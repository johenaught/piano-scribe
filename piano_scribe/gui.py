"""Piano Scribe desktop front end (tkinter, stdlib only).

Voice-memo style: a list of your sessions (projects) on the left; clicking a
session opens its details — record/import, transcribe with progress + cancel,
note table with editing and undo, a piano-roll score view, original-audio vs
synthesized-detected-notes playback, and MIDI/MusicXML export.

All heavy work (capture, transcription, rendering) runs in worker threads;
the UI stays responsive. Runs offline; no cloud.

Launch:  pythonw -m piano_scribe.gui        (or `piano-scribe-gui`)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
import winsound
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from .audio_io import load_clip
from .capture import record_microphone, session_metadata
from .corpus import load_labels  # noqa: F401  (kept for future real-corpora workflows)
from .models import CancelledError, get_backend
from .pipeline import ProgressTracker, transcribe_audio
from .project import Project, ProjectError
from .score_quantize import quantize_notes
from .types import NoteEvent, TranscribeSettings, midi_name

SESSIONS_DIR = str((Path.home() / "Documents" / "Piano Scribe Sessions"))

BGC = "#1F232B"        # window background
CARD = "#2A2F3A"       # session card
CARD_ACTIVE = "#33405C"
TXT = "#E8EAF0"
MUTED = "#9AA3B2"
ACCENT = "#4A9EFF"
GOOD = "#4CAF7D"
WARN = "#E5B567"


# ------------------------------------------------------------------ model ---

@dataclass
class Session:
    path: Path
    meta: dict
    recording: dict

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.path.name

    @property
    def status(self) -> str:
        return self.meta.get("status", "empty")

    @property
    def created(self) -> str:
        return (self.meta.get("created") or "")[:16].replace("T", " ")

    @property
    def duration_s(self) -> float:
        return float((self.recording or {}).get("duration_s", 0.0) or 0.0)

    @property
    def note_count(self) -> int:
        p = self.path / "corrected" / "notes.json"
        if p.exists():
            try:
                return len(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                return 0
        return 0


def list_sessions(root: Path) -> list[Session]:
    out = []
    if root.exists():
        for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
            if not d.is_dir():
                continue
            try:
                meta = json.loads((d / "project.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            rec = {}
            rp = d / "recording" / "recording.json"
            if rp.exists():
                try:
                    rec = json.loads(rp.read_text(encoding="utf-8"))
                except Exception:
                    rec = {}
            out.append(Session(d, meta, rec))
    return out


def _fmt_duration(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


# ------------------------------------------------------------------- app ---

class App(tk.Tk):
    def __init__(self, sessions_root: Path | None = None) -> None:
        super().__init__()
        self.sessions_root = Path(sessions_root or SESSIONS_DIR)
        self.sessions_root.mkdir(parents=True, exist_ok=True)

        self.title("Piano Scribe")
        self.geometry("1280x780")
        self.minsize(980, 620)
        self.configure(bg=BGC)
        self.option_add("*Font", ("Segoe UI", 10))

        self.proj: Project | None = None
        self.session: Session | None = None
        self._tracker: ProgressTracker | None = None
        self._worker: threading.Thread | None = None
        self._rec_stop: threading.Event | None = None
        self._rec_start: float = 0.0
        self._denoise_var = tk.BooleanVar(value=False)
        self._msg_queue: list[tuple[str, str]] = []
        self._playing = False

        self._style = ttk.Style(self)
        self._style.theme_use("clam")
        self._style.configure("Treeview", background="#232733", fieldbackground="#232733",
                              foreground=TXT, rowheight=24, borderwidth=0)
        self._style.configure("Treeview.Heading", background=CARD, foreground=TXT,
                              font=("Segoe UI", 9, "bold"))
        self._style.map("Treeview", background=[("selected", "#3B4B6B")])
        self._style.configure("TProgressbar", background=ACCENT, troughcolor="#232733")

        self._build_header()
        self._build_body()
        self._build_statusbar()
        self._refresh_sessions()
        self.after(120, self._pump_msgs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- chrome ---
    def _build_header(self) -> None:
        bar = tk.Frame(self, bg=BGC)
        bar.pack(side="top", fill="x")
        tk.Label(bar, text="Piano Scribe", font=("Segoe UI", 16, "bold"),
                 fg=TXT, bg=BGC).pack(side="left", padx=14, pady=8)
        tk.Label(bar, text="local transcription • no cloud", font=("Segoe UI", 9),
                 fg=MUTED, bg=BGC).pack(side="left")
        tk.Button(bar, text="＋ New session", command=self._new_session, bg=ACCENT, fg="white",
                  activebackground="#6FB0FF", relief="flat", padx=14, pady=5,
                  font=("Segoe UI", 10, "bold")).pack(side="right", padx=12)
        tk.Button(bar, text="Add existing project…", command=self._add_existing,
                  bg=CARD, fg=TXT, relief="flat", padx=10, pady=5).pack(side="right", padx=8)

    def _build_body(self) -> None:
        body = tk.Frame(self, bg=BGC)
        body.pack(fill="both", expand=True)

        # left: session list (voice-memo style)
        left = tk.Frame(body, bg=BGC, width=330)
        left.pack(side="left", fill="y", padx=(6, 2))
        left.pack_propagate(False)
        tk.Label(left, text="SESSIONS", font=("Segoe UI", 9, "bold"),
                 fg=MUTED, bg=BGC).pack(anchor="w", padx=10)
        self._list_canvas = tk.Canvas(left, bg=BGC, highlightthickness=0)
        self._list_frame = tk.Frame(self._list_canvas, bg=BGC)
        self._list_win = self._list_canvas.create_window((0, 0), window=self._list_frame, anchor="nw")
        self._list_canvas.pack(fill="both", expand=True)
        self._list_canvas.bind("<Configure>", lambda e: self._list_canvas.itemconfigure(
            self._list_win, width=e.width))

        # right: detail
        right = tk.Frame(body, bg=BGC)
        right.pack(side="left", fill="both", expand=True, padx=2, pady=(24, 0))
        self._nb = ttk.Notebook(right)
        self._nb.pack(fill="both", expand=True)

        self._tab_session = tk.Frame(self._nb, bg=BGC)
        self._tab_notes = tk.Frame(self._nb, bg=BGC)
        self._tab_score = tk.Frame(self._nb, bg=BGC)
        self._nb.add(self._tab_session, text="  Session  ")
        self._nb.add(self._tab_notes, text="  Notes  ")
        self._nb.add(self._tab_score, text="  Score  ")

        self._build_tab_session(self._tab_session)
        self._build_tab_notes(self._tab_notes)
        self._build_tab_score(self._tab_score)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self, bg=CARD)
        bar.pack(side="bottom", fill="x")
        self._status = tk.Label(bar, text="Ready.", fg=MUTED, bg=CARD, anchor="w",
                                font=("Segoe UI", 9))
        self._status.pack(side="left", fill="x", padx=10, pady=3)

    def _status_msg(self, text: str, kind: str = "info") -> None:
        color = {"info": MUTED, "ok": GOOD, "warn": WARN}[kind]
        self._status.configure(text=text, fg=color)

    def _ask(self, title: str, prompt: str) -> str | None:
        import tkinter.simpledialog as sd
        return sd.askstring(title, prompt, parent=self)

    # -------------------------------------------------------- session list ---
    def _refresh_sessions(self) -> None:
        for w in self._list_frame.winfo_children():
            w.destroy()
        sessions = list_sessions(self.sessions_root)
        if not sessions:
            tk.Label(self._list_frame, text="No sessions yet.\n\nClick “＋ New session” above.",
                     fg=MUTED, bg=BGC, justify="left").pack(padx=10, pady=30)
        for s in sessions:
            active = self.session is not None and s.path == self.session.path
            card = tk.Frame(self._list_frame, bg=CARD_ACTIVE if active else CARD,
                            padx=10, pady=7, cursor="hand2")
            card.pack(fill="x", padx=6, pady=5)
            # click anywhere on the card (or its labels) opens the session
            def _bind_click(widget, s=s):
                widget.bind("<Button-1>", lambda _e, s=s: self._open_session(s))
            for w in [card] + _all_descendants(card):
                _bind_click(w)
            tk.Label(card, text=s.name, font=("Segoe UI", 11, "bold"), fg=TXT, bg=card["bg"]).pack(anchor="w")
            sub = f"{s.created}"
            if s.duration_s:
                sub += f"   •   {_fmt_duration(s.duration_s)}"
            if s.note_count:
                sub += f"   •   {s.note_count} notes"
            tk.Label(card, text=sub, font=("Segoe UI", 8), fg=MUTED, bg=card["bg"]).pack(anchor="w")
            badge = {"empty": "empty", "recorded": "recorded", "ready": "transcribed",
                     "reviewed": "reviewed"}.get(s.status, s.status)
            tk.Label(card, text=badge, font=("Segoe UI", 8, "italic"), fg=ACCENT if s.status in ("ready", "reviewed") else MUTED,
                     bg=card["bg"]).pack(anchor="w")

    def _open_session(self, s: Session) -> None:
        try:
            self.proj = Project.open(s.path)
        except ProjectError as e:
            messagebox.showerror("Piano Scribe", str(e), parent=self)
            return
        self.session = s
        self._status_msg(f"Opened {s.name}.")
        self._refresh_sessions()
        self._reload_detail()

    def _reload_detail(self) -> None:
        self._reload_session_tab()
        self._reload_notes_table()
        self._draw_score()

    # ------------------------------------------------------ session tab -----
    def _build_tab_session(self, parent: tk.Frame) -> None:
        info = tk.Frame(parent, bg=BGC)
        info.pack(fill="x", padx=12, pady=6)
        self._info_title = tk.Label(info, text="No session selected", font=("Segoe UI", 14, "bold"),
                                    fg=TXT, bg=BGC)
        self._info_title.pack(anchor="w")
        self._info_sub = tk.Label(info, text="", fg=MUTED, bg=BGC)
        self._info_sub.pack(anchor="w")

        steps = tk.Frame(parent, bg=BGC)
        steps.pack(fill="x", padx=12, pady=10)

        def step(title: str, text: str) -> None:
            box = tk.Frame(steps, bg=CARD, padx=12, pady=8)
            box.pack(side="left", fill="both", expand=True, padx=6)
            tk.Label(box, text=title, font=("Segoe UI", 10, "bold"), fg=TXT, bg=CARD).pack(anchor="w")
            tk.Label(box, text=text, fg=MUTED, bg=CARD, justify="left", wraplength=240).pack(anchor="w", pady=(2, 6))
            return box

        box1 = step("1 · Capture", "Record from your microphone or import an audio file. The original is saved untouched.")
        b = tk.Frame(box1, bg=CARD)
        b.pack(fill="x")
        self.btn_record = tk.Button(b, text="● Record", command=self._record, bg="#C5484D",
                                    fg="white", relief="flat", padx=12, pady=4, font=("Segoe UI", 10, "bold"))
        self.btn_record.pack(side="left", padx=4)
        self.btn_import = tk.Button(b, text="Import file…", command=self._import_audio, bg=CARD, fg=TXT, relief="flat", padx=10, pady=4)
        self.btn_import.pack(side="left", padx=4)
        self.rec_elapsed = tk.Label(box1, text="", fg=TXT, bg=CARD)
        self.rec_elapsed.pack(anchor="w", pady=(4, 0))

        box2 = step("2 · Transcribe", "Runs locally. Progress and cancel below.")
        self.btn_transcribe = tk.Button(box2, text="▶ Transcribe", command=self._transcribe,
                                        bg=ACCENT, fg="white", relief="flat", padx=12, pady=4,
                                        font=("Segoe UI", 10, "bold"))
        self.btn_transcribe.pack(side="left", padx=4)
        self.btn_cancel = tk.Button(box2, text="Cancel", command=self._cancel_work,
                                    bg=CARD, fg=TXT, relief="flat", padx=10, pady=4, state="disabled")
        self.btn_cancel.pack(side="left", padx=4)
        self.btn_denoise = tk.Checkbutton(box2, text="light noise reduction (A/B test)",
                                          variable=self._denoise_var, fg=TXT, bg=BGC,
                                          activebackground=BGC, selectcolor=BGC, font=("Segoe UI", 9))
        self.btn_denoise.pack(anchor="w", pady=4)
        self.progress = ttk.Progressbar(box2, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 2))
        self.prog_label = tk.Label(box2, text="", fg=MUTED, bg=CARD)
        self.prog_label.pack(anchor="w")

        box3 = step("3 · Review & export", "Fix notes, listen to the comparison, export.")
        b3 = tk.Frame(box3, bg=CARD)
        b3.pack(fill="x")
        self.btn_midi = tk.Button(b3, text="Export MIDI", command=lambda: self._export("midi"),
                                  bg=CARD, fg=TXT, relief="flat", padx=10, pady=4)
        self.btn_midi.pack(side="left", padx=4)
        self.btn_xml = tk.Button(b3, text="Export MusicXML", command=lambda: self._export("musicxml"),
                                 bg=CARD, fg=TXT, relief="flat", padx=10, pady=4)
        self.btn_xml.pack(side="left", padx=4)
        self.btn_mscore = tk.Button(b3, text="Open in MuseScore…", command=self._open_musescore,
                                    bg=CARD, fg=TXT, relief="flat", padx=10, pady=4)
        self.btn_mscore.pack(side="left", padx=4)
        b4 = tk.Frame(box3, bg=CARD)
        b4.pack(fill="x", pady=4)
        self.btn_play_audio = tk.Button(b4, text="▶ Play recording", command=self._play_recording,
                                        bg="#3A7BBF", fg="white", relief="flat", padx=10, pady=4)
        self.btn_play_audio.pack(side="left", padx=4)
        self.btn_play_synth = tk.Button(b4, text="▶ Play detected notes", command=self._play_detected,
                                        bg="#3A7BBF", fg="white", relief="flat", padx=10, pady=4)
        self.btn_play_synth.pack(side="left", padx=4)
        self.btn_stop = tk.Button(b4, text="■ Stop", command=self._stop_playback,
                                  bg=CARD, fg=TXT, relief="flat", padx=10, pady=4)
        self.btn_stop.pack(side="left", padx=4)
        tk.Label(b4, text="compare original vs. what was detected", font=("Segoe UI", 8),
                 fg=MUTED, bg=CARD).pack(side="left", padx=6)

        danger = tk.Frame(parent, bg=BGC)
        danger.pack(side="bottom", fill="x", padx=12, pady=6)
        self.btn_delete = tk.Button(danger, text="Delete session…", command=self._delete_session,
                                    bg="#3A2B2B", fg=MUTED, relief="flat", padx=10, pady=4)
        self.btn_delete.pack(side="left")

    def _reload_session_tab(self) -> None:
        if not self.proj:
            self._info_title.configure(text="No session selected")
            self._info_sub.configure(text="Create a new session or pick one from the list.")
            for w in (self.btn_record, self.btn_import, self.btn_transcribe):
                w.configure(state="disabled")
            return
        s = self.session
        self._info_title.configure(text=s.name)
        sub = f"{s.created}   •   {self.proj.root}"
        if s.duration_s:
            sub += f"   •   {_fmt_duration(s.duration_s)} recording"
        self._info_sub.configure(text=sub)
        for w in (self.btn_record, self.btn_import, self.btn_transcribe):
            w.configure(state="normal")
        busy = self._worker is not None and self._worker.is_alive()
        if busy:
            self.btn_record.configure(state="disabled")

    # ---------------------------------------------------------- new/delete ---
    def _new_session(self) -> None:
        name = self._ask("New session", "Session name:")
        if not name:
            return
        path = self.sessions_root / name
        if path.exists():
            messagebox.showerror("Piano Scribe", f"A session named {name!r} already exists.", parent=self)
            return
        Project.create(path, name)
        self._status_msg(f"Created session {name!r}.", "ok")
        self._refresh_sessions()
        for s in list_sessions(self.sessions_root):
            if s.path == path:
                self._open_session(s)
                break

    def _add_existing(self) -> None:
        d = filedialog.askdirectory(parent=self, title="Pick a piano-scribe project folder")
        if not d:
            return
        p = Path(d)
        if not (p / "project.json").exists():
            messagebox.showerror("Piano Scribe", "That folder has no project.json — not a piano-scribe project.", parent=self)
            return
        link = self.sessions_root / p.name
        if link.exists():
            messagebox.showinfo("Piano Scribe", "A session with that name is already listed.", parent=self)
        else:
            try:
                import os
                os.symlink(p, link, target_is_directory=True) if hasattr(os, "symlink") else shutil.copytree(p, link)
            except (OSError, NotImplementedError):
                shutil.copytree(p, link)
        self._refresh_sessions()

    def _delete_session(self) -> None:
        if not self.proj:
            return
        if not messagebox.askyesno("Piano Scribe",
                                   f"Delete session {self.proj.meta().get('name')!r} permanently?\n"
                                   "This removes the recording and all notes.", parent=self):
            return
        root = self.proj.root
        self.proj = None
        self.session = None
        shutil.rmtree(root, ignore_errors=True)
        self._reload_detail()
        self._refresh_sessions()
        self._status_msg("Session deleted.", "warn")

    # ------------------------------------------------------------- capture ---
    def _record(self) -> None:
        if not self.proj:
            return
        if self._rec_stop is not None:
            self._stop_record()
            return
        cap = self.proj.root / "recording" / "_capture.wav"
        self._rec_stop = threading.Event()
        self._rec_start = time.time()
        device = None

        def work() -> None:
            try:
                session = record_microphone(cap, seconds=None, device=device,
                                            stop_event=self._rec_stop)
                self.proj.save_recording(cap, session_metadata(session))
                cap.unlink(missing_ok=True)
                self._msg_queue.append(("rec_done", f"Saved {session.duration_s:.1f} s of audio."))
            except Exception as e:  # noqa: BLE001
                self._msg_queue.append(("rec_fail", str(e)))

        self.btn_record.configure(text="■ Stop", bg="#C5484D")
        self.btn_import.configure(state="disabled")
        self.rec_elapsed.configure(text="recording…")
        self._status_msg("Recording — click Stop when done.", "warn")
        threading.Thread(target=work, daemon=True).start()

    def _stop_record(self) -> None:
        if self._rec_stop is not None:
            self._rec_stop.set()

    def _import_audio(self) -> None:
        if not self.proj:
            return
        path = filedialog.askopenfilename(
            parent=self, title="Import audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")])
        if not path:
            return
        from .audio_io import probe
        try:
            self.proj.save_recording(Path(path), probe(Path(path)))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Piano Scribe", f"Could not import: {e}", parent=self)
            return
        self._status_msg(f"Imported {Path(path).name}.", "ok")
        self._reload_session_tab()
        self._refresh_sessions()

    # ---------------------------------------------------------- transcribe ---
    def _transcribe(self) -> None:
        if not self.proj or self.proj.recording_path is None:
            messagebox.showinfo("Piano Scribe", "Record or import audio first.", parent=self)
            return
        if self._worker is not None and self._worker.is_alive():
            return
        self._tracker = ProgressTracker()
        settings = TranscribeSettings(
            model="basic-pitch",
            min_note_length_s=0.05,
            min_confidence=0.25,
            chunk_s=45.0,
            overlap_s=2.0,
            denoise=self._denoise_var.get(),
        )
        backend = get_backend(settings.model)

        def work() -> None:
            try:
                clip = load_clip(self.proj.recording_path, sr=backend.spec.sample_rate)
                res = transcribe_audio(clip, settings, backend,
                                       self._tracker.progress, self._tracker.cancel_event)
                run_id = self.proj.save_run(res.notes, {
                    "run_id": res.run_id, "model": res.model, "model_version": res.model_version,
                    "runtime": res.runtime, "settings": res.settings,
                    "audio_length_s": res.audio_length_s, "processing_time_s": res.processing_time_s,
                    "status": res.status, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                self.proj.adopt_run(run_id, keep_edits=False)
                state = res.status
                n = len(res.notes)
                self._msg_queue.append(("work_done", f"({state}) {n} notes in {res.processing_time_s:.1f} s."))
            except CancelledError:
                self._msg_queue.append(("work_cancel", "Cancelled."))
            except Exception as e:  # noqa: BLE001
                self._msg_queue.append(("work_fail", str(e)))

        self.progress.configure(value=0)
        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self.btn_transcribe.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_record.configure(state="disabled")
        self._status_msg("Transcribing locally…", "warn")
        self._poll_progress()

    def _poll_progress(self) -> None:
        if self._tracker is not None:
            frac, stage = self._tracker.last_progress
            self.progress.configure(value=frac * 100)
            self.prog_label.configure(text=stage)
            self._status_msg(f"Transcribing — {stage} ({frac * 100:.0f}%)", "warn")
        if self._worker is not None and self._worker.is_alive():
            self.after(120, self._poll_progress)
        else:
            self.progress.configure(value=0)
            self.prog_label.configure(text="")
            self.btn_transcribe.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            self.btn_record.configure(state="normal")

    def _cancel_work(self) -> None:
        if self._tracker is not None:
            self._tracker.cancel()
            self._status_msg("Cancelling…", "warn")

    # -------------------------------------------------------------- notes ---
    def _build_tab_notes(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=BGC)
        bar.pack(fill="x", padx=8, pady=(4, 2))
        for text, cmd in (("Edit selected…", self._edit_note),
                          ("Delete selected", self._delete_note),
                          ("Undo last edit", self._undo_note)):
            tk.Button(bar, text=text, command=cmd, bg=CARD, fg=TXT, relief="flat",
                      padx=10, pady=4).pack(side="left", padx=4)
        tk.Label(bar, text="confidence < 0.5 shown in amber", font=("Segoe UI", 8),
                 fg=MUTED, bg=BGC).pack(side="right", padx=6)
        cols = ("idx", "note", "pitch", "onset", "end", "dur", "conf", "vel", "state")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
        heads = {"idx": "#", "note": "Note", "pitch": "Pitch", "onset": "Onset (s)",
                 "end": "End (s)", "dur": "Dur (s)", "conf": "Conf", "vel": "Vel", "state": "State"}
        widths = {"idx": 38, "note": 60, "pitch": 50, "onset": 80, "end": 80,
                  "dur": 70, "conf": 60, "vel": 46, "state": 90}
        for c in cols:
            self.tree.heading(c, text=heads[c])
            self.tree.column(c, width=widths[c], anchor="center", stretch=False)
        self.tree.column("note", stretch=True)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=8)
        vsb.pack(side="right", fill="y")
        self.tree.tag_configure("low", foreground=WARN)
        self.tree.tag_configure("del", foreground="#6B7380")

    def _reload_notes_table(self) -> None:
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())
        if not self.proj:
            return
        for i, n in enumerate(self.proj.notes):
            tag = "del" if n.state == "deleted" else ("low" if n.confidence is not None and n.confidence < 0.5 else "")
            self.tree.insert("", "end", iid=str(i), values=(
                i, midi_name(n.pitch), n.pitch, f"{n.onset:.3f}", f"{n.end:.3f}",
                f"{n.end - n.onset:.3f}",
                f"{n.confidence:.2f}" if n.confidence is not None else "–",
                n.velocity if n.velocity else "–", n.state), tags=(tag,))

    def _selected_index(self) -> int | None:
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def _edit_note(self) -> None:
        idx = self._selected_index()
        if idx is None or not self.proj:
            return
        n = self.proj.notes[idx]
        dlg = tk.Toplevel(self)
        dlg.title(f"Edit note {idx}")
        dlg.configure(bg=BGC)
        dlg.transient(self)
        vals = {}
        rows = [("pitch", "Pitch (MIDI)", str(n.pitch)), ("onset", "Onset (s)", f"{n.onset:.3f}"),
                ("end", "End (s)", f"{n.end:.3f}"), ("velocity", "Velocity", str(n.velocity if n.velocity else ""))]
        for r, (key, label, val) in enumerate(rows):
            tk.Label(dlg, text=label, fg=TXT, bg=BGC, font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w", padx=8, pady=6)
            e = tk.Entry(dlg, bg="#232733", fg=TXT, insertbackground=TXT, width=12)
            e.insert(0, val)
            e.grid(row=r, column=1, padx=8, pady=6)
            vals[key] = e
        def apply() -> None:
            try:
                fields = {}
                for key, e in vals.items():
                    raw = e.get().strip()
                    if raw:
                        fields[key] = float(raw) if key in ("onset", "end") else int(round(float(raw)))
                if fields:
                    self.proj.edit_note(idx, **fields)
                    self._reload_notes_table()
                    self._draw_score()
                    self._status_msg(f"Edited note {idx}.", "ok")
                dlg.destroy()
            except Exception as ex:  # noqa: BLE001
                messagebox.showerror("Piano Scribe", f"Invalid value: {ex}", parent=dlg)
        tk.Button(dlg, text="Apply", command=apply, bg=ACCENT, fg="white", relief="flat",
                  padx=12, pady=4).grid(row=len(rows), column=0, columnspan=2, pady=10)
        dlg.grab_set()

    def _delete_note(self) -> None:
        idx = self._selected_index()
        if idx is None or not self.proj:
            return
        self.proj.delete_note(idx)
        self._reload_notes_table()
        self._draw_score()
        self._status_msg(f"Deleted note {idx}.", "ok")

    def _undo_note(self) -> None:
        if not self.proj:
            return
        op = self.proj.undo()
        self._reload_notes_table()
        self._draw_score()
        self._status_msg(f"Undid {op.op if op else 'nothing'}." if op else "Nothing to undo.", "ok" if op else "info")

    # -------------------------------------------------------------- score ---
    def _build_tab_score(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=BGC)
        bar.pack(fill="x", padx=8, pady=4)
        tk.Label(bar, text="Piano roll — performed timing; measure lines from the session tempo/meter.",
                 font=("Segoe UI", 9), fg=MUTED, bg=BGC).pack(side="left")
        self.score_info = tk.Label(bar, text="", font=("Segoe UI", 9), fg=TXT, bg=BGC)
        self.score_info.pack(side="right", padx=8)
        self.score_canvas = tk.Canvas(parent, bg="#14161B", highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.score_canvas.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self.score_canvas.xview)
        self.score_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.score_canvas.pack(side="left", fill="both", expand=True, padx=8)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x", padx=8)
        self.score_canvas.bind("<Configure>", lambda e: self.score_canvas.configure(scrollregion=self.score_canvas.bbox("all")))
        self._score_items: list[int] = []
        self._score_selected = None

    def _draw_score(self) -> None:
        if not hasattr(self, "score_canvas"):
            return
        c = self.score_canvas
        c.delete("all")
        if not self.proj or not self.proj.notes:
            c.create_text(400, 200, text="No notes yet — record and transcribe first.",
                          fill=MUTED, font=("Segoe UI", 12))
            self.score_info.configure(text="")
            return
        notes = sorted(self.proj.notes, key=lambda n: (n.onset, n.pitch))
        t0 = max(0.0, notes[0].onset - 0.5)
        t1 = max(n.end for n in notes) + 0.5
        lo = max(21, min(n.pitch for n in notes) - 3)
        hi = min(108, max(n.pitch for n in notes) + 3)
        PADL, PADT, PADB = 46, 26, 26
        W, H = 900, max(300, 14 * (hi - lo + 1))
        # beat grid from session tempo
        try:
            timing = self.proj.timing
            spb = timing.seconds_per_beat()
            bpm = timing.beats_per_measure()
        except Exception:
            spb, bpm = 0.5, 4.0
        x_of = lambda t: PADL + (t - t0) / (t1 - t0) * (W - PADL - 20)
        y_of = lambda p: PADT + (hi - p) / (hi - lo) * (H - PADT - PADB)
        # clef zones + C lines
        for p in range(lo, hi + 1):
            y = y_of(p)
            color = "#232733" if p % 12 in (0, 7) else "#1B1F26"
            c.create_line(PADL, y, W - 20, y, fill=color)
            c.create_text(PADL - 8, y, text=midi_name(p), fill=MUTED, font=("Segoe UI", 7))
        # measure lines
        beat = 0.0
        while t0 + beat * spb <= t1 + 0.01:
            x = x_of(t0 + beat * spb)
            c.create_line(x, PADT, x, H - PADB, fill="#3A4154", dash=(4, 4))
            c.create_text(x, 14, text=f"m{int(beat // bpm) + 1}", fill=MUTED, font=("Segoe UI", 7))
            beat += bpm
        c.create_text(PADL, 14, text=f"{t0:.1f}s", fill=MUTED, font=("Segoe UI", 7), anchor="w")
        # notes
        for i, n in enumerate(notes):
            x0, x1 = x_of(n.onset), x_of(n.end)
            y = y_of(n.pitch)
            color = {"added": GOOD, "deleted": "#555C66", "edited": "#E5B567"}.get(n.state,
                    "#4A9EFF" if n.confidence is None or n.confidence >= 0.5 else "#E5B567")
            rect = c.create_rectangle(x0, y - 4, max(x0 + 2, x1), y + 4, fill=color, outline="")
            self._score_items.append(rect)
            c.tag_bind(rect, "<Button-1>", lambda _e, i=i, n=n, rect=rect: self._score_click(i, n, rect))
        c.create_text(W / 2, H - 8, text="left-click a note to inspect it", fill=MUTED, font=("Segoe UI", 8))
        c.configure(scrollregion=(0, 0, W, H))
        self.score_info.configure(text=f"{len(notes)} notes   •   {lo}–{hi} ({t1 - t0:.1f} s)")

    def _score_click(self, i: int, n: NoteEvent, rect: int) -> None:
        if self._score_selected is not None:
            self.score_canvas.itemconfigure(self._score_selected, outline="")
        self._score_selected = rect
        self.score_canvas.itemconfigure(rect, outline="white", width=2)
        self.score_info.configure(
            text=f"#{i} {midi_name(n.pitch)}  onset {n.onset:.3f}s  end {n.end:.3f}s"
                 f"  conf {n.confidence if n.confidence is not None else '–'}  state {n.state}")

    # ------------------------------------------------------------- export ---
    def _export(self, fmt: str) -> None:
        if not self.proj:
            return
        try:
            timing = self.proj.timing
            score = quantize_notes(self.proj.notes, timing)
            if fmt == "midi":
                from .export_midi import export_midi
                path = self.proj.export_path(f"{self.proj.meta().get('name', 'score')}.mid")
                export_midi(score, path)
            else:
                from .export_musicxml import export_musicxml
                path = self.proj.export_path(f"{self.proj.meta().get('name', 'score')}.musicxml")
                export_musicxml(score, path, title=self.proj.meta().get("name", "Untitled"))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Piano Scribe", f"Export failed: {e}", parent=self)
            return
        self._status_msg(f"Exported → {path}", "ok")
        if messagebox.askyesno("Piano Scribe", f"Exported to:\n{path}\n\nOpen the containing folder?",
                               parent=self):
            try:
                import os
                os.startfile(path.parent)  # noqa: S606
            except Exception:
                pass

    def _open_musescore(self) -> None:
        ms = shutil.which("museScore4") or shutil.which("musescore3") or shutil.which("MuseScore4")
        if not ms:
            messagebox.showinfo("Piano Scribe",
                                "MuseScore not found on PATH. Install it (free, musescore.org)\n"
                                "to view and print the sheet music.", parent=self)
            return
        if not self.proj:
            return
        try:
            timing = self.proj.timing
            score = quantize_notes(self.proj.notes, timing)
            from .export_musicxml import export_musicxml
            path = self.proj.export_path(f"{self.proj.meta().get('name', 'score')}.musicxml")
            export_musicxml(score, path, title=self.proj.meta().get("name", "Untitled"))
            subprocess.Popen([ms, str(path)])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Piano Scribe", str(e), parent=self)

    # ------------------------------------------------------------ playback ---
    def _play_recording(self) -> None:
        if not self.proj or self.proj.recording_path is None:
            return
        self._play_file(self.proj.recording_path)

    def _play_detected(self) -> None:
        if not self.proj or not self.proj.notes:
            messagebox.showinfo("Piano Scribe", "No detected notes to play.", parent=self)
            return
        try:
            from .synth import render_notes
            audio = render_notes(self.proj.notes)
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="pianoscribe_")
            import os
            os.close(fd)
            from .audio_io import save_wav
            save_wav(audio, 22050, Path(tmp))
            self._play_file(Path(tmp), delete_after=True)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Piano Scribe", f"Could not synthesize playback: {e}", parent=self)

    def _play_file(self, path: Path, delete_after: bool = False) -> None:
        self._stop_playback()
        try:
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._playing = True
            if delete_after:

                def cleanup() -> None:
                    time.sleep(6.0)
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                threading.Thread(target=cleanup, daemon=True).start()
            self._status_msg(f"Playing {path.name}…", "warn")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Piano Scribe", f"Playback failed: {e}", parent=self)

    def _stop_playback(self) -> None:
        winsound.PlaySound(None, winsound.PURGE)
        self._playing = False
        self._status_msg("Stopped playback.")

    # ----------------------------------------------------------- plumbing ---
    def _pump_msgs(self) -> None:
        while self._msg_queue:
            kind, text = self._msg_queue.pop(0)
            if kind in ("work_done",):
                self._status_msg(text, "ok")
                if self.proj:
                    self._reload_detail()
                self._refresh_sessions()
            elif kind == "work_cancel":
                self._status_msg(text, "warn")
            elif kind == "work_fail":
                self._status_msg(f"Transcription failed: {text}", "warn")
                messagebox.showerror("Piano Scribe", f"Transcription failed:\n{text}", parent=self)
            elif kind == "rec_done":
                self._status_msg(text, "ok")
                self.btn_record.configure(text="● Record")
                self.btn_import.configure(state="normal")
                self.rec_elapsed.configure(text="")
                self._reload_session_tab()
                self._refresh_sessions()
                if messagebox.askyesno("Piano Scribe", "Recording saved. Transcribe it now?",
                                       parent=self):
                    self._transcribe()
            elif kind == "rec_fail":
                self.btn_record.configure(text="● Record")
                self.btn_import.configure(state="normal")
                self.rec_elapsed.configure(text="")
                self._status_msg(f"Recording failed: {text}", "warn")
        self.after(120, self._pump_msgs)

    def _on_close(self) -> None:
        self._stop_playback()
        self.destroy()


def _all_descendants(widget) -> list:
    out = []
    for w in widget.winfo_children():
        out.append(w)
        out.extend(_all_descendants(w))
    return out


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()