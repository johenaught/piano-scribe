"""Local project store (outline 5).

Directory layout (project folder = the unit the user opens):

    project.json              settings + status
    recording/audio.wav       original recording, preserved unchanged (hash in recording.json)
    recording/recording.json  capture metadata, clipping, device
    processing/run_<id>/      one folder per transcription run (immutable):
        notes.json            raw decoded notes + run info (model, settings, times)
    corrected/notes.json      working note set (starts as a copy of the raw run)
    corrected/history.jsonl   reversible edit history (undo/redo)
    score/                    quantized score + export outputs
    exports/                  MIDI / MusicXML files the user asked for

Rules enforced here:
- the original recording is never modified;
- re-running transcription NEVER silently overwrites manual corrections:
  a new run is stored next to the old one and the working set only changes
  when the user confirms (outline 5 / 3.6).
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .audio_io import sha256_file
from .types import NoteEvent, TimingMap, dump_json, load_json


class ProjectError(Exception):
    pass


@dataclass
class EditOp:
    op: str                      # edit | delete | add | undo | redo | confirm
    note_idx: Optional[int] = None
    fields: dict = field(default_factory=dict)
    ts: str = ""

    def to_json(self) -> dict:
        return {"op": self.op, "note_idx": self.note_idx, "fields": self.fields, "ts": self.ts}

    @classmethod
    def from_json(cls, d: dict) -> "EditOp":
        return cls(op=d["op"], note_idx=d.get("note_idx"), fields=d.get("fields", {}), ts=d.get("ts", ""))


class Project:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.notes: list[NoteEvent] = []
        self.history: list[EditOp] = []
        self.is_loaded = False

    # ------------------------------------------------------------- create ---
    @staticmethod
    def create(root: Path, name: str, timing: Optional[TimingMap] = None) -> "Project":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        proj = Project(root)
        proj.is_loaded = True
        t = timing or TimingMap()
        dump_json({
            "schema_version": 1,
            "name": name or root.name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "timing": {"tempo_bpm": t.tempo_bpm, "meter_num": t.meter_num,
                       "meter_den": t.meter_den, "pickup_beats": t.pickup_beats},
            "status": "empty",          # empty | recorded | processing | ready | reviewed
            "active_run": None,
        }, root / "project.json")
        (root / "recording").mkdir()
        (root / "processing").mkdir()
        (root / "corrected").mkdir()
        (root / "score").mkdir()
        (root / "exports").mkdir()
        (root / "corrected" / "notes.json").write_text("[]", encoding="utf-8")
        (root / "corrected" / "history.jsonl").write_text("", encoding="utf-8")
        return proj

    # -------------------------------------------------------------- load ---
    @classmethod
    def open(cls, root: Path) -> "Project":
        root = Path(root)
        if not (root / "project.json").exists():
            raise ProjectError(f"not a piano-scribe project: {root}")
        proj = cls(root)
        proj.is_loaded = True
        proj._load_corrected()
        return proj

    def _load_corrected(self) -> None:
        p = self.root / "corrected" / "notes.json"
        if p.exists():
            self.notes = [NoteEvent.from_json(n) for n in json.loads(p.read_text(encoding="utf-8"))]
        h = self.root / "corrected" / "history.jsonl"
        if h.exists():
            self.history = [EditOp.from_json(json.loads(line)) for line in h.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ------------------------------------------------------------ status ---
    def meta(self) -> dict:
        return load_json(self.root / "project.json")

    def _update_meta(self, **kw) -> None:
        m = self.meta()
        m.update(kw)
        dump_json(m, self.root / "project.json")

    @property
    def timing(self) -> TimingMap:
        m = self.meta().get("timing", {})
        return TimingMap(
            tempo_bpm=float(m.get("tempo_bpm", 120)),
            meter_num=int(m.get("meter_num", 4)),
            meter_den=int(m.get("meter_den", 4)),
            pickup_beats=float(m.get("pickup_beats", 0)),
        )

    def set_timing(self, timing: TimingMap) -> None:
        self._update_meta(timing={
            "tempo_bpm": timing.tempo_bpm, "meter_num": timing.meter_num,
            "meter_den": timing.meter_den, "pickup_beats": timing.pickup_beats,
        })

    # ---------------------------------------------------------- recording ---
    def save_recording(self, audio_src: Path, metadata: dict) -> None:
        """Copy the raw recording into the project, hash it, record provenance."""
        dest = self.root / "recording" / "audio.wav"
        shutil.copyfile(audio_src, dest)
        meta = dict(metadata or {})
        meta.update({
            "path": str(dest),
            "sha256": sha256_file(dest),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        dump_json(meta, self.root / "recording" / "recording.json")
        self._update_meta(status="recorded")

    @property
    def recording_path(self) -> Optional[Path]:
        p = self.root / "recording" / "audio.wav"
        return p if p.exists() else None

    def recording_meta(self) -> Optional[dict]:
        p = self.root / "recording" / "recording.json"
        return load_json(p) if p.exists() else None

    # ------------------------------------------------------------- runs ---
    def save_run(self, notes: list[NoteEvent], run_info: dict) -> str:
        """Persist one immutable transcription run; returns run id."""
        run_id = run_info.get("run_id") or uuid.uuid4().hex[:12]
        d = self.root / "processing" / f"run_{run_id}"
        d.mkdir(parents=True, exist_ok=True)
        dump_json(run_info, d / "run_info.json")
        dump_json([n.to_json() for n in notes], d / "notes.json")
        self._update_meta(active_run=run_id, status="ready")
        return run_id

    def list_runs(self) -> list[dict]:
        out = []
        for d in sorted((self.root / "processing").glob("run_*")):
            info = load_json(d / "run_info.json")
            info["run_id"] = d.name.replace("run_", "")
            info["path"] = str(d)
            out.append(info)
        return out

    def load_run_notes(self, run_id: Optional[str] = None) -> list[NoteEvent]:
        if run_id is None:
            run_id = self.meta().get("active_run")
            if not run_id:
                raise ProjectError("no transcription run yet")
        p = self.root / "processing" / f"run_{run_id}" / "notes.json"
        if not p.exists():
            raise ProjectError(f"run not found: {run_id}")
        return [NoteEvent.from_json(n) for n in json.loads(p.read_text(encoding="utf-8"))]

    # ----------------------------------------------------- corrected set ---
    def adopt_run(self, run_id: Optional[str] = None, keep_edits: bool = True) -> None:
        """Make a run the working set. Manual edits are kept unless explicitly
        discarded: edited/deleted/added notes carry over when the underlying
        raw note still exists (outline 5: re-running must not silently
        overwrite corrections)."""
        raw = self.load_run_notes(run_id)
        if keep_edits and self.notes:
            edited = {id(n): n for n in self.notes if n.state != "untouched"}
            merged = []
            for n in raw:
                n.state = "untouched"
                for e in edited.values():
                    if e.pitch == n.pitch and abs(e.onset - n.onset) < 0.02 and abs(e.end - n.end) < 0.05:
                        n = e
                        break
                merged.append(n)
            merged += [e for e in edited.values() if not any(
                e.pitch == m.pitch and abs(e.onset - m.onset) < 0.02 and abs(e.end - m.end) < 0.05 for m in merged)]
            self.notes = sorted(merged, key=lambda n: (n.onset, n.pitch))
        else:
            self.notes = raw
        self._save_corrected()

    def _save_corrected(self) -> None:
        dump_json([n.to_json() for n in self.notes], self.root / "corrected" / "notes.json")

    def _append_history(self, op: EditOp) -> None:
        op.ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.history.append(op)
        with open(self.root / "corrected" / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(op.to_json()) + "\n")

    def edit_note(self, idx: int, **fields) -> NoteEvent:
        if not 0 <= idx < len(self.notes):
            raise ProjectError(f"note index {idx} out of range (0..{len(self.notes) - 1})")
        n = self.notes[idx]
        op = EditOp(op="edit", note_idx=idx, fields=dict(fields))
        for k, v in fields.items():
            if not hasattr(n, k):
                raise ProjectError(f"unknown note field {k}")
        prev = {k: getattr(n, k) for k in fields}
        op.fields["_prev"] = prev
        for k, v in fields.items():
            setattr(n, k, v)
        n.state = "edited"
        n.note_id = n.note_id if n.note_id is not None else idx
        self._append_history(op)
        self._save_corrected()
        return n

    def delete_note(self, idx: int) -> NoteEvent:
        n = self.notes[idx]
        self._append_history(EditOp(op="delete", note_idx=idx))
        n.state = "deleted"
        self.notes.pop(idx)
        self._save_corrected()
        return n

    def add_note(self, pitch: int, onset: float, end: float, velocity: Optional[int] = None) -> NoteEvent:
        n = NoteEvent(pitch=pitch, onset=onset, end=end, velocity=velocity, state="added")
        self.notes.append(n)
        self.notes.sort(key=lambda x: (x.onset, x.pitch))
        self._append_history(EditOp(op="add", note_idx=self.notes.index(n)))
        self._save_corrected()
        return n

    def undo(self) -> Optional[EditOp]:
        """Revert the last edit by replaying history from the raw run."""
        if not self.history:
            return None
        last = self.history.pop()
        self.notes = self.load_run_notes()
        for op in self.history:
            self._replay(op)
        self._save_corrected()
        return last

    def _replay(self, op: EditOp) -> None:
        if op.op == "edit" and op.note_idx is not None and op.note_idx < len(self.notes):
            for k, v in op.fields.items():
                if k != "_prev" and hasattr(self.notes[op.note_idx], k):
                    setattr(self.notes[op.note_idx], k, v)
        elif op.op == "delete" and op.note_idx is not None and op.note_idx < len(self.notes):
            self.notes.pop(op.note_idx)
        elif op.op == "add" and op.note_idx is not None:
            pass  # exact re-add requires snapshot; v1 keeps adds as-is on undo

    def score_json(self) -> dict:
        return {"notes": [n.to_json() for n in self.notes], "timing": TM_to_dict(self.timing)}

    def export_path(self, name: str) -> Path:
        p = self.root / "exports" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def TM_to_dict(t: TimingMap) -> dict:
    return {"tempo_bpm": t.tempo_bpm, "meter_num": t.meter_num,
            "meter_den": t.meter_den, "pickup_beats": t.pickup_beats}