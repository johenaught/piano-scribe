"""Command-line interface for the desktop reference pipeline.

Mirrors the user workflow (outline 2): create project -> record/import ->
calibrate -> transcribe -> review/edit -> export. A GUI (Flutter) replaces
this surface later; the pipeline and project store behind it stay.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .audio_io import AudioClip, clip_stats, load_clip
from .calibration import check_known_notes, estimate_noise_floor
from .capture import HAVE_SOUNDDEVICE, record_microphone, session_metadata
from .evaluate import EvalConfig, evaluate_pair, load_note_json, summarize
from .models import get_backend
from .pipeline import ProgressTracker, transcribe_audio
from .project import Project, ProjectError
from .score_quantize import TimingMap, quantize_notes
from .types import NoteEvent, midi_name


def _project(args) -> Project:
    return Project.open(Path(args.project))


def _print_notes(notes, limit: int = 200) -> None:
    print(f"{'#':>4} {'note':<6} {'pitch':>5} {'onset':>8} {'end':>8} {'dur':>6} {'conf':>6} {'vel':>4} state")
    for i, n in enumerate(notes[:limit]):
        print(f"{i:>4} {midi_name(n.pitch):<6} {n.pitch:>5} {n.onset:8.3f} {n.end:8.3f} "
              f"{n.end - n.onset:6.3f} {n.confidence if n.confidence is not None else float('nan'):6.3f} "
              f"{n.velocity if n.velocity else 0:>4} {n.state}")
    if len(notes) > limit:
        print(f"... {len(notes) - limit} more")


# ------------------------------------------------------------------ cmds ---

def cmd_new(args) -> int:
    root = Path(args.path)
    Project.create(root, args.name, TimingMap(tempo_bpm=args.tempo, meter_num=args.meter_num, meter_den=args.meter_den))
    print(f"created project: {root}")
    return 0


def cmd_import(args) -> int:
    proj = _project(args)
    meta = {}
    if args.audio:
        from .audio_io import probe
        meta = probe(Path(args.audio))
    proj.save_recording(Path(args.audio), meta)
    print(f"imported recording -> {proj.root / 'recording' / 'audio.wav'}")
    print(json.dumps(meta, indent=2))
    return 0


def cmd_record(args) -> int:
    proj = _project(args)
    try:
        session = record_microphone(
            proj.root / "recording" / "audio.wav",
            seconds=args.seconds,
            device=args.device,
            on_progress=lambda f: print(f"\rrecording... {f * 100:5.1f}%", end="", flush=True) if args.verbose else None,
        )
    except RuntimeError as e:
        print(f"recording failed: {e}", file=sys.stderr)
        return 1
    print("\n" + " " * 30, end="\r")
    proj.save_recording(proj.root / "recording" / "audio.wav", session_metadata(session))
    print(f"saved {session.path} ({session.duration_s:.1f}s, {session.sample_rate} Hz, {session.channels} ch)")
    if session.clipping_frames:
        print(f"WARNING: {session.clipping_frames} clipped frames detected -- lower input gain and re-record")
    return 0


def cmd_calibrate(args) -> int:
    proj = _project(args)
    clip = load_clip(proj.recording_path, sr=None)
    nf = estimate_noise_floor(clip.mono_mix(), clip.sample_rate, silence_seconds=args.seconds)
    print(f"silence sample ({nf.notes_seconds}s): RMS {nf.rms_dbfs:.1f} dBFS, 99th-pct peak {nf.peak_dbfs:.1f} dBFS")
    print(f"noise floor estimate: {nf.estimate if nf.estimate is not None else 'inaudible'}")
    stats = clip_stats(clip.mono_mix())
    print(f"levels: {stats.hint}")
    proj._update_meta(calibration={"noise_floor": nf.to_json(), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with open(proj.root / "recording" / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(nf.to_json(), f, indent=2)
    return 0


def cmd_transcribe(args) -> int:
    proj = _project(args)
    if proj.recording_path is None:
        print("no recording yet -- record or import first", file=sys.stderr)
        return 1
    from .types import TranscribeSettings
    settings = TranscribeSettings(
        model=args.model,
        min_pitch=args.min_pitch, max_pitch=args.max_pitch,
        onset_threshold=args.onset, frame_threshold=args.frame,
        min_note_length_s=args.min_len, min_confidence=args.conf_threshold,
        merge_gap_s=args.merge_gap,
        chunk_s=args.chunk_s, overlap_s=args.overlap_s,
        denoise=args.denoise,
    )
    backend = get_backend(args.model)
    tracker = ProgressTracker()
    clip = load_clip(proj.recording_path, sr=backend.spec.sample_rate)
    t0 = time.perf_counter()
    try:
        res = transcribe_audio(clip, settings, backend, tracker.progress, tracker.cancel_event)
    except Exception as e:  # noqa: BLE001
        print(f"transcription failed: {e}", file=sys.stderr)
        return 1
    run_id = proj.save_run(res.notes, {
        "run_id": res.run_id, "model": res.model, "model_version": res.model_version,
        "runtime": res.runtime, "settings": res.settings,
        "audio_length_s": res.audio_length_s, "processing_time_s": res.processing_time_s,
        "status": res.status, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    proj.adopt_run(run_id, keep_edits=False)
    print(f"run {run_id}: {len(res.notes)} notes in {res.processing_time_s:.1f}s "
          f"({res.audio_length_s / max(res.processing_time_s, 1e-9):.2f}x realtime) [{res.status}]")
    _print_notes(res.notes)
    return 0


def cmd_runs(args) -> int:
    proj = _project(args)
    runs = proj.list_runs()
    if not runs:
        print("no runs yet")
        return 0
    for r in runs:
        print(f"{r['run_id']}  {r['model']}  {r.get('status')}  {r['processing_time_s']:.1f}s  "
              f"{r.get('started_at', '')}")
    return 0


def cmd_notes(args) -> int:
    proj = _project(args)
    notes = proj.notes
    if args.uncertain:
        notes = [n for n in notes if n.confidence is not None and n.confidence < args.threshold]
        print(f"notes below confidence {args.threshold}:")
    _print_notes(notes, limit=args.limit)
    return 0


def cmd_edit(args) -> int:
    proj = _project(args)
    fields = {}
    if args.pitch is not None:
        fields["pitch"] = args.pitch
    if args.onset is not None:
        fields["onset"] = args.onset
    if args.end is not None:
        fields["end"] = args.end
    if args.velocity is not None:
        fields["velocity"] = args.velocity
    if args.delete:
        n = proj.delete_note(args.index)
        print(f"deleted note {args.index} ({midi_name(n.pitch)} @ {n.onset:.3f})")
        return 0
    if not fields:
        print("nothing to change (use --pitch/--onset/--end/--velocity/--delete)", file=sys.stderr)
        return 1
    n = proj.edit_note(args.index, **fields)
    print(f"edited note {args.index} -> {midi_name(n.pitch)} @ {n.onset:.3f}s .. {n.end:.3f}s")
    return 0


def cmd_undo(args) -> int:
    proj = _project(args)
    op = proj.undo()
    print(f"undone: {op.op if op else 'nothing to undo'}")
    return 0


def cmd_export(args) -> int:
    proj = _project(args)
    timing = proj.timing
    from .score_quantize import quantize_notes
    score = quantize_notes(proj.notes, timing, strength=args.strength, staff_split=args.split)
    if args.fmt == "midi":
        from .export_midi import export_midi
        path = proj.export_path(f"score_{time.strftime('%Y%m%d_%H%M%S')}.mid")
        export_midi(score, path)
    elif args.fmt == "musicxml":
        from .export_musicxml import export_musicxml
        path = proj.export_path(f"score_{time.strftime('%Y%m%d_%H%M%S')}.musicxml")
        export_musicxml(score, path, title=proj.meta().get("name", "Untitled transcription"))
    else:
        print(f"unknown format {args.fmt}", file=sys.stderr)
        return 1
    print(f"exported {path}")
    return 0


def cmd_evaluate(args) -> int:
    ref = load_note_json(Path(args.ref))
    pred = load_note_json(Path(args.pred))
    cfg = EvalConfig(tol_on=args.tol_on, tol_off=args.tol_off)
    res = evaluate_pair(ref, pred, cfg)
    print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0


def cmd_corpus(args) -> int:
    from .corpus import build_corpus
    manifest = build_corpus(Path(args.out), pieces=args.pieces, conditions=args.conditions,
                            seed=args.seed, overwrite=args.overwrite)
    print(f"corpus: {len(manifest)} clips -> {args.out}")
    from collections import Counter
    for split, n in Counter(m.split for m in manifest).items():
        print(f"  {split}: {n}")
    return 0


def cmd_levels(args) -> int:
    clip = load_clip(Path(args.audio))
    s = clip_stats(clip.mono_mix())
    print(json.dumps({"peak_dbfs": s.peak_dbfs, "rms_dbfs": s.rms_dbfs,
                      "clipping_frames": s.clipping_frames, "level_ok": s.level_ok, "hint": s.hint}, indent=2))
    return 0


# ------------------------------------------------------------------ main ---

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="piano-scribe", description="Local solo-piano transcription reference pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_project(sp) -> None:
        sp.add_argument("project", help="project directory")

    sp = sub.add_parser("new", help="create a project")
    sp.add_argument("path")
    sp.add_argument("--name", default=None)
    sp.add_argument("--tempo", type=float, default=120.0)
    sp.add_argument("--meter-num", type=int, default=4)
    sp.add_argument("--meter-den", type=int, default=4)
    sp.set_defaults(fn=cmd_new)

    sp = sub.add_parser("import", help="import a local audio file as the recording")
    add_project(sp)
    sp.add_argument("--audio", required=True)
    sp.set_defaults(fn=cmd_import)

    sp = sub.add_parser("record", help="record from the microphone")
    add_project(sp)
    sp.add_argument("--seconds", type=float, default=None, help="duration; omit to record until Ctrl+C")
    sp.add_argument("--device", type=int, default=None)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(fn=cmd_record)

    sp = sub.add_parser("calibrate", help="estimate the noise floor from the recorded silence sample")
    add_project(sp)
    sp.add_argument("--seconds", type=float, default=3.0)
    sp.set_defaults(fn=cmd_calibrate)

    sp = sub.add_parser("transcribe", help="run the local pipeline")
    add_project(sp)
    sp.add_argument("--model", choices=["basic-pitch", "bytedance"], default="basic-pitch")
    sp.add_argument("--min-pitch", type=int, default=21)
    sp.add_argument("--max-pitch", type=int, default=108)
    sp.add_argument("--onset", type=float, default=0.5)
    sp.add_argument("--frame", type=float, default=0.5)
    sp.add_argument("--min-len", type=float, default=0.05)
    sp.add_argument("--conf-threshold", type=float, default=0.25,
                    help="decoder confidence floor (model-relative; 0 disables)")
    sp.add_argument("--merge-gap", type=float, default=0.045)
    sp.add_argument("--chunk-s", type=float, default=45.0)
    sp.add_argument("--overlap-s", type=float, default=2.0)
    sp.add_argument("--denoise", action="store_true")
    sp.set_defaults(fn=cmd_transcribe)

    sp = sub.add_parser("runs", help="list transcription runs")
    add_project(sp)
    sp.set_defaults(fn=cmd_runs)

    sp = sub.add_parser("notes", help="show the working note set")
    add_project(sp)
    sp.add_argument("--uncertain", action="store_true")
    sp.add_argument("--threshold", type=float, default=0.6)
    sp.add_argument("--limit", type=int, default=200)
    sp.set_defaults(fn=cmd_notes)

    sp = sub.add_parser("edit", help="edit/delete a note (index from `notes`)")
    add_project(sp)
    sp.add_argument("index", type=int)
    sp.add_argument("--pitch", type=int)
    sp.add_argument("--onset", type=float)
    sp.add_argument("--end", type=float)
    sp.add_argument("--velocity", type=int)
    sp.add_argument("--delete", action="store_true")
    sp.set_defaults(fn=cmd_edit)

    sp = sub.add_parser("undo")
    add_project(sp)
    sp.set_defaults(fn=cmd_undo)

    sp = sub.add_parser("export", help="export score (midi|musicxml)")
    add_project(sp)
    sp.add_argument("--fmt", choices=["midi", "musicxml"], required=True)
    sp.add_argument("--strength", type=float, default=1.0)
    sp.add_argument("--split", type=int, default=60)
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("evaluate", help="compare reference and predicted note JSON")
    sp.add_argument("ref")
    sp.add_argument("pred")
    sp.add_argument("--tol-on", type=float, default=0.05)
    sp.add_argument("--tol-off", type=float, default=0.10)
    sp.set_defaults(fn=cmd_evaluate)

    sp = sub.add_parser("corpus", help="build the synthetic pilot corpus (Phase 1 test set)")
    sp.add_argument("out")
    sp.add_argument("--pieces", nargs="*", default=None)
    sp.add_argument("--conditions", nargs="*", default=None)
    sp.add_argument("--seed", type=int, default=20260919)
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(fn=cmd_corpus)

    sp = sub.add_parser("levels", help="input-level check for a WAV file")
    sp.add_argument("audio")
    sp.set_defaults(fn=cmd_levels)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except ProjectError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\naborted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())