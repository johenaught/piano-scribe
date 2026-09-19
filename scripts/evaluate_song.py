"""Song-level evaluation: real repertoire through the whole chain.

MIDI (same engraving as the published sheet music) -> score-derived labels ->
render audio with the piano-scribe synth -> transcribe with the local model ->
compare. This is the Phase 1/2 realistic-content check: the notes on the
published score are the ground truth; every miss and every phantom note is a
deliverable finding.

Usage:
    python scripts/evaluate_song.py data/songs/*.mid [--conf 0.25]
                                       [--tol-on 0.05] [--out data/songs/report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from piano_scribe.audio_io import save_wav
from piano_scribe.evaluate import EvalConfig, evaluate_pair
from piano_scribe.models import get_backend
from piano_scribe.pipeline import transcribe_audio
from piano_scribe.synth import render_notes
from piano_scribe.types import NoteEvent, TranscribeSettings, midi_name


def midi_to_labels(path: Path) -> tuple[list[NoteEvent], dict]:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    notes: list[NoteEvent] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append(NoteEvent(
                pitch=int(n.pitch),
                onset=float(n.start),
                end=float(n.end),
                velocity=max(1, min(127, int(round(n.velocity or 64)))),
            ))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    tempo = pm.estimate_tempo()
    meta = {
        "path": str(path),
        "name": path.stem,
        "tempo_est": round(float(tempo), 1) if tempo else None,
        "n_notes": len(notes),
        "duration_s": round(max((n.end for n in notes), default=0.0), 2),
    }
    return notes, meta


def evaluate_song(path: Path, settings: TranscribeSettings, eval_cfg: EvalConfig,
                  render: bool = True, reuse_render: bool = False) -> dict:
    labels, meta = midi_to_labels(path)
    backend = get_backend(settings.model)
    tmp = path.with_suffix(".render.wav")
    if render:
        if reuse_render and tmp.exists():
            audio_path = tmp
        else:
            audio = render_notes(labels)
            save_wav(audio, 22050, tmp)
            audio_path = tmp
    else:
        audio_path = None

    from piano_scribe.audio_io import load_clip
    clip = load_clip(audio_path, sr=backend.spec.sample_rate)
    t0 = time.perf_counter()
    res = transcribe_audio(clip, settings, backend)
    elapsed = time.perf_counter() - t0
    ev = evaluate_pair(labels, res.notes, eval_cfg)
    ev_loose = evaluate_pair(labels, res.notes, EvalConfig(tol_on=0.10, tol_off=eval_cfg.tol_off))

    # diff details
    matched_ref = {m[0] for m in ev.matched}
    matched_pred = {m[1] for m in ev.matched}
    missed = [labels[i] for i in range(len(labels)) if i not in matched_ref]
    phantoms = [res.notes[i] for i in range(len(res.notes)) if i not in matched_pred]
    toj = lambda n: {"note": midi_name(n.pitch), "pitch": n.pitch,
                     "onset": round(n.onset, 3), "end": round(n.end, 3),
                     "conf": round(n.confidence, 3) if n.confidence is not None else None}

    return {
        "meta": meta,
        "result": {k: v for k, v in ev.to_dict().items()},
        "result_loose_100ms": {"f1": round(ev_loose.f1, 3), "precision": round(ev_loose.precision, 3),
                               "recall": round(ev_loose.recall, 3), "tp": ev_loose.tp,
                               "fp": ev_loose.fp, "fn": ev_loose.fn},
        "n_missed": len(missed),
        "n_phantoms": len(phantoms),
        "missed": [toj(n) for n in missed[:25]],
        "phantoms": [toj(n) for n in phantoms[:25]],
        "processing_s": round(elapsed, 2),
        "proc_s_per_min": round(elapsed / meta["duration_s"] * 60.0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("midis", nargs="+", type=Path)
    ap.add_argument("--model", default="basic-pitch")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="decoder confidence floor (model-relative)")
    ap.add_argument("--min-len", type=float, default=0.05)
    ap.add_argument("--tol-on", type=float, default=0.05)
    ap.add_argument("--tol-off", type=float, default=0.10)
    ap.add_argument("--reuse-render", action="store_true",
                    help="reuse the last rendered WAV (fast threshold sweeps)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    settings = TranscribeSettings(model=args.model, min_confidence=args.conf,
                                  min_note_length_s=args.min_len)
    eval_cfg = EvalConfig(tol_on=args.tol_on, tol_off=args.tol_off)
    report = {"settings": settings.json(), "eval": {"tol_on_s": args.tol_on},
              "songs": {}}
    for mid in args.midis:
        print(f"--- {mid.name} ---", flush=True)
        try:
            r = evaluate_song(mid, settings, eval_cfg, reuse_render=args.reuse_render)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            report["songs"][mid.stem] = {"error": str(e)}
            continue
        m = r["meta"]
        res = r["result"]
        print(f"  {m['name']}: {m['duration_s']}s {m['n_notes']} score notes | "
              f"F1 {res['f1']:.3f} P {res['precision']:.3f} R {res['recall']:.3f} | "
              f"F1@100ms {r['result_loose_100ms']['f1']:.3f} | "
              f"missed {r['n_missed']} phantom {r['n_phantoms']} | "
              f"|dOnset| {res['onset_mae']}s |dEnd| {res['end_mae']}s "
              f"(end-mism {res['end_mismatch']}) | {r['proc_s_per_min']}s/min")
        if r["missed"]:
            print("   missed (first 8): " + ", ".join(
                f"{x['note']}@{x['onset']}s" for x in r["missed"][:8]))
        if r["phantoms"]:
            print("   phantoms (first 8): " + ", ".join(
                f"{x['note']}@{x['onset']}s(c{x['conf']})" for x in r["phantoms"][:8]))
        report["songs"][mid.stem] = r

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())