"""Phase 2 benchmark runner: corpus -> baseline report (outline 7, 8, 11.11).

Usage:
    python scripts/benchmark.py <corpus_dir> [--out report.md] [--model basic-pitch]
                                 [--split test val] [--conditions clean noise_18 ...]
                                 [--chunk-s 45] [--jobs 1]

Writes <report dir>/benchmark.json (machine-readable) and benchmark.md (human).
Processing time per clip is measured (seconds per minute of audio).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piano_scribe import __version__  # noqa: E402
from piano_scribe.audio_io import load_clip  # noqa: E402
from piano_scribe.evaluate import EvalConfig, evaluate_pair  # noqa: E402
from piano_scribe.models import get_backend  # noqa: E402
from piano_scribe.corpus import load_labels, load_manifest  # noqa: E402
from piano_scribe.pipeline import transcribe_audio  # noqa: E402
from piano_scribe.types import TranscribeSettings  # noqa: E402


def run(corpus_dir: Path, out_dir: Path, model: str, splits: list[str],
        conditions: list[str], settings: TranscribeSettings,
        eval_cfg: EvalConfig) -> tuple[dict, dict]:
    manifest = load_manifest(corpus_dir / "manifest.json")
    entries = [
        m for m in manifest.values()
        if (not splits or m["split"] in splits)
        and (not conditions or m["condition"] in conditions)
    ]
    entries.sort(key=lambda m: m["id"])
    backend = get_backend(model)
    results: dict[str, dict] = {}
    timing: dict[str, float] = {}
    for i, m in enumerate(entries):
        clip_id = m["id"]
        print(f"[{i + 1}/{len(entries)}] {clip_id}", flush=True)
        try:
            clip = load_clip(Path(m["audio_path"]), sr=backend.spec.sample_rate)
            t0 = time.perf_counter()
            res = transcribe_audio(clip, settings, backend)
            elapsed = time.perf_counter() - t0
            ref = load_labels(Path(m["label_path"]))
            ev = evaluate_pair(ref, res.notes, eval_cfg)
            results[clip_id] = {
                "piece": m["piece"], "condition": m["condition"], "split": m["split"],
                "tags": m["tags"], "duration_s": clip.duration,
                "n_ref": ev.n_ref, "n_pred": ev.n_pred, "tp": ev.tp, "fp": ev.fp, "fn": ev.fn,
                "octave_errors": ev.octave_errors, "end_mismatch": ev.end_mismatch,
                "precision": ev.precision, "recall": ev.recall, "f1": ev.f1,
                "onset_mae_s": ev.onset_mae, "end_mae_s": ev.end_mae,
                "processing_time_s": elapsed,
                "audio_seconds_per_minute": elapsed / clip.duration * 60.0,
                "status": "ok",
            }
        except Exception as e:  # noqa: BLE001
            results[clip_id] = {"status": "error", "error": str(e), "piece": m["piece"],
                                "condition": m["condition"], "split": m["split"], "tags": m["tags"]}
            traceback.print_exc()
    return results, {"model": model, "piano_scribe": __version__,
                     "settings": settings.json(), "tol_on_s": eval_cfg.tol_on,
                     "tol_off_s": eval_cfg.tol_off}


def aggregate(results: dict[str, dict]) -> dict:
    ok = [r for r in results.values() if r.get("status") == "ok"]
    if not ok:
        return {"clips": len(results), "ok": 0}
    tp = sum(r["tp"] for r in ok)
    fp = sum(r["fp"] for r in ok)
    fn = sum(r["fn"] for r in ok)
    ons = [r["onset_mae_s"] for r in ok if r.get("onset_mae_s") is not None]
    ends = [r["end_mae_s"] for r in ok if r.get("end_mae_s") is not None]
    times = [r["audio_seconds_per_minute"] for r in ok]
    n_pred = sum(r["n_pred"] for r in ok)
    n_ref = sum(r["n_ref"] for r in ok)
    return {
        "clips": len(results), "ok_clips": len(ok),
        "ref_notes": n_ref, "pred_notes": n_pred,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / n_pred, 4) if n_pred else 0.0,
        "recall": round(tp / n_ref, 4) if n_ref else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else 0.0,
        "onset_mae_s": round(sum(ons) / len(ons), 4) if ons else None,
        "end_mae_s": round(sum(ends) / len(ends), 4) if ends else None,
        "end_mismatch": sum(r.get("end_mismatch", 0) for r in ok),
        "octave_errors": sum(r["octave_errors"] for r in ok),
        "seconds_per_minute_audio": round(sum(times) / len(times), 3) if times else None,
    }


def group(results: dict[str, dict], key: str) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for cid, r in results.items():
        vals = r.get(key)
        if isinstance(vals, str):
            vals = [vals]
        for v in vals or []:
            groups.setdefault(v, {})[cid] = r
    return {g: aggregate(sub) for g, sub in sorted(groups.items())}


def fmt(a: dict) -> str:
    if a.get("ok_clips", 0) == 0:
        return f"{a.get('clips', 0)} clips (all failed)"
    return (f"{a['f1']:.3f} F1  P {a['precision']:.3f} R {a['recall']:.3f}  "
            f"tp {a['tp']} fp {a['fp']} fn {a['fn']}  "
            f"|Δonset| {a['onset_mae_s']}s |Δend| {a['end_mae_s']}s (mism {a['end_mismatch']})  "
            f"oct {a['octave_errors']}  proc {a['seconds_per_minute_audio']}s/min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default="basic-pitch", choices=["basic-pitch", "bytedance"])
    ap.add_argument("--split", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--chunk-s", type=float, default=45.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="decoder confidence floor (0 disables; model-relative)")
    ap.add_argument("--tol-on", type=float, default=0.05)
    ap.add_argument("--tol-off", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = (args.out or args.corpus / "reports").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = TranscribeSettings(model=args.model, chunk_s=args.chunk_s,
                                  overlap_s=args.overlap_s, min_confidence=args.conf)
    eval_cfg = EvalConfig(tol_on=args.tol_on, tol_off=args.tol_off)

    t0 = time.time()
    results, meta = run(args.corpus, out_dir, args.model, args.split, args.conditions, settings, eval_cfg)
    meta["wall_clock_s"] = round(time.time() - t0, 1)

    stats = {
        "meta": meta,
        "overall": aggregate(results),
        "by_condition": group(results, "condition"),
        "by_split": group(results, "split"),
        "by_tag": group(results, "tags"),
        "per_clip": {cid: r for cid, r in sorted(results.items())},
    }
    (out_dir / "benchmark.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    lines = [
        "# Piano transcription benchmark", "",
        f"- pipeline: piano-scribe {meta['piano_scribe']}",
        f"- model: {meta['model']}  tolerances: onset ±{meta['tol_on_s']}s, end ±{meta['tol_off_s']}s",
        f"- settings: {json.dumps(meta['settings'])}",
        f"- date: {time.strftime('%Y-%m-%d %H:%M:%S')}  wall clock: {meta['wall_clock_s']}s", "",
        "## Overall", "", f"| {fmt(stats['overall'])} |", "",
        "## By condition", "",
    ]
    for k, v in stats["by_condition"].items():
        lines.append(f"- **{k}**: {fmt(v)}")
    lines += ["", "## By split", ""]
    for k, v in stats["by_split"].items():
        lines.append(f"- **{k}**: {fmt(v)}")
    lines += ["", "## By tag (musical case)", ""]
    for k, v in stats["by_tag"].items():
        lines.append(f"- **{k}**: {fmt(v)}")
    lines += ["", "## Per clip", ""]
    lines.append("| id | split | cond | n_ref | n_pred | tp | fp | fn | oct | F1 | proc s/min |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cid, r in results.items():
        if r.get("status") != "ok":
            lines.append(f"| {cid} | {r.get('split')} | {r.get('condition')} | - | - | - | - | - | - | - | ERROR: {r.get('error', '')} |")
            continue
        lines.append(f"| {cid} | {r['split']} | {r['condition']} | {r['n_ref']} | {r['n_pred']} | {r['tp']} | "
                     f"{r['fp']} | {r['fn']} | {r['octave_errors']} | {r['f1']:.3f} | {r['audio_seconds_per_minute']:.1f} |")
    (out_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nOverall:  " + fmt(stats["overall"]))
    print(f"report -> {out_dir / 'benchmark.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())