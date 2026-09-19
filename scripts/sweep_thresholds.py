"""Threshold sweep on one song (fast: renders are reused).

Sweeps (onset_threshold, frame_threshold, min_confidence) over the cached
render and prints a compact table: F1 / P / R / phantoms / missed / onset MAE.

Run once per piece (after evaluate_song rendered it):
    python scripts/sweep_thresholds.py data/songs/fur_elise.mid
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from piano_scribe.audio_io import load_clip  # noqa: E402
from piano_scribe.evaluate import EvalConfig, evaluate_pair  # noqa: E402
from piano_scribe.models import get_backend  # noqa: E402
from piano_scribe.pipeline import transcribe_audio  # noqa: E402
from piano_scribe.types import TranscribeSettings  # noqa: E402

CONFIGS = [
    # (onset_threshold, frame_threshold, min_confidence, infer_onsets, melodia_trick)
    (0.50, 0.50, 0.25, True,  True),    # upstream defaults (reference)
    (0.50, 0.50, 0.25, False, True),    # infer_onsets OFF
    (0.60, 0.50, 0.25, False, True),
    (0.70, 0.60, 0.25, False, True),
    (0.60, 0.60, 0.25, False, False),   # + melodia OFF
    (0.70, 0.60, 0.25, False, False),
    (0.80, 0.60, 0.25, False, False),
    (0.70, 0.60, 0.30, False, False),
    (0.60, 0.50, 0.20, False, False),
    (0.70, 0.60, 0.20, False, False),
]
EVAL = EvalConfig(tol_on=0.05, tol_off=0.10)


def main() -> int:
    from evaluate_song import midi_to_labels

    mid = Path(sys.argv[1])
    backend = get_backend("basic-pitch")
    labels, meta = midi_to_labels(mid)
    render = mid.with_suffix(".render.wav")
    if not render.exists():
        print("no cached render; run evaluate_song once first", file=sys.stderr)
        return 1
    clip = load_clip(render, sr=backend.spec.sample_rate)

    print(f"{'onset':>5} {'frame':>5} {'conf':>5} {'inf':>3} {'mel':>3} | {'F1':>6} {'P':>6} {'R':>6} "
          f"{'miss':>5} {'phant':>6} {'dOnset':>7}")
    for onset, frame, conf, infer, melodia in CONFIGS:
        settings = TranscribeSettings(model="basic-pitch", onset_threshold=onset,
                                      frame_threshold=frame, min_confidence=conf,
                                      infer_onsets=infer, melodia_trick=melodia)
        t0 = time.perf_counter()
        res = transcribe_audio(clip, settings, backend)
        dt = time.perf_counter() - t0
        ev = evaluate_pair(labels, res.notes, EVAL)
        n_miss = ev.n_ref - ev.tp
        n_phant = ev.n_pred - ev.tp
        print(f"{onset:5.2f} {frame:5.2f} {conf:5.2f} {str(infer)[0]:>3} {str(melodia)[0]:>3} | "
              f"{ev.f1:6.3f} {ev.precision:6.3f} {ev.recall:6.3f} {n_miss:5d} {n_phant:6d} "
              f"{ev.onset_mae:7.4f}  [{dt / meta['duration_s'] * 60:5.2f}s/min]")
    return 0


if __name__ == "__main__":
    sys.exit(main())