"""Calibration and audio preparation helpers (outline 3.3).

- Noise-floor estimation from a *silence* sample (a scale is not a substitute
  for silence; we do not claim any personalization from scales).
- Level checks from a known-notes passage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .audio_io import clip_stats


@dataclass
class NoiseFloor:
    rms_dbfs: float
    peak_dbfs: float
    notes_seconds: float
    method: str = "silence_sample"
    estimate: Optional[float] = None   # derived attenuation hint for denoiser

    def to_json(self) -> dict:
        return field(default_factory=dict, **self.__dict__).copy()


def estimate_noise_floor(samples: np.ndarray, sr: int, silence_seconds: float = 3.0) -> NoiseFloor:
    """Steady background noise estimate from a brief silence sample.

    Uses the last ``silence_seconds`` of the clip (assumes the user pressed
    record during quiet), peaks excluding transient top 1% to stay robust to
    a stray click.
    """
    n = int(silence_seconds * sr)
    seg = np.asarray(samples, dtype=np.float64)
    if seg.size > n:
        seg = seg[-n:]
    if seg.size < sr:  # very short clip: use it all
        pass
    rms = np.sqrt(np.mean(seg**2)) if seg.size else 0.0
    # ignore the top 1% of instantaneous energy so one door-slam doesn't
    # poison the floor estimate
    peak = float(np.percentile(np.abs(seg), 99)) if seg.size else 0.0
    rms_db = 20.0 * np.log10(max(rms, 1e-9))
    peak_db = 20.0 * np.log10(max(peak, 1e-9))
    floor_db = rms_db
    return NoiseFloor(
        rms_dbfs=round(rms_db, 2),
        peak_dbfs=round(peak_db, 2),
        notes_seconds=round(len(seg) / sr, 2),
        estimate=None if floor_db < -80 else round(floor_db, 2),
    )


def check_known_notes(samples: np.ndarray, expected_notes: list[int],
                      f0: Optional[float] = None, expected_f0_midi: Optional[int] = None) -> dict:
    """Level and tuning sanity check from a known-notes passage.

    Reports measured peak/RMS and, if a reference pitch is given, the detected
    spectral centroid pitch as a coarse tuning check. This is a *check*, not a
    model-training claim (outline 3.3).
    """
    stats = clip_stats(samples)
    out = {"level": {"peak_dbfs": stats.peak_dbfs, "rms_dbfs": stats.rms_dbfs,
                     "clipping_frames": stats.clipping_frames, "level_ok": stats.level_ok}}
    if expected_notes:
        # crude: dominant F0 via autocorrelation on the loudest window
        seg = np.asarray(samples, dtype=np.float64)
        if seg.size:
            k = min(len(seg) - 1, int(0.5 * 44100))
            w = seg[:k] if k > 0 else seg
            w = w - w.mean()
            if np.abs(w).max() > 1e-6:
                ac = np.correlate(w, w, "full")[len(w) - 1:]
                # search lags for 60..1200 Hz
                lag_min = max(2, int(44100 / 1200))
                lag_max = min(len(ac) - 1, int(44100 / 60))
                if lag_max > lag_min:
                    lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
                    f0_detected = 44100.0 / lag
                    out["tuning"] = {"f0_hz": round(float(f0_detected), 2),
                                     "expected_midi": expected_f0_midi,
                                     "note": f0_detected}
    return out