"""Microphone capture for the desktop reference app (outline 3.2).

Uses sounddevice (PortAudio) with a raw, minimally-processed input path.
The recording is written via the dependency-free 16-bit writer as soon as
capture stops ('stop and save the original immediately', outline 2.G).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio_io import clip_stats, write_wav_raw

try:
    import sounddevice as sd
    HAVE_SOUNDDEVICE = True
except ImportError:  # pragma: no cover
    HAVE_SOUNDDEVICE = False


@dataclass
class RecordingSession:
    """Everything captured for one recording (persisted with the project)."""
    path: Path
    sample_rate: int
    channels: int
    device: str
    started_at: str
    duration_s: float
    clipping_frames: int
    peak_dbfs: float
    interrupted: bool = False
    interrupted_reason: str = ""


def available_input_devices() -> list[dict]:
    if not HAVE_SOUNDDEVICE:  # pragma: no cover
        return []
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append({"index": i, "name": d["name"], "channels": d["max_input_channels"], "sr": int(d["default_samplerate"])})
    return out


def record_microphone(
    out_path: Path,
    seconds: Optional[float] = None,
    sample_rate: int = 44100,
    channels: int = 1,
    device: Optional[int] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    stop_event: Optional[object] = None,
) -> RecordingSession:
    """Record from the default (or chosen) input device.

    ``seconds=None`` records until ``stop_event`` is set or Ctrl+C.
    Returns a RecordingSession; the WAV file is written before returning.
    """
    if not HAVE_SOUNDDEVICE:
        raise RuntimeError("Recording requires sounddevice: pip install 'piano-scribe[capture]'")

    dev = device if device is not None else sd.default.device[0]
    info = sd.query_devices(dev)
    sr = int(info["default_samplerate"]) if sample_rate is None else sample_rate
    chan = min(channels, int(info["max_input_channels"]))
    if chan < channels:
        channels = chan
    if channels < 1:  # pragma: no cover
        raise RuntimeError(f"device {dev} has no input channels")

    frames: list[np.ndarray] = []
    started = time.time()
    stop_flag = False
    interrupted = False
    reason = ""

    def callback(indata, frames_n, t, status):  # noqa: ARG001
        nonlocal stop_flag, interrupted
        frames.append(indata.copy())
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            interrupted = True
            raise sd.CallbackStop

    try:
        with sd.InputStream(samplerate=sr, device=dev, channels=channels,
                            dtype="float32", callback=callback):
            deadline = (started + seconds) if seconds else None
            while True:
                if deadline is not None and time.time() >= deadline:
                    break
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    interrupted = True
                    reason = "stop requested"
                    break
                if on_progress is not None and deadline is not None:
                    on_progress(min(1.0, (time.time() - started) / seconds))
                time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
        reason = "interrupted (Ctrl+C)"
    except sd.CallbackStop:
        pass

    if not frames:
        raise RuntimeError("no audio captured")

    audio = np.concatenate(frames, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_raw(audio, sr, out_path)   # save immediately (outline 2.G)

    stats = clip_stats(audio)
    session = RecordingSession(
        path=out_path, sample_rate=sr, channels=channels, device=str(info["name"]),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        duration_s=round(len(audio) / sr, 3), clipping_frames=stats.clipping_frames,
        peak_dbfs=stats.peak_dbfs, interrupted=interrupted, interrupted_reason=reason,
    )
    return session


def session_metadata(session: RecordingSession) -> dict:
    return {
        "path": str(session.path),
        "sample_rate": session.sample_rate,
        "channels": session.channels,
        "device": session.device,
        "started_at": session.started_at,
        "duration_s": session.duration_s,
        "clipping_frames": session.clipping_frames,
        "peak_dbfs": session.peak_dbfs,
        "interrupted": session.interrupted,
        "interrupted_reason": session.interrupted_reason,
    }