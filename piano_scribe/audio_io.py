"""Audio import and preparation (outline 3.2 / 3.3).

Raw audio is preserved unchanged; every transformation produces a
*derivative* so the pipeline keeps a raw path and denoising can be disabled
or compared.
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    HAVE_SF = True
except ImportError:  # pragma: no cover
    HAVE_SF = False

try:
    import librosa
    HAVE_LIBROSA = True
except ImportError:  # pragma: no cover
    HAVE_LIBROSA = False

# Model input convention for the built-in backends.
MODEL_SR = 22050
MODEL_MONO = True


@dataclass
class AudioClip:
    """Loaded audio plus provenance (outline 3.2: keep sample rate/channels)."""
    samples: np.ndarray            # float32, shape (n,) mono or (n, ch)
    sample_rate: int
    path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def mono_mix(self) -> np.ndarray:
        """Unweighted downmix; simple and predictable (outline 3.3)."""
        if self.samples.ndim == 1 or self.samples.shape[1] == 1:
            return self.samples.reshape(-1).astype(np.float32)
        return self.samples.mean(axis=1).astype(np.float32)

    def channels(self) -> int:
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[1]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def probe(path: Path) -> dict:
    """Read metadata without loading the samples (sf.info)."""
    if not HAVE_SF:
        raise RuntimeError("pip install soundfile")
    info = sf.info(str(path))
    return {
        "path": str(path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration": float(info.frames / info.samplerate),
        "format": info.format,
        "sha256": sha256_file(path),
    }


def load_clip(path, sr: Optional[int] = None, mono: bool = False) -> AudioClip:
    """Load a local audio file; resample to ``sr`` if requested."""
    if not HAVE_SF:
        raise RuntimeError("pip install soundfile")
    samples, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
    meta = probe(Path(path))
    clip = AudioClip(samples=samples, sample_rate=int(native_sr), path=Path(path), metadata=meta)
    if sr and sr != native_sr:
        clip.samples = resample(clip.samples, native_sr, sr, mono=mono)
        clip.sample_rate = sr
    elif mono and clip.samples.ndim == 2:
        clip.samples = clip.mono_mix()
    return clip


def save_wav(samples: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not HAVE_SF:
        raise RuntimeError("pip install soundfile")
    sf.write(str(path), samples, sr)
    return None


def read_wav_raw(path: Path) -> tuple[np.ndarray, int]:
    """Small built-in .wav reader used by the recording path (no deps)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        n = w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        raw = raw.reshape(-1, n_ch)
    return raw, sr


def write_wav_raw(samples: np.ndarray, sr: int, path: Path) -> None:
    """Built-in 16-bit PCM writer (no deps) for the safe-save recording path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if samples.ndim == 1:
        n_ch = 1
    else:
        samples = samples.reshape(-1)
        n_ch = 1
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(n_ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


@dataclass
class ClipStats:
    peak_dbfs: float
    rms_dbfs: float
    clipping_frames: int
    level_ok: bool
    hint: str


def clip_stats(samples: np.ndarray) -> ClipStats:
    """Input-level check (outline 2.C): clipping and weak-input detection."""
    s = np.asarray(samples, dtype=np.float64)
    if s.size == 0:
        return ClipStats(float("-inf"), float("-inf"), 0, False, "empty audio")
    peak = np.max(np.abs(s))
    rms = np.sqrt(np.mean(s**2)) if s.size else 0.0
    clip_frames = int(np.sum(np.abs(s) > 0.999))
    peak_db = 20.0 * np.log10(max(peak, 1e-9))
    rms_db = 20.0 * np.log10(max(rms, 1e-9))
    if clip_frames > 0:
        ok, hint = False, f"clipping detected ({clip_frames} frames at full scale)"
    elif peak_db < -24.0:
        ok, hint = False, f"very weak input (peak {peak_db:.1f} dBFS); move closer or raise gain"
    elif rms_db < -45.0:
        ok, hint = False, f"quiet signal (RMS {rms_db:.1f} dBFS)"
    else:
        ok, hint = True, f"levels ok (peak {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS)"
    return ClipStats(round(peak_db, 2), round(rms_db, 2), clip_frames, ok, hint)


def normalize_peak(samples: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Peak-normalize a derivative copy (never touches the original)."""
    s = np.asarray(samples, dtype=np.float32)
    peak = np.max(np.abs(s)) if s.size else 0.0
    if peak <= 1e-9:
        return s
    return (s * (target / peak)).astype(np.float32)


def resample(samples: np.ndarray, src_sr: int, dst_sr: int, mono: bool = False) -> np.ndarray:
    """Resample (librosa if present, else scipy), optionally downmixing."""
    if mono and samples.ndim == 2:
        x = samples.mean(axis=1)
    else:
        x = samples
    if src_sr == dst_sr:
        return np.asarray(x, dtype=np.float32)
    if HAVE_LIBROSA:
        return librosa.resample(x.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)
    from scipy.signal import resample_poly
    return resample_poly(x, dst_sr, src_sr).astype(np.float32)


def prepare_for_model(clip: AudioClip, denoise: bool = False) -> np.ndarray:
    """Produce the model-input derivative: mono, resampled, peak-normalized.

    Denoising is optional and conservative (spectral gate); the raw path is
    always available by passing ``denoise=False`` (outline 3.3).
    """
    x = clip.mono_mix()
    x = resample(x, clip.sample_rate, MODEL_SR, mono=True)
    if denoise:
        x = spectral_gate(x, MODEL_SR)
    return normalize_peak(x)


def spectral_gate(samples: np.ndarray, sr: int, noise_floor_db: Optional[float] = None) -> np.ndarray:
    """Very conservative noise gate: suppress only bins far below the floor.

    Designed to be A/B-tested against the raw path before any use
    (outline 3.3: apply only when evaluation shows benefit).
    """
    from scipy import signal

    if noise_floor_db is None:
        # estimate floor from the quietest 5% of STFT frames
        f, t, Z = signal.stft(samples, fs=sr, nperseg=2048, noverlap=1024)
        mag = np.abs(Z)
        floor_per_bin = np.percentile(mag, 5, axis=1, keepdims=True)
        noise_floor_db = 20.0 * np.log10(max(float(np.mean(floor_per_bin)), 1e-6))
    thr = 10.0 ** (noise_floor_db / 20.0)
    f, t, Z = signal.stft(samples, fs=sr, nperseg=2048, noverlap=1024)
    mag = np.abs(Z)
    mask = mag > (thr * 2.0)   # keep everything at least 6 dB above the floor
    Z = Z * mask
    _, x = signal.istft(Z, fs=sr, nperseg=2048, noverlap=1024)
    return np.asarray(x, dtype=np.float32)