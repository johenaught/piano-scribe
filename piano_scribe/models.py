"""Replaceable transcription backends (outline 3.4).

A backend converts prepared audio into raw timed note events. Everything
above (pipeline, decode, evaluation) talks to the ``TranscriptionModel``
interface only, so the model can be swapped without touching the rest of
the app. We do NOT train a model here; we benchmark existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .types import NoteEvent, TranscribeSettings


class CancelledError(Exception):
    pass


@dataclass
class ModelSpec:
    name: str
    version: str
    runtime: str            # e.g. 'ONNX Runtime (CPU)'
    sample_rate: int        # expected input sample rate
    min_pitch: int
    max_pitch: int
    notes: str = "pitch, onset, end, confidence"
    license_note: str = ""


class TranscriptionModel:
    """Protocol: transcribe prepared mono audio -> list[NoteEvent]."""

    spec: ModelSpec

    def transcribe(
        self,
        samples: np.ndarray,                 # mono float32, model sample rate
        settings: TranscribeSettings,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event=None,
    ) -> list[NoteEvent]:
        raise NotImplementedError


_BASIC_PITCH_MODEL_PATH = None
_BASIC_PITCH_MODEL = None
_BASIC_PITCH_VERSION = ""


def _load_basic_pitch(force_runtime: str = "onnx"):
    """Load the basic-pitch ICASSP 2022 model, preferring the ONNX
    serialization executed by ONNX Runtime (the outline's Windows runtime).

    The pip package ships the TF/CoreML/TFLite/ONNX artifacts side by side;
    the ONNX file keeps the app lean (no TensorFlow) and gives an exact
    match with the planned native runtime on Windows (outline 4)."""
    global _BASIC_PITCH_MODEL_PATH, _BASIC_PITCH_MODEL, _BASIC_PITCH_VERSION
    if _BASIC_PITCH_MODEL is not None:
        return _BASIC_PITCH_MODEL
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        import importlib.metadata as md
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "basic-pitch not installed. Run: pip install 'piano-scribe[baseline]'"
        ) from e
    _BASIC_PITCH_MODEL_PATH = ICASSP_2022_MODEL_PATH
    try:
        _BASIC_PITCH_VERSION = md.version("basic-pitch")
    except Exception:
        _BASIC_PITCH_VERSION = "unknown"
    if force_runtime == "onnx":
        onnx_path = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
        if onnx_path.exists():
            _BASIC_PITCH_MODEL_PATH = onnx_path
    _BASIC_PITCH_MODEL = Model(_BASIC_PITCH_MODEL_PATH)
    return _BASIC_PITCH_MODEL


class BasicPitchBackend(TranscriptionModel):
    """Baseline per outline 3.4/11.1: Spotify Basic Pitch (Apache-2.0 code).

    On Windows the package defaults to the ONNX-serialized model executed by
    ONNX Runtime, which is also the planned Windows deployment runtime
    (outline 4). For evaluation we always read the raw note events rather
    than its MIDI writer, so the rest of the pipeline owns decoding.
    """

    def __init__(self) -> None:
        self.spec = ModelSpec(
            name="basic-pitch",
            version="icassp_2022 (package-shipped)",
            runtime="ONNX Runtime (Windows default)",
            sample_rate=22050,
            min_pitch=21,
            max_pitch=108,
            license_note="Apache-2.0 (code and model artifacts shipped by the package)",
        )

    def transcribe(self, samples, settings, progress_cb=None, cancel_event=None):
        model = _load_basic_pitch()

        if progress_cb:
            progress_cb(0.2, "inference (ONNX Runtime / TensorFlow)")
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise CancelledError("cancelled before model run")

        # Use basic-pitch's low-level decode so WE own its note-decoding knobs
        # (the high-level predict() hardcodes infer_onsets=True and
        # melodia_trick=True, both phantom sources for chordal piano).
        import tempfile
        from basic_pitch.inference import run_inference
        from basic_pitch.note_creation import model_output_to_notes
        from .audio_io import save_wav

        with tempfile.TemporaryDirectory() as td:
            chunk_path = Path(td) / "chunk.wav"
            save_wav(samples, self.spec.sample_rate, chunk_path)
            model_output = run_inference(chunk_path, model)
        _, note_events = model_output_to_notes(
            model_output,
            onset_thresh=settings.onset_threshold,
            frame_thresh=settings.frame_threshold,
            infer_onsets=settings.infer_onsets,
            min_note_len=int(round(settings.min_note_length_s * 1000
                                   / 1000 * (22050 / 256))),  # frames @10ms
            min_freq=midi_to_hz(settings.min_pitch),
            max_freq=midi_to_hz(settings.max_pitch),
            include_pitch_bends=False,
            melodia_trick=settings.melodia_trick,
            midi_tempo=120,
        )
        if progress_cb:
            progress_cb(0.9, "decoding note events")
        notes = []
        for ne in note_events:
            # (start_time, end_time, pitch, amplitude, pitch_bends)
            start, end, pitch, amplitude = ne[0], ne[1], ne[2], ne[3]
            notes.append(NoteEvent(
                pitch=int(round(pitch)),
                onset=float(start),
                end=float(end),
                confidence=float(amplitude),
                velocity=int(max(1, min(127, round(amplitude * 127)))),
            ))
        return notes


class ByteDanceBackend(TranscriptionModel):
    """Piano-specific research baseline (outline 11.1).

    Code: Apache-2.0 (bytedance/piano_transcription). The pretrained
    checkpoint is downloaded at first use from Zenodo record 4034264 --
    inspect its license separately for distribution (outline 10/11.1).
    PyTorch-based; Windows officially untested upstream. Optional install:
    pip install 'piano-scribe[bytedance]'.
    """

    def __init__(self) -> None:
        self.spec = ModelSpec(
            name="bytedance-piano-transcription",
            version="2020 checkpoint (zenodo.org/records/4034264)",
            runtime="PyTorch CPU (research baseline; expect ~50 MB + torch)",
            sample_rate=16000,
            min_pitch=21,
            max_pitch=108,
            license_note="Code Apache-2.0; checkpoint license on Zenodo record 4034264 — verify before distribution",
        )

    def transcribe(self, samples, settings, progress_cb=None, cancel_event=None):
        try:
            from piano_transcription_inference import PianoTranscription, sample_rate as _pt_sr
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "piano_transcription_inference not installed. Run: "
                "pip install 'piano-scribe[bytedance]' (checkpoint downloads from Zenodo on first use)"
            ) from e
        from scipy.signal import resample_poly

        if progress_cb:
            progress_cb(0.1, "loading checkpoint (first use downloads from Zenodo)")
        full = resample_poly(samples, _pt_sr, self.spec.sample_rate)
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise CancelledError("cancelled before model run")
        transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)
        est = transcriptor.transcribe(full, None)  # noqa: B023 - returns dict; None => no midi write
        events = []
        for ev in est.get("est_note_events", []):
            # dict: {'onset_time', 'offset_time', 'midi_note', 'velocity'}
            events.append(NoteEvent(
                pitch=int(round(ev["midi_note"])),
                onset=float(ev["onset_time"]),
                end=float(ev["offset_time"]),
                confidence=None,
                velocity=int(round(ev["velocity"])) if "velocity" in ev else None,
            ))
        return events


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


_BACKENDS = {
    "basic-pitch": BasicPitchBackend,
    "bytedance": ByteDanceBackend,
}


def get_backend(name: str) -> TranscriptionModel:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_BACKENDS)}")
    return _BACKENDS[name]()