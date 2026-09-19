"""Core data types shared across the pipeline (outline section 5)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Span of the supported instrument (88-key piano: A0 .. C8).
MIDI_PIANO_MIN = 21
MIDI_PIANO_MAX = 108

# Free-text note names for editing/reports.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_name(pitch: int) -> str:
    """'A0'..'C8' style name for a MIDI pitch."""
    octave = pitch // 12 - 1
    return f"{NOTE_NAMES[pitch % 12]}{octave}"


# Correction state: kept alongside the interpreted score (outline 3.5/5).
class CorrectionState:
    UNTOUCHED = "untouched"   # straight from the model, not reviewed
    CONFIRMED = "confirmed"   # reviewed and kept
    EDITED = "edited"         # pitch/onset/end/velocity edited by user
    ADDED = "added"           # added by user
    DELETED = "deleted"       # deleted by user (kept for history/recovery)
    UNCERTAIN = "uncertain"   # flagged by decoder or review heuristics


@dataclass
class NoteEvent:
    """A single timed note in the AUDIO domain (performed timing).

    ``end`` is the acoustic note-off estimate (key release + pedal handling is
    recorded separately as ``sustain_pedal``; a written score duration is NOT
    this field -- see ScoreNote in score_quantize).
    """

    pitch: int                  # MIDI pitch, 21..108 for piano
    onset: float                # seconds, absolute in the recording
    end: float                  # seconds, > onset
    confidence: Optional[float] = None   # model confidence, 0..1 if available
    velocity: Optional[int] = None       # estimated intensity; NOT exact key velocity
    sustain_pedal: bool = False          # sound extended by sustain pedal
    state: str = CorrectionState.UNTOUCHED
    note_id: Optional[int] = None        # stable id within a project

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "NoteEvent":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class TranscribeSettings:
    """Settings for one transcription run (persisted per run, outline 5)."""
    model: str = "basic-pitch"           # backend key
    min_pitch: int = MIDI_PIANO_MIN
    max_pitch: int = MIDI_PIANO_MAX
    onset_threshold: float = 0.60        # tuned on real repertoire (see research/basic_pitch_study.md)
    frame_threshold: float = 0.45
    infer_onsets: bool = False           # basic-pitch: derive onsets from frame jumps (phantom source for piano)
    melodia_trick: bool = False          # melody post-filter; hurts chordal music
    min_note_length_s: float = 0.05      # decayed/unsupported blips below this are dropped
    min_confidence: float = 0.0          # decoder drops notes below this (model-relative)
    suppress_octave_leak: bool = True    # drop weak harmonic-echo detections at a strong note's onset
    merge_gap_s: float = 0.09            # same-pitch gap smaller than this => one note (echo-merge)
    chunk_s: float = 45.0                # processing chunk length
    overlap_s: float = 2.0               # chunk overlap for boundary continuity
    denoise: bool = False                # conservative spectral gate (off by default)
    calibration: Optional[dict] = field(default_factory=dict)  # noise floor snapshot

    def json(self) -> dict:
        return asdict(self)


@dataclass
class TimingMap:
    """User-supplied tempo/meter used by the score layer (outline 3.7)."""
    tempo_bpm: float = 120.0
    meter_num: int = 4
    meter_den: int = 4
    pickup_beats: float = 0.0            # offset in beats before measure 1

    def beats_per_measure(self) -> float:
        return self.meter_num * (4.0 / self.meter_den)

    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo_bpm


@dataclass
class RunInfo:
    """Metadata for one processing run (reproducibility, outline 11.11)."""
    run_id: str
    model: str
    model_version: str = ""
    settings: dict = field(default_factory=dict)
    started_at: str = ""
    duration_s: float = 0.0
    audio_length_s: float = 0.0
    processing_time_s: float = 0.0
    status: str = "ok"                   # ok | cancelled | error | partial


def dump_json(obj: Any, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)