"""Note decoding and temporal consistency (outline 3.5).

Turns raw model predictions (already frame-decoded into notes by the backend)
into clean note events:

- clip to the audio range and the supported piano range;
- merge overlapping detections of the same pitch (a note that flickers in
  frame-activation must become ONE note event) while preserving genuine
  repeated strikes (gap >= ``merge_gap_s`` is a new strike);
- suppress unsupported ultra-short detections;
- drop exact duplicates.

Stitching across processing-window boundaries lives in ``pipeline`` (it needs
the chunk geometry); the per-chunk cleanup lives here.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .types import NoteEvent, MIDI_PIANO_MIN, MIDI_PIANO_MAX, TranscribeSettings


def clean_notes(
    events: Iterable[NoteEvent],
    settings: TranscribeSettings,
    audio_duration: Optional[float] = None,
    time_offset: float = 0.0,
    window_start: Optional[float] = None,
) -> list[NoteEvent]:
    """Temporal-consistency cleanup for one (chunk of) decoded notes.

    ``time_offset`` shifts absolute timestamps (used when stitching chunks);
    ``window_start`` (absolute) drops notes the backend emitted outside the
    chunk's own time window.
    All rules are deterministic and documented; evaluation keeps the raw
    predictions for comparison (outline 3.6: retain uncorrected predictions).
    """
    out: list[NoteEvent] = []
    for ev in events:
        on = ev.onset + time_offset
        end = ev.end + time_offset
        if end <= on:
            continue
        if window_start is not None and on < window_start - 0.001:
            continue
        if audio_duration is not None:
            if on < -0.001 or on > audio_duration + 0.001:
                continue
            end = min(end, audio_duration + 0.01)
        if not (settings.min_pitch <= ev.pitch <= settings.max_pitch):
            continue
        if end - on < settings.min_note_length_s:
            continue  # unsupported short detection (outline 3.5)
        if settings.min_confidence > 0.0 and ev.confidence is not None and ev.confidence < settings.min_confidence:
            continue  # frame spatter with no acoustic support (skip models without confidence)
        out.append(NoteEvent(
            pitch=ev.pitch, onset=on, end=end,
            confidence=ev.confidence, velocity=ev.velocity,
            sustain_pedal=ev.sustain_pedal, state=ev.state,
        ))
    return _merge_same_pitch_overlaps(out, settings)


def _merge_same_pitch_overlaps(notes: list[NoteEvent], settings: TranscribeSettings) -> list[NoteEvent]:
    """Merge same-pitch detections that overlap or kiss within merge_gap_s.

    Genuine repeats survive: a same-pitch note whose onset is >= merge_gap_s
    after the previous note's end is a new strike and is kept as-is.
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n.pitch, n.onset))
    merged: list[NoteEvent] = []
    for n in notes:
        if merged and merged[-1].pitch == n.pitch and n.onset - merged[-1].end <= settings.merge_gap_s:
            prev = merged[-1]
            # keep the earlier onset, extend the end, keep best confidence
            conf = max([c for c in (prev.confidence, n.confidence) if c is not None] or [None])
            merged[-1] = NoteEvent(
                pitch=prev.pitch, onset=prev.onset, end=max(prev.end, n.end),
                confidence=conf,
                velocity=prev.velocity or n.velocity,
                sustain_pedal=prev.sustain_pedal or n.sustain_pedal,
                state=prev.state if prev.state != "untouched" else n.state,
            )
        elif merged and merged[-1].pitch == n.pitch and n.onset < merged[-1].end:
            # same pitch, overlapping but gap rule passed: genuine repeat
            # struck before the previous ring fully died away -- keep both
            merged.append(n)
        else:
            merged.append(n)
    return merged


def suppress_blips(notes: list[NoteEvent], settings: TranscribeSettings) -> list[NoteEvent]:
    """Drop unsupported ultra-short detections (outline 3.5)."""
    return [n for n in notes if n.end - n.onset >= settings.min_note_length_s]


def dedupe(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Drop exact duplicates (same pitch/onset/end from chunk overlap)."""
    seen = set()
    out = []
    for n in sorted(notes, key=lambda n: (n.pitch, n.onset, n.end)):
        key = (round(n.pitch), round(n.onset, 4), round(n.end, 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def match_pitch_notes(ref: Optional[int], pred: Optional[int]) -> bool:
    return ref == pred


def sort_by_time(notes: Iterable[NoteEvent]) -> list[NoteEvent]:
    return sorted(notes, key=lambda n: (n.onset, n.pitch))