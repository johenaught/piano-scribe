"""Rhythm and score interpretation (outline 3.7).

Performed timing is kept separate from written musical timing. Given a tempo
and meter (user-supplied at project creation), note events are quantized into
measures, note values, rests and ties; notes are allocated to a grand staff
(treble/bass) and to musical voices.

Deterministic v1 rules (documented, and the outline explicitly defers fancy
interpretation): no rubato modelling, no tuplets, no ornaments.
Staff split defaults to MIDI 60 (middle C) boundary; a fixed pitch split does
NOT reliably identify hands (outline 3.7), so borderline notes are flagged
``ambiguous_staff`` for the user to confirm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import NoteEvent, TimingMap

# MusicXML divisions: ticks per quarter note.
DIV = 480


@dataclass
class ScoreNote:
    pitch: int
    staff: str                       # 'treble' | 'bass'
    voice: int                       # per-staff voice index
    measure: int                     # 1-based
    beat: float                      # beat position within measure (1.0 = downbeat)
    dur_beats: float
    velocity: Optional[int] = None   # estimated intensity (not exact key velocity)
    tie_after: bool = False          # continues into the next measure (written tie)
    tie_from: bool = False
    ambiguous_staff: bool = False
    src_onset: float = 0.0           # original performed timing, always kept
    src_end: float = 0.0

    def duration_divs(self, meter_den: int) -> int:
        """Written duration in divisions (1 quarter note = DIV)."""
        return int(round(self.dur_beats * DIV * 4.0 / meter_den))


@dataclass
class Score:
    notes: list[ScoreNote]
    timing: TimingMap
    quantization_strength: float = 1.0   # 0..1 blend of snapped vs. raw timing
    grid_divs: int = 60                  # 32nd-note grid

    def beats_per_measure(self) -> float:
        return self.timing.beats_per_measure()


def _snap(value: float, grid: float) -> float:
    return round(value / grid) * grid


def quantize_notes(
    notes: list[NoteEvent],
    timing: TimingMap,
    strength: float = 1.0,
    staff_split: int = 60,
    grid: Optional[float] = None,
) -> Score:
    """Quantize performed notes onto the measure grid with ties.

    ``strength`` blends snapped and raw timing (1.0 = fully snapped);
    original onsets/ends are preserved on every ScoreNote (outline 3.7:
    adjustable quantization strength, preserve original timings).
    """
    spb = timing.seconds_per_beat()
    if grid is None:
        grid = 0.5  # eighth-note grid in beats
    snapped = []
    for n in sorted(notes, key=lambda x: (x.onset, x.pitch)):
        beat_raw = n.onset / spb + timing.pickup_beats
        beat_snapped = _snap(beat_raw, grid)
        beat = beat_raw * (1.0 - strength) + beat_snapped * strength
        dur_raw = max(n.end - n.onset, 0.05) / spb
        dur_snapped = _snap(dur_raw, grid)
        dur = max(dur_raw * (1.0 - strength) + dur_snapped * strength, grid * 0.5)
        snapped.append((beat, dur, n))
    snapped.sort(key=lambda x: (x[0], x[2].pitch))

    # measure layout
    bpm = timing.beats_per_measure()
    measure_len = bpm
    score_notes: list[ScoreNote] = []
    for beat, dur, n in snapped:
        measure = max(0, int(beat // measure_len))
        beat_in = beat - measure * measure_len
        if beat_in < 0.0 or beat_in >= measure_len + 1e-6:
            beat_in = 0.0  # guard: pickup offsets outside measure 1 collapse to the downbeat
        ambiguous = n.pitch in range(staff_split - 3, staff_split + 4)
        sn = ScoreNote(
            pitch=n.pitch,
            staff="treble" if n.pitch >= staff_split else "bass",
            voice=0,
            measure=measure + 1,
            beat=round(beat_in, 5),
            dur_beats=round(dur, 5),
            velocity=n.velocity,
            ambiguous_staff=ambiguous,
            src_onset=n.onset,
            src_end=n.end,
        )
        score_notes.append(sn)

    # split long notes across measure boundaries with ties
    tied: list[ScoreNote] = []
    for sn in score_notes:
        end_beat_in = sn.beat + sn.dur_beats
        while end_beat_in - measure_len > 1e-6:
            piece = ScoreNote(
                pitch=sn.pitch, staff=sn.staff, voice=sn.voice,
                measure=sn.measure, beat=sn.beat,
                dur_beats=round(measure_len - sn.beat, 5),
                tie_after=True, tie_from=sn.tie_from,
                ambiguous_staff=sn.ambiguous_staff,
                src_onset=sn.src_onset, src_end=sn.src_end,
            )
            tied.append(piece)
            sn = ScoreNote(
                pitch=sn.pitch, staff=sn.staff, voice=sn.voice,
                measure=sn.measure + 1, beat=0.0,
                dur_beats=round(end_beat_in - measure_len, 5),
                tie_after=False, tie_from=True,
                ambiguous_staff=sn.ambiguous_staff,
                src_onset=sn.src_onset, src_end=sn.src_end,
            )
            end_beat_in = sn.dur_beats
        tied.append(sn)
    score_notes = tied

    # voice assignment per staff: greedy horizontal streams (max 2 voices)
    by_staff: dict[str, list[ScoreNote]] = {"treble": [], "bass": []}
    for sn in score_notes:
        by_staff[sn.staff].append(sn)
    for staff, evs in by_staff.items():
        streams: list[list[ScoreNote]] = []
        for sn in sorted(evs, key=lambda x: (x.measure, x.beat)):
            placed = False
            for s in streams:
                last = s[-1]
                last_end = (last.measure - 1) * measure_len + last.beat + last.dur_beats
                this_start = (sn.measure - 1) * measure_len + sn.beat
                if last_end <= this_start + 1e-6:
                    s.append(sn)
                    placed = True
                    break
            if not placed:
                streams.append([sn])
        for si, s in enumerate(streams[:2]):
            for sn in s:
                sn.voice = si
        # any extra streams collapse into voice 1 (rare in v1)
        for s in streams[2:]:
            for sn in s:
                sn.voice = 1

    score_notes.sort(key=lambda x: (x.measure, x.beat, x.staff, x.pitch))
    return Score(notes=score_notes, timing=timing, quantization_strength=strength, grid_divs=int(grid * 480))