"""Quantization + staff/voice assignment tests (outline 3.7)."""

import pytest

from piano_scribe.types import NoteEvent, TimingMap
from piano_scribe.score_quantize import quantize_notes


def n(pitch, onset, end, vel=90):
    return NoteEvent(pitch=pitch, onset=onset, end=end, velocity=vel)


def test_simple_4_4_grid():
    timing = TimingMap(tempo_bpm=120, meter_num=4, meter_den=4)  # 1 beat = 0.5 s
    score = quantize_notes([n(60, 0.0, 0.45), n(64, 0.5, 0.95)], timing)
    assert len(score.notes) == 2
    a, b = score.notes
    assert a.measure == 1 and a.beat == pytest.approx(0.0) and a.dur_beats == pytest.approx(1.0)
    assert b.beat == pytest.approx(1.0)


def test_staff_split_default_middle_c():
    timing = TimingMap()
    notes = [n(59, 0.0, 0.4), n(60, 0.0, 0.4), n(72, 0.0, 0.4)]
    score = quantize_notes(notes, timing)
    by = {x.pitch: x.staff for x in score.notes}
    assert by[59] == "bass" and by[60] == "treble" and by[72] == "treble"


def test_ambiguous_staff_flagged():
    timing = TimingMap()
    score = quantize_notes([n(60, 0.0, 0.4), n(67, 0.0, 0.4)], timing)
    assert score.notes[0].ambiguous_staff  # 60 within +/- 3 of split
    assert not score.notes[1].ambiguous_staff


def test_tie_across_measure():
    timing = TimingMap(tempo_bpm=120, meter_num=4, meter_den=4)  # measure = 2 s, 4 beats
    # ~4.4 beats long note starting at the downbeat -> 0.5 beats into measure 2
    score = quantize_notes([n(60, 0.0, 2.2)], timing)  # 4.4 beats -> snapped 4.5
    notes = sorted(score.notes, key=lambda x: x.measure)
    assert len(notes) == 2
    assert notes[0].tie_after and notes[1].tie_from
    assert notes[0].dur_beats == pytest.approx(4.0)
    assert notes[1].measure == 2 and notes[1].beat == pytest.approx(0.0)
    assert notes[1].dur_beats == pytest.approx(0.5)


def test_voices_sequence_same_staff():
    timing = TimingMap(tempo_bpm=120, meter_num=4, meter_den=4)
    # two interleaved treble streams (melody above accompaniment rhythm)
    notes = [
        n(72, 0.0, 1.4),   # stream A
        n(64, 0.25, 0.7),  # stream B
        n(65, 0.95, 1.4),  # stream B continues
        n(74, 1.5, 2.0),   # stream A continues
    ]
    score = quantize_notes(notes, timing)
    treble = sorted((x for x in score.notes if x.staff == "treble"), key=lambda x: x.beat)
    voices = {t.pitch: t.voice for t in treble}
    assert voices[72] == 0 and voices[74] == 0
    assert voices[64] == 1 and voices[65] == 1


def test_durations_and_velocity_kept():
    timing = TimingMap(tempo_bpm=120, meter_num=4, meter_den=4)
    score = quantize_notes([n(60, 0.0, 0.45, vel=77)], timing)
    assert score.notes[0].velocity == 77


def test_original_timings_preserved():
    timing = TimingMap(tempo_bpm=120, meter_num=4, meter_den=4)
    score = quantize_notes([n(60, 0.0, 0.41)], timing)
    assert score.notes[0].src_onset == pytest.approx(0.0)
    assert score.notes[0].src_end == pytest.approx(0.41)