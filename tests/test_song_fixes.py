"""Tests for the song-testing fixes: octave-leak suppression and
notation-end sharpening ("timing off" / "phantom notes" fixes)."""

import pytest

from piano_scribe.decode import suppress_octave_leak
from piano_scribe.score_quantize import sharpened_for_notation
from piano_scribe.types import NoteEvent, TranscribeSettings


def n(pitch, on, end, conf=0.8, vel=80):
    return NoteEvent(pitch=pitch, onset=on, end=end, confidence=conf, velocity=vel)


def settings(**kw):
    kw.setdefault("min_pitch", 21)
    kw.setdefault("max_pitch", 108)
    kw.setdefault("min_note_length_s", 0.05)
    kw.setdefault("merge_gap_s", 0.045)
    return TranscribeSettings(**kw)


# ------------------------------------------------- octave-leak suppression ---

def test_weak_octave_echo_removed():
    # classic harmonic leakage: E4 strong, weak E5 echo at the same instant
    notes = [n(64, 1.0, 1.5, conf=0.82), n(76, 1.005, 1.5, conf=0.30)]
    out = suppress_octave_leak(notes, settings())
    assert [x.pitch for x in out] == [64]


def test_suboctave_echo_removed_too():
    notes = [n(64, 1.0, 1.5, conf=0.86), n(52, 1.003, 1.5, conf=0.28)]
    out = suppress_octave_leak(notes, settings())
    assert [x.pitch for x in out] == [64]


def test_real_octave_doubling_preserved():
    # genuine octaves in the score are played at comparable strength
    notes = [n(60, 1.0, 1.5, conf=0.80), n(72, 1.002, 1.5, conf=0.75)]
    out = suppress_octave_leak(notes, settings())
    assert {x.pitch for x in out} == {60, 72}


def test_no_strong_note_keeps_everything():
    notes = [n(55, 1.0, 1.5, conf=0.4), n(67, 1.01, 1.5, conf=0.35)]
    out = suppress_octave_leak(notes, settings())
    assert {x.pitch for x in out} == {55, 67}


def test_suppression_can_be_disabled():
    s = settings()
    s.suppress_octave_leak = False
    notes = [n(64, 1.0, 1.5, conf=0.82), n(76, 1.005, 1.5, conf=0.30)]
    assert len(suppress_octave_leak(notes, s)) == 2


def test_weak_fifth_above_removed():
    # 3rd harmonic (19 semitones up) is the strongest non-octave overtone
    notes = [n(64, 1.0, 1.5, conf=0.84), n(83, 1.004, 1.5, conf=0.30)]
    out = suppress_octave_leak(notes, settings())
    assert [x.pitch for x in out] == [64]


def test_weak_sixteenth_kept():
    # a minor third (3 semitones) is NOT a harmonic of the fundamental
    notes = [n(64, 1.0, 1.5, conf=0.85), n(67, 1.01, 1.5, conf=0.31)]
    out = suppress_octave_leak(notes, settings())
    assert {x.pitch for x in out} == {64, 67}


def test_strong_harmonics_kept():
    # loud chord tones at harmonic intervals are all kept
    notes = [n(48, 1.0, 1.5, conf=0.8), n(60, 1.0, 1.5, conf=0.78), n(67, 1.01, 1.5, conf=0.7)]
    out = suppress_octave_leak(notes, settings())
    assert {x.pitch for x in out} == {48, 60, 67}


def test_non_octave_weak_note_kept():
    # weak FIFTH at same onset (could be a real chord tone) is preserved
    notes = [n(64, 1.0, 1.5, conf=0.85), n(71, 1.01, 1.5, conf=0.30)]
    out = suppress_octave_leak(notes, settings())
    assert {x.pitch for x in out} == {64, 71}


# ------------------------------------------------- notation-end sharpening ---

def test_overlapping_same_pitch_capped():
    # struck again before the decay died: written rhythm must not overlap
    notes = [n(60, 1.0, 2.8), n(60, 2.0, 3.1)]
    out = sharpened_for_notation(notes, gap_s=0.03)
    assert out[0].end == pytest.approx(1.97)
    assert out[1].end == pytest.approx(3.1)   # last note keeps its end


def test_non_overlapping_untouched():
    notes = [n(60, 1.0, 1.4), n(60, 2.0, 2.4)]
    out = sharpened_for_notation(notes)
    assert out[0].end == pytest.approx(1.4)


def test_chord_notes_not_capped():
    # different pitches in one chord keep their ends (no same-pitch overlap)
    notes = [n(60, 1.0, 2.0), n(64, 1.0, 2.0), n(67, 1.0, 2.0)]
    out = sharpened_for_notation(notes)
    assert {x.pitch: x.end for x in out} == {60: 2.0, 64: 2.0, 67: 2.0}


def test_trill_pairs_sharpen_to_clean_gaps():
    notes = [n(72, 1.0, 1.06, conf=0.9), n(72, 1.09, 1.15, conf=0.9)]
    out = sharpened_for_notation(notes, gap_s=0.02)
    assert out[0].end <= 1.07
    assert out[0].end < out[1].onset


def test_sharpening_keeps_original_acoustic_info():
    notes = [n(60, 1.0, 2.8), n(60, 2.0, 3.1)]
    out = sharpened_for_notation(notes)
    assert out[0].end == pytest.approx(1.97)
    # confidence/velocity carried over
    assert out[0].confidence == notes[0].confidence
    assert out[0].velocity == notes[0].velocity