"""Decoder temporal-consistency unit tests (outline 3.5)."""

import pytest

from piano_scribe.decode import clean_notes, dedupe, suppress_blips
from piano_scribe.types import NoteEvent, TranscribeSettings


def n(pitch, on, end, conf=0.8):
    return NoteEvent(pitch=pitch, onset=on, end=end, confidence=conf)


def settings(**kw):
    kw.setdefault("min_pitch", 21)
    kw.setdefault("max_pitch", 108)
    kw.setdefault("min_note_length_s", 0.05)
    kw.setdefault("merge_gap_s", 0.045)
    return TranscribeSettings(**kw)


def test_merges_overlapping_same_pitch():
    out = clean_notes([n(60, 1.0, 1.6), n(60, 1.3, 1.9)], settings())
    assert len(out) == 1
    assert out[0].onset == pytest.approx(1.0)
    assert out[0].end == pytest.approx(1.9)


def test_preserves_genuine_repeated_notes():
    # gap 0.10s > merge_gap 0.045 -> two distinct notes
    out = clean_notes([n(60, 1.0, 1.3), n(60, 1.4, 1.7)], settings())
    assert len(out) == 2
    assert out[0].onset == pytest.approx(1.0)
    assert out[1].onset == pytest.approx(1.4)


def test_merges_kissing_detections():
    # gap exactly at the merge threshold boundary (0.03 < 0.045)
    out = clean_notes([n(60, 1.0, 1.3), n(60, 1.33, 1.6)], settings())
    assert len(out) == 1


def test_suppresses_ultra_short_blips():
    s = settings()
    out = suppress_blips([n(60, 1.0, 1.02), n(64, 2.0, 2.1)], s)
    assert [x.pitch for x in out] == [64]


def test_clips_out_of_range_pitches():
    out = clean_notes([n(12, 0.1, 0.5), n(110, 0.2, 0.6), n(60, 0.3, 0.7)], settings())
    assert [x.pitch for x in out] == [60]


def test_clips_to_audio_duration():
    out = clean_notes([n(60, 8.0, 12.0), n(64, 0.1, 0.5)], settings(), audio_duration=5.0)
    assert len(out) == 1
    assert out[0].end <= 5.01


def test_drops_inverted_duration():
    out = clean_notes([n(60, 2.0, 1.0)], settings())
    assert out == []


def test_time_offset_shifts_absolute_times():
    out = clean_notes([n(60, 1.0, 1.5)], settings(), time_offset=44.0)
    assert out[0].onset == pytest.approx(45.0)
    assert out[0].end == pytest.approx(45.5)


def test_dedupe():
    notes = [n(60, 1.0, 1.5, 0.9), n(60, 1.0, 1.5, 0.9)]
    assert len(dedupe(notes)) == 1


def test_merge_diff_pitch_never_merges():
    out = clean_notes([n(60, 1.0, 1.6), n(64, 1.1, 1.7)], settings())
    assert len(out) == 2