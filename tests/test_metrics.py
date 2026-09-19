"""Metrics unit tests (outline 8): tolerances, octave errors, matching."""

from piano_scribe.evaluate import EvalConfig, evaluate_pair
from piano_scribe.types import NoteEvent


def n(pitch, on, end, conf=0.9):
    return NoteEvent(pitch=pitch, onset=on, end=end, confidence=conf)


def test_perfect_match():
    ref = [n(60, 1.0, 1.5), n(64, 2.0, 2.5)]
    pred = [n(60, 1.01, 1.51), n(64, 2.02, 2.52)]
    r = evaluate_pair(ref, pred)
    assert r.tp == 2 and r.fp == 0 and r.fn == 0
    assert r.f1 == 1.0
    assert r.onset_mae < 0.02


def test_missed_and_false():
    ref = [n(60, 1.0, 1.5), n(64, 2.0, 2.5)]
    pred = [n(60, 1.0, 1.5), n(67, 3.0, 3.5)]  # 64 missed, 67 false
    r = evaluate_pair(ref, pred)
    assert r.tp == 1 and r.fn == 1 and r.fp == 1
    assert r.recall == 0.5 and r.precision == 0.5


def test_octave_error_counted_separately():
    ref = [n(60, 1.0, 1.5)]
    pred = [n(72, 1.02, 1.5)]  # one octave up at the same onset
    r = evaluate_pair(ref, pred)
    assert r.tp == 0 and r.fp == 1 and r.octave_errors == 1


def test_onset_tolerance_boundary():
    cfg = EvalConfig(tol_on=0.05, tol_off=0.10)
    ref = [n(60, 1.0, 1.5)]
    r_ok = evaluate_pair(ref, [n(60, 1.049, 1.5)], cfg)
    r_no = evaluate_pair(ref, [n(60, 1.06, 1.5)], cfg)
    assert r_ok.tp == 1
    assert r_no.tp == 0


def test_duration_tolerance_reported_not_gating():
    cfg = EvalConfig(tol_on=0.05, tol_off=0.10)
    ref = [n(60, 1.0, 1.5)]
    ok = evaluate_pair(ref, [n(60, 1.0, 1.58)], cfg)
    bad = evaluate_pair(ref, [n(60, 1.0, 1.72)], cfg)
    # duration is NOT a match gate (acoustic end != key release, outline 11.2)
    assert ok.tp == 1 and ok.end_mismatch == 0
    assert bad.tp == 1 and bad.end_mismatch == 1


def test_repeat_not_matched_as_same_note():
    # same pitch twice: pred shorter -> second repeat is a false extra
    ref = [n(60, 1.0, 1.5), n(60, 2.0, 2.5)]
    pred = [n(60, 1.0, 1.5), n(60, 2.0, 2.5), n(60, 2.01, 2.3)]
    r = evaluate_pair(ref, pred)
    assert r.tp == 2 and r.fp == 1