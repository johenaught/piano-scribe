"""Audio I/O and corpus determinism tests."""

import json

import numpy as np
import pytest

from piano_scribe.audio_io import AudioClip, clip_stats, load_clip, prepare_for_model, save_wav
from piano_scribe.corpus import build_corpus, PIECES
from piano_scribe.synth import render_notes
from piano_scribe.types import NoteEvent


def test_load_clip_resamples(tmp_path):
    sr = 44100
    wav = tmp_path / "x.wav"
    n = 1 + int(0.5 * sr)
    t = np.arange(n) / sr
    save_wav((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, wav)
    clip = load_clip(wav, sr=22050)
    assert clip.sample_rate == 22050
    assert clip.duration == pytest.approx(0.5, abs=0.02)


def test_prepare_for_model_mixes_and_normalizes():
    rng = np.random.default_rng(1)
    s = np.stack([rng.normal(0, 0.1, 44100), rng.normal(0, 0.05, 44100)], axis=1).astype(np.float32)
    clip = AudioClip(samples=s, sample_rate=44100)
    x = prepare_for_model(clip)
    assert x.ndim == 1
    assert np.max(np.abs(x)) <= 0.96


def test_audio_stats_clipping_detection():
    s = np.zeros(44100, dtype=np.float32)
    s[100] = 1.5          # clipped frame
    s[200] = 0.5
    stats = clip_stats(s)
    assert stats.clipping_frames > 0
    assert not stats.level_ok


def test_render_is_deterministic():
    notes = [NoteEvent(pitch=60, onset=0.2, end=0.8, velocity=90)]
    a = render_notes(notes)
    b = render_notes(notes)
    assert np.array_equal(a, b)


def test_corpus_rebuild_keeps_hashes(tmp_path):
    m1 = build_corpus(tmp_path, pieces=["sparse_silence"], conditions=["clean"])
    sha1 = m1[0].sha256_audio
    audio_bytes1 = (tmp_path / "audio" / "sparse_silence__clean.wav").read_bytes()
    m2 = build_corpus(tmp_path, pieces=["sparse_silence"], conditions=["clean"])
    assert m2[0].sha256_audio == sha1
    assert (tmp_path / "audio" / "sparse_silence__clean.wav").read_bytes() == audio_bytes1
    label = json.loads((tmp_path / "labels" / "sparse_silence__clean.json").read_text())
    assert label["provenance"].startswith("machine-verified")
    assert len(label["notes"]) > 0


def test_corpus_pieces_have_valid_notes():
    rng = np.random.default_rng(7)
    for p in PIECES:
        notes = p.build(rng)
        assert len(notes) >= 5, p.pid
        assert all(21 <= n.pitch <= 108 for n in notes), p.pid
        assert all(n.end > n.onset for n in notes), p.pid