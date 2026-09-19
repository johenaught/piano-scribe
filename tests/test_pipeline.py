"""Pipeline chunking/stitching tests (outline 6): absolute timestamps and
boundary continuity with overlapping windows."""

import numpy as np
import pytest

from piano_scribe.models import TranscriptionModel
from piano_scribe.pipeline import transcribe_audio
from piano_scribe.types import NoteEvent, TranscribeSettings
from piano_scribe.audio_io import AudioClip


class FakeBackend(TranscriptionModel):
    """Emits chunk-relative notes (the real contract) for a fixed set of
    absolute events, including a note whose attack precedes a chunk boundary
    and is re-detected by the next chunk with a small onset lag."""

    ABS_EVENTS = [
        (60, 0.5, 1.0, 0.90),     # plain note
        (60, 42.9, 45.4, 0.91),   # attack just before the first boundary (43.0)
        (64, 44.9, 45.1, 0.80),   # entirely inside the first overlap region
    ]

    def __init__(self, sr=22050):
        self.spec = type("Spec", (), {
            "name": "fake", "version": "0", "runtime": "test",
            "sample_rate": sr, "min_pitch": 0, "max_pitch": 127,
            "license_note": "", "notes": "",
        })()
        self._calls = 0

    def transcribe(self, samples, settings, progress_cb=None, cancel_event=None):
        t0 = self._calls * (settings.chunk_s - settings.overlap_s)
        self._calls += 1
        win_end = t0 + settings.chunk_s
        rel = []
        for pitch, a, e, c in self.ABS_EVENTS:
            if e <= t0 or a >= win_end:      # no overlap with this window
                continue
            rel.append(NoteEvent(
                pitch=pitch,
                onset=max(0.0, a - t0),
                end=min(e, win_end) - t0,
                confidence=c,
            ))
        return rel


def test_absolute_timestamps_across_chunks():
    sr = 22050
    backend = FakeBackend(sr)
    settings = TranscribeSettings(chunk_s=45.0, overlap_s=2.0)
    audio = AudioClip(samples=np.zeros(48 * sr, dtype=np.float32), sample_rate=sr)
    res = transcribe_audio(audio, settings, backend)
    by_pitch = {n.pitch: n for n in res.notes}
    # boundary re-detection merges into the earlier note (rule B)
    assert by_pitch[60].onset == pytest.approx(42.9, abs=0.01)
    assert by_pitch[60].end == pytest.approx(45.4, abs=0.01)
    # overlap-region note detected by both chunks merges (rule A)
    assert by_pitch[64].onset == pytest.approx(44.9, abs=0.01)
    assert by_pitch[64].end == pytest.approx(45.1, abs=0.01)
    assert len([n for n in res.notes if n.pitch == 64]) == 1
    assert len(res.notes) == 3
    assert res.notes[0].onset == pytest.approx(0.5)


def test_result_matches_lone_notes_case():
    sr = 22050
    backend = FakeBackend(sr)
    settings = TranscribeSettings(chunk_s=100.0, overlap_s=2.0)  # single chunk over all notes
    audio = AudioClip(samples=np.zeros(110 * sr, dtype=np.float32), sample_rate=sr)
    res = transcribe_audio(audio, settings, backend)
    assert len(res.notes) == 3
    assert {round(n.pitch) for n in res.notes} == {60, 64}