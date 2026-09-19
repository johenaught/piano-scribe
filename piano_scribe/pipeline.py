"""Chunked transcription pipeline (outline 6).

Long recordings are processed in bounded chunks to control memory use;
note continuity and correct absolute timestamps are maintained across chunk
boundaries. Progress callbacks and a cancel event keep the pipeline usable
from UI threads; intermediate results can be recovered from the project
store (outline 5: save intermediate results).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .audio_io import AudioClip, prepare_for_model
from .decode import clean_notes, dedupe
from .models import CancelledError, TranscriptionModel
from .types import NoteEvent, TranscribeSettings


@dataclass
class TranscriptionResult:
    notes: list[NoteEvent]
    model: str
    model_version: str = ""
    runtime: str = ""
    settings: dict = field(default_factory=dict)
    audio_length_s: float = 0.0
    processing_time_s: float = 0.0
    status: str = "ok"                   # ok | cancelled
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def transcribe_audio(
    audio: AudioClip,
    settings: TranscribeSettings,
    backend: TranscriptionModel,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event=None,
) -> TranscriptionResult:
    """Run the full local pipeline over one audio clip.

    - prepares model input (mono, resampled, normalized; optional denoise),
    - splits into bounded chunks with overlap,
    - cleans notes per chunk (absolute timestamps restored),
    - stitches chunk boundaries,
    - runs a final global dedupe/merge pass.
    """
    if settings.chunk_s <= settings.overlap_s + 1.0:
        raise ValueError("chunk_s must exceed overlap_s + 1.0")
    start = time.perf_counter()
    x = prepare_for_model(audio, denoise=settings.denoise)
    seconds = len(x) / backend.spec.sample_rate
    if progress_cb:
        progress_cb(0.0, "preparing audio")
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        raise CancelledError("cancelled before processing")

    sr = backend.spec.sample_rate
    chunk_n = int(settings.chunk_s * sr)
    step_n = int((settings.chunk_s - settings.overlap_s) * sr)
    if len(x) <= chunk_n:
        chunks = [(0, x)]
    else:
        chunks = []
        t = 0
        while t < len(x):
            chunks.append((t, x[t : min(t + chunk_n, len(x))]))
            if t + chunk_n >= len(x):
                break
            t += step_n

    n_chunks = len(chunks)
    chunk_results: list[list[NoteEvent]] = []
    cancelled = False
    for i, (t0, seg) in enumerate(chunks):
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            cancelled = True
            break
        t0_s = t0 / sr
        seg_duration = t0_s + len(seg) / sr
        raw = backend.transcribe(seg, settings, progress_cb=None, cancel_event=cancel_event)
        cleaned = clean_notes(
            raw,
            settings,
            audio_duration=seg_duration,
            time_offset=t0_s,
            window_start=t0_s,
        )
        chunk_results.append(cleaned)
        if progress_cb:
            progress_cb((i + 1) / n_chunks * 0.85 + 0.05, f"processing chunk {i + 1}/{n_chunks}")

    if cancelled:
        notes = dedupe([n for c in chunk_results for n in c])
        if progress_cb:
            progress_cb(1.0, "cancelled")
        return TranscriptionResult(
            notes=notes, model=backend.spec.name, model_version=backend.spec.version,
            runtime=backend.spec.runtime, settings=settings.json(),
            audio_length_s=seconds, processing_time_s=time.perf_counter() - start,
            status="cancelled",
        )

    stitched = _stitch_chunks(chunk_results, [c[0] / sr for c in chunks], settings)
    notes = dedupe(stitched)
    notes.sort(key=lambda n: (n.onset, n.pitch))
    if progress_cb:
        progress_cb(1.0, "done")
    return TranscriptionResult(
        notes=notes, model=backend.spec.name, model_version=backend.spec.version,
        runtime=backend.spec.runtime, settings=settings.json(),
        audio_length_s=seconds, processing_time_s=time.perf_counter() - start,
    )


def _stitch_chunks(
    chunk_notes: list[list[NoteEvent]],
    chunk_starts: list[float],
    settings: TranscribeSettings,
) -> list[NoteEvent]:
    """Re-join notes broken by chunk boundaries (outline 3.5/6).

    Two duplicate patterns from overlapping-window processing are repaired:
    A. same-pitch notes from adjacent chunks that are the SAME strike (onsets
       within 60 ms) -- e.g. a note fully inside the overlap region detected
       by both chunks. Merged, keeping the earliest onset and longest end.
    B. a note anchored at a chunk boundary that the later chunk re-detects
       with a slightly later onset (model context lag): right-side onset
       within [b - 0.1, b + 0.35] s of the boundary whose left-side same-pitch
       partner ends at [b - 0.35, b + 0.1]. Merged onto the earlier note.

    Genuine repeated strikes are preserved in both cases (strikes separated by
    >= merge_gap_s stay distinct).
    """
    out: list[NoteEvent] = []
    for ci, notes in enumerate(chunk_notes):
        if ci == 0:
            out.extend(notes)
            continue
        prev = chunk_notes[ci - 1]
        b = chunk_starts[ci]
        for r in notes:
            buddy = None
            for l in prev:
                if l.pitch != r.pitch:
                    continue
                # A. same strike detected by both chunks (overlap region)
                if abs(l.onset - r.onset) < 0.06 and l.end >= r.onset - 0.05:
                    buddy = l
                    break
            if buddy is None:
                # B. continuation re-detected just after a chunk boundary:
                #    the attack fell just before the boundary (overlap region),
                #    the later chunk re-detects it with a small onset lag.
                if b - 0.1 <= r.onset <= b + 0.35:
                    for l in prev:
                        if (l.pitch == r.pitch
                                and b - settings.overlap_s <= l.onset <= b
                                and l.end >= r.onset - 0.05):
                            buddy = l
                            break
            if buddy is None:
                out.append(r)
                continue
            buddy.end = max(buddy.end, r.end)
            if buddy.confidence is None:
                buddy.confidence = r.confidence
            elif r.confidence is not None:
                buddy.confidence = max(buddy.confidence, r.confidence)
            buddy.sustain_pedal = buddy.sustain_pedal or r.sustain_pedal
    return out


class ProgressTracker:
    """Thread-safe progress + cancellation helper for UI threads."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._last: list[tuple[float, str]] = []

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def cancel_event(self):
        return self._cancel

    def progress(self, frac: float, stage: str) -> None:
        self._last.append((float(frac), str(stage)))

    @property
    def last_progress(self) -> tuple[float, str]:
        return self._last[-1] if self._last else (0.0, "")