"""Synthetic piano renderer (outline 11.9).

Renders labeled note events to piano-ish audio so the transcription pipeline
can be exercised with EXACT ground truth before any real recordings exist.
This is a clearly-labeled development tool, not a substitute for real
microphone evaluation (outline 11.9: 'A model may learn the renderer instead
of robust piano acoustics' -- real-room recordings remain the Phase 1 test
set requirement).

Design: additive partials with inharmonicity, fast attack, exponential decay
(register-dependent), hammer noise at onset, soft clipping. Sustain-pedal
notes ring on past their label end (key release) -- the label end is
CAREFULLY NOT the acoustic end of the sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .types import NoteEvent

RENDER_SR = 22050


@dataclass
class RenderParams:
    partials: int = 10
    attack_s: float = 0.006
    thump_amp: float = 0.12
    # base decay full-time constant; high register decays faster
    tau_at_a0: float = 3.0
    tau_at_c8: float = 0.6
    inharmonicity_b: float = 0.0004


def midi_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _tau(pitch: int, p: RenderParams) -> float:
    f = (p.tau_at_c8 - p.tau_at_a0) / (108 - 21)
    return float(p.tau_at_a0 + f * max(0, pitch - 21))


def render_notes(notes: Iterable[NoteEvent], sr: int = RENDER_SR, p: Optional[RenderParams] = None) -> np.ndarray:
    """Render note events into mono float32 audio at ``sr`` (22050 default).

    Notes with ``sustain_pedal=True`` are NOT truncated at their label end:
    the acoustic tail continues (decay-only), modelling pedal resonance.
    Pedal-release/off events are not represented in v1 rendering.
    """
    p = p or RenderParams()
    notes = [n for n in notes if n.end > n.onset]
    if not notes:
        return np.zeros(1, dtype=np.float32)
    tail = 4.0 * max(_tau(n.pitch, p) for n in notes)
    length = int((max(n.end for n in notes) + tail + 0.2) * sr)
    buf = np.zeros(length, dtype=np.float64)

    t_all = np.arange(length, dtype=np.float64) / sr
    for n in notes:
        onset = int(n.onset * sr)
        tau = _tau(n.pitch, p)
        note_len = n.end - n.onset
        # acoustic tail: ring length is the longer of (label length + release
        # tail) and (decay floor); pedal notes keep ringing through the tail
        ring = min(tail, note_len + 3.0 * tau)
        dur = int((note_len + (3.0 * tau if n.sustain_pedal else 0.6 * tau)) * sr)
        if onset >= length:
            continue
        idx = np.arange(onset, min(onset + dur, length), dtype=np.int64)
        if idx.size == 0:
            continue
        t = idx / sr - n.onset
        vel = n.velocity if n.velocity else 96
        amp = 0.03 + 0.85 * (vel / 127.0) ** 1.6
        f0 = midi_hz(n.pitch)
        env = np.exp(-t / tau)
        # attack ramp (~6 ms)
        att = int(p.attack_s * sr)
        if att > 0:
            ramp = np.linspace(0.0, 1.0, min(att, t.size))
            env[: ramp.size] *= ramp
        sig = np.zeros_like(idx, dtype=np.float64)
        for h in range(1, p.partials + 1):
            fh = h * f0 * np.sqrt(1.0 + p.inharmonicity_b * h * h)
            ah = (1.0 / h) ** 1.35
            if h == 2:
                ah *= 0.9
            if fh > sr / 2.0 * 0.95:
                break
            phase = np.random.default_rng((n.pitch * 1009 + h * 37) % (2**31)).uniform(0, 2 * np.pi)
            sig += ah * np.cos(2 * np.pi * fh * t + phase)
        # hammer thump: brief filtered noise at the attack
        rng = np.random.default_rng((n.pitch * 7919 + int(n.onset * 1000)) % (2**31))
        thump_len = min(int(0.004 * sr), idx.size)
        if thump_len > 8:
            noise = rng.standard_normal(thump_len)
            noise *= np.exp(-np.arange(thump_len) / (0.0012 * sr))
            sig[:thump_len] += noise * p.thump_amp * amp
        buf[idx] += sig * amp * env

    # soft clip then normalize so peaks sit near -3 dBFS
    buf = np.tanh(buf * 1.15)
    peak = np.max(np.abs(buf)) if buf.size else 0.0
    if peak > 1e-9:
        buf *= 0.7 / peak
    return buf.astype(np.float32)[:length]


def notes_to_event_list(notes: list[tuple[int, float, float, Optional[int], bool]]) -> list[NoteEvent]:
    """Convenience: (pitch, onset, end[, velocity, sustain_pedal]) -> NoteEvent."""
    out = []
    for row in notes:
        pitch, onset, end = row[0], row[1], row[2]
        vel = row[3] if len(row) > 3 and row[3] is not None else 96
        pedal = bool(row[4]) if len(row) > 4 else False
        out.append(NoteEvent(pitch=pitch, onset=onset, end=end, velocity=vel, sustain_pedal=pedal))
    return out