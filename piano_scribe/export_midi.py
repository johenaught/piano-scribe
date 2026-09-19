"""MIDI export (outline 3.8 / 9 first-release boundary).

Small dependency-free Standard MIDI File writer (format 1): conductor track
(tempo, time signature), one track per staff. Ties are written as a single
held note across measures (ties are written notation in MusicXML, not MIDI).
"""

from __future__ import annotations

import struct
from pathlib import Path

from .score_quantize import Score, TimingMap

DIV = 480  # ticks per quarter note


def _vlq(value: int) -> bytes:
    out = bytearray([value & 0x7F])
    value >>= 7
    while value > 0:
        out.insert(0, 0x80 | (value & 0x7F))
        value >>= 7
    return bytes(out)


def _event(delta: int, data: bytes) -> bytes:
    return _vlq(delta) + data


def _meta(tick: int, last: int, mtype: int, data: bytes) -> bytes:
    return _vlq(max(0, tick - last)) + bytes([0xFF, mtype, len(data)]) + data


def _tempo_meta(bpm: float) -> bytes:
    return struct.pack(">I", int(round(6e7 / bpm)))[1:]


def export_midi(score: Score, out_path: Path) -> Path:
    """Write a format-1 MIDI file: conductor + one note track per staff.

    Velocity: the model/decoder estimate when present (intensity estimate,
    not exact key velocity), else 90.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timing = score.timing
    tpb = DIV * 4.0 / timing.meter_den          # ticks per beat (meter-aware)
    bpm = timing.beats_per_measure()

    def tick_of(sn) -> int:
        return int(round(((sn.measure - 1) * bpm + sn.beat) * tpb))

    def dur_ticks(sn) -> int:
        return max(1, int(round(sn.dur_beats * tpb)))

    by_staff: dict[str, list] = {"treble": [], "bass": []}
    for sn in score.notes:
        by_staff[sn.staff].append(sn)

    tracks_bytes = []
    # conductor track
    data = b""
    data += _meta(0, 0, 0x01, b"piano-scribe")
    data += _meta(0, 0, 0x51, _tempo_meta(timing.tempo_bpm))
    data += _meta(0, 0, 0x58, bytes([timing.meter_num, timing.meter_den.bit_length() - 1, 24, 8]))
    data += _vlq(DIV) + bytes([0xFF, 0x2F, 0x00])
    tracks_bytes.append(b"MTrk" + struct.pack(">I", len(data)) + data)

    for staff in ("treble", "bass"):
        evs = sorted(by_staff[staff], key=lambda n: (tick_of(n), n.pitch))
        events: list[tuple[int, int, bytes]] = []  # (tick, order, payload)
        for sn in evs:
            if sn.tie_from:
                continue  # already sounding from the previous measure
            t0 = tick_of(sn)
            t1 = t0 + dur_ticks(sn)
            vel = 90 if sn.velocity is None else max(1, min(127, sn.velocity))
            ch = 0 if staff == "treble" else 1
            events.append((t0, 0, bytes([0x90 | ch, sn.pitch, vel])))
            events.append((t1, -1, bytes([0x80 | ch, sn.pitch, 0])))
        events.sort(key=lambda e: (e[0], e[1]))
        data = b""
        last = 0
        for tick, _, payload in events:
            data += _event(max(0, tick - last), payload)
            last = tick
        data += _vlq(DIV) + bytes([0xFF, 0x2F, 0x00])
        tracks_bytes.append(b"MTrk" + struct.pack(">I", len(data)) + data)

    with open(out_path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks_bytes), DIV))
        for t in tracks_bytes:
            f.write(t)
    return out_path


def export_notes_midi(notes, out_path: Path) -> Path:
    """Direct (pitch, onset_s, end_s, velocity, channel) -> single-track MIDI.

    Used by tests and quick checks. Tempo defaults to 120 bpm (quarter = 0.5 s).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events = []
    for pitch, onset, end, vel, ch in notes:
        t0 = int(round(onset * 2 * DIV))          # 120 bpm: 1 quarter = 0.5 s
        t1 = int(round(end * 2 * DIV))
        events.append((t0, 0, bytes([0x90 | ch, pitch, max(1, min(127, vel))])))
        events.append((max(t0 + 1, t1), -1, bytes([0x80 | ch, pitch, 0])))
    events.sort(key=lambda e: (e[0], e[1]))
    data = b""
    last = 0
    for tick, _, payload in events:
        data += _event(max(0, tick - last), payload)
        last = tick
    data += _vlq(DIV) + bytes([0xFF, 0x2F, 0x00])
    with open(out_path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 1, DIV))
        f.write(b"MTrk" + struct.pack(">I", len(data)) + data)
    return out_path


def load_timing_map(d: dict) -> TimingMap:
    return TimingMap(
        tempo_bpm=float(d.get("tempo_bpm", 120.0)),
        meter_num=int(d.get("meter_num", 4)),
        meter_den=int(d.get("meter_den", 4)),
        pickup_beats=float(d.get("pickup_beats", 0.0)),
    )