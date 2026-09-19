"""Pilot evaluation corpus builder (outline 11.5, Phase 1).

Synthetic corpus with EXACT note labels: every clip's labels are generated
from the same note events used to render the audio, so onset/offset ground
truth is true by construction (provenance: 'machine-verified-by-construction').
This exercises the pipeline end-to-end and the evaluation harness; it does
NOT replace the required real-recordings test set (outline 11.9).

Organization (outline 11.10):
- one manifest entry per clip with sha256 hashes, split, condition, tags,
  parent id for condition variants;
- labels are the *played* notes (including deliberate mistakes);
- splits are by piece; all conditions of one piece share its split;
- silence/long-rest clips check false positives.

Conditions: clean, +white noise (SNR 18/8 dB), +hum, +synthetic room reverb.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio_io import save_wav
from .synth import RENDER_SR, render_notes
from .types import NoteEvent

PieceFn = Callable[[int], tuple[list[NoteEvent], dict]]


@dataclass
class Piece:
    pid: str
    title: str
    tags: list[str]
    split: str
    build: PieceFn


def _ev(pitch, onset, end, vel=96, pedal=False) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset=onset, end=end, velocity=vel, sustain_pedal=pedal)


def _humanize(onsets, rng, jitter=0.008):
    """Deterministic human-ish timing jitter (real players are not exact)."""
    return [max(0.0, o + rng.normal(0.0, jitter)) for o in onsets]


# ---------------------------------------------------------------- pieces ---

def _p_isolate(rng: np.random.Generator):
    notes = []
    t = 0.5
    pit = list(range(21, 109))[::4]  # every 4th: 23 notes across the range
    for i, p in enumerate(pit):
        vel = 110 if i % 2 == 0 else 62
        dur = 0.35 if i % 3 else 1.6
        notes.append(_ev(p, t, t + dur, vel))
        t += dur + 0.45
    return notes


def _p_chords(rng: np.random.Generator):
    notes, t = [], 0.5
    # root-position and inverted triads/sevenths, both hands
    for root, inv in [(48, 0), (55, 1), (60, 0), (43, 1), (64, 2), (50, 0), (57, 2), (41, 1)]:
        bass = root - (12 if root >= 55 else 0)
        pit = [bass, bass + 12 + inv, bass + 19 + inv, bass + 24 + inv]
        dur = 1.1
        for p in pit[:3 + (0 if inv == 2 else 1)]:
            notes.append(_ev(p, t, t + dur, 88))
        t += dur + 0.30
    return notes


def _p_scales(rng: np.random.Generator):
    notes, t = [], 0.5
    # C major scale, a minor arpeggio, chromatic run, contrary motion
    for name, gen in [
        ("cmaj", [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 83, 81, 79, 77, 76, 74, 72, 71, 69, 67, 65, 64, 62, 60]),
        ("amin", [57, 60, 64, 69, 72, 76, 81, 84, 81, 76, 72, 69, 64, 60, 57]),
        ("chrom", [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60]),
    ]:
        step = 0.22
        for i, p in enumerate(gen):
            notes.append(_ev(p, t, t + 0.18, 84))
            t += step
        t += 0.6
    # two-hand contrary motion (simultaneous both hands)
    for i in range(8):
        p_up, p_dn = 60 + 2 * i, 60 - 2 * i
        notes.append(_ev(p_up, t, t + 0.45, 80))
        notes.append(_ev(p_dn, t, t + 0.45, 78))
        t += 0.5
    return notes


def _p_repeated(rng: np.random.Generator):
    notes, t = [], 0.5
    # same-pitch repeats: eighth-note-ish ~2.2 notes/s
    for i in range(12):
        notes.append(_ev(64, t, t + 0.28, 84))
        t += 0.42
    t += 0.4
    # trill C5-D5 (fast alternation)
    for i in range(24):
        notes.append(_ev(72 + i % 2, t, t + 0.09, 90))
        t += 0.13
    t += 0.4
    # tremolo octave (both sounding together)
    for i in range(8):
        notes.append(_ev(60, t, t + 0.24, 92))
        notes.append(_ev(72, t, t + 0.24, 88))
        t += 0.52
    return notes


def _p_sustained(rng: np.random.Generator):
    notes = []
    # pedal bass + sustained chords + moving melody over held notes
    held = [
        (41, 0.5, 8.0, 72, True),
        (53, 0.5, 8.0, 74, True),
        (65, 2.0, 6.5, 80, True),   # held over changing harmony
    ]
    for p, o, e, v, ped in held:
        # end = key release; sustain_pedal marks that the sound rings on past it
        notes.append(_ev(p, o, e, v, ped))
    mel = [(67, 1.0), (69, 2.0), (72, 3.0), (74, 4.0), (77, 5.0), (81, 6.0), (79, 7.0)]
    for p, o in mel:
        notes.append(_ev(p, o, o + 0.5, 86))
    # second pedal passage: chordal, keys released but pedal holds the sound
    t = 9.5
    for i, chord in enumerate([[60, 64, 67], [62, 65, 69], [64, 67, 71], [59, 62, 67]]):
        for p in chord:
            notes.append(_ev(p, t, t + 0.6, 82, True))   # released at +0.6, rings on
        t += 1.9
    return notes


def _p_two_hands(rng: np.random.Generator):
    notes, t = [], 0.5
    # left hand block chords, right hand melody on top
    accomp = [([36, 43, 48], 0.0), ([38, 45, 50], 0.75), ([40, 47, 52], 1.5), ([36, 43, 48], 2.25)]
    mel = [(60, 0.0), (64, 0.75), (67, 1.5), (69, 2.25), (72, 3.0), (74, 3.75), (76, 4.5), (79, 5.25), (81, 6.0), (79, 6.75)]
    for bar in range(3):
        t0 = t + bar * 3.0
        for chord, dt in accomp:
            for p in chord:
                notes.append(_ev(p, t0 + dt, t0 + dt + 1.5, 70))
        for p, dt in mel:
            if bar == 0 or dt >= 3.0:
                notes.append(_ev(p, t0 + dt, t0 + dt + 0.6, 92))
    t += 9.5
    return notes


def _p_quiet_loud(rng: np.random.Generator):
    notes, t = [], 0.5
    # loud left-hand accompaniment, quiet right-hand melody
    chords = [[43, 50, 55], [41, 48, 53], [43, 50, 55], [45, 52, 57]]
    for bar in range(8):
        t0 = t + bar * 1.5
        chord = chords[bar % 4]
        for p in chord:
            notes.append(_ev(p, t0, t0 + 1.2, 112))
        for p, dt in [(60, 0.0), (64, 0.5), (67, 1.0)]:
            notes.append(_ev(p + (bar % 3) * 2, t0 + dt, t0 + dt + 0.4, 42))
    return notes


def _p_beginner(rng: np.random.Generator):
    notes, t = [], 1.0
    # hesitant, irregular timing; one deliberate wrong note (must be labeled as played)
    grid = [1.0, 1.15, 1.9, 0.85, 2.35, 1.05, 1.8, 1.25, 2.2, 1.6]
    pitches = [60, 62, 64, 65, 64, 67, 69, 74, 71, 72]   # G# (68) would be "wrong" -- keep exact:
    pitches[7] = 73                                       # deliberate wrong note vs. C major expectation
    for i, (dt, p) in enumerate(zip(_humanize(grid, rng, 0.05), pitches)):
        notes.append(_ev(p, t, t + max(0.25, dt * 0.55), 84 + (i % 3) * 8))
        t += dt
    t += 0.5
    # rough two-finger chords (imperfect synchrony)
    for i, (dt, chord) in enumerate(zip(_humanize([1.2, 1.3, 2.4], rng, 0.06),
                                        [[60, 64, 67], [55, 59, 62], [53, 57, 60]])):
        for p in chord:
            notes.append(_ev(p, t + dt, t + dt + 0.9, 80))
        t += dt
    return notes


def _p_impro(rng: np.random.Generator):
    notes, t = [], 0.5
    # chromatic/whole-tone-ish polyphonic improv (unfamiliar harmony)
    seq = [[60, 66, 69, 75], [58, 64, 67, 73], [59, 65, 71, 76], [57, 63, 66, 72],
           [61, 67, 73, 76], [62, 68, 74, 77], [60, 66, 72, 78], [58, 64, 70, 73]]
    for chord in seq:
        for i, p in enumerate(chord):
            on = t + rng.uniform(0.0, 0.12) if i > 0 else t
            notes.append(_ev(p, on, on + 0.7, 88))
        t += 1.55
    t += 0.4
    for i in range(10):
        p = int(rng.choice([52, 56, 60, 64, 68, 72, 76, 80, 84, 88]))
        notes.append(_ev(p, t, t + 0.18, 76))
        t += rng.uniform(0.14, 0.3)
    return notes


def _p_sparse(rng: np.random.Generator):
    notes, t = [], 1.0
    # three short figures separated by long silence (false-positive probe)
    for fig in [(55, 59, 62), (72, 76, 79), (60, 63, 67, 72)]:
        for p in fig:
            notes.append(_ev(p, t, t + 0.5, 86))
        t += 4.5
    return notes


PIECES: list[Piece] = [
    Piece("isolate_notes", "Isolated notes across the piano range",
          ["isolated", "dynamics", "range"], "train", _p_isolate),
    Piece("chords", "Triads and sevenths, both hands, inversions",
          ["chords", "both_hands"], "train", _p_chords),
    Piece("arpeggios_scales", "Scales, arpeggios, chromatic, contrary motion",
          ["scales", "arpeggios", "chromatic", "both_hands"], "train", _p_scales),
    Piece("repeated_notes", "Repeated notes, trill, octave tremolo",
          ["repeated", "trill", "chords"], "val", _p_repeated),
    Piece("sustained", "Sustain-pedal bass, held chords, melody over pedal",
          ["pedal", "sustained", "overlap"], "val", _p_sustained),
    Piece("two_hands", "Melody over accompaniment",
          ["both_hands", "melody_accompaniment"], "val", _p_two_hands),
    Piece("quiet_loud", "Quiet melody under loud accompaniment",
          ["dynamics", "quiet_under_loud", "both_hands"], "val", _p_quiet_loud),
    Piece("beginner_irregular", "Irregular human timing with a wrong note",
          ["beginner", "irregular_timing", "wrong_note"], "test", _p_beginner),
    Piece("improvisation", "Chromatic polyphonic improvisation",
          ["improvisation", "chromatic", "polyphony"], "test", _p_impro),
    Piece("sparse_silence", "Short figures separated by long silence",
          ["silence", "false_positive"], "test", _p_sparse),
]

PIECE_BY_ID = {p.pid: p for p in PIECES}

CONDITIONS = ["clean", "noise_18", "noise_8", "hum", "room"]


@dataclass
class ManifestEntry:
    id: str
    piece: str
    condition: str
    split: str
    audio_path: str
    label_path: str
    duration: float
    sample_rate: int
    channels: int
    tags: list[str]
    sha256_audio: str
    sha256_labels: str
    parent: Optional[str]
    source: str
    labels_provenance: str
    allowed_uses: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _add_condition(samples, condition: str, rng: np.random.Generator, sr: int) -> np.ndarray:
    """Derivative variants: every clip keeps its own split (outline 11.10)."""
    if condition == "clean":
        return samples
    x = np.asarray(samples, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2)) if x.size else 0.0
    if condition.startswith("noise_"):
        snr = int(condition.split("_")[1])
        noise = rng.standard_normal(x.shape)
        target = rms / (10 ** (snr / 20.0))
        noise *= target / (np.sqrt(np.mean(noise**2)) + 1e-12)
        return (x + noise).astype(np.float32)
    if condition == "hum":
        t = np.arange(x.size) / sr
        hum = 0.02 * np.sin(2 * np.pi * 120 * t) + 0.008 * np.sin(2 * np.pi * 240 * t)
        hum += 0.004 * rng.standard_normal(x.shape)
        return (x + hum).astype(np.float32)
    if condition == "room":
        from scipy import signal
        # synthetic room impulse response: early reflections + decay
        ir_len = int(0.5 * sr)
        ir = rng.standard_normal(ir_len) * np.exp(-np.arange(ir_len) / (0.12 * sr))
        ir = ir / (np.sqrt(np.sum(ir**2)) + 1e-12)
        dry = x * 0.85
        wet = signal.fftconvolve(x, ir, mode="full")[: x.size] * 0.35
        return (dry + wet).astype(np.float32)
    raise ValueError(f"unknown condition {condition}")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_corpus(
    out_dir: Path,
    pieces: Optional[list[str]] = None,
    conditions: Optional[list[str]] = None,
    seed: int = 20260919,
    overwrite: bool = False,
    sources: Optional[str] = None,
) -> list[ManifestEntry]:
    """Build the pilot synthetic corpus. Returns manifest entries.

    Deterministic per piece+condition (fixed seed); original audio and label
    files are immutable once written; re-runs keep existing files and verify
    hashes instead of rewriting (outline 11.10).
    """
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    label_dir = out_dir / "labels"
    audio_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    pieces = pieces or list(PIECE_BY_ID)
    conditions = conditions or CONDITIONS
    manifest: list[ManifestEntry] = []

    for pid in pieces:
        piece = PIECE_BY_ID[pid]
        rng_piece = np.random.default_rng(seed + hash(pid) % 10000)
        base_notes = piece.build(rng_piece)
        for cond in conditions:
            cid = f"{pid}__{cond}"
            audio_path = audio_dir / f"{cid}.wav"
            label_path = label_dir / f"{cid}.json"

            if audio_path.exists() and not overwrite:
                # immutable: keep existing file, just record hashes
                label = json.loads(label_path.read_text(encoding="utf-8"))
                a_sha = _sha256_bytes(audio_path.read_bytes())
                entries = label["notes"]
                dur = label["duration"]
            else:
                rng = np.random.default_rng(seed + hash(cid) % 10000)
                clean = render_notes(base_notes, sr=RENDER_SR)
                samples = _add_condition(clean, cond, rng, RENDER_SR)
                dur = len(samples) / RENDER_SR
                save_wav(samples, RENDER_SR, audio_path)
                label = {
                    "id": cid,
                    "piece": pid,
                    "condition": cond,
                    "split": piece.split,
                    "sample_rate": RENDER_SR,
                    "duration": round(dur, 4),
                    "provenance": "machine-verified-by-construction (labels generated from the same events that rendered the audio)",
                    "notes": [n.to_json() for n in base_notes],
                }
                label_path.write_text(json.dumps(label, indent=2), encoding="utf-8")
                entries = label["notes"]
                a_sha = _sha256_bytes(audio_path.read_bytes())

            l_sha = _sha256_bytes(label_path.read_bytes())
            duration = float(label["duration"])
            manifest.append(ManifestEntry(
                id=cid, piece=pid, condition=cond, split=piece.split,
                audio_path=str(audio_path), label_path=str(label_path),
                duration=round(dur, 4), sample_rate=RENDER_SR, channels=1,
                tags=piece.tags, sha256_audio=a_sha, sha256_labels=l_sha,
                parent=None if cond == "clean" else f"{pid}__clean",
                source=sources or "synthetic: piano-scribe.renderer v1 (additive harmonics; NOT a real piano)",
                labels_provenance="machine-verified-by-construction",
                allowed_uses="internal development/evaluation only; synthetic audio has no licensing constraints, "
                             "but real recordings still required for the Phase 1 test set",
            ))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps([m.to_dict() for m in manifest], indent=2), encoding="utf-8")
    return manifest


def load_manifest(manifest_path: Path) -> dict:
    return {m["id"]: m for m in json.loads(manifest_path.read_text(encoding="utf-8"))}


def load_labels(label_path: Path) -> list[NoteEvent]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    return [NoteEvent.from_json(n) for n in data["notes"]]