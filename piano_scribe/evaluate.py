"""Evaluation metrics (outline 8 / 11.11).

Matching: pitch must match exactly and onset within ``tol_on`` seconds
(octave errors counted separately). Durations are deliberately NOT a hard
match criterion: a decaying instrument's acoustic note end (what a model
hears) differs from the label end (usually key release / written rhythm),
and the outline explicitly separates these (§11.2, §8). End agreement is
still measured: every matched pair contributes end-MAE, and pairs with
|end error| > ``tol_off`` are counted in ``end_mismatch`` -- including
sustain-pedal passages, so pedal trouble is never hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .types import NoteEvent


@dataclass
class EvalConfig:
    tol_on: float = 0.05      # seconds, onset tolerance (match criterion)
    tol_off: float = 0.10     # seconds, end tolerance (REPORTING criterion)
    tol_octave: float = 0.10  # seconds, onset tolerance for octave-error check


@dataclass
class EvalResult:
    n_ref: int = 0
    n_pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    octave_errors: int = 0
    onset_mae: float = 0.0
    end_mae: float = 0.0
    end_mismatch: int = 0             # matched notes whose |end error| > tol_off
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    matched: list = field(default_factory=list)   # (ref_idx, pred_idx, onset_err, end_err)
    fp_details: list = field(default_factory=list)
    fn_details: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("matched", None)
        return d


def _octave(pitch: int, step: int) -> int:
    return pitch + 12 * step


def match_notes(ref: list[NoteEvent], pred: list[NoteEvent], cfg: EvalConfig,
                tag_filter: Optional[str] = None) -> EvalResult:
    """Greedy best-first matching by (onset distance, end distance)."""
    pred = sorted(pred, key=lambda n: (n.onset, n.end))
    ref = sorted(ref, key=lambda n: (n.onset, n.end))

    # candidate pairs within onset tolerance, sorted by fit quality
    cands = []
    for ri, r in enumerate(ref):
        for pi, p in enumerate(pred):
            if p.pitch != r.pitch:
                continue
            don = abs(p.onset - r.onset)
            if don > cfg.tol_on:
                continue
            if p.onset >= r.end + 0.001 or r.onset >= p.end + 0.001:
                continue  # same pitch, non-overlapping: repeat, not a match
            dend = abs(p.end - r.end)
            cands.append((don + dend, ri, pi, don, dend))
    cands.sort(key=lambda c: (c[0], c[1]))

    r_used, p_used = set(), set()
    matched = []
    for _, ri, pi, don, dend in cands:
        if ri in r_used or pi in p_used:
            continue
        r_used.add(ri)
        p_used.add(pi)
        matched.append((ri, pi, don, dend))

    res = EvalResult(
        n_ref=len(ref), n_pred=len(pred),
        tp=len(matched),
        fp=len(pred) - len(matched),
        fn=len(ref) - len(matched),
        matched=matched,
    )
    if matched:
        res.onset_mae = sum(m[2] for m in matched) / len(matched)
        res.end_mae = sum(m[3] for m in matched) / len(matched)
        res.end_mismatch = sum(1 for m in matched if m[3] > cfg.tol_off)
    res.precision = res.tp / res.n_pred if res.n_pred else 0.0
    res.recall = res.tp / res.n_ref if res.n_ref else 0.0
    res.f1 = 2 * res.precision * res.recall / (res.precision + res.recall) if (res.precision + res.recall) else 0.0

    # octave-error accounting: pred pitch exactly ±12 off an unmatched ref onset
    for pi, p in enumerate(pred):
        if pi in p_used:
            continue
        for ri, r in enumerate(ref):
            if ri in r_used:
                continue
            if abs(p.pitch - r.pitch) == 12 and abs(p.onset - r.onset) <= cfg.tol_octave:
                if p.onset < r.end + 0.001 and r.onset < p.end + 0.001:
                    res.octave_errors += 1
                    res.fp_details.append({"reason": "octave", "pred": p.to_json(), "ref": r.to_json()})
                    break
    return res


def evaluate_pair(ref_notes: list[NoteEvent], pred_notes: list[NoteEvent],
                  cfg: Optional[EvalConfig] = None) -> EvalResult:
    return match_notes(ref_notes, pred_notes, cfg or EvalConfig())


def load_note_json(path: Path) -> list[NoteEvent]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("notes", data.get("note_events", []))
    return [NoteEvent.from_json(n) for n in data]


def summarize(results: dict[str, EvalResult], cfg: EvalConfig) -> dict:
    """Aggregate across clips: micro-averaged note counts and MAEs."""
    tp = sum(r.tp for r in results.values())
    fp = sum(r.fp for r in results.values())
    fn = sum(r.fn for r in results.values())
    n_pred = sum(r.n_pred for r in results.values())
    n_ref = sum(r.n_ref for r in results.values())
    on = [d for r in results.values() for _, _, d, _ in r.matched]
    en = [d for r in results.values() for _, _, _, d in r.matched]
    return {
        "tolerances": {"onset_s": cfg.tol_on, "end_report_s": cfg.tol_off, "octave_onset_s": cfg.tol_octave},
        "clips": len(results),
        "ref_notes": n_ref,
        "pred_notes": n_pred,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / n_pred, 4) if n_pred else 0.0,
        "recall": round(tp / n_ref, 4) if n_ref else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (tp + fp + fn) else 0.0,
        "onset_mae_s": round(sum(on) / len(on), 4) if on else None,
        "end_mae_s": round(sum(en) / len(en), 4) if en else None,
        "end_mismatch": sum(r.end_mismatch for r in results.values()),
        "octave_errors": sum(r.octave_errors for r in results.values()),
    }


def breakdown(results: dict[str, EvalResult], manifest: dict, key: str) -> dict:
    """Per-condition or per-category breakdown (outline 8: track separately)."""
    groups: dict[str, list[str]] = {}
    for cid, res in results.items():
        meta = manifest.get(cid, {})
        tags = meta.get("tags", [])
        vals = meta.get(key, [])
        if isinstance(vals, str):
            vals = [vals]
        for v in vals + tags if key == "tags" else vals:
            groups.setdefault(v, []).append(cid)
    out = {}
    for g, ids in sorted(groups.items()):
        sub = {i: results[i] for i in ids if i in results}
        out[g] = summarize(sub, EvalConfig())
    return out