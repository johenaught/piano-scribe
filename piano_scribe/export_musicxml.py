"""MusicXML 4.0 export (outline 3.8 / Phase 5 gate).

Score-partwise grand staff (P1 treble, P2 bass), measure-by-measure with
attributes (divisions, time, clef), notes with ties, rests for gaps.
Deterministic v1 output: no tuplets/ornaments, no key signature yet (C major;
spelling control is deferred, outline 3.7).
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from .score_quantize import Score, ScoreNote

DIV = 480

_STEP = ["C", "D", "E", "F", "G", "A", "B"]
_TYPE_BY_DIVS = {1920: "whole", 960: "half", 480: "quarter", 240: "eighth", 120: "16th", 60: "32nd"}


def _div_type(dur: int) -> tuple[str, str]:
    """(divisions, type-string) for a supported note value (rounds down)."""
    for d in (1920, 960, 480, 240, 120, 60):
        if dur >= d:
            return str(d), _TYPE_BY_DIVS[d]
    return "60", "32nd"


def _pitch_elem(pitch: int) -> str:
    pc = pitch % 12
    step_idx = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6][pc]   # natural name per pitch class
    alter = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0][pc]       # accidental
    step = _STEP[step_idx]
    octave = pitch // 12 - 1
    out = f"<step>{step}</step>"
    if alter:
        out += f"<alter>{alter}</alter>"
    out += f"<octave>{octave}</octave>"
    return out


def _note_elem(sn: ScoreNote, meter_den: int) -> str:
    voice = f"<voice>{sn.voice + 1}</voice>"
    dur = sn.duration_divs(meter_den)
    d, t = _div_type(dur)
    tie_elems = ""
    notations = ""
    if sn.tie_after:
        tie_elems = "<tie type=\"start\"/>"
        notations = "<notations><tied type=\"start\"/></notations>"
    elif sn.tie_from:
        tie_elems = "<tie type=\"stop\"/>"
        notations = "<notations><tied type=\"stop\"/></notations>"
    return (
        f"<note>{voice}<pitch>{_pitch_elem(sn.pitch)}</pitch>"
        f"<duration>{d}</duration><type>{t}</type>{tie_elems}{notations}</note>"
    )


def _rest_elem(dur: int, meter_den: int) -> str:
    d, t = _div_type(dur)
    return f"<note><rest/><duration>{d}</duration><type>{t}</type></note>"


def export_musicxml(score: Score, out_path: Path, title: str = "Untitled transcription") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timing = score.timing
    den = timing.meter_den
    sec_per_beat = timing.seconds_per_beat()

    # group by staff then measure
    by_staff: dict[str, dict[int, list[ScoreNote]]] = {"treble": {}, "bass": {}}
    for sn in score.notes:
        by_staff[sn.staff].setdefault(sn.measure, []).append(sn)

    def measure_elems(staff: str, m: int) -> str:
        clef = "G" if staff == "treble" else "F"
        line = 2 if staff == "treble" else 4
        notes = sorted(by_staff[staff].get(m, []), key=lambda n: (n.beat, n.pitch))
        attrs = [f"<divisions>{DIV}</divisions>"]
        if m == 1:
            attrs.append(f"<time><beats>{timing.meter_num}</beats><beat-type>{den}</beat-type></time>")
        attrs.append(f"<clef><sign>{clef}</sign><line>{line}</line></clef>")
        body = f"<attributes>{''.join(attrs)}</attributes>"
        # rests fill gaps between note starts within the measure
        cursor = 0.0  # beat position within measure
        for sn in notes:
            if sn.beat > cursor + 1e-6:
                dur = int(round((sn.beat - cursor) * sec_per_beat / timing.seconds_per_beat() * (DIV * 4.0 / den)))
                body += _rest_elem(max(60, dur), den)
            body += _note_elem(sn, den)
            cursor = sn.beat + float(sn.duration_divs(den)) * den / (DIV * 4.0)
        if m == 1 and not notes:
            pass
        return f"<measure number=\"{m}\">{body}</measure>"

    max_meas = max([max(by, default=1) for by in by_staff.values()] or [1])
    parts = []
    for pid, staff in [("P1", "treble"), ("P2", "bass")]:
        measures = "".join(measure_elems(staff, m) for m in range(1, max_meas + 1))
        parts.append(f"<part id=\"{pid}\">\n    {measures}\n  </part>")

    doc = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work><work-title>{escape(title)}</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Piano (treble)</part-name><score-instrument id="P1-I1"><instrument-name>Grand Piano</instrument-name></score-instrument></score-part>
    <score-part id="P2"><part-name>Piano (bass)</part-name><score-instrument id="P2-I1"><instrument-name>Grand Piano</instrument-name></score-instrument></score-part>
  </part-list>
  {''.join(parts)}
</score-partwise>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path