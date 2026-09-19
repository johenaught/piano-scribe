"""Export tests: bytes written are valid MIDI / well-formed MusicXML."""

import xml.etree.ElementTree as ET

import pretty_midi
import pytest

from piano_scribe.export_midi import export_midi, export_notes_midi
from piano_scribe.export_musicxml import export_musicxml
from piano_scribe.score_quantize import quantize_notes
from piano_scribe.types import NoteEvent, TimingMap


def test_midi_roundtrip_via_pretty_midi(tmp_path):
    mid = tmp_path / "out.mid"
    export_notes_midi([(60, 0.0, 0.5, 90, 0), (64, 0.5, 1.0, 80, 0), (67, 0.0, 1.0, 70, 1)], mid)
    pm = pretty_midi.PrettyMIDI(str(mid))
    # channels split into separate pretty_midi instruments; aggregate all
    notes = sorted((int(n.pitch), n.start, n.end) for inst in pm.instruments for n in inst.notes)
    assert notes == [(60, 0.0, 0.5), (64, 0.5, 1.0), (67, 0.0, 1.0)]


def test_midi_score_export_has_tempo(tmp_path):
    timing = TimingMap(tempo_bpm=96, meter_num=3, meter_den=4)
    score = quantize_notes(
        [NoteEvent(pitch=60, onset=0.0, end=0.3, velocity=90),
         NoteEvent(pitch=72, onset=0.0, end=1.0, velocity=90)],
        timing,
    )
    mid = tmp_path / "score.mid"
    export_midi(score, mid)
    pm = pretty_midi.PrettyMIDI(str(mid))
    tc = pm.get_tempo_changes()
    if isinstance(tc, tuple):          # (times, tempos) convention
        tempos = tc[1]
    else:
        tc = np.asarray(tc)
        tempos = tc[:, 1] if tc.ndim == 2 else tc
    assert tempos[0] == pytest.approx(96.0)
    pitches = sorted(n.pitch for inst in pm.instruments for n in inst.notes)
    assert pitches == [60, 72]


def test_musicxml_well_formed_with_measures_and_ties(tmp_path):
    timing = TimingMap(tempo_bpm=120, meter_num=4, meter_den=4)
    # bass note spanning into measure 2 (tie), treble note on beat 2
    score = quantize_notes(
        [NoteEvent(pitch=48, onset=0.0, end=2.2, velocity=88),
         NoteEvent(pitch=64, onset=0.5, end=0.9, velocity=88)],
        timing,
    )
    xml_path = tmp_path / "score.musicxml"
    export_musicxml(score, xml_path, title="Test")
    root = ET.parse(str(xml_path)).getroot()
    parts = [p for p in root.iter("part")]
    assert len(parts) == 2
    measures = [m for m in root.iter("measure")]
    assert len(measures) >= 3                 # each part has measures 1-2
    notes = [m for m in root.iter("note")]
    assert len(notes) >= 3
    ties = [t for t in root.iter("tie")]
    assert any(t.attrib["type"] == "start" for t in ties)
    divs = [d.text for d in root.iter("divisions")]
    assert divs and divs[0] == "480"


def test_musicxml_accidentals_spelled_correctly(tmp_path):
    # C#4 (61) -> step C alter 1; D4 (62) -> step D; F#3 (54) -> step F alter 1
    timing = TimingMap(tempo_bpm=120, meter_num=4, meter_den=4)
    score = quantize_notes(
        [NoteEvent(pitch=61, onset=0.0, end=0.4, velocity=90),
         NoteEvent(pitch=62, onset=0.5, end=0.9, velocity=90),
         NoteEvent(pitch=54, onset=0.0, end=0.4, velocity=90)],
        timing,
    )
    xml_path = tmp_path / "acc.musicxml"
    export_musicxml(score, xml_path)
    root = ET.parse(str(xml_path)).getroot()
    pitches = [(n.findtext("step"), n.findtext("alter"), n.findtext("octave"))
               for n in root.iter("pitch")]
    assert ("C", "1", "4") in pitches
    assert ("D", None, "4") in pitches
    assert ("F", "1", "3") in pitches


def test_musicxml_rests_fill_gaps(tmp_path):
    timing = TimingMap(tempo_bpm=120, meter_num=4, meter_den=4)
    score = quantize_notes([NoteEvent(pitch=60, onset=1.0, end=1.4, velocity=90)], timing)
    xml_path = tmp_path / "r.musicxml"
    export_musicxml(score, xml_path)
    root = ET.parse(str(xml_path)).getroot()
    rests = len([n for n in root.iter("note") if n.find("rest") is not None])
    assert rests >= 1