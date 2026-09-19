# piano-scribe

Local-first solo-piano transcription: record → transcribe (on-device) →
review/edit → MIDI / MusicXML export. Built from `solo_piano_app_outline(1).txt`.

**Status: development reference pipeline (Phases 1–4 of the outline).**
Python is the research/reference implementation per outline §4; the installed
app UI is planned as Dart/Flutter. All processing is local; no cloud, no
accounts, no uploads.

```
record ─▶ original.wav (immutable) ─▶ local model ─▶ note events ─▶ review/edit ─▶ MIDI, MusicXML
              ▲                                    ▲                 ▲
         capture metadata +       chunked windows,      reversible edit history
         calibration (3.2/3.3)    boundary stitching    (project store, §5)
                                  (pipeline, §6)
```

## What's implemented

| Outline area | Here |
|---|---|
| §3.2 audio capture/import | `audio_io.py`, `capture.py` (sounddevice → WAV, clipping/level checks) |
| §3.3 calibration | `calibration.py` (noise floor from silence; raw path always kept) |
| §3.4 engine | `models.py` — replaceable backends: **basic-pitch** (baseline, ONNX-aware on Windows), **bytedance** (piano-specific, optional) |
| §3.5 decoding | `decode.py` — same-pitch merge, genuine repeats preserved, blip suppression, chunk-window hygiene |
| §6 chunked processing | `pipeline.py` — bounded chunks + overlap, boundary stitching, progress/cancel |
| §3.7 score interpretation | `score_quantize.py` — tempo/meter, quantization, ties, grand staff, voices (deterministic v1; original timings preserved) |
| §3.8 export | `export_midi.py`, `export_musicxml.py` — dependency-free writers |
| §5 project store | `project.py` — immutable original, per-run results, correction state, undo history |
| §7 Phases 1–2 | `corpus.py` (synthetic pilot corpus with verified labels, splits, hashes), `evaluate.py`, `scripts/benchmark.py` |

Deliberately deferred (first-release boundary, §9): live preview, tuplets/
ornaments/pedal notation, key-signature control, PDF export, denoiser
training data, Markov decoding (only if it beats the baseline on held-out
data, §3.6/6), the Flutter UI itself.

## Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows (bash)
pip install -e . "[baseline]" "[capture]" "[dev]"        # basic-pitch + mic + pytest

# 1. Project + recording
piano-scribe new ~/projects/demo --name "demo" --tempo 96 --meter-num 4 --meter-den 4
piano-scribe record ~/projects/demo --seconds 45      # or omit --seconds; Ctrl+C stops
piano-scribe levels ~/projects/demo/recording/audio.wav   # clipping/weak-input check

# 2. Transcribe (local model)
piano-scribe transcribe ~/projects/demo

# 3. Review / edit (CLI for now; the GUI replaces this surface later)
piano-scribe notes ~/projects/demo
piano-scribe edit ~/projects/demo 12 --pitch 64 --end 8.3
piano-scribe edit ~/projects/demo 7 --delete
piano-scribe undo ~/projects/demo

# 4. Export
piano-scribe export ~/projects/demo --fmt midi
piano-scribe export ~/projects/demo --fmt musicxml
```

## Phase 2 benchmark (prove transcription in Python)

```bash
piano-scribe corpus data/corpus                 # builds the 50-clip synthetic pilot set
python scripts/benchmark.py data/corpus         # basic-pitch baseline -> reports/benchmark.md/json
python scripts/benchmark.py data/corpus --model bytedance   # piano-specific comparison (needs torch)
```

Current synthetic-clean baseline (test split, basic-pitch, tolerances
onset ±50 ms): see `data/corpus/reports/benchmark.md`. **Important:** these
numbers are on synthesized piano-like audio with exact labels — they exercise
the harness, not a promise about real recordings (outline §8: no accuracy
promises before benchmarking on actual device recordings).

## Evaluation semantics (read before trusting any number)

- Match = same pitch + onset within ±50 ms + temporal overlap (§8, §11.11).
- Duration is deliberately **not** a match gate: a model reports the acoustic
  end (incl. decay/pedal resonance); labels carry key-release. End agreement
  is always reported: matched-pair end-MAE plus `end_mismatch` (|Δend| > 100 ms)
  per split/condition/tag, so pedal and tail problems stay visible (§8).
- Octave errors are counted separately from false positives.

## Corpus notes (§7 Phase 1, §11.5)

`data/corpus` is synthetic (renderer: additive partials + hammer noise), with
labels that are exact by construction ("machine-verified-by-construction").
Splits are by piece; every condition variant of a piece shares its split;
audio and labels are hashed into `manifest.json` and immutable once written.
Conditions: clean, white noise (SNR 18/8 dB), 120 Hz hum, synthetic room.
Coverage: isolated notes/dynamics/range, chords, scales/arpeggios, repeated
notes/trills/tremolo, sustain pedal + held notes, two hands, quiet-under-loud,
beginner timing with a deliberate wrong note (labeled as played), chromatic
improvisation, silence probes.

**The synthetic set does not replace the real-recording test set** (§11.9).
Next step: ~100–200 short clips on actual target devices with verified labels
(MIDI-capable digital piano + mic, §11.6).

## Project layout (per project folder)

```
project.json               settings, timing, status
recording/audio.wav        original recording — preserved unchanged (§5)
recording/recording.json   device, sample rate, clipping, sha256
processing/run_*/          one immutable folder per transcription run
corrected/notes.json       working note set (user fixable)
corrected/history.jsonl    reversible edit history
score/  exports/           quantized score outputs
```

## Model notes (§10/§11)

- **basic-pitch** (baseline): Apache-2.0. On Windows the package favors the
  ONNX serialization under ONNX Runtime — the same runtime planned for the
  Windows app (§4).
- **bytedance/piano_transcription**: Apache-2.0 code; PyTorch checkpoint from
  Zenodo (record 4034264) — verify the checkpoint's own license before any
  distribution (§10).
- **MAESTRO v3**: CC BY-NC-SA 4.0 — usable for research benchmarks, **not**
  unrestricted commercial training data (§11.3).

See `research/model_research.md` for the full decision record.