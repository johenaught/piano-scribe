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

## Installation

**Requirements:** Windows 10/11. Everything runs offline — the model is
bundled/installed locally; no cloud, no account.

### Option A — the app (easiest)

Download `PianoScribe.exe` from the [Releases page](https://github.com/johenaught/piano-scribe/releases)
and double-click it. It is a self-contained Windows app — model included, no
installation, works offline. Windows SmartScreen may warn about an unsigned
app; click **More info → Run anyway**.

### Option B — one-click installer (from the source zip)

1. On the [Releases page](https://github.com/johenaught/piano-scribe/releases),
   download the **Source code (zip)** and extract it anywhere.
2. Right-click `scripts\install.ps1` → **Run with PowerShell**.
   (Or run `powershell -ExecutionPolicy Bypass -File scripts\install.ps1`.)
3. It downloads the model packages once (~300 MB), then puts a **Piano Scribe**
   shortcut on your Desktop. Everything after that is offline.

### Option C — from source (developers)

```bash
git clone https://github.com/johenaught/piano-scribe.git
cd piano-scribe
python -m venv .venv                       # Python 3.11+
.venv\Scripts\activate
pip install -e ".[baseline,capture,dev]"   # model, microphone, tests
piano-scribe-gui                           # or: pythonw -m piano_scribe.gui
```

A desktop shortcut is created by `scripts/make_shortcut.ps1`.
Uninstall: delete the Desktop shortcut and (option B) `%LOCALAPPDATA%\PianoScribe`;
sessions in `Documents\Piano Scribe Sessions` are yours to keep.

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

### Run the desktop app

```bash
piano-scribe-gui          # or: pythonw -m piano_scribe.gui
```

Voice-memo style window: sessions on the left, and each session opens a
detail pane with Record / Import / Transcribe (progress + cancel) / note
editing / piano-roll score / original-vs-detected playback comparison /
MIDI + MusicXML export. Sessions live in `~/Documents/Piano Scribe Sessions`
(projects created by the CLI can be added via "Add existing project…").

The app makes its state unmistakable: while recording, a red banner with a
live timer fills the top of the window and every other action is locked;
while transcribing, a blue banner shows progress. Sessions move between
computers with **⇅ Export session… / ⇅ Import session…** (a single
`.pianoscribe` file containing the recording, notes, and exports).

CLI equivalents work on the same project folders:

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

Current baseline (default tuned settings, tolerances onset ±50 ms):
see `data/corpus/reports/benchmark.md` — overall **F1 0.773, P 0.905, R 0.675**
on the synthetic corpus (down from 0.584 F1 / 983 false positives at
upstream defaults), and `data/songs/report_tuned.json` for the real
repertoire probes (Für Elise, Gymnopédie 1, Moonlight Mvt 1, Canon in D):
F1 0.71–0.89 per piece with 7–20× fewer phantom notes than the baseline.
**Important:** these numbers are on synthesized piano-like audio with exact
labels — they exercise the harness, not a promise about real recordings
(outline §8: no accuracy promises before benchmarking on actual device
recordings). What the tuning did, and what we learned from Basic Pitch's own
decoder, is documented in `research/basic_pitch_study.md`.

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