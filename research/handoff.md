# Handoff notes & next steps (outline §7 gates, §11.12 checklist)

Status: 2026-09-19. Phases 1–4 foundation built and verified on this machine.

## What's verified

- 30 unit tests green (decode, quantization, metrics, exports, pipeline stitching).
- Full CLI workflow: project → import → transcribe (basic-pitch, local) →
  edit/undo → export MIDI + MusicXML (parsed by pretty_midi / ElementTree).
- Phase 1 synthetic corpus: 50 clips, 10 pieces × 5 conditions, splits by
  piece, hashed immutable manifest (`data/corpus`).
- Phase 2 baseline report: `data/corpus/reports/benchmark.md` (basic-pitch,
  onset ±50 ms matcher; end reported separately, not gating).

## Baseline results (synthetic corpus only — NOT phone recordings!)

**Tuned defaults (2026-09-19, after repertoire-driven threshold tuning):**

| group | F1 | notes |
|---|---|---|
| overall (all conditions) | **0.773** | **P 0.905 / R 0.675**, fp 983→137, octave errors 40→11 |
| held-out test split | 0.90+ | precision ~0.95 on all splits |
| repeated notes / trills | recall still low | the remaining weak case for tuning |
| quiet-under-loud | improved | same threshold stack |
| noise/room conditions | ≈clean | robust |

**Real-repertoire probes** (`data/songs/`, score-backed labels, onset ±50 ms):

| piece | F1 before → after | P | R | phantoms before → after |
|---|---|---|---|---|
| Für Elise | 0.596 → **0.769** | 0.944 | 0.649 | 366 → 37 |
| Gymnopédie No. 1 | 0.629 → **0.885** | 0.933 | 0.841 | 311 → 31 |
| Moonlight Mvt 1 | — → **0.759** | 0.951 | 0.632 | 722 → 49 |
| Canon in D (arr.) | 0.492 → **0.714** | 0.919 | 0.584 | 805 → 94 |

What changed and why (learned from Basic Pitch's own decoder — see
`research/basic_pitch_study.md`): onset_threshold 0.5→0.6, frame_threshold
0.5→0.45, `infer_onsets=False`, `melodia_trick=False`, confidence floor 0.18,
same-pitch echo merge 0.09 s, harmonic-echo suppression, and
notation-sharpened note ends for exports/playback.

**Candidate comparison (test split, clean, same harness):**

| model | F1 | P | R | proc s/min audio |
|---|---|---|---|---|
| basic-pitch (ONNX/TF, CPU) | **0.762** | 0.712 | 0.820 | ~0.9 |
| ByteDance piano_transcription (torch CPU) | 0.016 | 0.020 | 0.014 | ~102 |

Caveat: the bytedance model was trained on MAESTRO (Disklavier acoustics);
the synthetic renderer's timbre penalizes it more than it penalizes
basic-pitch's instrument-agnostic training. This is a harness check, not the
final verdict — the real-reference corpus decides. But it already proves the
outline's point: never assume "piano-specific" wins; benchmark on the actual
recordings.

Diagnosis to carry forward: the failing tags point at DECODER work first
(§11.1 escalate order A — fix decoding/thresholds before any training):
1. repeated notes/trills: same-pitch strike separation (merge_gap, frame
   threshold sweep on val split only — do not tune on test).
2. quiet-under-loud: per-note confidence/velocity floor, maybe onset
   threshold sweep.
3. chords: missing inner notes of dense chords — check frame_threshold and
   the 0.25 min-confidence floor interaction.

## Open items (in outline order)

- **Phase 1 real test set** (§11.5/11.6): ~100–200 short clips, MIDI-capable
  digital piano + phone mic, silent/noisy matched takes, latency measured at
  start/middle/end. This replaces/extends the synthetic corpus for decisions.
- **Phase 2 comparison**: ByteDance baseline run (checkpoint license first).
  First result on the synthetic corpus is in — see next section. Then any
  denoise A/B (§3.3: only if it wins held-out).
- **Phase 3**: measure the exported pipeline on target devices (iPhone,
  midrange Android, this Windows machine), RAM/heat/time; verify ONNX
  operators and CQT preprocessing parity in the native runtime.
- **Phase 4 app**: Flutter desktop (Windows first) around the same project
  store; the CLI is the reference implementation. Note: basic-pitch's own
  streaming of long audio replaces our Python chunking in the native build;
  keep the stitch rules in the ported decoder.
- **Phase 5 notation**: MusicXML exporter exists (v1: no key sig, no tuplets);
  validate exports in MuseScore, then iterate.
- **§11.3 licensing**: MAESTRO = research-only (CC BY-NC-SA); ByteDance Zenodo
  checkpoint terms unverified; form the permission checklist before bundling
  anything in a distributable.

## ByteDance backend integration notes

- `pip install piano_transcription_inference` (latest on PyPI is 0.0.6; my
  first attempt `>=0.3.0` resolved to nothing) and **torch CPU separately** —
  the package does not declare torch as a dependency.
- The package downloads its ~165 MB checkpoint with `wget`, which does not
  exist in git-bash on Windows: fetch it manually with
  `curl -L -o "note_F1=0.9677_pedal_F1=0.9186.pth" "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"`
  into `~/piano_transcription_inference_data/`.
- `transcribe()` returns `{'est_note_events': [...], 'est_pedal_events': [...]}`
  where each event is a dict with keys `onset_time`, `offset_time`, `midi_note`,
  `velocity`. Pedal events are currently ignored by our backend (v1).
- Windows upstream status: "not tested"; runs here on torch CPU.

## Reproduce

```bash
python -m pytest -q
piano-scribe corpus data/corpus
python scripts/benchmark.py data/corpus          # full 50-clip baseline (~1 min)
```

## Known limitations (honest)

- Synthetic renderer ≠ piano acoustics (§11.9): treat absolute F1 numbers as
  harness validation, not product performance.
- |Δend| mismatch everywhere: model reports acoustic end; labels carry key
  release. The product must decide how to map acoustic ends → written rhythm
  (§3.7) and must never present model duration as exact key velocity/hold.
- Chunk-stitch heuristics (rules A/B) tuned for 45 s/2 s windows; re-validate
  if defaults change.
- `undo` replays history from the raw run; re-added notes are not perfectly
  restored in v1 (documented in project.py).