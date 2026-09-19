# Model research notes (outline §10, §11)

Date: 2026-09-19. Sources: official repos/pages only (fetched during build).

## Candidates

### Basic Pitch — baseline (chosen ✓)
- Repo: https://github.com/spotify/basic-pitch — **Apache-2.0** (code + shipped model artifacts).
- ICASSP 2022 paper: "A Lightweight Instrument-Agnostic Model for Polyphonic Note
  Transcription and Multipitch Estimation" (arXiv:2203.09893).
- Model formats shipped: TensorFlow, ONNX, CoreML, TensorFlow Lite. On Windows the
  pip package defaults to **ONNX → ONNX Runtime**; that matches the outline's
  Windows runtime plan (§4) and gives us one native-runtime story to validate.
- Input: resampled mono 22 050 Hz; accepts mp3/ogg/wav/flac/m4a via librosa.
- License of *training data*: repo doesn't state; the model was trained on Spotify
  in-house data (not released). Weights distributed under Apache-2.0 by the project;
  re-distribution on our side = copying the same artifacts → Apache applies.
- Instrument-agnostic, polyphonic, single instrument at a time → fits "one pianist".

### ByteDance piano_transcription — piano-specific research baseline
- Repo: https://github.com/bytedance/piano_transcription — **Apache-2.0** code.
- PyTorch CRNN (Regress_onset_offset_frame_velocity), trained on MAESTRO v2.0.0
  (Liszt demo: https://github.com/bytedance/piano_transcription).
- Inference package: https://github.com/qiuqiangkong/piano_transcription_inference
  (`pip install piano_transcription_inference`); checkpoint downloads on first use
  from **Zenodo record 4034264** — that record's own terms must be inspected before
  bundling the checkpoint in a distributed product (§10: code/weights/data licenses
  are separate questions).
- Windows officially "not tested" upstream; we gate it behind the optional extra.
- This is the correct second baseline for the §11.1 comparison, if its checkpoint
  license permits and CPU perf is acceptable.

## Datasets

### MAESTRO v3 (research benchmark)
- https://magenta.tensorflow.org/datasets/maestro — ~199 h piano audio+MIDI,
  ~3 ms alignment, pedals (CC 64/66/67), composition-separated splits.
- **CC BY-NC-SA 4.0** → fine for research benchmarks, NOT unrestricted
  commercial training data (§11.3 warning).
- v3.0.0: 101 GB zip (SHA256 6680fea5…), midi-only 56 MB.
- Useful for reporting comparable benchmark numbers later; too clean for
  phone-recording realism — the pilot corpus (§11.5) covers that gap.

### GiantMIDI-Piano — weak labels only
- MIDI was *transcribed by an automatic model* → not verified ground truth
  (§11.3); repository license does not cover the third-party audio. Not used.

## Runtime decision map (§4)

| Target | Runtime | Notes |
|---|---|---|
| Windows | ONNX Runtime | Basic Pitch's default on Windows; verified present in this pipeline |
| iOS | Core ML (or ONNX via onnxruntime-c/objc) | Basic Pitch ships CoreML format; onnxruntime-c also supports CoreML execution provider |
| Android | LiteRT (or ONNX via onnxruntime-android) | Basic Pitch ships TFLite; NNAPI/XNNPACK accelerators available in onnxruntime |

ONNX Runtime across all three is the fallback if operator/perf parity holds —
one export path, three platforms (§4 "Alternative").

## Decisions taken (Phase 2)

1. **Do not train anything yet** (§11.1): evaluate existing pretrained models on
   a real corpus first. Escape order: fix recording/resampling/thresholds →
   fine-tune → distill → train from scratch.
2. Pipeline contracts: backends emit chunk-relative note events; the pipeline
   owns windowing/stitching; the decoder owns merge/repeat/blip rules; the
   evaluator owns stated-tolerance matching (onset gate, end reporting).
3. Evaluation semantics per §8/§11.11: note P/R/F1 with stated tolerances,
   onset MAE, end MAE + end-mismatch (acoustic end ≠ key release, §11.2),
   octave errors separate, all broken down by condition/tag/split.
4. Synthetic corpus first (outline §11.9) to exercise the harness end-to-end;
   the real-recording corpus (§11.5/11.6, MIDI-capable digital piano + phone
   mic, matched silent/noisy takes) is the next data task.

## Licenses checklist (before shipping any bundle)
- [ ] basic-pitch artifacts: Apache-2.0 (OK to bundle; keep license notice).
- [ ] ByteDance checkpoint (Zenodo 4034264): inspect record terms.
- [ ] MAESTRO: research-only, CC BY-NC-SA 4.0 — excluded from commercial
      training data.
- [ ] Any future soundfont/rendered MIDI used for synthetic training: check
      MIDI rights and sample-library license separately (§11.9).
- [ ] Piano sample renderer: ours is additive-synthesis (no samples) → no
      sample-library rights issue for the pilot corpus.