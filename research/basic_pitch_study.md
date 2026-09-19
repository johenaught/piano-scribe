# Basic Pitch study — what we learned and adopted

Date: 2026-09-19. Source studied: spotify/basic-pitch (Apache-2.0), the
installed package (v0.3.x) — `basic_pitch/inference.py`,
`basic_pitch/note_creation.py`, the ICASSP 2022 paper.

## How Basic Pitch works (what we learned)

1. **Front end**: audio → CQT (constant-Q transform) with harmonic stacking —
   each pitch bin sums energy from its own harmonic partials, which gives the
   model octave-robustness at the input level instead of the output level.
2. **Model**: compact CNN (4 conv blocks, melodia-style head) producing, per
   time-frame, three maps: `frame` (note activity), `onset` (attack strength),
   `contour` (monophonic salience for pitch bends).
3. **Decoder** (`model_output_to_notes`): two-threshold hysteresis —
   a note starts when onset AND frame activity exceed thresholds, tracks while
   activity stays above `frame_thresh`, ends when it drops. Post-passes:
   - `infer_onsets=True` (their default): *adds* onsets wherever frame
     activity jumps inside a sounding note — intended to catch legato
     re-strikes, but it also manufactures attacks on loud sustained chords.
   - `melodia_trick=True` (their default): melodia-style post-filter that
     prunes low-salience multi-note stacks — tuned for monophonic melodies;
     for chordal piano it suppresses real inner voices (and, combined with a
     strong fundamental, can leave only octaves).
   - `min_note_len` frame floor, frequency-range clip, optional pitch bends.
4. Their **tuning procedure** (README/paper): sweep onset/frame thresholds per
   target domain; treat accuracy claims as domain-specific. That is exactly
   the procedure we now run (`scripts/sweep_thresholds.py`).

## What we adopted (ported into piano-scribe)

- **We now call the low-level decode** (`run_inference` + `model_output_to_notes`)
  instead of `predict()`, so the decoder knobs are ours:
  - `infer_onsets=False` — biggest single phantom cut for piano (measured
    below); genuine legato re-strikes are rare in piano vs. sustained wind,
    and our own same-pitch echo-merge catches near misses.
  - `melodia_trick=False` — keeps inner chord voices.
  - `include_pitch_bends=False` — pianos do not bend; saves contour work.
  - threshold sweep as the standard tuning procedure, with held-out
    evaluation (corpus `val` split; songs as content probes).
- Their frame/onset hysteresis concept is reflected in our two-stage decode:
  model decode → our temporal consistency (echo-merge, blip floor,
  harmonic-leak suppression, confidence gate).

## What we changed beyond Basic Pitch (measured improvements)

- **Notation sharpening** (`sharpened_for_notation`): the model reports the
  acoustic end (decay tail); written rhythm must end before the next attack
  on the same pitch. Applied at quantize/export/playback; acoustic ends stay
  in the project. This directly addresses "timing was off".
- **Harmonic-echo suppression**: weak (≤½×strong, <55%) detections a
  harmonic interval (12/19/24 st) away from a strong onset-mate are dropped.
- **Same-pitch echo-merge** (`merge_gap_s=0.09`): double detections of one
  note within ~100 ms collapse into one — the "E3 echo chain" pattern.
- **Tuned defaults** (vs. upstream, on real repertoire, onset tol ±50 ms):

| setting | upstream default | ours (tuned) | effect on 4-song set |
|---|---|---|---|
| onset_threshold | 0.5 | 0.60 | phantoms 100–354 → tens |
| frame_threshold | 0.5 | 0.45 | recall back up |
| infer_onsets | True | False | **phantom count −~80%** |
| melodia_trick | True | False | inner voices survive |
| min_confidence | — | 0.18 | spatter floor |

## Measured result (score-backed ground truth, synthetic render)

Für Elise: F1 0.597 → **0.769**, precision 0.61 → **0.94**, phantoms 354 → **37**,
onset MAE 6.8 ms. Full before/after in `data/songs/report_baseline.json` vs
`data/songs/report_tuned.json`. (Render caveat: our additive synth differs
from real piano acoustics; real recordings remain the final judge — but the
phantom pathology here mirrored the user report on real audio.)

## Remaining known weaknesses (honest)

- Recall (0.62–0.65) now lags precision: dense fast passages (Moonlight
  bass, Canon arpeggios) still drop notes. Better frame_threshold tuning per
  texture and the planned decoder timing rules are next (§3.5/3.6).
- End times stay acoustic (end MAE ~0.5 s); handled by sharpening at
  notation time, not by pretending the model knows release timing.
- Synthetic renderer's overlapping tails create interference the model hears
  as extra attacks; real-piano validation may be *better*, not worse.