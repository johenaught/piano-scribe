"""piano-scribe: local-first solo piano transcription reference pipeline.

Development pipeline only (per the product outline, the shipped app is a
Flutter desktop/mobile client; Python is the research and reference
implementation). No audio leaves the device. No cloud services.

Modules mirror the outline's application modules:
    3.2  audio_io / capture     audio capture and import
    3.3  calibration            silence/noise estimation, level checks
    3.4  models                 replaceable transcription backends
    3.5  decode                 note decoding and temporal consistency
    3.7  score_quantize         rhythm/score interpretation
    3.8  export_midi/export_musicxml
    5    project                local project store, edit history
    7    corpus, evaluate       test set + Phase 2 evaluation harness
"""

__version__ = "0.1.0"