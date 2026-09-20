# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['scripts/pianoscribe_app.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/Joss H/piano-scribe/.venv/Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.onnx', 'basic_pitch/saved_models/icassp_2022')],
    hiddenimports=['piano_scribe.gui', 'basic_pitch.inference', 'basic_pitch.note_creation', 'sounddevice'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'keras', 'torch', 'piano_transcription_inference', 'matplotlib', 'pretty_midi'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PianoScribe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
