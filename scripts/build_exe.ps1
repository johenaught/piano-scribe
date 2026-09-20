# Builds the standalone Windows app (PyInstaller one-file, GUI, offline model).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "no .venv found - create it first (see README)" }

# PyInstaller + build tools
& $py -m pip install -q pyinstaller
if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed" }

$onnx = Join-Path $root ".venv\Lib\site-packages\basic_pitch\saved_models\icassp_2022\nmp.onnx"
if (-not (Test-Path $onnx)) { throw "basic-pitch ONNX model not found: $onnx" }

$dataArg = "$onnx;basic_pitch\saved_models\icassp_2022"

& $py -m PyInstaller `
    --noconfirm --clean --onefile --noconsole `
    --name "PianoScribe" `
    --add-data $dataArg `
    --hidden-import "piano_scribe.gui" `
    --hidden-import "basic_pitch.inference" `
    --hidden-import "basic_pitch.note_creation" `
    --hidden-import "sounddevice" `
    --exclude-module "tensorflow" `
    --exclude-module "keras" `
    --exclude-module "torch" `
    --exclude-module "piano_transcription_inference" `
    --exclude-module "matplotlib" `
    --exclude-module "pretty_midi" `
    "scripts\pianoscribe_app.py"

if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$exe = Join-Path $root "dist\PianoScribe.exe"
Write-Output "BUILD OK: $exe  ($([math]::Round((Get-Item $exe).Length / 1MB, 1)) MB)"