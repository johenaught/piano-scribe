# One-click installer for Piano Scribe (Windows, PowerShell).
#
# Works in two ways:
#   1. From the GitHub source zip:  download the repo zip, extract it, then
#      run:  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#   2. Direct from GitHub:           powershell -ExecutionPolicy Bypass -File install.ps1 -ZipUrl <url-to-repo-zip>
#
# Installs into %LOCALAPPDATA%\PianoScribe, builds the environment, and puts a
# "Piano Scribe" shortcut on the Desktop. Everything runs offline afterwards.

param(
    [string]$ZipUrl = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$src = Split-Path -Parent $PSScriptRoot
$installRoot = Join-Path $env:LOCALAPPDATA "PianoScribe"
$log = Join-Path $env:TEMP "piano-scribe-install.log"

function Log($msg) { Write-Host $msg; Add-Content -Path $log -Value (Get-Date -Format s) + " " + $msg }

# --- 1. source location -----------------------------------------------------
if ($ZipUrl -ne "") {
    Log "Downloading source from GitHub..."
    $zip = Join-Path $env:TEMP "piano-scribe-src.zip"
    Invoke-WebRequest -Uri $ZipUrl -OutFile $zip
    $src = Join-Path $env:TEMP "piano-scribe-src"
    if (Test-Path $src) { Remove-Item $src -Recurse -Force }
    Expand-Archive -Path $zip -DestinationPath $src
    $src = Join-Path $src ($(Get-ChildItem $src | Where-Object { $_.PSIsContainer } | Select-Object -First 1).Name)
} else {
    if (-not (Test-Path (Join-Path $src "pyproject.toml"))) {
        throw "install.ps1 must run from inside the piano-scribe repo (pyproject.toml not found)"
    }
}

New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
Set-Location $installRoot

# --- 2. python 3.11+ --------------------------------------------------------
$pyExe = $null
$candidates = @()
if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += (Get-Command python).Source }
foreach ($c in $candidates) {
    try {
        $v = & $c -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ([int]($v.Split('.')[0]) -eq 3 -and [int]($v.Split('.')[1]) -ge 11) { $pyExe = $c; break }
    } catch {}
}
if (-not $pyExe) {
    throw "Python 3.11+ not found. Install it first: https://www.python.org/downloads/ (check 'Add to PATH'), then run this again."
}
Log "Using Python: $pyExe"

# --- 3. venv + packages ------------------------------------------------------
$venv = Join-Path $installRoot ".venv"
if ((Test-Path $venv) -and -not $Force) {
    Log "Environment already installed at $installRoot (use -Force to rebuild)."
} else {
    Log "Creating environment (this downloads the model packages once) ..."
    if (Test-Path $venv) { Remove-Item $venv -Recurse -Force }
    & $py -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    $vpy = Join-Path $venv "Scripts\python.exe"
    & $vpy -m pip install -q --upgrade pip
    & $vpy -m pip install -q -e "$src[baseline,capture]"
    if ($LASTEXITCODE -ne 0) { throw "package install failed (see $log)" }
}

# --- 4. desktop shortcut ----------------------------------------------------
& powershell -ExecutionPolicy Bypass -File (Join-Path $src "scripts\make_shortcut.ps1") `
    -Pythonw (Join-Path $venv "Scripts\pythonw.exe") -Repo $src

Log ""
Log "DONE. Double-click 'Piano Scribe' on your Desktop to start."
Log "Sessions are stored in Documents\Piano Scribe Sessions. No internet needed."
Read-Host "Press Enter to close"