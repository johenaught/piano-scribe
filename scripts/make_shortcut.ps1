# Creates the "Piano Scribe" desktop shortcut (run from the piano-scribe repo root).
$ErrorActionPreference = "Stop"
$repo = "C:\Users\Joss H\piano-scribe"
$pythonw = Join-Path $repo ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found: $pythonw" }

$desktop = [Environment]::GetFolderPath("Desktop")
if (-not (Test-Path $desktop) -or $desktop -eq "") { $desktop = Join-Path $env:USERPROFILE "Desktop" }
$lnk = Join-Path $desktop "Piano Scribe.lnk"

$shell = New-Object -ComObject "WScript.Shell"
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $pythonw
$sc.Arguments = "-m piano_scribe.gui"
$sc.WorkingDirectory = $repo
$sc.IconLocation = "$pythonw,0"
$sc.Description = "Piano Scribe - local piano transcription"
$sc.Save()

Write-Output "shortcut created: $lnk"