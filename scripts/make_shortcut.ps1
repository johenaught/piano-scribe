# Creates the "Piano Scribe" desktop shortcut.
# Usage:
#   - dev machine:   powershell -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1
#   - installer:     powershell ... -Pythonw <path>\pythonw.exe -Repo <folder>
param(
    [string]$Pythonw = "",
    [string]$Repo = ""
)
$ErrorActionPreference = "Stop"

if ($Pythonw -eq "") {
    $Pythonw = "C:\Users\Joss H\piano-scribe\.venv\Scripts\pythonw.exe"
}
if ($Repo -eq "") {
    $Repo = "C:\Users\Joss H\piano-scribe"
}
if (-not (Test-Path $Pythonw)) { throw "pythonw.exe not found: $Pythonw" }

$desktop = [Environment]::GetFolderPath("Desktop")
if (-not (Test-Path $desktop) -or $desktop -eq "") { $desktop = Join-Path $env:USERPROFILE "Desktop" }
$lnk = Join-Path $desktop "Piano Scribe.lnk"

$shell = New-Object -ComObject "WScript.Shell"
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $Pythonw
$sc.Arguments = "-m piano_scribe.gui"
$sc.WorkingDirectory = $Repo
$sc.IconLocation = "$Pythonw,0"
$sc.Description = "Piano Scribe - local piano transcription"
$sc.Save()

Write-Output "shortcut created: $lnk"