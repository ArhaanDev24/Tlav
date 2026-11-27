# Tlav installer for Windows. Creates %USERPROFILE%\.tlav\venv.
$ErrorActionPreference = "Stop"
$TlavHome = if ($env:TLAV_HOME) { $env:TLAV_HOME } else { Join-Path $env:USERPROFILE ".tlav" }
$RepoUrl = if ($env:TLAV_REPO) { $env:TLAV_REPO } else { "https://github.com/ArhaanDev24/Tlavv" }

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Write-Error "Python 3.8+ is required: https://www.python.org/downloads/windows/"; exit 1 }

New-Item -ItemType Directory -Force -Path $TlavHome | Out-Null
python -m venv (Join-Path $TlavHome "venv")
$pip = Join-Path $TlavHome "venv\Scripts\pip.exe"
if (Test-Path (Join-Path $PSScriptRoot "pyproject.toml")) { & $pip install --quiet --upgrade $PSScriptRoot }
else { & $pip install --quiet --upgrade "git+$RepoUrl" }

$scripts = Join-Path $TlavHome "venv\Scripts"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$scripts*") {
  [Environment]::SetEnvironmentVariable("Path", "$userPath;$scripts", "User")
}
& (Join-Path $scripts "tlav.exe") welcome
if ($LASTEXITCODE -ne 0) { Write-Host "`nTlav installed. Open a new terminal, then just type:`n  tlav" }
else { Write-Host "Open a new terminal so the tlav command is found." }
