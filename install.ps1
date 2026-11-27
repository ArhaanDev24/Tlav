# Tlav installer for Windows.
#
#   irm https://tlav.online/install.ps1 | iex
#
# Works in Windows PowerShell 5.1 and PowerShell 7, needs no git, and is not blocked by the
# default "running scripts is disabled" policy when run this way. Creates %USERPROFILE%\.tlav\venv
# and puts `tlav` on your PATH (in this window too, so you can type `tlav` straight away).
$ErrorActionPreference = "Stop"
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}

$TlavHome = if ($env:TLAV_HOME) { $env:TLAV_HOME } else { Join-Path $HOME ".tlav" }
$Source = if ($env:TLAV_SOURCE) { $env:TLAV_SOURCE } else { "https://github.com/ArhaanDev24/Tlav/archive/refs/heads/main.zip" }

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }
function Fail($text) { Write-Host ""; Write-Host "  $text" -ForegroundColor Red; Write-Host ""; throw "Tlav was not installed." }

# A real Python 3.8+ (not the Microsoft Store shortcut, which only opens the Store).
function Find-Python {
  foreach ($cmd in @(@("py", "-3"), @("python"), @("python3"))) {
    if (-not (Get-Command $cmd[0] -ErrorAction SilentlyContinue)) { continue }
    $pre = @($cmd | Select-Object -Skip 1)
    try { $ok = & $cmd[0] @pre -c "import sys; print('ok' if sys.version_info >= (3, 8) else 'old')" 2>$null } catch { $ok = $null }
    if ($ok -eq "ok") { return ,$cmd }
  }
  return $null
}

Say ""
Say "  Tlav: old PCs, new minds. Installing..." "White"

$py = Find-Python
if (-not $py -and (Get-Command winget -ErrorAction SilentlyContinue)) {
  Say "  Tlav needs Python 3.8 or newer, and it isn't installed yet."
  $answer = Read-Host "  Install Python 3.12 now with winget? [Y/n]"
  if ($answer -eq "" -or $answer -match "^[Yy]") {
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
    $py = Find-Python
  }
}
if (-not $py) {
  Fail ("Python 3.8+ is required. Get it from https://www.python.org/downloads/windows/`n" +
        "  (tick 'Add python.exe to PATH' in the installer), then run this command again.")
}
$pyExe = $py[0]; $pyPre = @($py | Select-Object -Skip 1)
Say ("  Python:  " + (& $pyExe @pyPre -c "import sys; print(sys.version.split()[0])"))

New-Item -ItemType Directory -Force -Path $TlavHome | Out-Null
$venv = Join-Path $TlavHome "venv"
& $pyExe @pyPre -m venv $venv
if ($LASTEXITCODE -ne 0) { Fail "Could not create Tlav's Python environment in $venv." }

$scripts = if (Test-Path (Join-Path $venv "Scripts")) { Join-Path $venv "Scripts" } else { Join-Path $venv "bin" }
$vpy = @("python.exe", "python") | ForEach-Object { Join-Path $scripts $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1

# Install from this folder when run from a checkout, otherwise download the latest Tlav.
$here = if ($PSScriptRoot) { $PSScriptRoot } else { "" }
$target = if ($here -and (Test-Path (Join-Path $here "pyproject.toml"))) { $here } else { $Source }
Say "  Getting Tlav..."
& $vpy -m pip install --quiet --disable-pip-version-check --upgrade --force-reinstall $target
if ($LASTEXITCODE -ne 0) { Fail "Downloading or installing Tlav failed (see the message above). Check your internet connection and try again." }

$tlav = @("tlav.exe", "tlav") | ForEach-Object { Join-Path $scripts $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $tlav) { Fail "Tlav installed, but its command was not found in $scripts." }

# On PATH for new windows, and for this one right now.
try {
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  if (-not $userPath) { $userPath = "" }
  if (($userPath -split ";") -notcontains $scripts) {
    [Environment]::SetEnvironmentVariable("Path", (($userPath.TrimEnd(";") + ";" + $scripts).TrimStart(";")), "User")
  }
} catch {}
$now = [Environment]::GetEnvironmentVariable("PATH")
if (($now -split [IO.Path]::PathSeparator) -notcontains $scripts) {
  [Environment]::SetEnvironmentVariable("PATH", $scripts + [IO.Path]::PathSeparator + $now)
}

Say ""
& $tlav welcome
if ($LASTEXITCODE -ne 0) { Say "  Tlav is installed. Next, type:  tlav" "White" }
