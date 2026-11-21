<#
.SYNOPSIS
    VoiceOS setup script — creates the project virtual environment,
    installs dependencies and prepares runtime directories.

.DESCRIPTION
    Safe to run repeatedly (idempotent). Does NOT download speech models;
    see docs/MODEL_SETUP.md for offline model installation.

.PARAMETER Dev
    Also install development dependencies (pytest).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Dev
#>
[CmdletBinding()]
param(
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    $msg" -ForegroundColor Yellow }

# ------------------------------------------------------------------
# 1. Detect Python
# ------------------------------------------------------------------
Write-Step "Detecting Python"
$pyCmd = $null
foreach ($candidate in @("py", "python", "python3")) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) { $pyCmd = $candidate; break }
}
if (-not $pyCmd) {
    Write-Error "Python was not found. Install Python 3.12+ from https://www.python.org and re-run."
}

$versionText = & $pyCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
if (-not $versionText) {
    Write-Error "Could not determine the Python version for '$pyCmd'."
}
$version = [version]$versionText
Write-Ok "Found $pyCmd -> Python $version"
if ($version -lt [version]"3.12") {
    Write-Error "VoiceOS requires Python >= 3.12 (found $version)."
}

# Resolve the real interpreter for venv creation (py launcher may be a shim).
$pythonExe = & $pyCmd -c "import sys; print(sys.executable)" 2>$null
if (-not $pythonExe) { $pythonExe = $pyCmd }

# ------------------------------------------------------------------
# 2. Virtual environment
# ------------------------------------------------------------------
$venvDir = Join-Path $RepoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Step "Virtual environment already exists at .venv (keeping it)"
} else {
    Write-Step "Creating virtual environment at .venv"
    & $pythonExe -m venv $venvDir
    if (-not (Test-Path $venvPython)) {
        Write-Error "Virtual environment creation failed; $venvPython not found."
    }
    Write-Ok "Created .venv"
}

# ------------------------------------------------------------------
# 3. Dependencies
# ------------------------------------------------------------------
Write-Step "Installing runtime dependencies"
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Error "Runtime dependency installation failed." }
Write-Ok "Runtime dependencies installed"

if ($Dev) {
    Write-Step "Installing development dependencies"
    & $venvPython -m pip install -r (Join-Path $RepoRoot "requirements-dev.txt")
    if ($LASTEXITCODE -ne 0) { Write-Error "Development dependency installation failed." }
    Write-Ok "Development dependencies installed"
}

# ------------------------------------------------------------------
# 4. Runtime directories
# ------------------------------------------------------------------
Write-Step "Preparing runtime directories"
$envScript = @'
import sys
sys.path.insert(0, sys.argv[1])
from src.core import paths
paths.runtime_dir(); paths.logs_dir(); paths.cache_dir()
paths.kws_model_dir(); paths.stt_model_dir(); paths.tts_model_dir()
paths.database_path(); paths.file_index_db_path()
print(paths.runtime_dir())
'@
$runtimeDir = & $venvPython -c $envScript $RepoRoot
if ($LASTEXITCODE -ne 0 -or -not $runtimeDir) {
    Write-Warn2 "Could not resolve runtime dir via src.core.paths; using LOCALAPPDATA fallback"
    $runtimeDir = Join-Path $env:LOCALAPPDATA "VoiceOS"
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $runtimeDir "logs"), `
        (Join-Path $runtimeDir "cache"), `
        (Join-Path $runtimeDir "models\kws"), `
        (Join-Path $runtimeDir "models\stt"), `
        (Join-Path $runtimeDir "models\tts") | Out-Null
}
Write-Ok "Runtime dir: $runtimeDir"

# ------------------------------------------------------------------
# 5. Configuration + model check
# ------------------------------------------------------------------
Write-Step "Validating configuration"
& $venvPython (Join-Path $RepoRoot "scripts\verify_installation.py")
$verifyExit = $LASTEXITCODE

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Runtime data: $runtimeDir"
Write-Host "  Run VoiceOS:  .\scripts\run.ps1"
Write-Host ""
if ($verifyExit -ne 0) {
    Write-Warn2 "verify_installation reported failures (usually missing offline speech models)."
    Write-Warn2 "Install models per docs/MODEL_SETUP.md, then re-run verify_installation.py."
}
exit $verifyExit
