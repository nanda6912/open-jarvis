<#
.SYNOPSIS
    Launch VoiceOS using the repository virtual environment.

.DESCRIPTION
    Works from any current working directory and any drive. Uses the
    project venv when present, otherwise falls back to `py -3`/`python`.
    The application itself enforces single-instance behaviour, so
    launching twice is safe (the second process exits quietly).

.PARAMETER Console
    Run with a visible console for debug output. Without it, VoiceOS is
    started detached (windowed, no console) on Windows.

.EXAMPLE
    .\scripts\run.ps1
    .\scripts\run.ps1 -Console
#>
[CmdletBinding()]
param(
    [switch]$Console
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runPy = Join-Path $RepoRoot "run.py"
$pythonArgs = @()

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$venvPythonw = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"

if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    # Fall back to the system launcher; VoiceOS may still run if deps exist.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $python = "py"; $pythonArgs = @("-3")
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $python = "python"
    } else {
        Write-Error "No Python found and .venv is missing. Run scripts\setup.ps1 first."
    }
}

Push-Location $RepoRoot
try {
    if ($Console) {
        & $python @pythonArgs $runPy @args
        exit $LASTEXITCODE
    }

    # Detached windowed launch (no console) unless -Console was given.
    if (Test-Path $venvPythonw) {
        Start-Process -FilePath $venvPythonw -ArgumentList "`"$runPy`"" -WorkingDirectory $RepoRoot
    } elseif (Test-Path $venvPython) {
        Start-Process -FilePath $venvPython -ArgumentList "`"$runPy`"" -WorkingDirectory $RepoRoot
    } else {
        & $python @pythonArgs $runPy @args
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
