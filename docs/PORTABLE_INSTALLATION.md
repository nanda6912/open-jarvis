# Portable Installation

VoiceOS contains **no machine-specific paths** in production code. All
runtime locations resolve through `src/core/paths.py` at startup.

## Resolution order

| Concern | Default (Windows) | Override |
|---|---|---|
| Runtime root | `%LOCALAPPDATA%\VoiceOS` | `VOICEOS_HOME` (or legacy `VOICEOS_DATA_DIR`) |
| KWS models | `<root>\models\kws` | `VOICEOS_KWS_MODEL_DIR` |
| STT models | `<root>\models\stt` | `VOICEOS_STT_MODEL_DIR` |
| TTS models | `<root>\models\tts` | `VOICEOS_TTS_MODEL_DIR` |
| Main DB | `<root>\voiceos.db` | `VOICEOS_DB_PATH` |
| File index | `<root>\file_index.db` | `VOICEOS_FILE_INDEX_DB_PATH` |
| Logs | `<root>\logs` | `VOICEOS_LOG_DIR` |
| Screenshots | `<root>\screenshots` | — |
| Reports (diagnostics) | `<root>\reports` | — |

Non-Windows fallback: `~/.local/share/voiceos` (the app is only
supported on Windows; this is a graceful default).

`.env` in the repo root is loaded via python-dotenv; real environment
variables always take precedence over `.env` values.

## Fresh vs. existing installations

**Fresh install**

- Runtime dirs are created on demand (`paths.*` helpers `mkdir` safely).
- HUD monitor defaults to `auto` → primary display.
- Microphone defaults to Windows default input; devices are discovered
  at runtime — nothing is hard-coded.
- Missing models produce clear `[FAIL]` diagnostics from
  `verify_installation.py`, not crashes.

**Existing install (upgrade)**

- `voiceos.db`, settings, monitor/microphone preferences and models in
  `%LOCALAPPDATA%\VoiceOS\` are used unchanged.
- The legacy `<repo>\data\file_index.db` is copied once to
  `<root>\file_index.db` via the SQLite backup API and verified with
  `PRAGMA integrity_check`; the source file is never deleted and the
  migration is idempotent (skipped when the destination already has
  content).
- Logs move from `<repo>\logs\` to `<root>\logs\`; old log files are
  left untouched.

## Clone-to-run checklist

```powershell
git clone <repository-url>
cd Jarvis
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1            # .venv + deps + runtime dirs
.\scripts\verify_installation.py
# install models per docs/MODEL_SETUP.md, re-run verifier
.\scripts\run.ps1
```

`scripts\setup.ps1` and `scripts\run.ps1` work from any working
directory, any drive and paths containing spaces (they resolve the
repo root from `$PSScriptRoot`).

## Custom data location example

```powershell
$env:VOICEOS_HOME = "D:\VoiceOSData"
.\scripts\run.ps1
```

Or persist it in `.env`:

```
VOICEOS_HOME=D:\VoiceOSData
```
