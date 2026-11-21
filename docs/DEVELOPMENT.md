# Development Guide

## Environment setup

```powershell
git clone <repository-url>
cd Jarvis
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1 -Dev        # .venv + runtime & dev deps
.\scripts\verify_installation.py
```

Requires Python 3.12+ (3.13 verified) and Windows 10/11. Model files
are not in the repo — see `docs/MODEL_SETUP.md`; tests that need real
models skip or use fixtures, so a fresh clone can run the suite
without them.

## Running

```powershell
.\.venv\Scripts\python.exe run.py            # console/debug
.\scripts\run.ps1 -Console                   # same, via script
.\scripts\run.ps1                            # detached (no console)
.\.venv\Scripts\python.exe run.py --startup status   # headless CLI
```

Logs: `%LOCALAPPDATA%\VoiceOS\logs\voiceos.log` (rotating) and
`boot_diag.log` (startup stage timings). Boot markers also print
elapsed-since-process-creation, useful for slow-start diagnosis.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest            # everything
.\.venv\Scripts\python.exe -m pytest -q tests/test_configuration.py
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

- `pytest.ini` sets `pythonpath = .` and `testpaths = tests`.
- Tests use `tmp_path`/temporary DBs; real user data is never touched
  (except tests that explicitly monkeypatch `VOICEOS_DATA_DIR`).
- Marker `slow` exists for model-loading tests.
- Known failure: `test_live_reliability.py::...test_fixture[hey_jarvis.wav]`
  (Moonshine transcribes the fixture as "Hage-arvis") — documented,
  not masked.

## Project layout

See the README "Project Structure" section and `docs/ARCHITECTURE.md`.
The `src/` package uses `src.<area>.<module>` imports; `run.py` and
`scripts/` insert the repo root into `sys.path`.

## Adding a command handler

1. Define the intent pattern in `src/commands/intent.py`
   (`RuleBasedIntentRecognizer`) and add the `Intent` type.
2. Implement a handler in the appropriate `src/commands/*_handlers.py`
   (app, file, monitor, system, status) using services from `_Svc` —
   never call Windows APIs directly from handlers.
3. Register it in `create_app_registry` (`src/commands/app_handlers.py`)
   with the correct permission (`safe`/`system`).
4. Add corpus entries in `tests/fixtures/command_corpus.json` and/or
   coverage in the intent/command tests.
5. Spoken responses must go through the session/TTS path; handlers
   return structured results, they never speak directly.

## Adding supported hardware

- **Microphone**: devices are enumerated dynamically
  (`audio_device_manager.py`); pin via tray selector or
  `voice.microphone_*` settings. The Intel array + Realtek headset
  combo on the dev machine is the reference configuration — see
  `tests/fixtures/eval_physical_intel.json` /
  `eval_physical_realtek.json` for recorded evaluation manifests.
- **Monitor**: `MonitorManager` enumerates displays live; logical
  numbering is layout-based. Persisted choice lives in
  `user_settings` (`hud.monitor`).
- New device classes need entries in the eval manifests plus recorded
  fixtures under `tests/fixtures/physical/` if you want regression
  coverage comparable to the existing hardware.

## Debugging

- `boot_diag.log` — timestamped startup stage markers.
- `voiceos.log` — full application log (INFO+).
- `VOICEOS_KWS_PARITY_DIAGNOSTIC=1` — records KWS frame-level parity
  data to `<runtime>/reports/v1074_production_kws_parity.json`.
- `scripts/diagnose_wake_word_input.py` — capture-time audio health.
- `scripts/probe_hud_geometry.py` — HUD geometry probe.
- `run.py --startup status` — headless startup-registration check.

## Scripts

`scripts/` holds three groups:

- **Benchmarks**: `benchmark_kws.py`, `benchmark_stt.py`,
  `benchmark_v105_models.py`, `compare_live_kws_replay.py`,
  `record_eval_samples.py`.
- **Diagnostics**: `diagnose_wake_word_input.py`,
  `probe_hud_geometry.py`.
- **Manual/hardware validation**: `test_v*.py`,
  `test_all_applications.py`, `test_file_folder_discovery.py` —
  interactive scripts that open real windows/apps; run only when you
  can supervise them.

## Conventions

- Compact, idiomatic Python; `from __future__ import annotations`.
- Type hints on public APIs; Protocols for engine interfaces.
- No new dependencies without need — the stack is deliberately small.
- Comments explain *why*; don't add noise comments.
