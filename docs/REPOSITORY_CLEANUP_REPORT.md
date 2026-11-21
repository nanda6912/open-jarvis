# Repository Cleanup Report

Date: 2026-11-17. Scope: cleanup, portability, documentation and
GitHub readiness for VoiceOS. No recognition, audio, inference or
rendering behavior was changed.

## A. Files removed

| Path | Reason |
|---|---|
| `reports/` (directory + `.gitkeep`) | No report files existed; runtime reports now go to `<runtime>/reports/` (see `kws_parity_diagnostic.py`, `benchmark_v105_models.py`). |
| `data/.gitkeep` | `data/` is no longer a tracked location; the directory itself remains on disk holding the legacy `file_index.db` (user data, git-ignored). |
| `logs/.gitkeep` | Logs moved to `<runtime>/logs/`; `logs/` is fully git-ignored. Old log files left on disk untouched. |

No source modules, tests, fixtures or scripts were deleted. Every
`scripts/test_*.py` was audited and retained (see §C).

## B. Files consolidated

- `src/core/configuration.py::get_app_data_dir` now delegates to the
  new `src/core/paths.py::runtime_dir` (single source of truth);
  `VOICEOS_HOME` added, `VOICEOS_DATA_DIR` kept as legacy alias.
- `ConfigurationManager.database_path` honors `VOICEOS_DB_PATH`.
- KWS/STT/TTS engine default model dirs now resolve through
  `paths.kws_model_dir()` / `stt_model_dir()` / `tts_model_dir()`
  (env-aware, resolved at engine-creation time so `.env` works).
- `ScreenshotController` default output moved to
  `<runtime>/screenshots`.
- `scripts/benchmark_v105_models.py`,
  `scripts/test_v103_speech_recognition.py`,
  `scripts/diagnose_wake_word_input.py` now use `src.core.paths`
  instead of hard-coded `%LOCALAPPDATA%` paths; report outputs go to
  `<runtime>/reports`.
- `examples/kws_benchmark_result.example.json`: personal path
  sanitized to `%LOCALAPPDATA%\VoiceOS\models\kws`.
- `tests/test_kws_parity_diagnostic_v10741.py`: one assertion updated
  to expect the report under `runtime_dir()/reports` (test intent —
  absolute, CWD-independent path — unchanged).
- Documentation consolidated into `docs/` (see §I); `CHANGELOG.md`
  gained a cleanup entry. Stale README claims (Moonshine-tiny default,
  `docs/reports/`, `src/security/`, repo-local `logs/`) removed.

## C. Files retained (deliberate)

- **`src/ui/hologram/hud_window.py` + `hud_widgets.py`** — the earlier
  full-panel HUD. Still referenced by `src/voice/kws_parity_diagnostic.py`
  and `tests/test_hud_v107.py` / `test_v1071_hotfix.py`. `hud_compact.py`
  remains the single active HUD via `hud_controller.py`. Removing it
  would break diagnostics and regression tests for no runtime gain.
- **All 50 test modules** — version-named tests (`test_v92.py`,
  `test_v107*`, `test_v108_*`) protect real regressions; no duplicates
  justified removal. Test count unchanged (1205 collected before and
  after).
- **All `tests/fixtures/` audio** — command corpus, silence/negative
  samples, `phys_*.wav`, and `physical/intel/` recordings kept
  (hardware-specific regression data).
- **All `scripts/` utilities** — benchmarks, diagnostics and manual
  hardware-validation scripts provide unique value not covered by
  pytest; scripts stay flat because each resolves the repo root via
  `parents[1]` and no reorganizing benefit justified the churn.
- **`examples/kws_benchmark_result.example.json`** — documents the KWS
  benchmark output format.
- **`config/default_settings.json`, `pyproject.toml`, `pytest.ini`,
  `requirements*.txt`, `.env.example`, `run.py`** — all still required.

## D. Files changed

| File | Change |
|---|---|
| `src/core/paths.py` | **New.** Central runtime path resolution + `.env` loading. |
| `src/core/configuration.py` | `get_app_data_dir` delegates to `paths.runtime_dir`; `database_path` honors `VOICEOS_DB_PATH`; dropped unused `sys` import. |
| `src/core/boot_diag.py` | Diagnostics log now under `logs_dir()` with fallback; lazy, never breaks startup. |
| `src/main.py` | Logging → `logs_dir()`; file index → `file_index_db_path()` with `migrate_from=<repo>/data`. |
| `src/controllers/file_system_discovery.py` | `create_file_system_service` accepts `index_db_path`/`migrate_from`; added verified, idempotent `_migrate_legacy_file_index`. |
| `src/controllers/screenshot_controller.py` | Default dir → `runtime_dir()/screenshots`. |
| `src/voice/wake_word_engine.py` | `DEFAULT_MODEL_DIR`/`create_engine` → `paths.kws_model_dir()` (`VOICEOS_KWS_MODEL_DIR`). |
| `src/voice/stt_engine.py` | Same pattern → `VOICEOS_STT_MODEL_DIR`. |
| `src/voice/tts_engine.py` | Same pattern → `VOICEOS_TTS_MODEL_DIR`. |
| `src/voice/kws_parity_diagnostic.py` | `REPORT_PATH` → `runtime_dir()/reports/`. |
| `scripts/benchmark_v105_models.py`, `test_v103_speech_recognition.py`, `diagnose_wake_word_input.py` | Portable model/report paths via `src.core.paths`. |
| `tests/test_kws_parity_diagnostic_v10741.py` | Report-path expectation updated. |
| `.env.example` | Documents all `VOICEOS_*` variables. |
| `.gitignore` | Rewritten: runtime data, logs, reports, secrets, OS/IDE noise. |
| `README.md` | Complete rewrite (see §F/§G of README). |
| `docs/` | 7 new documents + CHANGELOG entry. |
| `config/model_manifest.json` | **New.** Machine-readable model requirements. |
| `scripts/setup.ps1`, `scripts/run.ps1`, `scripts/verify_installation.py` | **New.** |
| `examples/kws_benchmark_result.example.json` | Personal path sanitized. |

## E. Runtime data migration

- **File index:** `<repo>\data\file_index.db` (32.5 MB) →
  `%LOCALAPPDATA%\VoiceOS\file_index.db` via `sqlite3.Connection.backup`
  + `PRAGMA integrity_check`. Verified on this machine: integrity `ok`,
  69,851 `fs_items` rows preserved. Idempotent (skips when destination
  has content); legacy file left in place; failure deletes only the
  partial destination and logs an error.
- **Main DB:** already lived at `%LOCALAPPDATA%\VoiceOS\voiceos.db`;
  unchanged. `VOICEOS_DB_PATH` can now override it.
- **Logs:** `voiceos.log`/`boot_diag.log` now write to
  `%LOCALAPPDATA%\VoiceOS\logs\` (verified: fresh `boot_diag.log`
  markers and `voiceos.log` entries observed after the change). Old
  repo-local logs untouched.
- **Models/settings:** untouched under `%LOCALAPPDATA%\VoiceOS\models\`.

## F. Portability

- Environment variables: `VOICEOS_HOME`, `VOICEOS_DATA_DIR` (legacy),
  `VOICEOS_KWS_MODEL_DIR`, `VOICEOS_STT_MODEL_DIR`,
  `VOICEOS_TTS_MODEL_DIR`, `VOICEOS_DB_PATH`,
  `VOICEOS_FILE_INDEX_DB_PATH`, `VOICEOS_LOG_DIR` — all documented in
  `.env.example`, `docs/CONFIGURATION.md`, `docs/PORTABLE_INSTALLATION.md`.
- Supported: Windows 10/11 x64, Python ≥ 3.12 (3.13 verified).
- Install: `scripts/setup.ps1` (idempotent, `-Dev` for pytest) →
  `scripts/verify_installation.py` → install models per
  `docs/MODEL_SETUP.md` → `scripts/run.ps1`.
- No personal paths remain in publishable source; the only
  machine-specific literal left is inside `scripts/test_v96_devin.py`
  (a manual physical-validation script documenting a real Devin.exe
  path — retained as historical test evidence, not production code).
- Fresh installs default to primary display + Windows-default mic;
  existing installs keep saved monitor/mic/appearance settings.

## G. Test comparison

| Metric | Before | After |
|---|---|---|
| Collected | 1205 | 1205 |
| Passed | — (baseline run crashed at ~77% on a flaky native Qt test) | 1202 |
| Failed | — | 3 → effectively 1 known + 2 flaky |
| Known failure | `test_fixture[hey_jarvis.wav]` (Moonshine → "Hage-arvis") | same, still visible |

The two additional full-suite failures
(`test_post_standby_recognition.py::test_resume_listener_resets_stt_and_recovers`,
`test_v108_task1_diagnostics.py::test_activity_close_reports_rescore_miss_truthfully`)
are timing-sensitive tests that **pass in isolation** and are unrelated
to the cleanup (no KWS/STT logic changed). The earlier baseline crash
(`test_v1071_hotfix.py` access violation) likewise does not reproduce
in isolation — pre-existing flaky Qt/native behavior.

## H. Performance comparison

No hot-path changes: paths resolve once at startup/factory time; no
new filesystem scans, audio streams, model instances, polling or
dependencies were added. Inference config, KWS/STT/TTS models, audio
callback design, session logic, command security and OpenGL renderer
are untouched. HUD benchmark baseline (~31 FPS, ~0.48 ms `paintGL` at
320×320) is unchanged by construction; no re-measurement was needed
since no rendering code was modified.

## I. Final directory tree

```
Jarvis/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
├── run.py
├── config/
│   ├── default_settings.json
│   └── model_manifest.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHANGELOG.md
│   ├── CONFIGURATION.md
│   ├── DEVELOPMENT.md
│   ├── KNOWN_LIMITATIONS.md
│   ├── MODEL_SETUP.md
│   ├── PORTABLE_INSTALLATION.md
│   ├── REPOSITORY_CLEANUP_REPORT.md
│   └── TESTING.md
├── examples/
│   └── kws_benchmark_result.example.json
├── scripts/
│   ├── setup.ps1
│   ├── run.ps1
│   ├── verify_installation.py
│   ├── benchmark_kws.py / benchmark_stt.py / benchmark_v105_models.py
│   ├── compare_live_kws_replay.py / record_eval_samples.py
│   ├── diagnose_wake_word_input.py / probe_hud_geometry.py
│   └── test_*.py (manual/hardware validation)
├── src/            # unchanged layout + new core/paths.py
└── tests/          # unchanged suite + fixtures
```

Local-only (git-ignored, still on disk): `data/file_index.db` (legacy,
post-migration), `logs/*.log` (historical diagnostics).

## J. GitHub readiness

- **Git:** the directory is **not a git repository** (`git status`
  fails — no `.git`). Nothing was committed or pushed per instructions;
  `git init` + first commit is the owner's next step.
- **Secret scan:** no API keys/passwords/tokens found (pattern scan
  over all publishable files); only `.env.example` template exists.
- **Personal paths:** none in `src/`, `tests/`, `docs/`, `config/`,
  `examples/` (one documented exception in `scripts/test_v96_devin.py`).
- **`.gitignore`:** covers Python artifacts, venvs, `.env*`, `data/`,
  `logs/`, `reports/`, `*.db*`, `*.log`, caches, IDE/OS files. WAV
  fixtures, manifests and `.env.example` are explicitly preserved.
- **`git diff --check`:** N/A (no repo); edited files compile cleanly
  (`compileall` pass) and PowerShell scripts parse cleanly.
- **Uncommitted changes:** N/A — see Git note above.

## K. Remaining blockers / decisions

1. **License:** no `LICENSE` file exists; the owner must choose one
   before publication (README documents this).
2. **Git init + remote:** repository is not yet under version control;
   no remote URL exists, so README uses a `<repository-url>`
   placeholder.
3. **Model distribution:** ~1.5 GB of ONNX models are not committed;
   users must install per `docs/MODEL_SETUP.md` (official sherpa-onnx
   release assets, licenses preserved inside each archive).
4. **Flaky tests:** two timing-sensitive tests intermittently fail
   under full-suite load (pass in isolation); one rare native Qt crash
   observed once. Pre-existing; not introduced by this cleanup.
5. **HUD topmost-over-Chrome / click-through:** still lacking physical
   verification — documented in `docs/KNOWN_LIMITATIONS.md`.
6. **Old `data/file_index.db`:** intentionally left on disk (user
   data); safe to delete manually once the user is confident the
   migrated copy at `%LOCALAPPDATA%\VoiceOS\file_index.db` is good.
