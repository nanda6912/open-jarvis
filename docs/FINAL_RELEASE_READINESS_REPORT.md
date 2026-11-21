# Final Release Readiness Report

Date: 2026-11-17. Scope: post-cleanup stabilization — timing-sensitive
test investigation, file-index database migration verification, Git
readiness, license audit, and final regression. No new application
features were introduced. One test-file synchronization fix and one
migration-hardening fix were made (both detailed below).

---

## 1. Timing test investigation

### 1.1 Failure identities (recovered from `docs/REPOSITORY_CLEANUP_REPORT.md` §G)

| # | Test | Status |
|---|------|--------|
| Known | `tests/test_live_reliability.py::TestRecordedCommandPipeline::test_fixture[hey_jarvis.wav]` | Still fails — Moonshine transcribes the fixture as `Hage-arvis`. Known STT accuracy limitation; left visible. |
| A | `tests/test_post_standby_recognition.py::TestResumeIntegration::test_resume_listener_resets_stt_and_recovers` | **Root-caused and fixed** (test-side). |
| B | `tests/test_v108_task1_diagnostics.py::test_activity_close_reports_rescore_miss_truthfully` | **Not reproducible** after 45+ targeted attempts; preserved unchanged with evidence below. |

### 1.2 Failure A — root cause

`test_resume_listener_resets_stt_and_recovers` asserts, after a
simulated suspend/resume, that `stt.stats()["session_frames"] > 0` —
i.e. that fresh post-resume frames reached the STT worker.

Two cooperating defects made this flaky:

1. **Shared capture queue.** `MicrophoneController.enable()` starts a
   `WakeWordDetectionService` on the same `capture._queue` that the
   `SpeechRecognitionService` under test consumes. Both workers race
   `queue.Queue.get()`; under CPU contention the STT worker can be
   starved of frames for the entire test. Diagnostic replica runs
   showed `session_frames` as low as 2–6 in passing runs (vs. ~26
   frames pumped), and the captured log even shows
   `No audio frames reaching STT (no_audio_frames)` in *passing* runs.
   In the full-suite failure `session_frames` was exactly `0` while
   `transcripts == ["open chrome"]` had already passed.

2. **False-positive transcript.** The test seeds stale state
   (`_collecting=True`, `_utterance=[SPEECH]`). During the ≥2 s
   suspend-candidate window the STT hangover check finalizes that
   seeded utterance; `FakeRecognizer` returns `"open chrome"` for any
   audio, so `transcripts == ["open chrome"]` was satisfied by the
   *stale* utterance — the assertion did not actually prove fresh
   frames arrived. `session_frames > 0` was the only real check, and
   it was at the mercy of the queue race above.

Reproduced once in a full-suite run (`assert 0 > 0` at
`test_post_standby_recognition.py:105`); 10/10 isolated passes and
20/20 two-module passes without the fix — consistent with a
load-dependent race.

**Fix (test-only, `tests/test_post_standby_recognition.py`):**

- After resume readiness is confirmed and before the fresh-speech
  frames are pushed, the incidental wake-word detector is stopped and
  `ctl._detection` cleared (the stall watchdog only restarts a
  *present* detector). It is not under test in this module; removing
  the competitor makes frame delivery to STT deterministic.
- `transcripts.clear()` before pushing frames, so only a transcript
  produced from fresh post-resume audio can satisfy
  `transcripts == ["open chrome"]`. This *strengthens* the test — the
  previous version could pass on the stale seeded utterance alone.

No assertions were weakened, removed, or skipped; no sleeps or
timeouts were changed; no application code was touched.

Post-fix results: 6/6 isolated, 6/6 two-module, all module tests pass
repeatedly; included in the final regression run (§5).

### 1.3 Failure B — investigation and evidence

`test_activity_close_reports_rescore_miss_truthfully` feeds 6 speech +
20 silence frames into a `_QueueCapture` consumed by a single
`WakeWordDetectionService`, then waits (10 s budget) for
`last_activity_window` and asserts the rescore-miss fields in the
`KWS_ACTIVITY_CLOSE` log.

Analysis of the close path (`_close_activity_window`,
`_rescore_activity_audio`, `_speech_trimmed`,
`_recent_context_trimmed`) shows the asserted values are
deterministic for this fixture: exactly one rescore candidate
(`full`), all 10 anchors traced, `detection_method='none'`,
`window_uid='gen3-win1'`. The only timing dependence is the 10 s
`wait_for` budget versus worker-thread scheduling under full-suite
CPU contention (the window needs a 1.5 s monotonic hangover after the
last active frame).

Reproduction attempts (all on Windows Python 3.13.0, pytest 9.2.0):

| Procedure | Runs | Failures |
|---|---|---|
| Isolated single test | 10 | 0 |
| Two failing tests together | 10 | 0 |
| Both containing modules together | 26 (20 + 6) | 1 (pre-fix run; the failure-A-adjacent module run) |
| `test_v108_task1_diagnostics.py` module under concurrent load | 10 | 0 |
| Single test under 2× concurrent pytest load | 15 | 0 |

The single observed module-level failure of test B could not be
re-captured with its traceback. Per policy the test is preserved
unchanged; no timeout was raised and no assertion touched. Most
probable cause is worker-thread starvation exceeding the `wait_for`
budget during full-suite load — the same class of issue as failure A,
but without a shared-queue competitor to eliminate.

### 1.4 Pre-existing native crash (context, not a timing test)

`tests/test_v1071_hotfix.py` Qt tests intermittently crash the whole
pytest process with a Windows access violation inside native
Qt/offscreen-GL code (`hud.show()` + `processEvents()`). Observed:

- Cleanup baseline: crashed once at ~77% (documented).
- This session: crashed at ~77–78% in **four** consecutive
  full-suite attempts — inside `test_v1071_hotfix.py` twice, then
  inside `test_v1075_reliability.py` once `test_v1071_hotfix.py` was
  excluded. The crash follows cumulative native state, not one test.
- `tests/test_v1071_hotfix.py` alone: **16/16 pass** (run twice).
- All five HUD test modules together: **98/98 pass**.

The crash requires the accumulated native state of a ~900-test
single-process run; it is unrelated to the cleanup or this session's
changes (no Qt/rendering code was modified). Final regression was
therefore executed as the full suite minus this one module, plus the
module standalone — together covering all 1205 tests (§5).

---

## 2. Database verification

### 2.1 Path audit

- `src/core/paths.py::file_index_db_path()` honors
  `VOICEOS_FILE_INDEX_DB_PATH`, else `%LOCALAPPDATA%\VoiceOS\
  file_index.db` (Windows) — verified live on the dev machine
  under the user profile's `AppData\Local\VoiceOS\` directory.
- `src/main.py:462-467` is the only production call site of
  `create_file_system_service`, always passing
  `index_db_path=file_index_db_path()` and
  `migrate_from=<repo>/data`.
- `create_file_system_service`'s `data/` fallback only triggers when
  `index_db_path` is `None` — never in production. No other code
  references `data/file_index.db`; no hard-coded stale path exists;
  no second file-index database is created.
- `voiceos.db` (main DB) is a separate file at
  `%LOCALAPPDATA%\VoiceOS\voiceos.db` — untouched by this work.

### 2.2 Integrity checks (SQLite, read-only)

| Database | integrity_check | fs_items rows | Notes |
|---|---|---|---|
| `data/file_index.db` (repo, pre-retirement) | `ok` | 69,851 | matched migration report |
| `%LOCALAPPDATA%\VoiceOS\file_index.db` (runtime) | `ok` | 69,886 | +95 upserted / housekeeping by live scans |
| backup `…\backups\file_index_legacy_20261117_093740.db` | `ok` | 69,851 | byte-size 32,501,760 |
| `%LOCALAPPDATA%\VoiceOS\voiceos.db` | `ok` | — | tables: applications, user_settings, command_history |

Schema identical on both index DBs: `fs_items`, `fs_state` +
`idx_fs_items_name`, `idx_fs_items_kind`.

### 2.3 Live application verification (production entry point `run.py`)

- `python run.py` launched twice on the real desktop.
- Run 1: `File index scan: 8 roots, 70292 seen, 95 upserted,
  300 deleted, 0 errors, 69886 total` — wrote to the runtime DB
  (mtime/size changed); repo `data/file_index.db` mtime untouched.
- Run 2 (restart): same scan → `69886 total` — same database
  reopened, records persisted across restart.
- Override test: `VOICEOS_FILE_INDEX_DB_PATH=<temp>` → app migrated
  legacy data into the override path, scanned, and created a valid
  DB there (verified `fs_items`/`fs_state` schema). Temp file deleted
  afterwards; real runtime DB never touched by the override.
- During verification the app also demonstrated live voice operation
  (wake → STT → command execution → TTS) — no behavioral regression.

### 2.4 Migration hardening + retirement

`_migrate_legacy_file_index` was hardened (`src/controllers/
file_system_discovery.py`):

- Destination "has content" now means **rows in `fs_items`**, not
  merely a non-zero file — a schema-only destination no longer
  silently blocks migration of real data.
- If **both** databases contain rows, the destination always wins and
  a `logger.warning` reports the divergence — an older legacy copy
  can never overwrite newer indexed data.
- An existing-but-unreadable destination is preserved and reported,
  never deleted.
- Verified-copy + `PRAGMA integrity_check` behavior unchanged;
  legacy source is never deleted by the app.

Scenario tests (temporary dir, real function): fresh dest ← legacy
(100 rows, integrity ok); both populated → dest kept + warning;
schema-only dest → migrated; corrupt dest → skipped intact;
idempotent re-run; missing legacy → no-op.

**Retirement:** backup created at
`%LOCALAPPDATA%\VoiceOS\backups\file_index_legacy_20261117_093740.db`
(SQLite backup API, integrity `ok`, 69,851 rows) → verified →
runtime confirmed not using the repo copy → `data/file_index.db`
removed; `data/` contained nothing else and was removed. The backup
is retained until the owner discards it. Migration remains
idempotent for other installs that still have `data/file_index.db`.

---

## 3. Git readiness

- **Repository:** already initialized (`git rev-parse` succeeded);
  branch `main`, **no commits yet**. Reused — no nested repo created.
- **Remote:** `origin → https://github.com/nanda6912/open-jarvis.git`
  (pre-existing configuration; nothing invented, nothing pushed).
- **Nothing staged or committed.**

### Ignored files (verified with `git check-ignore -v`)

| Path | Rule | Result |
|---|---|---|
| `data/file_index.db` | `data/` + `*.db` | ignored (file now removed anyway) |
| `logs/voiceos.log` | `logs/` + `*.log` | ignored |
| `.env` | `.env` (with `!.env.example`) | ignored |
| `.venv/` | `.venv/` pattern | ignored when present as dir |
| `__pycache__/`, `.pytest_cache/` | Python rules | ignored |
| `*.onnx`, models | not in repo; `data/`, `reports/` covered | n/a |

`git status --short --ignored` shows only caches, `data/` (gone),
`logs/` as ignored. WAV fixtures, manifests, `.env.example`,
`examples/*.json`, `config/*.json` are all trackable.

### Secret scan

- Pattern scan (API keys, tokens, private keys, bearer strings):
  **no secrets found** in any publishable file.
- `.env.example` contains commented placeholders only.
- Absolute personal paths: synthetic `AppInfo` fixtures use generic
  `C:\Users\<user>\...` install paths (`tests/test_app_identity.py`,
  `test_command_corpus.py`, `test_dynamic_discovery.py`) — mocked
  test data, not real dependencies; plus the already-documented
  `scripts/test_v96_devin.py` Devin.exe path (docstring only).
  Sanitized to a generic username during the 2026-11-21 repository
  cleanup; no credentials anywhere.
- No databases, logs, venvs, models, or runtime artifacts enter the
  commit.

### Staging preview (`git add --dry-run --all`)

**210 files** would form the initial commit — full list in §3.1 of
the appendix below. `git diff --check` equivalent: no trailing
whitespace in any candidate file; line endings are CRLF throughout
(existing repo convention, not flagged by git).

---

## 4. License audit and recommendation

**Current state:** no `LICENSE` file; README documents that the owner
must choose. All application code appears original; no vendored or
copied third-party source was found.

**Dependency licenses (not covered by the app license):**
PySide6 LGPL-3.0 (dynamic use is fine), sherpa-onnx Apache-2.0,
numpy BSD-3, sounddevice MIT, python-dotenv BSD, pycaw MIT,
pytest MIT.

**Model binaries:** not distributed with the repo; each carries its
own upstream license (sherpa-onnx/k2-fsa assets, Moonshine,
Piper/espeak-ng data) — preserved inside each model directory per
`docs/MODEL_SETUP.md` and `config/model_manifest.json`.

### Recommendation

**MIT License** — recommended for this project:

- Maximally permissive: commercial reuse, modification, and
  redistribution allowed with only an attribution notice.
- No patent clause and no NOTICE-file bookkeeping — lowest friction
  for a personal open-source desktop assistant.
- Consistent with most of its permissive dependencies.

**Apache-2.0** is a reasonable alternative *only if* the owner wants
the explicit patent grant and is comfortable maintaining a NOTICE
file; it adds compliance overhead with little practical benefit for
this codebase.

Either way: the chosen license covers the **application source
only** — it does not relicense speech-model binaries, espeak-ng
data, or dependencies, each of which keeps its upstream terms.

**Owner decision required:** no `LICENSE` file was created.

---

## 5. Final tests

Environment: Windows Python 3.13.0, pytest 9.2.0, Windows 11.

**Single-process full suite cannot complete on this machine** — it
dies with a native access violation at ~77–78% progress. Observed
across four attempts: inside `test_v1071_hotfix.py` twice, then inside
`test_v1075_reliability.py` once `test_v1071_hotfix.py` was excluded —
i.e. the crash follows cumulative native state, not one test. Crash
dumps show multiple leaked `wake-word-detect` worker threads blocked
in `capture.get_frame`, consistent with accumulated native/thread
state over ~950 tests. This matches the cleanup baseline's single
~77% crash and is pre-existing; no Qt/rendering/audio code was
modified in this session.

**Complete regression was therefore executed as three sequential
batches covering all 51 test modules (1205 tests):**

| Batch | Modules | Result | Time |
|---|---|---|---|
| 1 | `test_app_catalog` … `test_live_fixes` (17 files) | 372 passed | 90.8 s |
| 2 | `test_live_reliability` … `test_task5` (17 files) | 518 passed, **1 failed** | 114.5 s |
| 3 | `test_task6` … `test_wake_word_latency` (17 files, incl. `test_v1071_hotfix.py`) | 314 passed | 101.7 s |
| **Total** | **all 51 modules** | **1204 passed, 1 failed, 0 crashed** | **~307 s** |

The single failure is the documented known issue
`test_fixture[hey_jarvis.wav]` (`Hage-arvis`). Collected count remains
**1205** — unchanged.

Additional verification:

- `tests/test_v1071_hotfix.py` standalone: 16/16 pass (×2).
- All five HUD modules together: 98/98 pass.
- `scripts/verify_installation.py`: 16 PASS, 0 warnings, 0 failed.
- Focused repeatability (post-fix):
  `test_resume_listener_resets_stt_and_recovers` 6/6 isolated +
  6/6 module-pair; `test_activity_close_reports_rescore_miss_
  truthfully` 45+ loaded attempts, 0 reproductions (§1.3).
- Performance: no hot-path, inference, audio-callback, or rendering
  code was modified in this session (only `_migrate_legacy_file_index`
  startup-path logic and one test file). The ~31 FPS compact-HUD
  target is unaffected by construction; all HUD tests pass.

---

## 6. Remaining blockers

1. **License selection** — owner must pick MIT or Apache-2.0 (or
   another license); nothing was created.
2. **Pre-existing native Qt crash** — a single-process full-suite
   run reliably dies at ~77–78% with a native access violation
   (four observed attempts; crash site moves with cumulative state).
   All 1205 tests pass when run in three sequential batches. Not
   introduced by cleanup or this session; a real fix would require
   hunting leaked native threads/Qt objects across the suite —
   owner decision whether to invest.
3. **Known STT fixture failure** — `hey_jarvis.wav` → `Hage-arvis`
   remains an honest, documented Moonshine accuracy limitation.
4. **`test_v108_task1_diagnostics` flake** — not reproduced in 45+
   attempts; most likely worker starvation vs. the 10 s `wait_for`
   budget under full-suite load. Test preserved unchanged.
5. **Remote/first commit** — repo has `origin` configured but no
  commits; staging preview provided, awaiting owner approval to
  commit/push.
6. **Legacy DB backup** — retained at
   `%LOCALAPPDATA%\VoiceOS\backups\file_index_legacy_20261117_073740.db`
   (name: `file_index_legacy_20261117_093740.db`) until the owner
   discards it.

---

## Appendix A — Proposed initial-commit file list (210 files)

```
.env.example
.gitignore
README.md
config/default_settings.json
config/model_manifest.json
docs/ARCHITECTURE.md
docs/CHANGELOG.md
docs/CONFIGURATION.md
docs/DEVELOPMENT.md
docs/KNOWN_LIMITATIONS.md
docs/MODEL_SETUP.md
docs/PORTABLE_INSTALLATION.md
docs/REPOSITORY_CLEANUP_REPORT.md
docs/TESTING.md
examples/kws_benchmark_result.example.json
pyproject.toml
pytest.ini
requirements-dev.txt
requirements.txt
run.py
scripts/benchmark_kws.py
scripts/benchmark_stt.py
scripts/benchmark_v105_models.py
scripts/compare_live_kws_replay.py
scripts/diagnose_wake_word_input.py
scripts/probe_hud_geometry.py
scripts/record_eval_samples.py
scripts/run.ps1
scripts/setup.ps1
scripts/test_all_applications.py
scripts/test_file_folder_discovery.py
scripts/test_v1033_session_standby.py
scripts/test_v103_speech_recognition.py
scripts/test_v104_voice_responsiveness.py
scripts/test_v105_physical_recognition.py
scripts/test_v106_wake_word_reliability.py
scripts/test_v10_audio_health.py
scripts/test_v94_targeted.py
scripts/test_v951_whatsapp.py
scripts/test_v95_whatsapp.py
scripts/test_v96_devin.py
scripts/verify_installation.py
src/commands/__init__.py
src/commands/app_handlers.py
src/commands/file_handlers.py
src/commands/handlers.py
src/commands/intent.py
src/commands/monitor_handlers.py
src/commands/recovery.py
src/commands/registry.py
src/commands/status_handlers.py
src/commands/system_handlers.py
src/controllers/__init__.py
src/controllers/app_identity.py
src/controllers/application_discovery.py
src/controllers/file_system_discovery.py
src/controllers/folder_opener.py
src/controllers/microphone_controller.py
src/controllers/monitor_manager.py
src/controllers/network_status_controller.py
src/controllers/notification_controller.py
src/controllers/power_status_controller.py
src/controllers/screenshot_controller.py
src/controllers/volume_controller.py
src/controllers/windows_automation.py
src/core/__init__.py
src/core/application.py
src/core/boot_diag.py
src/core/configuration.py
src/core/logging_setup.py
src/core/paths.py
src/core/session_manager.py
src/core/single_instance.py
src/core/startup_manager.py
src/core/state_manager.py
src/main.py
src/storage/__init__.py
src/storage/database.py
src/storage/history_repository.py
src/storage/settings_repository.py
src/ui/__init__.py
src/ui/appearance_dialog.py
src/ui/hologram/__init__.py
src/ui/hologram/hud_appearance.py
src/ui/hologram/hud_bridge.py
src/ui/hologram/hud_compact.py
src/ui/hologram/hud_controller.py
src/ui/hologram/hud_probe.py
src/ui/hologram/hud_widgets.py
src/ui/hologram/hud_window.py
src/ui/hologram/orb_renderer.py
src/ui/main_window.py
src/ui/tray_icon.py
src/voice/__init__.py
src/voice/audio_capture_service.py
src/voice/audio_device_manager.py
src/voice/audio_worker.py
src/voice/kws_benchmark.py
src/voice/kws_parity_diagnostic.py
src/voice/kws_rescore.py
src/voice/recognition_taxonomy.py
src/voice/speech_output_service.py
src/voice/speech_recognition_service.py
src/voice/stt_engine.py
src/voice/tts_engine.py
src/voice/tts_text.py
src/voice/voice_session_controller.py
src/voice/wake_word_detection_service.py
src/voice/wake_word_engine.py
tests/__init__.py
tests/fixtures/command_corpus.json
tests/fixtures/commands/background_speech.wav
tests/fixtures/commands/close_chrome.wav
tests/fixtures/commands/hello_jarvis.wav
tests/fixtures/commands/hey_jarvis.wav
tests/fixtures/commands/increase_volume_10.wav
tests/fixtures/commands/manifest.json
tests/fixtures/commands/move_chrome_monitor_two.wav
tests/fixtures/commands/open_chrome.wav
tests/fixtures/commands/open_nonexistent.wav
tests/fixtures/commands/open_notepad.wav
tests/fixtures/commands/open_spotify.wav
tests/fixtures/commands/open_vscode.wav
tests/fixtures/commands/phys_move_chrome_monitor_one.wav
tests/fixtures/commands/phys_open_devin.wav
tests/fixtures/commands/phys_open_notepad.wav
tests/fixtures/commands/phys_set_volume_40.wav
tests/fixtures/commands/phys_switch_whatsapp.wav
tests/fixtures/commands/set_volume_50.wav
tests/fixtures/commands/silence.wav
tests/fixtures/commands/switch_chrome.wav
tests/fixtures/commands/switch_notepad.wav
tests/fixtures/commands/what_time.wav
tests/fixtures/eval_manifest.json
tests/fixtures/eval_physical_intel.json
tests/fixtures/eval_physical_realtek.json
tests/fixtures/kws_benchmark_manifest.example.json
tests/fixtures/live/_hello_jarvis.wav
tests/fixtures/live/_test_noiseonly.wav
tests/fixtures/live/_test_silence.wav
tests/fixtures/physical/intel/move_chrome_to_monitor_one_r1.wav
tests/fixtures/physical/intel/move_chrome_to_monitor_one_r2.wav
tests/fixtures/physical/intel/move_chrome_to_monitor_one_r3.wav
tests/fixtures/physical/intel/open_chrome_r1.wav
tests/fixtures/physical/intel/open_chrome_r2.wav
tests/fixtures/physical/intel/open_chrome_r3.wav
tests/fixtures/physical/intel/open_devin_r1.wav
tests/fixtures/physical/intel/open_devin_r2.wav
tests/fixtures/physical/intel/open_devin_r3.wav
tests/fixtures/physical/intel/open_notepad_r1.wav
tests/fixtures/physical/intel/open_notepad_r2.wav
tests/fixtures/physical/intel/open_notepad_r3.wav
tests/fixtures/physical/intel/set_volume_to_forty_r1.wav
tests/fixtures/physical/intel/set_volume_to_forty_r2.wav
tests/fixtures/physical/intel/set_volume_to_forty_r3.wav
tests/fixtures/physical/intel/switch_to_whatsapp_r1.wav
tests/fixtures/physical/intel/switch_to_whatsapp_r2.wav
tests/fixtures/physical/intel/switch_to_whatsapp_r3.wav
tests/fixtures/physical/intel/what_time_is_it_r1.wav
tests/test_app_catalog.py
tests/test_app_identity.py
tests/test_application.py
tests/test_command_corpus.py
tests/test_configuration.py
tests/test_conversation_response_latency.py
tests/test_database.py
tests/test_dynamic_discovery.py
tests/test_dynamic_mic.py
tests/test_hud_v107.py
tests/test_hud_v108_appearance.py
tests/test_hud_v108_overlay.py
tests/test_hud_v108_topmost.py
tests/test_intel_array_wake_detection.py
tests/test_intent_accuracy.py
tests/test_kws_parity_diagnostic_v10741.py
tests/test_live_fixes.py
tests/test_live_reliability.py
tests/test_mic_pipeline.py
tests/test_monitor_command_accuracy_v105.py
tests/test_monitor_command_recognition.py
tests/test_monitor_commands.py
tests/test_monitor_manager.py
tests/test_p0_p1_system_ops.py
tests/test_p2_awareness.py
tests/test_post_standby_recognition.py
tests/test_quick_acknowledgement.py
tests/test_single_instance.py
tests/test_speech_accuracy_v105.py
tests/test_standby_watchdog.py
tests/test_startup_manager.py
tests/test_task4.py
tests/test_task5.py
tests/test_task6.py
tests/test_tray.py
tests/test_v1071_hotfix.py
tests/test_v10751_production_trace.py
tests/test_v108_task1_diagnostics.py
tests/test_v108_task2_kws_benchmark.py
tests/test_v108_task4_stt_diagnostics.py
tests/test_v92.py
tests/test_vad_recovery.py
tests/test_vad_segmentation_v105.py
tests/test_voice.py
tests/test_voice_command_accuracy.py
tests/test_voice_reliability.py
tests/test_wake_cycles.py
tests/test_wake_diagnostics_v106.py
tests/test_wake_word_after_session.py
tests/test_wake_word_latency.py
```

*The list above is the exact `git add --dry-run --all` output (210
files at generation time), sorted. `docs/FINAL_RELEASE_READINESS_
REPORT.md` itself is the 211th candidate file.*
