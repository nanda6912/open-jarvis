# VoiceOS — Changelog

Milestone-level history. Package version remains `1.0.0`
(`pyproject.toml`, `config/default_settings.json`); V-numbers are
documentation milestones, not semver releases.

## Repository cleanup & portability (current)

- Centralized runtime path resolution in `src/core/paths.py`
  (`VOICEOS_HOME`, `VOICEOS_*_MODEL_DIR`, `VOICEOS_DB_PATH`,
  `VOICEOS_LOG_DIR`, legacy `VOICEOS_DATA_DIR` alias).
- Application logs and `boot_diag.log` moved from `<repo>\logs\` to
  `<runtime>\logs\`.
- File-index database moved from `<repo>\data\file_index.db` to
  `<runtime>\file_index.db` with verified one-time migration
  (SQLite backup + integrity check; legacy file preserved).
- KWS parity diagnostic report redirected to `<runtime>\reports\`.
- Added `config/model_manifest.json`, `scripts/setup.ps1`,
  `scripts/run.ps1`, `scripts/verify_installation.py`.
- Rewrote README; added `docs/` set (architecture, development,
  configuration, model setup, testing, limitations,
  portable installation, cleanup report).
- Removed `reports/` placeholder; `data/` and `logs/` are now fully
  git-ignored runtime locations (no longer tracked).
- No behavioral, recognition, audio or rendering changes.

## V10.7.1 — HUD startup & diagnostic hotfix

- Fixed HUD startup `NameError`: `main.py` passed
  `microphone_controller=micro_controller`; the real variable is
  `mic_controller`. HUD now initializes on VoiceOS startup.
- Wake-attempt diagnostics renamed to accurate terminology: tracker
  now records **audio activity windows** (`ACTIVITY_WITHOUT_KEYWORD` /
  `WAKE_MATCHED`), never "failed wake attempts". Log line now reports
  `active_frames`, `active_audio_seconds`, `keyword_detected`, queue
  and drop counters.
- `active_audio_seconds` is now computed from real KWS-input sample
  counts at 16 kHz (was `voiced_frames * 0.1` assumed frame period).
- V10.6 record/stats names kept as compatibility aliases
  (`attempts`, `last_attempt`, `attempts_recorded`, `attempts_missed`,
  `detected`, `keyword`).
- HUD: new "WAKE WORD DETECTED" status shown only on a real keyword
  emission; WAKE WORD panel row now shows detector state
  (processing/suspended/match count/drops) instead of "—".
- KWS untouched: threshold 0.25, six variants, model, stream
  lifecycle, 0.008 activity gate and 1.5 s hangover all preserved.
- 20 new/updated tests; suite 1,064 passed + 1 pre-existing live-STT
  fixture failure.

## V10.7 — 3D reactive holographic HUD

- New `src/ui/hologram/` package: genuine `QOpenGLWidget` 3D orb
  (sphere core, 3 torus rings, particle shell, custom GLSL Fresnel/
  pulse/scan shaders), floating SYSTEM/DESKTOP/ACTIVITY glass panels,
  live waveform, event bridge, read-only system probes, lifecycle
  controller with Monitor-2 (LEFT) placement.
- Voice-reactive: standby breathing, wake pulse, listening level from
  shared mic stats, processing scan, TTS glow driven by a measured
  ~10 Hz RMS envelope of the actual Piper waveform (new
  `SpeechOutputService.set_level_callback` + `ttsEnvelope` signal).
- Additive voice-side hooks only: `VoiceSessionController.last_intent`,
  `ttsEnvelope` signal, `hud` config section, tray show/hide action.
- 30 HUD tests; measured 30.4 fps / ~13% single-core / ~240 MB RSS on
  Intel Iris Xe. Physical voice validation pending.

## V10.6 — Wake-word reliability investigation

- Verified sherpa-onnx 2.0.1 KWS API: `get_result` returns a keyword
  string or `""`; **no confidence/score is exposed** — the 0.25
  threshold cannot reject near-misses.
- Fixed stale KWS stream across standby handover: `resume_detection()`
  now calls `engine.reset()` after the queue flush, so post-detection
  frames can no longer leak into the next standby period.
- Added PortAudio overflow accounting: `status_warnings`,
  `overflow_callbacks`, bounded timestamped `recent_status_events` in
  capture stats (driver-level drops were previously invisible).
- Added per-attempt speech diagnostics to `WakeWordDetectionService`:
  every voiced stretch produces a bounded record (missed *and*
  detected) with RMS/peak, per-channel snapshot, selected channel,
  inference count/ms, queue depth, drops delta.
- Added bounded (~5 s) in-memory ring buffer of exact KWS-input audio;
  diagnostic only, never persisted, cleared on stop/session end.
- 10 new tests; 110-row physical trial worksheet
  (`reports/v106_wake_word_trials.csv`). **Physical root cause still
  pending.**

## V10.5 — Speech accuracy & command understanding

- Fixed empty-transcript diagnostics: utterance stats are now snapshotted
  before `_reset_utterance()` (was logging `voiced=0, frames=1` always).
- Noise-floor calibration no longer learns from frames at/above the
  effective threshold (fixes 0.0955 vs 0.00005 contamination).
- Hangover no longer finalizes utterances while the capture gate is
  paused (TTS-boundary clipping fix).
- `vad_accepted_stt_empty` now gets a throttled spoken clarification.
- "Open this PC"/"my computer" resolve to File Explorer, never PC
  Manager. New `SEARCH_WEB` intent routes web requests to the browser
  before file-search rules. Monitor phrases accept into/onto and
  normalize "less"→"left".
- `scripts/benchmark_v105_models.py` + physical-trial harness;
  Moonshine Base INT8 kept as production STT. Physical Intel
  recordings decode poorly for both models — capture conditions
  suspected; physical validation outstanding.

## V10.4 — Responsiveness & command accuracy

- Wake-word instrumentation (per-channel levels, KWS input quality,
  inference timing, keyword SHA-256), rolling 1 s windows.
- Quick/full/silent acknowledgement modes; misheard-command recovery
  (`recovery.py`) with conservative fuzzy matching.
- Monitor-command recognition tests; response-latency metrics.

## V10.3.3 — Post-Goodbye wake recovery

- Root cause: `reset_stream()` left the sherpa KWS stream poisoned by
  trailing queued frames → alternating detect/miss. Fixed with
  `create_stream()` after every detection (preserved since).
- Channel selection frozen while capture gate paused (TTS bleed could
  hijack the selected mic channel).

## V10.3.1 / V10.3 — Intel-array & VAD reliability

- Intel Smart Mic Array 4-channel handling: open up to 4 ch, select
  strongest channel via EMA (never blind averaging).
- Adaptive VAD: noise-floor estimation, floor/cap bounds, pre-roll,
  hangover endpointing, starvation vs below-VAD distinction.

## V10.2 — Suspend/resume recovery

- Trusted Windows resume notifications; STT state reset
  (`notify_system_resume`); post-resume queue flush and watchdog
  verification window.

## V10.1 — File/folder discovery

- Local file-system index + discovery (`file_system_discovery.py`),
  OPEN_FOLDER/OPEN_FILE/SEARCH_FILES/SEARCH_FOLDERS/SEARCH_LOCAL_ITEMS/
  OPEN_LOCAL_ITEM intents.

## V10 — Audio health & standby diagnostics

- Callback-liveness tracking, missed-interval detection, standby
  watchdog, signal/clipping counters.

## V9.x — App/window reliability

- WhatsApp/Devin targeted fixes, application identity resolution,
  window geometry, monitor enumeration groundwork.

## V9.4/V9.5/V9.6 — Targeted reliability

- Per-application reliability passes (WhatsApp state, Devin).

## Tasks 1–6 (foundation)

- Project scaffold, config system, SQLite storage, PySide6 tray UI,
  single-instance mutex, startup-folder registration, mic capture,
  wake-word detection, voice session, intent registry, Windows
  app/window automation, multi-monitor management.
