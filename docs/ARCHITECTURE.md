# VoiceOS Architecture

This document describes the **current** implementation. Historical
milestone reports are summarized in `CHANGELOG.md`.

## Overview

VoiceOS is a Windows desktop application: a wake-word-activated,
fully-offline voice assistant with a holographic HUD. All speech
inference runs locally through ONNX models (sherpa-onnx); there are no
cloud services and no API keys.

## Process model

`run.py` → `src.main:main()`:

1. Parse args (`--startup enable|disable|status` runs headless).
2. Acquire the single-instance lock (`src/core/single_instance.py`,
   Windows named mutex with PID-file fallback).
3. `setup_logging(logs_dir())` — rotating `voiceos.log` under the
   runtime log dir.
4. `VoiceOSApplication` (`src/core/application.py`) — loads
   `ConfigurationManager`, `StateManager`, `SessionManager`, SQLite
   `Database` + repositories.
5. Qt `QApplication`, system tray (`src/ui/tray_icon.py`), main window
   (`src/ui/main_window.py`).
6. Voice pipeline (below).
7. Windows automation services + command registry.
8. Voice-session controller, HUD controller (isolated try/except),
   session countdown timer, tray wiring, event loop.

## Audio path

```
MicrophoneController ──► AudioCaptureService ──► 16 kHz mono float32
      │                        │
      │                   bounded frame queue
      │                   ┌────┴─────────────┐
      │              standby mode        session mode
      │                   │                   │
      │            WakeWordDetection    SpeechRecognition
      │            Service → KWS        Service → VAD → STT
      │            engine (sherpa-onnx) engine (Moonshine)
      └── audio_stats / device events ──► UI + HUD
```

- **Single stream**: one `sounddevice` input stream feeds both KWS and
  STT. `MicrophoneController` owns device selection (auto / pinned),
  Windows-default following, standby handover and suspend/resume
  recovery (`WM_POWERBROADCAST` native filter + clock-drift fallback).
- **Audio gate**: `SpeechOutputService` pauses capture while Jarvis
  speaks so TTS never feeds back into STT/KWS.
- Worker threads: KWS worker, STT worker, TTS worker — the GUI thread
  only receives queued `VoiceSignals`.

## Recognition path

`VoiceSessionController` coordinates:

1. Wake phrase → `AppState.STANDBY → ACTIVATED`, session activated,
   TTS speaks the activation response.
2. Each utterance → VAD segmentation → `SpeechRecognizer.transcribe`
   → `RuleBasedIntentRecognizer` (`src/commands/intent.py`).
3. `CommandRegistry` (`src/commands/registry.py`) dispatches intents to
   allowlisted handlers (`src/commands/*_handlers.py`) with permission
   checks, entity validation and guarded `CommandResult`s.
4. `MisheardCommandRecovery` (`src/commands/recovery.py`) offers
   conservative fuzzy reinterpretation against currently-running apps.
5. Responses are spoken via `SpeechOutputService`; transcripts and
   outcomes are recorded to `command_history`.

### Windows system control (Milestone 1)

- Power intents (`SLEEP_COMPUTER`, `RESTART_COMPUTER`,
  `SHUTDOWN_COMPUTER`, `CANCEL_PENDING_POWER_ACTION`) run through
  `PowerManager` (`src/controllers/power_manager.py`): sleep via the
  Win32 power API; shutdown/restart via fixed-argument `shutdown.exe`
  (`/t 15`, never a shell). Handlers in
  `src/commands/system_control_handlers.py` require the existing
  yes/no confirmation; the 15 s countdown arms only after the spoken
  acknowledgement finishes (the session controller's `after_speech`
  hook in `_handle_tts_finished`).
- Desktop intents (`SHOW_DESKTOP`, `MINIMIZE_ACTIVE_WINDOW`,
  `RESTORE_ACTIVE_WINDOW`, `CLOSE_ACTIVE_WINDOW`,
  `SWITCH_NEXT_WINDOW`, `SWITCH_PREVIOUS_WINDOW`,
  `SWITCH_NEXT_VIRTUAL_DESKTOP`, `SWITCH_PREVIOUS_VIRTUAL_DESKTOP`,
  `SNAP_ACTIVE_WINDOW_LEFT`, `SNAP_ACTIVE_WINDOW_RIGHT`) run through
  `DesktopController` (`src/controllers/desktop_control.py`), which
  verifies the foreground `hwnd` before acting and uses only fixed
  key chords for snap/virtual-desktop switching.

## Persistence

SQLite (`src/storage/database.py`), short-lived connections, foreign
keys on, schema created idempotently:

- `voiceos.db` — `applications`, `user_settings`, `command_history`.
- `file_index.db` — `fs_items`/`fs_state` file/folder index built by
  `FileSystemIndexer` (names/metadata only, never contents).

## UI

- `src/ui/main_window.py` — dark-themed status window.
- `src/ui/tray_icon.py` — `QSystemTrayIcon`, menu, wiring.
- `src/ui/hologram/` — HUD:
  - `hud_controller.py` — lifecycle, settings, monitor placement.
  - `hud_compact.py` — **active** compact always-on-top window.
  - `orb_renderer.py` — OpenGL orb (~30 FPS target).
  - `hud_bridge.py` — voice signals → HUD state (throttled timers).
  - `hud_probe.py` — desktop/window/system probes.
  - `hud_appearance.py` — appearance presets/parameters.
  - `hud_window.py` + `hud_widgets.py` — earlier full-panel HUD,
    retained for diagnostics (`kws_parity_diagnostic`) and tests.

## Runtime paths

`src/core/paths.py` is the single place that resolves filesystem
locations. See `docs/CONFIGURATION.md` for the environment variables
and `docs/PORTABLE_INSTALLATION.md` for the directory layout.

## Failure isolation

- Missing models → null engines + warnings (app still runs).
- HUD failure → logged, voice pipeline unaffected.
- Command handler failure → structured error result + spoken apology.
- Diagnostics (`boot_diag`, `kws_parity_diagnostic`) never raise into
  the audio path.
