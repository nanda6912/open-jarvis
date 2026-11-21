"""VoiceOS application entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is importable when run as ``python -m src.main``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QSystemTrayIcon  # noqa: E402

if sys.platform == "win32":  # noqa: E402
    import ctypes  # noqa: E402
    from PySide6.QtCore import QAbstractNativeEventFilter  # noqa: E402

from src.core.application import VoiceOSApplication  # noqa: E402
from src.core.boot_diag import mark as boot_mark  # noqa: E402
from src.core.logging_setup import setup_logging  # noqa: E402
from src.core.paths import file_index_db_path, logs_dir  # noqa: E402
from src.core.single_instance import InstanceGuard  # noqa: E402
from src.core.startup_manager import StartupError, StartupManager  # noqa: E402
from src.storage.database import DatabaseError  # noqa: E402
from src.ui.main_window import MainWindow  # noqa: E402
from src.ui.tray_icon import VoiceOSTray, wire_tray  # noqa: E402
from src.controllers.microphone_controller import MicrophoneController  # noqa: E402
from src.voice.audio_worker import VoiceSignals  # noqa: E402
from src.voice.speech_output_service import SpeechOutputService  # noqa: E402
from src.voice.speech_recognition_service import (  # noqa: E402
    SpeechRecognitionService,
)
from src.voice.stt_engine import create_recognizer  # noqa: E402
from src.voice.tts_engine import create_synthesizer  # noqa: E402
from src.voice.voice_session_controller import (  # noqa: E402
    VoiceSessionController,
)

logger = logging.getLogger(__name__)


if sys.platform == "win32":
    _WM_POWERBROADCAST = 0x0218
    _PBT_APMSUSPEND = 0x0004
    _PBT_APMRESUMESUSPEND = 0x0007
    _PBT_APMRESUMEAUTOMATIC = 0x0012

    class _MSG(ctypes.Structure):
        _fields_ = [
            ("hwnd", ctypes.c_void_p),
            ("message", ctypes.c_uint),
            ("wParam", ctypes.c_size_t),
            ("lParam", ctypes.c_size_t),
            ("time", ctypes.c_uint32),
            ("pt_x", ctypes.c_long),
            ("pt_y", ctypes.c_long),
        ]

    class _PowerBroadcastFilter(QAbstractNativeEventFilter):
        """Forwards WM_POWERBROADCAST suspend/resume to the mic
        controller — the trusted OS resume signal that complements
        the controller's validated clock-drift fallback."""

        def __init__(self, mic_controller: MicrophoneController):
            super().__init__()
            self._mic = mic_controller

        def nativeEventFilter(self, eventType, message):
            try:
                msg = _MSG.from_address(int(message))
            except Exception:  # noqa: BLE001
                return False, 0
            if msg.message != _WM_POWERBROADCAST:
                return False, 0
            if msg.wParam == _PBT_APMSUSPEND:
                self._mic.notify_system_suspend("WM_POWERBROADCAST")
            elif msg.wParam in (
                _PBT_APMRESUMESUSPEND, _PBT_APMRESUMEAUTOMATIC
            ):
                self._mic.notify_system_resume("WM_POWERBROADCAST")
            return False, 0


def reconcile_auto_start(
    app_core: VoiceOSApplication, startup_manager: StartupManager
) -> None:
    """Apply a persisted automatic-startup preference, if one exists.

    The user's choice made via the tray menu is stored in the
    ``user_settings`` table under ``startup.auto_start``. When no explicit
    preference exists the current registration state is left untouched,
    so development runs never modify the Startup folder unexpectedly.
    """
    override = app_core.settings_repository.get("startup.auto_start")
    if override is None:
        logger.debug("No persisted auto-start preference; skipping reconcile")
        return
    enabled = override.strip().lower() in ("1", "true", "yes", "on")
    try:
        changed = startup_manager.set_enabled(enabled)
        logger.info(
            "Auto-start reconciled to %s (changed=%s)", enabled, changed
        )
    except StartupError:
        logger.exception("Failed to reconcile automatic startup")


def announce_tray_startup(tray: VoiceOSTray, app_core: VoiceOSApplication) -> None:
    """Show a notification balloon pointing the user to the tray icon.

    Helps discoverability when VoiceOS starts minimized: Windows may place
    new tray icons in the hidden-icons overflow area, so a balloon makes
    it obvious where to right-click. Controlled by the
    ``interface.show_notifications`` setting.
    """
    if not app_core.config.get("interface.show_notifications", True):
        return
    try:
        tray.showMessage(
            app_core.config.app_name,
            f"{app_core.config.assistant_name} is running in the system tray. "
            "Right-click the tray icon for options.",
            QSystemTrayIcon.MessageIcon.Information,
            6000,
        )
        logger.info("Tray startup notification shown")
    except Exception:  # noqa: BLE001 - notification is best-effort
        logger.debug("Tray notification unavailable", exc_info=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="VoiceOS - Windows voice assistant (Jarvis).",
    )
    parser.add_argument(
        "--startup",
        choices=("enable", "disable", "status"),
        metavar="ACTION",
        help=(
            "Manage Windows automatic startup without launching the GUI: "
            "enable, disable or show status."
        ),
    )
    return parser.parse_args(argv)


def handle_startup_command(
    action: str,
    app_core: VoiceOSApplication | None = None,
    startup_manager: StartupManager | None = None,
) -> int:
    """Execute a ``--startup`` CLI command and return an exit code.

    This path never acquires the single-instance lock and never creates a
    ``QApplication`` — it only touches the Startup folder and the
    ``user_settings`` table.

    Args:
        action: One of ``enable``, ``disable``, ``status``.
        app_core: Optional pre-built application facade (for tests).
        startup_manager: Optional pre-built startup manager (for tests).

    Returns:
        0 on success, 1 on failure.
    """
    setup_logging(logs_dir())
    logger.info("CLI command: --startup %s", action)
    boot_mark("main: CLI command entered")

    app_core = app_core or VoiceOSApplication()
    startup_manager = startup_manager or StartupManager()

    try:
        # The settings repository needs its table to exist.
        app_core.database.initialize()
    except DatabaseError as exc:
        print(f"Error: cannot initialize database: {exc}")
        return 1

    repo = app_core.settings_repository

    try:
        if action == "enable":
            created = startup_manager.register()
            repo.set("startup.auto_start", "true")
            verified = startup_manager.is_registered()
            state = "enabled" if verified else "FAILED verification"
            print(
                f"Automatic startup {state}. "
                f"Shortcut: {startup_manager.shortcut_path} "
                f"({'created' if created else 'already present'})"
            )
            return 0 if verified else 1

        if action == "disable":
            removed = startup_manager.unregister()
            repo.set("startup.auto_start", "false")
            verified = not startup_manager.is_registered()
            print(
                f"Automatic startup {'disabled' if verified else 'FAILED verification'}. "
                f"Startup entry {'removed' if removed else 'was not present'}."
            )
            return 0 if verified else 1

        # status
        registered = startup_manager.is_registered()
        persisted = repo.get("startup.auto_start")
        print(f"Automatic startup: {'enabled' if registered else 'disabled'}")
        print(f"Startup entry: {startup_manager.shortcut_path}")
        print(f"Registered in Startup folder: {'yes' if registered else 'no'}")
        print(f"Persisted preference (user_settings): {persisted or 'not set'}")
        return 0
    except StartupError as exc:
        print(f"Error: {exc}")
        logger.error("Startup command failed: %s", exc)
        return 1


def _wake_label(mic_controller: MicrophoneController) -> str:
    """Human-readable wake-word status for the UI."""
    if mic_controller.is_listening:
        return "listening"
    if mic_controller.status == "idle":
        return "activated"
    if mic_controller.status == "error":
        return "error"
    return "off"


def main(argv: list[str] | None = None) -> int:
    """Start the VoiceOS application.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on clean exit).
    """
    args = parse_args(argv)
    boot_mark("main: arguments parsed")

    # CLI management commands run headless: no instance lock, no GUI.
    if args.startup is not None:
        return handle_startup_command(args.startup)

    # 1. Single-instance lock (before logging so a second launch exits
    #    quietly without creating windows, tray icons or log noise).
    guard = InstanceGuard()
    if not guard.acquire():
        sys.stderr.write("VoiceOS is already running.\n")
        return 0
    boot_mark("main: instance lock acquired")

    try:
        return _run()
    finally:
        guard.release()


def _run() -> int:
    """Initialize and run the application (caller holds the lock)."""
    log_dir = logs_dir()
    setup_logging(log_dir)
    logger.info("Single-instance lock acquired")
    logger.info("Starting VoiceOS")

    app_core = VoiceOSApplication()
    try:
        app_core.initialize()
    except DatabaseError:
        logger.exception("Startup failed during initialization")
        return 1
    boot_mark("main: core initialized (config+db)")

    # 3-5. Startup registration reconcile + Qt application.
    startup_manager = StartupManager()
    reconcile_auto_start(app_core, startup_manager)

    qt_app = QApplication.instance() or QApplication(sys.argv)
    qt_app.setApplicationName(app_core.config.app_name)
    qt_app.setApplicationVersion(app_core.config.version)
    qt_app.setOrganizationName(app_core.config.app_name)
    qt_app.setQuitOnLastWindowClosed(False)
    boot_mark("main: QApplication created")

    # 6-8. Tray + main window.
    tray = VoiceOSTray(qt_app)
    tray_available = QSystemTrayIcon.isSystemTrayAvailable()
    if not tray_available:
        logger.warning(
            "System tray is not available; running with main window only"
        )

    # Voice pipeline: mic controller + Qt signal bridge (GUI thread).
    voice_signals = VoiceSignals(qt_app)
    mic_controller = MicrophoneController(
        settings_get=app_core.settings_repository.get,
        settings_set=app_core.settings_repository.set,
        threshold=float(app_core.config.get("voice.wake_word_threshold", 0.25)),
    )
    mic_controller.set_callbacks(
        on_wake_word=voice_signals.wakeWordDetected.emit,
        on_status=voice_signals.micStatusChanged.emit,
        on_error=voice_signals.micError.emit,
        on_stats=voice_signals.audioStats.emit,
    )
    if sys.platform == "win32":
        # Trusted OS suspend/resume notifications; the controller's
        # clock-drift check remains as the validated fallback.
        power_filter = _PowerBroadcastFilter(mic_controller)
        qt_app.installNativeEventFilter(power_filter)

    # STT engine + utterance recognition service (worker thread).
    stt_enabled = bool(app_core.config.get("voice.stt_enabled", True))
    recognizer = create_recognizer(
        model_dir=(
            app_core.config.get("voice.stt_model_dir") or None
        ),
        model_type=str(app_core.config.get("voice.stt_engine", "auto")),
        num_threads=int(app_core.config.get("voice.stt_num_threads", 2)),
    ) if stt_enabled else None
    if recognizer is None:
        from src.voice.stt_engine import NullSpeechRecognizer

        recognizer = NullSpeechRecognizer()
    stt_service = SpeechRecognitionService(
        capture=mic_controller.capture,
        recognizer=recognizer,
        speech_threshold=float(
            app_core.config.get("voice.stt_speech_threshold", 0.01)
        ),
    )
    # On confirmed OS resume the mic controller flushes stale audio;
    # the STT worker additionally drops partial utterances and
    # recalibrates its VAD noise floor on fresh audio only.
    mic_controller.set_resume_listener(stt_service.notify_system_resume)

    # TTS engine + queued speech output (worker thread). The capture
    # service acts as the audio gate so Jarvis never hears itself.
    tts_enabled = bool(app_core.config.get("voice.tts_enabled", True))
    synthesizer = create_synthesizer(
        model_dir=(
            app_core.config.get("voice.tts_model_dir") or None
        ),
        num_threads=int(app_core.config.get("voice.tts_num_threads", 2)),
        speed=float(app_core.config.get("voice.tts_speed", 1.0)),
    ) if tts_enabled else None
    if synthesizer is None:
        from src.voice.tts_engine import NullSpeechSynthesizer

        synthesizer = NullSpeechSynthesizer()
    tts_service = SpeechOutputService(
        synthesizer=synthesizer,
        gate=mic_controller.capture,
    )
    tts_service.start()
    # V10.4: warm the TTS runtime once at startup so the wake
    # acknowledgement doesn't pay cold-start synthesis cost. The audio
    # is discarded; this is a single short synthesis, not a stream.
    try:
        prewarm_s = tts_service.prewarm("Ready.")
        logger.info("TTS prewarm completed in %.3fs", prewarm_s)
    except Exception:  # noqa: BLE001
        logger.debug("TTS prewarm failed", exc_info=True)

    # Windows automation services + allowlisted app registry (Task 6).
    from src.commands.app_handlers import _Svc, create_app_registry
    from src.controllers.application_discovery import ApplicationDiscovery
    from src.controllers.windows_automation import (
        ApplicationLauncher,
        ProcessMonitor,
        WindowManager,
    )

    from src.controllers.folder_opener import FolderOpener
    from src.controllers.file_system_discovery import (
        create_file_system_service,
    )
    from src.controllers.monitor_manager import MonitorManager
    from src.controllers.network_status_controller import (
        NetworkStatusController,
    )
    from src.controllers.notification_controller import (
        NotificationController,
    )
    from src.controllers.power_manager import PowerManager
    from src.controllers.desktop_control import DesktopController
    from src.controllers.power_status_controller import (
        PowerStatusController,
    )
    from src.controllers.screenshot_controller import ScreenshotController
    from src.controllers.volume_controller import VolumeController
    from src.controllers.windows_automation import WorkstationLocker

    app_discovery = ApplicationDiscovery()
    app_monitor = ProcessMonitor()
    app_windows = WindowManager(monitor=app_monitor)
    app_launcher = ApplicationLauncher(
        discovery=app_discovery, monitor=app_monitor
    )
    monitor_manager = MonitorManager()

    # Runtime identity diagnostics: proves which code/modules this
    # process loaded and how the logical monitor map resolves, so a
    # stale or mismatched instance is visible in the logs.
    try:
        import os as _os
        import sys as _sys
        import importlib as _il

        _mods = {}
        for _mn in (
            "src.commands.app_handlers",
            "src.commands.monitor_handlers",
            "src.controllers.monitor_manager",
            "src.controllers.windows_automation",
            "src.controllers.app_identity",
            "src.controllers.application_discovery",
        ):
            try:
                _mods[_mn] = getattr(_il.import_module(_mn), "__file__",
                                     "?")
            except Exception:  # noqa: BLE001
                _mods[_mn] = "import-failed"
        logger.info(
            "VoiceOS runtime diagnostics: pid=%s exe=%s root=%s "
            "version=%s",
            _os.getpid(), _sys.executable,
            _os.getcwd(),
            app_core.config.get("application.version", "?"),
        )
        for _mn, _path in _mods.items():
            logger.info("  module %s -> %s", _mn, _path)
        _mons = monitor_manager.monitors()
        for _m in _mons:
            logger.info(
                "  monitor map: logical %d = %s bounds=%s work=%s "
                "primary=%s",
                _m.index, _m.device, _m.bounds, _m.work_area,
                _m.primary,
            )
        logger.info(
            "  default launch destination: Monitor 1 (primary); "
            "no persisted monitor override configured"
        )
    except Exception:  # noqa: BLE001 - diagnostics are best-effort
        logger.debug("Runtime diagnostics failed", exc_info=True)

    # V10.1 local file/folder discovery: persistent SQLite index +
    # background refresh worker (daemon thread; never blocks GUI/mic).
    fs_service, fs_indexer = create_file_system_service(
        index_db_path=file_index_db_path(),
        migrate_from=PROJECT_ROOT / "data",
        known_folder_opener=FolderOpener(),
        start_background=True,
    )

    app_registry = create_app_registry(
        _Svc(
            app_discovery, app_launcher, app_monitor, app_windows,
            monitors=monitor_manager,
            folders=FolderOpener(),
            fs=fs_service,
            volume=VolumeController(),
            locker=WorkstationLocker(),
            power=PowerStatusController(),
            power_ops=PowerManager(),
            desktop=DesktopController(windows=app_windows),
            network=NetworkStatusController(),
            screenshot=ScreenshotController(),
            notifier=NotificationController(),
        )
    )

    # V10.4 misheard-command recovery: candidates are the applications
    # that currently own windows (same evidence LIST_RUNNING uses), so
    # fuzzy reinterpretation only ever targets apps that can actually
    # be acted on.
    from src.commands.recovery import MisheardCommandRecovery
    from src.controllers import app_identity as _app_identity

    def _running_app_candidates() -> list:
        try:
            apps = list(app_discovery.discover())
        except Exception:  # noqa: BLE001
            return []
        try:
            identities = _app_identity.identities_for(apps)
            try:
                wins = app_windows.list_windows(include_hidden=True)
            except TypeError:  # pragma: no cover - older window managers
                wins = app_windows.list_windows()
        except Exception:  # noqa: BLE001
            return []
        found = {}
        for w in wins:
            app = _app_identity.app_for_window(w, apps, identities)
            if app is not None:
                found[app.exe_path or app.name] = app
        return list(found.values())

    command_recovery = MisheardCommandRecovery(
        app_names=_running_app_candidates,
        app_resolver=app_discovery.resolve,
        monitor_resolver=monitor_manager.resolve,
    )

    # Dynamic discovery lifecycle: registry-change invalidation +
    # periodic background refresh (daemon threads, never block GUI/mic).
    app_discovery.start_registry_watcher()
    app_discovery.start_background_refresh(interval=600.0)

    # Conversation/session controller (Qt-free core logic).
    session_controller = VoiceSessionController(
        state_manager=app_core.state_manager,
        session_manager=app_core.session_manager,
        speech_recognition=stt_service,
        speech_output=tts_service,
        microphone_controller=mic_controller,
        activation_response=str(
            app_core.config.get("voice.activation_response", "At your service.")
        ),
        exit_phrase=str(app_core.config.get("voice.exit_phrase", "Goodbye Jarvis")),
        wake_phrase=str(app_core.config.get("voice.wake_word", "Hello Jarvis")),
        history_recorder=app_core.history_repository.record,
        command_registry=app_registry,
        command_recovery=command_recovery,
        ack_mode=str(app_core.config.get("voice.ack_mode", "full")),
        quick_ack_response=str(
            app_core.config.get("voice.quick_ack_response", "Yes.")
        ),
        on_transcript=voice_signals.transcriptReceived.emit,
        on_response=voice_signals.jarvisResponse.emit,
        on_activity=voice_signals.assistantActivity.emit,
        on_session_state=voice_signals.sessionStateChanged.emit,
        on_no_speech=voice_signals.noSpeech.emit,
    )

    window = MainWindow(
        app_core,
        close_to_tray=tray_available,
        microphone_controller=mic_controller,
    )
    window.set_engine_status(
        f"{stt_service.engine_name} · {getattr(recognizer, 'model_dir', None) or 'default'}",
        f"{tts_service.engine_name} · {getattr(synthesizer, 'model_dir', None) or 'default'}",
    )
    try:
        window.set_monitor_summary(monitor_manager.summary())
    except Exception:  # noqa: BLE001 - display info is best-effort
        logger.debug("Monitor summary unavailable", exc_info=True)

    def _on_wake_word(keyword: str) -> None:
        started = session_controller.handle_wake_word(keyword)
        window.notify_wake_word(keyword)
        logger.info(
            "Wake-word activation: keyword=%r session_started=%s state=%s",
            keyword,
            started,
            app_core.state_manager.state.name,
        )

    voice_signals.wakeWordDetected.connect(_on_wake_word)
    voice_signals.micStatusChanged.connect(
        lambda mic, detail: window.update_audio_status(
            f"{mic}: {detail}", _wake_label(mic_controller)
        )
    )
    voice_signals.micError.connect(
        lambda msg: logger.warning("Microphone error: %s", msg)
    )
    voice_signals.audioStats.connect(window.update_audio_stats)
    voice_signals.transcriptReceived.connect(window.update_transcript)
    voice_signals.noSpeech.connect(window.notify_no_speech)
    voice_signals.jarvisResponse.connect(window.update_jarvis_response)
    voice_signals.assistantActivity.connect(window.set_activity)
    voice_signals.sessionCountdown.connect(window.set_session_countdown)
    voice_signals.sessionStateChanged.connect(
        lambda active: window.set_session_countdown(-1 if not active else 0)
    )

    # V10.7 holographic HUD — presentation layer only. Fully isolated:
    # any failure here leaves the voice pipeline untouched.
    hud_controller = None
    try:
        from src.ui.hologram.hud_controller import HudController
        from src.ui.hologram.hud_probe import DesktopProbe

        # Real measured TTS amplitude -> HUD orb (queued to GUI thread).
        tts_service.set_level_callback(
            lambda levels, dur: voice_signals.ttsEnvelope.emit(levels, dur)
        )
        hud_controller = HudController(
            settings_get=app_core.settings_repository.get,
            settings_set=app_core.settings_repository.set,
            monitor_manager=monitor_manager,
            config_get=app_core.config.get,
        )
        hud_controller.start(
            voice_signals=voice_signals,
            state_manager=app_core.state_manager,
            session_controller=session_controller,
            microphone_controller=mic_controller,
            stt_service=stt_service,
            tts_service=tts_service,
            desktop_probe=DesktopProbe(
                window_manager=app_windows,
                monitor_manager=monitor_manager,
                state_manager=app_core.state_manager,
            ),
        )
    except Exception:  # noqa: BLE001
        logger.exception("HUD startup failed; continuing without HUD")
        hud_controller = None

    # Voice-session housekeeping: inactivity timeout + countdown.
    def _session_tick() -> None:
        remaining = session_controller.tick()
        voice_signals.sessionCountdown.emit(
            -1 if remaining is None else remaining
        )

    session_timer = QTimer(qt_app)
    session_timer.setInterval(1000)
    session_timer.timeout.connect(_session_tick)
    session_timer.start()

    wire_tray(
        tray,
        window,
        app_core,
        startup_manager,
        app_core.settings_repository,
        microphone_controller=mic_controller,
        hud_controller=hud_controller,
    )

    # Start listening if enabled by configuration.
    if (
        tray_available
        and app_core.config.get("voice.listening_enabled", True)
        and app_core.config.get("voice.microphone_enabled", True)
        and app_core.config.get("voice.wake_word_enabled", True)
    ):
        mic_controller.enable()
        window.update_audio_status(
            f"{mic_controller.status}: {mic_controller.status_detail}",
            _wake_label(mic_controller),
        )

    if tray_available:
        tray.show()
        tray.refresh_auto_start(startup_manager.is_registered())
        tray.diagnose()
        boot_mark("main: tray visible")
        logger.info("System tray initialized")

    # 9. Respect start_minimized (only meaningful when the tray exists).
    start_minimized = bool(app_core.config.get("startup.start_minimized", True))
    if not (tray_available and start_minimized):
        window.show_window()
    else:
        logger.info("Starting minimized to system tray")
        announce_tray_startup(tray, app_core)

    boot_mark("main: entering event loop")
    logger.info("Entering Qt event loop")
    exit_code = qt_app.exec()

    # 10. Clean shutdown (idempotent if tray Exit already ran it).
    if hud_controller is not None:
        try:
            hud_controller.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("HUD shutdown failed", exc_info=True)
    session_controller.shutdown()
    stt_service.shutdown()
    tts_service.shutdown()
    mic_controller.shutdown()
    app_core.shutdown()
    logger.info("VoiceOS exited with code %s", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
