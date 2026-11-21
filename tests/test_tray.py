"""Tests for the system tray, window hide/restore and exit behaviour.

These run with Qt's offscreen platform so no real tray or window is
created.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QSystemTrayIcon  # noqa: E402

from src.core.application import VoiceOSApplication  # noqa: E402
from src.core.configuration import ConfigurationManager  # noqa: E402
from src.core.state_manager import AppState  # noqa: E402
from src.ui.main_window import MainWindow  # noqa: E402
from src.ui.tray_icon import VoiceOSTray, _make_tray_icon, wire_tray  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """Provide a single offscreen QApplication for the module."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def app_core(tmp_path) -> VoiceOSApplication:
    """Provide an initialized application core with a temp database."""
    app = VoiceOSApplication(database_path=tmp_path / "app.db")
    app.initialize()
    return app


@pytest.fixture
def window(app_core, qapp) -> MainWindow:
    win = MainWindow(app_core, close_to_tray=True)
    yield win
    win.deleteLater()


@pytest.fixture
def tray(qapp) -> VoiceOSTray:
    t = VoiceOSTray()
    yield t
    t.hide()


class FakeStartupManager:
    """In-memory stand-in for StartupManager."""

    def __init__(self) -> None:
        self.enabled = False
        self.set_enabled_calls: list[bool] = []

    def set_enabled(self, enabled: bool) -> bool:
        self.set_enabled_calls.append(enabled)
        changed = self.enabled != enabled
        self.enabled = enabled
        return changed

    def is_registered(self) -> bool:
        return self.enabled

    def register(self) -> bool:
        return self.set_enabled(True)

    def unregister(self) -> bool:
        return self.set_enabled(False)


class TestTrayMenu:
    """Tests for tray icon menu structure and signals."""

    def test_menu_actions_exist(self, tray: VoiceOSTray) -> None:
        assert tray.contextMenu() is not None
        texts = [a.text() for a in tray.contextMenu().actions()]
        assert any("Open VoiceOS" in t for t in texts)
        assert any("Hide VoiceOS" in t for t in texts)
        assert any("automatic startup" in t for t in texts)
        assert any("Exit Jarvis" in t for t in texts)

    def test_tooltip(self, tray: VoiceOSTray) -> None:
        assert "Jarvis" in tray.toolTip()

    def test_double_click_emits_open(self, tray: VoiceOSTray) -> None:
        received = []
        tray.openRequested.connect(lambda: received.append(True))
        tray.activated.emit(QSystemTrayIcon.ActivationReason.DoubleClick)
        assert received == [True]

    def test_single_click_does_not_emit_open(self, tray: VoiceOSTray) -> None:
        received = []
        tray.openRequested.connect(lambda: received.append(True))
        tray.activated.emit(QSystemTrayIcon.ActivationReason.Trigger)
        assert received == []

    def test_refresh_auto_start_text(self, tray: VoiceOSTray) -> None:
        tray.refresh_auto_start(True)
        assert tray.auto_start_action.isChecked() is True
        assert tray.auto_start_action.text() == "Disable automatic startup"
        tray.refresh_auto_start(False)
        assert tray.auto_start_action.isChecked() is False
        assert tray.auto_start_action.text() == "Enable automatic startup"


class TestTrayIconValidity:
    """Tests for the generated tray icon and diagnostics."""

    def test_icon_is_not_null(self) -> None:
        icon = _make_tray_icon()
        assert icon.isNull() is False

    def test_icon_has_visible_pixels(self) -> None:
        pm = _make_tray_icon().pixmap(64, 64)
        assert pm.isNull() is False
        img = pm.toImage()
        painted = sum(
            1
            for y in range(img.height())
            for x in range(img.width())
            if img.pixelColor(x, y).alpha() > 0
        )
        # A meaningful fraction of the icon must be non-transparent.
        assert painted > pm.width() * pm.height() * 0.2

    def test_diagnose_returns_bool(self, tray: VoiceOSTray) -> None:
        assert isinstance(tray.diagnose(), bool)


class TestTrayStartupAnnouncement:
    """Tests for the minimized-startup tray notification."""

    class _FakeTray:
        def __init__(self) -> None:
            self.messages = []

        def showMessage(self, title, message, icon, timeout):
            self.messages.append((title, message, icon, timeout))

    def _app(self, tmp_path, notifications: bool) -> VoiceOSApplication:
        import json

        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(
            json.dumps({"interface": {"show_notifications": notifications}}),
            encoding="utf-8",
        )
        config = ConfigurationManager(config_path=cfg_file)
        return VoiceOSApplication(config=config, database_path=tmp_path / "a.db")

    def test_announcement_shown_by_default(self, tmp_path) -> None:
        from src.main import announce_tray_startup

        app = self._app(tmp_path, notifications=True)
        tray = self._FakeTray()
        announce_tray_startup(tray, app)
        assert len(tray.messages) == 1
        title, message, _icon, _timeout = tray.messages[0]
        assert title == "VoiceOS"
        assert "Jarvis" in message

    def test_announcement_respects_setting(self, tmp_path) -> None:
        from src.main import announce_tray_startup

        app = self._app(tmp_path, notifications=False)
        tray = self._FakeTray()
        announce_tray_startup(tray, app)
        assert tray.messages == []

    def test_announcement_failure_is_swallowed(self, tmp_path) -> None:
        from src.main import announce_tray_startup

        class _BoomTray:
            def showMessage(self, *args):
                raise RuntimeError("no balloon support")

        app = self._app(tmp_path, notifications=True)
        announce_tray_startup(_BoomTray(), app)  # must not raise


class TestWindowHideRestore:
    """Tests for hide-to-tray and restore behaviour."""

    def test_close_hides_window_and_keeps_state(
        self, window: MainWindow, app_core: VoiceOSApplication
    ) -> None:
        window.show_window()
        assert window.isVisible() is True
        window.close()
        assert window.isVisible() is False
        # Hiding must NOT stop the application.
        assert app_core.state_manager.state is AppState.STANDBY

    def test_show_window_restores(self, window: MainWindow) -> None:
        window.show_window()
        window.hide_window()
        assert window.isVisible() is False
        window.show_window()
        assert window.isVisible() is True

    def test_no_duplicate_windows(
        self, app_core: VoiceOSApplication, qapp
    ) -> None:
        win1 = MainWindow(app_core, close_to_tray=True)
        win2 = MainWindow(app_core, close_to_tray=True)
        win1.show_window()
        win1.show_window()  # idempotent
        assert win1.isVisible() is True
        win1.hide_window()
        win2.show_window()
        assert win2.isVisible() is True
        win1.deleteLater()
        win2.deleteLater()

    def test_close_without_tray_shuts_down(
        self, app_core: VoiceOSApplication, qapp
    ) -> None:
        win = MainWindow(app_core, close_to_tray=False)
        win.show()
        win.close()
        assert app_core.state_manager.state is AppState.STOPPED


class TestTrayWiring:
    """Tests for tray action wiring."""

    def _wire(self, window, app_core, startup):
        tray = VoiceOSTray()
        wire_tray(
            tray,
            window,
            app_core,
            startup,
            app_core.settings_repository,
        )
        return tray

    def test_open_and_hide_actions(
        self, window: MainWindow, app_core: VoiceOSApplication
    ) -> None:
        startup = FakeStartupManager()
        tray = self._wire(window, app_core, startup)
        window.hide_window()
        tray.open_action.trigger()
        assert window.isVisible() is True
        tray.hide_action.trigger()
        assert window.isVisible() is False
        tray.hide()

    def test_exit_action_shuts_down(
        self, window: MainWindow, app_core: VoiceOSApplication
    ) -> None:
        startup = FakeStartupManager()
        tray = self._wire(window, app_core, startup)
        tray.exit_action.trigger()
        assert app_core.state_manager.state is AppState.STOPPED
        tray.hide()

    def test_auto_start_toggle_registers_and_persists(
        self, window: MainWindow, app_core: VoiceOSApplication
    ) -> None:
        startup = FakeStartupManager()
        tray = self._wire(window, app_core, startup)

        tray.autoStartToggled.emit(True)
        assert startup.enabled is True
        assert startup.set_enabled_calls == [True]
        assert app_core.settings_repository.get("startup.auto_start") == "true"

        tray.refresh_auto_start(startup.is_registered())
        tray.autoStartToggled.emit(False)
        assert startup.enabled is False
        assert app_core.settings_repository.get("startup.auto_start") == "false"
        tray.hide()

    def test_auto_start_failure_reverts_checkbox(
        self, window: MainWindow, app_core: VoiceOSApplication
    ) -> None:
        from src.core.startup_manager import StartupError

        class FailingStartup(FakeStartupManager):
            def set_enabled(self, enabled: bool) -> bool:
                raise StartupError("no access")

        startup = FailingStartup()
        tray = self._wire(window, app_core, startup)
        tray.auto_start_action.setChecked(True)
        tray.autoStartToggled.emit(True)
        # Checkbox must be reverted and not persisted.
        assert tray.auto_start_action.isChecked() is False
        assert app_core.settings_repository.get("startup.auto_start") is None
        tray.hide()
