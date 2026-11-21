"""System tray icon and menu for VoiceOS.

Provides a ``QSystemTrayIcon`` with an Open / Hide / Automatic-startup /
Exit menu, plus a ``wire_tray`` helper that connects tray actions to the
application, window and startup manager.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from src.core.application import VoiceOSApplication
from src.core.startup_manager import StartupError, StartupManager
from src.storage.settings_repository import SettingsRepository
from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def _make_tray_icon() -> QIcon:
    """Generate a simple 'J' tray icon programmatically (no asset files)."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Rounded background with a light border so it stands out on both
    # light and dark taskbars.
    path = QPainterPath()
    path.addRoundedRect(4, 4, size - 8, size - 8, 14, 14)
    painter.fillPath(path, QColor("#1e1e2e"))
    painter.setPen(QPen(QColor("#cba6f7"), 2.5))
    painter.drawPath(path)

    # Stylized "J": vertical stroke with a bottom hook curving left.
    pen = QPen(QColor("#cba6f7"), 5, Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    j = QPainterPath()
    j.moveTo(37, 13)
    j.lineTo(37, 35)
    j.quadTo(37, 53, 19, 51)
    j.lineTo(19, 45)
    painter.drawPath(j)
    painter.end()

    return QIcon(pixmap)


class VoiceOSTray(QSystemTrayIcon):
    """VoiceOS system tray icon with application menu.

    Signals:
        openRequested: User asked to show the main window.
        hideRequested: User asked to hide the main window.
        exitRequested: User asked to exit the application.
        autoStartToggled: Automatic-startup checkbox changed (bool = enabled).
    """

    openRequested = Signal()
    hideRequested = Signal()
    exitRequested = Signal()
    autoStartToggled = Signal(bool)
    micToggled = Signal(bool)
    hudToggled = Signal(bool)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Create the tray icon and its context menu."""
        super().__init__(_make_tray_icon(), parent)
        self.setToolTip("VoiceOS — Jarvis")

        menu = QMenu()
        self.open_action = QAction("Open VoiceOS", menu)
        self.hide_action = QAction("Hide VoiceOS", menu)
        self.auto_start_action = QAction("Enable automatic startup", menu)
        self.auto_start_action.setCheckable(True)
        self.mic_action = QAction("Microphone listening", menu)
        self.mic_action.setCheckable(True)
        self.hud_action = QAction("Show holographic HUD", menu)
        self.hud_action.setCheckable(True)
        self.hud_action.setChecked(True)
        # V10.8 compact-overlay controls.
        self.hud_lock_action = QAction("Lock HUD position", menu)
        self.hud_lock_action.setCheckable(True)
        self.hud_lock_action.setChecked(True)
        self.hud_reset_action = QAction("Reset HUD position", menu)
        self.hud_size_menu = QMenu("HUD size", menu)
        self.hud_size_group = QActionGroup(self.hud_size_menu)
        self.hud_size_group.setExclusive(True)
        self.hud_size_actions: dict[int, QAction] = {}
        for label, side in (("Small (240)", 240),
                            ("Medium (320)", 320),
                            ("Large (420)", 420)):
            act = QAction(label, self.hud_size_menu)
            act.setCheckable(True)
            act.setData(side)
            self.hud_size_menu.addAction(act)
            self.hud_size_group.addAction(act)
            self.hud_size_actions[side] = act
        # V10.8 appearance controls: preset submenu + slider dialog.
        self.hud_appearance_menu = QMenu("HUD appearance", menu)
        self.hud_appearance_group = QActionGroup(self.hud_appearance_menu)
        self.hud_appearance_group.setExclusive(True)
        self.hud_preset_actions: dict[str, QAction] = {}
        for label, preset in (("Subtle", "subtle"),
                              ("Balanced", "balanced"),
                              ("Intense", "intense")):
            act = QAction(label, self.hud_appearance_menu)
            act.setCheckable(True)
            act.setData(preset)
            self.hud_appearance_menu.addAction(act)
            self.hud_appearance_group.addAction(act)
            self.hud_preset_actions[preset] = act
        self.hud_appearance_menu.addSeparator()
        self.hud_customize_action = QAction(
            "Customize appearance…", self.hud_appearance_menu)
        self.hud_appearance_dialog = None
        self.exit_action = QAction("Exit Jarvis", menu)

        menu.addAction(self.open_action)
        menu.addAction(self.hide_action)
        menu.addSeparator()
        menu.addAction(self.auto_start_action)
        menu.addSeparator()
        menu.addAction(self.mic_action)
        menu.addAction(self.hud_action)
        menu.addAction(self.hud_lock_action)
        menu.addAction(self.hud_reset_action)
        menu.addMenu(self.hud_size_menu)
        menu.addMenu(self.hud_appearance_menu)
        menu.addSeparator()
        menu.addAction(self.exit_action)
        self.setContextMenu(menu)

        self.open_action.triggered.connect(self.openRequested)
        self.hide_action.triggered.connect(self.hideRequested)
        self.auto_start_action.triggered.connect(self.autoStartToggled)
        self.mic_action.triggered.connect(self.micToggled)
        self.hud_action.triggered.connect(self.hudToggled)
        self.exit_action.triggered.connect(self.exitRequested)
        self.activated.connect(self._on_activated)

        logger.debug("Tray icon and menu initialized")

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Open the main window on tray icon double-click."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.openRequested.emit()

    def refresh_auto_start(self, registered: bool) -> None:
        """Update the automatic-startup action to reflect reality."""
        self.auto_start_action.setChecked(registered)
        self.auto_start_action.setText(
            "Disable automatic startup" if registered else "Enable automatic startup"
        )

    def diagnose(self) -> bool:
        """Log tray/icon diagnostics and return whether the icon is shown.

        Returns:
            True if the tray is available and the icon reports visible.
        """
        available = QSystemTrayIcon.isSystemTrayAvailable()
        icon_null = self.icon().isNull()
        visible = self.isVisible()
        logger.info(
            "Tray diagnostics: available=%s icon_null=%s visible=%s geometry=%s",
            available,
            icon_null,
            visible,
            self.geometry(),
        )
        if not available:
            logger.warning("System tray reports unavailable")
        elif icon_null:
            logger.error("Tray icon is null; nothing will be displayed")
        elif not visible:
            logger.warning(
                "Tray icon is not visible after show(); it may be located "
                "in the Windows hidden-icons overflow area"
            )
        return bool(available and not icon_null and visible)


def wire_tray(
    tray: VoiceOSTray,
    window: MainWindow,
    app_core: VoiceOSApplication,
    startup_manager: StartupManager,
    settings_repository: SettingsRepository,
    microphone_controller=None,
    hud_controller=None,
) -> None:
    """Connect tray signals to application behaviour.

    Args:
        tray: The system tray icon.
        window: The main window.
        app_core: The core application facade.
        startup_manager: Startup registration manager.
        settings_repository: Repository used to persist the user's
            automatic-startup preference.
    """

    def on_open() -> None:
        window.show_window()
        logger.info("Main window shown from tray")

    def on_hide() -> None:
        window.hide_window()
        logger.info("Main window hidden from tray")

    def on_exit() -> None:
        logger.info("Exit requested from tray")
        app_core.shutdown()
        if QApplication.instance() is not None:
            QApplication.instance().quit()

    def on_auto_start(checked: bool) -> None:
        try:
            changed = startup_manager.set_enabled(checked)
        except StartupError as exc:
            logger.error("Startup registration failed: %s", exc)
            tray.auto_start_action.setChecked(not checked)
            tray.refresh_auto_start(startup_manager.is_registered())
            return
        settings_repository.set(
            "startup.auto_start", "true" if checked else "false"
        )
        tray.refresh_auto_start(startup_manager.is_registered())
        logger.info(
            "Automatic startup %s (changed=%s)",
            "enabled" if checked else "disabled",
            changed,
        )

    def on_mic(checked: bool) -> None:
        if microphone_controller is None:
            return
        if checked:
            ok = microphone_controller.enable()
            if not ok:
                tray.mic_action.setChecked(False)
        else:
            microphone_controller.disable()
        tray.mic_action.setChecked(microphone_controller.is_listening)

    tray.openRequested.connect(on_open)
    tray.hideRequested.connect(on_hide)
    tray.exitRequested.connect(on_exit)
    tray.autoStartToggled.connect(on_auto_start)
    tray.micToggled.connect(on_mic)

    if hud_controller is not None:
        def on_hud(checked: bool) -> None:
            if checked:
                hud_controller.show()
            else:
                hud_controller.hide()

        tray.hudToggled.connect(on_hud)
        tray.hud_action.setChecked(hud_controller.is_shown)

        # V10.8 overlay controls.
        def on_hud_lock(checked: bool) -> None:
            hud_controller.set_position_locked(checked)

        def on_hud_reset() -> None:
            hud_controller.reset_position()

        def on_hud_size(action: QAction) -> None:
            side = action.data()
            if side is not None:
                hud_controller.set_size(int(side))

        try:
            tray.hud_lock_action.setChecked(
                hud_controller.is_position_locked())
        except Exception:  # noqa: BLE001
            pass
        for act in tray.hud_size_group.actions():
            try:
                act.setChecked(int(act.data()) == hud_controller.hud_size())
            except Exception:  # noqa: BLE001
                act.setChecked(int(act.data()) == 320)
        tray.hud_lock_action.triggered.connect(on_hud_lock)
        tray.hud_reset_action.triggered.connect(on_hud_reset)
        tray.hud_size_group.triggered.connect(on_hud_size)

        # V10.8 appearance controls.
        appearance = getattr(hud_controller, "appearance", None)
        if appearance is not None:
            def on_hud_preset(action: QAction) -> None:
                appearance.set_preset(str(action.data()))

            def on_hud_customize() -> None:
                dlg = getattr(tray, "hud_appearance_dialog", None)
                if dlg is None:
                    from src.ui.appearance_dialog import (
                        AppearanceDialog)
                    dlg = AppearanceDialog(
                        appearance, parent=window)
                    tray.hud_appearance_dialog = dlg
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()

            def _sync_preset_checks() -> None:
                current = appearance.preset
                for name, act in tray.hud_preset_actions.items():
                    act.setChecked(current == name)

            tray.hud_appearance_group.triggered.connect(on_hud_preset)
            tray.hud_customize_action.triggered.connect(on_hud_customize)
            # Keep preset checks in sync when sliders mark "custom".
            prev_cb = appearance.on_change

            def _chain_and_sync():
                _sync_preset_checks()
                if prev_cb is not None:
                    prev_cb()

            appearance.on_change = _chain_and_sync
            _sync_preset_checks()
    else:
        # No HUD this run — disable its position controls.
        for act in (tray.hud_lock_action, tray.hud_reset_action):
            act.setEnabled(False)
        tray.hud_size_menu.setEnabled(False)
        tray.hud_appearance_menu.setEnabled(False)

    if microphone_controller is not None:
        tray.mic_action.setChecked(microphone_controller.is_listening)
