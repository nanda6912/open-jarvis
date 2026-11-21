"""Small HUD appearance settings dialog (V10.8).

Launched from the tray menu — seven 0–100 % sliders plus preset
buttons, Reset to Balanced and Close. Every change is applied
immediately through :class:`HudAppearance` (which persists and
notifies the controller), so the hologram previews live without
restart.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.ui.hologram.hud_appearance import PARAMS

logger = logging.getLogger(__name__)

_LABELS = {
    "brightness": "Overall brightness",
    "glow": "Sphere glow",
    "opacity": "Overall opacity",
    "ring": "Ring brightness",
    "particle": "Particle brightness",
    "waveform": "Waveform intensity",
    "speed": "Animation speed",
}


class AppearanceDialog(QDialog):
    """Slider editor bound to a :class:`HudAppearance` instance."""

    def __init__(self, appearance, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._appearance = appearance
        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._preset_buttons: dict[str, QPushButton] = {}
        self._syncing = False

        self.setWindowTitle("HUD appearance")
        self.setModal(False)
        self.resize(380, 460)

        root = QVBoxLayout(self)

        # Preset row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        for name in ("subtle", "balanced", "intense"):
            btn = QPushButton(name.capitalize(), self)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c, n=name: self._on_preset(n))
            preset_row.addWidget(btn)
            self._preset_buttons[name] = btn
        preset_row.addStretch(1)
        reset_btn = QPushButton("Reset to Balanced", self)
        reset_btn.clicked.connect(self._on_reset)
        preset_row.addWidget(reset_btn)
        root.addLayout(preset_row)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for param in PARAMS:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setSingleStep(1)
            slider.setPageStep(5)
            value_label = QLabel("0%")
            value_label.setMinimumWidth(42)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                     | Qt.AlignmentFlag.AlignVCenter)
            row = QWidget(self)
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(8)
            row_lay.addWidget(slider, 1)
            row_lay.addWidget(value_label)
            slider.valueChanged.connect(
                lambda v, p=param: self._on_slider(p, v))
            form.addRow(_LABELS[param], row)
            self._sliders[param] = slider
            self._value_labels[param] = value_label
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        # Chain onto the controller's change callback so external
        # preset changes (tray menu) also refresh the sliders.
        prev_cb = getattr(appearance, "on_change", None)

        def _chain():
            self.refresh()
            if prev_cb is not None:
                prev_cb()

        appearance.on_change = _chain
        self.refresh()

    # ------------------------------------------------------------------
    def _on_preset(self, name: str) -> None:
        self._appearance.set_preset(name)
        self.refresh()

    def _on_reset(self) -> None:
        self._appearance.reset()
        self.refresh()

    def _on_slider(self, param: str, percent: int) -> None:
        if self._syncing:
            return
        self._appearance.set_value_percent(param, percent)
        self._value_labels[param].setText(f"{percent}%")
        self._refresh_preset_buttons()

    def _refresh_preset_buttons(self) -> None:
        current = self._appearance.preset
        for name, btn in self._preset_buttons.items():
            btn.setChecked(current == name)

    def refresh(self) -> None:
        """Re-sync controls from the appearance manager."""
        self._syncing = True
        try:
            for param, slider in self._sliders.items():
                pct = int(round(self._appearance.value(param) * 100))
                slider.setValue(pct)
                self._value_labels[param].setText(f"{pct}%")
            self._refresh_preset_buttons()
        finally:
            self._syncing = False

    def closeEvent(self, event) -> None:  # noqa: N802
        self.hide()
        event.accept()

    def reject(self) -> None:
        self.hide()


__all__ = ["AppearanceDialog"]
