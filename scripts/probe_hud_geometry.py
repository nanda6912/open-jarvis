"""One-shot diagnostic: why does HudController._recheck_placement see
hud.geometry() != want every 5 s on a single-monitor desktop?

Creates a frameless translucent widget (same window flags as
HologramHUD), places it like the controller does, then prints
screen/widget geometry for ~10 s and exits. Read-only diagnostic;
closes itself.
"""

import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget


def main() -> int:
    app = QApplication(sys.argv)
    screens = QGuiApplication.screens()
    print("screens:", len(screens))
    for s in screens:
        g = s.geometry()
        ag = s.availableGeometry()
        print(
            f"  name={s.name()!r} model={s.model()!r} "
            f"geom=({g.x()},{g.y()},{g.width()},{g.height()}) "
            f"avail=({ag.x()},{ag.y()},{ag.width()},{ag.height()}) "
            f"dpr={s.devicePixelRatio():.3f}"
        )
    screen = screens[0]
    want = screen.geometry()

    w = QWidget()
    w.setWindowFlags(
        Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
    )
    w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    w.resize(10, 10)
    w.setGeometry(want)
    w.show()
    app.processEvents()

    state = {"n": 0}

    def tick():
        state["n"] += 1
        actual = w.geometry()
        print(
            f"[{state['n']}] want=({want.x()},{want.y()},"
            f"{want.width()},{want.height()}) "
            f"actual=({actual.x()},{actual.y()},"
            f"{actual.width()},{actual.height()}) "
            f"equal={actual == want} "
            f"screen={w.screen().name() if w.screen() else None!r}"
        )
        if state["n"] >= 4:
            app.quit()

    QTimer.singleShot(500, tick)
    QTimer.singleShot(2000, tick)
    QTimer.singleShot(4000, tick)
    QTimer.singleShot(6000, tick)
    # Also test setGeometry convergence explicitly.
    QTimer.singleShot(8000, lambda: (w.setGeometry(want), app.processEvents(),
                                    tick(), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
