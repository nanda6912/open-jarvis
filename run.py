"""VoiceOS launcher.

Run from the project root with::

    python run.py
"""

from src.core.boot_diag import mark

mark("run.py: script start")

from src.main import main  # noqa: E402

mark("run.py: src.main imported (incl. PySide6)")

if __name__ == "__main__":
    raise SystemExit(main())
