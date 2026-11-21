"""VoiceOS installation verifier.

Checks that the current Python environment can run VoiceOS: interpreter
version, dependencies, configuration, runtime paths, offline speech
models, microphone presence and Qt/OpenGL availability.

Read-only with respect to user settings: it never starts audio capture,
never modifies settings, and only creates the runtime directories it
reports on. Model binaries are validated by file presence only — they
are never loaded.

Usage (from the repository root, using the project venv)::

    .venv\\Scripts\\python.exe scripts\\verify_installation.py

Exit code is 0 when there are no FAIL results, 1 otherwise.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_PYTHON = (3, 12)

PASS, WARN, FAIL = "PASS", "WARNING", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(status: str, area: str, detail: str) -> None:
    _results.append((status, area, detail))
    print(f"[{status:<7}] {area}: {detail}")


def check_python() -> None:
    v = sys.version_info
    ok = v >= MIN_PYTHON
    record(
        PASS if ok else FAIL,
        "Python",
        f"{v.major}.{v.minor}.{v.micro} (requires >={MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
    )


def check_dependencies() -> None:
    required = {
        "PySide6": "PySide6",
        "dotenv": "python-dotenv",
        "numpy": "numpy",
        "sounddevice": "sounddevice",
        "sherpa_onnx": "sherpa-onnx",
        "pycaw": "pycaw",
    }
    for module, dist in required.items():
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "Dependencies", f"{dist} ({module}) not importable: {exc}")
        else:
            record(PASS, "Dependencies", f"{dist} importable")
    try:
        import pytest  # noqa: F401

        record(PASS, "Dependencies", "pytest importable (dev)")
    except Exception:  # noqa: BLE001
        record(WARN, "Dependencies", "pytest not installed (needed for tests only)")


def check_configuration() -> None:
    try:
        from src.core.configuration import ConfigurationManager

        cfg = ConfigurationManager()
        record(PASS, "Configuration", f"settings loaded; db={cfg.database_path}")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Configuration", f"settings failed to load: {exc}")
        return
    try:
        from src.core.paths import runtime_dir

        record(PASS, "Paths", f"runtime dir: {runtime_dir()}")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Paths", f"runtime dir resolution failed: {exc}")


def check_database() -> None:
    try:
        from src.core.paths import cache_dir

        probe = cache_dir() / "_verify_probe.db"
        try:
            conn = sqlite3.connect(str(probe))
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            conn.close()
            probe.unlink(missing_ok=True)
        finally:
            probe.unlink(missing_ok=True)
        record(PASS, "Database", "SQLite read/write probe succeeded")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Database", f"SQLite probe failed: {exc}")


def _find(base: Path, pattern: str) -> list[Path]:
    try:
        return sorted(base.rglob(pattern))
    except OSError:
        return []


def check_models() -> None:
    manifest_path = PROJECT_ROOT / "config" / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Models", f"cannot read {manifest_path}: {exc}")
        return

    from src.core import paths

    dir_resolvers = {
        "kws": paths.kws_model_dir,
        "stt": paths.stt_model_dir,
        "tts": paths.tts_model_dir,
    }
    env_names = {"kws": "VOICEOS_KWS_MODEL_DIR",
                 "stt": "VOICEOS_STT_MODEL_DIR",
                 "tts": "VOICEOS_TTS_MODEL_DIR"}

    for entry in manifest.get("models", []):
        family = entry.get("family", "?")
        model_id = entry.get("id", "?")
        base = Path(dir_resolvers.get(family, paths.runtime_dir)())
        missing = []
        for req in entry.get("required", []):
            pattern = req.get("glob", "")
            if not _find(base, pattern):
                missing.append(pattern)
        if missing:
            env = env_names.get(family, "")
            record(
                FAIL,
                "Models",
                f"{model_id}: missing under {base} "
                f"(globs: {', '.join(missing)}); "
                f"set {env} or see docs/MODEL_SETUP.md",
            )
        else:
            record(PASS, "Models", f"{model_id}: all required files present in {base}")
        # A present-but-empty model dir is a common mistake.
        try:
            if base.is_dir() and not any(base.iterdir()):
                record(WARN, "Models", f"{model_id}: model dir is empty: {base}")
        except OSError:
            pass


def check_microphone() -> None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if inputs:
            default = sd.default.device
            name = (
                devices[default[0]].get("name", "?")
                if isinstance(default, (list, tuple)) and default[0] is not None
                else "?"
            )
            record(PASS, "Microphone", f"{len(inputs)} input device(s); default={name}")
        else:
            record(FAIL, "Microphone", "no input-capable audio devices found")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Microphone", f"device query failed: {exc}")


def check_qt_opengl() -> None:
    try:
        import PySide6.QtCore  # noqa: F401
        import PySide6.QtWidgets  # noqa: F401
        import PySide6.QtOpenGLWidgets  # noqa: F401

        record(PASS, "Qt/OpenGL", "PySide6 widgets + OpenGL widgets importable")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Qt/OpenGL", f"PySide6/OpenGL import failed: {exc}")


def main() -> int:
    print("VoiceOS installation verification")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Interpreter:   {sys.executable}")
    print("-" * 60)

    check_python()
    check_dependencies()
    check_configuration()
    check_database()
    check_models()
    check_microphone()
    check_qt_opengl()

    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for status, _, _ in _results:
        counts[status] += 1
    print("-" * 60)
    print(
        f"Summary: {counts[PASS]} passed, {counts[WARN]} warnings, "
        f"{counts[FAIL]} failed"
    )
    if counts[FAIL]:
        print("Result: FAIL — see items marked [FAIL] above.")
        return 1
    if counts[WARN]:
        print("Result: PASS with warnings.")
    else:
        print("Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
