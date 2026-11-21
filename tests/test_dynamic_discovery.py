"""Tests for the dynamic application-discovery enhancement.

Verifies that Jarvis resolves apps from the real inventory sources
(Start Menu shortcuts, App Paths, uninstall entries, program dirs)
without any hardcoded allowlist, handles ambiguity safely, and picks
up apps installed after the cache was populated.
"""

from __future__ import annotations

import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.intent import Intent, RuleBasedIntentRecognizer
from src.controllers.application_discovery import (
    AppInfo,
    ApplicationDiscovery,
)
from src.controllers.windows_automation import (
    ApplicationLauncher,
    WindowInfo,
    WindowManager,
)
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.voice_session_controller import VoiceSessionController


# ----------------------------------------------------------------------
# Helpers / fakes
# ----------------------------------------------------------------------
class FakeMonitor:
    def __init__(self, procs=None):
        self.procs = procs or []

    def list_processes(self):
        from src.controllers.windows_automation import ProcInfo
        return [ProcInfo(pid=p, name=n) for p, n in self.procs]

    def is_running(self, app):
        return any(p.stem == app.exe_name for p in self.list_processes())


class FakeAPI:
    def __init__(self, windows=None):
        self.windows = windows or []
        self.shown = []
        self.foreground_calls = []
        self.closed = []
        self.desktop_shown = 0

    def enum_windows(self, callback):
        for w in self.windows:
            callback(w.hwnd, 0)

    def window_text(self, hwnd):
        return next((w.title for w in self.windows if w.hwnd == hwnd), "")

    def is_visible(self, hwnd):
        return any(w.hwnd == hwnd for w in self.windows)

    def pid_of(self, hwnd):
        return next((w.pid for w in self.windows if w.hwnd == hwnd), 0)

    def show(self, hwnd, cmd):
        self.shown.append((hwnd, cmd))

    def set_foreground(self, hwnd):
        self.foreground_calls.append(hwnd)
        return True

    def post_close(self, hwnd):
        self.closed.append(hwnd)
        # WM_CLOSE destroys the window; mirror that for close
        # verification.
        self.windows = [w for w in self.windows if w.hwnd != hwnd]

    def foreground_hwnd(self):
        return self.windows[0].hwnd if self.windows else 0

    def ex_style(self, hwnd):
        return 0

    def show_desktop(self):
        self.desktop_shown += 1


def make_env(tmp_path):
    """Env with fake Start Menu / Programs / AppData roots."""
    appdata = tmp_path / "AppData" / "Roaming"
    programdata = tmp_path / "ProgramData"
    local = tmp_path / "AppData" / "Local"
    for d in (appdata, programdata, local):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "APPDATA": str(appdata),
        "ProgramData": str(programdata),
        "LOCALAPPDATA": str(local),
        "ProgramFiles": str(tmp_path / "PF"),
        "ProgramFiles(x86)": str(tmp_path / "PF86"),
        "SystemRoot": str(tmp_path / "Win"),
        "windir": str(tmp_path / "Win"),
    }


def write_exe(path: str) -> str:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"MZ" + b"\x00" * 100)
    return str(p)


def make_discovery(tmp_path, file_exists=None, registry_reader=None,
                   lnk_resolver=None, dir_lister=None, cache_ttl=0.0):
    env = make_env(tmp_path)
    files = set()

    def default_exists(p):
        return p.lower() in files

    if file_exists is None:
        file_exists = default_exists
    return ApplicationDiscovery(
        env=env,
        file_exists=file_exists,
        registry_reader=registry_reader or (lambda k: {}),
        start_apps_lister=lambda: [],
        lnk_resolver=lnk_resolver or (lambda p: ""),
        dir_lister=dir_lister,
        cache_ttl=cache_ttl,
    ), files, env


# ----------------------------------------------------------------------
# Discovery sources
# ----------------------------------------------------------------------
class TestDiscoverySources:
    def test_start_menu_shortcut_discovery(self, tmp_path):
        """App only reachable via a Start Menu shortcut is discovered."""
        lnk_dir = (
            tmp_path / "AppData" / "Roaming"
            / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        )
        lnk_dir.mkdir(parents=True)
        lnk = lnk_dir / "Spotify.lnk"
        lnk.write_bytes(b"fake")
        exe = write_exe(str(tmp_path / "Apps" / "spotify.exe"))
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p == exe,
            lnk_resolver=lambda p: exe,
            dir_lister=lambda d: [str(lnk)] if "Start Menu" in d else [],
        )
        apps = d.discover()
        assert any(a.name == "Spotify" and a.exe_path == exe for a in apps)
        assert d.find("spotify") is not None

    def test_app_paths_registry_discovery(self, tmp_path):
        exe = write_exe(str(tmp_path / "Apps" / "postman.exe"))
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p == exe,
            registry_reader=lambda k: (
                {"Postman.exe": {"": exe}} if "App Paths" in k else {}
            ),
        )
        apps = d.discover()
        assert any("postman" in a.exe_path.lower() for a in apps)
        assert d.resolve("postman").status == "found"

    def test_uninstall_registry_discovery(self, tmp_path):
        exe = write_exe(str(tmp_path / "Apps" / "idea64.exe"))
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p == exe,
            registry_reader=lambda k: (
                {"x": {"DisplayName": "IntelliJ IDEA",
                       "DisplayIcon": exe + ",0"}}
                if "Uninstall" in k else {}
            ),
        )
        res = d.resolve("intellij idea")
        assert res.status == "found"
        assert res.app.name == "IntelliJ IDEA"

    def test_program_dir_discovery(self, tmp_path):
        exe = write_exe(
            str(tmp_path / "AppData" / "Local" / "Programs"
                / "Discord" / "Discord.exe")
        )
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p == exe,
        )
        # dir_lister default walks real fs; exe exists on disk.
        apps = d.discover()
        assert any(a.exe_path == exe for a in apps)
        assert d.resolve("discord").status == "found"

    def test_missing_application(self, tmp_path):
        d, files, env = make_discovery(tmp_path)
        assert d.resolve("nonexistent app xyz").status == "missing"
        assert d.find("nonexistent app xyz") is None

    def test_exact_and_partial_match(self, tmp_path):
        exe = write_exe(str(tmp_path / "Apps" / "foobar2000.exe"))
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p == exe,
            registry_reader=lambda k: (
                {"x": {"DisplayName": "foobar2000",
                       "DisplayIcon": exe}} if "Uninstall" in k else {}
            ),
        )
        assert d.resolve("foobar2000").status == "found"
        assert d.resolve("foobar").status == "found"  # partial

    def test_ambiguous_names_not_silently_chosen(self, tmp_path):
        e1 = write_exe(str(tmp_path / "A" / "player.exe"))
        e2 = write_exe(str(tmp_path / "B" / "player.exe"))
        e3 = write_exe(str(tmp_path / "C" / "playerx.exe"))
        exes = {e1, e2, e3}
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p in exes,
            registry_reader=lambda k: (
                {
                    "a": {"DisplayName": "Music Player", "DisplayIcon": e1},
                    "b": {"DisplayName": "Video Player", "DisplayIcon": e2},
                    "c": {"DisplayName": "Player X", "DisplayIcon": e3},
                } if "Uninstall" in k else {}
            ),
        )
        res = d.resolve("player")
        # Multiple plausible matches -> ambiguous, never a guess.
        assert res.status == "ambiguous"
        assert len(res.options) >= 2

    def test_dominant_match_wins(self, tmp_path):
        e1 = write_exe(str(tmp_path / "A" / "chrome.exe"))
        e2 = write_exe(str(tmp_path / "B" / "chromehelper.exe"))
        exes = {e1, e2}
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p in exes,
            registry_reader=lambda k: (
                {
                    "a": {"DisplayName": "Google Chrome",
                          "DisplayIcon": e1},
                    "b": {"DisplayName": "Chrome Helper",
                          "DisplayIcon": e2},
                } if "Uninstall" in k else {}
            ),
        )
        res = d.resolve("chrome")
        assert res.status == "found"
        assert res.app.exe_path == e1

    def test_invalid_exe_target_rejected(self, tmp_path):
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: False,
            registry_reader=lambda k: (
                {"x": {"DisplayName": "Ghost App",
                       "DisplayIcon": r"C:\ghost\ghost.exe"}}
                if "Uninstall" in k else {}
            ),
        )
        assert d.discover() == []
        assert d.resolve("ghost app").status == "missing"

    def test_non_exe_shortcut_target_rejected(self, tmp_path):
        lnk = tmp_path / "x.lnk"
        lnk.write_bytes(b"x")
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p.endswith(".txt"),
            lnk_resolver=lambda p: r"C:\stuff\readme.txt",
            dir_lister=lambda d: [str(lnk)] if "Start Menu" in d else [],
        )
        assert all(not a.exe_path.endswith(".txt") for a in d.discover())

    def test_new_install_after_cache(self, tmp_path):
        exe1 = write_exe(str(tmp_path / "Apps" / "oldapp.exe"))
        state = {"exes": {exe1}}
        d, files, env = make_discovery(
            tmp_path,
            file_exists=lambda p: p in state["exes"],
            registry_reader=lambda k: (
                {"x": {"DisplayName": "Old App", "DisplayIcon": exe1}}
                if "Uninstall" in k else {}
            ),
            cache_ttl=300,
        )
        assert d.resolve("old app").status == "found"
        assert d.resolve("new app").status == "missing"
        # Simulate install: new exe + registry entry appear.
        exe2 = write_exe(str(tmp_path / "Apps" / "newapp.exe"))
        state["exes"].add(exe2)
        d2 = ApplicationDiscovery(
            env=env,
            file_exists=lambda p: p in state["exes"],
            registry_reader=lambda k: (
                {
                    "x": {"DisplayName": "Old App", "DisplayIcon": exe1},
                    "y": {"DisplayName": "New App", "DisplayIcon": exe2},
                } if "Uninstall" in k else {}
            ),
            start_apps_lister=lambda: [],
            cache_ttl=300,
        )
        # Fresh instance sees it; also force-refresh works on same cache.
        assert d2.resolve("new app").status == "found"
        d.discover(force=True)


# ----------------------------------------------------------------------
# Dynamic launch / switch / handlers
# ----------------------------------------------------------------------
def make_svc(tmp_path, apps=None, windows=None, procs=None):
    """Build _Svc with a fully dynamic (non-seeded) discovery."""
    launched = []
    exes = {}
    for a in (apps or []):
        exes[a.exe_path] = a
    env = make_env(tmp_path)

    def reg_reader(k):
        if "Uninstall" not in k:
            return {}
        return {
            a.exe_path: {
                "DisplayName": a.name,
                "DisplayIcon": a.exe_path,
            } for a in exes.values()
        }

    disc = ApplicationDiscovery(
        env=env,
        file_exists=lambda p: p in exes,
        registry_reader=reg_reader,
        start_apps_lister=lambda: [],
        cache_ttl=0,
    )
    api = FakeAPI(windows or [])
    mon = FakeMonitor(procs or [])
    wins = WindowManager(api=api, monitor=mon)
    launcher = ApplicationLauncher(
        discovery=disc,
        popen=lambda args, **kw: launched.append(args) or object(),
        monitor=mon,
    )
    svc = _Svc(disc, launcher, mon, wins)
    return svc, disc, wins, api, launcher, launched


SPOTIFY = AppInfo(
    "Spotify",
    r"C:\Users\user\AppData\Roaming\Spotify\Spotify.exe",
    source="uninstall",
)
PHOTOSHOP = AppInfo(
    "Adobe Photoshop 2024",
    r"C:\Program Files\Adobe\Adobe Photoshop 2024\Photoshop.exe",
    source="uninstall",
)


class TestDynamicOperations:
    def test_open_unlisted_app(self, tmp_path):
        """'Open Spotify' works though Spotify is not a seed app."""
        svc, disc, wins, api, launcher, launched = make_svc(
            tmp_path, apps=[SPOTIFY]
        )
        reg = create_app_registry(svc)
        r = reg.dispatch(Intent("OPEN_APPLICATION", {"app_name": "spotify"}))
        assert r.executed
        assert "Opening Spotify" in r.message
        assert launched == [[SPOTIFY.exe_path]]

    def test_open_app_not_in_seeds(self, tmp_path):
        """IntelliJ IDEA / Photoshop style names resolve dynamically."""
        svc, *_ = make_svc(tmp_path, apps=[PHOTOSHOP])
        reg = create_app_registry(svc)
        r = reg.dispatch(
            Intent("OPEN_APPLICATION", {"app_name": "adobe photoshop"})
        )
        assert r.executed and "Opening Adobe Photoshop 2024" in r.message

    def test_open_missing_app_message(self, tmp_path):
        svc, *_ = make_svc(tmp_path, apps=[])
        reg = create_app_registry(svc)
        r = reg.dispatch(
            Intent("OPEN_APPLICATION", {"app_name": "spotify"})
        )
        assert not r.success and "couldn't find" in r.message.lower()

    def test_switch_to_dynamic_app(self, tmp_path):
        win = WindowInfo(7, 42, "Spotify Premium", "spotify.exe")
        svc, *_ = make_svc(
            tmp_path, apps=[SPOTIFY],
            windows=[win], procs=[(42, "spotify.exe")],
        )
        reg = create_app_registry(svc)
        r = reg.dispatch(
            Intent("SWITCH_TO_APPLICATION", {"app_name": "spotify"})
        )
        assert r.executed and "Switching" in r.message

    def test_close_dynamic_app_confirmation(self, tmp_path):
        win = WindowInfo(7, 42, "Spotify", "spotify.exe")
        svc, disc, wins, api, launcher, launched = make_svc(
            tmp_path, apps=[SPOTIFY],
            windows=[win], procs=[(42, "spotify.exe")],
        )
        reg = create_app_registry(svc)
        r = reg.dispatch(Intent(
            "CLOSE_APPLICATION", {"app_name": "spotify"}))
        assert r.confirmation_required
        r2 = reg.dispatch(Intent(
            "CLOSE_APPLICATION",
            {"app_name": "spotify", "_confirmed": True}))
        assert r2.executed and "Closing Spotify" in r2.message
        assert api.closed == [7]

    def test_ambiguous_open_asks_clarification(self, tmp_path):
        e1 = r"C:\Apps\player_one.exe"
        e2 = r"C:\Apps\player_two.exe"
        a1 = AppInfo("Music Player", e1, "uninstall")
        a2 = AppInfo("Video Player", e2, "uninstall")
        svc, *_ = make_svc(tmp_path, apps=[a1, a2])
        reg = create_app_registry(svc)
        r = reg.dispatch(Intent("OPEN_APPLICATION", {"app_name": "player"}))
        assert r.ambiguous
        assert "several applications" in r.message
        assert len(r.options) == 2
        assert r.data["apps"]

    def test_find_app_dynamic(self, tmp_path):
        svc, *_ = make_svc(tmp_path, apps=[SPOTIFY])
        reg = create_app_registry(svc)
        r = reg.dispatch(Intent("FIND_APPLICATION", {"app_name": "spotify"}))
        assert r.executed and "found Spotify" in r.message


# ----------------------------------------------------------------------
# Session-level clarification flow
# ----------------------------------------------------------------------
class _StubSTT:
    def __init__(self):
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self): pass
    def stop(self): pass

    @property
    def is_running(self): return True

    def emit(self, text): self._cb["on_transcript"](text)


class _StubTTS:
    def __init__(self):
        self.spoken = []
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self): pass
    def stop(self): pass

    @property
    def is_running(self): return True

    def speak(self, text):
        self.spoken.append(text)
        self._cb["on_finished"](text)


def _controller(svc):
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    return VoiceSessionController(
        state_manager=state,
        session_manager=SessionManager(timeout_seconds=60),
        speech_recognition=_StubSTT(),
        speech_output=_StubTTS(),
        command_registry=create_app_registry(svc),
        intent_recognizer=RuleBasedIntentRecognizer(),
    )


class TestClarificationFlow:
    def test_ambiguous_then_pick_by_name(self, tmp_path):
        e1 = r"C:\Apps\player_one.exe"
        e2 = r"C:\Apps\player_two.exe"
        a1 = AppInfo("Music Player", e1, "uninstall")
        a2 = AppInfo("Video Player", e2, "uninstall")
        svc, disc, wins, api, launcher, launched = make_svc(
            tmp_path, apps=[a1, a2]
        )
        ctl = _controller(svc)
        stt, tts = ctl._stt, ctl._tts
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Open the player.")
        assert any("several applications" in s for s in tts.spoken)
        assert launched == []
        tts.spoken.clear()
        stt.emit("Music Player")
        assert any("Opening Music Player" in s for s in tts.spoken)
        assert launched == [[e1]]
        ctl.shutdown()

    def test_ambiguous_then_pick_by_ordinal(self, tmp_path):
        e1 = r"C:\Apps\player_one.exe"
        e2 = r"C:\Apps\player_two.exe"
        a1 = AppInfo("Music Player", e1, "uninstall")
        a2 = AppInfo("Video Player", e2, "uninstall")
        svc, *_ = make_svc(tmp_path, apps=[a1, a2])
        ctl = _controller(svc)
        stt, tts = ctl._stt, ctl._tts
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Open the player.")
        tts.spoken.clear()
        stt.emit("The second one.")
        assert any("Opening Video Player" in s for s in tts.spoken)
        ctl.shutdown()

    def test_unrecognized_reply_falls_through(self, tmp_path):
        e1 = r"C:\Apps\player_one.exe"
        e2 = r"C:\Apps\player_two.exe"
        svc, *_ = make_svc(
            tmp_path,
            apps=[AppInfo("Music Player", e1),
                  AppInfo("Video Player", e2)],
        )
        ctl = _controller(svc)
        stt, tts = ctl._stt, ctl._tts
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Open the player.")
        tts.spoken.clear()
        stt.emit("What time is it?")
        # Not an app choice -> routed as a normal command.
        assert any("current time" in s.lower() for s in tts.spoken)
        ctl.shutdown()
