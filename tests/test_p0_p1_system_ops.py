"""Tests for P0 + P1 Windows operations:

- P0: power phrases must not end the voice session and must never be
  executed (UNSUPPORTED_WINDOWS_ACTION).
- P1.1: OPEN_FOLDER via allowlisted known-folder resolution.
- P1.2: master output volume (increase/decrease/mute/unmute/set/get).
- P1.3: LOCK_WORKSTATION behind the confirmation lifecycle.

All Windows/COM interaction is faked — no real folders are opened, no
real volume changes, no real lock.
"""

from __future__ import annotations

import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.handlers import (
    SUPPORTED_FOLDERS,
    UnsupportedActionHandler,
    create_default_registry,
)
from src.commands.intent import Intent, RuleBasedIntentRecognizer
from src.commands.system_handlers import (
    LockWorkstationHandler,
    OpenKnownFolderHandler,
)
from src.controllers.folder_opener import (
    FolderOpener,
    KnownFolderResolver,
    _KNOWN_FOLDERS,
)
from src.controllers.volume_controller import VolumeController
from src.controllers.windows_automation import WorkstationLocker
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.voice_session_controller import VoiceSessionController


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeResolver:
    def __init__(self, paths=None):
        self.paths = paths or {}
        self.requested = []

    def resolve(self, name):
        self.requested.append(name)
        from pathlib import Path

        p = self.paths.get(name)
        return Path(p) if p else None


class FakeOpener:
    def __init__(self, ok=True, detail="ok"):
        self.ok = ok
        self.detail = detail
        self.opened = []

    def open(self, name):
        self.opened.append(name)
        return self.ok, self.detail


class FakeVolume:
    def __init__(self, level=40, muted=False, available=True):
        self.level = level
        self.muted = muted
        self.available = available
        self.calls = []

    def _avail(self):
        return self.available

    def adjust(self, delta):
        self.calls.append(("adjust", delta))
        if not self.available:
            return None
        self.level = max(0, min(self.level + delta, 100))
        return self.level

    def set_percent(self, pct):
        self.calls.append(("set", pct))
        if not self.available:
            return None
        self.level = max(0, min(pct, 100))
        return self.level

    def get_percent(self):
        self.calls.append(("get",))
        return self.level if self.available else None

    def is_muted(self):
        return self.muted if self.available else None

    def set_mute(self, muted):
        self.calls.append(("mute", muted))
        if not self.available:
            return False
        self.muted = muted
        return True


class FakeLocker:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = 0

    def lock(self):
        self.calls += 1
        return self.ok


def make_svc(**kw):
    return _Svc(
        None, None, None, None,
        folders=kw.get("folders"),
        volume=kw.get("volume"),
        locker=kw.get("locker"),
    )


# ----------------------------------------------------------------------
# P0 — power phrase mappings
# ----------------------------------------------------------------------
class TestPowerPhrases:
    @pytest.fixture
    def rec(self):
        return RuleBasedIntentRecognizer()

    @pytest.mark.parametrize("text", [
        "Goodbye Jarvis",
        "Goodbye.",
        "Bye.",
        "End session.",
        "That's all.",
        "That is all for now.",
        "Go to standby.",
        "Go back to standby.",
        "Stop listening.",
    ])
    def test_session_end_phrases(self, rec, text):
        assert rec.recognize(text).name == "END_SESSION"

    # Milestone 1: sleep/restart/shutdown are now real intents behind
    # the confirmation lifecycle; only hibernate/sign-out remain
    # unsupported.
    @pytest.mark.parametrize("text,expected", [
        ("Go to sleep.", "SLEEP_COMPUTER"),
        ("Sleep this computer.", "SLEEP_COMPUTER"),
        ("Sleep the PC.", "SLEEP_COMPUTER"),
        ("Put the computer to sleep.", "SLEEP_COMPUTER"),
        ("Send my laptop to sleep.", "SLEEP_COMPUTER"),
        ("Shut down.", "SHUTDOWN_COMPUTER"),
        ("Shut down Windows.", "SHUTDOWN_COMPUTER"),
        ("Shut down the computer.", "SHUTDOWN_COMPUTER"),
        ("Power off.", "SHUTDOWN_COMPUTER"),
        ("Power off the computer.", "SHUTDOWN_COMPUTER"),
        ("Power down.", "SHUTDOWN_COMPUTER"),
        ("Restart.", "RESTART_COMPUTER"),
        ("Restart Windows.", "RESTART_COMPUTER"),
        ("Turn off the computer.", "SHUTDOWN_COMPUTER"),
        ("Switch off the machine.", "SHUTDOWN_COMPUTER"),
    ])
    def test_power_phrases_now_supported(self, rec, text, expected):
        assert rec.recognize(text).name == expected, text

    @pytest.mark.parametrize("text", [
        "Hibernate.",
        "Sign out.",
        "Sign out of Windows.",
    ])
    def test_power_phrases_are_unsupported(self, rec, text):
        intent = rec.recognize(text)
        assert intent.name == "UNSUPPORTED_WINDOWS_ACTION", text
        assert intent.entities.get("action")

    @pytest.mark.parametrize("text", [
        "Turn off the volume.",
        "Turn off the sound.",
        "Mute.",
        "Increase volume.",
        "What time is it?",
        "Close Chrome.",
        "Open Downloads.",
        "Lock my computer.",
    ])
    def test_non_power_phrases_unaffected(self, rec, text):
        assert rec.recognize(text).name != "UNSUPPORTED_WINDOWS_ACTION"

    def test_unsupported_dispatch_never_executes(self):
        reg = create_default_registry()
        result = reg.dispatch(
            Intent("UNSUPPORTED_WINDOWS_ACTION", {"action": "shut_down"})
        )
        assert result.success is False
        assert result.executed is False
        assert "shut down" in result.message.lower()


# ----------------------------------------------------------------------
# P0 — session integration: power phrases don't end the session
# ----------------------------------------------------------------------
class _StubSTT:
    def __init__(self):
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def is_running(self):
        return True

    def emit(self, text):
        self._cb["on_transcript"](text)


class _StubTTS:
    def __init__(self):
        self.spoken = []
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def is_running(self):
        return True

    def speak(self, text):
        self.spoken.append(text)
        self._cb["on_finished"](text)


def _controller(**kw):
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    session = SessionManager(timeout_seconds=60)
    stt, tts = _StubSTT(), _StubTTS()
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=session,
        speech_recognition=stt,
        speech_output=tts,
        **kw,
    )
    return ctl, state, session, stt, tts


class TestPowerSessionIntegration:
    @pytest.mark.parametrize("text", [
        "Go to sleep.",
        "Shut down Windows.",
        "Power off the computer.",
        "Sleep this computer.",
    ])
    def test_power_phrase_keeps_session_alive(self, text):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit(text)
        assert session.is_active, text
        assert "Goodbye." not in tts.spoken
        # Milestone 1: these are real intents now — with the default
        # (stub) registry they are unregistered, but the session must
        # still stay alive and never execute a power action.
        ctl.shutdown()

    def test_session_end_still_works(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("That's all.")
        assert "Goodbye." in tts.spoken
        ctl.shutdown()


# ----------------------------------------------------------------------
# P1.1 — known folders
# ----------------------------------------------------------------------
class TestFolderResolution:
    def test_all_allowlisted_folders_resolve(self, tmp_path):
        dirs = {name: tmp_path / name for name in _KNOWN_FOLDERS}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        resolver = KnownFolderResolver(
            env={"USERPROFILE": str(tmp_path)},
            query_known=lambda guid: None,
        )
        for name in (
            "desktop", "documents", "downloads",
            "pictures", "music", "videos",
        ):
            path = resolver.resolve(name)
            assert path is not None, name
            assert path.is_dir()

    def test_aliases_resolve_to_canonical(self, tmp_path):
        (tmp_path / "Pictures").mkdir()
        (tmp_path / "Videos").mkdir()
        resolver = KnownFolderResolver(
            env={"USERPROFILE": str(tmp_path)},
            query_known=lambda guid: None,
        )
        assert resolver.resolve("photos") == tmp_path / "Pictures"
        assert resolver.resolve("movies") == tmp_path / "Videos"

    def test_known_folder_api_path_preferred(self, tmp_path):
        api_dir = tmp_path / "ApiDownloads"
        api_dir.mkdir()
        resolver = KnownFolderResolver(
            env={"USERPROFILE": str(tmp_path / "nope")},
            query_known=lambda guid: str(api_dir),
        )
        assert resolver.resolve("downloads") == api_dir

    def test_missing_folder_returns_none(self, tmp_path):
        resolver = KnownFolderResolver(
            env={"USERPROFILE": str(tmp_path)},
            query_known=lambda guid: None,
        )
        assert resolver.resolve("downloads") is None

    def test_invalid_name_returns_none(self):
        resolver = KnownFolderResolver(
            env={}, query_known=lambda guid: None,
            is_dir=lambda p: True,
        )
        assert resolver.resolve("secretstuff") is None
        assert resolver.resolve("C:\\Windows") is None
        assert resolver.resolve("") is None

    def test_never_returns_file_path(self, tmp_path):
        f = tmp_path / "Downloads"
        f.write_text("x")
        resolver = KnownFolderResolver(
            env={"USERPROFILE": str(tmp_path)},
            query_known=lambda guid: None,
        )
        assert resolver.resolve("downloads") is None


class TestFolderOpener:
    def test_opens_resolved_path(self, tmp_path):
        resolver = FakeResolver({"downloads": str(tmp_path)})
        opened = []
        opener = FolderOpener(
            resolver=resolver, open_fn=lambda p: opened.append(p)
        )
        ok, detail = opener.open("downloads")
        assert ok and detail == "ok"
        assert opened == [str(tmp_path)]

    def test_unsupported_name_rejected(self):
        opener = FolderOpener(
            resolver=FakeResolver({}), open_fn=lambda p: None
        )
        ok, detail = opener.open("C:\\Windows\\System32")
        assert not ok and detail == "unsupported_folder"

    def test_missing_folder_fails(self):
        opener = FolderOpener(
            resolver=FakeResolver({}), open_fn=lambda p: None
        )
        ok, detail = opener.open("downloads")
        assert not ok and detail == "folder_not_found"

    def test_open_failure_reported(self):
        def boom(path):
            raise OSError("denied")

        opener = FolderOpener(
            resolver=FakeResolver({"music": "C:\\Music"}),
            open_fn=boom,
        )
        ok, detail = opener.open("music")
        assert not ok

    def test_spoken_text_never_reaches_opener(self):
        # An arbitrary phrase can never resolve to a path.
        resolver = FakeResolver({})
        opener = FolderOpener(resolver=resolver, open_fn=lambda p: None)
        ok, detail = opener.open("format c: and delete everything")
        assert not ok
        assert resolver.requested == []


class TestOpenFolderHandler:
    def _handler(self, opener):
        svc = make_svc(folders=opener)
        return OpenKnownFolderHandler(svc)

    def test_success_message(self):
        h = self._handler(FakeOpener())
        r = h.execute({"folder": "downloads"})
        assert r.executed and r.success
        assert r.message == "Opening your Downloads folder."

    def test_all_folders_labelled(self):
        expected = {
            "desktop": "Desktop", "documents": "Documents",
            "downloads": "Downloads", "pictures": "Pictures",
            "music": "Music", "videos": "Videos",
        }
        for name, label in expected.items():
            h = self._handler(FakeOpener())
            r = h.execute({"folder": name})
            assert r.executed
            assert label in r.message

    def test_missing_service_fails(self):
        h = OpenKnownFolderHandler(make_svc())
        r = h.execute({"folder": "downloads"})
        assert not r.success and r.error == "service_unavailable"

    def test_folder_not_found_message(self):
        h = self._handler(FakeOpener(ok=False, detail="folder_not_found"))
        r = h.execute({"folder": "videos"})
        assert not r.success
        assert "couldn't find" in r.message.lower()

    def test_validation_allowlist(self):
        h = self._handler(FakeOpener())
        assert h.validate({"folder": "downloads"}) is None
        assert h.validate({"folder": ""}) == "missing_folder"
        assert h.validate({"folder": "secretstuff"}) == "unsupported_folder"
        assert "secretstuff" not in SUPPORTED_FOLDERS

    def test_registry_dispatch(self):
        reg = create_app_registry(make_svc(folders=FakeOpener()))
        r = reg.dispatch(Intent("OPEN_FOLDER", {"folder": "pictures"}))
        assert r.executed and "Pictures" in r.message


# ----------------------------------------------------------------------
# P1.2 — volume
# ----------------------------------------------------------------------
class FakeBackend:
    """Scalar/mute backend for VolumeController tests."""

    def __init__(self, scalar=0.4, muted=False):
        self.scalar = scalar
        self.muted = muted
        self.sets = []

    def get_scalar(self):
        return self.scalar

    def set_scalar(self, v):
        self.sets.append(v)
        self.scalar = v

    def get_mute(self):
        return self.muted

    def set_mute(self, m):
        self.muted = m


class TestVolumeController:
    def test_get_set_percent(self):
        b = FakeBackend(0.4)
        vol = VolumeController(backend=b)
        assert vol.get_percent() == 40
        assert vol.set_percent(55) == 55
        assert b.scalar == 0.55

    def test_clamps_range(self):
        b = FakeBackend(0.4)
        vol = VolumeController(backend=b)
        assert vol.set_percent(250) == 100
        assert vol.set_percent(-5) == 0

    def test_adjust_up_down(self):
        b = FakeBackend(0.4)
        vol = VolumeController(backend=b)
        assert vol.adjust(10) == 50
        assert vol.adjust(-60) == 0
        assert vol.adjust(200) == 100

    def test_mute_unmute(self):
        b = FakeBackend()
        vol = VolumeController(backend=b)
        assert vol.set_mute(True) is True
        assert vol.is_muted() is True
        assert vol.set_mute(False) is True
        assert vol.is_muted() is False

    def test_missing_endpoint_returns_none(self):
        vol = VolumeController(backend_factory=lambda: None)
        assert vol.get_percent() is None
        assert vol.set_percent(50) is None
        assert vol.adjust(5) is None
        assert vol.set_mute(True) is False
        assert vol.is_muted() is None

    def test_factory_called_per_thread(self):
        calls = []

        def factory():
            calls.append(1)
            return FakeBackend()

        vol = VolumeController(backend=factory)
        assert vol.get_percent() == 40
        import threading

        t = threading.Thread(target=vol.get_percent)
        t.start()
        t.join()
        assert len(calls) == 2  # one backend per thread


class TestVolumeHandlers:
    def _reg(self, volume=None):
        return create_app_registry(make_svc(volume=volume))

    def test_increase(self):
        vol = FakeVolume(level=40)
        r = self._reg(vol).dispatch(Intent("INCREASE_VOLUME", {}))
        assert r.executed and "45 percent" in r.message
        assert vol.calls == [("adjust", 5)]

    def test_increase_with_amount(self):
        vol = FakeVolume(level=40)
        r = self._reg(vol).dispatch(
            Intent("INCREASE_VOLUME", {"amount": 20})
        )
        assert r.executed and "60 percent" in r.message
        assert vol.calls == [("adjust", 20)]

    def test_decrease(self):
        vol = FakeVolume(level=40)
        r = self._reg(vol).dispatch(Intent("DECREASE_VOLUME", {}))
        assert r.executed and "35 percent" in r.message
        assert vol.calls == [("adjust", -5)]

    def test_mute_unmute(self):
        vol = FakeVolume()
        reg = self._reg(vol)
        r1 = reg.dispatch(Intent("MUTE_VOLUME", {}))
        r2 = reg.dispatch(Intent("UNMUTE_VOLUME", {}))
        assert r1.executed and r1.message == "Muted."
        assert r2.executed and r2.message == "Unmuted."
        assert vol.muted is False
        assert ("mute", True) in vol.calls and ("mute", False) in vol.calls

    def test_set_volume(self):
        vol = FakeVolume()
        r = self._reg(vol).dispatch(
            Intent("SET_VOLUME", {"amount": 40})
        )
        assert r.executed
        assert "40 percent" in r.message
        assert vol.calls == [("set", 40)]

    def test_set_volume_validation(self):
        reg = self._reg(FakeVolume())
        bad = reg.dispatch(Intent("SET_VOLUME", {"amount": 250}))
        assert not bad.success and bad.error == "amount_out_of_range"
        missing = reg.dispatch(Intent("SET_VOLUME", {}))
        assert not missing.success and missing.error == "missing_amount"

    def test_get_volume(self):
        vol = FakeVolume(level=40)
        r = self._reg(vol).dispatch(Intent("GET_VOLUME", {}))
        assert r.executed and "40 percent" in r.message

    def test_get_volume_muted(self):
        vol = FakeVolume(level=40, muted=True)
        r = self._reg(vol).dispatch(Intent("GET_VOLUME", {}))
        assert r.executed and "muted" in r.message.lower()

    def test_no_endpoint_fails_honestly(self):
        vol = FakeVolume(available=False)
        reg = self._reg(vol)
        for intent, name in (
            (Intent("INCREASE_VOLUME", {}), "increase"),
            (Intent("MUTE_VOLUME", {}), "mute"),
            (Intent("SET_VOLUME", {"amount": 50}), "set"),
            (Intent("GET_VOLUME", {}), "get"),
        ):
            r = reg.dispatch(intent)
            assert not r.success, name
            assert r.error == "no_audio_endpoint"

    def test_missing_service_fails(self):
        reg = create_app_registry(make_svc())
        r = reg.dispatch(Intent("INCREASE_VOLUME", {}))
        assert not r.success and r.error == "service_unavailable"


# ----------------------------------------------------------------------
# P1.3 — lock workstation
# ----------------------------------------------------------------------
class TestWorkstationLocker:
    def test_lock_success(self):
        locker = WorkstationLocker(lock_fn=lambda: 1)
        assert locker.lock() is True

    def test_lock_failure(self):
        locker = WorkstationLocker(lock_fn=lambda: 0)
        assert locker.lock() is False

    def test_lock_exception(self):
        def boom():
            raise OSError("no")

        locker = WorkstationLocker(lock_fn=boom)
        assert locker.lock() is False


class TestLockHandler:
    def _handler(self, locker):
        return LockWorkstationHandler(make_svc(locker=locker))

    def test_requires_confirmation(self):
        locker = FakeLocker()
        r = self._handler(locker).execute({})
        assert r.confirmation_required and not r.executed
        assert locker.calls == 0

    def test_confirmed_locks(self):
        locker = FakeLocker(ok=True)
        r = self._handler(locker).execute({"_confirmed": True})
        assert r.executed and r.success
        assert "Locking your workstation" in r.message
        assert locker.calls == 1

    def test_api_failure_not_reported_as_success(self):
        locker = FakeLocker(ok=False)
        r = self._handler(locker).execute({"_confirmed": True})
        assert not r.success
        assert "couldn't lock" in r.message.lower()

    def test_missing_service(self):
        r = LockWorkstationHandler(make_svc()).execute({"_confirmed": True})
        assert not r.success and r.error == "service_unavailable"


class TestLockConfirmationFlow:
    """End-to-end through VoiceSessionController._pending_confirm."""

    def _setup(self, locker):
        svc = make_svc(locker=locker)
        ctl, state, session, stt, tts = _controller(
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        tts.spoken.clear()
        return ctl, session, stt, tts, locker

    def test_yes_locks(self):
        locker = FakeLocker(ok=True)
        ctl, session, stt, tts, _ = self._setup(locker)
        stt.emit("Lock my computer.")
        assert locker.calls == 0
        assert any("lock" in s.lower() for s in tts.spoken)
        stt.emit("Yes.")
        assert locker.calls == 1
        assert session.is_active
        ctl.shutdown()

    def test_no_cancels(self):
        locker = FakeLocker()
        ctl, session, stt, tts, _ = self._setup(locker)
        stt.emit("Lock Windows.")
        stt.emit("No.")
        assert locker.calls == 0
        ctl.shutdown()

    def test_unrelated_command_clears_confirmation(self):
        locker = FakeLocker()
        ctl, session, stt, tts, _ = self._setup(locker)
        stt.emit("Lock my computer.")
        stt.emit("What time is it?")
        stt.emit("Yes.")
        assert locker.calls == 0
        ctl.shutdown()

    def test_expired_confirmation_does_not_lock(self):
        locker = FakeLocker()
        ctl, session, stt, tts, _ = self._setup(locker)
        stt.emit("Lock workstation.")
        # Age the pending confirmation past the window.
        pending = ctl._pending_confirm
        assert pending is not None
        pending["at"] -= ctl._CONFIRM_WINDOW + 1
        stt.emit("Yes.")
        assert locker.calls == 0
        ctl.shutdown()
