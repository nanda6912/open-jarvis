"""Tests for Windows System Control (Milestone 1).

Power management (sleep / scheduled shutdown / scheduled restart /
cancel) and desktop/window controls. ALL destructive Windows
operations are mocked — nothing here sleeps, restarts, shuts down,
closes windows or sends real keystrokes.
"""

from __future__ import annotations

import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.intent import Intent, RuleBasedIntentRecognizer
from src.commands.system_control_handlers import (
    CancelPowerActionHandler,
    CloseActiveWindowHandler,
    ShutdownComputerHandler,
)
from src.controllers.desktop_control import DesktopController
from src.controllers.power_manager import (
    SCHEDULE_DELAY_SECONDS,
    PowerManager,
)
from src.controllers.windows_automation import WindowInfo
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.voice_session_controller import VoiceSessionController


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class FakePower:
    """Records schedule/sleep/abort calls; real pending semantics."""

    def __init__(self, clock=None, schedule_ok=True, abort_ok=True,
                 sleep_ok=True):
        self.clock = clock or FakeClock()
        self.calls = []
        self._schedule_ok = schedule_ok
        self._abort_ok = abort_ok
        self._sleep_ok = sleep_ok
        self._pending = None

    def pending_action(self):
        p = self._pending
        if p is None:
            return None
        if self.clock() >= p["deadline"]:
            self._pending = None
            return None
        return p["action"]

    def sleep(self):
        self.calls.append("sleep")
        return self._sleep_ok

    def schedule(self, action):
        self.calls.append(("schedule", action))
        if not self._schedule_ok or self.pending_action() is not None:
            return False
        self._pending = {
            "action": action,
            "deadline": self.clock() + SCHEDULE_DELAY_SECONDS,
        }
        return True

    def cancel(self):
        self.calls.append("cancel")
        if self.pending_action() is None:
            return False, None, "nothing_pending"
        action = self._pending["action"]
        if not self._abort_ok:
            return False, action, "abort_failed"
        self._pending = None
        return True, action, None


class FakeDesktop:
    """DesktopController-shaped fake; records every operation."""

    def __init__(self, active=None):
        self.active = active or WindowInfo(
            hwnd=100, pid=1, title="Notepad", process="notepad.exe"
        )
        self.calls = []
        self.closed = []
        self.close_result = (True, "ok")
        self.switch_result = (True, "ok", "Chrome")
        self.desktop_ok = True
        self.snap_result = (True, "ok", "Notepad")
        self.min_result = (True, "ok", "Notepad")
        self.restore_result = (True, "ok", "Notepad")

    def active_window(self):
        return self.active

    def minimize_active(self):
        self.calls.append("minimize_active")
        return self.min_result

    def restore_active(self):
        self.calls.append("restore_active")
        return self.restore_result

    def close_window(self, hwnd):
        self.closed.append(hwnd)
        return self.close_result

    def switch_window(self, direction):
        self.calls.append(("switch_window", direction))
        return self.switch_result

    def switch_desktop(self, direction):
        self.calls.append(("switch_desktop", direction))
        return self.desktop_ok

    def snap_active(self, direction):
        self.calls.append(("snap", direction))
        return self.snap_result


class FakeWindowManager:
    """Minimal WindowManager stand-in for DesktopController tests."""

    def __init__(self, windows, fg_hwnd):
        self._windows = windows
        self._fg = fg_hwnd
        self.focused = []
        self.closed = []
        self.api = type("A", (), {"foreground_hwnd": lambda s: self._fg})()

    def list_windows(self):
        return list(self._windows)

    def focus(self, hwnd):
        self.focused.append(hwnd)
        return True

    def close_window(self, hwnd):
        self.closed.append(hwnd)
        return True

    def foreground_window(self):
        for w in self._windows:
            if w.hwnd == self._fg:
                return w
        return None

    def show_desktop(self):
        pass

    def minimize(self, hwnd):
        pass

    def restore(self, hwnd):
        pass


# ----------------------------------------------------------------------
# STT/TTS stubs (manual-finish TTS for countdown-timing tests)
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


class _ManualTTS:
    """Queues speech; the test decides when each utterance finishes."""

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
        return True

    def finish_next(self):
        """Finish the oldest queued utterance (FIFO)."""
        text = self.spoken.pop(0)
        self._cb["on_finished"](text)
        return text


def _controller(registry, tts=None, timeout=60):
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    session = SessionManager(timeout_seconds=timeout)
    stt = _StubSTT()
    tts = tts or _ManualTTS()
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=session,
        speech_recognition=stt,
        speech_output=tts,
        command_registry=registry,
    )
    return ctl, state, session, stt, tts


def _power_svc(power=None, desktop=None):
    return _Svc(
        None, None, None, None,
        power_ops=power if power is not None else FakePower(),
        desktop=desktop if desktop is not None else FakeDesktop(),
    )


def _wake(ctl, stt, tts):
    assert ctl.handle_wake_word()
    tts.finish_next()  # activation ack


# ======================================================================
# PowerManager unit tests
# ======================================================================
class TestPowerManager:
    def test_schedule_records_pending(self):
        clock = FakeClock()
        scheduled = []
        pm = PowerManager(
            sleep_fn=lambda: True,
            schedule_fn=lambda a: scheduled.append(a) or True,
            abort_fn=lambda: True,
            clock=clock,
        )
        assert pm.schedule("shutdown") is True
        assert scheduled == ["shutdown"]
        assert pm.pending_action() == "shutdown"

    def test_pending_expires_after_delay(self):
        clock = FakeClock()
        pm = PowerManager(
            schedule_fn=lambda a: True, abort_fn=lambda: True,
            clock=clock,
        )
        pm.schedule("restart")
        clock.advance(SCHEDULE_DELAY_SECONDS - 0.5)
        assert pm.pending_action() == "restart"
        clock.advance(1.0)
        assert pm.pending_action() is None

    def test_duplicate_schedule_refused(self):
        pm = PowerManager(schedule_fn=lambda a: True,
                          abort_fn=lambda: True, clock=FakeClock())
        assert pm.schedule("shutdown") is True
        assert pm.schedule("restart") is False
        assert pm.pending_action() == "shutdown"

    def test_schedule_failure_leaves_no_pending(self):
        pm = PowerManager(schedule_fn=lambda a: False,
                          abort_fn=lambda: True, clock=FakeClock())
        assert pm.schedule("shutdown") is False
        assert pm.pending_action() is None

    def test_cancel_only_when_pending(self):
        aborted = []
        pm = PowerManager(schedule_fn=lambda a: True,
                          abort_fn=lambda: aborted.append(1) or True,
                          clock=FakeClock())
        ok, action, err = pm.cancel()
        assert ok is False and err == "nothing_pending"
        assert aborted == []
        pm.schedule("shutdown")
        ok, action, err = pm.cancel()
        assert ok is True and action == "shutdown" and err is None
        assert aborted == [1]
        assert pm.pending_action() is None

    def test_cancel_abort_failure_keeps_pending(self):
        pm = PowerManager(schedule_fn=lambda a: True,
                          abort_fn=lambda: False, clock=FakeClock())
        pm.schedule("restart")
        ok, action, err = pm.cancel()
        assert ok is False and err == "abort_failed"
        assert pm.pending_action() == "restart"

    def test_sleep_delegates(self):
        slept = []
        pm = PowerManager(sleep_fn=lambda: slept.append(1) or True,
                          clock=FakeClock())
        assert pm.sleep() is True and slept == [1]

    def test_default_schedule_uses_fixed_args(self, monkeypatch):
        """The real default path must call shutdown.exe with a FIXED
        argument list — never a shell, never user text."""
        import src.controllers.power_manager as pm_mod

        calls = []

        class _Proc:
            returncode = 0

        def fake_run(argv, **kw):
            calls.append((argv, kw))
            return _Proc()

        monkeypatch.setattr(pm_mod.subprocess, "run", fake_run)
        assert pm_mod._default_schedule("shutdown") is True
        argv, kw = calls[0]
        assert argv[0] == "shutdown.exe"
        assert "/s" in argv and "/t" in argv and "15" in argv
        assert kw.get("shell") in (None, False)
        assert pm_mod._default_schedule("restart") is True
        assert "/r" in calls[1][0]
        assert pm_mod._default_schedule("bogus") is False

    def test_default_abort_uses_fixed_args(self, monkeypatch):
        import src.controllers.power_manager as pm_mod

        calls = []

        class _Proc:
            returncode = 0

        monkeypatch.setattr(
            pm_mod.subprocess, "run",
            lambda argv, **kw: calls.append(argv) or _Proc(),
        )
        assert pm_mod._default_abort() is True
        assert calls[0] == ["shutdown.exe", "/a"]


# ======================================================================
# Intent recognition for new phrases
# ======================================================================
class TestNewIntentRecognition:
    @pytest.fixture
    def rec(self):
        return RuleBasedIntentRecognizer()

    @pytest.mark.parametrize("text,expected", [
        ("Sleep my computer.", "SLEEP_COMPUTER"),
        ("Sheep my computer.", "SLEEP_COMPUTER"),
        ("Sheep.", "SLEEP_COMPUTER"),
        ("Put my PC to sleep.", "SLEEP_COMPUTER"),
        ("Put Windows to sleep.", "SLEEP_COMPUTER"),
        ("Put the laptop to sleep.", "SLEEP_COMPUTER"),
        ("Go to sleep.", "SLEEP_COMPUTER"),
        ("Restart my computer.", "RESTART_COMPUTER"),
        ("Restart my PC.", "RESTART_COMPUTER"),
        ("Reboot Windows.", "RESTART_COMPUTER"),
        ("Reboot the system.", "RESTART_COMPUTER"),
        ("Shut down my computer.", "SHUTDOWN_COMPUTER"),
        ("Turn off my PC.", "SHUTDOWN_COMPUTER"),
        ("Power off my computer.", "SHUTDOWN_COMPUTER"),
        ("Turn my computer off.", "SHUTDOWN_COMPUTER"),
        ("Shut down.", "SHUTDOWN_COMPUTER"),
        ("Shed down computer.", "SHUTDOWN_COMPUTER"),
        ("Sherdown my PC.", "SHUTDOWN_COMPUTER"),
        ("Restart.", "RESTART_COMPUTER"),
        ("Power off.", "SHUTDOWN_COMPUTER"),
        ("Cancel shutdown.", "CANCEL_PENDING_POWER_ACTION"),
        ("Cancel restart.", "CANCEL_PENDING_POWER_ACTION"),
        ("Cancel sherdown.", "CANCEL_PENDING_POWER_ACTION"),
        ("Cancel shed down.", "CANCEL_PENDING_POWER_ACTION"),
        ("Abort the scheduled power action.", "CANCEL_PENDING_POWER_ACTION"),
        ("Show desktop.", "SHOW_DESKTOP"),
        ("Get all windows out of the way.", "SHOW_DESKTOP"),
        ("Close this window.", "CLOSE_ACTIVE_WINDOW"),
        ("Close the active window.", "CLOSE_ACTIVE_WINDOW"),
        ("Switch to the next window.", "SWITCH_NEXT_WINDOW"),
        ("Switch to the previous application.", "SWITCH_PREVIOUS_WINDOW"),
        ("Move to the next desktop.", "SWITCH_NEXT_VIRTUAL_DESKTOP"),
        ("Move to the previous desktop.", "SWITCH_PREVIOUS_VIRTUAL_DESKTOP"),
        ("Snap this window left.", "SNAP_ACTIVE_WINDOW_LEFT"),
        ("Snap this window right.", "SNAP_ACTIVE_WINDOW_RIGHT"),
        ("Minimize active window.", "MINIMIZE_ACTIVE_WINDOW"),
        ("Restore active window.", "RESTORE_ACTIVE_WINDOW"),
    ])
    def test_new_phrases(self, rec, text, expected):
        assert rec.recognize(text).name == expected, text

    @pytest.mark.parametrize("text", [
        "Hibernate.", "Sign out.", "Sign out of Windows.",
    ])
    def test_still_unsupported(self, rec, text):
        assert rec.recognize(text).name == "UNSUPPORTED_WINDOWS_ACTION"

    @pytest.mark.parametrize("text", [
        "Turn off the volume.", "Turn off the sound.", "Mute.",
        "Restart the download.", "go to sleep at ten",
        "I am going to sleep.", "Close Chrome.", "Switch to Chrome",
        "Minimize Chrome.", "Restore the window.", "What time is it?",
        # 2026-11-20 log near-misses: ordinary uses of the misheard
        # words must never become power intents.
        "The sheep are grazing.", "I need to shed the load.",
        "Cancel the meeting.", "For no.", "Shepard the sheep.",
    ])
    def test_no_false_positives(self, rec, text):
        name = rec.recognize(text).name
        assert name not in (
            "SLEEP_COMPUTER", "RESTART_COMPUTER", "SHUTDOWN_COMPUTER",
            "CANCEL_PENDING_POWER_ACTION",
        ), text

    def test_for_no_is_unknown_not_a_confirmation_word(self, rec):
        # Observed 2026-11-20: "For no" followed a restart ask. It is
        # neither a yes nor a no and must classify UNKNOWN so the
        # confirmation simply expires without executing anything.
        assert rec.recognize("For no").name == "UNKNOWN"
        assert rec.recognize("For no.").name == "UNKNOWN"


# ======================================================================
# Handler-level power tests
# ======================================================================
class TestPowerHandlers:
    def _handler(self, power, intent_name="SHUTDOWN_COMPUTER"):
        from src.commands.system_control_handlers import (
            RestartComputerHandler,
            ShutdownComputerHandler,
            SleepComputerHandler,
        )
        cls = {
            "SHUTDOWN_COMPUTER": ShutdownComputerHandler,
            "RESTART_COMPUTER": RestartComputerHandler,
            "SLEEP_COMPUTER": SleepComputerHandler,
        }[intent_name]
        return cls(_power_svc(power))

    @pytest.mark.parametrize("name", [
        "SHUTDOWN_COMPUTER", "RESTART_COMPUTER", "SLEEP_COMPUTER",
    ])
    def test_requires_confirmation_first(self, name):
        power = FakePower()
        h = self._handler(power, name)
        res = h.execute({})
        assert res.confirmation_required is True
        assert res.executed is False
        assert "yes" in res.message.lower()
        # Nothing scheduled/slept yet.
        assert power.calls == []

    @pytest.mark.parametrize("name,action", [
        ("SHUTDOWN_COMPUTER", "shutdown"),
        ("RESTART_COMPUTER", "restart"),
    ])
    def test_confirmed_defers_schedule_to_after_speech(self, name, action):
        power = FakePower()
        h = self._handler(power, name)
        ent = {}
        h.execute(ent)  # arms confirmation
        res = h.execute({**ent, "_confirmed": True})
        assert res.executed is True
        assert "15 seconds" in res.message
        assert "scheduled" not in res.message.lower().replace(
            "in 15 seconds", "")
        # Countdown must NOT begin until the ack finishes playing.
        assert ("schedule", action) not in power.calls
        cb = res.data["after_speech"]
        assert cb() is None
        assert ("schedule", action) in power.calls
        assert power.pending_action() == action

    def test_confirmed_sleep_runs_after_speech(self):
        power = FakePower()
        h = self._handler(power, "SLEEP_COMPUTER")
        ent = {}
        h.execute(ent)
        res = h.execute({**ent, "_confirmed": True})
        assert "sleep" in res.message.lower()
        assert "sleep" not in power.calls
        res.data["after_speech"]()
        assert "sleep" in power.calls

    def test_no_execution_without_confirmation(self):
        power = FakePower()
        h = self._handler(power)
        res = h.execute({"_confirmed": True})  # never armed
        assert res.success is False
        assert res.error == "confirmation_mismatch"
        assert power.calls == []

    def test_stale_mismatched_confirmation_rejected(self):
        power = FakePower()
        h = self._handler(power, "SHUTDOWN_COMPUTER")
        ent = {}
        h.execute(ent)  # arms "shutdown"
        # Simulate a confirmation smuggled for a different action.
        res = h.execute({**ent, "_confirmed": True,
                         "_armed_action": "restart"})
        assert res.success is False
        assert res.error == "confirmation_mismatch"
        assert power.calls == []

    def test_duplicate_request_refused(self):
        power = FakePower()
        h = self._handler(power, "RESTART_COMPUTER")
        ent = {}
        h.execute(ent)
        h.execute({**ent, "_confirmed": True}).data["after_speech"]()
        # Second request while countdown pending.
        res = h.execute({})
        assert res.success is False
        assert res.error == "power_action_pending"
        assert "already scheduled" in res.message

    def test_schedule_failure_reported_via_followup(self):
        power = FakePower(schedule_ok=False)
        h = self._handler(power, "SHUTDOWN_COMPUTER")
        ent = {}
        h.execute(ent)
        res = h.execute({**ent, "_confirmed": True})
        msg = res.data["after_speech"]()
        assert "couldn't schedule" in msg

    def test_sleep_failure_reported_via_followup(self):
        power = FakePower(sleep_ok=False)
        h = self._handler(power, "SLEEP_COMPUTER")
        ent = {}
        h.execute(ent)
        res = h.execute({**ent, "_confirmed": True})
        msg = res.data["after_speech"]()
        assert "couldn't" in msg


class TestCancelHandler:
    def test_cancel_with_nothing_pending(self):
        power = FakePower()
        h = CancelPowerActionHandler(_power_svc(power))
        res = h.execute({})
        assert res.success is False
        assert res.error == "nothing_pending"
        assert "no power action" in res.message

    def test_cancel_shutdown(self):
        power = FakePower()
        h = CancelPowerActionHandler(_power_svc(power))
        power.schedule("shutdown")
        res = h.execute({})
        assert res.success is True and res.executed is True
        assert "shutdown" in res.message
        assert power.pending_action() is None

    def test_cancel_restart(self):
        power = FakePower()
        h = CancelPowerActionHandler(_power_svc(power))
        power.schedule("restart")
        res = h.execute({})
        assert res.success is True
        assert "restart" in res.message

    def test_cancel_expired_pending(self):
        clock = FakeClock()
        power = FakePower(clock=clock)
        h = CancelPowerActionHandler(_power_svc(power))
        power.schedule("shutdown")
        clock.advance(SCHEDULE_DELAY_SECONDS + 1)
        res = h.execute({})
        assert res.success is False
        assert res.error == "nothing_pending"


# ======================================================================
# Session-level power flow (confirmation -> ack TTS -> countdown ->
# cancel while listening)
# ======================================================================
class TestPowerSessionFlow:
    def _setup(self, power=None):
        power = power or FakePower()
        registry = create_app_registry(_power_svc(power))
        return _controller(registry) + (power,)

    def test_full_shutdown_flow_with_cancel(self):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)

        stt.emit("Shut down my computer.")
        # Confirmation question spoken; finish it.
        assert any("shut down" in s for s in tts.spoken)
        tts.finish_next()

        stt.emit("yes")
        # Ack queued; countdown must NOT have started yet.
        assert ("schedule", "shutdown") not in power.calls
        tts.finish_next()  # ack finishes -> countdown arms now
        assert ("schedule", "shutdown") in power.calls
        assert power.pending_action() == "shutdown"

        # Mic is live again: cancel during the countdown.
        stt.emit("Cancel shutdown.")
        assert power.pending_action() is None
        assert any("cancel" in s.lower() for s in tts.spoken)
        tts.finish_next()

    def test_rejected_confirmation_never_schedules(self):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Restart my PC.")
        tts.finish_next()
        stt.emit("no")
        tts.finish_next()
        assert power.calls == []
        assert power.pending_action() is None

    def test_stale_confirmation_rejected(self):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Shut down.")
        tts.finish_next()
        # Age the pending confirmation past the 30 s window.
        ctl._pending_confirm["at"] -= 60
        stt.emit("yes")
        tts.finish_next()
        assert ("schedule", "shutdown") not in power.calls
        assert power.pending_action() is None

    def test_cancel_when_nothing_pending(self):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Cancel shutdown.")
        assert any("no power action" in s for s in tts.spoken)
        tts.finish_next()

    def test_sleep_executes_after_ack(self):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Put my PC to sleep.")
        tts.finish_next()
        stt.emit("yes")
        assert "sleep" not in power.calls
        tts.finish_next()
        assert "sleep" in power.calls

    def test_session_timeout_clears_pending_confirm(self):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Shut down my computer.")
        tts.finish_next()
        assert ctl._pending_confirm is not None
        # Force expiry and tick the session; drain TTS to finalize.
        session._last_activity_monotonic -= 120
        ctl.tick()
        while tts.spoken:
            tts.finish_next()
        assert not session.is_active
        stt.emit("yes")
        while tts.spoken:
            tts.finish_next()
        assert ("schedule", "shutdown") not in power.calls

    # -- Power-cancel while a confirmation is still pending ----------
    # 2026-11-20 log: "Cancel sherdown." arrived while Jarvis was
    # still asking "Should I shut down?" and classified UNKNOWN,
    # leaving the confirmation to expire silently. A cancel utterance
    # must now drop the pending confirmation and acknowledge it —
    # without ever dispatching or confirming the power intent.

    def _pending_power_flow(self, request, cancel_text):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit(request)
        tts.finish_next()  # confirmation ask spoken
        assert ctl._pending_confirm is not None
        stt.emit(cancel_text)
        return ctl, stt, tts, power

    def test_cancel_phrase_cancels_pending_shutdown_confirmation(self):
        ctl, stt, tts, power = self._pending_power_flow(
            "Shut down my computer.", "Cancel shutdown."
        )
        # Pending confirmation dropped; decline ack spoken; and the
        # CANCEL handler was never dispatched (nothing was scheduled,
        # so power.cancel() must not even run).
        assert ctl._pending_confirm is None
        assert any("won't shut" in s for s in tts.spoken)
        assert power.calls == []
        tts.finish_next()
        # A later "yes" cannot resurrect the cancelled request.
        stt.emit("yes")
        while tts.spoken:
            tts.finish_next()
        assert ("schedule", "shutdown") not in power.calls
        assert power.pending_action() is None

    def test_misheard_cancel_cancels_pending_shutdown_confirmation(self):
        # Exact 2026-11-20 transcript.
        ctl, stt, tts, power = self._pending_power_flow(
            "Shut down my computer.", "Cancel sherdown."
        )
        assert ctl._pending_confirm is None
        assert any("won't shut" in s for s in tts.spoken)
        assert power.calls == []
        tts.finish_next()

    def test_cancel_phrase_cancels_pending_restart_confirmation(self):
        ctl, stt, tts, power = self._pending_power_flow(
            "Restart my PC.", "Cancel restart."
        )
        assert ctl._pending_confirm is None
        assert any("won't restart" in s for s in tts.spoken)
        assert power.calls == []
        tts.finish_next()

    def test_misheard_sleep_request_still_requires_confirmation(self):
        # "Sheep my computer." (2026-11-20 transcript) maps to
        # SLEEP_COMPUTER but must still ask before acting.
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Sheep my computer.")
        assert any("sleep" in s.lower() for s in tts.spoken)
        assert power.calls == []
        tts.finish_next()
        stt.emit("no")
        tts.finish_next()
        assert power.calls == []

    def test_for_no_does_not_confirm_or_cancel_power_request(self):
        # 2026-11-20: "For no" followed the restart ask. It must not
        # confirm, must not cancel-via-new-path, and must not schedule.
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Restart my PC.")
        tts.finish_next()
        stt.emit("For no")
        while tts.spoken:
            tts.finish_next()
        assert ("schedule", "restart") not in power.calls
        assert "sleep" not in power.calls
        assert power.pending_action() is None
        # The pending confirmation was consumed by the unknown
        # utterance; a subsequent "yes" cannot execute it.
        stt.emit("yes")
        while tts.spoken:
            tts.finish_next()
        assert ("schedule", "restart") not in power.calls
        assert power.pending_action() is None

    # -- Cancel during the armed 15 s countdown ----------------------

    def test_misheard_cancel_aborts_scheduled_shutdown(self):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Shut down my computer.")
        tts.finish_next()
        stt.emit("yes")
        tts.finish_next()  # ack done -> countdown armed
        assert power.pending_action() == "shutdown"
        stt.emit("Cancel sherdown.")
        assert power.pending_action() is None
        assert any("cancel" in s.lower() for s in tts.spoken)
        tts.finish_next()

    def test_cancel_aborts_scheduled_restart(self):
        ctl, state, session, stt, tts, power = self._setup()
        _wake(ctl, stt, tts)
        stt.emit("Restart my PC.")
        tts.finish_next()
        stt.emit("yes")
        tts.finish_next()
        assert power.pending_action() == "restart"
        stt.emit("Cancel restart.")
        assert power.pending_action() is None
        assert any("cancel" in s.lower() for s in tts.spoken)
        tts.finish_next()


# ======================================================================
# Desktop control handler tests
# ======================================================================
class TestDesktopHandlers:
    def _registry(self, desktop=None):
        return create_app_registry(_power_svc(desktop=desktop))

    def test_show_desktop(self):
        desktop = FakeDesktop()
        # SHOW_DESKTOP uses svc.windows; give it a fake too.
        wm = FakeWindowManager(
            [WindowInfo(hwnd=1, pid=1, title="A")], 1
        )
        svc = _Svc(None, None, None, wm, desktop=desktop)
        reg = create_app_registry(svc)
        res = reg.dispatch(Intent("SHOW_DESKTOP", {}))
        assert res.success and res.executed

    def test_minimize_active(self):
        desktop = FakeDesktop()
        reg = self._registry(desktop)
        res = reg.dispatch(Intent("MINIMIZE_ACTIVE_WINDOW", {}))
        assert res.success and res.executed
        assert "minimize_active" in desktop.calls
        assert "Notepad" in res.message

    def test_minimize_missing_window(self):
        desktop = FakeDesktop()
        desktop.min_result = (False, "no_window", "")
        reg = self._registry(desktop)
        res = reg.dispatch(Intent("MINIMIZE_ACTIVE_WINDOW", {}))
        assert res.success is False
        assert "couldn't find" in res.message.lower()

    def test_restore_active(self):
        desktop = FakeDesktop()
        reg = self._registry(desktop)
        res = reg.dispatch(Intent("RESTORE_ACTIVE_WINDOW", {}))
        assert res.success and res.executed
        assert "restore_active" in desktop.calls

    def test_close_active_requires_confirmation(self):
        desktop = FakeDesktop()
        reg = self._registry(desktop)
        res = reg.dispatch(Intent("CLOSE_ACTIVE_WINDOW", {}))
        assert res.confirmation_required is True
        assert res.executed is False
        assert desktop.closed == []
        # The hwnd was captured for the confirmed dispatch.
        intent = Intent("CLOSE_ACTIVE_WINDOW", {})
        res1 = reg.dispatch(intent)
        assert intent.entities["_hwnd"] == 100

    def test_close_active_confirmed_closes_captured_hwnd(self):
        desktop = FakeDesktop()
        reg = self._registry(desktop)
        intent = Intent("CLOSE_ACTIVE_WINDOW", {})
        reg.dispatch(intent)  # arms + captures hwnd
        res = reg.dispatch(intent)  # same entities, now confirmed below
        # Simulate the session's yes: set _confirmed and re-dispatch.
        intent.entities["_confirmed"] = True
        res = reg.dispatch(intent)
        assert res.success and res.executed
        assert desktop.closed == [100]
        assert "Notepad" in res.message

    def test_close_active_window_gone(self):
        desktop = FakeDesktop()
        desktop.close_result = (False, "window_gone")
        reg = self._registry(desktop)
        intent = Intent("CLOSE_ACTIVE_WINDOW", {})
        reg.dispatch(intent)
        intent.entities["_confirmed"] = True
        res = reg.dispatch(intent)
        assert res.success is False
        assert "no longer open" in res.message

    def test_close_active_no_window(self):
        desktop = FakeDesktop(active=None)
        desktop.active = None
        reg = self._registry(desktop)
        res = reg.dispatch(Intent("CLOSE_ACTIVE_WINDOW", {}))
        assert res.success is False
        assert res.error == "no_window"

    @pytest.mark.parametrize("name,direction", [
        ("SWITCH_NEXT_WINDOW", "next"),
        ("SWITCH_PREVIOUS_WINDOW", "previous"),
    ])
    def test_switch_window(self, name, direction):
        desktop = FakeDesktop()
        reg = self._registry(desktop)
        res = reg.dispatch(Intent(name, {}))
        assert res.success and res.executed
        assert ("switch_window", direction) in desktop.calls

    def test_switch_window_failure(self):
        desktop = FakeDesktop()
        desktop.switch_result = (False, "no_other_window", "")
        reg = self._registry(desktop)
        res = reg.dispatch(Intent("SWITCH_NEXT_WINDOW", {}))
        assert res.success is False
        assert "no other window" in res.message

    @pytest.mark.parametrize("name,direction", [
        ("SWITCH_NEXT_VIRTUAL_DESKTOP", "next"),
        ("SWITCH_PREVIOUS_VIRTUAL_DESKTOP", "previous"),
    ])
    def test_switch_virtual_desktop(self, name, direction):
        desktop = FakeDesktop()
        reg = self._registry(desktop)
        res = reg.dispatch(Intent(name, {}))
        assert res.success and res.executed
        assert ("switch_desktop", direction) in desktop.calls

    def test_virtual_desktop_unavailable(self):
        desktop = FakeDesktop()
        desktop.desktop_ok = False
        reg = self._registry(desktop)
        res = reg.dispatch(Intent("SWITCH_NEXT_VIRTUAL_DESKTOP", {}))
        assert res.success is False
        assert "couldn't switch" in res.message

    @pytest.mark.parametrize("name,direction", [
        ("SNAP_ACTIVE_WINDOW_LEFT", "left"),
        ("SNAP_ACTIVE_WINDOW_RIGHT", "right"),
    ])
    def test_snap(self, name, direction):
        desktop = FakeDesktop()
        reg = self._registry(desktop)
        res = reg.dispatch(Intent(name, {}))
        assert res.success and res.executed
        assert ("snap", direction) in desktop.calls

    def test_snap_no_window(self):
        desktop = FakeDesktop()
        desktop.snap_result = (False, "no_window", "")
        reg = self._registry(desktop)
        res = reg.dispatch(Intent("SNAP_ACTIVE_WINDOW_LEFT", {}))
        assert res.success is False
        assert "couldn't find" in res.message

    def test_missing_desktop_service(self):
        reg = create_app_registry(_Svc(None, None, None, None))
        res = reg.dispatch(Intent("MINIMIZE_ACTIVE_WINDOW", {}))
        assert res.success is False
        assert res.error == "service_unavailable"


# ======================================================================
# DesktopController unit tests (fake WindowManager, fake chords)
# ======================================================================
class TestDesktopController:
    def _wins(self):
        return [
            WindowInfo(hwnd=10, pid=1, title="Top"),
            WindowInfo(hwnd=20, pid=2, title="Middle"),
            WindowInfo(hwnd=30, pid=3, title="Bottom"),
        ]

    def test_switch_next_wraps_zorder(self):
        wm = FakeWindowManager(self._wins(), fg_hwnd=20)
        combos = []
        dc = DesktopController(windows=wm,
                               send_combo=lambda c: combos.append(c) or True)
        ok, detail, title = dc.switch_window("next")
        assert ok and wm.focused == [30] and title == "Bottom"
        ok, detail, title = dc.switch_window("previous")
        assert ok and wm.focused[-1] == 10

    def test_switch_single_window_fails(self):
        wm = FakeWindowManager(self._wins()[:1], fg_hwnd=10)
        dc = DesktopController(windows=wm, send_combo=lambda c: True)
        ok, detail, _ = dc.switch_window("next")
        assert ok is False and detail == "no_other_window"

    def test_switch_unknown_foreground_fails(self):
        wm = FakeWindowManager(self._wins(), fg_hwnd=999)
        dc = DesktopController(windows=wm, send_combo=lambda c: True)
        ok, detail, _ = dc.switch_window("next")
        assert ok is False and detail == "no_active_window"

    def test_close_window_only_live_hwnd(self):
        wm = FakeWindowManager(self._wins(), fg_hwnd=20)
        dc = DesktopController(windows=wm, send_combo=lambda c: True)
        ok, detail = dc.close_window(999)
        assert ok is False and detail == "window_gone"
        assert wm.closed == []
        ok, detail = dc.close_window(20)
        assert ok is True and wm.closed == [20]

    def test_snap_sends_fixed_chord_only_with_window(self):
        combos = []
        wm = FakeWindowManager(self._wins(), fg_hwnd=20)
        dc = DesktopController(windows=wm,
                               send_combo=lambda c: combos.append(c) or True)
        ok, detail, title = dc.snap_active("left")
        assert ok and combos == ["win+left"]
        ok, detail, title = dc.snap_active("right")
        assert combos[-1] == "win+right"

    def test_snap_no_window_sends_nothing(self):
        combos = []
        wm = FakeWindowManager([], fg_hwnd=0)
        wm.foreground_window = lambda: None
        dc = DesktopController(windows=wm,
                               send_combo=lambda c: combos.append(c) or True)
        ok, detail, _ = dc.snap_active("left")
        assert ok is False and detail == "no_window"
        assert combos == []

    def test_virtual_desktop_chords(self):
        combos = []
        wm = FakeWindowManager(self._wins(), fg_hwnd=20)
        dc = DesktopController(windows=wm,
                               send_combo=lambda c: combos.append(c) or True)
        assert dc.switch_desktop("next") is True
        assert dc.switch_desktop("previous") is True
        assert combos == ["win+ctrl+right", "win+ctrl+left"]
        assert dc.switch_desktop("bogus") is False

    def test_combo_failure_reported(self):
        wm = FakeWindowManager(self._wins(), fg_hwnd=20)
        dc = DesktopController(windows=wm, send_combo=lambda c: False)
        ok, detail, _ = dc.snap_active("left")
        assert ok is False and detail == "combo_failed"
        assert dc.switch_desktop("next") is False
