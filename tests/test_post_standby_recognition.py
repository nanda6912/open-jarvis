"""V10.3 post-standby recognition integration tests.

End-to-end (mocked) checks that after a simulated suspend/resume the
pipeline reaches evidence-based readiness, the STT VAD state is reset
via the resume listener, and only fresh post-resume frames are
accepted as new commands.
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np
import pytest

from src.voice.speech_recognition_service import SpeechRecognitionService
from src.voice.voice_session_controller import VoiceSessionController
from tests.test_dynamic_mic import FakeSD, mic
from tests.test_mic_pipeline import FakeRecognizer
from tests.test_standby_watchdog import make_controller, pump

FRAME_N = 1600
SILENCE = np.zeros(FRAME_N, dtype=np.float32)
SPEECH = (np.random.RandomState(1).randn(FRAME_N)
          .astype(np.float32) * 0.08)


class TestResumeIntegration:
    def test_resume_listener_resets_stt_and_recovers(self):
        """Suspend -> resume: stale VAD state is reset, queue flushed,
        fresh callbacks verified, detector READY again."""
        sd = FakeSD([mic("Mic A", rate=16000.0)])
        ctl = make_controller(sd)
        transcripts = []
        stt = SpeechRecognitionService(
            ctl.capture, FakeRecognizer("open chrome"),
            speech_threshold=0.02, silence_hangover=0.15,
            min_utterance=0.05, frame_timeout=0.05,
            on_transcript=lambda t: transcripts.append(t),
        )
        ctl.set_resume_listener(stt.notify_system_resume)

        ctl.enable()
        try:
            stop = threading.Event()
            pt = threading.Thread(
                target=pump, args=(sd, 3600.0, 0.05, stop),
                daemon=True)
            pt.start()
            try:
                time.sleep(0.3)
                stt.start()
                time.sleep(0.2)

                # Seed stale post-suspend state as if captured pre-freeze.
                stt._noise_floor.extend([0.4] * 30)
                stt._effective_threshold = 0.019
                stt._collecting = True
                stt._utterance = [SPEECH]

                # Simulate an OS suspend/resume: wall-clock drift with
                # a real callback gap, then resume pumping.
                ctl._last_wall_time = time.time() - 600.0
                ctl._last_mono_time = time.monotonic()
                stop.set()          # callbacks stop (frozen process)
                pt.join(timeout=2)
                # Keep callbacks stopped until the drift candidate is
                # confirmed as a real suspend (stale pipeline required).
                deadline = time.monotonic() + 10.0
                while (time.monotonic() < deadline
                       and ctl._resume_events < 1):
                    time.sleep(0.05)
                assert ctl._resume_events >= 1
                stop2 = threading.Event()
                pt2 = threading.Thread(
                    target=pump, args=(sd, 3600.0, 0.05, stop2),
                    daemon=True)
                pt2.start()
                try:
                    deadline = time.monotonic() + 6.0
                    while time.monotonic() < deadline:
                        if ctl.audio_readiness().get("ready"):
                            break
                        time.sleep(0.05)

                    # STT VAD was reset by the resume listener; any
                    # repopulated noise floor contains only fresh
                    # post-resume frames (stale 0.4 entries gone).
                    assert stt._collecting is False
                    assert all(v < 0.1 for v in stt._noise_floor)
                    assert (stt._effective_threshold
                            == stt._speech_threshold)

                    # The wake-word detector enabled by ctl.enable()
                    # shares capture._queue with STT and can win the
                    # scheduling race for every pushed frame. It is not
                    # under test here — stop it so the frames below
                    # deterministically reach the recognizer.
                    det = getattr(ctl, "_detection", None)
                    if det is not None:
                        det.stop()
                        # Clearing the reference also stops the stall
                        # watchdog from recreating the detector.
                        ctl._detection = None
                    # A transcript may already have been produced by
                    # the seeded stale utterance during the freeze;
                    # clear it so only fresh post-resume speech counts.
                    transcripts.clear()

                    # Fresh speech after resume reaches the recognizer.
                    for _ in range(6):
                        ctl.capture._queue.put(SPEECH.copy())
                    for _ in range(6):
                        ctl.capture._queue.put(SILENCE.copy())
                    deadline = time.monotonic() + 4.0
                    while time.monotonic() < deadline and not transcripts:
                        time.sleep(0.02)
                    assert transcripts == ["open chrome"]
                    assert stt.stats()["session_frames"] > 0
                finally:
                    stop2.set()
                    pt2.join(timeout=2)
            finally:
                stt.stop()
        finally:
            ctl.shutdown()

    def test_no_stale_frames_after_resume(self):
        """Frames queued before the freeze must be flushed so they
        cannot be transcribed as a new command."""
        sd = FakeSD([mic("Mic A", rate=16000.0)])
        ctl = make_controller(sd)
        ctl.enable()
        try:
            pump(sd, 0.3)
            flush_calls = []
            orig_flush = ctl.capture.flush
            ctl.capture.flush = lambda: (
                flush_calls.append(1), orig_flush())
            # Freeze: wall clock jumps, callbacks stop. The 2s
            # candidate window must elapse with a stale pipeline.
            ctl._last_wall_time = time.time() - 900.0
            ctl._last_mono_time = time.monotonic()
            deadline = time.monotonic() + 8.0
            while (time.monotonic() < deadline
                   and ctl._resume_events < 1):
                time.sleep(0.05)
            assert ctl._resume_events >= 1
            pump(sd, 0.3)
            # The resume path must have flushed pre-freeze audio.
            assert flush_calls
        finally:
            ctl.shutdown()


class TestRecognitionLog:
    def _controller(self):
        c = VoiceSessionController.__new__(VoiceSessionController)
        c._recognition_log = collections.deque(maxlen=50)
        # Attributes read by _record_recognition that __init__ would
        # normally provide (bare-__new__ fixture).
        c._current_interaction = None
        c._current_command = None
        c._command_open = False
        return c

    def test_record_recognition_shape(self):
        c = self._controller()
        c._record_recognition(
            "open chrome", "executed", "OPEN_APPLICATION",
        )
        log = c.recognition_log()
        assert len(log) == 1
        entry = log[0]
        for key in ("wall", "monotonic", "text", "outcome",
                    "intent", "detail"):
            assert key in entry
        assert entry["outcome"] == "executed"
        assert entry["intent"] == "OPEN_APPLICATION"

    def test_log_is_bounded(self):
        c = self._controller()
        for i in range(60):
            c._record_recognition(f"cmd {i}", "executed", "X")
        assert len(c.recognition_log()) == 50
