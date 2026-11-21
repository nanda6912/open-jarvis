"""V10.8 Task 1 — diagnostics/telemetry correctness regressions.

Covers only observability fixes:

* ``KWS_ACTIVITY_CLOSE`` rescore lifecycle aggregation (attempted /
  succeeded / candidates / anchors / duration / detection method).
* STT empty-utterance RMS min/max/mean/median from one sorted snapshot.
* Voice-session interaction timing with distinct per-phase monotonic
  timestamps and session/command correlation ids.
* Explicit lifetime vs detector-generation vs activity-window counters.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.speech_recognition_service import SpeechRecognitionService
from src.voice.voice_session_controller import VoiceSessionController
from src.voice.wake_word_detection_service import WakeWordDetectionService

from tests.test_dynamic_mic import FakeEngine, wait_for
from tests.test_v10751_production_trace import _TracingRescoreEngine, _events
from tests.test_v1075_reliability import (
    SR,
    _QueueCapture,
    _feed,
    _silence_frame,
    _speech_frame,
)
from tests.test_vad_segmentation_v105 import _FakeRecognizer, _frame, _service


# ----------------------------------------------------------------------
# KWS rescore lifecycle aggregation
# ----------------------------------------------------------------------
def _window_ready_service(engine, generation=3):
    cap = _QueueCapture()
    svc = WakeWordDetectionService(
        capture=cap,
        engine=engine,
        threshold=0.25,
        cooldown_seconds=0.0,
        generation=generation,
    )
    svc._kws_sample_rate = SR
    return svc


def _prime_window(svc, seconds=0.8):
    n = int(seconds * SR)
    svc._window_audio.append(np.full(n, 0.05, dtype=np.float32))
    svc._window_audio_samples = n
    svc._diag_buffer.clear()
    svc._diag_buffer_samples = 0


def test_rescore_report_aggregates_miss_candidates_and_anchors():
    engine = _TracingRescoreEngine(rescore_keyword=None)
    svc = _window_ready_service(engine)
    try:
        _prime_window(svc)
        out = svc._rescore_activity_audio(1)
        assert out is None
        report = svc._last_rescore_report
        assert report["attempted"] is True
        assert report["succeeded"] is False
        assert report["candidates_attempted"] == 1
        assert report["candidates_completed"] == 1
        assert report["anchors_requested"] == len(svc._rescore_lead_seconds)
        assert report["anchors_evaluated"] == len(svc._rescore_lead_seconds)
        assert report["anchors_evaluated_complete"] is True
        assert report["elapsed_ms"] is not None
        assert report["error"] is None
        assert report["hit_candidate"] is None
    finally:
        svc.stop()


def test_rescore_report_aggregates_hit_candidate():
    engine = _TracingRescoreEngine()  # hits on first candidate
    svc = _window_ready_service(engine)
    try:
        _prime_window(svc)
        out = svc._rescore_activity_audio(1)
        assert out is not None
        result, completed = out
        assert result == ("HELLO JARVIS", 0.1)
        assert completed == 1
        report = svc._last_rescore_report
        assert report["succeeded"] is True
        assert report["hit_candidate"] == "full"
        assert report["candidates_attempted"] == 1
        assert report["candidates_completed"] == 1
        assert report["anchors_evaluated"] == len(svc._rescore_lead_seconds)
    finally:
        svc.stop()


def test_activity_close_reports_rescore_miss_truthfully(caplog):
    cap = _QueueCapture()
    engine = _TracingRescoreEngine(rescore_keyword=None)
    svc = WakeWordDetectionService(
        capture=cap, engine=engine, threshold=0.25,
        cooldown_seconds=0.0, generation=3,
    )
    try:
        with caplog.at_level(logging.INFO):
            svc.start()
            _feed(cap, [_speech_frame() for _ in range(6)])
            _feed(cap, [_silence_frame() for _ in range(20)])
            assert wait_for(
                lambda: svc.last_activity_window is not None,
                timeout=10.0,
            )
            # V10.8: rescore runs on the dedicated worker; the window
            # record's rescore_* fields settle when the detection
            # worker drains the completed job.
            assert wait_for(
                lambda: (
                    svc.last_activity_window is not None
                    and svc.last_activity_window.get(
                        "rescore_succeeded") is not None
                ),
                timeout=10.0,
            )
        win = svc.last_activity_window
        assert win["rescored"] is False
        assert win["rescore_attempted"] is True
        assert win["rescore_succeeded"] is False
        assert win["rescore_candidates_attempted"] >= 1
        assert win["rescore_candidates_completed"] == (
            win["rescore_candidates_attempted"]
        )
        assert win["rescore_anchors_requested"] > 0
        assert win["rescore_anchors_evaluated"] == (
            win["rescore_anchors_requested"]
        )
        assert win["rescore_elapsed_ms"] is not None
        assert win["detection_method"] == "none"
        assert win["window_uid"] == "gen3-win1"
        close = _events(caplog, "KWS_ACTIVITY_CLOSE")[-1].getMessage()
        assert "rescore_attempted=True" in close
        assert "rescore_scheduled=True" in close
        assert "detection_method='rescore_pending'" in close
        assert "window_uid='gen3-win1'" in close
        result = _events(caplog, "KWS_RESCORE_RESULT")[-1].getMessage()
        assert "rescore_succeeded" not in result  # fields are flat
        assert "succeeded=False" in result
        assert "candidates_attempted=1" in result
        assert "candidates_completed=1" in result
        assert "detection_method='none'" in result
        assert "window_uid='gen3-win1'" in result
    finally:
        svc.stop()


def test_activity_close_reports_rescore_hit_method(caplog):
    cap = _QueueCapture()
    engine = _TracingRescoreEngine()
    svc = WakeWordDetectionService(
        capture=cap, engine=engine, threshold=0.25,
        cooldown_seconds=0.0, generation=4,
    )
    try:
        with caplog.at_level(logging.INFO):
            svc.start()
            _feed(cap, [_speech_frame() for _ in range(6)])
            _feed(cap, [_silence_frame() for _ in range(20)])
            assert wait_for(
                lambda: svc.last_activity_window is not None
                and svc.last_activity_window["rescored"],
                timeout=10.0,
            )
        win = svc.last_activity_window
        assert win["rescore_attempted"] is True
        assert win["rescore_succeeded"] is True
        assert win["rescore_hit_candidate"] == "full"
        assert win["detection_method"] == "rescore"
        result = _events(caplog, "KWS_RESCORE_RESULT")[-1].getMessage()
        assert "succeeded=True" in result
        assert "detection_method='rescore'" in result
    finally:
        svc.stop()


def test_activity_close_skip_reports_not_attempted(caplog):
    cap = _QueueCapture()
    svc = WakeWordDetectionService(
        capture=cap, engine=FakeEngine(score=0.0), threshold=0.25,
        cooldown_seconds=0.0, generation=5,
    )
    try:
        with caplog.at_level(logging.INFO):
            svc.start()
            _feed(cap, [_speech_frame() for _ in range(6)])
            _feed(cap, [_silence_frame() for _ in range(20)])
            assert wait_for(
                lambda: svc.last_activity_window is not None,
                timeout=10.0,
            )
        win = svc.last_activity_window
        assert win["rescore_attempted"] is False
        assert win["rescore_candidates_attempted"] == 0
        assert win["rescore_candidates_completed"] == 0
        assert win["detection_method"] == "none"
        assert _events(caplog, "KWS_RESCORE_SKIPPED")
    finally:
        svc.stop()


# ----------------------------------------------------------------------
# STT RMS snapshot correctness
# ----------------------------------------------------------------------
def test_stt_empty_log_uses_sorted_rms_snapshot(caplog):
    rec = _FakeRecognizer("")
    svc, _cap, _rec = _service(recognizer=rec)
    try:
        with caplog.at_level(logging.INFO):
            # Chronological RMS: high -> low -> medium -> silence. The old
            # code logged first/last as min/max (inverted here).
            svc._process_frame(_frame(0.08, seed=1))
            svc._process_frame(_frame(0.02, seed=2))
            svc._process_frame(_frame(0.05, seed=3))
            svc._last_voiced = time.monotonic() - 1.0
            svc._process_frame(_frame(0.0005, seed=4))
        assert rec.calls, "empty-decode log path was not reached"
        msgs = [
            r.getMessage() for r in caplog.records
            if "No speech recognized in utterance" in r.getMessage()
        ]
        assert msgs, "empty-utterance diagnostic was not logged"
        msg = msgs[-1]
        assert "rms_min=0.0005" in msg
        assert "rms_max=0.0800" in msg
        assert "rms_mean=0.0376" in msg
        assert "rms_med=0.0500" in msg
        # Onset frame counts as voiced: 3 voiced frames, 1 rejected.
        assert "voiced=3" in msg
        assert "rejected=1" in msg
    finally:
        svc.stop()


# ----------------------------------------------------------------------
# Interaction timing phases + correlation ids
# ----------------------------------------------------------------------
class _ManualSTT:
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

    def begin(self, text=None):
        self._cb["on_started"](text or self.spoken[-1])

    def end(self, text=None):
        self._cb["on_finished"](text or self.spoken[-1])


def _phase_controller(ack_mode="quick"):
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    stt, tts = _ManualSTT(), _ManualTTS()
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=SessionManager(timeout_seconds=60),
        speech_recognition=stt,
        speech_output=tts,
        microphone_controller=None,
        ack_mode=ack_mode,
        quick_ack_response="Yes.",
    )
    return ctl, stt, tts


def _assert_nondecreasing(rec, keys):
    values = [rec.get(k) for k in keys]
    present = [v for v in values if v is not None]
    assert len(present) == len(values), f"missing phase timestamps: {keys}"
    assert present == sorted(present), f"phase order violated: {keys}"


def test_interaction_timing_uses_distinct_phase_timestamps():
    ctl, stt, tts = _phase_controller()
    try:
        ctl.handle_wake_word()
        rec = ctl._current_interaction
        assert rec["session_id"].startswith("session-")
        tts.begin("Yes.")
        time.sleep(0.005)
        tts.end("Yes.")
        stt.emit("What time is it?")
        response = tts.spoken[-1]
        tts.begin(response)
        tts.end(response)
        ctl._end_with_goodbye()
        tts.begin("Goodbye.")
        tts.end("Goodbye.")

        timings = ctl.interaction_timings()
        assert len(timings) == 1
        rec = timings[0]
        _assert_nondecreasing(rec, [
            "wake_detected_monotonic",
            "activated_monotonic",
            "ack_tts_started_monotonic",
            "ack_tts_finished_monotonic",
            "first_transcript_monotonic",
            "first_command_processing_started_monotonic",
            "first_command_processing_finished_monotonic",
            "first_response_tts_started_monotonic",
            "first_response_tts_finished_monotonic",
            "goodbye_detected_monotonic",
            "farewell_tts_started_monotonic",
            "farewell_tts_finished_monotonic",
            "standby_restored_monotonic",
        ])
        # Legacy single-TTS fields refer to the FIRST spoken segment
        # (the ack), not the last TTS before standby.
        assert rec["tts_started_monotonic"] == rec[
            "ack_tts_started_monotonic"
        ]
        assert rec["tts_finished_monotonic"] == rec[
            "ack_tts_finished_monotonic"
        ]
        assert rec["tts_playback_seconds"] == rec["ack_playback_seconds"]
        assert rec["command_count"] == 1
        cmd = rec["commands"][0]
        assert cmd["session_id"] == rec["session_id"]
        assert cmd["command_id"].startswith(rec["session_id"])
        assert cmd["response_tts_started_monotonic"] == rec[
            "first_response_tts_started_monotonic"
        ]
        assert cmd["processing_seconds"] is not None
        assert cmd["response_playback_seconds"] is not None
        logged = ctl.recognition_log()
        command_entries = [e for e in logged if e.get("command_id")]
        assert command_entries
        assert command_entries[-1]["session_id"] == rec["session_id"]
        assert command_entries[-1]["command_id"] == cmd["command_id"]
    finally:
        ctl.shutdown()


# ----------------------------------------------------------------------
# Counter scope separation
# ----------------------------------------------------------------------
def test_kws_stats_separate_lifetime_generation_and_window_counters():
    cap = _QueueCapture()
    svc1 = WakeWordDetectionService(
        capture=cap, engine=FakeEngine(score=0.0), threshold=0.25,
        cooldown_seconds=0.0, generation=1,
    )
    svc2 = WakeWordDetectionService(
        capture=cap, engine=FakeEngine(score=0.0), threshold=0.25,
        cooldown_seconds=0.0, generation=2,
    )
    try:
        assert svc1._window_uid(7) == "gen1-win7"
        assert svc2._window_uid(7) == "gen2-win7"
        stats = svc1.stats()
        assert stats["detector_generation"] == 1
        assert stats["generation_frames_processed"] == (
            stats["frames_processed"]
        )
        assert "lifetime_frames_captured" in stats
        assert "activity_window_seq" in stats
    finally:
        svc1.stop()
        svc2.stop()
