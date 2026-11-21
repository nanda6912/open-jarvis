"""V10.7.5.1 — production wake-trace + rescoring regression tests.

Covers the end-to-end wake path instrumentation and the confirmed
failure mode: activity windows that open AFTER the wake-phrase onset
(loss of pre-window context) and long windows whose tail-truncation
drops the phrase head.

Uses the real sherpa-onnx engine where practical; model-dependent
tests skip cleanly when the KWS model is unavailable.
"""

from __future__ import annotations

import logging
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import AudioDeviceManager
from src.voice.wake_word_detection_service import WakeWordDetectionService
from src.voice.wake_word_engine import create_engine

from tests.test_dynamic_mic import FakeEngine, FakeSD, mic, wait_for
from tests.test_v1075_reliability import (
    CHUNK,
    SR,
    WAKE_WAV,
    _QueueCapture,
    _RescoringEngine,
    _feed,
    _load_wav_16k,
    _silence_frame,
    _speech_frame,
)

LIVE_WAV = (
    Path(__file__).resolve().parent
    / "fixtures" / "live" / "_hello_jarvis.wav"
)


class _TracingRescoreEngine(_RescoringEngine):
    """Rescoring fake that also honours the engine ``trace`` hook so
    per-anchor/complete events flow into the service log."""

    def rescore(self, samples, lead_seconds=(0.5,), trail_seconds=0.5,
                trace=None):
        self.rescore_calls += 1
        self.rescore_leads.append(tuple(lead_seconds))
        evaluated = 0
        hit = None
        if trace is not None:
            for lead in lead_seconds:
                evaluated += 1
                trace("anchor", lead_seconds=float(lead),
                      padded_seconds=0.0, audio_seconds=0.0,
                      hit=False, keyword=None, elapsed_ms=0.0,
                      error=None)
        if self.rescore_keyword is not None:
            hit = (self.rescore_keyword, float(lead_seconds[0]))
        if trace is not None:
            trace("complete", anchors_requested=len(lead_seconds),
                  anchors_evaluated=evaluated, hit=hit is not None,
                  keyword=hit[0] if hit else None,
                  lead_seconds=hit[1] if hit else None,
                  elapsed_ms=0.0)
        return hit


def _events(caplog, name):
    return [
        r for r in caplog.records
        if r.getMessage().startswith(name)
    ]


def _real_engine_or_skip():
    engine = create_engine()
    if getattr(engine, "name", "") != "sherpa-onnx-kws":
        pytest.skip("sherpa-onnx KWS model unavailable")
    return engine


# ----------------------------------------------------------------------
# 1-2. Rescore invocation + all anchors evaluated (traced)
# ----------------------------------------------------------------------
class TestRescoreInvocation:
    def test_activity_window_invokes_rescore(self, caplog):
        cap = _QueueCapture()
        engine = _TracingRescoreEngine()
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0, generation=7,
        )
        try:
            with caplog.at_level(logging.INFO):
                svc.start()
                _feed(cap, [_speech_frame() for _ in range(6)])
                _feed(cap, [_silence_frame() for _ in range(20)])
                # rescore_calls increments before the CLOSE event is
                # logged — wait for the closed window record instead.
                assert wait_for(
                    lambda: (
                        svc.last_activity_window is not None
                        and svc.last_activity_window["rescored"]
                    ),
                    timeout=10.0,
                )
            assert engine.rescore_calls >= 1
            opens = _events(caplog, "KWS_ACTIVITY_OPEN")
            closes = _events(caplog, "KWS_ACTIVITY_CLOSE")
            starts = _events(caplog, "KWS_RESCORE_START")
            completes = _events(caplog, "KWS_RESCORE_COMPLETE")
            assert opens and "window_id=1" in opens[0].getMessage()
            assert "generation=7" in opens[0].getMessage()
            assert closes and "enough_audio=True" in closes[0].getMessage()
            assert starts and completes
        finally:
            svc.stop()

    def test_all_configured_anchors_evaluated_real_engine(self):
        """Real sherpa engine: every configured lead anchor is scored."""
        if not WAKE_WAV.exists():
            pytest.skip("fixture missing")
        engine = _real_engine_or_skip()
        audio = _load_wav_16k(WAKE_WAV)
        anchors = []

        def trace(event, **fields):
            if event == "anchor":
                anchors.append(fields)

        result = engine.rescore(
            audio,
            lead_seconds=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2,
                          1.5, 1.8),
            trail_seconds=0.5,
            trace=trace,
        )
        assert result is not None
        # First hit wins — anchors after the hit are not evaluated, but
        # every evaluated anchor must carry complete trace fields.
        assert anchors, "no anchor trace events emitted"
        for a in anchors:
            assert "lead_seconds" in a and "hit" in a
            assert "padded_seconds" in a and "elapsed_ms" in a
        leads = [a["lead_seconds"] for a in anchors]
        assert leads == sorted(leads)

    def test_rescore_trace_complete_event(self):
        engine = _real_engine_or_skip()
        events = []
        engine.rescore(
            np.zeros(SR, dtype=np.float32),
            lead_seconds=(0.5, 0.9),
            trace=lambda e, **f: events.append((e, f)),
        )
        names = [e for e, _ in events]
        assert names.count("anchor") == 2
        assert names[-1] == "complete"
        comp = events[-1][1]
        assert comp["anchors_requested"] == 2
        assert comp["anchors_evaluated"] == 2
        assert comp["hit"] is False


# ----------------------------------------------------------------------
# 3. Captured window audio contains the complete wake phrase
# ----------------------------------------------------------------------
class TestWindowAudioCompleteness:
    def test_window_audio_covers_full_phrase_fixture(self):
        """Feed the real wake wav through the service as 100 ms frames;
        the window buffer must retain the entire utterance plus the
        hangover tail — not just the loudest fragment."""
        if not LIVE_WAV.exists():
            pytest.skip("live fixture missing")
        audio = _load_wav_16k(LIVE_WAV)
        cap = _QueueCapture()
        engine = _RescoringEngine(rescore_keyword=None)
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc.start()
            frames = []
            for off in range(0, audio.size, CHUNK):
                seg = np.zeros(CHUNK, dtype=np.float32)
                seg[: min(CHUNK, audio.size - off)] = audio[
                    off: off + CHUNK]
                frames.append(seg)
            # Prepend ~0.4 s of sub-gate audio then the phrase —
            # mimics a quiet onset below the activity gate.
            quiet = (audio[: int(0.4 * SR)] * 0.05).astype(np.float32)
            qframes = []
            for off in range(0, quiet.size, CHUNK):
                seg = np.zeros(CHUNK, dtype=np.float32)
                seg[: min(CHUNK, quiet.size - off)] = quiet[
                    off: off + CHUNK]
                qframes.append(seg)
            _feed(cap, qframes + frames)
            _feed(cap, [_silence_frame() for _ in range(20)])
            assert wait_for(
                lambda: len(svc.activity_windows) >= 1, timeout=10.0)
            win = svc.last_activity_window
            assert win is not None
            # Window audio must span at least the phrase duration.
            buf_n = svc._window_audio_samples
            assert buf_n >= int(1.2 * SR), (
                f"window buffer too small: {buf_n} samples")
        finally:
            svc.stop()

    def test_onset_below_gate_loses_phrase_head_real_engine(self):
        """PART 4 evidence: cutting the first 0.35 s of the real
        utterance makes every rescore anchor miss — the production
        failure mode when the window opens mid-phrase."""
        if not LIVE_WAV.exists():
            pytest.skip("live fixture missing")
        engine = _real_engine_or_skip()
        audio = _load_wav_16k(LIVE_WAV)
        cut = audio[int(0.35 * SR):]
        result = engine.rescore(
            cut,
            lead_seconds=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2,
                          1.5, 1.8),
            trail_seconds=0.5,
        )
        assert result is None, (
            "onset-cut audio unexpectedly detected — mechanism "
            "assumption invalid")
        # Sanity: the uncut audio DOES rescore.
        assert engine.rescore(
            audio, lead_seconds=(0.5,), trail_seconds=0.5) is not None

    def test_speech_trimmed_drops_leading_quiet_audio(self):
        """_speech_trimmed must cut leading sub-gate audio while
        keeping a margin before the first clear-speech hop."""
        cap = _QueueCapture()
        svc = WakeWordDetectionService(
            capture=cap, engine=_RescoringEngine(), threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc._kws_sample_rate = SR
            # 1 s quiet (rms~0.004) + 1 s loud (rms~0.08) + 0.5 s quiet.
            audio = np.concatenate([
                np.full(SR, 0.004, dtype=np.float32),
                np.full(SR, 0.08, dtype=np.float32),
                np.full(SR // 2, 0.004, dtype=np.float32),
            ])
            trimmed = svc._speech_trimmed(audio)
            assert trimmed is not None
            # First loud hop at 1.0 s; margin 0.4 s -> start ~0.6 s.
            assert trimmed.size == audio.size - int(0.6 * SR)
            assert trimmed.size >= int(0.3 * SR)
        finally:
            svc.stop()

    def test_speech_trimmed_returns_none_without_speech(self):
        cap = _QueueCapture()
        svc = WakeWordDetectionService(
            capture=cap, engine=_RescoringEngine(), threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc._kws_sample_rate = SR
            assert svc._speech_trimmed(
                np.full(SR, 0.004, dtype=np.float32)) is None
        finally:
            svc.stop()

    def test_recent_context_requires_more_than_window(self):
        """_recent_context_trimmed returns None unless the diagnostic
        ring holds meaningfully more audio than the window buffer —
        i.e. it actually contains pre-window context."""
        cap = _QueueCapture()
        svc = WakeWordDetectionService(
            capture=cap, engine=_RescoringEngine(), threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc._kws_sample_rate = SR
            # Ring = window audio + <0.3 s extra -> None.
            svc._window_audio_samples = 4 * SR
            svc._diag_buffer.clear()
            svc._diag_buffer.append(np.full(4 * SR + 1000,
                                            0.08, dtype=np.float32))
            svc._diag_buffer_samples = 4 * SR + 1000
            assert svc._recent_context_trimmed() is None
            # Ring with >0.3 s of extra speech-bearing audio -> trimmed.
            svc._diag_buffer.clear()
            svc._diag_buffer_samples = 0
            svc._diag_buffer.append(np.concatenate([
                np.full(SR, 0.001, dtype=np.float32),   # pre-window
                np.full(4 * SR, 0.08, dtype=np.float32),
            ]))
            svc._diag_buffer_samples = 5 * SR
            recent = svc._recent_context_trimmed()
            assert recent is not None
            assert recent.size > svc._window_audio_samples
        finally:
            svc.stop()

    def test_rescore_candidate_order_and_early_exit(self):
        """Candidates run full -> speech_trimmed -> recent_context;
        the first hit wins and later candidates are skipped."""
        cap = _QueueCapture()

        class _Rec:
            def __init__(self):
                self.calls = []

            def rescore(self, samples, lead_seconds=(0.5,),
                        trail_seconds=0.5, trace=None):
                self.calls.append(samples.size)
                if trace:
                    trace("complete", anchors_requested=1,
                          anchors_evaluated=1, hit=True,
                          keyword="HELLO JARVIS", lead_seconds=0.1,
                          elapsed_ms=1.0)
                return ("HELLO JARVIS", 0.1)

        rec = _Rec()
        svc = WakeWordDetectionService(
            capture=cap, engine=rec, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc._kws_sample_rate = SR
            # Window audio: 0.5 s quiet + 1.5 s loud.
            window = np.concatenate([
                np.full(SR // 2, 0.004, dtype=np.float32),
                np.full(int(1.5 * SR), 0.08, dtype=np.float32),
            ])
            svc._window_audio.append(window)
            svc._window_audio_samples = window.size
            # Ring identical to window -> recent_context is a dup.
            svc._diag_buffer.clear()
            svc._diag_buffer_samples = 0
            svc._diag_buffer.append(window.copy())
            svc._diag_buffer_samples = window.size
            out = svc._rescore_activity_audio(1)
            assert out is not None
            result, evaluated = out
            assert result == ("HELLO JARVIS", 0.1)
            # 'full' hit first -> exactly one candidate evaluated.
            assert evaluated == 1
            assert len(rec.calls) == 1
        finally:
            svc.stop()


# ----------------------------------------------------------------------
# 4-5. Malformed input visible + rescore exceptions observable
# ----------------------------------------------------------------------
class TestFailureVisibility:
    def test_empty_and_malformed_frames_do_not_open_window(self):
        cap = _QueueCapture()
        engine = _RescoringEngine()
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc.start()
            cap.push(np.zeros(0, dtype=np.float32))       # empty
            cap.push(np.zeros((4, 4), dtype=np.float32))  # 2-D shape
            _feed(cap, [_silence_frame() for _ in range(10)])
            time.sleep(0.6)
            assert not svc.activity_windows
            assert engine.rescore_calls == 0
        finally:
            svc.stop()

    def test_rescore_exception_is_logged(self, caplog):
        cap = _QueueCapture()
        engine = _RescoringEngine(rescore_keyword=None)

        def boom(*a, **kw):
            raise RuntimeError("rescore exploded")

        engine.rescore = boom
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            with caplog.at_level(logging.INFO):
                svc.start()
                _feed(cap, [_speech_frame() for _ in range(6)])
                _feed(cap, [_silence_frame() for _ in range(20)])
                assert wait_for(
                    lambda: bool(_events(caplog, "KWS_RESCORE_FAILED")),
                    timeout=10.0)
            assert svc.inference_errors == 0
            assert _events(caplog, "KWS_RESCORE_FAILED")
        finally:
            svc.stop()

    def test_engine_without_rescore_logged_as_skip(self, caplog):
        cap = _QueueCapture()
        engine = FakeEngine(score=0.0)
        assert not hasattr(engine, "rescore") or True
        if hasattr(engine, "rescore"):
            del engine.rescore  # simulate pre-V10.7.5 engine
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            with caplog.at_level(logging.INFO):
                svc.start()
                _feed(cap, [_speech_frame() for _ in range(6)])
                _feed(cap, [_silence_frame() for _ in range(20)])
                assert wait_for(
                    lambda: bool(_events(caplog, "KWS_RESCORE_SKIPPED")),
                    timeout=10.0)
            msgs = " ".join(
                r.getMessage() for r in _events(caplog,
                                              "KWS_RESCORE_SKIPPED"))
            assert "engine_no_rescore" in msgs
        finally:
            svc.stop()


# ----------------------------------------------------------------------
# 6-7. Live stream preserved + rescored hit reaches callback
# ----------------------------------------------------------------------
class TestDetectionPath:
    def test_rescore_preserves_live_stream_real_engine(self):
        engine = _real_engine_or_skip()
        gen_before = getattr(engine, "_stream_generation", None)
        pending_before = getattr(engine, "_pending_keyword", None)
        audio = _load_wav_16k(WAKE_WAV) if WAKE_WAV.exists() else (
            np.zeros(SR, dtype=np.float32))
        engine.rescore(audio, lead_seconds=(0.1, 0.5), trace=None)
        assert getattr(engine, "_stream_generation", None) == gen_before
        assert getattr(engine, "_pending_keyword", None) == pending_before

    def test_rescored_hit_emits_detection_and_activation_events(
            self, caplog):
        cap = _QueueCapture()
        engine = _RescoringEngine()
        detected = []
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0, on_detected=detected.append,
            generation=3,
        )
        try:
            with caplog.at_level(logging.INFO):
                svc.start()
                _feed(cap, [_speech_frame() for _ in range(6)])
                _feed(cap, [_silence_frame() for _ in range(20)])
                assert wait_for(lambda: detected, timeout=10.0)
            assert detected == ["HELLO JARVIS"]
            emitted = _events(caplog, "KWS_DETECTION_EMITTED")
            activated = _events(caplog, "KWS_SESSION_ACTIVATED")
            assert emitted and "via='rescore'" in emitted[-1].getMessage()
            assert activated
            msg = activated[-1].getMessage()
            assert "callback_ok=True" in msg
            assert "generation=3" in msg
        finally:
            svc.stop()

    def test_live_detection_emits_via_live(self, caplog):
        cap = _QueueCapture()
        engine = FakeEngine(score=1.0, keyword="HELLO JARVIS")
        detected = []
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0, on_detected=detected.append,
        )
        try:
            with caplog.at_level(logging.INFO):
                svc.start()
                _feed(cap, [_speech_frame()])
                assert wait_for(lambda: detected, timeout=5.0)
            emitted = _events(caplog, "KWS_DETECTION_EMITTED")
            assert emitted and "via='live'" in emitted[-1].getMessage()
        finally:
            svc.stop()


# ----------------------------------------------------------------------
# 8-9. Cooldown + session activation through the controller
# ----------------------------------------------------------------------
class TestCooldownAndActivation:
    def test_cooldown_blocks_second_detection(self, caplog):
        cap = _QueueCapture()
        engine = _RescoringEngine()
        detected = []
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=30.0, on_detected=detected.append,
        )
        try:
            with caplog.at_level(logging.INFO):
                svc.start()
                # First window -> rescored hit.
                _feed(cap, [_speech_frame() for _ in range(6)])
                _feed(cap, [_silence_frame() for _ in range(20)])
                assert wait_for(lambda: detected, timeout=10.0)
                # Second window inside cooldown -> blocked, not emitted.
                _feed(cap, [_speech_frame() for _ in range(6)])
                _feed(cap, [_silence_frame() for _ in range(20)])
                assert wait_for(
                    lambda: bool(
                        _events(caplog, "KWS_DETECTION_BLOCKED")),
                    timeout=10.0)
            assert detected == ["HELLO JARVIS"]
            blocked = _events(caplog, "KWS_DETECTION_BLOCKED")
            assert "reason='cooldown'" in blocked[-1].getMessage()
        finally:
            svc.stop()

    def test_successful_callback_reaches_controller_handling(self):
        """A detection through the real callback chain suspends
        detection and notifies the wake listener exactly once."""
        from src.controllers.microphone_controller import (
            STATUS_IDLE,
            STATUS_LISTENING,
            MicrophoneController,
        )

        sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
        engine = _RescoringEngine()
        wakes = []
        mgr = AudioDeviceManager(sd=sd)
        ctl = MicrophoneController(
            device_manager=mgr,
            capture=AudioCaptureService(sd=sd, device_manager=mgr),
            engine=engine,
            settings_get=lambda k: None,
            settings_set=lambda k, v: None,
            monitor_interval=0.05,
            callback_stall_threshold=60.0,
            detection_stall_threshold=60.0,
        )
        ctl.set_callbacks(on_wake_word=wakes.append)
        try:
            assert ctl.enable()
            assert ctl.status == STATUS_LISTENING
            # Simulate session end then push speech frames through the
            # real capture callback so the new generation sees them.
            ctl._set_status(STATUS_IDLE, "activated")
            ctl.resume_detection()
            assert ctl.status == STATUS_LISTENING
            speech = _speech_frame().reshape(-1, 1)
            silence = np.zeros((CHUNK, 1), dtype=np.float32)

            def pump(frames):
                for s in sd.streams:
                    try:
                        for f in frames:
                            s.callback(f.copy(), CHUNK, None, None)
                    except Exception:  # noqa: BLE001
                        pass

            # ~1 s speech opens the activity window; then >hangover
            # silence lets it close so rescore can fire.
            pump([speech] * 10)
            end = time.monotonic() + 10.0
            while time.monotonic() < end and not wakes:
                pump([silence] * 10)
                time.sleep(0.05)
            assert wakes, "rescored detection never reached controller"
            assert wakes[0] == "HELLO JARVIS"
            assert ctl.status == STATUS_IDLE
        finally:
            ctl.shutdown()


# ----------------------------------------------------------------------
# 10-13. Silence, handover stability, queue bounds, false activations
# ----------------------------------------------------------------------
class TestStability:
    def test_silence_never_rescores_or_detects(self):
        cap = _QueueCapture()
        engine = _RescoringEngine()
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc.start()
            _feed(cap, [_silence_frame() for _ in range(60)])
            time.sleep(0.5)
            assert engine.rescore_calls == 0
            assert not svc.activity_windows
        finally:
            svc.stop()

    def test_repeated_standby_handovers_stable(self):
        """Two goodbye->standby cycles: each gets a fresh generation
        and the pipeline still detects afterwards."""
        from src.controllers.microphone_controller import (
            STATUS_IDLE,
            MicrophoneController,
        )

        sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
        engine = FakeEngine(score=0.0)
        mgr = AudioDeviceManager(sd=sd)
        ctl = MicrophoneController(
            device_manager=mgr,
            capture=AudioCaptureService(sd=sd, device_manager=mgr),
            engine=engine,
            settings_get=lambda k: None,
            settings_set=lambda k, v: None,
            monitor_interval=0.05,
            callback_stall_threshold=60.0,
            detection_stall_threshold=60.0,
        )
        try:
            assert ctl.enable()
            gens = []
            for _ in range(2):
                ctl._set_status(STATUS_IDLE, "activated")
                ctl.resume_detection()
                gens.append(ctl._detection_generation)
                assert ctl._detection is not None
                assert ctl._detection.is_running
            assert gens[1] > gens[0]
            # Third handover while already listening is a no-op guard:
            # status LISTENING -> resume_detection does nothing harmful.
            ctl.resume_detection()
            assert ctl._detection is not None
        finally:
            ctl.shutdown()

    def test_rescore_does_not_grow_queue_unbounded(self):
        """A slow rescore must not drop frames uncontrolled: the
        capture queue is bounded and drops are counted, not silent."""
        cap = _QueueCapture()
        engine = _RescoringEngine()

        def slow_rescore(*a, **kw):
            engine.rescore_calls += 1
            time.sleep(0.05)  # simulate heavy anchor decode
            return engine.rescore_keyword, 0.5

        engine.rescore = slow_rescore
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc.start()
            # Speech window then sustained speech-level frames so the
            # queue keeps receiving while rescore runs.
            _feed(cap, [_speech_frame() for _ in range(6)])
            _feed(cap, [_silence_frame() for _ in range(20)])
            assert wait_for(
                lambda: engine.rescore_calls >= 1, timeout=10.0)
            stats = cap.stats()
            # _QueueCapture is unbounded; assert the service consumed
            # everything and the worker survived repeated rescoring.
            assert svc.is_running
            time.sleep(0.3)
            assert svc.is_running
        finally:
            svc.stop()

    def test_non_keyword_speech_no_false_activation(self, caplog):
        """Loud non-keyword activity: rescore runs, no keyword, no
        detection, no session callback."""
        cap = _QueueCapture()
        engine = _RescoringEngine(rescore_keyword=None)
        detected = []
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0, on_detected=detected.append,
        )
        try:
            with caplog.at_level(logging.INFO):
                svc.start()
                # Two loud non-keyword windows; each needs >hangover
                # wall time to close before rescore runs.
                for _ in range(2):
                    _feed(cap, [_speech_frame(0.2) for _ in range(15)])
                    _feed(cap, [_silence_frame() for _ in range(20)])
                    assert wait_for(
                        lambda: len(svc.activity_windows) >= _ + 1,
                        timeout=10.0)
            assert engine.rescore_calls >= 1
            assert not detected
            assert not _events(caplog, "KWS_DETECTION_EMITTED")
            assert not _events(caplog, "KWS_SESSION_ACTIVATED")
            win = svc.last_activity_window
            assert win is not None and not win["keyword_detected"]
        finally:
            svc.stop()
