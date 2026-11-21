"""V10.7.4.1 KWS parity diagnostic reliability tests.

Covers the empty-``windows`` instrumentation defect:

* diagnostic_status health section (enabled/attached/frames/jobs);
* partial-window preservation across suspend/resume/shutdown;
* deferred finalize so frame-450 bookkeeping is captured;
* single-worker analysis with explicit pending/completed/failed states;
* atomic report checkpoints independent of the process CWD;
* attachment to the real WakeWordDetectionService + engine factory.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src.voice import kws_parity_diagnostic as diag
from src.voice.wake_word_detection_service import (
    WakeWordDetectionService,
)

SR = 16000
FRAME = 1600  # 100 ms of 16 kHz mono — one KWS input frame
REAL_REPORT_PATH = diag.REPORT_PATH


class FakeCapture:
    """Frame source double; each queued item is one KWS frame."""

    sample_rate = SR
    input_channels = 4
    device = None
    paused = False

    def __init__(self):
        self._queue = queue.Queue()
        self.frames_dropped = 0
        self.status_warnings = 0
        self.overflow_callbacks = 0
        self.channel_switch_count = 0

    def push_frames(self, n, level=0.001):
        block = (level * np.ones(FRAME, dtype=np.float32))
        for _ in range(n):
            self._queue.put(block.copy())

    def get_frame(self, timeout=0.5):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self):
        return {
            "sample_rate": SR,
            "input_channels": 4,
            "selected_channel": 0,
            "queue_depth": self._queue.qsize(),
            "frames_captured": 0,
            "frames_dropped": self.frames_dropped,
            "overflow_callbacks": self.overflow_callbacks,
            "status_warnings": self.status_warnings,
            "channel_switch_count": self.channel_switch_count,
        }


class FakeEngine:
    name = "fake-kws"
    keyword_count = 6
    keywords_sha256 = "abc"
    keywords_file = None
    inference_ms_ema = 1.0
    _stream_generation = 0

    def __init__(self):
        self.inferences = 0
        self.reset_count = 0
        self._pending = None

    def process(self, samples):
        self.inferences += 1
        return 0.0

    def keyword_detected(self):
        kw, self._pending = self._pending, None
        return kw

    def reset(self):
        self.reset_count += 1
        self._stream_generation += 1

    def close(self):
        pass


def _wait_for(cond, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def _service(cap, engine):
    return WakeWordDetectionService(
        cap, engine, threshold=0.25, cooldown_seconds=0.0)


def _read_report(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _wait_report(path, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rep = _read_report(path)
        if rep is not None:
            return rep
        time.sleep(0.05)
    return None


def _wait_report_where(path, pred, timeout=15.0):
    """Poll the report until pred(report) is truthy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rep = _read_report(path)
        if rep is not None and pred(rep):
            return rep
        time.sleep(0.05)
    return _read_report(path)


@pytest.fixture
def diag_on(tmp_path, monkeypatch):
    """Enabled diagnostic writing to an isolated report file."""
    report = tmp_path / "parity.json"
    monkeypatch.setenv(diag.ENV_FLAG, "1")
    diag.reset_for_tests()
    monkeypatch.setattr(diag, "REPORT_PATH", report)
    rec = diag.recorder
    yield rec, report
    try:
        rec._supersede()
    except Exception:
        pass


@pytest.fixture
def diag_off(monkeypatch):
    monkeypatch.delenv(diag.ENV_FLAG, raising=False)
    diag.reset_for_tests()
    yield diag.recorder
    try:
        diag.recorder._supersede()
    except Exception:
        pass


def _patch_fast_analysis(monkeypatch, rec, result=None):
    """Replace the heavy replay/STT analysis with a fast stub.

    NOTE: set as an instance attribute, so it is NOT bound — the
    worker calls it with (widx, frames, audio, events).
    """
    state = {"calls": 0}

    def fake(widx, frames, audio, events):
        state["calls"] += 1
        return dict(result or {"conclusion": "test-analysis"})

    monkeypatch.setattr(rec, "_analyze_window", fake)
    return state


# ----------------------------------------------------------------------
# Test A — diagnostic disabled
# ----------------------------------------------------------------------
def test_a_disabled_records_nothing(diag_off, tmp_path, monkeypatch):
    report = tmp_path / "parity.json"
    monkeypatch.setattr(diag, "REPORT_PATH", report)
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(60)
        assert _wait_for(lambda: svc.frames_processed >= 60, 5)
    finally:
        svc.stop()
    assert diag.recorder._frames_recorded == 0
    assert diag.recorder._frames_seen == 0
    assert not report.exists()
    assert svc.frames_processed >= 60  # normal KWS behaviour preserved


# ----------------------------------------------------------------------
# Test B — enabled, 50 frames: partial-window status persisted
# ----------------------------------------------------------------------
def test_b_partial_window_status(diag_on):
    rec, report = diag_on
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(50)
        assert _wait_for(
            lambda: rec._frames_recorded >= 50, 5)
        rep = _wait_report_where(
            report,
            lambda r: r["diagnostic_status"]["current_window_frames"]
            >= 50)
        assert rep is not None
        st = rep["diagnostic_status"]
        assert st["enabled"] is True
        assert st["collection_state"] == "collecting"
        assert st["current_window_frames"] == 50
        assert st["required_window_frames"] == 450
        assert st["frames_seen"] == 50
        assert st["frames_recorded"] == 50
        assert rep["active_window"]["frames"] == 50
        # No completed analysis may be claimed for a partial window.
        assert all(
            w.get("window", {}).get("frames") != 50
            or w.get("status") != "analyzed"
            for w in rep["windows"]
        )
    finally:
        svc.stop()


# ----------------------------------------------------------------------
# Test C — exactly 450 frames: one window, one analysis job
# ----------------------------------------------------------------------
def test_c_full_window_single_analysis(diag_on, monkeypatch):
    rec, report = diag_on
    state = _patch_fast_analysis(monkeypatch, rec)
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(450)
        assert _wait_for(lambda: state["calls"] == 1, 10)
        rep = _wait_report_where(
            report,
            lambda r: any(
                w.get("status") == "analyzed" for w in r["windows"]))
        assert rep is not None
        done = [
            w for w in rep["windows"]
            if w.get("status") == "analyzed"
        ]
        assert len(done) == 1
        assert done[0]["window"]["frames"] == 450
        assert done[0]["frame_records"][-1]["idx"] == 449
        st = rep["diagnostic_status"]
        assert st["analysis_jobs_completed"] == 1
        assert st["completed_windows"] == 1
        # Window finalized -> a new window may already be collecting.
        assert st["collection_state"] in (
            "collecting", "service_attached_no_frames",
            "detector_stopped_window_open",
        )
    finally:
        svc.stop()


# ----------------------------------------------------------------------
# Test D — detector suspended at frame 200: partial preserved
# ----------------------------------------------------------------------
def test_d_suspend_preserves_partial(diag_on):
    rec, report = diag_on
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(200)
        assert _wait_for(lambda: rec._frames_recorded >= 200, 5)
        svc.suspend()
        assert _wait_for(
            lambda: rec._stopped_reason is not None, 5)
        rep = _wait_report_where(
            report,
            lambda r: r["diagnostic_status"]["collection_state"]
            == "detector_stopped_window_open")
        assert rep is not None
        st = rep["diagnostic_status"]
        assert st["collection_state"] == "detector_stopped_window_open"
        assert st["current_window_frames"] == 200
        assert rep["active_window"]["status"] == "stopped_open"
        assert rep["active_window"]["stopped_reason"] in (
            "suspend", "stop")
        # Seal it explicitly (as resume/shutdown would) and verify the
        # lifecycle trail + interruption reason are preserved.
        rec.shutdown()
        rep = _read_report(report)
        partials = [
            w for w in rep["windows"]
            if w.get("status") == "interrupted_partial"
        ]
        assert len(partials) == 1
        p = partials[0]
        assert p["window"]["frames"] == 200
        assert p["interruption"]["reason"] in ("suspend", "stop")
        kinds = [e["kind"] for e in p["detector_events"]]
        assert "suspend" in kinds and "stop" in kinds
        assert p["analysis"]["status"] == "skipped"
        # No frame loss: every seen frame was recorded.
        assert rec._frames_seen == rec._frames_recorded == 200
    finally:
        try:
            svc.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Test E — resume: new lifecycle generation, window split
# ----------------------------------------------------------------------
def test_e_resume_splits_window(diag_on):
    rec, report = diag_on
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(200)
        assert _wait_for(lambda: rec._frames_recorded >= 200, 5)
        svc.suspend()
        assert _wait_for(lambda: rec._stopped_reason, 5)
        svc.resume()
        cap.push_frames(100)
        assert _wait_for(lambda: len(rec._frames) >= 100, 5)
        rep = _wait_report_where(
            report,
            lambda r: (r.get("active_window") or {}).get("frames")
            == 100)
        assert rep is not None
        partials = [
            w for w in rep["windows"]
            if w.get("status") == "interrupted_partial"
        ]
        assert len(partials) == 1
        assert partials[0]["window"]["frames"] == 200
        # New window collecting the post-resume frames.
        assert rep["active_window"]["frames"] == 100
        assert rep["diagnostic_status"]["collection_state"] == \
            "collecting"
        # Ordering: sealed window index < active window index.
        assert partials[0]["window"]["index"] < \
            rep["active_window"]["index"]
        # The new window begins with a fresh frame idx sequence.
        assert rec._frames[0]["idx"] == 0
        assert rec._next_frame == 100
    finally:
        svc.stop()


# ----------------------------------------------------------------------
# Test F — analysis exception: window failed, app stays responsive
# ----------------------------------------------------------------------
def test_f_analysis_failure_isolated(diag_on, monkeypatch):
    rec, report = diag_on
    calls = {"n": 0}

    def flaky(widx, frames, audio, events):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"conclusion": "second-window-ok"}

    monkeypatch.setattr(rec, "_analyze_window", flaky)
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(450)
        assert _wait_for(lambda: calls["n"] >= 1, 10)
        assert _wait_for(
            lambda: any(
                w.get("status") == "failed" for w in rec._windows), 10)
        # Application remains responsive: keep feeding frames.
        cap.push_frames(30)
        assert _wait_for(lambda: rec._frames_recorded >= 480, 5)
        assert svc.is_running
        # A second full window is still analyzed despite the first
        # failure.
        cap.push_frames(420)
        assert _wait_for(lambda: calls["n"] >= 2, 15)
        rep = _wait_report_where(
            report,
            lambda r: any(
                w.get("status") == "analyzed" for w in r["windows"]))
        assert rep is not None
        statuses = [w.get("status") for w in rep["windows"]]
        assert "failed" in statuses
        assert "analyzed" in statuses
        st = rep["diagnostic_status"]
        assert st["analysis_jobs_failed"] >= 1
        assert st["analysis_jobs_completed"] >= 1
        failed = [w for w in rep["windows"]
                  if w.get("status") == "failed"][0]
        assert "boom" in failed["analysis"]["error"]
    finally:
        svc.stop()


# ----------------------------------------------------------------------
# Test G — concurrent report updates lose nothing, JSON stays valid
# ----------------------------------------------------------------------
def test_g_concurrent_windows_all_present(diag_on, monkeypatch):
    rec, report = diag_on
    _patch_fast_analysis(monkeypatch, rec)
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(900)  # two full windows back-to-back
        assert _wait_for(
            lambda: sum(
                1 for w in rec._windows if w.get("status") == "analyzed"
            ) >= 2, 15)
        # Hammer checkpoints from multiple threads while the writer
        # serializes them.
        threads = [
            threading.Thread(target=rec._request_checkpoint)
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rep = _wait_report_where(
            report,
            lambda r: sum(
                1 for w in r["windows"]
                if w.get("status") == "analyzed") >= 2)
        assert rep is not None
        analyzed = [
            w for w in rep["windows"] if w.get("status") == "analyzed"]
        assert len(analyzed) >= 2
        idxs = sorted(w["window"]["index"] for w in analyzed)
        assert idxs == sorted(set(idxs))
        # File is valid JSON (atomic replace, no torn writes).
        assert _read_report(report) is not None
    finally:
        svc.stop()


# ----------------------------------------------------------------------
# Test H — shutdown: incomplete capture + pending analysis explicit
# ----------------------------------------------------------------------
def test_h_shutdown_reports_interruption(diag_on, monkeypatch):
    rec, report = diag_on
    release = threading.Event()

    def blocking(widx, frames, audio, events):
        release.wait(timeout=30)
        return {"conclusion": "late"}

    monkeypatch.setattr(rec, "_analyze_window", blocking)
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(100)
        assert _wait_for(lambda: rec._frames_recorded >= 100, 5)
        diag.shutdown_diagnostic()
        rep = _read_report(report)
        assert rep is not None
        assert rep["shutdown"] is True
        st = rep["diagnostic_status"]
        assert st["shutdown"] is True
        assert st["collection_state"] == "stopped"
        partials = [
            w for w in rep["windows"]
            if w.get("status") == "interrupted_partial"
        ]
        assert len(partials) == 1
        assert partials[0]["interruption"]["reason"] == \
            "process_shutdown"
        assert partials[0]["window"]["frames"] == 100
    finally:
        release.set()
        try:
            svc.stop()
        except Exception:
            pass


def test_h2_pending_analysis_marked(diag_on, monkeypatch):
    rec, report = diag_on
    started = threading.Event()
    release = threading.Event()

    def blocking(widx, frames, audio, events):
        started.set()
        release.wait(timeout=30)
        return {"conclusion": "late"}

    monkeypatch.setattr(rec, "_analyze_window", blocking)
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(450)
        assert started.wait(timeout=10)
        diag.shutdown_diagnostic()
        rep = _read_report(report)
        assert rep is not None
        # The in-flight analysis is explicitly not claimed complete.
        assert rep["shutdown"] is True
        st = rep["diagnostic_status"]
        assert st["analysis_jobs_pending"] >= 1
        running = [
            w for w in rep["windows"]
            if (w.get("analysis") or {}).get("status") == "running"
        ]
        assert len(running) == 1
        assert running[0]["status"] == "analyzing"
    finally:
        release.set()
        try:
            svc.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Test I — report path is absolute, independent of CWD
# ----------------------------------------------------------------------
def test_i_report_path_absolute(tmp_path, monkeypatch):
    # The module constant must resolve inside the project regardless
    # of the process working directory.
    from src.core.paths import runtime_dir

    expected = (
        runtime_dir() / "reports" / "v1074_production_kws_parity.json"
    )
    assert diag.REPORT_PATH == expected
    assert diag.REPORT_PATH.is_absolute()

    report = tmp_path / "parity.json"
    monkeypatch.setenv(diag.ENV_FLAG, "1")
    diag.reset_for_tests()
    monkeypatch.setattr(diag, "REPORT_PATH", report)
    rec = diag.recorder
    try:
        monkeypatch.chdir(tmp_path)  # different working directory
        cap, eng = FakeCapture(), FakeEngine()
        svc = _service(cap, eng)
        svc.start()
        try:
            cap.push_frames(50)
            assert _wait_for(lambda: rec._frames_recorded >= 50, 5)
            assert _wait_for(
                lambda: (lambda r: r is not None and
                         r["diagnostic_status"]["frames_recorded"] >= 50
                         )(_read_report(report)), 10)
        finally:
            svc.stop()
        # Report landed at the patched absolute path even though the
        # process CWD is tmp_path.
        rep = _read_report(report)
        assert rep is not None
        assert rep["diagnostic_status"]["frames_recorded"] == 50
    finally:
        rec._supersede()


# ----------------------------------------------------------------------
# Test J — real WakeWordDetectionService + production engine factory
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_j_real_service_attachment(tmp_path, monkeypatch):
    """Frames must flow through the actual production path: real
    create_engine() (sherpa-onnx) + real WakeWordDetectionService."""
    from src.voice.wake_word_engine import create_engine

    report = tmp_path / "parity.json"
    monkeypatch.setenv(diag.ENV_FLAG, "1")
    diag.reset_for_tests()
    monkeypatch.setattr(diag, "REPORT_PATH", report)
    rec = diag.recorder
    engine = create_engine()
    cap = FakeCapture()
    svc = WakeWordDetectionService(
        cap, engine, threshold=0.25, cooldown_seconds=0.0)
    svc.start()
    try:
        cap.push_frames(50)
        assert _wait_for(lambda: rec._frames_recorded >= 50, 15)
        # The recorded frames came through engine.process on the
        # production worker thread with the real sherpa engine.
        fr = rec._frames[0]
        assert fr["engine_name"] == "sherpa-onnx-kws"
        assert fr["thread_name"] == "wakeword-detect"
        assert fr["service_id"] == id(svc)
        assert rec._status_dict()["production_service_attached"]
        rep = _wait_report_where(
            report,
            lambda r: r["diagnostic_status"]["frames_recorded"] >= 50)
        assert rep is not None
        assert rep["diagnostic_status"]["frames_recorded"] >= 50
        assert rep["diagnostic_status"]["collection_state"] == \
            "collecting"
    finally:
        svc.stop()
        rec._supersede()


# ----------------------------------------------------------------------
# Status/report-shape checks
# ----------------------------------------------------------------------
def test_status_fields_present(diag_on):
    rec, report = diag_on
    st = diag.report_status()
    for key in (
        "enabled", "initialized", "recorder_attached",
        "production_service_attached", "frames_seen",
        "frames_recorded", "current_window_frames",
        "required_window_frames", "completed_windows",
        "analysis_jobs_pending", "analysis_jobs_completed",
        "analysis_jobs_failed", "last_frame_timestamp",
        "last_report_write_timestamp", "last_error",
        "collection_state",
    ):
        assert key in st, key
    assert st["collection_state"] == "no_production_frames_received"


def test_no_pcm_in_report(diag_on, monkeypatch):
    rec, report = diag_on
    _patch_fast_analysis(monkeypatch, rec)
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc.start()
    try:
        cap.push_frames(450)
        assert _wait_for(
            lambda: any(
                w.get("status") == "analyzed" for w in rec._windows),
            10)
        rep = _wait_report_where(
            report,
            lambda r: any(
                w.get("status") == "analyzed" for w in r["windows"]))
        assert rep is not None
        text = report.read_text(encoding="utf-8")
        # Frame records carry hashes/metadata only — no sample arrays.
        assert '"samples"' not in text
        assert '"audio"' not in text
        w = [w for w in rep["windows"]
             if w.get("status") == "analyzed"][0]
        assert "pcm" not in json.dumps(w.get("frame_records", [{}])[0]
                                       ).lower()
    finally:
        svc.stop()
