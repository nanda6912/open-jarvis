"""Regression tests for the V10.8 Task 2 offline KWS benchmark."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from src.voice import kws_benchmark as kb


def _write_wav(path: Path, audio: np.ndarray, rate: int = 16000) -> Path:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * kb.TARGET_SAMPLE_RATE)) / kb.TARGET_SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _case(case_id: str, label: str, audio: np.ndarray, wake_end: float | None = None) -> dict:
    return {
        "id": case_id,
        "label": label,
        "expected": label == "positive",
        "duration_seconds": round(audio.size / kb.TARGET_SAMPLE_RATE, 6),
        "wake_end_seconds": wake_end,
        "audio": audio.astype(np.float32),
    }


class FakeEngine:
    """Deterministic stand-in: fires once after enough voiced audio."""

    name = "fake-kws"
    model_dir = "fake"
    keyword_count = 6
    keywords_sha256 = "0" * 64

    def __init__(self, voiced_threshold: int = 1600):
        self.voiced_threshold = voiced_threshold
        self.reset()
        self.rescore_calls = 0

    def reset(self) -> None:
        self._voiced = 0
        self._fired = False
        self._pending = None

    def process(self, samples: np.ndarray) -> float:
        voiced = int(np.count_nonzero(np.abs(samples) > 0.05))
        self._voiced += voiced
        if not self._fired and self._voiced >= self.voiced_threshold:
            self._fired = True
            self._pending = "hello jarvis"
            return 1.0
        return 0.0

    def keyword_detected(self):
        kw, self._pending = self._pending, None
        return kw

    def rescore(self, samples, lead_seconds=(0.5,), trail_seconds=0.5, trace=None):
        self.rescore_calls += 1
        voiced = int(np.count_nonzero(np.abs(samples) > 0.05))
        hit = voiced >= self.voiced_threshold
        evaluated = 0
        for lead in lead_seconds or ():
            evaluated += 1
            if trace:
                trace("anchor", lead_seconds=float(lead), hit=hit,
                      keyword="hello jarvis" if hit else None, elapsed_ms=1.0)
            if hit:
                break
        if trace:
            trace("complete", anchors_requested=len(lead_seconds or ()),
                  anchors_evaluated=evaluated, hit=hit,
                  keyword="hello jarvis" if hit else None,
                  lead_seconds=0.1 if hit else None, elapsed_ms=2.0)
        return ("hello jarvis", 0.1) if hit else None

    def close(self) -> None:
        pass


def test_audio_transforms_and_wake_end(tmp_path: Path):
    src = _write_wav(tmp_path / "src.wav", _tone(0.5))
    entry = {
        "id": "t",
        "wav": "src.wav",
        "label": "positive",
        "wake_end_seconds": 0.4,
        "transforms": {
            "pad_before_seconds": 1.0,
            "pad_after_seconds": 0.5,
            "gain": 0.5,
            "speed": 2.0,
        },
    }
    case = kb.load_case(entry, tmp_path)
    # speed 2.0 halves 0.5s -> 0.25s; pads add 1.5s -> 1.75s total.
    assert case["duration_seconds"] == pytest.approx(1.75, abs=0.01)
    # wake_end 0.4 / 2.0 + 1.0 pad = 1.2
    assert case["wake_end_seconds"] == pytest.approx(1.2, abs=0.01)
    assert case["audio"].dtype == np.float32
    assert case["transforms"]["speed"] == 2.0


def test_load_wav_resamples_and_normalizes(tmp_path: Path):
    audio = _tone(0.2)
    path = _write_wav(tmp_path / "tone.wav", audio, rate=16000)
    loaded, rate = kb.load_wav_16k(path)
    assert rate == 16000
    assert loaded.dtype == np.float32
    assert loaded.size == pytest.approx(audio.size, abs=4)
    assert np.max(np.abs(loaded)) <= 1.0


def test_fresh_stream_positive_negative_metrics():
    pos_audio = np.concatenate([
        np.zeros(8000, dtype=np.float32),
        _tone(0.3),
        np.zeros(8000, dtype=np.float32),
    ])
    neg_audio = np.zeros(16000, dtype=np.float32)
    cases = [
        _case("pos", "positive", pos_audio, wake_end=0.9),
        _case("neg_silence", "negative", neg_audio),
    ]
    engine = FakeEngine()
    rows = kb.evaluate_fresh_stream(engine, cases, cooldown_seconds=0.0)
    assert rows[0]["detected"] is True
    assert rows[0]["latency_seconds"] is not None
    assert rows[1]["detected"] is False
    assert rows[1]["false_activation"] is False
    summary = kb.summarize("fresh_stream", rows, cases)
    assert summary["true_detections"] == 1
    assert summary["missed_detections"] == 0
    assert summary["false_activation_cases"] == 0
    assert summary["recall"] == 1.0
    assert summary["false_positive_rate"] == 0.0
    assert summary["negative_evaluated_seconds"] == pytest.approx(1.0)
    assert summary["detection_latency_seconds"]["count"] == 1


def test_live_stream_attributes_across_cases():
    pos = _case("p1", "positive", np.concatenate([np.zeros(4000, np.float32), _tone(0.4)]), wake_end=0.5)
    neg = _case("n1", "negative", np.zeros(8000, np.float32))
    engine = FakeEngine()
    rows = kb.evaluate_live_stream(engine, [pos, neg], gap_seconds=0.05, cooldown_seconds=0.0)
    by_id = {r["case_id"]: r for r in rows}
    assert by_id["p1"]["detected"] is True
    assert by_id["n1"]["detected"] is False
    assert by_id["n1"]["false_activation"] is False


def test_rescore_aggregation_and_summary():
    pos_audio = np.concatenate([np.zeros(4000, np.float32), _tone(0.4)])
    neg_audio = np.zeros(8000, np.float32)
    cases = [
        _case("pos", "positive", pos_audio, wake_end=0.5),
        _case("neg", "negative", neg_audio),
    ]
    engine = FakeEngine()
    leads = (0.1, 0.2, 0.3)
    rows = kb.evaluate_rescore(engine, cases, lead_seconds=leads, trail_seconds=0.0)
    assert rows[0]["hit"] is True
    assert rows[0]["anchors_evaluated"] == 1  # fake stops at first hit
    assert rows[1]["hit"] is False
    summary = kb.summarize("rescore", rows, cases)
    assert summary["anchors_evaluated"] == 1 + 3
    assert summary["anchors_requested"] == 3 + 3
    assert summary["rescore_elapsed_ms"]["count"] == 2
    assert summary["rescore_elapsed_ms"]["p50"] == pytest.approx(2.0)


def test_run_benchmark_no_inputs(tmp_path: Path):
    manifest = tmp_path / "empty.json"
    manifest.write_text("[]", encoding="utf-8")
    result = kb.run_benchmark(manifest, lambda: FakeEngine(), modes=("fresh_stream",))
    assert result["status"] == "no_inputs"
    assert result["inputs"]["case_count"] == 0
    assert result["summary"]["fresh_stream"]["evaluated_cases"] == 0
    text = kb.render_markdown(result)
    assert "No recordings were supplied" in text
    json.dumps(result, default=kb.json_default)


def test_load_cases_records_missing_file(tmp_path: Path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"id": "missing", "wav": "nope.wav", "label": "positive"},
        {"id": "bad", "wav": "x.wav", "label": "weird"},
    ]), encoding="utf-8")
    cases, errors = kb.load_cases(manifest)
    assert cases == []
    assert len(errors) == 2


def test_manifest_example_loads():
    manifest = Path("tests/fixtures/kws_benchmark_manifest.example.json")
    if not manifest.exists():
        pytest.skip("example manifest missing")
    cases, errors = kb.load_cases(manifest)
    assert errors == []
    assert len(cases) >= 5
    assert any(c["label"] == "positive" for c in cases)
    assert any(c["label"] == "negative" for c in cases)
    for c in cases:
        assert "audio" not in c or c["audio"].dtype == np.float32


def test_privacy_flags_and_no_audio_in_output():
    manifest = Path("tests/fixtures/kws_benchmark_manifest.example.json")
    if not manifest.exists():
        pytest.skip("example manifest missing")
    result = kb.run_benchmark(manifest, lambda: FakeEngine(), modes=("fresh_stream",))
    assert result["privacy"]["microphone_used"] is False
    assert result["privacy"]["audio_persisted"] is False
    assert result["privacy"]["network_transmission"] is False
    for case in result["cases"]:
        assert "audio" not in case
