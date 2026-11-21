

"""Offline wake-word benchmark framework for VoiceOS.

This module evaluates recorded local audio through the existing KWS
implementation only. It never opens a microphone, never records audio,
and never transmits data. All inputs are explicit local files referenced
by a JSON manifest.
"""

from __future__ import annotations

import json
import math
import time
import wave
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

TARGET_SAMPLE_RATE = 16000
FRAME_SAMPLES = 1600  # 100 ms @ 16 kHz, matching production cadence
SCHEMA_VERSION = "1.0"
BENCHMARK_NAME = "voiceos-kws"

# Production rescore anchors (kept identical to the live defaults; the
# benchmark may override them for experiments but defaults are unchanged).
DEFAULT_RESCROE_LEADS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8)
DEFAULT_RESCROE_TRAIL = 0.5
# Correctly spelled aliases; the RESCROE names are kept for compatibility.
DEFAULT_RESCORE_LEADS = DEFAULT_RESCROE_LEADS
DEFAULT_RESCORE_TRAIL = DEFAULT_RESCROE_TRAIL

EngineFactory = Callable[[], Any]
TraceFn = Callable[..., None]


def load_wav_16k(path: Path | str) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono 16 kHz float32.

    Returns ``(samples, source_sample_rate)``. Supports PCM int16/int32/
    uint8 and IEEE float32/float64 WAV payloads.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as w:
        rate = int(w.getframerate())
        channels = int(w.getnchannels())
        width = int(w.getsampwidth())
        frames = int(w.getnframes())
        raw = w.readframes(frames)
        comptype = w.getcomptype()

    if comptype != "NONE":
        raise ValueError(f"compressed WAV not supported: {path}")

    if width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        # Ambiguous: could be int32 PCM or float32. WAV fmt tag is not
        # exposed by the wave module portably, so prefer float32 when the
        # values look like [-1, 1] floats; otherwise treat as int32 PCM.
        as_float = np.frombuffer(raw, dtype=np.float32).copy()
        as_int = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        finite = np.isfinite(as_float)
        looks_float = bool(
            finite.all()
            and (as_float.size == 0 or np.max(np.abs(as_float)) <= 1.5)
            and not np.allclose(as_float, as_float.astype(np.int32), atol=1e-3)
        )
        audio = as_float if looks_float else as_int
    elif width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        a = np.empty(b.shape[0], dtype=np.float32)
        a[:] = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        a = np.where(a >= 8388608, a - 16777216, a).astype(np.float32) / 8388608.0
        audio = a
    else:
        raise ValueError(f"unsupported WAV sample width {width}: {path}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != TARGET_SAMPLE_RATE:
        from src.voice.audio_capture_service import resample_to_16k

        audio = resample_to_16k(audio.astype(np.float32), float(rate))

    return audio.astype(np.float32, copy=False), rate


def _interp_resample(audio: np.ndarray, out_len: int) -> np.ndarray:
    if audio.size == 0 or out_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if out_len == audio.size:
        return audio.astype(np.float32, copy=True)
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=True)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Change duration by ``speed`` (>1 = faster/shorter)."""
    if speed <= 0:
        raise ValueError("speed must be > 0")
    if abs(speed - 1.0) < 1e-9:
        return audio.astype(np.float32, copy=True)
    out_len = max(1, int(round(audio.size / speed)))
    return _interp_resample(audio, out_len)


def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    if gain < 0:
        raise ValueError("gain must be >= 0")
    return np.clip(audio.astype(np.float32) * gain, -1.0, 1.0)


def pad_audio(audio: np.ndarray, before_s: float, after_s: float) -> np.ndarray:
    before = int(round(before_s * TARGET_SAMPLE_RATE))
    after = int(round(after_s * TARGET_SAMPLE_RATE))
    before = max(0, before)
    after = max(0, after)
    if before == 0 and after == 0:
        return audio.astype(np.float32, copy=True)
    return np.concatenate(
        [
            np.zeros(before, dtype=np.float32),
            audio.astype(np.float32, copy=False),
            np.zeros(after, dtype=np.float32),
        ]
    )


def mix_background(
    audio: np.ndarray,
    background: np.ndarray,
    gain: float,
    offset_s: float = 0.0,
) -> np.ndarray:
    """Mix a background noise bed under ``audio`` at ``gain``."""
    if background.size == 0 or gain <= 0:
        return audio.astype(np.float32, copy=True)
    out = audio.astype(np.float32, copy=True)
    offset = int(round(offset_s * TARGET_SAMPLE_RATE))
    bg = apply_gain(background, gain)
    start = max(0, offset)
    # Tile/truncate the background to cover the output.
    need = out.size - start
    if need <= 0:
        return out
    reps = int(math.ceil(need / bg.size))
    tiled = np.tile(bg, reps)[:need]
    out[start : start + need] += tiled
    return np.clip(out, -1.0, 1.0)


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (p / 100.0) * (len(vals) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    frac = k - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _load_manifest(path: Path | str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    if not isinstance(data, list):
        raise ValueError("manifest must be a JSON list or {'cases': [...]}")
    return data


def load_case(entry: dict[str, Any], base_dir: Path | str) -> dict[str, Any]:
    """Load and transform one manifest entry into benchmark audio."""
    base = Path(base_dir)
    wav = Path(str(entry.get("wav") or entry.get("file") or ""))
    if not wav.is_absolute():
        wav = base / wav
    if not wav.exists():
        raise FileNotFoundError(f"missing wav for case {entry.get('id')}: {wav}")

    label = str(entry.get("label") or ("positive" if entry.get("expected") else "negative")).lower()
    if label not in ("positive", "negative"):
        raise ValueError(f"invalid label {label!r} for case {entry.get('id')}")

    audio, source_rate = load_wav_16k(wav)
    transforms = dict(entry.get("transforms") or {})
    applied: dict[str, Any] = {}

    speed = float(transforms.get("speed", 1.0) or 1.0)
    if speed != 1.0:
        audio = apply_speed(audio, speed)
        applied["speed"] = speed

    gain = transforms.get("gain")
    if gain is not None:
        audio = apply_gain(audio, float(gain))
        applied["gain"] = float(gain)

    bg_spec = transforms.get("background")
    if bg_spec:
        bg_wav = Path(str(bg_spec.get("wav") or ""))
        if not bg_wav.is_absolute():
            bg_wav = base / bg_wav
        bg_audio, _ = load_wav_16k(bg_wav)
        audio = mix_background(
            audio,
            bg_audio,
            float(bg_spec.get("gain", 0.1)),
            float(bg_spec.get("offset_seconds", 0.0)),
        )
        applied["background"] = {
            "wav": str(bg_wav),
            "gain": float(bg_spec.get("gain", 0.1)),
            "offset_seconds": float(bg_spec.get("offset_seconds", 0.0)),
        }

    before_s = float(transforms.get("pad_before_seconds", 0.0) or 0.0)
    after_s = float(transforms.get("pad_after_seconds", 0.0) or 0.0)
    if before_s or after_s:
        audio = pad_audio(audio, before_s, after_s)
        applied["pad_before_seconds"] = before_s
        applied["pad_after_seconds"] = after_s

    duration = audio.size / float(TARGET_SAMPLE_RATE)
    wake_end = entry.get("wake_end_seconds")
    if wake_end is None:
        wake_end = duration if label == "positive" else None
    else:
        wake_end = float(wake_end)
        if speed != 1.0:
            wake_end = wake_end / speed
        wake_end += before_s

    case_id = str(entry.get("id") or wav.stem)
    return {
        "id": case_id,
        "label": label,
        "expected": label == "positive",
        "wav": str(wav),
        "source_sample_rate": source_rate,
        "sample_rate": TARGET_SAMPLE_RATE,
        "duration_seconds": round(duration, 6),
        "wake_end_seconds": None if wake_end is None else round(float(wake_end), 6),
        "audio": audio,
        "transforms": applied,
        "reference": entry.get("reference"),
        "notes": entry.get("notes"),
    }


def load_cases(manifest_path: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all manifest cases; returns (cases, load_errors)."""
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, entry in enumerate(_load_manifest(manifest_path)):
        try:
            cases.append(load_case(entry, base))
        except Exception as exc:  # noqa: BLE001 - record and continue
            errors.append(
                {
                    "index": i,
                    "id": entry.get("id") if isinstance(entry, dict) else None,
                    "wav": entry.get("wav") if isinstance(entry, dict) else None,
                    "error": repr(exc),
                }
            )
    return cases, errors


def _chunks(audio: np.ndarray, frame_samples: int) -> Iterable[np.ndarray]:
    for off in range(0, audio.size, frame_samples):
        yield audio[off : off + frame_samples]


def _detect(engine: Any, frame: np.ndarray) -> tuple[bool, Optional[str]]:
    conf = engine.process(frame)
    kw = None
    try:
        kw = engine.keyword_detected()
    except Exception:  # noqa: BLE001
        kw = None
    detected = bool(kw) or (isinstance(conf, (int, float)) and float(conf) >= 0.5)
    return detected, kw


def evaluate_fresh_stream(
    engine: Any,
    cases: list[dict[str, Any]],
    frame_samples: int = FRAME_SAMPLES,
    cooldown_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    """Evaluate each case on a fresh engine stream (reset per case)."""
    results = []
    for case in cases:
        row = {
            "case_id": case["id"],
            "label": case["label"],
            "expected": case["expected"],
            "duration_seconds": case["duration_seconds"],
            "detected": False,
            "detection_count": 0,
            "false_activation": False,
            "first_detection_seconds": None,
            "latency_seconds": None,
            "keyword": None,
            "error": None,
        }
        try:
            try:
                engine.reset()
            except Exception:  # noqa: BLE001
                pass
            last_hit = -1e9
            stream_t = 0.0
            for frame in _chunks(case["audio"], frame_samples):
                detected, kw = _detect(engine, frame)
                stream_t += frame.size / float(TARGET_SAMPLE_RATE)
                t = stream_t
                if detected and (t - last_hit) >= cooldown_seconds:
                    row["detection_count"] += 1
                    row["detected"] = True
                    row["keyword"] = kw or row["keyword"] or "detected"
                    if row["first_detection_seconds"] is None:
                        row["first_detection_seconds"] = round(t, 6)
                        wake_end = case.get("wake_end_seconds")
                        if wake_end is not None:
                            row["latency_seconds"] = round(max(0.0, t - wake_end), 6)
                    last_hit = t
            row["false_activation"] = (not case["expected"]) and row["detected"]
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)
        results.append(row)
    return results


def evaluate_live_stream(
    engine: Any,
    cases: list[dict[str, Any]],
    frame_samples: int = FRAME_SAMPLES,
    gap_seconds: float = 1.0,
    cooldown_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    """Continuous live-style evaluation across concatenated cases.

    Cases are streamed back-to-back with gap_seconds of silence
    between them through ONE engine stream, matching production's
    continuous-capture model. Detections are attributed to the case whose
    time interval contains the detection frame.
    """
    results = [
        {
            "case_id": c["id"],
            "label": c["label"],
            "expected": c["expected"],
            "duration_seconds": c["duration_seconds"],
            "detected": False,
            "detection_count": 0,
            "false_activation": False,
            "first_detection_seconds": None,
            "latency_seconds": None,
            "keyword": None,
            "error": None,
        }
        for c in cases
    ]
    if not cases:
        return results

    gap_samples = int(round(gap_seconds * TARGET_SAMPLE_RATE))
    gap_frames = [
        np.zeros(min(frame_samples, gap_samples - i), dtype=np.float32)
        for i in range(0, max(0, gap_samples), frame_samples)
    ]
    gap_frames = [g for g in gap_frames if g.size]
    gap_dur = sum(g.size for g in gap_frames) / float(TARGET_SAMPLE_RATE)

    starts: list[float] = []
    t = 0.0
    for case in cases:
        starts.append(t)
        t += case["duration_seconds"] + gap_dur

    pending_frames: list[np.ndarray] = []
    for case in cases:
        pending_frames.extend(_chunks(case["audio"], frame_samples))
        pending_frames.extend(gap_frames)
    for _ in range(len(gap_frames)):
        if pending_frames:
            pending_frames.pop()

    last_hit = -1e9
    stream_t = 0.0
    for frame in pending_frames:
        stream_t += frame.size / float(TARGET_SAMPLE_RATE)
        hit_t = stream_t
        detected, kw = _detect(engine, frame)
        if detected and (hit_t - last_hit) >= cooldown_seconds:
            last_hit = hit_t
            # KWS is causal: a detection during the inter-case gap must trail
            # the preceding case's audio, so each case owns [start, next start).
            owner_idx = 0
            for i, start in enumerate(starts):
                if start <= hit_t + 1e-9:
                    owner_idx = i
                else:
                    break
            owner = cases[owner_idx]
            owner_start = starts[owner_idx]
            row = next(r for r in results if r["case_id"] == owner["id"])
            row["detection_count"] += 1
            row["detected"] = True
            row["keyword"] = kw or row["keyword"] or "detected"
            rel_t = hit_t - owner_start
            if row["first_detection_seconds"] is None:
                row["first_detection_seconds"] = round(rel_t, 6)
                wake_end = owner.get("wake_end_seconds")
                if wake_end is not None:
                    row["latency_seconds"] = round(max(0.0, rel_t - wake_end), 6)
            row["false_activation"] = (not owner["expected"]) and row["detected"]
    return results


def evaluate_rescore(
    engine: Any,
    cases: list[dict[str, Any]],
    lead_seconds: tuple[float, ...] = DEFAULT_RESCROE_LEADS,
    trail_seconds: float = DEFAULT_RESCROE_TRAIL,
) -> list[dict[str, Any]]:
    """Evaluate the existing engine.rescore() path per case."""
    results = []
    for case in cases:
        row = {
            "case_id": case["id"],
            "label": case["label"],
            "expected": case["expected"],
            "duration_seconds": case["duration_seconds"],
            "hit": False,
            "detected": False,
            "false_activation": False,
            "keyword": None,
            "lead_seconds": None,
            "anchors_requested": 0,
            "anchors_evaluated": 0,
            "elapsed_ms": None,
            "anchor_events": 0,
            "error": None,
        }
        events: list[tuple[str, dict[str, Any]]] = []

        def trace(event: str, **fields: Any) -> None:
            events.append((event, dict(fields)))

        try:
            t0 = time.perf_counter()
            result = engine.rescore(
                case["audio"],
                lead_seconds=lead_seconds,
                trail_seconds=trail_seconds,
                trace=trace,
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            row["wall_elapsed_ms"] = round(wall_ms, 3)
            if result:
                row["hit"] = True
                row["detected"] = True
                row["keyword"] = result[0] if isinstance(result, tuple) else str(result)
                row["lead_seconds"] = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            for event, fields in events:
                if event == "anchor":
                    row["anchor_events"] += 1
                elif event == "complete":
                    row["anchors_requested"] = int(fields.get("anchors_requested", row["anchors_requested"]))
                    row["anchors_evaluated"] = int(fields.get("anchors_evaluated", row["anchor_events"]))
                    row["elapsed_ms"] = fields.get("elapsed_ms", row["elapsed_ms"])
                    if fields.get("hit") and not row["hit"]:
                        row["hit"] = True
                        row["detected"] = True
                        row["keyword"] = fields.get("keyword") or row["keyword"]
                        row["lead_seconds"] = fields.get("lead_seconds")
            row["false_activation"] = (not case["expected"]) and row["detected"]
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)
        results.append(row)
    return results


def _empty_summary() -> dict[str, Any]:
    return {
        "evaluated_cases": 0,
        "positive_cases": 0,
        "negative_cases": 0,
        "negative_evaluated_seconds": 0.0,
        "true_detections": 0,
        "missed_detections": 0,
        "false_activation_cases": 0,
        "false_activation_events": 0,
        "recall": None,
        "false_positive_rate": None,
        "false_positives_per_hour": None,
        "detection_latency_seconds": {"count": 0, "p50": None, "p95": None},
        "rescore_elapsed_ms": {"count": 0, "mean": None, "p50": None, "p95": None},
        "anchors_evaluated": 0,
        "anchors_requested": 0,
    }


def summarize(mode: str, results: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _empty_summary()
    case_by_id = {c["id"]: c for c in cases}
    positives = [r for r in results if r.get("label") == "positive"]
    negatives = [r for r in results if r.get("label") == "negative"]
    summary["positive_cases"] = len(positives)
    summary["negative_cases"] = len(negatives)
    summary["evaluated_cases"] = len(results)
    summary["negative_evaluated_seconds"] = round(
        sum(float(r.get("duration_seconds") or 0.0) for r in negatives), 6
    )

    tp = sum(1 for r in positives if r.get("detected") or r.get("hit"))
    summary["true_detections"] = tp
    summary["missed_detections"] = max(0, len(positives) - tp)
    summary["false_activation_cases"] = sum(
        1 for r in negatives if r.get("false_activation") or ((not r.get("expected", True)) and (r.get("detected") or r.get("hit")))
    )
    summary["false_activation_events"] = sum(
        int(r.get("detection_count") or 0) for r in negatives if (r.get("detected") or r.get("hit"))
    )
    denom = tp + summary["missed_detections"]
    summary["recall"] = round(tp / denom, 6) if denom else None
    if negatives:
        summary["false_positive_rate"] = round(summary["false_activation_cases"] / len(negatives), 6)
    if summary["negative_evaluated_seconds"] > 0:
        hours = summary["negative_evaluated_seconds"] / 3600.0
        summary["false_positives_per_hour"] = round(summary["false_activation_cases"] / hours, 6)

    latencies = [r.get("latency_seconds") for r in positives if r.get("latency_seconds") is not None]
    latencies = [float(v) for v in latencies if v is not None]
    summary["detection_latency_seconds"] = {
        "count": len(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
    }

    if mode == "rescore":
        elapsed = [float(r["elapsed_ms"]) for r in results if r.get("elapsed_ms") is not None]
        summary["rescore_elapsed_ms"] = {
            "count": len(elapsed),
            "mean": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
            "p50": percentile(elapsed, 50),
            "p95": percentile(elapsed, 95),
        }
        summary["anchors_evaluated"] = int(sum(int(r.get("anchors_evaluated") or 0) for r in results))
        summary["anchors_requested"] = int(sum(int(r.get("anchors_requested") or 0) for r in results))
    return summary


def run_benchmark(
    manifest_path: Path | str,
    engine_factory: EngineFactory,
    modes: Iterable[str] = ("live", "fresh_stream", "rescore"),
    frame_samples: int = FRAME_SAMPLES,
    gap_seconds: float = 1.0,
    cooldown_seconds: float = 3.0,
    rescore_leads: tuple[float, ...] = DEFAULT_RESCROE_LEADS,
    rescore_trail: float = DEFAULT_RESCROE_TRAIL,
    engine_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the offline KWS benchmark and return a JSON-serializable dict."""
    modes = tuple(modes)
    cases, errors = load_cases(manifest_path)
    manifest_path = Path(manifest_path)

    engine = engine_factory()
    try:
        info = dict(engine_info or {})
        info.setdefault("name", getattr(engine, "name", "unknown"))
        info.setdefault("model_dir", str(getattr(engine, "model_dir", "")))
        info.setdefault("keyword_count", getattr(engine, "keyword_count", None))
        info.setdefault("keywords_sha256", getattr(engine, "keywords_sha256", None))
    finally:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass

    results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    if "live" in modes:
        live_engine = engine_factory()
        try:
            results["live"] = evaluate_live_stream(
                live_engine, cases, frame_samples=frame_samples,
                gap_seconds=gap_seconds, cooldown_seconds=cooldown_seconds,
            )
        finally:
            try:
                live_engine.close()
            except Exception:  # noqa: BLE001
                pass
        summaries["live"] = summarize("live", results["live"], cases)
    if "fresh_stream" in modes:
        fs_engine = engine_factory()
        try:
            results["fresh_stream"] = evaluate_fresh_stream(
                fs_engine, cases, frame_samples=frame_samples,
                cooldown_seconds=cooldown_seconds,
            )
        finally:
            try:
                fs_engine.close()
            except Exception:  # noqa: BLE001
                pass
        summaries["fresh_stream"] = summarize("fresh_stream", results["fresh_stream"], cases)
    if "rescore" in modes:
        rs_engine = engine_factory()
        try:
            results["rescore"] = evaluate_rescore(
                rs_engine, cases, lead_seconds=rescore_leads,
                trail_seconds=rescore_trail,
            )
        finally:
            try:
                rs_engine.close()
            except Exception:  # noqa: BLE001
                pass
        summaries["rescore"] = summarize("rescore", results["rescore"], cases)

    public_cases = [
        {k: v for k, v in c.items() if k != "audio"} for c in cases
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": "completed" if cases else "no_inputs",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(manifest_path),
        "engine": info,
        "config": {
            "frame_samples": frame_samples,
            "frame_ms": round(frame_samples / TARGET_SAMPLE_RATE * 1000.0, 3),
            "sample_rate": TARGET_SAMPLE_RATE,
            "gap_seconds": gap_seconds,
            "cooldown_seconds": cooldown_seconds,
            "modes": list(modes),
            "rescore": {
                "lead_seconds": list(rescore_leads),
                "trail_seconds": rescore_trail,
            },
        },
        "inputs": {
            "case_count": len(cases),
            "positive_cases": sum(1 for c in cases if c["label"] == "positive"),
            "negative_cases": sum(1 for c in cases if c["label"] == "negative"),
            "total_audio_seconds": round(sum(c["duration_seconds"] for c in cases), 6),
            "load_errors": errors,
        },
        "cases": public_cases,
        "results": results,
        "summary": summaries,
        "privacy": {
            "microphone_used": False,
            "audio_persisted": False,
            "network_transmission": False,
            "note": "Offline evaluation of explicit local WAV fixtures only.",
        },
    }


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_markdown(result: dict[str, Any]) -> str:
    """Render a human-readable report from a benchmark result dict."""
    lines = [
        "# VoiceOS Wake-Word Benchmark",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Created (UTC): {result.get('created_utc')}",
        f"- Manifest: `{result.get('manifest')}`",
        f"- Engine: `{(result.get('engine') or {}).get('name')}` "
        f"model_dir=`{(result.get('engine') or {}).get('model_dir')}` "
        f"keywords={((result.get('engine') or {}).get('keyword_count'))} "
        f"sha256=`{str((result.get('engine') or {}).get('keywords_sha256'))[:12]}`",
        "",
        "## Inputs",
        "",
    ]
    inputs = result.get("inputs") or {}
    lines += [
        f"- Cases: {inputs.get('case_count', 0)} "
        f"(positive={inputs.get('positive_cases', 0)}, negative={inputs.get('negative_cases', 0)})",
        f"- Total audio: {_fmt(inputs.get('total_audio_seconds'), 3)} s",
        f"- Load errors: {len(inputs.get('load_errors') or [])}",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(result.get("config") or {}, indent=2, sort_keys=True),
        "```",
        "",
    ]
    if result.get("status") == "no_inputs":
        lines += [
            "No recordings were supplied. This is an empty-input run, not a "
            "zero-accuracy measurement.",
            "",
        ]
    for mode, summary in (result.get("summary") or {}).items():
        lines += [
            f"## Mode: `{mode}`",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Evaluated cases | {summary.get('evaluated_cases')} |",
            f"| True detections | {summary.get('true_detections')} |",
            f"| Missed detections | {summary.get('missed_detections')} |",
            f"| False activation cases | {summary.get('false_activation_cases')} |",
            f"| False activation events | {summary.get('false_activation_events')} |",
            f"| Recall | {_fmt(summary.get('recall'), 4)} |",
            f"| Negative evaluated seconds | {_fmt(summary.get('negative_evaluated_seconds'), 3)} |",
            f"| False-positive rate (per negative case) | {_fmt(summary.get('false_positive_rate'), 4)} |",
            f"| False positives per hour | {_fmt(summary.get('false_positives_per_hour'), 3)} |",
            f"| Detection latency p50 (s) | {_fmt((summary.get('detection_latency_seconds') or {}).get('p50'), 4)} |",
            f"| Detection latency p95 (s) | {_fmt((summary.get('detection_latency_seconds') or {}).get('p95'), 4)} |",
        ]
        if mode == "rescore":
            res = summary.get("rescore_elapsed_ms") or {}
            lines += [
                f"| Rescore elapsed mean (ms) | {_fmt(res.get('mean'), 3)} |",
                f"| Rescore elapsed p50 (ms) | {_fmt(res.get('p50'), 3)} |",
                f"| Rescore elapsed p95 (ms) | {_fmt(res.get('p95'), 3)} |",
                f"| Anchors requested | {summary.get('anchors_requested')} |",
                f"| Anchors evaluated | {summary.get('anchors_evaluated')} |",
            ]
        lines.append("")
    lines += [
        "## Privacy",
        "",
        "- Microphone used: **no**",
        "- Raw audio persisted by benchmark: **no**",
        "- Network transmission: **no**",
        "",
        "Only explicit local WAV fixtures listed in the manifest are evaluated.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], json_path: Path | str, report_path: Optional[Path | str] = None) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_markdown(result), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dumps_result(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True, default=json_default)


__all__ = [
    "TARGET_SAMPLE_RATE",
    "FRAME_SAMPLES",
    "SCHEMA_VERSION",
    "BENCHMARK_NAME",
    "DEFAULT_RESCROE_LEADS",
    "DEFAULT_RESCROE_TRAIL",
    "DEFAULT_RESCORE_LEADS",
    "DEFAULT_RESCORE_TRAIL",
    "load_wav_16k",
    "apply_speed",
    "apply_gain",
    "pad_audio",
    "mix_background",
    "percentile",
    "load_case",
    "load_cases",
    "evaluate_fresh_stream",
    "evaluate_live_stream",
    "evaluate_rescore",
    "summarize",
    "run_benchmark",
    "render_markdown",
    "write_outputs",
    "dumps_result",
]
