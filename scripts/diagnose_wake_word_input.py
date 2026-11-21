"""V10.3.1 wake-word input diagnostic for the Intel 4-channel array.

Reproduces the production audio path on the real microphone and
answers: which physical channel carries "Hello Jarvis", does the
production channel-selection deliver it to the KWS engine, and does
the engine detect per channel?

Phases:
  1. Fixture sanity — hello_jarvis.wav (padded) through a fresh KWS
     engine proves the model/keyword file still detects.
  2. Live capture — opens the configured input device at 16 kHz with
     up to 4 channels while ALSO running the production
     AudioCaptureService + WakeWordDetectionService stack. A bounded
     in-memory ring buffer keeps the raw multichannel audio (never
     persisted).
  3. Offline per-channel replay — each channel is fed through its own
     fresh KWS engine; the production-selected channel stream is fed
     through another. Reports which channel(s) detect the phrase.

During the live window, speak "Hello Jarvis" 3-5 times at normal
volume, plus once quietly and once with background noise if possible.

Usage:
    py scripts/diagnose_wake_word_input.py [--seconds 40]
        [--device QUERY] [--channels N]

Writes reports/v1031_channel_diagnostics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.voice.wake_word_engine import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    create_engine,
)

REPORTS = ROOT / "reports"
FIXTURE = ROOT / "tests" / "fixtures" / "commands" / "hello_jarvis.wav"


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return (np.frombuffer(w.readframes(w.getnframes()),
                              dtype=np.int16)
                .astype(np.float32) / 32768.0)


def fixture_sanity() -> dict:
    """Engine must detect the padded fixture — proves model health."""
    eng = create_engine()
    if eng.name == "null":
        return {"engine": "null", "detected": False,
                "note": "KWS model unavailable"}
    audio = _load_wav(FIXTURE)
    pad = np.zeros(TARGET_SAMPLE_RATE // 2,
                   dtype=np.float32)  # 0.5 s context each side
    audio = np.concatenate([pad, audio, pad])
    detected = False
    kws_rms_max = 0.0
    for i in range(0, len(audio) - 1599, 1600):
        chunk = audio[i:i + 1600].copy()
        kws_rms_max = max(
            kws_rms_max,
            float(np.sqrt(np.mean(chunk * chunk))))
        if eng.process(chunk) >= 0.25:
            detected = True
            kw = eng.keyword_detected()
            break
    eng.close()
    return {"engine": eng.name, "detected": detected,
            "keyword": kw if detected else None,
            "fixture": str(FIXTURE.name),
            "fixture_rms_max": round(kws_rms_max, 5)}


def _kws_detect(audio_1d: np.ndarray) -> dict:
    """Feed a mono 16 kHz float32 signal through a fresh KWS engine."""
    eng = create_engine()
    if eng.name == "null":
        return {"engine": "null", "detected": False}
    pad = np.zeros(TARGET_SAMPLE_RATE // 2, dtype=np.float32)
    audio = np.concatenate([pad, audio_1d, pad])
    detected = False
    peak_rms = 0.0
    for i in range(0, len(audio) - 1599, 1600):
        chunk = audio[i:i + 1600].copy()
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        peak_rms = max(peak_rms, rms)
        if eng.process(chunk) >= 0.25:
            detected = True
            kw = eng.keyword_detected()
            break
    eng.close()
    return {"engine": eng.name, "detected": detected,
            "keyword": kw if detected else None,
            "peak_chunk_rms": round(peak_rms, 5)}


def list_input_devices(sd) -> list[dict]:
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({
                "index": i, "name": d.get("name"),
                "channels": d.get("max_input_channels"),
                "rate": d.get("default_samplerate"),
            })
    return out


def pick_device(sd, query: str | None):
    devices = list_input_devices(sd)
    if query:
        q = query.lower()
        for d in devices:
            if q in (d["name"] or "").lower():
                return d, devices
        # Fall back to substring on stable parts
        for d in devices:
            if any(tok in (d["name"] or "").lower()
                   for tok in q.split()):
                return d, devices
        return None, devices
    # Default: sounddevice default input
    try:
        di = sd.default.device[0] if isinstance(
            sd.default.device, (list, tuple)) else sd.default.device
        for d in devices:
            if d["index"] == di:
                return d, devices
    except Exception:  # noqa: BLE001
        pass
    return (devices[0] if devices else None), devices


def run_live(sd, device: dict, seconds: float,
             channels: int | None) -> dict:
    """Capture raw multichannel audio + run production stack."""
    from src.voice.audio_capture_service import AudioCaptureService
    from src.voice.audio_device_manager import AudioDeviceManager
    from src.voice.wake_word_detection_service import (
        WakeWordDetectionService,
    )

    mgr = AudioDeviceManager(sd=sd)
    dev = None
    for d in mgr.list_input_devices():
        if d.index == int(device["index"]):
            dev = d
            break
    if dev is None:
        dev = mgr.resolve_device(None)  # fall back to default
    max_ch = int(getattr(dev, "max_input_channels", 1) or 1)
    use_ch = min(max_ch, channels or 4, 4)
    use_ch = max(use_ch, 1)

    # Ring buffer of raw (frames, channels) audio — memory only.
    ring = deque(maxlen=int(seconds * TARGET_SAMPLE_RATE))
    ring_frames = []

    def cb(indata, frames, time_info, status):
        arr = np.asarray(indata)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        ring_frames.append(arr.copy())

    prod_detections = []
    prod_stats_log = []

    cap = AudioCaptureService(sd=sd, device_manager=mgr)
    engine = create_engine()
    det = WakeWordDetectionService(
        cap, engine, threshold=0.25,
        on_detected=lambda kw: prod_detections.append(
            {"t": time.time(), "keyword": kw}),
    )

    # Open the diagnostic raw stream (same rate/channels as prod would
    # use) alongside the production capture stream.
    raw_stream = sd.InputStream(
        device=device["index"], channels=use_ch,
        samplerate=TARGET_SAMPLE_RATE, blocksize=1600,
        dtype="float32", callback=cb)

    t0 = time.monotonic()
    try:
        cap.start(device_id=dev.device_id)
        det.start()
        raw_stream.start()
        print(f"[diagnose] capturing {seconds:.0f}s on "
              f"{device['name']!r} ({use_ch} ch) — speak "
              "'Hello Jarvis' now, 3-5 times.")
        while time.monotonic() - t0 < seconds:
            time.sleep(1.0)
            st = det.stats()
            prod_stats_log.append({
                "t": round(time.monotonic() - t0, 1),
                "captured": st.get("frames_captured"),
                "processed": st.get("frames_processed"),
                "queue_depth": st.get("queue_depth"),
                "kws_rms": st.get("kws_input_rms"),
                "kws_peak": st.get("kws_input_peak"),
                "channel_rms": st.get("channel_rms"),
                "selected_channel": st.get("selected_channel"),
                "detections": st.get("detections"),
            })
            print(f"  t={time.monotonic()-t0:5.1f}s "
                  f"cap={st.get('frames_captured')} "
                  f"proc={st.get('frames_processed')} "
                  f"q={st.get('queue_depth')} "
                  f"kws_rms={st.get('kws_input_rms')} "
                  f"ch={st.get('channel_rms')} "
                  f"sel={st.get('selected_channel')} "
                  f"det={st.get('detections')}")
    finally:
        det.stop()
        cap.stop()
        raw_stream.stop()
        raw_stream.close()
        engine.close()

    # Concatenate raw capture -> (n_samples, channels)
    if ring_frames:
        raw = np.concatenate(ring_frames, axis=0)
    else:
        raw = np.zeros((0, use_ch), dtype=np.float32)
    n_ch = raw.shape[1] if raw.ndim > 1 else 1

    # Per-channel stats + independent KWS replay per channel.
    channel_report = []
    for c in range(n_ch):
        mono = raw[:, c].reshape(-1) if raw.ndim > 1 else raw.reshape(-1)
        rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        # speech-band estimate: 90th percentile of 100ms RMS
        if mono.size >= 1600:
            w = mono[: len(mono) // 1600 * 1600].reshape(-1, 1600)
            wrms = np.sqrt(np.mean(w * w, axis=1))
            p90 = float(np.percentile(wrms, 90))
            act = float(np.mean(wrms > 0.01))
        else:
            p90, act = 0.0, 0.0
        replay = _kws_detect(mono) if mono.size else {
            "detected": False, "engine": "n/a"}
        channel_report.append({
            "channel": c,
            "rms": round(rms, 6),
            "peak": round(peak, 5),
            "p90_frame_rms": round(p90, 5),
            "active_frame_ratio": round(act, 3),
            "kws_replay": replay,
        })

    # All-channel mean (what a naive average would feed KWS).
    if raw.ndim > 1 and raw.shape[1] > 1:
        mean_mono = raw.mean(axis=1).astype(np.float32)
        avg_report = _kws_detect(mean_mono)
    else:
        avg_report = {"detected": False, "engine": "n/a",
                      "note": "single channel"}

    # STT verification: transcribe high-energy speech segments so we
    # know what was actually said (in-memory only, never persisted).
    stt_report = _stt_check(raw, n_ch)

    final_cap = cap.stats()
    return {
        "device": {"index": device["index"],
                   "name": device["name"],
                   "channels": device["channels"]},
        "channels_opened": n_ch,
        "prod_stream_channels": final_cap.get("input_channels"),
        "prod_selected_channel": final_cap.get("selected_channel"),
        "prod_channel_switch_count":
            final_cap.get("channel_switch_count"),
        "prod_channel_rms": final_cap.get("channel_rms"),
        "prod_channel_peaks": final_cap.get("channel_peaks"),
        "prod_channel_ema": final_cap.get("channel_ema"),
        "prod_kws_max_rms": final_cap.get("max_rms"),
        "prod_detections": prod_detections,
        "prod_final_stats": {
            k: final_cap.get(k) for k in (
                "frames_captured", "frames_processed",
                "frames_dropped", "queue_depth", "signal_frames",
                "clipped_frames", "max_rms", "callback_age_seconds",
            )
        },
        "prod_stats_log": prod_stats_log,
        "channel_report": channel_report,
        "all_channel_mean_kws": avg_report,
        "stt_verification": stt_report,
        "raw_seconds": round(raw.shape[0] / TARGET_SAMPLE_RATE, 2)
        if raw.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--device", default=None,
                    help="substring of input device name "
                         "(default: Windows default input)")
    ap.add_argument("--index", type=int, default=None,
                    help="exact sounddevice input index")
    ap.add_argument("--channels", type=int, default=None,
                    help="force channel count (default: up to 4)")
    args = ap.parse_args()

    import sounddevice as sd

    result = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixture_sanity": fixture_sanity(),
        "input_devices": list_input_devices(sd),
    }
    if args.index is not None:
        device, _ = pick_device(sd, None)
        for d in result["input_devices"]:
            if d["index"] == args.index:
                device = d
                break
    else:
        device, _ = pick_device(sd, args.device)
    if device is None:
        result["error"] = "no input device found"
        _write(result)
        return 1
    result["target_device"] = device
    print(f"[diagnose] target device: {device}")
    result["live"] = run_live(sd, device, args.seconds, args.channels)
    _write(result)
    print(f"[diagnose] wrote {REPORTS/'v1031_channel_diagnostics.json'}")
    return 0


def _stt_check(raw: np.ndarray, n_ch: int) -> dict:
    """Transcribe speech segments from the strongest channel.

    Segments = contiguous runs of 100 ms frames with RMS > 0.02,
    merged across gaps < 0.6 s. Each segment is transcribed with the
    production offline STT engine. Audio stays in memory.
    """
    if raw.size == 0:
        return {"skipped": "no audio"}
    if raw.ndim > 1 and raw.shape[1] > 1:
        # Strongest channel by RMS — dead/reference channels must not
        # dilute the speech signal.
        ch_rms = [float(np.sqrt(np.mean(raw[:, c] ** 2)))
                  for c in range(raw.shape[1])]
        best = int(np.argmax(ch_rms))
        mono = raw[:, best].astype(np.float32)
    else:
        mono = raw.reshape(-1).astype(np.float32)
    hop = TARGET_SAMPLE_RATE // 10  # 100 ms
    n = len(mono) // hop
    if n == 0:
        return {"skipped": "audio too short"}
    w = mono[: n * hop].reshape(n, hop)
    wrms = np.sqrt(np.mean(w * w, axis=1))
    active = wrms > 0.02
    if not active.any():
        return {"skipped": "no speech-level segments",
                "max_frame_rms": round(float(wrms.max()), 5)}

    # Merge active frames into segments (gap tolerance 6 frames).
    segs, start, gap = [], None, 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 6:
                segs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        segs.append((start, n))
    segs = [(s, e) for s, e in segs if (e - s) >= 3]  # >=0.3 s
    if not segs:
        return {"skipped": "no segments >=0.3s"}

    from src.core.paths import stt_model_dir

    models = stt_model_dir()
    if not models.exists():
        return {"skipped": "STT model not installed"}
    from src.voice.stt_engine import create_recognizer
    rec = create_recognizer(models)
    if getattr(rec, "name", "") == "null":
        rec.close()
        return {"skipped": "STT unavailable"}

    out = []
    try:
        for s, e in segs[:8]:
            a, b = s * hop, e * hop
            seg = mono[a:b]
            pad = np.zeros(hop, dtype=np.float32)
            t0 = time.perf_counter()
            text = (rec.transcribe(np.concatenate([pad, seg, pad]))
                    or "").strip()
            out.append({
                "start_s": round(s * 0.1, 1),
                "dur_s": round((e - s) * 0.1, 1),
                "peak_rms": round(float(wrms[s:e].max()), 4),
                "transcript": text,
                "latency_ms": round(
                    (time.perf_counter() - t0) * 1000, 1),
            })
    finally:
        rec.close()
    return {"segments": out, "segment_count": len(out)}


def _write(result: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "v1031_channel_diagnostics.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
