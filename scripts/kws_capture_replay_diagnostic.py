"""Wake-word capture & replay diagnostic (manual, explicitly triggered).

Captures a few seconds of the EXACT mono float32 16 kHz audio that the
production KWS pipeline hands to the engine, then replays it through
controlled paths to localize a wake-word failure:

  Test A — live: the production AudioCaptureService +
           WakeWordDetectionService stack listens on the real
           microphone while we record every frame that reaches
           ``engine.process()``.
  Test B — replay: the recorded KWS-input frames are fed through a
           FRESH engine's ``process()`` (same 100 ms cadence).
  Test C — rescore: the recorded audio is replayed through a real
           WakeWordDetectionService backed by a paced replay queue, so
           the production activity-window + rescore path runs
           end-to-end. Additionally ``engine.rescore()`` is invoked
           directly on the whole clip with production anchors.

Privacy: nothing is recorded or persisted unless ``--save`` is given.
Saved WAVs are plain 16 kHz mono int16 and can be removed with
``--delete``.

Usage:
    py scripts/kws_capture_replay_diagnostic.py --list-devices
    py scripts/kws_capture_replay_diagnostic.py --seconds 6
    py scripts/kws_capture_replay_diagnostic.py            # Enter to stop
    py scripts/kws_capture_replay_diagnostic.py --seconds 6 --save diag.wav
    py scripts/kws_capture_replay_diagnostic.py --delete diag.wav
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.voice.audio_capture_service import (  # noqa: E402
    AudioCaptureService,
    TARGET_SAMPLE_RATE,
)
from src.voice.audio_device_manager import AudioDeviceManager  # noqa: E402
from src.voice.wake_word_detection_service import (  # noqa: E402
    WakeWordDetectionService,
)
from src.voice.wake_word_engine import create_engine  # noqa: E402

FRAME = 1600  # production frame: 100 ms @ 16 kHz


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
class _RecordingEngine:
    """Proxy that records every frame handed to ``process()``.

    This is the exact KWS input stream — post channel-select,
    post-resample, post-queue — with no extra copying inside the
    pipeline. All other engine attributes delegate to the real engine.
    """

    def __init__(self, inner):
        self._inner = inner
        self.frames: list[np.ndarray] = []
        self.lock = threading.Lock()

    @property
    def name(self):
        return getattr(self._inner, "name", "recording-proxy")

    def process(self, samples_16k):
        with self.lock:
            self.frames.append(
                np.asarray(samples_16k, dtype=np.float32).reshape(-1).copy()
            )
        return self._inner.process(samples_16k)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _ReplayCapture:
    """Capture fake that replays recorded frames at real-time cadence."""

    def __init__(self, frames: list[np.ndarray], pace: bool = True):
        self._frames = list(frames)
        self._idx = 0
        self._pace = pace
        self._next_at = time.monotonic()
        self._lock = threading.Lock()
        self.is_running = True
        self.paused = False
        self.device = None
        self.stats_calls = 0

    def get_frame(self, timeout: float = 0.25):
        with self._lock:
            if self._idx >= len(self._frames):
                return None
            frame = self._frames[self._idx]
            self._idx += 1
        if self._pace:
            # Maintain ~100 ms/frame so activity-window wall-clock
            # semantics match live capture.
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(min(wait, timeout))
            self._next_at = max(now, self._next_at) + 0.1
        return frame

    def stats(self):
        self.stats_calls += 1
        with self._lock:
            depth = len(self._frames) - self._idx
        return {
            "frames_captured": self._idx,
            "frames_queued": self._idx,
            "frames_dropped": 0,
            "queue_depth": depth,
            "running": self.is_running,
            "sample_rate": TARGET_SAMPLE_RATE,
            "device": "replay",
            "input_channels": 1,
            "selected_channel": None,
            "channel_rms": [],
            "channel_peaks": [],
        }

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def flush(self):
        with self._lock:
            self._idx = len(self._frames)


def _concat(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames).astype(np.float32)


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _save_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _delete_wav(path: Path) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
        print(f"deleted {p}")
    else:
        print(f"not found: {p}")


# ----------------------------------------------------------------------
# phases
# ----------------------------------------------------------------------
def run_live_capture(args) -> dict:
    mgr = AudioDeviceManager()
    devices = [d for d in mgr.list_input_devices()]
    if args.list_devices:
        print("Input devices:")
        for d in devices:
            print(f"  index={d.index} id={d.device_id!r} "
                  f"name={d.name!r} rate={d.default_samplerate} "
                  f"max_ch={getattr(d, 'max_input_channels', '?')}")
        return {}

    device_id = None
    if args.device is not None:
        # Accept numeric index or substring of the device name/id.
        for d in devices:
            if (str(args.device) == str(d.index)
                    or str(args.device) == str(d.device_id)
                    or str(args.device).lower() in d.name.lower()):
                device_id = d.device_id
                break
        if device_id is None:
            raise SystemExit(f"device {args.device!r} not found; "
                             "run with --list-devices")

    capture = AudioCaptureService(device_manager=mgr)
    inner_engine = create_engine()
    if getattr(inner_engine, "name", "") != "sherpa-onnx-kws":
        raise SystemExit("KWS model unavailable (null engine); "
                         "check VOICEOS_KWS_MODEL_DIR / model files")
    engine = _RecordingEngine(inner_engine)

    live_hits: list[dict] = []
    det = WakeWordDetectionService(
        capture=capture,
        engine=engine,
        threshold=args.threshold,
        cooldown_seconds=args.cooldown,
        on_detected=lambda kw: live_hits.append(
            {"t": time.time(), "keyword": kw}),
    )

    print("=" * 70)
    print("TEST A — LIVE CAPTURE")
    print("Speak 'Hello Jarvis' a few times (normal voice, ~20 cm),")
    print("pausing ~2 s between attempts.")
    print("=" * 70)
    capture.start(device_id=device_id)
    dev = capture.device
    det.start()
    print(f"device={dev.name!r} rate={capture.sample_rate} "
          f"ch={capture.input_channels} sel_ch={capture._selected_channel}")
    t0 = time.monotonic()
    try:
        if args.seconds:
            while time.monotonic() - t0 < args.seconds:
                time.sleep(0.5)
                _progress(det, engine, time.monotonic() - t0)
        else:
            print("press Enter to stop capture...")
            # Read Enter on a thread so the main thread can print.
            done = threading.Event()

            def _wait_enter():
                try:
                    input()
                except EOFError:
                    pass
                done.set()

            threading.Thread(target=_wait_enter, daemon=True).start()
            while not done.wait(timeout=0.5):
                _progress(det, engine, time.monotonic() - t0)
    finally:
        elapsed = time.monotonic() - t0
        det.stop()
        capture.stop()
        try:
            inner_engine.close()
        except Exception:  # noqa: BLE001
            pass

    with engine.lock:
        frames = list(engine.frames)
    audio = _concat(frames)
    cap_stats = capture.stats()
    result = {
        "test": "A_live",
        "elapsed_seconds": round(elapsed, 2),
        "device": dev.name if dev else None,
        "sample_rate": capture.sample_rate,
        "input_channels": capture.input_channels,
        "selected_channel": cap_stats.get("selected_channel"),
        "frames_recorded": len(frames),
        "audio_seconds": round(audio.size / TARGET_SAMPLE_RATE, 3),
        "rms": round(_rms(audio), 5),
        "peak": round(float(np.max(np.abs(audio))) if audio.size else 0.0, 4),
        "detections": live_hits,
        "frames_dropped": cap_stats.get("frames_dropped"),
        "overflow_callbacks": cap_stats.get("overflow_callbacks"),
        "clipped_frames": cap_stats.get("clipped_frames"),
        "last_activity_window": det.last_activity_window,
        "activity_windows": len(det.activity_windows),
    }
    _print_result(result)
    return {"live": result, "audio": audio, "frames": frames}


def _progress(det, engine, elapsed):
    st = det.stats()
    with engine.lock:
        n = len(engine.frames)
    print(f"\r  t={elapsed:5.1f}s frames={n} "
          f"proc={st.get('frames_processed')} q={st.get('queue_depth')} "
          f"det={st.get('detections')} rms={st.get('kws_input_rms')}",
          end="", flush=True)


def _print_result(res: dict) -> None:
    print()
    for k, v in res.items():
        if k == "last_activity_window":
            continue
        print(f"  {k}: {v}")
    win = res.get("last_activity_window")
    if win:
        print(f"  last_window: detected={win.get('keyword_detected')} "
              f"rescored={win.get('rescored')} "
              f"method={win.get('detection_method')}")


def run_replay_fresh(audio: np.ndarray, frames: list[np.ndarray],
                     threshold: float = 0.25) -> dict:
    """Test B: fresh engine, production-style process() per frame."""
    engine = create_engine()
    if getattr(engine, "name", "") != "sherpa-onnx-kws":
        return {"test": "B_replay_fresh", "error": "engine unavailable"}
    hits = []
    t0 = time.monotonic()
    try:
        # Feed exactly the recorded frames (100 ms each) — identical
        # to what live process() consumed.
        for i, frame in enumerate(frames):
            conf = engine.process(np.ascontiguousarray(frame))
            kw = engine.keyword_detected()
            if kw:
                hits.append({"frame": i, "t_s": round(i * 0.1, 2),
                             "keyword": kw, "confidence": conf})
        elapsed = time.monotonic() - t0
    finally:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass
    return {
        "test": "B_replay_fresh",
        "frames": len(frames),
        "detections": hits,
        "elapsed_seconds": round(elapsed, 2),
    }


def run_replay_service(frames: list[np.ndarray],
                       threshold: float = 0.25) -> dict:
    """Test C1: full detection service over paced replay (windows +
    rescore path exactly as production)."""
    engine = create_engine()
    if getattr(engine, "name", "") != "sherpa-onnx-kws":
        return {"test": "C_replay_service", "error": "engine unavailable"}
    cap = _ReplayCapture(frames, pace=True)
    hits: list[dict] = []
    det = WakeWordDetectionService(
        capture=cap, engine=engine, threshold=threshold,
        cooldown_seconds=0.0,
        on_detected=lambda kw: hits.append(
            {"t": time.time(), "keyword": kw}),
    )
    t0 = time.monotonic()
    det.start()
    # Wait until replay exhausted + hangover window closes.
    deadline = t0 + (len(frames) * 0.1) + 4.0
    while time.monotonic() < deadline:
        if cap._idx >= len(frames) and det._activity_open is False \
                and cap._idx >= len(frames):
            # give the worker a moment to drain + close windows
            time.sleep(0.5)
            if not det._activity_open:
                break
        time.sleep(0.2)
    det.stop()
    try:
        engine.close()
    except Exception:  # noqa: BLE001
        pass
    windows = [
        {
            "window_id": w.get("window_id"),
            "keyword_detected": w.get("keyword_detected"),
            "rescored": w.get("rescored"),
            "detection_method": w.get("detection_method"),
            "rescore_succeeded": w.get("rescore_succeeded"),
            "rescore_elapsed_ms": w.get("rescore_elapsed_ms"),
            "rescore_error": w.get("rescore_error"),
            "audio_seconds": w.get("audio_seconds"),
        }
        for w in det.activity_windows
    ]
    return {
        "test": "C_replay_service",
        "frames": len(frames),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "detections": hits,
        "activity_windows": windows,
        "frames_processed": det.frames_processed,
        "inference_errors": det.inference_errors,
    }


def run_replay_rescore(audio: np.ndarray) -> dict:
    """Test C2: direct engine.rescore() on the whole clip."""
    engine = create_engine()
    if getattr(engine, "name", "") != "sherpa-onnx-kws":
        return {"test": "C_replay_rescore", "error": "engine unavailable"}
    from src.voice import kws_rescore  # noqa: E402

    policy = kws_rescore.RescorePolicy()
    t0 = time.monotonic()
    try:
        result = engine.rescore(
            audio,
            lead_seconds=policy.lead_seconds,
            trail_seconds=policy.trail_seconds,
        )
        elapsed = time.monotonic() - t0
    finally:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass
    return {
        "test": "C_replay_rescore",
        "audio_seconds": round(audio.size / TARGET_SAMPLE_RATE, 3),
        "anchors_requested": len(policy.lead_seconds),
        "result": result,
        "elapsed_ms": round(elapsed * 1000.0, 1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true",
                   help="list input devices and exit")
    p.add_argument("--device", default=None,
                   help="device index, id, or name substring "
                        "(default: production/auto resolution)")
    p.add_argument("--seconds", type=float, default=None,
                   help="capture duration (default: until Enter)")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--cooldown", type=float, default=3.0)
    p.add_argument("--save", type=Path, default=None,
                   help="save captured KWS-input audio as 16 kHz WAV")
    p.add_argument("--delete", type=Path, default=None,
                   help="delete a previously saved diagnostic WAV")
    p.add_argument("--skip-replay", action="store_true",
                   help="only run Test A (live capture)")
    args = p.parse_args(argv)

    if args.delete is not None:
        _delete_wav(args.delete)
        return 0

    capres = run_live_capture(args)
    if not capres:
        return 0

    audio = capres["audio"]
    frames = capres["frames"]
    if audio.size == 0:
        print("no KWS-input frames captured; nothing to replay")
        return 1

    if args.save is not None:
        _save_wav(args.save, audio)
        print(f"saved {audio.size / TARGET_SAMPLE_RATE:.2f}s of KWS-input "
              f"audio to {args.save}")

    if args.skip_replay:
        return 0

    print()
    print("=" * 70)
    print("TEST B — REPLAY THROUGH FRESH ENGINE (production frames)")
    print("=" * 70)
    res_b = run_replay_fresh(audio, frames, threshold=args.threshold)
    _print_result(res_b)

    print()
    print("=" * 70)
    print("TEST C1 — REPLAY THROUGH DETECTION SERVICE (windows+rescore)")
    print("=" * 70)
    res_c1 = run_replay_service(frames, threshold=args.threshold)
    _print_result(res_c1)

    print()
    print("=" * 70)
    print("TEST C2 — DIRECT engine.rescore() ON WHOLE CLIP")
    print("=" * 70)
    res_c2 = run_replay_rescore(audio)
    _print_result(res_c2)

    print()
    print("SUMMARY")
    print(f"  A live detections:      {len(capres['live']['detections'])}")
    print(f"  B fresh replay:         {len(res_b.get('detections', []))}")
    print(f"  C1 service replay:      {len(res_c1.get('detections', []))}")
    c2 = res_c2.get("result")
    print(f"  C2 direct rescore:      {'HIT ' + repr(c2) if c2 else 'miss'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
