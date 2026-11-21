"""Physical voice-recognition baseline benchmark for VoiceOS.

DIAGNOSTIC ONLY — uses the production audio capture, KWS engine,
thresholds, VAD and STT model exactly as configured, but NEVER routes
transcripts to command handlers and NEVER executes commands. No raw
PCM is persisted; only metrics, transcripts and timings are written.

Subcommands:

    kws       20 positive "Hello Jarvis" trials (first-attempt recall,
              live-vs-rescore method, latency, signal metrics)
    kws-neg   10 negative trials (ordinary phrases, expect no wake)
    stt       20 known-phrase STT trials (exact/WER/CER vs ground truth)

Each trial is user-paced: the script waits for Enter, arms a window,
then records what the production pipeline actually did. Results are
appended to reports/voice_benchmark_<mode>_<timestamp>.json and .csv.

Usage (VoiceOS venv, repo root):

    py scripts/voice_recognition_benchmark.py kws
    py scripts/voice_recognition_benchmark.py kws-neg
    py scripts/voice_recognition_benchmark.py stt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"

STT_PHRASES = [
    "Open Chrome.",
    "Move Chrome to the left monitor.",
    "What is today's date?",
    "What time is it?",
    "Set volume to twenty percent.",
    "Open the Downloads folder.",
    "Switch to Chrome.",
    "Maximize Chrome.",
    "Minimize Chrome.",
    "List running applications.",
    "Search for Java developer jobs in Hyderabad.",
    "Open a new tab.",
    "Search YouTube for Spring Boot tutorials.",
    "Open GitHub.",
    "Move the active window to the right monitor.",
    "Show the desktop.",
    "Open the Settings app.",
    "Find my resumes folder.",
    "Search for PostgreSQL interview questions.",
    "Tell me today's date and time.",
]

NEGATIVE_PHRASES = [
    "Good morning.",
    "What is the weather today?",
    "Open Chrome.",
    "Tell me the time.",
    "Find my downloads folder.",
    "What is happening today?",
    "Open the browser.",
    "Show running applications.",
    "How are you?",
    "Search for Java tutorials.",
]


def _resources():
    """CPU/RAM snapshot via psutil (already a project dependency)."""
    try:
        import psutil
        proc = psutil.Process()
        return {
            "cpu_percent": proc.cpu_percent(interval=None),
            "rss_mb": round(proc.memory_info().rss / 1e6, 1),
            "avail_ram_mb": round(psutil.virtual_memory().available / 1e6, 1),
        }
    except Exception:  # noqa: BLE001
        return {}


def _capture_snapshot(cap) -> dict:
    """Signal/queue metrics for the window — no PCM, metrics only."""
    try:
        s = cap.stats()
    except Exception:  # noqa: BLE001
        return {}
    keys = (
        "frames_captured", "frames_dropped", "frames_queued",
        "queue_depth", "callback_seq", "callback_age_seconds",
        "channel_rms", "channel_peaks", "max_channel_rms",
        "selected_channel", "input_channels", "clipped_frames",
        "signal_frames", "indata_rms", "indata_peak", "last_rms",
        "last_peak", "sample_rate", "device", "paused",
        "overflow_callbacks", "status_warnings",
    )
    return {k: s.get(k) for k in keys}


def _percentiles(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"p50": None, "p95": None, "n": 0}
    def pct(p):
        i = min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1))))
        return round(vals[i], 4)
    return {"p50": pct(50), "p95": pct(95), "n": len(vals)}


def _wer_cer(reference: str, hypothesis: str):
    """Word/char error rates — whitespace-normalized tokens only."""
    def norm(t):
        return [w for w in t.strip().lower().split() if w]
    ref, hyp = norm(reference), norm(hypothesis)

    def edit_distance(a, b):
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(
                    prev[j] + 1, cur[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                ))
            prev = cur
        return prev[-1]

    wer = edit_distance(ref, hyp) / max(1, len(ref))
    cer = edit_distance(list(reference.lower()), list(hypothesis.lower())) \
        / max(1, len(reference))
    return round(wer, 4), round(cer, 4)


class _Pipeline:
    """Production capture + detector (KWS modes) or STT (stt mode)."""

    def __init__(self, mode: str):
        from src.voice.audio_capture_service import AudioCaptureService
        from src.voice.audio_device_manager import AudioDeviceManager
        from src.voice.wake_word_engine import create_engine
        from src.voice.wake_word_detection_service import (
            WakeWordDetectionService,
        )

        self.mode = mode
        self._mgr = AudioDeviceManager()
        self._cap = AudioCaptureService(device_manager=self._mgr)
        self._engine = create_engine()
        self._det = None
        self._stt = None
        self._wake_event = threading.Event()
        self._stop = threading.Event()

    # -- lifecycle ----------------------------------------------------
    def start_kws(self, threshold: float, cooldown: float):
        device = self._cap.start(device_id=None)
        print(f"[capture] {device.name} @ {self._cap.sample_rate:.0f} Hz")
        self._new_detector(threshold, cooldown)

    def _new_detector(self, threshold, cooldown):
        from src.voice.wake_word_detection_service import (
            WakeWordDetectionService,
        )
        if self._det is not None:
            try:
                self._det.stop()
            except Exception:  # noqa: BLE001
                pass
        det = WakeWordDetectionService(
            capture=self._cap,
            engine=self._engine,
            threshold=threshold,
            cooldown_seconds=cooldown,
            on_detected=lambda kw: self._on_wake(kw),
        )
        self._det = det
        det.start()

    def _on_wake(self, keyword: str):
        self._wake_event.set()
        self.last_wake = dict(getattr(self._det, "last_detection", None)
                              or {})

    def start_stt(self, **stt_kw):
        from src.voice.speech_recognition_service import (
            SpeechRecognitionService,
        )
        from src.voice.stt_engine import create_recognizer
        device = self._cap.start(device_id=None)
        print(f"[capture] {device.name} @ {self._cap.sample_rate:.0f} Hz")
        self._stt = SpeechRecognitionService(
            capture=self._cap,
            recognizer=create_recognizer(),
            **stt_kw,
        )
        self.last_transcript = None
        self.last_no_speech = None
        self._stt.set_callbacks(
            on_transcript=lambda t: setattr(
                self, "last_transcript", t),
            on_no_speech=lambda r: setattr(
                self, "last_no_speech", r),
            on_error=lambda m: setattr(
                self, "last_stt_error", m),
        )
        self._stt.start()

    def stop(self):
        self._stop.set()
        for svc in (self._det, self._stt):
            if svc is not None:
                try:
                    svc.stop()
                except Exception:  # noqa: BLE001
                    pass
        try:
            self._cap.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._engine.close()
        except Exception:  # noqa: BLE001
            pass

    # -- readiness ----------------------------------------------------
    def ready(self) -> bool:
        """Evidence-based READY: callbacks fresh + detector consuming."""
        if not self._cap.is_running:
            return False
        stall = self._cap.callback_stall_seconds()
        if stall is None or stall > 2.0:
            return False
        if self._det is not None:
            return self._det.is_running
        if self._stt is not None:
            return self._stt.is_running
        return True


def _wait_ready(pipe: _Pipeline, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pipe.ready():
            return True
        time.sleep(0.1)
    return False


def _pct(values):
    return _percentiles(values)


# ----------------------------------------------------------------------
def run_kws(trials: int, negative: bool = False,
            threshold: float = 0.25, cooldown: float = 3.0):
    from src.voice.wake_word_engine import create_engine  # noqa: F401
    phrases = NEGATIVE_PHRASES if negative else None
    pipe = _Pipeline("kws")
    pipe.start_kws(threshold=threshold, cooldown=cooldown)
    results = []
    try:
        if not _wait_ready(pipe):
            print("WARN: pipeline never reached READY; continuing anyway")
        for i in range(1, trials + 1):
            input(f"\n=== Trial {i}/{trials} ===\n"
                  + (f"Say: \"{phrases[i-1]}\" (NO wake phrase)\n"
                     if negative else
                     "Wait for READY, then say \"Hello Jarvis\" ONCE.\n")
                  + "Press Enter to arm the 5 s detection window...")
            # Fresh detector generation per trial = clean standby cycle.
            pipe._new_detector(threshold, cooldown)
            pipe._wake_event.clear()
            pipe.last_wake = {}
            t_arm = time.monotonic()
            res = {
                "trial": i,
                "phrase": phrases[i - 1] if negative else "Hello Jarvis",
                "armed_monotonic": t_arm,
                "resources": _resources(),
                "capture_before": _capture_snapshot(pipe._cap),
            }
            detected = pipe._wake_event.wait(timeout=5.0)
            res["detected"] = bool(detected)
            res["detection_latency_seconds"] = (
                round(time.monotonic() - t_arm, 3) if detected else None
            )
            res["detection"] = pipe.last_wake or None
            res["capture_after"] = _capture_snapshot(pipe._cap)
            det_stats = (
                pipe._det.stats() if pipe._det is not None else {}
            )
            res["detector_stats"] = {
                k: det_stats.get(k) for k in (
                    "frames_processed", "inference_errors",
                    "detections", "queue_depth", "frames_dropped",
                )
            }
            results.append(res)
            verdict = ("FALSE WAKE!" if negative and detected
                       else "missed" if not negative and not detected
                       else "ok")
            print(f"  -> {verdict} "
                  f"(method={res['detection'].get('detection_method') if res['detection'] else '-'}"
                  f" latency={res['detection_latency_seconds']})")
            time.sleep(0.5)  # let queues settle before next arm
    finally:
        pipe.stop()
    return results


def run_stt():
    pipe = _Pipeline("stt")
    pipe.start_stt()
    results = []
    try:
        if not _wait_ready(pipe):
            print("WARN: STT pipeline never reached READY")
        for i, phrase in enumerate(STT_PHRASES, 1):
            input(f"\n=== STT trial {i}/{len(STT_PHRASES)} ===\n"
                  f"Expected: \"{phrase}\"\n"
                  "Speak it once at normal volume, then press Enter...")
            pipe.last_transcript = None
            pipe.last_no_speech = None
            pipe.last_stt_error = None
            t0 = time.monotonic()
            # Wait for the utterance to finalize (transcript/no_speech/error)
            # or a 12 s cap.
            while time.monotonic() - t0 < 12.0:
                if (pipe.last_transcript is not None
                        or pipe.last_no_speech is not None
                        or getattr(pipe, "last_stt_error", None)):
                    break
                time.sleep(0.05)
            attempt = dict(pipe._stt.last_attempt or {})
            actual = pipe.last_transcript or ""
            outcome = (
                "transcript" if pipe.last_transcript is not None
                else "no_speech" if pipe.last_no_speech is not None
                else "stt_error" if getattr(pipe, "last_stt_error", None)
                else "timeout"
            )
            wer, cer = _wer_cer(phrase, actual) if actual else (None, None)
            rec = {
                "trial": i,
                "expected": phrase,
                "actual": actual,
                "exact_match": actual.strip().lower()
                == phrase.strip().lower().rstrip(".!?"),
                "wer": wer,
                "cer": cer,
                "vad_outcome": outcome,
                "no_speech_reason": pipe.last_no_speech,
                "attempt": attempt,
                "decode_seconds": attempt.get("decode_seconds")
                or attempt.get("stt_seconds"),
                "resources": _resources(),
                "elapsed_seconds": round(time.monotonic() - t0, 3),
            }
            results.append(rec)
            print(f"  -> {outcome}: {actual!r} "
                  f"(WER={wer} CER={cer})")
            time.sleep(0.4)
    finally:
        pipe.stop()
    return results


def _write(results, mode: str):
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = REPORTS / f"voice_benchmark_{mode}_{stamp}.json"
    cpath = REPORTS / f"voice_benchmark_{mode}_{stamp}.csv"
    jpath.write_text(json.dumps(
        {"mode": mode, "generated": stamp, "trials": results},
        indent=2, default=str))
    if results:
        flat = []
        for r in results:
            row = {k: v for k, v in r.items()
                   if k not in ("capture_before", "capture_after",
                                "detector_stats", "attempt", "detection")}
            if "attempt" in r:
                row.update({f"att_{k}": v for k, v in
                            (r.get("attempt") or {}).items()})
            if "detection" in r and r.get("detection"):
                row.update({f"det_{k}": v for k, v in
                            r["detection"].items()})
            flat.append(row)
        keys = sorted({k for row in flat for k in row})
        with cpath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
    print(f"\nWrote {jpath}\nWrote {cpath}")


def _summary_kws(results, negative: bool):
    det = [r for r in results if r["detected"]]
    lat = [r["detection_latency_seconds"] for r in det]
    methods = {}
    for r in det:
        m = (r.get("detection") or {}).get("detection_method") or "?"
        methods[m] = methods.get(m, 0) + 1
    print("\n===== KWS SUMMARY =====")
    print(f"trials={len(results)} detected={len(det)} "
          f"missed={len(results) - len(det)}")
    print(f"methods={methods}")
    print(f"latency {_pct(lat)}")
    if negative:
        print(f"FALSE WAKE count: {len(det)}")


def _summary_stt(results):
    n = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    wers = [r["wer"] for r in results if r["wer"] is not None]
    cers = [r["cer"] for r in results if r["cer"] is not None]
    outcomes = {}
    for r in results:
        outcomes[r["vad_outcome"]] = outcomes.get(r["vad_outcome"], 0) + 1
    dec = [r["decode_seconds"] for r in results
           if isinstance(r.get("decode_seconds"), (int, float))]
    print("\n===== STT SUMMARY =====")
    print(f"trials={n} exact={exact} ({exact / max(1, n):.0%})")
    print(f"mean WER={sum(wers) / max(1, len(wers)):.3f} "
          f"mean CER={sum(cers) / max(1, len(cers)):.3f}")
    print(f"outcomes={outcomes}")
    print(f"decode latency {_pct(dec)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=
                                 argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("kws", help="positive wake-word trials")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--cooldown", type=float, default=3.0)
    p = sub.add_parser("kws-neg", help="negative wake-word trials")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--cooldown", type=float, default=3.0)
    sub.add_parser("stt", help="known-phrase STT trials")
    args = ap.parse_args(argv)

    if args.mode == "kws":
        res = run_kws(args.trials, threshold=args.threshold,
                      cooldown=args.cooldown)
        _summary_kws(res, negative=False)
        _write(res, "kws")
    elif args.mode == "kws-neg":
        res = run_kws(args.trials, negative=True,
                      threshold=args.threshold, cooldown=args.cooldown)
        _summary_kws(res, negative=True)
        _write(res, "kws_neg")
    else:
        res = run_stt()
        _summary_stt(res)
        _write(res, "stt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
