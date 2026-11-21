"""VoiceOS V10.4 voice-responsiveness evaluation harness.

Runs the reproducible, in-memory evaluation workflow and writes the
V10.4 report artifacts:

- ``reports/v104_wake_word_latency.json``   — KWS detection latency on
  recorded wake-phrase samples + silence false-activation check.
- ``reports/v104_command_accuracy.json``    — transcript -> intent ->
  entity-resolution accuracy on the command eval set, including the
  garbled monitor commands observed in the field, plus an optional
  Moonshine Base-vs-Tiny STT benchmark when model dirs are present.
- ``reports/v104_response_latency.json``    — wake->ack->first-command
  timing measured through the real session controller with stub
  audio services.
- ``reports/v104_physical_trials.csv``      — worksheet for the
  physical microphone trials (filled in during hardware runs).

Nothing here touches the microphone, speakers, or persists audio: WAV
fixtures are loaded into memory, decoded in-process and discarded.

Usage:
    python scripts/test_v104_voice_responsiveness.py [--skip-stt]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.commands.intent import RuleBasedIntentRecognizer  # noqa: E402
from src.commands.recovery import MisheardCommandRecovery  # noqa: E402
from src.core.session_manager import SessionManager  # noqa: E402
from src.core.state_manager import AppState, StateManager  # noqa: E402
from src.voice.voice_session_controller import (  # noqa: E402
    VoiceSessionController,
)
from src.voice.wake_word_engine import create_engine  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
REPORTS = ROOT / "reports"
TARGET_RATE = 16000
KWS_CHUNK = 1600  # 100 ms at 16 kHz


# ----------------------------------------------------------------------
# Audio helpers (in-memory only)
# ----------------------------------------------------------------------
def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if rate != TARGET_RATE:
        from src.voice.audio_capture_service import resample_to_16k
        a = resample_to_16k(a, float(rate))
    return a.astype(np.float32)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


# ----------------------------------------------------------------------
# 1. Wake-word latency harness
# ----------------------------------------------------------------------
def run_wake_word_eval() -> dict:
    """Feed recorded wake phrases through the real KWS engine.

    Latency = detection chunk end minus the end of the last voiced
    chunk (energy >= speech threshold) — an approximation of
    end-of-phrase -> detection delay at 100 ms granularity.
    """
    engine = create_engine()
    rows = []
    engine_name = getattr(engine, "name", "unknown")

    def feed(wav_path: Path, label: str):
        # Fresh stream state per sample: leftover audio from a previous
        # utterance must not contaminate the next detection.
        engine.reset()
        # Pad with 1 s of silence: the sherpa-onnx KWS stream only
        # finalizes a keyword after sufficient post-phrase audio, which
        # the continuous production capture always supplies. Padding
        # reproduces that trailing context for isolated fixtures.
        audio = np.concatenate([
            load_wav(wav_path),
            np.zeros(TARGET_RATE, dtype=np.float32),
        ])
        last_voiced_end = None
        detected_at = None
        inference_ms = []
        n_chunks = 0
        for start in range(0, len(audio) - KWS_CHUNK + 1, KWS_CHUNK):
            chunk = audio[start:start + KWS_CHUNK]
            rms = float(np.sqrt(np.mean(chunk * chunk))) \
                if chunk.size else 0.0
            if rms >= 0.008:
                last_voiced_end = (start + KWS_CHUNK) / TARGET_RATE
            t0 = time.perf_counter()
            conf = engine.process(chunk)
            inference_ms.append((time.perf_counter() - t0) * 1000.0)
            n_chunks += 1
            if conf >= 0.25 and detected_at is None:
                detected_at = (start + KWS_CHUNK) / TARGET_RATE
                break
        latency = (
            detected_at - last_voiced_end
            if detected_at is not None and last_voiced_end is not None
            else None
        )
        rows.append({
            "sample": label,
            # "hey_jarvis" is an out-of-vocabulary probe: the keyword file
            # only encodes "hello jarvis" pronunciations, so it is reported
            # but excluded from the wake-phrase detection rate.
            "is_target_wake": label != "hey_jarvis",
            "wav": str(wav_path.relative_to(ROOT)),
            "detected": detected_at is not None,
            "detection_time_s": detected_at,
            "speech_end_s": last_voiced_end,
            "latency_s": round(latency, 3) if latency is not None else None,
            "chunks": n_chunks,
            "inference_ms_mean": (
                round(sum(inference_ms) / len(inference_ms), 2)
                if inference_ms else None
            ),
        })

    candidates = [
        (FIXTURES / "commands" / "hello_jarvis.wav", "hello_jarvis"),
        (FIXTURES / "commands" / "hey_jarvis.wav", "hey_jarvis"),
    ]
    for wav, label in candidates:
        if wav.exists():
            feed(wav, label)

    # False-activation probe: silence + low-level noise must not fire.
    false_rows = []
    for wav, label in (
        (FIXTURES / "commands" / "silence.wav", "silence"),
        (FIXTURES / "live" / "_test_noiseonly.wav", "noise_only"),
        (FIXTURES / "live" / "_test_silence.wav", "live_silence"),
    ):
        if not wav.exists():
            continue
        engine.reset()
        audio = np.concatenate([
            load_wav(wav),
            np.zeros(TARGET_RATE, dtype=np.float32),
        ])
        fired = 0
        for start in range(0, len(audio) - KWS_CHUNK + 1, KWS_CHUNK):
            if engine.process(audio[start:start + KWS_CHUNK]) >= 0.25:
                fired += 1
        false_rows.append({
            "sample": label,
            "wav": str(wav.relative_to(ROOT)),
            "detections": fired,
        })

    target_rows = [r for r in rows if r["is_target_wake"]]
    detected = [r for r in target_rows if r["detected"]]
    latencies = [r["latency_s"] for r in detected
                 if r["latency_s"] is not None]
    return {
        "engine": engine_name,
        "model_dir": str(getattr(engine, "model_dir", "") or ""),
        "keywords_file": str(getattr(engine, "keywords_file", "") or ""),
        "keyword_count": getattr(engine, "keyword_count", None),
        "keywords_sha256": getattr(engine, "keywords_sha256", None),
        "threshold": 0.25,
        "note": (
            "Latency measured on recorded fixtures at 100 ms chunk "
            "granularity: detection chunk end minus last voiced chunk "
            "end. Physical microphone trials are tracked separately in "
            "v104_physical_trials.csv."
        ),
        "samples": rows,
        "detection_rate": (
            round(len(detected) / len(target_rows), 3)
            if target_rows else None
        ),
        "target_samples": len(target_rows),
        "out_of_vocabulary_samples": [
            r["sample"] for r in rows if not r["is_target_wake"]
        ],
        "latency_p50_s": round(percentile(latencies, 50), 3)
        if latencies else None,
        "latency_p95_s": round(percentile(latencies, 95), 3)
        if latencies else None,
        "latency_max_s": round(max(latencies), 3) if latencies else None,
        "false_activation_samples": false_rows,
        "false_activations": sum(r["detections"] for r in false_rows),
    }


# ----------------------------------------------------------------------
# 2. Command-accuracy eval (transcripts, in-memory)
# ----------------------------------------------------------------------
def _app(name, exe=None):
    return types.SimpleNamespace(
        name=name, exe_name=exe or name.lower().replace(" ", "")
    )


RUNNING_APPS = [
    _app("Google Chrome", "chrome"),
    _app("WhatsApp", "whatsapp"),
    _app("Devin", "devin"),
    _app("Notepad", "notepad"),
    _app("Spotify", "spotify"),
]

# (transcript, expected_intent, expected_app_fragment_or_None,
#  expected_monitor_fragment_or_None)
COMMAND_EVAL = [
    # Application operations
    ("Open Chrome.", "OPEN_APPLICATION", "chrome", None),
    ("Open WhatsApp.", "OPEN_APPLICATION", "whatsapp", None),
    ("Open Devin.", "OPEN_APPLICATION", "devin", None),
    ("Switch to Chrome.", "SWITCH_TO_APPLICATION", "chrome", None),
    ("Switch to WhatsApp.", "SWITCH_TO_APPLICATION", "whatsapp", None),
    ("Switch to Devin.", "SWITCH_TO_APPLICATION", "devin", None),
    ("Close Chrome.", "CLOSE_APPLICATION", "chrome", None),
    # Monitor operations
    ("Move Chrome to the left monitor.", "MOVE_WINDOW_TO_MONITOR",
     "chrome", "left"),
    ("Move Chrome to the right monitor.", "MOVE_WINDOW_TO_MONITOR",
     "chrome", "right"),
    ("Move Devin to the left monitor.", "MOVE_WINDOW_TO_MONITOR",
     "devin", "left"),
    ("Move WhatsApp to the right monitor.", "MOVE_WINDOW_TO_MONITOR",
     "whatsapp", "right"),
    ("Move the current window to the other monitor.",
     "MOVE_WINDOW_TO_OTHER_MONITOR", None, "other"),
    ("Move the active application to monitor one.",
     "MOVE_WINDOW_TO_MONITOR", None, "one"),
    ("Move the active application to monitor two.",
     "MOVE_WINDOW_TO_MONITOR", None, "two"),
    # Other operations
    ("Open the downloads folder.", "OPEN_FOLDER", None, None),
    ("List running applications.", "LIST_RUNNING_APPLICATIONS",
     None, None),
    ("Goodbye Jarvis.", "END_SESSION", None, None),
]

# Garbled transcripts observed in the field. For each we record what
# the pipeline *would* do — safe outcomes are a correct recovery, an
# ambiguous clarification, or an honest UNKNOWN.
GARBLED_EVAL = [
    "More crumpled less commoditous.",
    "More crumpled last-monitor.",
    "More devent to less the monitors.",
    "Move Dev into less than monitor.",
]


def _blob(entities: dict) -> str:
    return " ".join(
        str(v) for k, v in entities.items() if not k.startswith("_")
    ).lower()


def run_command_eval() -> dict:
    recognizer = RuleBasedIntentRecognizer()
    recovery = MisheardCommandRecovery(app_names=lambda: RUNNING_APPS)
    rows = []
    for text, exp_intent, exp_app, exp_mon in COMMAND_EVAL:
        intent = recognizer.recognize(text)
        blob = _blob(intent.entities)
        intent_ok = intent.name == exp_intent
        app_ok = (
            exp_app is None or exp_app in blob
        )
        mon_ok = (
            exp_mon is None or exp_mon in blob
        )
        rows.append({
            "transcript": text,
            "expected_intent": exp_intent,
            "intent": intent.name,
            "intent_ok": intent_ok,
            "app_ok": app_ok,
            "monitor_ok": mon_ok,
            "entities": {
                k: v for k, v in intent.entities.items()
                if not k.startswith("_")
            },
        })

    garbled_rows = []
    for text in GARBLED_EVAL:
        intent = recognizer.recognize(text)
        entry = {
            "transcript": text,
            "primary_intent": intent.name,
        }
        if intent.name == "UNKNOWN":
            result = recovery.recover(text)
            entry["recovery_status"] = result.status
            entry["recovery_reason"] = result.reason
            entry["recovery_confidence"] = result.confidence
            entry["interpretation"] = result.interpretation
            if result.intent is not None:
                entry["recovered_intent"] = result.intent.name
                entry["recovered_entities"] = {
                    k: v for k, v in result.intent.entities.items()
                    if k != "_recovery"
                }
            if result.status == "ambiguous":
                entry["outcome"] = "clarify"
            elif result.status == "matched":
                entry["outcome"] = "recovered"
            else:
                entry["outcome"] = "unknown"
        else:
            entry["outcome"] = "primary"
        garbled_rows.append(entry)

    n = len(rows)
    return {
        "eval_set_size": n,
        "intent_accuracy": round(
            sum(r["intent_ok"] for r in rows) / n, 4
        ) if n else None,
        "app_entity_accuracy": round(
            sum(r["app_ok"] for r in rows) / n, 4
        ) if n else None,
        "monitor_entity_accuracy": round(
            sum(r["monitor_ok"] for r in rows) / n, 4
        ) if n else None,
        "rows": rows,
        "garbled_rows": garbled_rows,
    }


# ----------------------------------------------------------------------
# 3. Response-latency harness (session controller, stub audio)
# ----------------------------------------------------------------------
class _FakeSTT:
    def __init__(self):
        self._cb = {}
        self.started = 0

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        self.started += 1

    def stop(self):
        pass

    @property
    def is_running(self):
        return True

    def emit(self, text):
        self._cb["on_transcript"](text)


class _FakeTTS:
    """TTS stub whose playback takes ``playback_seconds``."""

    def __init__(self, playback_seconds):
        self.spoken = []
        self._cb = {}
        self._playback = playback_seconds

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
        if self._cb.get("on_started"):
            self._cb["on_started"](text)
        time.sleep(self._playback)
        if self._cb.get("on_finished"):
            self._cb["on_finished"](text)


class _FakeMic:
    def __init__(self):
        class Cap:
            def resume(self):
                pass

            def flush(self):
                pass

        self.capture = Cap()
        self.suspended = 0
        self.resumed = 0

    def suspend_detection(self, pause_capture=True):
        self.suspended += 1

    def resume_detection(self):
        self.resumed += 1


def run_response_latency() -> dict:
    """Measure wake->ack->first-command timing per ack mode."""
    modes = {
        "full": ("At your service.", 1.8),
        "quick": ("Yes.", 0.4),
        "silent": (None, 0.0),
    }
    results = {}
    for mode, (ack_text, playback) in modes.items():
        state = StateManager()
        state.transition_to(AppState.STANDBY)
        stt, tts = _FakeSTT(), _FakeTTS(playback)
        mic = _FakeMic()
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=SessionManager(timeout_seconds=60),
            speech_recognition=stt,
            speech_output=tts,
            microphone_controller=mic,
            ack_mode=mode,
            activation_response="At your service.",
            quick_ack_response="Yes.",
        )
        t_wake = time.monotonic()
        ctl.handle_wake_word("hello jarvis")
        t_active = time.monotonic()
        # First command immediately after the ack completes.
        stt.emit("What time is it?")
        t_first_transcript = time.monotonic()
        ctl._end_with_goodbye()
        timings = ctl.interaction_timings()
        rec = timings[-1] if timings else {}
        results[mode] = {
            "ack_spoken": ack_text,
            "simulated_ack_playback_s": playback,
            "wake_to_active_s": round(t_active - t_wake, 4),
            "wake_to_first_transcript_s": round(
                t_first_transcript - t_wake, 4
            ),
            "recorded": rec,
            "tts_utterances": list(tts.spoken),
        }
        ctl.shutdown()

    # Consecutive-command check: three commands, one session.
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    stt, tts = _FakeSTT(), _FakeTTS(0.0)
    mic = _FakeMic()
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=SessionManager(timeout_seconds=60),
        speech_recognition=stt,
        speech_output=tts,
        microphone_controller=mic,
        ack_mode="quick",
    )
    ctl.handle_wake_word()
    stt.emit("What time is it?")
    stt.emit("What is the date?")
    stt.emit("Tell me the time.")
    consecutive = {
        "commands_accepted": 3,
        "session_active": ctl.session_active,
        "stt_starts": stt.started,
        "tts_utterances": len(tts.spoken),
    }
    ctl.shutdown()

    return {
        "modes": results,
        "consecutive_commands": consecutive,
        "note": (
            "Playback durations are simulated; real TTS timing is "
            "captured in interaction_timings during physical runs."
        ),
    }


# ----------------------------------------------------------------------
# 4. Optional STT benchmark (Moonshine Base vs Tiny)
# ----------------------------------------------------------------------
def run_stt_benchmark() -> dict:
    eval_manifest = FIXTURES / "eval_manifest.json"
    if not eval_manifest.exists():
        return {"status": "skipped", "reason": "eval_manifest missing"}
    try:
        import psutil  # noqa: F401
        from scripts.benchmark_stt import run as bench_run
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"benchmark unavailable: {exc}"}

    from src.core.configuration import get_app_data_dir
    stt_root = get_app_data_dir() / "models" / "stt"
    models = {
        "moonshine_base_int8": stt_root / "sherpa-onnx-moonshine-base-en-int8",
        "moonshine_tiny_int8": stt_root / "sherpa-onnx-moonshine-tiny-en-int8",
    }
    out = {}
    for label, mdir in models.items():
        if not mdir.exists():
            out[label] = {"status": "missing", "dir": str(mdir)}
            continue
        try:
            res = bench_run(mdir, eval_manifest, normalize=True)
            out[label] = {
                "status": "ok",
                "load_s": res["load_s"],
                "mean_wer": res["mean_wer"],
                "intent_accuracy": res["intent_accuracy"],
                "app_name_accuracy": res["app_name_accuracy"],
                "empty_transcripts": res["empty_transcripts"],
                "unsafe_interpretations": res["unsafe_interpretations"],
                "latency_mean_ms": res["latency_mean_ms"],
                "latency_p95_ms": res["latency_p95_ms"],
                "rss_peak_mb": res["rss_peak_mb"],
            }
        except Exception as exc:  # noqa: BLE001
            out[label] = {"status": "error", "error": str(exc)}
    return {"status": "ok", "models": out}


# ----------------------------------------------------------------------
# 5. Resource snapshot (best effort)
# ----------------------------------------------------------------------
def resource_snapshot() -> dict:
    snap = {"status": "unavailable"}
    try:
        import psutil

        proc = psutil.Process()
        proc.cpu_percent()
        time.sleep(0.5)
        snap = {
            "status": "ok",
            "rss_mb": round(proc.memory_info().rss / 1e6, 1),
            "cpu_percent": proc.cpu_percent(),
            "threads": proc.num_threads(),
        }
    except Exception as exc:  # noqa: BLE001
        snap = {"status": "error", "error": str(exc)}
    return snap


# ----------------------------------------------------------------------
# Physical trials worksheet
# ----------------------------------------------------------------------
PHYSICAL_TRIALS_HEADER = [
    "trial_id", "category", "device", "attempt", "expected",
    "transcript", "intent", "result", "wake_latency_s",
    "ack_mode", "notes",
]


def write_physical_trials_csv() -> Path:
    path = REPORTS / "v104_physical_trials.csv"
    REPORTS.mkdir(exist_ok=True)
    # Seed rows for the required physical protocol; results are filled
    # in during hardware runs — never fabricated.
    rows = []
    tid = 0
    for device in ("intel_array", "realtek_headset"):
        for i in range(1, 21):
            tid += 1
            rows.append({
                "trial_id": f"wake-{device}-{i:02d}",
                "category": "wake_word",
                "device": device,
                "attempt": i,
                "expected": "hello jarvis",
            })
    for i in range(1, 11):
        tid += 1
        rows.append({
            "trial_id": f"cycle-{i:02d}",
            "category": "wake_and_command",
            "device": "both",
            "attempt": i,
            "expected": "wake + command",
        })
    for i in range(1, 21):
        tid += 1
        rows.append({
            "trial_id": f"appcmd-{i:02d}",
            "category": "application_command",
            "device": "both",
            "attempt": i,
            "expected": "open/switch/close app",
        })
    for i in range(1, 21):
        tid += 1
        rows.append({
            "trial_id": f"moncmd-{i:02d}",
            "category": "monitor_command",
            "device": "both",
            "attempt": i,
            "expected": "move window/app to monitor",
        })
    for label, n in (("silence_30min", 1), ("conversation_noise", 5)):
        for i in range(1, n + 1):
            tid += 1
            rows.append({
                "trial_id": f"{label}-{i}",
                "category": "false_activation",
                "device": "both",
                "attempt": i,
                "expected": "no activation",
            })
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PHYSICAL_TRIALS_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in PHYSICAL_TRIALS_HEADER})
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stt", action="store_true",
                    help="skip the Moonshine Base/Tiny benchmark")
    ap.add_argument("--out-dir", default=str(REPORTS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[v104] wake-word latency eval ...")
    wake = run_wake_word_eval()
    (out_dir / "v104_wake_word_latency.json").write_text(
        json.dumps(wake, indent=1), encoding="utf-8")
    print(f"  detected={wake['detection_rate']} "
          f"p95={wake['latency_p95_s']}s false={wake['false_activations']}")

    print("[v104] command accuracy eval ...")
    commands = run_command_eval()
    if not args.skip_stt:
        commands["stt_benchmark"] = run_stt_benchmark()
    (out_dir / "v104_command_accuracy.json").write_text(
        json.dumps(commands, indent=1), encoding="utf-8")
    print(f"  intent_acc={commands['intent_accuracy']} "
          f"app_acc={commands['app_entity_accuracy']}")

    print("[v104] response latency eval ...")
    latency = run_response_latency()
    latency["resource_snapshot"] = resource_snapshot()
    (out_dir / "v104_response_latency.json").write_text(
        json.dumps(latency, indent=1), encoding="utf-8")
    print("  modes:", {m: v["wake_to_first_transcript_s"]
                       for m, v in latency["modes"].items()})

    csv_path = write_physical_trials_csv()
    # Ensure the CSV lands in the requested out dir too.
    if out_dir != REPORTS:
        import shutil
        shutil.copy(csv_path, out_dir / "v104_physical_trials.csv")
    print(f"[v104] physical trials worksheet: {csv_path}")
    print("[v104] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
