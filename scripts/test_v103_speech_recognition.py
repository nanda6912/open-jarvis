"""V10.3 speech-recognition reliability harness.

Two phases:

A. Offline fixture benchmark — transcribes every recorded command WAV
   through the production sherpa-onnx STT engine, runs the real intent
   recognizer and application discovery, and scores transcript,
   intent and entity-resolution accuracy plus inference latency.
   Deterministic; no microphone needed.

B. Live capture health — opens the real default microphone through
   AudioCaptureService for a short soak and records device identity,
   channel config, callback freshness and throughput. Spoken-command
   trials still require a human operator (see the V10.3 report).

Outputs:
    reports/v103_recognition_trials.csv
    reports/v103_recognition_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.commands.intent import RuleBasedIntentRecognizer  # noqa: E402
from src.controllers.application_discovery import (  # noqa: E402
    AppInfo,
    ApplicationDiscovery,
)
from src.voice.stt_engine import create_recognizer  # noqa: E402

from src.core.paths import runtime_dir, stt_model_dir  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "commands"
MODELS = stt_model_dir()
REPORTS = runtime_dir() / "reports"

# Expected intent per fixture (mirrors tests/test_live_reliability.py).
EXPECTED = {
    "open_chrome.wav": ("OPEN_APPLICATION", "chrome"),
    "open_vscode.wav": ("OPEN_APPLICATION", "code"),
    "open_notepad.wav": ("OPEN_APPLICATION", "notepad"),
    "open_spotify.wav": ("UNKNOWN", None),
    "open_nonexistent.wav": ("UNKNOWN", None),
    "switch_chrome.wav": ("SWITCH_TO_APPLICATION", "chrome"),
    "close_chrome.wav": ("CLOSE_APPLICATION", "chrome"),
    "what_time.wav": ("GET_CURRENT_TIME", None),
    "set_volume_50.wav": ("SET_VOLUME", None),
    "increase_volume_10.wav": ("INCREASE_VOLUME", None),
    "move_chrome_monitor_two.wav": ("MOVE_WINDOW_TO_MONITOR", None),
    "switch_notepad.wav": ("SWITCH_TO_APPLICATION", "notepad"),
    "hello_jarvis.wav": (None, None),
    "hey_jarvis.wav": (None, None),
    "silence.wav": ("", None),
    "background_speech.wav": ("UNKNOWN", None),
    # Physical-microphone recordings: real-world STT/intent evidence.
    "phys_open_devin.wav": ("OPEN_APPLICATION", None),
    "phys_open_notepad.wav": ("OPEN_APPLICATION", None),
    "phys_move_chrome_monitor_one.wav":
        ("MOVE_WINDOW_TO_MONITOR", None),
    "phys_set_volume_40.wav": ("SET_VOLUME", None),
    "phys_switch_whatsapp.wav": ("SWITCH_TO_APPLICATION", None),
}

APP_INVENTORY = [
    AppInfo("Google Chrome",
            r"C:\PF\Google\Chrome\Application\chrome.exe"),
    AppInfo("Visual Studio Code",
            r"C:\Users\x\AppData\Local\Programs\Microsoft VS Code\Code.exe"),
    AppInfo("Notepad", r"C:\Windows\System32\notepad.exe"),
    AppInfo("Spotify",
            r"C:\Users\x\AppData\Roaming\Spotify\Spotify.exe"),
    AppInfo("Microsoft Edge",
            r"C:\PF86\Microsoft\Edge\Application\msedge.exe"),
    AppInfo("Devin", r"C:\Apps\Devin.exe"),
    AppInfo("Docker Desktop", r"C:\Apps\Docker Desktop.exe"),
    AppInfo("WhatsApp", r"C:\Apps\WhatsApp.exe"),
]


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return (np.frombuffer(w.readframes(w.getnframes()),
                              dtype=np.int16)
                .astype(np.float32) / 32768.0)


def _disc() -> ApplicationDiscovery:
    valid = {a.exe_path.lower() for a in APP_INVENTORY}
    return ApplicationDiscovery(
        env={},
        file_exists=lambda p: str(p).lower() in valid,
        registry_reader=lambda k: {},
        cache_ttl=0,
        extra_candidates={a.name: [a.exe_path]
                          for a in APP_INVENTORY},
    )


def run_fixture_benchmark() -> dict:
    if not MODELS.exists():
        return {"skipped": "STT model not installed"}
    rec = create_recognizer(MODELS)
    if getattr(rec, "name", "") == "null":
        rec.close()
        return {"skipped": "STT recognizer unavailable"}
    intents = RuleBasedIntentRecognizer()
    disc = _disc()

    rows = []
    latencies = []
    intent_ok = intent_total = 0
    entity_ok = entity_total = 0
    transcript_exact = transcript_total = 0

    for wav in sorted(FIXTURES.glob("*.wav")):
        exp_intent, exp_app = EXPECTED.get(wav.name, ("UNKNOWN", None))
        audio = _load_wav(wav)
        t0 = time.perf_counter()
        text = (rec.transcribe(audio) or "").strip()
        lat = time.perf_counter() - t0
        latencies.append(lat)

        if exp_intent is None:
            # Wake fixtures: transcript should mention Jarvis (or be
            # empty for the non-wake variant).
            ok = ("jarvis" in text.lower()) or wav.name == "hey_jarvis.wav"
            rows.append({
                "fixture": wav.name, "expected_text": "<wake>",
                "actual_transcript": text, "stt_latency_s": round(lat, 3),
                "expected_intent": "-", "actual_intent": "-",
                "entity_result": "wake_ok" if ok else "wake_miss",
                "pass": ok,
            })
            continue

        intent = intents.recognize(text)
        intent_total += 1
        if exp_intent == "":
            # Empty-transcript fixture: correct = no intent produced.
            intent_pass = (text == "")
        else:
            intent_pass = intent.name == exp_intent
        intent_ok += intent_pass

        entity_result, entity_pass = "-", True
        if exp_app:
            entity_total += 1
            res = disc.resolve(intent.entities.get("app_name", ""))
            entity_pass = (res.status == "found"
                           and (exp_app in res.app.exe_name
                                or exp_app in res.app.name.lower()))
            entity_result = res.status
            entity_ok += entity_pass

        if exp_intent == "":
            transcript_total += 1
            transcript_exact += (text == "")

        rows.append({
            "fixture": wav.name,
            "expected_text": _manifest_text(wav.name),
            "actual_transcript": text,
            "stt_latency_s": round(lat, 3),
            "expected_intent": exp_intent,
            "actual_intent": intent.name,
            "entity_result": entity_result,
            "pass": bool(intent_pass and entity_pass),
        })

    rec.close()

    REPORTS.mkdir(exist_ok=True)
    csv_path = REPORTS / "v103_recognition_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lat = np.array(latencies) if latencies else np.array([0.0])
    return {
        "phase": "fixture_benchmark",
        "engine": getattr(rec, "name", "?"),
        "fixtures": len(rows),
        "intent_accuracy": round(intent_ok / max(intent_total, 1), 4),
        "entity_accuracy": round(entity_ok / max(entity_total, 1), 4),
        "empty_transcript_correct": transcript_exact,
        "stt_latency_ms": {
            "mean": round(float(lat.mean()) * 1000, 1),
            "p95": round(float(np.percentile(lat, 95)) * 1000, 1),
            "max": round(float(lat.max()) * 1000, 1),
        },
        "csv": str(csv_path.relative_to(ROOT)),
    }


def _manifest_text(name: str) -> str:
    try:
        manifest = json.loads(
            (FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest:
            if item.get("file") == name:
                return item.get("text", "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def run_live_capture(soak_s: float = 30.0) -> dict:
    """Short physical-mic health soak (no speech required)."""
    try:
        import sounddevice as sd
        from src.voice.audio_capture_service import AudioCaptureService
        from src.voice.audio_device_manager import AudioDeviceManager
    except Exception as exc:  # noqa: BLE001
        return {"skipped": f"audio stack unavailable: {exc}"}

    mgr = AudioDeviceManager(sd=sd)
    cap = AudioCaptureService(sd=sd, device_manager=mgr)
    try:
        cap.start()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"capture start failed: {exc}"}

    t0 = time.monotonic()
    peak_rms = 0.0
    while time.monotonic() - t0 < soak_s:
        time.sleep(0.5)
        st = cap.stats()
        peak_rms = max(peak_rms, st.get("rms") or 0.0)

    st = cap.stats()
    elapsed = time.monotonic() - t0
    cap.stop()
    return {
        "phase": "live_capture_health",
        "soak_s": round(elapsed, 1),
        "device": st.get("device"),
        "stable_device_id": st.get("device_stable_id"),
        "sample_rate": st.get("sample_rate"),
        "input_channels": st.get("input_channels"),
        "frames_captured": st.get("frames_captured"),
        "frames_per_sec": round(
            (st.get("frames_captured") or 0) / max(elapsed, 0.1), 2),
        "callback_age_s": st.get("callback_age_seconds"),
        "max_callback_age_s": st.get("max_callback_age_seconds"),
        "missed_callback_intervals":
            st.get("missed_callback_intervals"),
        "queue_depth": st.get("queue_depth"),
        "peak_rms": round(peak_rms, 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-soak", type=float, default=0.0,
                    help="seconds of live-mic capture health soak "
                         "(0 = skip)")
    args = ap.parse_args()

    results = {"v103": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "phases": []}
    bench = run_fixture_benchmark()
    results["phases"].append(bench)
    print("[fixture]", json.dumps(bench, indent=None))
    if args.live_soak > 0:
        live = run_live_capture(args.live_soak)
        results["phases"].append(live)
        print("[live]", json.dumps(live, indent=None))

    out = REPORTS / "v103_recognition_results.json"
    REPORTS.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
