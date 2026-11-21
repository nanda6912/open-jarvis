"""V10.3.3 harness — wake-word detection across Goodbye->standby cycles.

Drives the REAL pipeline pieces that matter for the reported bug:

- real sherpa-onnx KWS engine + WakeWordDetectionService
- real VoiceSessionController / SpeechRecognitionService /
  SpeechOutputService / StateManager / SessionManager
- recorded ``hello_jarvis.wav`` pushed through the shared frame queue
  exactly as live capture would deliver it

Only hardware edges are faked (frame source, scripted STT transcripts,
silent synth/player). Each cycle:

    standby --hello jarvis--> ACTIVATED -> command -> LISTENING
            --goodbye jarvis--> farewell -> STANDBY
            --hello jarvis--> ACTIVATED ... (x N)

Writes ``reports/v1033_session_wake_results.json`` and
``reports/v1033_physical_trials.csv`` (worksheet for real-mic runs).

Usage::

    python scripts/test_v1033_session_standby.py [--cycles 5]
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

from src.core.session_manager import SessionManager  # noqa: E402
from src.core.state_manager import AppState, StateManager  # noqa: E402
from src.voice.speech_output_service import SpeechOutputService  # noqa: E402
from src.voice.speech_recognition_service import SpeechRecognitionService  # noqa: E402
from src.voice.voice_session_controller import VoiceSessionController  # noqa: E402
from src.voice.wake_word_detection_service import WakeWordDetectionService  # noqa: E402
from src.voice.wake_word_engine import create_engine  # noqa: E402

from tests.test_wake_cycles import (  # noqa: E402
    FakeFrameSource,
    FakePlayer,
    FakeSynthesizer,
    ScriptedRecognizer,
    _speak_utterance,
    _wait_for,
)

FIXTURES = ROOT / "tests" / "fixtures"
WAKE_WAV = FIXTURES / "commands" / "hello_jarvis.wav"
REPORTS = ROOT / "reports"
SR = 16000
CHUNK = 1600


def load_wav_16k(path: Path) -> np.ndarray:
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
    if rate != SR:
        from src.voice.audio_capture_service import resample_to_16k

        a = resample_to_16k(a, float(rate))
    return a.astype(np.float32)


class HarnessMic:
    """Mimics MicrophoneController's suspend/resume contract while
    driving a REAL WakeWordDetectionService on the shared queue."""

    def __init__(self, capture: FakeFrameSource, engine):
        self.capture = capture
        self._engine = engine
        self._service: WakeWordDetectionService | None = None
        self.suspend_count = 0
        self.resume_count = 0
        self.generations = 0
        self.on_wake = None  # set later -> controller.handle_wake_word

    # -- MicrophoneController-compatible surface --
    def suspend_detection(self, pause_capture: bool = True):
        self.suspend_count += 1
        if self._service is not None:
            self._service.stop()
            self._service = None
        if pause_capture:
            self.capture.pause()

    def resume_detection(self):
        self.resume_count += 1
        self.capture.resume()
        self.capture.flush()
        self.generations += 1
        self._service = WakeWordDetectionService(
            capture=self.capture,
            engine=self._engine,
            threshold=0.25,
            cooldown_seconds=3.0,
            on_detected=self._on_detected,
        )
        self._service.start()

    def enable(self):
        self.resume_detection()

    def shutdown(self):
        if self._service is not None:
            self._service.stop()
            self._service = None

    def _on_detected(self, keyword: str):
        # Same contract as MicrophoneController._handle_wake_word:
        # suspend KWS, then deliver to the session controller.
        self.suspend_detection(pause_capture=True)
        if self.on_wake is not None:
            self.on_wake(keyword)


def push_audio(cap: FakeFrameSource, audio: np.ndarray):
    for start in range(0, len(audio) - CHUNK + 1, CHUNK):
        cap.push(audio[start:start + CHUNK])


def run_cycles(cycles: int = 5) -> dict:
    engine = create_engine()
    if getattr(engine, "name", "") != "sherpa-onnx-kws":
        raise SystemExit("sherpa-onnx KWS model unavailable")
    variants = engine.keyword_count
    wake_audio = np.concatenate(
        [load_wav_16k(WAKE_WAV), np.zeros(SR, dtype=np.float32)]
    )

    state = StateManager()
    state.transition_to(AppState.STANDBY)
    session = SessionManager(timeout_seconds=60)
    capture = FakeFrameSource()
    mic = HarnessMic(capture, engine)

    transcripts = []
    for _ in range(cycles):
        transcripts.extend(["What time is it?", "Goodbye Jarvis"])
    stt = SpeechRecognitionService(
        capture=capture,
        recognizer=ScriptedRecognizer(transcripts),
        speech_threshold=0.02,
        silence_hangover=0.25,
        min_utterance=0.15,
        pre_roll=0.2,
        frame_timeout=0.1,
    )
    synth = FakeSynthesizer()
    player = FakePlayer(block=0.01)
    tts = SpeechOutputService(synthesizer=synth, player=player, gate=capture)
    tts.start()

    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=session,
        speech_recognition=stt,
        speech_output=tts,
        microphone_controller=mic,
        command_registry=None,
    )
    mic.on_wake = ctl.handle_wake_word
    mic.enable()

    results = []
    try:
        for i in range(1, cycles + 1):
            rec = {"cycle": i}
            t0 = time.monotonic()
            # -- wake -------------------------------------------------
            push_audio(capture, wake_audio)
            ok = _wait_for(lambda: session.is_active, timeout=15.0)
            rec["wake_detected"] = ok
            rec["wake_latency_s"] = round(time.monotonic() - t0, 3)
            rec["first_attempt"] = True  # single pushed utterance
            if not ok:
                rec["error"] = "wake not detected"
                results.append(rec)
                break
            _wait_for(lambda: state.state is AppState.LISTENING, 10.0)
            rec["activated"] = True

            # -- command ----------------------------------------------
            _speak_utterance(capture)
            _wait_for(
                lambda: any("I heard you" in c for c in synth.calls[-3:]),
                10.0,
            )
            rec["command_answered"] = True

            # -- goodbye ----------------------------------------------
            _speak_utterance(capture)
            ok = _wait_for(lambda: state.state is AppState.STANDBY, 15.0)
            rec["returned_to_standby"] = ok
            rec["session_inactive"] = not session.is_active
            # KWS must be listening again before next wake.
            rec["kws_resumed"] = mic._service is not None and (
                mic._service.is_running
            )
            rec["detection_generation"] = mic.generations
            results.append(rec)
            if not ok:
                break
    finally:
        try:
            ctl.shutdown()
        except Exception:  # noqa: BLE001
            pass
        stt.shutdown()
        tts.shutdown()
        mic.shutdown()

    detected = sum(1 for r in results if r.get("wake_detected"))
    latencies = [r["wake_latency_s"] for r in results
                 if r.get("wake_detected")]
    return {
        "engine": getattr(engine, "name", "unknown"),
        "keywords_file": str(getattr(engine, "keywords_file", "") or ""),
        "keyword_count": variants,
        "keywords_sha256": getattr(engine, "keywords_sha256", None),
        "threshold": 0.25,
        "cycles_requested": cycles,
        "cycles_completed": len(results),
        "wake_detections": detected,
        "detection_rate": (
            round(detected / len(results), 3) if results else None
        ),
        "wake_latency_s": latencies,
        "all_first_attempt": all(
            r.get("first_attempt") for r in results
        ),
        "standby_transitions_ok": all(
            r.get("returned_to_standby") for r in results
        ),
        "kws_resumed_each_cycle": all(
            r.get("kws_resumed") for r in results
        ),
        "note": (
            "Harness result with recorded fixture audio through the "
            "real KWS engine/service/controller. NOT a substitute for "
            "physical microphone trials (see v1033_physical_trials.csv)."
        ),
        "cycles": results,
    }


def write_physical_trials_csv(path: Path) -> None:
    header = [
        "trial_id", "cycle", "attempt", "expected", "wake_detected",
        "attempts_used", "returned_to_standby", "duplicate_activation",
        "false_activation", "notes",
    ]
    rows = []
    for cycle in range(1, 6):
        for attempt in range(1, 4):
            rows.append({
                "trial_id": f"postgoodbye-c{cycle:02d}-a{attempt}",
                "cycle": cycle,
                "attempt": attempt,
                "expected": "hello jarvis",
                "wake_detected": "",
                "attempts_used": "",
                "returned_to_standby": "",
                "duplicate_activation": "",
                "false_activation": "",
                "notes": "",
            })
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--out-dir", default=str(REPORTS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[v1033] running post-Goodbye wake cycles ...")
    result = run_cycles(args.cycles)
    (out_dir / "v1033_session_wake_results.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    csv_path = out_dir / "v1033_physical_trials.csv"
    write_physical_trials_csv(csv_path)

    print(f"  detections={result['wake_detections']}/"
          f"{result['cycles_completed']} "
          f"rate={result['detection_rate']}")
    print(f"  latencies={result['wake_latency_s']}")
    print(f"  standby_ok={result['standby_transitions_ok']} "
          f"kws_resumed={result['kws_resumed_each_cycle']}")
    print(f"[v1033] results: {out_dir / 'v1033_session_wake_results.json'}")
    print(f"[v1033] trials worksheet: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
