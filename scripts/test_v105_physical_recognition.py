"""V10.5 physical-recognition harness.

Two modes:

1. Default — writes ``reports/v105_physical_trials.csv``, a worksheet
   covering the full V10.5 controlled command vocabulary for real
   laptop-microphone runs (Intel array + Realtek headset, normal and
   quiet speech, post-standby cycles). Fill it in during live trials.

2. ``--wav-dir DIR`` — transcribes user-provided recordings IN MEMORY
   through the production STT engine + intent parser and writes
   ``reports/v105_command_accuracy.json``. Audio is never persisted.
   File names should look like ``<expected-slug>_r<n>.wav``; expected
   intent/entities come from the built-in eval set.

Usage::

    py -3 scripts/test_v105_physical_recognition.py
    py -3 scripts/test_v105_physical_recognition.py --wav-dir recordings
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"

# Controlled evaluation set (V10.5 spec section 7).
EVAL_SET = [
    # (slug, spoken command, expected_intent, expected entity fragment)
    ("open_chrome", "Open Chrome", "OPEN_APPLICATION", "chrome"),
    ("open_whatsapp", "Open WhatsApp", "OPEN_APPLICATION", "whatsapp"),
    ("open_devin", "Open Devin", "OPEN_APPLICATION", "devin"),
    ("open_docker", "Open Docker Desktop", "OPEN_APPLICATION", "docker"),
    ("open_calculator", "Open Calculator", "OPEN_APPLICATION",
     "calculator"),
    ("switch_chrome", "Switch to Chrome", "SWITCH_TO_APPLICATION",
     "chrome"),
    ("switch_whatsapp", "Switch to WhatsApp", "SWITCH_TO_APPLICATION",
     "whatsapp"),
    ("move_chrome_left", "Move Chrome to the left monitor",
     "MOVE_WINDOW_TO_MONITOR", "left"),
    ("move_chrome_right", "Move Chrome to the right monitor",
     "MOVE_WINDOW_TO_MONITOR", "right"),
    ("move_devin_left", "Move Devin to the left monitor",
     "MOVE_WINDOW_TO_MONITOR", "left"),
    ("move_whatsapp_right", "Move WhatsApp to the right monitor",
     "MOVE_WINDOW_TO_MONITOR", "right"),
    ("move_current_other", "Move the current window to the other "
     "monitor", "MOVE_WINDOW_TO_OTHER_MONITOR", "other"),
    ("open_downloads", "Open Downloads", "OPEN_FOLDER", "downloads"),
    ("open_desktop", "Open Desktop folder", "OPEN_FOLDER", "desktop"),
    ("open_jarvis", "Open Jarvis folder", "OPEN_FOLDER", "jarvis"),
    ("open_smart_parking", "Open Smart Parking System folder",
     "OPEN_FOLDER", "smart parking"),
    ("search_resume", "Search resume files", "SEARCH_FILES", "resume"),
    ("open_named_pdf", "Open my resume PDF", "OPEN_FILE", "resume"),
    ("what_time", "What time is it?", "GET_CURRENT_TIME", None),
    ("what_date", "What's today's date?", "GET_CURRENT_DATE", None),
    ("list_apps", "List running applications",
     "LIST_RUNNING_APPLICATIONS", None),
    ("goodbye", "Goodbye Jarvis", "END_SESSION", None),
]

MICS = ("intel_array", "realtek_headset")
CONDITIONS = ("normal", "quiet", "post_standby")
REPS = 3


def write_trials_csv(path: Path) -> None:
    header = [
        "trial_id", "mic", "condition", "rep", "expected_command",
        "expected_intent", "actual_transcript", "actual_intent",
        "entity_ok", "empty_transcript", "unsafe_execution",
        "latency_ms", "notes",
    ]
    rows = []
    for slug, spoken, intent, _frag in EVAL_SET:
        for mic in MICS:
            for cond in CONDITIONS:
                for rep in range(1, REPS + 1):
                    rows.append({
                        "trial_id": f"{slug}-{mic}-{cond}-r{rep}",
                        "mic": mic,
                        "condition": cond,
                        "rep": rep,
                        "expected_command": spoken,
                        "expected_intent": intent,
                        "actual_transcript": "",
                        "actual_intent": "",
                        "entity_ok": "",
                        "empty_transcript": "",
                        "unsafe_execution": "",
                        "latency_ms": "",
                        "notes": "",
                    })
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def _load_wav_16k(path: Path) -> np.ndarray:
    import wave
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
    if rate != 16000:
        from src.voice.audio_capture_service import resample_to_16k
        a = resample_to_16k(a, rate)
    return a.astype(np.float32)


def _slug_for(name: str):
    stem = re.sub(r"_r\d+$", "", Path(name).stem.lower())
    return stem


def run_wav_dir(wav_dir: Path, model_dir: Path | None) -> dict:
    """Transcribe provided recordings in memory; write accuracy JSON."""
    from src.voice.stt_engine import (
        DEFAULT_STT_MODEL_DIR,
        create_recognizer,
        prepare_utterance,
    )
    from src.commands.intent import RuleBasedIntentRecognizer

    model_dir = model_dir or DEFAULT_STT_MODEL_DIR
    rec = create_recognizer(model_dir, num_threads=2)
    if getattr(rec, "name", "") == "null":
        raise SystemExit(f"recognizer unavailable for {model_dir}")
    parser = RuleBasedIntentRecognizer()

    by_slug = {slug: (spoken, intent, frag) for slug, spoken, intent,
               frag in EVAL_SET}
    rows = []
    latencies = []
    for wav in sorted(wav_dir.glob("*.wav")):
        slug = _slug_for(wav.name)
        expected = by_slug.get(slug)
        if expected is None:
            rows.append({"wav": wav.name, "skipped": "unknown_slug"})
            continue
        spoken, exp_intent, frag = expected
        audio = prepare_utterance(_load_wav_16k(wav))
        t0 = time.perf_counter()
        text = rec.transcribe(audio)
        lat = time.perf_counter() - t0
        latencies.append(lat)
        intent = parser.recognize(text) if text.strip() else None
        iname = intent.name if intent else "-"
        blob = " ".join(
            str(v) for v in (intent.entities if intent else {}).values()
        ).lower() if intent else ""
        rows.append({
            "wav": wav.name,
            "expected_command": spoken,
            "expected_intent": exp_intent,
            "transcript": text,
            "actual_intent": iname,
            "entity_ok": (frag in blob) if frag else None,
            "empty": not text.strip(),
            "intent_ok": iname == exp_intent,
            "latency_ms": round(lat * 1000, 1),
        })
    try:
        rec.close()
    except Exception:
        pass

    scored = [r for r in rows if "skipped" not in r]
    n = len(scored)
    summary = {
        "source": "user_provided_recordings",
        "model_dir": str(model_dir),
        "n": n,
        "transcription_accuracy": (
            round(sum(1 for r in scored if not r["empty"]) / n, 4)
            if n else None),
        "intent_accuracy": (
            round(sum(1 for r in scored if r["intent_ok"]) / n, 4)
            if n else None),
        "empty_rate": (
            round(sum(1 for r in scored if r["empty"]) / n, 4)
            if n else None),
        "latency_mean_ms": (
            round(1000 * sum(latencies) / len(latencies), 1)
            if latencies else None),
        "latency_p95_ms": (
            round(1000 * sorted(latencies)[
                int(0.95 * (len(latencies) - 1))], 1)
            if latencies else None),
        "rows": rows,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", type=Path, default=None,
                    help="Directory of user-provided recordings "
                         "(<slug>_r<n>.wav). Audio is never persisted.")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPORTS)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.wav_dir is None:
        csv_path = args.out_dir / "v105_physical_trials.csv"
        write_trials_csv(csv_path)
        print(f"[v105] wrote trials worksheet: {csv_path}")
        print(f"[v105] {len(EVAL_SET)} commands x {len(MICS)} mics x "
              f"{len(CONDITIONS)} conditions x {REPS} reps = "
              f"{len(EVAL_SET) * len(MICS) * len(CONDITIONS) * REPS} "
              "physical trials")
        return 0

    summary = run_wav_dir(args.wav_dir, args.model_dir)
    out = args.out_dir / "v105_command_accuracy.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=1))
    print(f"[v105] results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
