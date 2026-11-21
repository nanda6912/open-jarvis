"""V10.6 wake-word reliability worksheet generator.

Writes ``reports/v106_wake_word_trials.csv`` — the structured physical
test plan for investigating missed "Hello Jarvis" detections on the
real Intel microphone (and Realtek headset for comparison).

Each row corresponds to one spoken wake attempt. During live runs,
fill in the outcome columns from the VoiceOS log / HUD. The new V10.6
diagnostics emit one "Wake attempt #N closed WITHOUT detection" line
per missed speech window with RMS/peak/channel/inference evidence, and
``attempts_missed``/``attempts_recorded`` counters appear in pipeline
stats — use them to cross-check the sheet.

Usage::

    py -3 scripts/test_v106_wake_word_reliability.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "v106_wake_word_trials.csv"


def _rows() -> list[dict]:
    rows = []
    tid = 0

    def add(block, mic, condition, attempts):
        nonlocal tid
        for i in range(1, attempts + 1):
            tid += 1
            rows.append({
                "trial_id": f"w{tid:03d}",
                "block": block,
                "mic": mic,
                "condition": condition,
                "attempt": i,
                "expected": "Hello Jarvis",
                "attempt_recorded": "",
                "detected": "",
                "attempts_used": "",
                "speech_end_to_detection_s": "",
                "kws_rms_max": "",
                "peak_max": "",
                "selected_channel": "",
                "channel_rms": "",
                "inferences": "",
                "queue_depth_max": "",
                "frames_dropped": "",
                "overflow_callbacks": "",
                "notes": "",
            })

    mics = ("intel_array", "realtek_headset")
    # Block A: 10 quiet attempts per mic.
    for mic in mics:
        add("A_quiet_10", mic, "quiet_room", 10)
    # Block B: 10 normal-volume attempts per mic.
    for mic in mics:
        add("B_normal_10", mic, "normal_volume", 10)
    # Block C: 10 distance variants per mic (near/arm's-length/far).
    for mic in mics:
        add("C_distance_10", mic, "near_20cm", 4)
        add("C_distance_10", mic, "arms_length", 3)
        add("C_distance_10", mic, "far_1m", 3)
    # Block D: 10 moderate background noise per mic.
    for mic in mics:
        add("D_noise_10", mic, "moderate_noise", 10)
    # Block E: 5 standby->wake->goodbye->standby cycles per mic.
    for mic in mics:
        for cycle in range(1, 6):
            for att in range(1, 4):
                tid += 1
                rows.append({
                    "trial_id": f"w{tid:03d}",
                    "block": f"E_cycle_{cycle}",
                    "mic": mic,
                    "condition": "standby_cycle",
                    "attempt": att,
                    "expected": "Hello Jarvis",
                    "attempt_recorded": "",
                    "detected": "",
                    "attempts_used": "",
                    "speech_end_to_detection_s": "",
                    "kws_rms_max": "",
                    "peak_max": "",
                    "selected_channel": "",
                    "channel_rms": "",
                    "inferences": "",
                    "queue_depth_max": "",
                    "frames_dropped": "",
                    "overflow_callbacks": "",
                    "notes": "goodbye->standby before each cycle",
                })
    return rows


def main() -> int:
    rows = _rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"[v106] wrote {len(rows)} trial rows: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
