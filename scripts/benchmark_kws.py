"""Offline wake-word benchmark runner for VoiceOS.

Evaluates explicit local WAV fixtures through the existing KWS engine.
Never opens a microphone, never records audio, and never transmits data.

Usage:
    python scripts/benchmark_kws.py --manifest tests/fixtures/kws_benchmark_manifest.json \
        --output reports/kws_benchmark.json --report reports/kws_benchmark.md

Manifest format (JSON list or {"cases": [...]}):
    {"id": "hello_jarvis", "wav": "commands/hello_jarvis.wav",
     "label": "positive", "wake_end_seconds": 1.4,
     "transforms": {"pad_before_seconds": 0.5, "pad_after_seconds": 1.0,
                    "gain": 0.9, "speed": 1.0,
                    "background": {"wav": "commands/silence.wav", "gain": 0.05}}}

``label`` must be ``positive`` (contains the wake phrase) or ``negative``
(ordinary speech / noise / silence with no wake phrase). ``wake_end_seconds``
is the end time of the wake phrase in the source audio and is used for
latency; if omitted for positives it defaults to the clip duration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voice.kws_benchmark import (  # noqa: E402
    DEFAULT_RESCORE_LEADS,
    DEFAULT_RESCORE_TRAIL,
    dumps_result,
    run_benchmark,
    write_outputs,
)
from src.voice.wake_word_engine import create_engine  # noqa: E402


def _parse_leads(text: str) -> tuple[float, ...]:
    vals = [float(x) for x in text.replace(",", " ").split() if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("at least one lead anchor is required")
    return tuple(vals)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="JSON manifest of labelled WAV fixtures")
    parser.add_argument("--output", required=True, type=Path,
                        help="Machine-readable JSON result path")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional human-readable Markdown report path")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="KWS model directory (default: production path)")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--mode", action="append", choices=("live", "fresh_stream", "rescore"),
                        help="Evaluation mode; repeat or omit for all three")
    parser.add_argument("--frame-ms", type=float, default=100.0,
                        help="Frame size in milliseconds (default 100)")
    parser.add_argument("--gap-seconds", type=float, default=1.0,
                        help="Silence inserted between live-stream cases")
    parser.add_argument("--cooldown-seconds", type=float, default=3.0,
                        help="Minimum seconds between counted detections")
    parser.add_argument("--rescore-leads", type=_parse_leads,
                        default=DEFAULT_RESCORE_LEADS,
                        help="Space/comma separated lead anchors (default: production)")
    parser.add_argument("--rescore-trail", type=float, default=DEFAULT_RESCORE_TRAIL)
    parser.add_argument("--allow-null-engine", action="store_true",
                        help="Run even when the KWS model is unavailable")
    args = parser.parse_args(argv)

    frame_samples = int(round(args.frame_ms * 16000.0 / 1000.0))
    if frame_samples <= 0:
        parser.error("--frame-ms must be positive")

    def factory():
        from src.voice.wake_word_engine import SherpaOnnxKeywordSpotter
        if args.model_dir is not None:
            return SherpaOnnxKeywordSpotter(args.model_dir, num_threads=args.num_threads)
        return create_engine()

    probe = factory()
    try:
        engine_name = getattr(probe, "name", "unknown")
        if engine_name == "null" and not args.allow_null_engine:
            raise SystemExit(
                "Wake-word model unavailable (null engine). Provide --model-dir "
                "or --allow-null-engine for plumbing-only runs."
            )
    finally:
        try:
            probe.close()
        except Exception:
            pass

    modes = tuple(args.mode) if args.mode else ("live", "fresh_stream", "rescore")
    result = run_benchmark(
        args.manifest,
        factory,
        modes=modes,
        frame_samples=frame_samples,
        gap_seconds=args.gap_seconds,
        cooldown_seconds=args.cooldown_seconds,
        rescore_leads=args.rescore_leads,
        rescore_trail=args.rescore_trail,
    )
    write_outputs(result, args.output, args.report)
    # Emit a compact JSON document to stdout as well for piping.
    print(dumps_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
