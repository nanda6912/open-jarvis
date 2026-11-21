"""V10.5 STT model benchmark.

Runs the shared ``benchmark_stt.run`` harness over every locally
available offline STT model and every evaluation manifest, then writes
``reports/v105_stt_benchmark.json``.

Evaluations:
  * ``tests/fixtures/eval_manifest.json`` — synthesized command corpus.
  * ``tests/fixtures/eval_physical_intel.json`` — real Intel-array
    microphone recordings (subset present on disk is used).
  * ``tests/fixtures/eval_physical_realtek.json`` — Realtek headset
    recordings, when present.

No model is replaced automatically: Moonshine Base INT8 stays the
production default unless a candidate shows a measurable overall
improvement across WER, app-name accuracy, intent accuracy, latency
and memory.

Usage:
    py -3 scripts/benchmark_v105_models.py [--models DIR ...]
        [--evals MANIFEST ...] [--out reports/v105_stt_benchmark.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_stt import run as benchmark_run  # noqa: E402

from src.core.paths import runtime_dir, stt_model_dir  # noqa: E402

DEFAULT_MODEL_ROOT = stt_model_dir()
DEFAULT_EVALS = [
    PROJECT_ROOT / "tests" / "fixtures" / "eval_manifest.json",
    PROJECT_ROOT / "tests" / "fixtures" / "eval_physical_intel.json",
    PROJECT_ROOT / "tests" / "fixtures" / "eval_physical_realtek.json",
]
DEFAULT_OUT = runtime_dir() / "reports" / "v105_stt_benchmark.json"


def discover_models(root: Path) -> list[Path]:
    """Every directory under ``root`` that looks like an STT model."""
    models = []
    if not root.is_dir():
        return models
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.onnx")):
            models.append(child)
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", type=Path, default=None,
                    help="Explicit model dirs (default: scan model root)")
    ap.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    ap.add_argument("--evals", nargs="*", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    models = args.models or discover_models(args.model_root)
    evals = [e for e in (args.evals or DEFAULT_EVALS) if e.exists()]
    if not models:
        raise SystemExit(f"no STT models found under {args.model_root}")
    if not evals:
        raise SystemExit("no evaluation manifests found")

    results = []
    for model_dir in models:
        for eval_path in evals:
            # Skip manifests whose wavs are entirely absent.
            entries = json.loads(eval_path.read_text(encoding="utf-8"))
            base = eval_path.parent
            present = [
                e for e in entries
                if (base / e["wav"]).exists() or Path(e["wav"]).exists()
            ]
            if not present:
                print(f"skip {eval_path.name}: no wavs present")
                continue
            # benchmark_stt.run iterates every manifest entry, so feed
            # it a filtered copy containing only wavs on disk.
            if len(present) != len(entries):
                filtered = eval_path.with_suffix(".v105.json")
                filtered.write_text(json.dumps(present, indent=1),
                                    encoding="utf-8")
                eval_run = filtered
            else:
                eval_run = eval_path
            t0 = time.perf_counter()
            print(f"=== {model_dir.name} x {eval_path.name} "
                  f"({len(present)}/{len(entries)} wavs) ===")
            summary = benchmark_run(
                model_dir, eval_run, normalize=True,
                threads=args.threads,
            )
            summary["eval_manifest"] = str(eval_path.relative_to(
                PROJECT_ROOT)) if eval_path.is_relative_to(PROJECT_ROOT) \
                else str(eval_path)
            summary["present_wavs"] = len(present)
            summary["total_wavs"] = len(entries)
            summary["wall_s"] = round(time.perf_counter() - t0, 1)
            results.append(summary)
            brief = {k: v for k, v in summary.items() if k != "rows"}
            print(json.dumps(brief, indent=1))

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": [str(m) for m in models],
        "evals": [str(e) for e in evals],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
