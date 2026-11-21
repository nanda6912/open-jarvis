"""Moonshine Tiny vs Base STT benchmark for VoiceOS.

Usage:
    python scripts/benchmark_stt.py --model-dir <stt model dir> \
        --eval tests/fixtures/eval_manifest.json [--no-normalize]

Measures transcription quality (WER, exact match, app-name, intent/
entity accuracy, empty transcripts, unsafe interpretations) and
resource use (load time, RSS memory, latency avg/p95, CPU%).

Eval manifest format (JSON list):
    {"wav": "file.wav", "reference": "expected transcript",
     "expected_intent": "OPEN_APPLICATION",       # optional
     "expected_entities": {"app_name": "Notepad"}} # optional
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.commands.intent import RuleBasedIntentRecognizer  # noqa: E402
from src.controllers.application_discovery import (  # noqa: E402
    AppInfo,
    ApplicationDiscovery,
)
from src.voice.stt_engine import (  # noqa: E402
    NullSpeechRecognizer,
    SherpaOnnxRecognizer,
    create_recognizer,
    prepare_utterance,
)

TARGET_RATE = 16000


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


def words(text: str) -> list[str]:
    return [w for w in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split()]


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # Word-level Levenshtein.
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (r[i - 1] != h[j - 1]),
            )
    return d[len(r)][len(h)] / len(r)


def make_discovery():
    apps = [
        AppInfo("Google Chrome", r"C:\PF\Google\Chrome\Application\chrome.exe"),
        AppInfo("Visual Studio Code", r"C:\U\x\Code.exe"),
        AppInfo("Notepad", r"C:\Windows\System32\notepad.exe"),
        AppInfo("Spotify", r"C:\U\x\Spotify.exe"),
        AppInfo("Microsoft Edge", r"C:\PF86\Microsoft\Edge\Application\msedge.exe"),
        AppInfo("Devin (User)", r"C:\U\x\Devin\Devin.exe"),
        AppInfo("WhatsApp", r"C:\PF\WhatsApp\WhatsApp.exe"),
    ]
    valid = {a.exe_path.lower() for a in apps}
    return ApplicationDiscovery(
        env={}, file_exists=lambda p: p.lower() in valid,
        registry_reader=lambda k: {}, cache_ttl=0,
        extra_candidates={a.name: [a.exe_path] for a in apps},
    )


def app_token_match(expected: str, transcript: str) -> bool:
    """True when every expected app token appears in the transcript."""
    et = set(words(expected))
    tt = set(words(transcript))
    return bool(et) and et <= tt


def entity_match(expected: dict, actual: dict) -> bool:
    for k, v in (expected or {}).items():
        got = actual.get(k)
        if isinstance(v, str):
            if str(got).strip().lower() != v.strip().lower():
                return False
        elif got != v:
            return False
    return True


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def run(model_dir: Path, eval_path: Path, normalize: bool,
        threads: int = 2) -> dict:
    import psutil

    proc = psutil.Process()
    proc.cpu_percent()  # prime
    rss_before = proc.memory_info().rss

    t0 = time.perf_counter()
    rec = create_recognizer(model_dir, num_threads=threads)
    load_s = time.perf_counter() - t0
    if getattr(rec, "name", "") == "null":
        raise SystemExit(f"recognizer unavailable for {model_dir}")
    rss_after_load = proc.memory_info().rss

    entries = json.loads(eval_path.read_text(encoding="utf-8"))
    base = eval_path.parent
    intents = RuleBasedIntentRecognizer()
    disc = make_discovery()

    rows = []
    latencies = []
    cpu_samples = []
    peak_rss = rss_after_load
    for e in entries:
        wav = base / e["wav"]
        if not wav.exists():
            wav = Path(e["wav"])
        audio = load_wav(wav)
        if normalize:
            audio = prepare_utterance(audio)
        c0 = proc.cpu_times()
        t1 = time.perf_counter()
        text = rec.transcribe(audio)
        lat = time.perf_counter() - t1
        c1 = proc.cpu_times()
        latencies.append(lat)
        cpu_samples.append(
            ((c1.user - c0.user) + (c1.system - c0.system)) / max(lat, 1e-6)
        )
        rss = proc.memory_info().rss
        peak_rss = max(peak_rss, rss)

        ref = e.get("reference", "")
        exp_intent = e.get("expected_intent")
        exp_entities = e.get("expected_entities") or {}
        intent = intents.recognize(text) if text.strip() else None

        # App-name accuracy: expected app token present in transcript.
        exp_app = exp_entities.get("app_name") or exp_entities.get("target")
        app_ok = (
            app_token_match(exp_app, text)
            if exp_app else None
        )
        # Unsafe: parser produced an executable app intent that resolves
        # to a *different* app than expected (silent wrong launch risk).
        unsafe = False
        wrong_app = None
        if intent is not None and intent.name in (
            "OPEN_APPLICATION", "CLOSE_APPLICATION",
            "SWITCH_TO_APPLICATION", "FIND_APPLICATION",
        ):
            name = intent.entities.get("app_name") or ""
            res = disc.resolve(name) if name else None
            if res is not None and res.status == "found" and res.app:
                if exp_app:
                    want = set(words(exp_app))
                    got = set(words(res.app.name)) | set(
                        words(res.app.exe_name)
                    )
                    if not want <= got:
                        unsafe = True
                        wrong_app = res.app.name
                elif intent.name == "OPEN_APPLICATION":
                    # No expected app recorded; a confident resolve of an
                    # odd transcript is still noteworthy, not unsafe.
                    pass

        intent_ok = (
            intent is not None and intent.name == exp_intent
        ) if exp_intent else None
        entities_ok = (
            entity_match(exp_entities, intent.entities)
            if intent is not None and exp_entities else None
        )

        rows.append({
            "wav": e["wav"],
            "reference": ref,
            "transcript": text,
            "wer": wer(ref, text),
            "exact": words(ref) == words(text) if ref else (text == ""),
            "empty": not text.strip(),
            "app_ok": app_ok,
            "intent_ok": intent_ok,
            "entities_ok": entities_ok,
            "unsafe": unsafe,
            "wrong_app": wrong_app,
            "latency_s": round(lat, 4),
            "intent_name": intent.name if intent else "-",
        })

    n = len(rows)
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "model_dir": str(model_dir),
        "normalize": normalize,
        "threads": threads,
        "load_s": round(load_s, 3),
        "rss_before_mb": round(rss_before / 1e6, 1),
        "rss_after_load_mb": round(rss_after_load / 1e6, 1),
        "rss_peak_mb": round(peak_rss / 1e6, 1),
        "n_utterances": n,
        "mean_wer": round(avg("wer") or 0.0, 4),
        "exact_match": round(sum(r["exact"] for r in rows) / n, 4),
        "empty_transcripts": sum(r["empty"] for r in rows),
        "app_name_accuracy": (
            round(sum(1 for r in rows if r["app_ok"]) /
                  sum(1 for r in rows if r["app_ok"] is not None), 4)
            if any(r["app_ok"] is not None for r in rows) else None
        ),
        "intent_accuracy": (
            round(sum(1 for r in rows if r["intent_ok"]) /
                  sum(1 for r in rows if r["intent_ok"] is not None), 4)
            if any(r["intent_ok"] is not None for r in rows) else None
        ),
        "entities_accuracy": (
            round(sum(1 for r in rows if r["entities_ok"]) /
                  sum(1 for r in rows if r["entities_ok"] is not None), 4)
            if any(r["entities_ok"] is not None for r in rows) else None
        ),
        "unsafe_interpretations": sum(r["unsafe"] for r in rows),
        "latency_mean_ms": round(1000 * sum(latencies) / n, 1),
        "latency_p50_ms": round(1000 * percentile(latencies, 50), 1),
        "latency_p95_ms": round(1000 * percentile(latencies, 95), 1),
        "latency_max_ms": round(1000 * max(latencies), 1),
        "cpu_util_mean": round(sum(cpu_samples) / n, 3),
        "rows": rows,
    }
    try:
        rec.close()
    except Exception:
        pass
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run(
        Path(args.model_dir), Path(args.eval),
        normalize=not args.no_normalize, threads=args.threads,
    )
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    # Print summary without rows.
    brief = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(brief, indent=1))
    print("\n--- per-utterance ---")
    for r in result["rows"]:
        flags = []
        if r["empty"]:
            flags.append("EMPTY")
        if r["unsafe"]:
            flags.append(f"UNSAFE->{r['wrong_app']}")
        if r["app_ok"] is False:
            flags.append("APP_MISS")
        if r["intent_ok"] is False:
            flags.append("INTENT_MISS")
        print(
            f"{r['wav']:42s} {r['latency_s']*1000:7.0f}ms "
            f"wer={r['wer']:.2f} {r['transcript']!r:55s} "
            f"[{r['intent_name']}] {' '.join(flags)}"
        )


if __name__ == "__main__":
    main()
