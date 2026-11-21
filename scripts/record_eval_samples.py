"""Record physical-microphone eval samples for the STT benchmark.

Explicitly initiated, local-only recording for testing — nothing is
captured during normal application operation and VoiceOS continuous
recording is never enabled.

Usage:
    python scripts/record_eval_samples.py --device intel [--hostapi WASAPI]
    python scripts/record_eval_samples.py --device realtek
    python scripts/record_eval_samples.py --list-devices [--verbose]

Semantic selectors resolve to exactly one physical endpoint:

- ``intel``   -> the Intel Microphone Array (Smart Sound Technology).
- ``realtek`` -> ``Headset Microphone (Realtek(R) Audio)``.

Resolution uses the full normalized device name plus optional host API
— never a PortAudio index (those change between runs) and never a
bare substring. Stereo Mix, PC Speaker, Sound Mapper, Primary Sound
Capture Driver and unrelated Realtek endpoints are always rejected.

Recordings land in ``tests/fixtures/physical/<device>/`` at 16 kHz
mono int16. Existing files are never overwritten without ``--force``
plus an interactive confirmation.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PHRASES = [
    "Move Chrome to monitor one.",
    "Open Devin.",
    "Set volume to forty.",
    "Switch to WhatsApp.",
    "Open Notepad.",
    "Open Chrome.",
    "What time is it?",
]
REPS = 3
TARGET_RATE = 16000
CAPTURE_SECONDS = 6.0
PRE_ROLL_S = 3.0

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# Endpoints that must never be treated as a microphone, regardless of
# channel counts or names.
_BLOCKED_PATTERNS = (
    r"\bstereo\s*mix\b",
    r"\bpc\s*speaker\b",
    r"\bsound\s*mapper\b",
    r"\bprimary\s*sound\s*capture\b",
    r"\bmicrophone\s*array\s*[123]\b",   # unnamed array aliases
)

# Preferred probe order for the capture rate.
_PROBE_RATES = (16000, 48000, 44100, 32000, 22050, 96000, 8000)


def _norm(text: str) -> str:
    """Lowercase, strip ®/™ symbols, collapse non-alnum to spaces."""
    t = (text or "").lower().replace("®", " ").replace("™", " ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def _category(name: str) -> str:
    """Semantic category for a device name (or 'other')."""
    n = _norm(name)
    if re.search(r"\b(stereo|mix)\b", n) and "microphone" not in n:
        return "stereo-mix"
    if "pc speaker" in n:
        return "output"
    if "sound mapper" in n or "primary sound capture" in n:
        return "wrapper"
    if "microphone array" in n:
        if "intel" in n or "smart sound" in n:
            return "intel-array"
        return "mic-array-other"
    if "headset microphone" in n and "realtek" in n:
        return "realtek-headset"
    if "realtek" in n:
        return "realtek-other"
    if "microphone" in n:
        return "mic-other"
    return "other"


def _is_intel_target(name: str) -> bool:
    n = _norm(name)
    return (
        "microphone array" in n
        and ("intel" in n or "smart sound" in n)
    )


def _is_realtek_target(name: str) -> bool:
    n = _norm(name)
    return "headset microphone" in n and "realtek" in n


def _blocked(name: str) -> bool:
    n = _norm(name)
    return any(re.search(p, n) for p in _BLOCKED_PATTERNS)


def _semantic_match(selector: str, name: str) -> bool:
    if selector == "intel":
        return _is_intel_target(name)
    if selector == "realtek":
        return _is_realtek_target(name)
    return False


def _hostapi_match(api_name: str, wanted: str) -> bool:
    if not wanted:
        return True
    w = wanted.lower()
    a = api_name.lower()
    aliases = {
        "mme": "mme",
        "mme ": "mme",
        "directsound": "windows directsound",
        "direct sound": "windows directsound",
        "wasapi": "windows wasapi",
        "wdm": "windows wdm ks",
        "wdm-ks": "windows wdm ks",
        "wdmks": "windows wdm ks",
    }
    return aliases.get(w, w) in a or w in a


def list_devices(sd, verbose: bool = False) -> None:
    hostapis = {i: h["name"] for i, h in enumerate(sd.query_hostapis())}
    # Map hostapi index -> devices belonging to it.
    api_of = {}
    for api_idx, h in enumerate(sd.query_hostapis()):
        for d_idx in h["devices"]:
            api_of[d_idx] = api_idx

    print("Input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = d["name"]
        api = hostapis.get(api_of.get(i), "?")
        cat = _category(name)
        tags = []
        if _is_intel_target(name):
            tags.append("intel")
        if _is_realtek_target(name):
            tags.append("realtek")
        if _blocked(name):
            tags.append("BLOCKED")
        print(
            f"  [{i}] hostapi={api} ch={d['max_input_channels']} "
            f"rate={d.get('default_samplerate')} category={cat} "
            f"match={','.join(tags) or '-'}"
        )
        print(f"       name: {name}")
        if verbose:
            rates = []
            for r in _PROBE_RATES:
                try:
                    sd.check_input_settings(
                        device=i, channels=1, samplerate=r,
                        dtype="float32",
                    )
                    rates.append(int(r))
                except Exception:
                    pass
            print(f"       supported_rates={rates or 'unknown'}")


def _iter_input_devices(sd):
    hostapis = {i: h["name"] for i, h in enumerate(sd.query_hostapis())}
    api_of = {}
    for api_idx, h in enumerate(sd.query_hostapis()):
        for d_idx in h["devices"]:
            api_of[d_idx] = api_idx
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        yield i, d, hostapis.get(api_of.get(i), "?")


def resolve_device(sd, selector: str, hostapi: str = ""):
    """Resolve a semantic selector to exactly one input endpoint.

    Returns (index, info_dict, capture_rate). Raises SystemExit listing
    candidates when the match is ambiguous — never guesses.
    """
    candidates = []
    for i, d, api in _iter_input_devices(sd):
        name = d["name"]
        if _blocked(name):
            continue
        if not _semantic_match(selector, name):
            continue
        if hostapi and not _hostapi_match(api, hostapi):
            continue
        candidates.append((i, d, api))

    if not candidates:
        raise SystemExit(
            f"No eligible input device for selector {selector!r}"
            + (f" on hostapi {hostapi!r}" if hostapi else "")
            + ". Run --list-devices --verbose to inspect."
        )
    if len(candidates) > 1:
        # Multiple host-API endpoints can expose the SAME physical
        # microphone (MME/DirectSound/WASAPI/WDM-KS views of one device).
        # If every candidate has the identical normalized name it is
        # still one physical mic — pick the preferred API explicitly
        # and say so. Distinct names remain ambiguous -> abort.
        # Windows truncates device names differently per host API
        # (MME shows ~32 chars, WASAPI the full string). Names that are
        # prefixes of each other are the same physical endpoint.
        def _same_physical(a: str, b: str) -> bool:
            na, nb = _norm(a), _norm(b)
            if not na or not nb:
                return False
            short, long_ = sorted((na, nb), key=len)
            return long_.startswith(short) and len(short) >= 12

        names = [d["name"] for _, d, _ in candidates]
        same_physical = all(
            _same_physical(names[0], n) for n in names[1:]
        )
        if same_physical:
            pref = ("windows wasapi", "windows directsound", "mme",
                    "windows wdm ks")
            candidates.sort(
                key=lambda c: next(
                    (i for i, p in enumerate(pref)
                     if p in c[2].lower()), len(pref)
                )
            )
            idx, info, api = candidates[0]
            others = ", ".join(
                f"[{i}] {a}" for i, _, a in candidates[1:]
            )
            print(
                f"Note: {len(candidates)} host-API views of the same "
                f"physical mic; using {api} "
                f"(skipped {others})."
            )
            rate = _probe_rate(sd, idx)
            return idx, info, api, rate
        print("Ambiguous device candidates — aborting:")
        for i, d, api in candidates:
            print(f"  [{i}] hostapi={api} name={d['name']}")
        raise SystemExit(
            "Multiple endpoints match; re-run with --hostapi "
            "(e.g. --hostapi WASAPI) to disambiguate."
        )

    idx, info, api = candidates[0]

    # Probe supported capture rates; prefer 16 kHz, else best native.
    rate = _probe_rate(sd, idx)
    return idx, info, api, rate


def _probe_rate(sd, device_index: int) -> int:
    """First supported rate from _PROBE_RATES via check_input_settings."""
    for r in _PROBE_RATES:
        try:
            sd.check_input_settings(
                device=device_index, channels=1, samplerate=r,
                dtype="float32",
            )
            return int(r)
        except Exception:
            continue
    raise SystemExit(
        f"Device [{device_index}] accepts none of the probe rates "
        f"{list(_PROBE_RATES)}."
    )


def record_utterance(sd, device_index: int, native_rate: int,
                     phrase: str, rep: int, out_dir: Path) -> Path:
    from src.voice.audio_capture_service import resample_to_16k

    print(f"\n>>> Say: {phrase!r}  (rep {rep}/{REPS})")
    print(
        f"    Recording in {PRE_ROLL_S:.0f}s at {native_rate} Hz "
        f"(-> {TARGET_RATE} Hz WAV) — speak after the pause."
    )
    time.sleep(PRE_ROLL_S)

    frames = []
    stream = sd.InputStream(
        device=device_index,
        samplerate=native_rate,
        channels=1,
        dtype="float32",
        blocksize=1024,
    )
    stream.start()
    print(f"    REC [started device={device_index} rate={native_rate}]")
    t0 = time.time()
    try:
        while time.time() - t0 < CAPTURE_SECONDS:
            data, _ = stream.read(1024)
            frames.append(data.copy())
    finally:
        stream.stop()
        stream.close()
    print("    REC [stopped]")

    audio = np.concatenate(frames).reshape(-1)
    if native_rate != TARGET_RATE:
        audio = resample_to_16k(audio, float(native_rate))
    frame = 160  # 10 ms at 16 kHz
    n = (audio.size // frame) * frame
    fr = audio[:n].reshape(-1, frame)
    rms = np.sqrt(np.mean(fr * fr, axis=1))
    voiced = np.nonzero(rms > 0.005)[0]
    if voiced.size:
        audio = audio[
            max(0, voiced[0] - 5) * frame:
            min(len(fr), voiced[-1] + 10) * frame
        ]
    if audio.size < TARGET_RATE // 10:
        raise RuntimeError("Captured audio too short — re-record this rep.")

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    path = out_dir / f"{slugify(phrase)}_r{rep}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm.tobytes())
    print(f"    saved {path.name} ({len(audio) / TARGET_RATE:.1f}s)")
    return path


def slugify(text: str) -> str:
    return "_".join(
        s for s in "".join(
            c if c.isalnum() else "_" for c in text.lower()
        ).split("_") if s
    )[:40]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--device", choices=["intel", "realtek"],
        help="semantic microphone selector",
    )
    ap.add_argument(
        "--hostapi",
        help="restrict to a host API: MME, DirectSound, WASAPI, WDM-KS",
    )
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="with --list-devices: show supported rates")
    ap.add_argument(
        "--force", action="store_true",
        help="overwrite existing recordings after explicit confirmation",
    )
    args = ap.parse_args()

    import sounddevice as sd

    if args.list_devices or not args.device:
        list_devices(sd, verbose=args.verbose)
        if not args.device:
            return

    idx, info, api, rate = resolve_device(sd, args.device, args.hostapi or "")
    out_dir = FIXTURES / "physical" / args.device
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSelected microphone: [{idx}] {info['name']}")
    print(f"  hostapi={api} default_rate={info.get('default_samplerate')} "
          f"max_input_channels={info.get('max_input_channels')}")
    print(f"  capture: {rate} Hz native -> {TARGET_RATE} Hz mono int16 "
          f"-> {out_dir}")
    print("  identity: semantic selector "
          f"{args.device!r} matched exactly one endpoint "
          f"(category={_category(info['name'])})")

    planned = [
        out_dir / f"{slugify(p)}_r{r}.wav"
        for p in PHRASES for r in range(1, REPS + 1)
    ]
    existing = [p for p in planned if p.exists()]
    if existing and not args.force:
        raise SystemExit(
            f"{len(existing)} recording(s) already exist in {out_dir}. "
            "Re-run with --force to overwrite, or delete them first."
        )
    if existing and args.force:
        answer = input(
            f"Overwrite {len(existing)} existing recordings in "
            f"{out_dir}? [y/N] "
        )
        if answer.strip().lower() != "y":
            raise SystemExit("Aborted; existing recordings preserved.")

    ok, failed = 0, []
    for phrase in PHRASES:
        for rep in range(1, REPS + 1):
            path = out_dir / f"{slugify(phrase)}_r{rep}.wav"
            if path.exists() and args.force:
                path.unlink()
            try:
                record_utterance(sd, idx, rate, phrase, rep, out_dir)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"    FAILED: {exc}")
                failed.append((phrase, rep))
    print(f"\nDone: {ok} recorded, {len(failed)} failed -> {out_dir}")
    if failed:
        for p, r in failed:
            print(f"  {p} rep {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
