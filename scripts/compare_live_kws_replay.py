"""V10.7.3 — live-KWS vs fresh-replay discrepancy experiment.

Runs the production VoiceOS capture + wake-word stack on the real
microphone WHILE recording the raw multichannel input in memory, then
replays the *same captured waveform* through independent diagnostic
KWS instances under controlled variations:

  A  live-equivalent: fresh engine, original 100 ms frame sequence
  B  fresh stream on a second spotter, identical frame sequence
  C  concatenated-waveform replay (diagnose_wake_word_input style)
  D  44.1 kHz capture resampled to 16 kHz (production resampler,
     per-frame and continuous variants)
  E  trailing-silence sweep
  F  INT8-all vs INT8-enc/FP32-dec/INT8-join decoder config
  G  tokens()/timestamps() dump on first detection
  H  stream-position sensitivity map on the strongest speech segment
  I  run-to-run variance on the detected segment (N fresh engines)
  J  replay of the exact production KWS input (last <=5 s diag ring)

Everything stays in memory; no PCM is persisted. Output:
reports/v1073_kws_stream_comparison.json (metrics only).

Usage:
    py scripts/compare_live_kws_replay.py [--seconds 45]
        [--index N | --device NAME] [--no-rate-probe]

During the capture window speak "Hello Jarvis" several times at
normal volume (~20 cm), pausing ~2 s between attempts. Note the wall
time of each attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.voice.audio_capture_service import (  # noqa: E402
    AudioCaptureService,
    resample_to_16k,
    TARGET_SAMPLE_RATE,
)
from src.voice.audio_device_manager import AudioDeviceManager  # noqa: E402
from src.voice.wake_word_detection_service import (  # noqa: E402
    WakeWordDetectionService,
)
from src.voice.wake_word_engine import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    SherpaOnnxKeywordSpotter,
    create_engine,
)

REPORT = ROOT / "reports" / "v1073_kws_stream_comparison.json"
CHUNK = 1600  # production frame: 100 ms at 16 kHz


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return (np.frombuffer(w.readframes(w.getnframes()),
                              dtype=np.int16)
                .astype(np.float32) / 32768.0)


def _engine_ok(eng) -> bool:
    return getattr(eng, "name", "") == "sherpa-onnx-kws"


def _feed_process(eng, audio: np.ndarray, chunk: int = CHUNK):
    """Production-style feeding: eng.process() per chunk.

    Returns list of per-chunk records (no PCM). eng.process() already
    performs accept_waveform + decode loop + get_result and internally
    freshens the stream after a keyword — identical to live.
    """
    out = []
    for i in range(0, len(audio) - chunk + 1, chunk):
        c = audio[i:i + chunk]
        t0 = time.perf_counter()
        conf = eng.process(np.ascontiguousarray(c, dtype=np.float32))
        ms = (time.perf_counter() - t0) * 1000.0
        out.append({
            "frame": i // chunk,
            "samples": int(c.size),
            "process_ms": round(ms, 3),
            "confidence": conf,
            "keyword": eng.keyword_detected(),
        })
    return out


def _detections(records):
    return [
        {"frame": r["frame"], "t_s": round(r["frame"] * CHUNK / 16000.0, 2),
         "keyword": r["keyword"]}
        for r in records if r["confidence"] >= 0.25 and r["keyword"]
    ]


def _fresh_stream_feed(spotter, audio: np.ndarray, chunk: int = CHUNK):
    """Manual sherpa calls on ONE fresh stream (no auto-reset)."""
    stream = spotter._spotter.create_stream()
    out = []
    for i in range(0, len(audio) - chunk + 1, chunk):
        c = np.ascontiguousarray(audio[i:i + chunk], dtype=np.float32)
        t0 = time.perf_counter()
        stream.accept_waveform(TARGET_SAMPLE_RATE, c)
        ndec = 0
        while spotter._spotter.is_ready(stream):
            spotter._spotter.decode_stream(stream)
            ndec += 1
        r = spotter._spotter.get_result(stream)
        ms = (time.perf_counter() - t0) * 1000.0
        out.append({
            "frame": i // chunk, "samples": int(c.size),
            "decode_calls": ndec, "process_ms": round(ms, 3),
            "result": r or "",
        })
    # token/timestamp dump (best effort, once)
    tok_info = {}
    try:
        toks = spotter._spotter.tokens(stream)
        tok_info["tokens"] = list(toks)[:50] if toks else []
    except Exception as exc:  # noqa: BLE001
        tok_info["tokens_error"] = str(exc)
    try:
        ts = spotter._spotter.timestamps(stream)
        tok_info["timestamps"] = [float(x) for x in list(ts)[:50]]
    except Exception as exc:  # noqa: BLE001
        tok_info["timestamps_error"] = str(exc)
    return out, tok_info


def _speech_segments(mono: np.ndarray, thr: float = 0.02,
                     min_frames: int = 3):
    """100 ms-frame RMS segments (same logic as diagnose STT check)."""
    hop = TARGET_SAMPLE_RATE // 10
    n = len(mono) // hop
    if n == 0:
        return []
    w = mono[: n * hop].reshape(n, hop)
    wrms = np.sqrt(np.mean(w * w, axis=1))
    active = wrms > thr
    segs, start, gap = [], None, 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 6:
                segs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        segs.append((start, n))
    return [(s, e, float(wrms[s:e].max()))
            for s, e in segs if (e - s) >= min_frames]


# ----------------------------------------------------------------------
# phase 1: live capture (production stack + raw rings, memory only)
# ----------------------------------------------------------------------
def run_capture(sd, device_index: int, seconds: float,
                rate_probe: bool) -> dict:
    mgr = AudioDeviceManager(sd=sd)
    dev = None
    for d in mgr.list_input_devices():
        if d.index == int(device_index):
            dev = d
            break
    if dev is None:
        raise RuntimeError(f"device index {device_index} not found")
    max_ch = int(getattr(dev, "max_input_channels", 1) or 1)
    use_ch = max(1, min(max_ch, 4))

    raw16_frames, raw44_frames = [], []

    def cb16(indata, frames, ti, status):
        raw16_frames.append(np.asarray(indata).copy())

    def cb44(indata, frames, ti, status):
        raw44_frames.append(np.asarray(indata).copy())

    prod_detections = []
    prod_kws_tail = {"audio": None}

    cap = AudioCaptureService(sd=sd, device_manager=mgr)
    engine = create_engine()
    det = WakeWordDetectionService(
        cap, engine, threshold=0.25,
        on_detected=lambda kw: prod_detections.append(
            {"t": time.time(), "keyword": kw}),
    )

    raw16 = sd.InputStream(device=device_index, channels=use_ch,
                           samplerate=TARGET_SAMPLE_RATE, blocksize=1600,
                           dtype="float32", callback=cb16)
    raw44 = None
    if rate_probe:
        raw44 = sd.InputStream(
            device=device_index, channels=use_ch,
            samplerate=float(dev.default_samplerate), blocksize=4096,
            dtype="float32", callback=cb44)

    stats_log = []
    t0 = time.monotonic()
    try:
        cap.start(device_id=dev.device_id)
        det.start()
        raw16.start()
        if raw44 is not None:
            raw44.start()
        print(f"[v1073] capturing {seconds:.0f}s on {dev.name!r} "
              f"({use_ch} ch) — speak 'Hello Jarvis' now, pausing "
              f"~2 s between attempts.")
        while time.monotonic() - t0 < seconds:
            time.sleep(1.0)
            st = det.stats()
            stats_log.append({
                "t": round(time.monotonic() - t0, 1),
                "captured": st.get("frames_captured"),
                "processed": st.get("frames_processed"),
                "queue_depth": st.get("queue_depth"),
                "kws_rms": st.get("kws_input_rms"),
                "kws_peak": st.get("kws_input_peak"),
                "selected_channel": st.get("selected_channel"),
                "detections": st.get("detections"),
            })
            print(f"  t={time.monotonic()-t0:5.1f}s "
                  f"cap={st.get('frames_captured')} "
                  f"proc={st.get('frames_processed')} "
                  f"kws_rms={st.get('kws_input_rms')} "
                  f"sel={st.get('selected_channel')} "
                  f"det={st.get('detections')}")
        # Grab the exact production KWS input BEFORE stop() clears it.
        try:
            prod_kws_tail["audio"] = det.diagnostic_audio()
        except Exception:  # noqa: BLE001
            pass
    finally:
        det.stop()
        cap.stop()
        raw16.stop(); raw16.close()
        if raw44 is not None:
            raw44.stop(); raw44.close()
        engine.close()

    raw16m = (np.concatenate(raw16_frames, axis=0)
              if raw16_frames else np.zeros((0, use_ch), np.float32))
    raw44m = (np.concatenate(raw44_frames, axis=0)
              if raw44_frames else np.zeros((0, use_ch), np.float32))
    return {
        "device": {"index": device_index, "name": dev.name,
                   "device_id": dev.device_id,
                   "default_samplerate": dev.default_samplerate,
                   "max_input_channels": max_ch},
        "channels_opened": int(raw16m.shape[1]) if raw16m.ndim > 1 else 1,
        "raw16_shape": list(raw16m.shape),
        "raw44_shape": list(raw44m.shape),
        "raw16": raw16m,
        "raw44": raw44m,
        "prod_detections": prod_detections,
        "prod_stats_log": stats_log,
        "prod_kws_tail": prod_kws_tail["audio"],
        "prod_final": {
            k: cap.stats().get(k) for k in (
                "frames_captured", "frames_dropped", "queue_depth",
                "input_channels", "selected_channel",
                "channel_switch_count", "overflow_callbacks",
                "status_warnings", "sample_rate")
        },
    }


# ----------------------------------------------------------------------
# phase 2: experiments on the captured waveform (memory only)
# ----------------------------------------------------------------------
def run_experiments(capres: dict) -> dict:
    out = {"experiments": {}, "notes": []}
    raw16 = capres["raw16"]
    if raw16.size == 0:
        out["error"] = "no audio captured"
        return out
    n_ch = raw16.shape[1] if raw16.ndim > 1 else 1
    sel = int(capres["prod_final"].get("selected_channel") or 0)
    sel = min(sel, n_ch - 1)
    ch0 = (raw16[:, sel].astype(np.float32).reshape(-1)
           if raw16.ndim > 1 else raw16.reshape(-1).astype(np.float32))
    out["selected_channel_used"] = sel
    out["signal_seconds"] = round(len(ch0) / TARGET_SAMPLE_RATE, 2)

    segs = _speech_segments(ch0)
    out["speech_segments"] = [
        {"start_s": round(s * 0.1, 1), "end_s": round(e * 0.1, 1),
         "peak_rms": round(p, 4)} for s, e, p in segs]

    # -- A: live-equivalent framed replay -------------------------------
    engA = create_engine()
    if _engine_ok(engA):
        recA = _feed_process(engA, ch0)
        out["experiments"]["A_live_equivalent_fresh_engine"] = {
            "engine": engA.name,
            "frames": len(recA),
            "detections": _detections(recA),
            "first_divergence": None,
        }
        # stash records for B comparison
        out["_recA"] = recA
    else:
        out["experiments"]["A_live_equivalent_fresh_engine"] = {
            "error": "engine unavailable"}
    engA.close()

    # -- B: fresh stream, identical sequence, second spotter -----------
    engB = create_engine()
    if _engine_ok(engB):
        recB, tokB = _fresh_stream_feed(engB, ch0)
        detB = [r for r in recB if r["result"]]
        out["experiments"]["B_fresh_stream_identical_frames"] = {
            "engine": engB.name,
            "frames": len(recB),
            "detections": [
                {"frame": r["frame"],
                 "t_s": round(r["frame"] * CHUNK / 16000.0, 2),
                 "keyword": r["result"]} for r in detB],
            "total_decode_calls": sum(r["decode_calls"] for r in recB),
            "tokens": tokB.get("tokens"),
            "timestamps": tokB.get("timestamps"),
            "tokens_error": tokB.get("tokens_error"),
            "timestamps_error": tokB.get("timestamps_error"),
        }
        # per-frame divergence A vs B
        recA = out.get("_recA") or []
        div = None
        for a, b in zip(recA, recB):
            ka = a["keyword"] or ""
            kb = b["result"] or ""
            if (a["confidence"] >= 0.25) != bool(kb):
                div = {"frame": a["frame"], "A": ka or a["confidence"],
                       "B": kb}
                break
        out["experiments"]["B_fresh_stream_identical_frames"][
            "first_divergence_vs_A"] = div
    else:
        out["experiments"]["B_fresh_stream_identical_frames"] = {
            "error": "engine unavailable"}
    engB.close()
    out.pop("_recA", None)

    # -- C: concatenated replay (diagnose-style) ------------------------
    engC = create_engine()
    if _engine_ok(engC):
        pad = np.zeros(TARGET_SAMPLE_RATE // 2, dtype=np.float32)
        cat = np.concatenate([pad, ch0, pad])
        recC = _feed_process(engC, cat)
        out["experiments"]["C_concatenated_replay"] = {
            "engine": engC.name,
            "pad_s": 0.5,
            "frames": len(recC),
            "detections": _detections(recC),
        }
    else:
        out["experiments"]["C_concatenated_replay"] = {
            "error": "engine unavailable"}
    engC.close()

    # -- D: 44.1 kHz -> 16 kHz resample variants -------------------------
    raw44 = capres.get("raw44")
    if raw44 is not None and raw44.size:
        nch44 = raw44.shape[1] if raw44.ndim > 1 else 1
        sel44 = min(sel, nch44 - 1)
        ch44 = (raw44[:, sel44].astype(np.float32).reshape(-1)
                if raw44.ndim > 1 else raw44.reshape(-1))
        rate = float(capres["device"]["default_samplerate"])
        dres = {"source_rate": rate, "selected_channel": sel44}
        # D1: continuous resample of whole signal (state preserved)
        dur = len(ch44) / rate
        n16 = int(round(dur * TARGET_SAMPLE_RATE))
        cont = np.interp(
            np.linspace(0, dur, n16, endpoint=False),
            np.linspace(0, dur, len(ch44), endpoint=False),
            ch44).astype(np.float32)
        # D2: production-style per-100ms-frame resample (no state)
        hop44 = int(round(rate * 0.1))
        per_frames = []
        for i in range(0, len(ch44) - hop44 + 1, hop44):
            seg = resample_to_16k(ch44[i:i + hop44].astype(np.float32),
                                  rate)
            per_frames.append(seg)
        per = (np.concatenate(per_frames).astype(np.float32)
               if per_frames else np.zeros(0, np.float32))
        for tag, sig in (("continuous", cont), ("per_frame", per)):
            engD = create_engine()
            if _engine_ok(engD):
                recD = _feed_process(engD, sig)
                dres[tag] = {
                    "samples_16k": int(sig.size),
                    "detections": _detections(recD),
                }
            else:
                dres[tag] = {"error": "engine unavailable"}
            engD.close()
        # reference: native 16k (already have A/C); store lengths only
        dres["native16k_samples"] = int(ch0.size)
        out["experiments"]["D_rate_44k_to_16k"] = dres
    else:
        out["experiments"]["D_rate_44k_to_16k"] = {
            "skipped": "no 44.1k probe (--no-rate-probe)"}

    # -- E: trailing-silence sweep on native 16k selected channel ------
    esweep = {}
    for trail_s in (0.0, 0.25, 0.5, 1.0, 2.0):
        sig = np.concatenate(
            [ch0, np.zeros(int(trail_s * TARGET_SAMPLE_RATE),
                           dtype=np.float32)])
        engE = create_engine()
        if _engine_ok(engE):
            recE = _feed_process(engE, sig)
            esweep[f"trail_{trail_s}s"] = _detections(recE)
        else:
            esweep[f"trail_{trail_s}s"] = {"error": "engine unavailable"}
        engE.close()
    out["experiments"]["E_trailing_silence"] = esweep

    # -- F: decoder config int8-all vs fp32 decoder --------------------
    fdec = _decoder_experiment(ch0)
    out["experiments"]["F_decoder_config"] = fdec

    # -- G already inside B (tokens/timestamps)

    # -- H: stream-position sensitivity on strongest segment -----------
    if segs:
        # pick segment with max peak_rms that is >= 0.5 s long
        cand = [s for s in segs if (s[1] - s[0]) >= 5]
        s0, e0, _ = max(cand or segs, key=lambda x: x[2])
        seg = ch0[s0 * 1600: e0 * 1600]
        lead_pad = np.zeros(8000, dtype=np.float32)
        trail_pad = np.zeros(8000, dtype=np.float32)
        hmap = []
        for lead in range(0, 4801, 400):
            sig = np.concatenate(
                [np.zeros(lead, dtype=np.float32), lead_pad, seg,
                 trail_pad])
            engH = create_engine()
            if _engine_ok(engH):
                recH = _feed_process(engH, sig)
                dets = _detections(recH)
                hmap.append({"lead_samples": lead,
                             "detected": bool(dets),
                             "first_frame": dets[0]["frame"] if dets else None})
            engH.close()
        out["experiments"]["H_position_sensitivity"] = {
            "segment_start_s": round(s0 * 0.1, 1),
            "segment_end_s": round(e0 * 0.1, 1),
            "segment_peak_rms": round(_, 4),
            "map": hmap,
        }
    else:
        out["experiments"]["H_position_sensitivity"] = {
            "skipped": "no speech segments"}

    # -- I: run variance on the strongest segment ----------------------
    if segs:
        cand = [s for s in segs if (s[1] - s[0]) >= 5]
        s0, e0, _ = max(cand or segs, key=lambda x: x[2])
        seg = ch0[s0 * 1600: e0 * 1600]
        sig = np.concatenate(
            [np.zeros(8000, dtype=np.float32), seg,
             np.zeros(8000, dtype=np.float32)])
        hits = 0
        trials = 8
        frames_hit = []
        for _t in range(trials):
            engI = create_engine()
            if _engine_ok(engI):
                recI = _feed_process(engI, sig)
                dets = _detections(recI)
                if dets:
                    hits += 1
                    frames_hit.append(dets[0]["frame"])
            engI.close()
        out["experiments"]["I_run_variance"] = {
            "segment_start_s": round(s0 * 0.1, 1),
            "trials": trials, "detections": hits,
            "first_frames": frames_hit,
        }
    else:
        out["experiments"]["I_run_variance"] = {
            "skipped": "no speech segments"}

    # -- J: exact production KWS input (last <=5 s) ---------------------
    tail = capres.get("prod_kws_tail")
    if tail is not None and getattr(tail, "size", 0):
        engJ = create_engine()
        if _engine_ok(engJ):
            recJ = _feed_process(engJ, tail.astype(np.float32))
            out["experiments"]["J_production_kws_tail_replay"] = {
                "engine": engJ.name,
                "tail_seconds": round(float(tail.size) / 16000.0, 2),
                "detections": _detections(recJ),
                "note": ("last <=5 s of exact live KWS input; "
                         "covers only the recording tail"),
            }
        else:
            out["experiments"]["J_production_kws_tail_replay"] = {
                "error": "engine unavailable"}
        engJ.close()
    else:
        out["experiments"]["J_production_kws_tail_replay"] = {
            "skipped": "no diagnostic tail captured"}

    return out


def _decoder_experiment(ch0: np.ndarray) -> dict:
    """INT8-all (production) vs INT8-enc + FP32-dec + INT8-join."""
    model = DEFAULT_MODEL_DIR
    sub = next(model.glob("sherpa-onnx-kws-*"), None)
    if sub is None:
        return {"error": "model dir not found"}
    enc = sorted(sub.glob("encoder*.onnx"))[0]
    dec_i8 = sorted(sub.glob("decoder*.int8.onnx"))
    dec_fp = sorted(p for p in sub.glob("decoder*.onnx")
                    if ".int8." not in p.name)
    join = sorted(sub.glob("joiner*.onnx"))[0]
    kw = sub / "keywords_jarvis.txt"
    toks = sub / "tokens.txt"
    res = {}
    import sherpa_onnx
    for tag, dec in (("int8_all", dec_i8[0] if dec_i8 else None),
                     ("fp32_decoder", dec_fp[0] if dec_fp else None)):
        if dec is None:
            res[tag] = {"skipped": "decoder file missing"}
            continue
        try:
            sp = sherpa_onnx.KeywordSpotter(
                tokens=str(toks), encoder=str(enc), decoder=str(dec),
                joiner=str(join), keywords_file=str(kw),
                num_threads=2, provider="cpu")
        except Exception as exc:  # noqa: BLE001
            res[tag] = {"error": str(exc)}
            continue
        # feed selected-channel audio, 0.5 s pads, via fresh stream
        sig = np.concatenate(
            [np.zeros(8000, np.float32), ch0, np.zeros(8000, np.float32)])
        rec, _tok = _fresh_stream_feed(
            type("W", (), {"_spotter": sp})(), sig)
        dets = [{"frame": r["frame"], "keyword": r["result"]}
                for r in rec if r["result"]]
        res[tag] = {"decoder": dec.name, "detections": dets}
    return res


# ----------------------------------------------------------------------
def _merge_report(live: dict) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    prev = {}
    if REPORT.exists():
        try:
            prev = json.loads(REPORT.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
    prev["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    prev["live_run"] = live
    REPORT.write_text(json.dumps(prev, indent=2, default=str),
                      encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="substring of input device name")
    ap.add_argument("--no-rate-probe", action="store_true")
    args = ap.parse_args()

    import sounddevice as sd

    index = args.index
    if index is None:
        # default: prefer the MME/DirectSound Intel array (index of the
        # truncated-name 4ch 44.1k device), else Windows default input.
        try:
            di = sd.default.device[0]
        except Exception:  # noqa: BLE001
            di = -1
        cands = [d for d in sd.query_devices()
                 if d.get("max_input_channels", 0) > 0]
        index = di if di >= 0 else (cands[0]["index"] if cands else None)
        if args.device:
            q = args.device.lower()
            for d in cands:
                if q in (d.get("name") or "").lower():
                    index = d["index"]
                    break
    if index is None:
        print("[v1073] no input device")
        return 1
    print(f"[v1073] device index {index}")

    capres = run_capture(sd, index, args.seconds,
                         rate_probe=not args.no_rate_probe)

    live_summary = {
        "device": capres["device"],
        "channels_opened": capres["channels_opened"],
        "raw16_shape": capres["raw16_shape"],
        "raw44_shape": capres["raw44_shape"],
        "seconds": round(
            capres["raw16"].shape[0] / TARGET_SAMPLE_RATE
            if capres["raw16"].size else 0.0, 2),
        "prod_detections": capres["prod_detections"],
        "prod_final": capres["prod_final"],
        "prod_stats_log": capres["prod_stats_log"],
        "prod_kws_tail_seconds": (
            round(capres["prod_kws_tail"].size / 16000.0, 2)
            if capres.get("prod_kws_tail") is not None else None),
    }
    print("[v1073] live detections:", capres["prod_detections"])

    exp = run_experiments(capres)
    # strip heavy internals before merge
    for k in ("raw16", "raw44", "prod_kws_tail"):
        capres.pop(k, None)
    live_summary["experiments"] = exp.pop("experiments", {})
    live_summary["experiment_notes"] = exp.get("notes", [])
    live_summary["speech_segments"] = exp.get("speech_segments")
    _merge_report(live_summary)
    print(f"[v1073] wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
