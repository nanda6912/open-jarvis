# VoiceOS — Physical Voice Recognition Baseline Report

Date: 2026-11-19 · Status: **baseline tooling ready; physical trials PENDING user participation**

This report separates **measured facts** (existing Nov 18–19 production logs,
offline fixtures) from **hypotheses** (require the controlled trials below).
No production KWS/STT model, threshold, VAD, channel-selection or
recognition code was changed for this benchmark.

---

## 1. Methodology

A new diagnostic-only runner was added:

```
py scripts/voice_recognition_benchmark.py kws       # 20 positive wake trials
py scripts/voice_recognition_benchmark.py kws-neg   # 10 negative trials
py scripts/voice_recognition_benchmark.py stt       # 20 known-phrase STT trials
```

Properties:

- Uses the **production** `AudioCaptureService` + `AudioDeviceManager`
  (same mic resolution as the app), production `create_engine()` KWS
  (keyword config + 0.25 threshold unchanged), production
  `create_recognizer()` STT (Moonshine Base INT8) and the production
  `SpeechRecognitionService` VAD.
- **Never instantiates `VoiceSessionController` or the command registry** —
  transcripts cannot reach command handlers; no Windows commands can run.
- Trials are user-paced (Enter to arm, 5 s detection window for KWS,
  12 s utterance cap for STT). A fresh detector generation per KWS trial
  mirrors a real standby cycle.
- Per trial it records: detection/miss, `detection_method`
  (live/rescore/fresh_stream), detection latency, `kws_rms`/`kws_peak`,
  `speech_end_to_detection_seconds`, capture stats (per-channel RMS/peaks,
  selected channel, clipped frames, queue depth, dropped frames,
  overflow/status warnings), STT `last_attempt` record (VAD outcome,
  voiced/rejected frames, effective threshold, noise floor, decode
  timing), expected-vs-actual transcript, exact-match/WER/CER, and
  psutil CPU%/RSS/available-RAM snapshots.
- Results: `reports/voice_benchmark_<mode>_<timestamp>.json` + `.csv`
  (one row per trial, ground truth included). No PCM is written.

Controlled conditions (per spec): same seat/distance, quiet room, normal
volume, no Windows gain/enhancement changes, KWS and STT measured in
separate runs.

## 2. Hardware / production configuration (measured)

- Host: Windows laptop, Python 3.13 venv `VoiceOS\venv`.
- Mic: Intel Smart Sound Technology microphone array (4-channel input;
  production capture opens up to 4 ch and selects strongest channel).
- KWS: sherpa-onnx engine via `create_engine()`, threshold 0.25,
  cooldown 3 s, async rescore enabled (`kws_rescore` policy).
- STT: Moonshine Base INT8 (`create_recognizer()` auto-discovery;
  eval models were moved out of the production dir in Task 2).
- VAD: adaptive threshold (`_effective_threshold` = configured
  `speech_threshold` adjusted by noise floor), hangover/pre-roll as in
  `SpeechRecognitionService`.

## 3. Evidence from existing Nov 18–19 production logs (MEASURED)

Source: `%LocalAppData%\VoiceOS\logs\voiceos.log` (+ rotated .1–.5).

| Metric | Value |
|---|---|
| Voice sessions started | 6 |
| Wake detections — live path | 3 |
| Wake detections — rescore path | 3 |
| Rescore attempts `hit=True` | 3 |
| Rescore attempts `hit=False` | 60 |
| `speech_end_to_detection` p50 / p95 | 1.80 s / 2.01 s (n=6) |
| Recognition outcomes | no_speech=34, intent_unknown=13, executed=11, entity_ambiguous=2, recovered=1, handler_failed=3, confirmation_requested=3, exit_session=2 |

Interpretation (facts only): rescore recovered 3 of 6 activations; the
majority of rescore evaluations (60) found no keyword — consistent with
the reported "five unsuccessful rescoring windows before a live
detection" pattern. `no_speech` dominates recognition outcomes, matching
the report of empty/quiet captures. Frame drops/queue backlog were not
reported in these logs (capture stats show bounded queue behavior).

Caveat: ground-truth spoken phrases were not recorded per event, so
these logs **cannot** separate "user spoke poorly/too quietly" from
"pipeline failed" — hence the controlled trials.

## 4. Controlled-trial results — PENDING USER

The interactive benchmark has not been run on the laptop yet. Required
steps for the user (≈15–20 min, quiet room, normal volume):

1. `kws` — 20 × "Hello Jarvis" once each; record first-attempt recall,
   live-vs-rescore split, latency p50/p95, per-window RMS/peak/clipping.
2. `kws-neg` — 10 ordinary phrases; expect zero activations.
3. `stt` — 20 fixed phrases; exact-match, WER/CER, VAD reject/empty
   counts, decode latency, signal metrics per utterance.

Acceptance rules: a missed first attempt counts as failure even if a
repetition later succeeds; no threshold changes to improve scores.

## 5. Root-cause hypotheses (RANKED — pending trial data)

1. **Capture acoustics / distance** (most likely, per Task 2 evidence):
   far-field Intel-array audio degrades both KWS margin and Moonshine
   accuracy; rescore exists precisely because live KWS misses occur.
   Trial signal metrics (RMS/peak per channel) will confirm or refute.
2. **VAD gating too aggressive for quiet speech**: `no_speech` dominates
   outcomes; if trials show `vad_rejected`/empty on clearly spoken
   phrases with adequate RMS, the effective threshold vs noise floor is
   suspect. Do NOT change it before data.
3. **Channel selection**: if per-channel RMS shows speech concentrated
   off channel 0 while selection picks another, first/last-word cutoff
   or level loss could explain inconsistent recognition.
4. **Endpoint/first-word truncation**: pre-roll/hangover constants vs
   actual speech onset — check `first/last word cut` in STT trials.
5. **Resource contention**: STT decode p95 vs CPU/RAM samples —
   correlation would show throttling; unlikely primary cause.
6. **TTS gating residue**: VAD state after playback pause — covered by
   Task 5 pause tests; trials will show if post-TTS utterances fail
   disproportionately.

## 6. What still needs the user

- All three benchmark modes above (physical voice required).
- Optional WAV capture for waveform-level analysis — NOT implemented;
  if needed it will be a separate explicit-consent local-only option
  with documented deletion. Current metrics (RMS/peak/clipping/duration)
  require no audio persistence.

## 7. Files

- `scripts/voice_recognition_benchmark.py` — new diagnostic runner.
- No production source files were modified for this benchmark.
- Results will land in `reports/voice_benchmark_*` when run.

*Stop point: awaiting user-run trial data before any fix work.*
