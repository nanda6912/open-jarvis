# Phase 1 Reliability Recovery Report

Repository: Jarvis working copy (local checkout)
Baseline backup: sibling backup directory `Jarvis_pre_phase1_20261118_033505`
Git state at baseline: no commits; all files untracked. No rollback assumed.

---

## Original baseline

- Test suite (pre-change): **1089 passed, 4 failed** in ~354 s.
- Known failures at baseline:
  1. `tests/test_dynamic_mic.py::TestWakeWordAfterSwitch::test_detection_survives_device_switch` — duplicate wake emission.
  2. `tests/test_live_reliability.py::TestRecordedCommandPipeline::test_fixture[hey_jarvis.wav]` — STT returned `Hage-arvis` (Task 2 scope).
  3. `tests/test_post_standby_recognition.py::TestResumeIntegration::test_resume_listener_resets_stt_and_recovers` — timing-sensitive; passed on isolated rerun.
  4. `tests/test_v108_task1_diagnostics.py::test_activity_close_reports_rescore_miss_truthfully` — timing-sensitive; passed on isolated rerun.
- Production logs (`%LocalAppData%\VoiceOS\logs`) showed KWS live inference running, rescore jobs completing asynchronously, and missed detections on low-level activity windows.

---

## Task 1 — Wake-word recognition

### Root cause (confirmed defect)

Race between wake handling and the standby watchdog in
`src/controllers/microphone_controller.py`:

- `_handle_wake_word` → `suspend_detection()` kept `status=LISTENING` while
  stopping the detection worker (join up to 2 s).
- The monitor thread's `_check_detection_liveness` saw a dead worker +
  `LISTENING` and called `_restart_detection`, creating a **new**
  `WakeWordDetectionService`.
- `suspend_detection` then executed `self._detection = None`, discarding the
  live replacement **without stopping it** → orphaned worker kept consuming
  frames.
- Each service has its own cooldown (`_last_detection` resets per instance),
  so the orphan emitted additional `hello jarvis` wake events and could
  suspend the newer standby detector. This produced the duplicate wake in
  `test_detection_survives_device_switch` (3 wakes, expected 2) and is a
  real production risk (duplicate/false activations, killed standby
  detection until watchdog restart).

Secondary gap: `on_detected=self._handle_wake_word` was identical for every
service instance, so a stale worker's late emission was indistinguishable
from the current detector's.

### Investigation notes (no defect found)

- Engine lock scope verified: `SherpaOnnxKeywordSpotter.rescore()` acquires
  `self._lock` **per 100 ms decode chunk** (`wake_word_engine.py` ~line 386);
  live `process()` interleaves — a rescore sweep cannot starve live KWS.
- Live detection proven in production logs: `KWS_DETECTION_EMITTED
  keyword='HELLO JARVIS' via='live'` on real 4-channel Intel array audio
  (rms_max ≈ 0.195, peak ≈ 0.49).
- Rescore proven offline: `reports/diag.wav` and
  `tests/fixtures/commands/hello_jarvis.wav` both rescore-hit `HELLO JARVIS`;
  detection survives 20 dB attenuation of the fixture — gain is not the
  limiting factor.
- The only organic missed windows (2026-11-18 16:51 / 16:58, window_id 27/28)
  contained low-level, mostly-silent audio (rms_max ≈ 0.067, 2 active frames
  of 17) — consistent with noise/non-speech rather than a clear wake attempt.
- `reports/diag_kws_*.wav` diagnostic captures contain no discernible speech
  (STT returns empty on raw audio); they are not positive wake samples.
- Keyword file `keywords_jarvis.txt` (6 variants) is structurally compatible
  with the model `tokens.txt`; no tokenization defect found.

### Files modified

- `src/controllers/microphone_controller.py` (only file changed)

### Functions modified / added

- `_new_detection_service(generation, on_first_frame=None)` — **new**. Builds
  each `WakeWordDetectionService` with an `on_detected` closure bound to that
  exact instance (holder pattern).
- `_handle_wake_word(service, keyword)` — added stale-service identity guard:
  emissions from any service that is not the currently installed
  `self._detection` are logged and dropped before they can suspend the
  replacement detector or reach the session callback.
- `suspend_detection()` — atomically detaches `self._detection` under
  `_restart_lock` **before** stopping it; stop/join runs outside the lock so
  a worker blocked in a callback cannot stall the join. Watchdog sees
  `_detection is None` → "nothing to restart".
- `disable()` — same detach-first-under-lock protocol (identical race
  pattern).
- All three `WakeWordDetectionService` construction sites now use
  `_new_detection_service`.

Preserved: async KWS rescoring, live inference path, per-service cooldown,
session lifecycle, KWS models/keywords/thresholds, STT, HUD, mic capture.

### Tests executed (post-change)

| Suite | Result |
|---|---|
| `test_detection_survives_device_switch` ×6 | 6/6 passed (previously failing) |
| `tests/test_dynamic_mic.py` | 20/20 passed |
| Wake/session regression batch (12 files: wake_cycles, wake_word_after_session, standby_watchdog, post_standby_recognition, v1075_reliability, v10751_production_trace, v108_async_rescore, v108_task1_diagnostics, wake_word_latency, wake_diagnostics_v106, intel_array_wake_detection, v1071_hotfix) | 119/119 passed |
| Voice/application batch (test_voice, voice_reliability, mic_pipeline, live_reliability, live_fixes, application, conversation_response_latency, quick_acknowledgement, vad_recovery, vad_segmentation_v105, speech_accuracy_v105, command_corpus, intent_accuracy) | 340 passed, 1 failed* |
| Remaining non-Qt batch (18 files) | 594/594 passed |
| HUD/monitor/tray batch | 135/135 passed |
| Custom stress: enable → wake → resume → device switch → wake ×20, checking wake count and `wakeword-detect`/`kws-rescore` thread counts | 0 failures — exactly 2 wakes, exactly 2 detection threads while listening, 0 orphaned threads after suspend |

\* `test_live_reliability.py::test_fixture[hey_jarvis.wav]` — known STT
model issue (`Hage-arvis`), tracked under Task 2, unrelated to this change.

### Physical validation — PENDING

Automated evidence shows detection works when real "Hello Jarvis" audio
arrives, but an end-to-end physical microphone session (wake → command →
response → goodbye → repeat) has not yet been run by the user. Recommend a
physical pass, optionally with `VOICEOS_KWS_PARITY_DIAGNOSTIC=1` to capture
window audio if misses recur.

---

## Task 2 — Speech-to-text accuracy

### Pipeline audit — no defect found

- `src/voice/stt_engine.py`: `transcribe()` → `prepare_utterance()`
  (peak-normalize toward 0.7, gain cap 6×) → `accept_waveform` →
  `decode_stream` → `result.text`. Decoder config and transcript
  extraction are correct; empty-input is guarded (`audio.size == 0`
  early-return in `prepare_utterance`; zero-length ONNX decode errors
  observed only in a manual diagnostic, not reachable via the service's
  `min_utterance` gate).
- `src/voice/speech_recognition_service.py`: adaptive energy VAD with
  noise-floor estimation (speech frames excluded from the floor — V10.5
  fix present), pre-roll, silence hangover, `max_utterance` cap,
  `min_utterance=0.3 s` drop. `_finalize()` concatenates the exact VAD
  frames — no edge trimming, no truncation; utterance audio reaches
  `transcribe()` intact.
- Sample rate/channel path verified earlier: 16 kHz mono float32 from
  `AudioCaptureService`; strongest-channel selection on the Intel array.

### Controlled evaluation (existing manifests)

Production model: sherpa-onnx Moonshine Base INT8
(`models/stt/sherpa-onnx-moonshine-base-en-int8`).

| Corpus | WER | CER | Empty | Intent acc | Latency mean/p95 |
|---|---|---|---|---|---|
| `eval_manifest.json` (19 synthesized command fixtures) | 0.246 | 0.190 | 1/19 | 0.765 | 49.5 / 77.3 ms |
| `eval_physical_intel.json` (19 present Intel-array recordings; 2 manifest entries missing files) | 0.984 | 0.722 | 8/19 | 0.211 | 201 / 336.5 ms |

### Alternative models benchmarked (evidence gathered, none adopted)

| Model | Physical WER | Physical CER | Empty | Latency mean | Notes |
|---|---|---|---|---|---|
| Moonshine Base INT8 (production) | 0.984 | 0.722 | 8/19 | 201 ms | best clean-corpus accuracy |
| Moonshine Tiny INT8 | 1.907 | 1.288 | 9/19 | 48 ms | much worse |
| Zipformer-en 2023-06-26 (int8) | 1.053 | 0.736 | 2/19 | 70 ms | clean-corpus WER 0.560 — far worse overall |
| Whisper base.en (int8) | 0.946 | 0.854 | 1/19 | 306 ms | ~0.04 WER better on physical only, 6× latency, ~480 MB RSS peak, hallucinates `(static)` on non-speech |

Candidate models were staged under
`%LocalAppData%\VoiceOS\models\stt\` for evaluation only; the production
model directory and settings are unchanged. Moonshine v2 quantized
(2026-02-27) exists upstream but uses a different file layout
(`encoder_model.ort` + `decoder_model_merged.ort`) unsupported by the
installed sherpa-onnx 2.0.1 `from_moonshine` API — evaluating it would
require a binding upgrade; deferred.

### Root cause of physical-corpus failures — capture acoustics, not the model

Evidence:

1. A real close-talk microphone recording
   (`tests/fixtures/live/_hello_jarvis.wav`, peak 0.92) transcribes
   perfectly (`Hello Jarvis`) with the production model.
2. Several `physical/intel` WAVs contain **no speech at all**
   (speech-band RMS ≈ 0.0002–0.0004 — recording captured silence;
   e.g. `open_notepad_r1`, `set_volume_to_forty_r2/r3`,
   `what_time_is_it_r1`). Empty transcripts there are correct behaviour.
3. Files that do contain speech have adequate SNR (≈ 23–70 dB) but are
   far-field/reverberant through the Intel array; **all four models
   fail similarly** on them → acoustic degradation beyond these models,
   not a Moonshine-specific defect.
4. Preprocessing variants tested on the physical corpus — raw,
   `prepare_utterance`, edge-trimmed, VAD-segmented, peak-normalized,
   gain 4×, RMS-normalized (0.08/0.15), spectral subtraction — none
   produced a material improvement (WER stayed ≈ 0.87–1.10 depending on
   scoring). Gain and stationary-noise suppression are not the cause.
5. The production VAD path (segment → `transcribe`) is not worse than
   whole-file decode; it is the same audio the benchmark feeds.

### Empty vs inaccurate transcripts — separated

- **Empty (8/19 files):** predominantly recordings with no captured
  speech (capture-side) plus a few very short/quiet utterances.
- **Inaccurate (non-empty):** far-field speech decoded as plausible but
  wrong words (`Open the window` for "Open Devin", `14` for "Set volume
  to forty", `Opened Chrome` for "Open Chrome") — model limitation on
  degraded acoustics.

### Files involved

- Read-only audit: `src/voice/stt_engine.py`,
  `src/voice/speech_recognition_service.py`,
  `scripts/benchmark_stt.py`, `scripts/benchmark_v105_models.py`,
  `scripts/record_eval_samples.py`, test manifests/fixtures.
- New evaluation artifacts (reports only, no source change):
  `reports/stt_eval_intel_tiny.json`,
  `reports/stt_eval_intel_zipformer.json`,
  `reports/stt_eval_intel_whisper.json`,
  `reports/stt_eval_manifest_zipformer.json`.
- Staged (not activated) candidate models under
  `%LocalAppData%\VoiceOS\models\stt\` — harmless extra directories;
  production selection unchanged. Remove manually if unwanted.

### Changes implemented

**None.** Every pipeline stage was verified correct and every tested
preprocessing/model alternative was equal or worse. The smallest safe
change is no change; replacing the model is not supported by evidence
(whisper's marginal physical-WER gain costs 6× latency, ~2.4× memory,
and introduces non-speech hallucinations — unsafe for a command
assistant).

### Known issues / remaining work

- `test_live_reliability.py::test_fixture[hey_jarvis.wav]` still fails:
  Moonshine transcribes the fixture as `Hage-arvis`. "hey jarvis" is not
  the production wake phrase; this is a fixture/model-pronunciation
  edge case, not a pipeline bug.
- Physical evaluation set needs re-recording at usable distance/gain
  (several current files contain no speech) before further model
  conclusions can be drawn about far-field performance.
- Optional future work (requires approval): evaluate Moonshine v2
  quantized after a sherpa-onnx binding upgrade, or a larger
  far-field-capable model, on a refreshed physical corpus.

---

## Task 3 — Microphone device management

### Root cause

Persisted device ids are `hostapi:index:name`, where `index` is a
**runtime PortAudio enumeration position**, not a permanent identity.
The Intel Smart Sound array was observed at index `1` at startup and
index `2` in a later session banner. After any enumeration reshuffle
the stored `hostapi:index` pair no longer matches, so:

- `resolve_device()` fell through exact-id matching to hostapi+name
  resolution and logged `resolved by name` — capture worked.
- `_record_active_device()` then applied a *stricter* check
  (`matches_device_id` on the `hostapi:index` pair **and** a full-name
  equality that did not tolerate Windows' ~32-char name truncation or
  `®`/`™` markers). It concluded the configured device was "unavailable"
  and reported `fallback: <id> unavailable` even though the resolved
  device **was** the configured microphone at its new index.

Result: contradictory logs, a permanently stuck "fallback" status, and
a persisted id that never self-healed.

Windows Core Audio endpoint ids were investigated as a more stable
identity (`pycaw` is already a dependency) but pycaw exposes render
endpoint abstractions without a practical capture-endpoint-id →
PortAudio-index mapping, and adding a new dependency was not justified.
The implemented identity is therefore a documented deterministic
fingerprint: **host API + normalized name** (index-free), with
capability checks (channels, probed sample rates) unchanged.

### Files modified

- `src/voice/audio_device_manager.py`
- `src/controllers/microphone_controller.py`
- `tests/test_dynamic_mic.py` (new `TestStableResolution` cases)

### Changes implemented

`audio_device_manager.py`:

- `_norm_name()` / `_names_match()`: name normalization stripping
  trademark markers and punctuation, plus word-boundary prefix matching
  to tolerate Windows name truncation (short side ≥16 chars, boundary
  required so `mic array 1` ≠ `mic array 10`).
- `_match_by_id()`: shared matcher returning `(device, strategy,
  name_candidates)` with strategies `exact` / `name` / `ambiguous` /
  `index` / `none`. Name matching is scoped to the stored host API;
  duplicate names on one host API return `ambiguous` and are **never**
  broken by an index guess.
- `find_device()` / `resolve_device()`: rewritten on `_match_by_id`.
  Resolution order: exact id → hostapi + normalized name → hostapi +
  index → explicit default. **Hardened after review:** hostapi:index
  is now used *only* for nameless legacy ids (e.g. `"2:1"`) that carry
  no better identity. A named id whose name matches nothing resolves
  to the explicit default with a warning — never to whatever device
  occupies the stale index after hot-unplug index reuse. Ambiguous
  duplicate names go to default rather than guessing. Every path logs
  the configured id, strategy, candidate count, selected device, host
  API, runtime index, channels and rate.
- `matches_stable_id()`: index-free identity check (hostapi +
  normalized name); nameless legacy ids fall back to hostapi:index.
- `same_physical_device()`: now uses `_names_match` (truncation/
  marker tolerant) instead of strict normalized equality.

`microphone_controller.py` (Task 1 changes preserved — verified intact):

- `_record_active_device()`: the "strict" test now uses
  `matches_stable_id()` instead of `matches_device_id()` + name
  equality. A device resolved by name at a new index is correctly
  recognized as the configured microphone: the persisted id
  self-heals to the fresh index and no false fallback is reported.
  Genuinely different fallback devices still set `_fallback_notice`,
  and pinned preferences are never overwritten by a fallback.
- `_id_name_matches()`: uses `AudioDeviceManager._names_match` for the
  same tolerant comparison.

### Environment fix (not a code change)

Task 2's staged evaluation models (`sherpa-onnx-zipformer-en-2023-06-26`,
`sherpa-onnx-moonshine-tiny-en-int8`, `sherpa-onnx-whisper-base.en`)
had been left inside `%LocalAppData%\VoiceOS\models\stt\`. Because
`create_recognizer()` auto-discovers the highest-file-count model dir,
the Zipformer candidate (7 files) outscored production Moonshine Base
(5 files) and was silently loaded — causing 6 `test_live_reliability`
fixture failures (`OPEN CROON`, `WHICH TO CRONE`, `SAID` on silence).
The three evaluation dirs were moved to
`%LocalAppData%\VoiceOS\eval_models\stt\` (preserved for future
evaluation, outside the auto-discovery path). `test_live_reliability`
is back to its baseline: only the known `hey_jarvis.wav` failure.

### Tests executed

| Batch | Result |
|---|---|
| `tests/test_dynamic_mic.py` (incl. 7 new Task 3 cases) | 27/27 passed |
| `test_mic_pipeline`, `test_voice`, `test_vad_recovery`, `test_standby_watchdog`, wake/session/rescore batch (16 files) | 180/180 passed |
| `test_voice_reliability`, `test_live_reliability`, `test_live_fixes`, `test_application`, latency/ack/VAD/corpus batch (10 files) | 245 passed + `test_live_reliability` re-run: 20/21 (only known `hey_jarvis.wav`) |

New tests cover: index-reshuffle non-fallback + persisted-id
self-heal, duplicate-name ambiguity → explicit default (no index
guess), duplicate names across different host APIs still resolving,
truncated/® name matching, unmatched-name id → explicit default (no
index bind), nameless legacy id → hostapi:index still works,
index-reuse-by-different-mic at controller level (fallback reported,
pinned preference preserved), `matches_stable_id` across reshuffles,
missing preferred device → default with fallback reported and pinned
preference preserved.

Post-review hardening regression run: `test_dynamic_mic.py` +
`test_mic_pipeline.py` + `test_voice.py` = **85/85 passed**; broader
wake/recovery/session batch (14 files) = **152/152 passed**.

### Physical / live validation

- Real PortAudio enumeration on this machine confirms the identity
  model: the Intel array appears under 4 host APIs with differently
  truncated names (MME `Microphone Array (Intel� Smart `, DirectSound/
  WASAPI `...for Digital Microphones)`, WDM-KS `Microphone Array 1/2/3`).
- Stale-index ids (`<hapi>:999:<name>`) resolve to the correct current
  device for **every** host-API variant; exact ids and
  `matches_stable_id` behave as designed.
- Interactive physical validation (actual spoken wake cycle on the
  Intel mic after restart/replug) remains PENDING — requires the user
  at the machine.

### Remaining limitations

- Identity is hostapi + normalized name, not a Windows Core Audio
  endpoint id; two physically distinct mics with identical names on
  the same host API are treated as ambiguous and fall back to the
  explicit default rather than guessing.
- Nameless legacy ids (`"hostapi:index"` with no name portion) can
  still bind a different physical device after hot-unplug index
  reuse — unavoidable without a stored name; logged as a warning.
  All ids written by the current version carry a name.
- Same-hostapi duplicate names are indistinguishable without endpoint
  ids; documented above.

## Task 4 — Voice session lifecycle

### State machine (audited, unchanged)

`STANDBY →(wake) ACTIVATED →(ack TTS finish / silent ack) LISTENING
→(utterance) PROCESSING →(dispatch) EXECUTING →(response TTS finish)
LISTENING … →("Goodbye Jarvis"/timeout) _ending →(farewell TTS drain)
STANDBY`. `VoiceSessionController` is Qt-free; STT/TTS callbacks arrive
on worker threads, `handle_wake_word`/`tick`/`shutdown` on the GUI
thread. Wake path: detection worker → `suspend_detection()` → Qt
signal → `handle_wake_word` → `session.activate()` → mic capture
resume+flush → `stt.start()` → ack `_speak`. TTS gate pauses capture
for the whole synthesize+playback window.

### Defects found and fixed (3, all narrow)

1. **STT zombie worker / stale transcript into a new session**
   (`speech_recognition_service.py`). `stop()` used one shared
   `_stop` Event cleared by the next `start()`; `stop()` joins only
   2 s, so a worker inside a slow decode survived and was resurrected
   — two workers then consumed the same frame queue, and an in-flight
   decode could emit a stale transcript that the new session would
   dispatch as a command. Fix: each `start()` now creates a private
   `threading.Event` (`_run_stop`) plus a `_run_generation` counter;
   the worker loop and all transcript/no-speech/error emissions are
   guarded by `_emit_allowed(generation, stop_event)`, so a retired
   run drops its result instead of emitting it. Direct/untracked
   calls (`generation=None`, used by unit tests) are unaffected.

2. **Stale command dispatch after session end**
   (`voice_session_controller.py`). `_handle_transcript` guards only
   at entry; a transcript mid-route on the STT worker could reach
   `registry.dispatch()` after `tick()`/goodbye/shutdown ended the
   session — executing a command post-mortem and driving the state
   machine back to EXECUTING from STANDBY. Fix: `_dispatch_and_respond`
   re-checks `session.is_active and not _ending` immediately before
   dispatch and records a `session_ended` outcome instead.

3. **Session end finalized on first TTS finish, not last queued**
   (`voice_session_controller.py`). `_handle_tts_finished` called
   `_finalize_end()` on ANY finish while `_ending` — a farewell queued
   behind a still-playing response restored standby (and wake-word
   detection) mid-speech. Conversely, if the farewell `speak()` was
   dropped (queue full) or the worker wasn't running, no finish ever
   arrived and the session hung in `_ending` forever (wake ignored,
   timeout could not re-fire). Fix: finalize only when `_tts_meta` is
   drained; new `_maybe_finalize_end()` runs after the end-path
   `_speak` calls to cover the no-playback path; non-ending finishes
   still return to LISTENING as before.

### Audited and found correct (no change)

- Duplicate wake suppression: `handle_wake_word` lock +
  `session.is_active/_ending` guard + Task 1 stale-service guard +
  KWS cooldown.
- Duplicate transcripts: `_DUPE_WINDOW` identical-text suppression.
- Timeout: 60 s default from config, `touch()` on transcript/no-speech
  (excludes `no_audio_frames`/`capture_failed` so pipeline faults
  cannot reset it); `tick()` fires once via `_ending` guard.
- TTS/mic coordination: capture paused for entire playback; hangover
  and starvation checks already skip while paused.
- Goodbye: exit phrase + `END_SESSION` intent both route to
  `_end_with_goodbye`; STT stopped at `_begin_end`; detection resumes
  only in `_finalize_end`.
- Sensitive commands: `_pending_confirm` yes/no flow unchanged.
- Session shutdown: `shutdown()` stops STT/TTS workers, deactivates,
  resumes detection, returns to STANDBY.
- Device switch mid-session: capture restart keeps the shared frame
  queue; STT worker is not restarted (verified in capture service).

### Files modified

- `src/voice/speech_recognition_service.py` — per-run stop event,
  generation counter, `_emit_allowed` guards.
- `src/voice/voice_session_controller.py` — stale-dispatch guard,
  TTS-drain finalize, `_maybe_finalize_end`.
- `tests/test_task4.py` — corrected `test_goodbye_ends_session` FIFO
  ordering (ack must finish before farewell, matching real playback);
  new `TestLifecycleHardening` (6 tests).

### Tests executed

| Batch | Result |
|---|---|
| `tests/test_task4.py` (incl. 6 new lifecycle tests) | 46/46 passed |
| Lifecycle/session batch: task5, task6, voice_reliability, wake_cycles, wake_word_after_session, post_standby, quick_ack, v1075, v10751, vad_recovery, vad_segmentation, live_reliability, live_fixes, v108_task4, conversation_latency, voice, standby_watchdog (17 files) | 387 passed, 1 failed — known `hey_jarvis.wav` STT fixture only |

New tests cover: end-drain across multiple queued utterances,
dropped-farewell session end, stale dispatch dropped post-end,
second wake starting a fresh session, STT restart never resurrecting
the old worker, stale-run `_finalize` emitting nothing.

### Physical validation — PENDING

Manual plan for the laptop: start VoiceOS → STANDBY → "Hello Jarvis"
→ ack → "Open Chrome" (executes once) → Jarvis responds → second
command same session → "Goodbye Jarvis" → STANDBY → repeat "Hello
Jarvis" cycle several times; also verify the 60 s timeout path.
Requires the user at the machine; not yet performed.

### Remaining limitations

- A TTS worker that dies mid-queue would orphan `_tts_meta` entries;
  `_speak_one` catches all playback errors and always emits finish, so
  this is theoretical only.
- The dispatch guard narrows but cannot eliminate the
  transcript-vs-end race to zero (the check is immediately before
  dispatch, not inside the handler); handlers themselves are fast
  OS calls.
- Timeout is inactivity-based; long TTS playback does not extend it
  (unchanged, per instructions).

## Task 5 — Audio recovery & watchdog

### Audit result

The existing watchdog (`MicrophoneController._check_device` →
`_check_pipeline_liveness` → `_handle_callback_stall` /
`_check_detection_liveness` / `_restart_detection`) already provides
bounded stall retries (`_STALL_MAX_RETRIES` per 120 s window →
exponential backoff to 300 s), detection-worker liveness with its own
bounded budget, callback-stall vs intentional-pause distinction
(paused streams still receive callbacks), post-resume verification,
and Task 1/3/4 synchronization (`_restart_lock`, detach-first suspend/
disable, service-instance wake guards, Task 3 device identity).

Six demonstrable defects were reproduced with the FakeSD harness
before any change, plus two lifecycle races found during the audit.

### Root causes and fixes

**`src/controllers/microphone_controller.py`**

1. `enable()` during `STATUS_IDLE` (active voice session) installed a
   second `WakeWordDetectionService` consuming the shared frame queue
   concurrently with the STT service. Fix: `enable()` is now a no-op
   for LISTENING *and* IDLE.
2. `enable()` from `STATUS_ERROR` overwrote `_detection` without
   stopping the surviving worker → orphaned consumer draining the
   queue. Fix: detach under `_restart_lock`, stop outside the lock
   (same protocol as `suspend_detection`/`disable`).
3. A failed `_restart_detection` left `_detection = None` forever;
   `_check_detection_liveness` early-returned on None, so standby
   stayed LISTENING-but-deaf permanently. Fix: persistent-None
   detection (`_detection_none_since`, grace `_DETECTION_MISSING_GRACE`
   = 8 s > worst-case suspend/disable detach window ~4.5 s) routes
   through the bounded `_restart_detection` budget.
4. `STATUS_ERROR` + dead stream never recovered: `stream_dead`
   required `should_run` (LISTENING/IDLE only), so a callback error
   followed by silent stream death left the controller in ERROR
   indefinitely. Fix: `stream_dead` also covers ERROR; recovery is
   throttled by the existing 5 s ERROR restart gate. The
   no-input-device branch no longer re-emits ERROR every tick.
5. `_restart_for_selection` flushed the queue but did not reset the
   KWS decoder stream or notify session-scoped consumers — pre-restart
   audio/state could leak into the new stream's first recognition
   cycle. Fix: `engine.reset()` (engine-lock serialized) + invoke the
   existing `_on_system_resume` listener (STT VAD/utterance reset),
   same rule as the OS-resume and standby-handover paths.
6. `disable()` ran `capture.stop()`/status transition outside
   `_restart_lock`, so an in-flight `_restart_for_selection` could
   reopen the stream after "microphone off"; `_restart_for_selection`
   also never checked DISABLED. Fix: restart aborts when DISABLED;
   disable's capture-stop + terminal status are serialized under
   `_restart_lock`.
7. `suspend_detection()`/`resume_detection()` could overwrite a
   completed DISABLED state (status flap to IDLE/LISTENING with an
   orphaned detector). Fix: suspend's terminal transition re-checks
   DISABLED under the lock; resume_detection's whole handover is now
   serialized under `_restart_lock` with an IDLE+running re-check.

**`src/voice/audio_capture_service.py`**

8. No stream-generation guard: a PortAudio callback still in flight
   from a replaced/closed stream could enqueue frames into the shared
   queue after `stop()`'s drain or after a replacement stream started.
   Fix: `_stream_generation` incremented on every `start()` (before
   stale-stream release) and `stop()`; each stream gets a guarded
   callback closure that drops stale-generation invocations.
   `_audio_callback` itself is unchanged.

### Invariants verified

- At most one active stream: `capture.start()` no-ops while running;
  stale stream released before open; concurrent restarts serialize on
  `_restart_lock` (test: two racing restarts → one active stream).
- At most one recovery owner: all restart paths hold `_restart_lock`;
  monitor is single-threaded.
- Bounded retries: stall budget (3/120 s → backoff ≤300 s), detect
  budget (3/120 s → 60 s backoff), ERROR restart throttle (5 s),
  missing-detector grace (8 s) feeding the same detect budget.
- Disabled/shutting-down mic cannot be restarted: DISABLED guard in
  `_restart_for_selection`, serialized disable transition, suspend/
  resume status re-checks.
- No pre-restart audio leaks: queue flush + stream-generation drop +
  KWS engine reset + STT VAD/utterance reset on every restart.
- TTS pause ≠ failure: paused streams keep receiving callbacks;
  `callback_stall_seconds` unaffected (existing + new tests).
- Task 3 identity/fallback untouched; no PCM persisted.

### Files modified

- `src/controllers/microphone_controller.py` — enable() idempotence +
  orphan-safe detection replacement, missing-detector recovery,
  ERROR-state dead-stream recovery, engine/STT reset on restart,
  DISABLED guards, serialized disable/suspend/resume transitions.
- `src/voice/audio_capture_service.py` — `_stream_generation` guard.
- `tests/test_audio_recovery.py` — new, 17 deterministic tests.
- `tests/test_wake_diagnostics_v106.py` — bare-controller fixture
  gained `_restart_lock` (harness fix, assertions unchanged).

### Tests executed

| Batch | Result |
|---|---|
| `tests/test_audio_recovery.py` (new) | 17/17 passed |
| standby_watchdog + dynamic_mic + mic_pipeline + voice + task4 | 151/151 passed |
| v1075_reliability + v10751 + v108_async_rescore + post_standby + voice_reliability + vad_recovery + live_fixes + quick_ack + intel_array + v1071_hotfix | 147/147 passed |
| wake_cycles + wake_word_after_session + wake_word_latency + wake_diagnostics_v106 + v108_task1 + conversation_latency + live_reliability | 57 passed, 1 failed — known `hey_jarvis.wav` STT fixture only |
| vad_segmentation + voice_command_accuracy + speech_accuracy + v92 + task5 + task6 | 222/222 passed |

Stress observation: repeated suspend/resume cycles, racing restarts,
and disable-during-restart tests showed **no duplicate streams, no
stale-callback leaks, and no orphan recovery workers**.

### Physical validation — PENDING

Requires the Windows laptop: verify standby survives long idle periods
(no stall restart loops), sleep/resume recovers within the verify
window, mic unplug/replug falls back per Task 3 rules, and TTS
playback never triggers a stall restart. Not yet performed.

### Remaining limitations

- `STATUS_ERROR` with a *live* stream (transient callback exception)
  is not auto-cleared; it self-heals on the next wake/session cycle.
  Deliberate — auto-clearing could mask recurring callback faults.
- ~~`resume_detection` could overwrite `_detection` without stopping a
  hypothetical survivor~~ — **resolved in Task 8** (see addendum
  below): detectors are marked retiring at detach time and a new
  consumer is installed only when no retired worker may still be
  alive.
- `_restart_detection` joins the old worker inside `_restart_lock`
  (bounded ~4.5 s worst case); kept as-is to preserve Task 1's
  detach-first protocol elsewhere.
- Hot-plug recovery for a *permanently absent* device remains a fixed
  ~5 s retry poll (pre-existing, intentional for replug detection).

---

## Addendum — 2026-11-20 runtime anomaly review

Source: `%LocalAppData%\VoiceOS\logs\voiceos.log` (Windows run,
`pythonw.exe` from the VoiceOS venv).

### Evidence

- ~18:34 activity window 4 closed with `duration_seconds=441.2` but
  only `audio_seconds=12` (`frames=153`, `samples=192000` @16 kHz) —
  the wall-clock span and captured-audio span disagree by ~7 min.
- 18:41:31 a rescore anchor reports `elapsed_ms=4.298e+05`; the job
  completes at ~431,500 ms total with only 2 of 10 candidates
  evaluated and `aborted=True`.
- 18:41:32 PortAudio `input overflow` logged.
- Subsequent pipeline stats: `rms=0`, `peak=0`, all channel RMS = 0,
  while callback/frame counters keep increasing.
- ~19:43 new activity windows open with normal sub-second rescore
  anchors; audio activity resumes.

### Analysis (no single cause proven; ranked by fit)

1. **System suspend/resume or process suspension — most consistent.**
   A ~7 min gap in which `time.monotonic()` kept advancing (the rescore
   job's `elapsed_ms` and the window's `duration_seconds` both measured
   it) while capture delivered exact zeros and callback counters still
   incremented matches a suspended-then-resumed process: PortAudio
   overflows on resume, the driver returns silence until the stream
   re-settles, and the rescore job's cooperative abort predicate then
   fires on the next chunk boundary (deadline/budget exceeded →
   `aborted=True` after 2 candidates). On Windows, `time.monotonic()`
   (QPC) can include suspend time, which is why a "4.3e5 ms" anchor
   elapsed is wall-clock, not CPU time.
2. **Audio-driver interruption — plausible contributor.** The
   PortAudio overflow + all-zero channels with live callbacks is the
   signature of a stream that kept calling back but delivered silence
   (device lost/reset). The callback-stall watchdog cannot see this:
   it only fires when callbacks *stop* (`callback_stall_seconds`),
   and zero-valued callbacks still increment the counters. Recovery
   at ~19:43 was therefore most likely OS/driver-level, not
   watchdog-triggered.
3. **Stalled KWS inference — unlikely as primary cause.** A hang
   inside `decode_stream` would hold `engine._lock` and stall live
   `process()` too; pipeline stats kept updating and live KWS frames
   kept being consumed. A *slow* rescore cannot starve live detection
   (per-100 ms-chunk lock release, V10.8) and cannot grow the bounded
   queue (`queue_maxsize=4`, `put_nowait` + reject).
4. **Rescore worker lifecycle — ruled out as the stall source.** The
   worker was alive and completed the job; `aborted=True` shows the
   deadline/budget predicate worked. The `duration_seconds=441.2` vs
   `audio_seconds=12` mismatch is activity-window bookkeeping spanning
   the silent gap — open/close monotonic timestamps are wall-clock,
   while `audio_seconds` counts only queued samples. Not a bug per se,
   but the metric is misleading across suspensions; recorded here as a
   known diagnostic limitation.

### Rescore timeout honesty

`should_abort` (job TTL `job_ttl_seconds=15`, budget
`job_budget_seconds=6.0`) is **cooperative**: it is checked between
candidates (`kws_rescore.run_candidates`) and between 100 ms decode
chunks inside `SherpaOnnxKeywordSpotter.rescore`. A native
`decode_stream` call that never returns **cannot** be interrupted or
killed — Python threads cannot be terminated. What is actually
guaranteed:

- live `engine.process()` interleaves between chunks (engine lock is
  per-chunk, not per-sweep);
- the rescore queue is bounded (4) and rejects rather than blocks;
- `stop()` joins the rescore worker with `timeout=2.0` and the thread
  is a daemon — `stop()` always returns; a still-stuck worker finishes
  whenever the native call returns and its result is discarded.

Residual risk requiring physical validation: a *permanent* native
hang inside `decode_stream` while holding `engine._lock` would stall
live `process()` as well; there is no in-process remedy short of
restarting the engine/service.

### Tests added (deterministic)

`tests/test_v108_async_rescore.py` (+3, file now 8 tests):

- `test_expired_queued_job_aborts_before_inference` — a job past its
  TTL aborts before any `engine.rescore()` call.
- `test_zero_valued_callbacks_do_not_block_recovery` — 300 exact-zero
  frames then speech: pipeline keeps consuming, window closes,
  rescore detects.
- `test_stop_returns_while_rescore_inflight` — `stop()` returns within
  the bounded join while a gated rescore is still blocked.

### Physical validation still required

- Reproduce/observe a real suspend-resume cycle and confirm the mic
  stream recovers (or document that it needs replug); the zero-RMS
  period self-recovered in this log but the trigger is unproven.
- Decide whether a "digital-silence" (all-zero callbacks) detector is
  worth adding to the mic watchdog — currently impossible to
  distinguish from a genuinely quiet room without false-positive risk.
- Confirm no real `shutdown.exe`/`powrprof` call occurred (log shows
  none; all power intents remained confirmation-gated).

---

## Addendum — Task 8: wake-word detector ownership race

### Confirmed root cause

`MicrophoneController` had three detector-replacement paths —
`enable()`, `resume_detection()` and `_restart_detection()` — that all
called `WakeWordDetectionService.stop()` and then **unconditionally**
installed a new service. `stop()` joins the worker for a bounded
~4.5 s (detection + rescore threads); if the worker outlived that
join — e.g. stuck inside a native `decode_stream` call, the same class
of stall seen in the 2026-11-20 log — the old thread kept draining
the shared capture frame queue while the replacement consumed it
too. The stale-service identity guard in `_handle_wake_word` prevents
duplicate *wake events*, but it does nothing about the old worker
stealing frames from the new consumer. `suspend_detection()` /
`disable()` had the same detach-then-stop-outside-lock shape, so a
survivor was invisible to any concurrent installer during the join
window.

The race is timing-dependent and was not observed firing in the
existing suite; it is closed structurally below rather than by adding
delays.

### Solution: retiring-worker ownership slot

`src/controllers/microphone_controller.py`:

- `_retiring_detectors` — list of services marked at **detach time**,
  under `_restart_lock`, *before* `stop()` is invoked. Marking before
  the join closes the window where a survivor was untracked.
- `_mark_retiring(service)` — records a still-running worker (dead
  threads cannot consume frames and are never tracked).
- `_detector_slot_available()` — purges entries whose `is_running`
  went false, then reports whether any retired worker may still be
  alive. Always called with `_restart_lock` held.
- `_stop_detector(service)` — uniform bounded `stop()` + logging.

Every install site now checks the slot:

- `enable()` — whole detector swap serialized under `_restart_lock`
  (with a status re-check so a concurrent enable cannot double-
  install); on a blocked slot it leaves `_detection=None`, stays
  LISTENING, and the existing missing-detector watchdog installs a
  replacement once the survivor exits.
- `resume_detection()` — the reuse branch still adopts a live
  detector; the create branch checks the slot and defers identically.
- `_restart_detection()` — defers *before* consuming the bounded
  retry budget when a retiree lives (an ownership wait is not a
  failed restart); after stopping the old worker it re-checks the
  slot: on a survivor it restores `self._detection = old` — ownership
  is never overwritten — and later ticks retry.
- `suspend_detection()` / `disable()` — mark the worker retiring at
  detach time (before the outside-lock stop/join) so resume/enable
  can never install while the suspended worker may still be alive.

**Invariant now enforced:** at most one live detector worker consumes
the shared frame queue. A stop/join timeout never produces a second
consumer; recovery is fail-closed through the existing bounded
watchdog (`_DETECTION_MISSING_GRACE` → `_restart_detection` budget +
backoff).

### Files modified

- `src/controllers/microphone_controller.py` — `_retiring_detectors`,
  `_mark_retiring`, `_detector_slot_available`, `_stop_detector`;
  slot checks in `enable()`/`resume_detection()`/`_restart_detection()`;
  retiring marks in `suspend_detection()`/`disable()`;
  `watchdog_stats()["retiring_detectors"]` counter.
- `tests/test_detector_ownership.py` — new, 11 deterministic tests.
- `tests/test_post_standby_recognition.py` — stale `__new__` fixture
  gained `_current_interaction`/`_current_command`/`_command_open`
  (harness fix, assertions unchanged; was failing before this task).

### Tests added (all deterministic — event-driven fake detector)

`tests/test_detector_ownership.py` (11 tests):

- repeated `resume_detection()` creates no duplicate detector;
- two barrier-synchronized concurrent resumes install exactly one
  worker;
- a worker surviving `stop()` keeps ownership — no replacement until
  it terminates, then exactly one replacement installs;
- deferrals do not consume the detect-restart budget;
- resume blocked while a suspended worker survives → LISTENING with
  `_detection=None`, watchdog missing-detector path also defers, then
  installs after the survivor exits;
- watchdog ticks during the survivor window never add a consumer;
- `disable()` racing `resume_detection()` ends DISABLED with zero
  live consumers, and a later resume is a no-op;
- stale detector emissions are dropped while the current detector's
  still work;
- shutdown leaves no orphan consumers;
- normal suspend→resume handover and enable-after-error still work.

### Test results

| Command | Result |
|---|---|
| `pytest tests/test_detector_ownership.py` | 11/11 passed |
| `pytest test_audio_recovery test_dynamic_mic test_mic_pipeline test_standby_watchdog test_wake_cycles test_wake_word_after_session test_post_standby_recognition test_v108_async_rescore test_voice_reliability test_task4 test_detector_ownership` | **197/197 passed** (~19.5 min) |

Known baseline failure unchanged: `test_live_reliability.py::
test_fixture[hey_jarvis.wav]` (STT returns `Hage-arvis`) — Task 2
scope, not affected by this change.

### Remaining limitations

- A *permanently* hung native `decode_stream` (worker never exits)
  now blocks detector replacement entirely — fail-closed by design:
  installing a second consumer would be worse. Recovery requires the
  process/engine to restart; there is no in-process thread kill.
- While a survivor briefly outlives its join it still consumes frames
  (bounded by `_stop` observation at the next loop iteration, ~100 ms
  cadence) — unavoidable without thread termination; emissions are
  still guarded by the stale-service identity check.
- `resume_detection`'s reuse branch can adopt a live detector whose
  stop was requested but not yet observed dead — it remains the
  single owner and exits normally; no second consumer is created.

### Physical validation — PENDING

Requires the Windows laptop: long standby idle (no restart loops),
sleep/resume recovery, mic replug fallback, TTS-pause behavior, and a
forced detector stall (if reproducible) to confirm fail-closed
deferral rather than duplicate consumers. Not yet performed.