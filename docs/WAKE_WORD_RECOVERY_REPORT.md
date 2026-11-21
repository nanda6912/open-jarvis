# VoiceOS — Wake Word Recovery Report ("Hello Jarvis")

Date: 2026-02-24 (continuation of the November 18 log investigation)
Scope: `src/voice/wake_word_detection_service.py`,
`src/controllers/microphone_controller.py`, `src/voice/kws_rescore.py`,
tests under `tests/test_v108_*` and `tests/test_v1075_reliability.py`.

---

## 1. Root cause identified

**Primary (verified in code + November 18 runtime logs):**
activity-window rescoring ran **synchronously on the wake-word
detection worker thread**. Every activity window that the live KWS
stream missed triggered `_rescore_activity_audio()`, which replayed up
to 3 candidates × 10 fresh-stream anchors through `engine.rescore()`.
Each anchor costs ~30–800 ms of ONNX inference (measured mean ≈ 400 ms
per candidate call, p95 ≈ 750 ms; multi-second totals for full
windows). During that time the detection worker performed **zero**
live inference: the capture queue backlogged (November 18 logs show
`queue_depth` up to 37 frames ≈ 3.7 s of stale audio) and every
subsequent live frame was inferred late or dropped from the
position-sensitive streaming window. Sixteen activity windows closed
with `detections=0` — consistent with a detector that spends most of
its wall-clock time rescoring old audio instead of listening to new
audio.

The async scaffolding for this fix already existed but was **dead
code**: `_rescore_queue`, `_rescore_thread`, `_rescore_results`,
`_rescore_pending_ids`, `kws_rescore.plan_candidates()` and
`kws_rescore.run_candidates()` were present yet never started or used
by the production path.

**Secondary (found during validation, fixed):**
`MicrophoneController._check_detection_liveness()` read
`self._detection` and decided to restart without holding
`_restart_lock`. `_restart_detection()` publishes `_detection = None`
while rebuilding the worker, so the 50 ms monitor thread and any
external caller could observe the gap — producing flaky
`test_dead_worker_is_recreated` failures (`new is None`) and, in
principle, redundant restarts. Fixed by holding `_restart_lock`
across the whole check, the same pattern already used by
`_check_capture_liveness` (see comment at
`microphone_controller.py:~1035`).

## 2. Exact files and functions involved

| File | Function / area | Role |
|---|---|---|
| `src/voice/wake_word_detection_service.py` | `_close_activity_window()` (~L496) | Now prepares a rescore job and enqueues it instead of running the sweep inline. Emits `KWS_ACTIVITY_CLOSE` promptly with `rescore_scheduled=True` / `detection_method='rescore_pending'`. |
| same | `_prepare_rescore_job()` (~L708) | Snapshots window audio into candidates via `kws_rescore.plan_candidates()` — no inference on the detection thread. |
| same | `_enqueue_rescore_job()` (~L765) | Bounded `put_nowait`; on `queue.Full` marks the record `rescore_attempted=False` and logs `KWS_RESCORE_SKIPPED reason='queue_full'`. Never blocks. |
| same | `_rescore_worker()` (~L808) | Dedicated `kws-rescore` thread; drains `_rescore_queue`, executes jobs, pushes `(job, report)` to `_rescore_results`. |
| same | `_rescore_should_abort()` (~L846) | Abort predicate: stop events, job TTL (`job_ttl_seconds=15`), per-job budget (`job_budget_seconds=6`). Checked between candidates and between 100 ms decode chunks inside `engine.rescore()`. |
| same | `_execute_rescore_job()` (~L861) | Runs `kws_rescore.run_candidates()` on the rescore thread; maps engine trace events onto the production `KWS_RESCORE_START/ANCHOR/COMPLETE/FAILED/SKIPPED` log names. |
| same | `_drain_rescore_results()` (~L1002) | Called at the top of every detection-worker loop iteration; applies reports to window records, logs `KWS_RESCORE_RESULT`, and emits hits via the normal `_emit_detection()` path — `on_detected` stays on the detection thread, preserving callback ordering. |
| same | `_apply_rescore_report()` (~L961) | Folds the async report into the closed window record (`rescored`, `rescore_*` fields, `detection_method='rescore'`). |
| same | `_rescore_activity_audio()` (~L1038) | Synchronous helper retained for diagnostics/tests; shares `plan_candidates`/`run_candidates` with production. |
| same | `start()` / `stop()` (~L1387/L1425) | `start()` launches the rescore thread; `stop()` sets both stop events, closes windows with `allow_rescore=False`, and joins both threads. |
| same | `_emit_detection()` | Guard added: a rescored hit cannot attach to a *newly opened* activity window belonging to a later utterance. |
| `src/controllers/microphone_controller.py` | `_check_detection_liveness()` (~L1153) | Whole check now under `_restart_lock` (RLock); fixes the `_detection=None` observation race. |
| `src/voice/kws_rescore.py` | `plan_candidates`, `run_candidates`, `RescorePolicy` | Pre-existing shared orchestration (unchanged); now actually used by production. |
| `tests/test_v108_async_rescore.py` | new | 5 regression tests for the async contract (see §7). |
| `tests/test_v108_task1_diagnostics.py` | 2 tests updated | Wait for async rescore completion; assert on new `KWS_RESCORE_RESULT` event for final state (`KWS_ACTIVITY_CLOSE` now truthfully reports the *pending* state). |

No changes to: keyword file, model, thresholds, capture/channel
selection, HUD, TTS, STT, command handling, or standby/session
lifecycle.

## 3. Before-and-after behavior

**Before**

- Window closes → `_rescore_activity_audio()` runs on the detection
  worker → up to ~7 s (3 candidates × 10 anchors × ~0.4 s) of zero live
  inference per missed window → capture queue depth grows (observed 37)
  → live frames inferred late/stale → position-sensitive KWS misses.
- Rescore hits emitted inline during window close.

**After**

- Window closes → job enqueued in ≤ µs → detection worker immediately
  resumes live inference → dedicated `kws-rescore` thread performs the
  anchor sweep → completed hit is drained on the detection worker and
  emitted through the identical `_emit_detection()` → controller →
  VoiceOS state-transition path as a live hit.
- Bounded queue (maxsize 4), per-job TTL 15 s, per-job budget 6 s,
  abort between anchors; stale results after `stop()` are discarded.
- Live inference is never blocked by rescore; rescore can never starve
  the primary detector (it has its own thread and its own engine
  calls are serialized by the engine's existing lock only against
  other engine calls — live `process()` and rescore `rescore()` share
  the engine lock, so they interleave safely rather than corrupt
  stream state; `rescore()` uses fresh streams and does not touch the
  live stream — verified by
  `test_rescore_does_not_touch_live_stream`).

## 4. Audio capture and replay results (Tests A/B/C)

Diagnostic: `scripts/kws_capture_replay_diagnostic.py`
(Test A live service, Test B fresh-engine replay of the exact
KWS-input frames, Test C detection-service replay + direct
`engine.rescore()`).

### 4a. November 18 production logs (evidence under investigation)

- Model/keywords load OK; sha256 matches; 4-ch Intel SST array @16 kHz,
  channel 0 selected; `frames_dropped=0`; peaks ≈ 0.99.
- 16 activity windows, `detections=0`, all rescore anchors `hit=False`,
  no rescore exceptions; queue depth up to 37; rescore operations of
  several seconds.

### 4b. Controlled captures on this machine (2026-02-24)

| Run | Device | Result |
|---|---|---|
| Diagnostic #1 (30 s) | Intel SST array, ch 4→0 | Speech-energy bursts present (rms ≤ 0.10, peak 0.37), 4 activity windows, **0 detections live / replay / rescore**. Saved: `reports/diag_kws_capture*.wav` family. |
| Diagnostic #2 (30 s, `--save`) | same | Same pattern; saved `reports/diag_kws_live_v108.wav`. STT (production moonshine model) on 3 s slices returned only garbled fragments (`"So if you can zoom"`, `"Sift your pencil"`, `"Thank you and God's good…"`) — **the captured audio does not contain an intelligible "Hello Jarvis"**. |
| Per-channel probe (20 s, `scripts/_perchannel_probe.py`) | same | ch0 ≡ ch1 (duplicated), ch2/ch3 digital silence. Overall rms 0.0018 — essentially no speech captured; consistent with nobody speaking during that window. |

**Interpretation (verified vs assumed):**

- *Verified:* the capture pipeline delivers real audio to KWS
  (speech-band energy bursts, correct rate/channels/dtype, no drops,
  no clipping). Replay of the exact captured frames through a fresh
  engine and through the full detection service reproduces the live
  result exactly — capture/preprocess is **not** the failure point.
- *Verified:* on the audio actually captured, neither live KWS, fresh
  replay, service rescore, nor a direct 10-anchor `engine.rescore()`
  (7.5–9.9 s) finds "Hello Jarvis" — because the phrase is not
  intelligibly present in that audio (STT agrees).
- *Assumed/pending:* whether a clearly spoken "Hello Jarvis" on this
  mic is detected live **after** the async fix — requires a capture
  where the user actually speaks the phrase (see §8).

## 5. Positive/negative benchmark (existing model, unchanged)

`scripts/benchmark_kws.py --manifest tests/fixtures/kws_benchmark_manifest.json`
(7 cases: 3 positive, 4 negative). Baseline vs post-change runs are
**identical** — the fix changes scheduling, not detection semantics.

| Mode | Recall (pos) | False positives | Rescore elapsed |
|---|---|---|---|
| live | 2/3 = 0.667 | 0/4 neg | — |
| fresh_stream | 1/3 = 0.333 | 0/4 | — |
| rescore | 2/3 = 0.667 | 0/4 | mean 403 ms, p50 396 ms, p95 751 ms per case |

Machine-readable: `reports/kws_benchmark_baseline.json` (before) and
`reports/kws_benchmark_async.json` (after).

Caveat: 3 positive fixtures is a small set; recall 0.667 means even a
perfectly scheduled pipeline can miss difficult utterances. The
position-sensitivity demonstrated by `fresh_stream` (1/3) vs `live`
(2/3) is exactly why the rescore recovery path exists — and why it must
not starve the live path.

## 6. Queue and inference timing

- Before: rescore anchors executed inline; November 18 logs show
  multi-second rescore operations and `queue_depth` up to 37
  (~3.7 s stale audio at 100 ms/frame).
- After: rescore jobs run on `kws-rescore` thread; the detection
  worker's per-iteration cost is unchanged (one `engine.process()` per
  frame + non-blocking queue ops). `test_rescore_does_not_block_live_
  frame_processing` proves `engine.process()` and
  `frames_processed` keep advancing while a rescore job is gated
  mid-flight. Rescore job bounds: queue maxsize 4, TTL 15 s, budget
  6 s, abort between anchors — a stuck sweep can delay at most one
  job, never live inference.
- Engine serialization is preserved: `engine.rescore()` uses fresh
  streams and shares the engine lock with `process()`; live stream
  state is untouched (`test_rescore_does_not_touch_live_stream`).

## 7. Test results

```
pytest tests/test_v1075_reliability.py \
       tests/test_v10751_production_trace.py \
       tests/test_v108_task1_diagnostics.py \
       tests/test_v108_async_rescore.py \
       tests/test_v108_task2_kws_benchmark.py \
       tests/test_kws_parity_diagnostic_v10741.py \
       tests/test_wake_diagnostics_v106.py \
       tests/test_wake_cycles.py \
       tests/test_wake_word_after_session.py \
       tests/test_wake_word_latency.py \
       tests/test_v1071_hotfix.py \
       tests/test_dynamic_mic.py \
       tests/test_mic_pipeline.py \
       tests/test_intel_array_wake_detection.py
→ 154 passed in 125.8s   (0 failures)
```

New regression tests (`tests/test_v108_async_rescore.py`, all passing):

- `test_rescore_does_not_block_live_frame_processing` — live
  `engine.process()` continues while a rescore job is blocked
  mid-flight; hit still emitted after release via `on_detected`.
- `test_rescore_hit_emitted_only_after_window_close` — ordering:
  window record persisted first, detection callback after.
- `test_rescore_result_after_stop_is_discarded` — post-stop results
  never emit; no worker resurrection.
- `test_rescore_queue_full_rejects_without_blocking` — bounded queue
  rejects with `rescore_rejected` accounting.
- `test_stop_leaves_no_rescore_worker` — clean shutdown, empty queue.

Previously failing, now fixed:

- `TestWorkerRecovery::test_dead_worker_is_recreated` — was flaky
  (~1 in 5) due to the `_detection=None` race; 8/8 consecutive passes
  after the `_restart_lock` fix.

## 8. Physical microphone validation — PENDING

**Not yet demonstrated.** Status on 2026-02-24:

- VoiceOS-equivalent stack starts cleanly on the Intel SST array
  (device index 2, 16 kHz, 4 ch → selected ch 0), 0 dropped frames,
  0 clipping, queue depth 0 throughout 30 s captures.
- Two 30 s captures containing speech-band energy bursts produced
  **no detections** — but STT verification shows the captured audio
  does **not** contain an intelligible "Hello Jarvis" (garbled
  transcripts only), so these runs neither confirm nor refute the fix.
- A 20 s per-channel probe captured essentially silence (nobody
  speaking during that window).

**Required to close acceptance:** a capture where a person clearly
speaks "Hello Jarvis" 3–4 times into the laptop microphone, run via

```
py scripts/kws_capture_replay_diagnostic.py --device 2 --seconds 30 --save reports/diag.wav
```

with Test A (live) expected to show `detections: [...]` and
`KWS_DETECTION_EMITTED`/`via='live'` or `via='rescore'`, followed by
the VoiceOS standby→session transition. Same captured audio must then
replay-detect in Tests B/C. Also still to verify physically:
"Goodbye Jarvis" end-of-session, next wake session, suspend/resume,
and mic recovery — covered by unit tests but not yet re-run live.

## 9. Remaining limitations

- Model recall on the small fixture set is 0.667; difficult
  utterances (fast/quiet/distant) can still miss. No model change was
  made or recommended — evidence does not support replacement.
- Rescore hits are emitted asynchronously (bounded by job TTL/budget);
  a hit can arrive up to ~seconds after the window closes. This is
  inherent to not blocking live inference and matches the pre-existing
  design intent.
- The `_check_detection_liveness` fix removes the observed race but
  the monitor-thread/test interleaving is inherently timing-sensitive;
  the 154-test suite passed with no flakes in the final run.
- Channel selection picks the highest-EMA-energy channel; on the Intel
  array ch0≡ch1 (duplicated) and ch2/ch3 are silent, so selection is
  effectively moot on this device — but a future array with speech on
  a non-selected channel could still underperform; per-channel level
  telemetry (`channel_rms`/`channel_ema` in capture stats) is
  available for diagnosis.
- Whether the November 18 zero-detection session was *fully* explained
  by rescore starvation versus additional utterance-quality issues
  cannot be proven retroactively; the fix removes the verified
  starvation mechanism and the remaining question is empirical (live
  test in §8).
