# Windows System Control — Milestone 1 Report

Scope: Windows power management (sleep / shutdown / restart / cancel)
and desktop/window controls, implemented inside the existing VoiceOS
command pipeline. No second registry or confirmation mechanism was
introduced; all destructive operations reuse the existing
`_pending_confirm` yes/no lifecycle.

## New intents and example voice commands

| Intent | Example phrases |
|---|---|
| `SLEEP_COMPUTER` | "Sleep my computer.", "Put my PC to sleep.", "Put Windows to sleep.", "Put the laptop to sleep.", "Go to sleep." |
| `RESTART_COMPUTER` | "Restart my computer.", "Restart my PC.", "Reboot Windows.", "Reboot the system." |
| `SHUTDOWN_COMPUTER` | "Shut down my computer.", "Turn off my PC.", "Power off my computer.", "Turn my computer off.", "Shut down." |
| `CANCEL_PENDING_POWER_ACTION` | "Cancel shutdown.", "Cancel restart.", "Abort the scheduled power action." |

Narrow STT-mishearing aliases added after the 2026-11-20 log review
(each still requires the full command shape — an explicit computer
object or end-anchored position — so ordinary uses of the misheard
words can never become power commands):

- `SLEEP_COMPUTER`: "Sheep my computer." / bare "Sheep." → sleep
  (the observed production mishearing of "sleep").
- `SHUTDOWN_COMPUTER`: "Shed down computer.", "Sherdown my PC." →
  shutdown (variants of "shut down").
- `CANCEL_PENDING_POWER_ACTION`: "Cancel sherdown.", "Cancel shed
  down." → cancel.
- "For no" (observed after a restart ask) classifies `UNKNOWN` — it
  is neither a yes nor a no and can never confirm or cancel anything;
  the pending confirmation simply expires.
| `SHOW_DESKTOP` (extended) | "Show desktop.", "Get all windows out of the way." |
| `MINIMIZE_ACTIVE_WINDOW` | "Minimize active window." (plus "Minimize this window." via `MINIMIZE_WINDOW` with empty target) |
| `RESTORE_ACTIVE_WINDOW` | "Restore active window." |
| `CLOSE_ACTIVE_WINDOW` | "Close this window.", "Close the window", "Close the active window." |
| `SWITCH_NEXT_WINDOW` | "Switch to the next window.", "next window" |
| `SWITCH_PREVIOUS_WINDOW` | "Switch to the previous application.", "previous window" |
| `SWITCH_NEXT_VIRTUAL_DESKTOP` | "Move to the next desktop.", "next desktop", "switch to next virtual desktop" |
| `SWITCH_PREVIOUS_VIRTUAL_DESKTOP` | "Move to the previous desktop." |
| `SNAP_ACTIVE_WINDOW_LEFT` | "Snap this window left.", "snap left" |
| `SNAP_ACTIVE_WINDOW_RIGHT` | "Snap this window right." |

`MINIMIZE_WINDOW`/`RESTORE_WINDOW` with no target already operated on
the foreground window; pronoun-only residues ("the", "this") no longer
trigger the "which application?" clarification, so "Minimize this
window." now minimizes the active window directly. `SHOW_DESKTOP`,
focus/maximize/minimize and monitor movement were reused, not
duplicated.

Hibernate and sign-out remain `UNSUPPORTED_WINDOWS_ACTION` (honest
refusal, never executed).

## Files created

- `src/controllers/power_manager.py` — `PowerManager`: injectable
  `sleep_fn`/`schedule_fn`/`abort_fn`/`clock`; fixed-argument
  `shutdown.exe` (`/s` or `/r`, `/t 15`, `/c <fixed comment>`),
  `shutdown.exe /a` abort, `powrprof.SetSuspendState` sleep; pending
  state with lazy expiry.
- `src/controllers/desktop_control.py` — `DesktopController`:
  verified-foreground-window ops, z-order window switching,
  Win+Ctrl+Arrow virtual-desktop chords, Win+Arrow snap chords.
- `src/commands/system_control_handlers.py` — all new handlers plus
  `create_system_control_handlers(svc)`.
- `tests/test_windows_system_control.py` — 101 deterministic tests.
- `docs/WINDOWS_SYSTEM_CONTROL_REPORT.md` — this file.

## Files modified

- `src/commands/intent.py` — new intent rules (cancel rule placed
  before power rules so "cancel shutdown" is never a fresh request);
  `_window_target` filler-residue fix; extended `SHOW_DESKTOP`;
  2026-11-20 follow-up: `_SHUTDOWN_SPOKEN`/`_SLEEP_SPOKEN` narrow
  mishearing variants (`sherdown`, `shed down`, `sheep`) applied only
  inside the existing command shapes.
- `src/commands/app_handlers.py` — `_Svc` gained `power_ops` and
  `desktop` services; registry wires the new handlers.
- `src/controllers/windows_automation.py` — added
  `WindowManager.close_window(hwnd)` (single-hwnd WM_CLOSE).
- `src/voice/voice_session_controller.py` — `_speak`/`_speak_response`
  accept `on_finished`; `_handle_tts_finished` invokes the callback
  after the utterance finishes and speaks any returned follow-up;
  `confirmation_required` results may carry `decline_message`;
  2026-11-20 follow-up: `_route_command` consumes a
  `CANCEL_PENDING_POWER_ACTION` utterance while a power confirmation
  is pending (`_POWER_CONFIRM_INTENTS`), dropping the ask with its
  decline message instead of letting it expire silently.
- `src/main.py` — constructs `PowerManager()` and
  `DesktopController(windows=app_windows)` and injects them.
- `tests/test_p0_p1_system_ops.py` — power phrases re-mapped to the
  new real intents (previously asserted unsupported).
- `docs/ARCHITECTURE.md` — documented the new control path.

## Power management safety design

1. Sleep/restart/shutdown always return `confirmation_required` on
   first dispatch; a fresh "yes" re-dispatches the *same stored intent*
   with `_confirmed=True`.
2. The handler records `_armed_action` on first dispatch; the confirmed
   path refuses (`confirmation_mismatch`) unless it matches — a
   confirmation can never be transferred to a different action.
3. Confirmation freshness is bounded by the existing 30 s
   `_CONFIRM_WINDOW`; expired pendings fall through to normal routing
   and can never execute.
4. Restart/shutdown arm `shutdown.exe /s|/r /t 15` — a real OS-level
   15 s cancellation window. The `after_speech` callback (invoked only
   after the acknowledgement TTS finishes) calls `PowerManager.schedule`,
   so the countdown cannot start while Jarvis is still speaking. The
   TTS gate already pauses capture during playback and resumes it
   after, so "cancel shutdown" is heard during the countdown.
5. `PowerManager` records pending state only after `shutdown.exe`
   exits 0; pending records expire lazily at the deadline; `cancel()`
   runs `shutdown.exe /a` only while an unexpired action is pending
   and reports `nothing_pending`/`abort_failed` honestly.
6. Duplicate requests while a countdown is pending are refused with
   "A <action> is already scheduled."
7. Scheduling is never reported as completion — the spoken message is
   "<Verb>ing Windows in 15 seconds. Say 'cancel <noun>' to abort."
   If `shutdown.exe` fails inside the deferred callback, a follow-up
   is spoken: "I couldn't schedule the <noun>."
8. All actions are logged (requested / confirmed / scheduled /
   cancelled / executed / failed) via `logger`.

## Confirmation and cancellation flow

- Request → `confirmation_required` result → session stores
  `_pending_confirm` and speaks the ask message.
- "yes" (within 30 s) → same intent re-dispatched with `_confirmed` →
  handler returns `executed` result whose message is the final
  acknowledgement; `after_speech` then arms the countdown.
- "no"/cancel words → `decline_message` spoken ("Okay, I won't shut it
  down."), nothing armed.
- A `CANCEL_PENDING_POWER_ACTION` utterance ("cancel shutdown",
  "cancel sherdown", …) while a *power* confirmation is pending is
  consumed by `VoiceSessionController._route_command`: it drops the
  pending confirmation and speaks the decline message — it is never
  dispatched as a command, so it cannot confirm or execute anything.
  (Added after the 2026-11-20 log showed "Cancel sherdown." classify
  UNKNOWN and leave the ask pending.)
- During the countdown the session stays active and listening;
  `CANCEL_PENDING_POWER_ACTION` aborts via `shutdown.exe /a`.
- Session end/timeout during a pending *confirmation* drops it (the
  session can no longer route "yes"). A *scheduled* countdown that was
  already confirmed is an OS-level timer and is not silently revoked —
  it remains cancellable only while the session is alive (see
  Limitations).

## TTS sequencing

`speech_output_service` pauses the mic gate for the whole
synthesize+playback window. The new `on_finished` meta hook fires in
`_handle_tts_finished` — i.e., strictly after playback — which is what
guarantees "the 15-second countdown begins only after the confirmation
acknowledgement has finished playing." Stub/non-playing TTS resolves
synchronously through the same path.

## Desktop control implementation

`DesktopController` wraps `WindowManager`:

- `minimize_active`/`restore_active`/`snap_active` act only on the
  verified foreground `hwnd`; no window → honest "couldn't find a
  window" failure.
- `close_window(hwnd)` re-enumerates top-level windows and posts
  WM_CLOSE only if the captured hwnd is still live — closing requires
  the standard confirmation and can never hit an unrelated foreground
  app.
- `switch_window` picks the z-order neighbour of the foreground window
  (EnumWindows order ≈ Alt+Tab order), wraps around, and verifies
  focus before reporting success.
- `switch_desktop`/`snap_active` send fixed chords (`win+ctrl+left|right`,
  `win+left|right`) through an injectable sender — never text-derived
  input. Chord delivery failure is reported honestly.

## Automated test results

- `tests/test_windows_system_control.py`: **120 passed** — covers
  confirmation accept/reject, no-execution-without-confirmation, stale
  and mismatched confirmations, shutdown/restart scheduling (deferred
  to post-TTS), cancellation of each, cancel-when-idle, countdown
  timing, sleep API success/failure, duplicate requests, session
  timeout during pending confirmation, all desktop intents, missing
  windows/services, and z-order/chord behavior with fakes.
- 2026-11-20 follow-up additions (all deterministic, fake power ops —
  no real sleep/restart/shutdown):
  - power-cancel utterance during a *pending confirmation* for
    shutdown and restart (incl. the exact "Cancel sherdown."
    transcript) → confirmation dropped, decline ack spoken,
    `power.calls == []`, later "yes" cannot resurrect it;
  - cancel during the armed countdown for shutdown ("Cancel
    sherdown.") and restart ("Cancel restart.");
  - "Sheep my computer." still requires confirmation; "For no" can
    neither confirm nor cancel nor schedule;
  - intent-level positives/negatives for every observed transcript
    plus ordinary-language near-misses ("The sheep are grazing.",
    "I need to shed the load.", "Cancel the meeting.", "Shepard the
    sheep.").
- `tests/test_v108_async_rescore.py`: **8 passed** — adds expired-
  queued-job aborts before inference, exact-zero callback recovery,
  and bounded `stop()` during an in-flight rescore (see
  PHASE_1_RELIABILITY_REPORT.md for the anomaly analysis).
- Regression run: `test_p0_p1_system_ops`, `test_task6`,
  `test_voice_reliability`, `test_intent_accuracy`,
  `test_voice_command_accuracy`, `test_command_corpus`,
  `test_monitor_commands`, `test_v1075_reliability` — **368 passed**
  (two stale `TestStandbyHandoverOrdering` tests were repaired to use
  the real `suspend_detection()` session-end path; they were failing
  before this change because `resume_detection` intentionally reuses
  a surviving detector).
- No real sleep/restart/shutdown/window/keystroke occurred; every OS
  call in tests is a mock or fixed-arg fake.

## Remaining limitations

- Virtual-desktop switching relies on the Win+Ctrl+Arrow shell chord;
  it cannot enumerate desktops or verify the switch landed, and fails
  honestly where virtual desktops are unavailable.
- Snap delivery is verified (window exists + chord sent) but the final
  snapped geometry is delegated to Windows.
- A confirmed 15 s countdown is an OS timer: if the voice session ends
  during the countdown, cancellation is unavailable until the next
  session (the pending record persists in `PowerManager` only for the
  process lifetime).
- A power-cancel utterance during the *pre-confirmation* ask now
  cancels the pending confirmation with an acknowledgement (see
  above); it still cannot arm-then-cancel a countdown that does not
  exist.
- Sleep is immediate post-ack (no countdown); it cannot be cancelled
  once the machine sleeps.
- Bare end-anchored power words ("…will restart.") still classify as
  power requests but always require confirmation before acting.

## 2026-11-20 runtime log review (follow-up)

Reviewed `%LocalAppData%\VoiceOS\logs\voiceos.log` from the Windows
run. Power-command evidence:

- 19:44:33 `RESTART_COMPUTER` recognized → confirmation requested.
- 19:44:40 "For no" → `UNKNOWN` (correct: neither yes nor no).
- 19:44:54 sleep attempt → `UNKNOWN` (transcription missed).
- 19:45:00 "Sheep my computer." → `UNKNOWN` → now `SLEEP_COMPUTER`
  (still confirmation-gated).
- 19:45:29 `SHUTDOWN_COMPUTER` recognized → confirmation requested.
- 19:45:44 "Cancel sherdown." → `UNKNOWN` → now
  `CANCEL_PENDING_POWER_ACTION`; while a power confirmation is
  pending it is consumed by the session controller as a cancellation
  (see Confirmation and cancellation flow).
- **No restart or shutdown was executed** — every destructive path
  remained behind confirmation, exactly as designed.

The same log contains a separate ~62-minute KWS/microphone anomaly
(18:41 rescore `elapsed_ms≈4.3e5`, PortAudio input overflow, all-zero
channel RMS with live callback counters, recovery ~19:43). Analysis
and the rescore-timeout honesty note are in
`docs/PHASE_1_RELIABILITY_REPORT.md` (2026-11-20 addendum). Desktop
controls were not exercised in that log — see Physical testing
instructions; nothing below is physically validated yet.

## Physical testing instructions

1. `py -3 run.py` (or the project's normal launcher), wake Jarvis.
2. Say "Shut down my computer." → hear the ask; say "no" → session
   continues, nothing happens.
2b. Say "Shut down my computer." → at the ask, say "Cancel shutdown."
   (and repeat with "Cancel sherdown." / "Cancel restart." during a
   restart ask) → hear the decline acknowledgement; nothing is armed.
   Verify a following "yes" does nothing.
3. Repeat, say "yes" → hear "Shutting down Windows in 15 seconds…";
   immediately say "Cancel shutdown." → hear "Cancelled."
4. Repeat with "yes" and do nothing → PC shuts down after ~15 s.
5. "Restart my PC." → "yes" → "cancel restart." variants; verify
   `shutdown /a` aborts (check `shutdown /a` manually if unsure).
6. "Put my PC to sleep." → "yes" → machine sleeps after the ack.
7. "Close this window." with Notepad focused → "yes" → window closes.
8. "Switch to the next window." / "previous window." cycles Alt+Tab
   order; "Move to the next desktop." requires ≥2 virtual desktops.
9. "Snap this window left/right." halves the active window.
