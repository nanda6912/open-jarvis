"""Voice session conversation controller.

Orchestrates a complete voice interaction session:

- Wake word ("Hello Jarvis") activates the session exactly once.
- Jarvis speaks the activation response, then listens for commands.
- Multiple consecutive commands are accepted within one session.
- Each recognized utterance produces an acknowledgement (no OS commands
  are executed yet — that is a later task).
- The end phrase ("Goodbye Jarvis") ends the session politely.
- An inactivity timeout (configurable, default 60 s) returns Jarvis to
  standby.
- Microphone capture is gated during TTS playback so Jarvis never
  transcribes its own voice.

The controller is Qt-free: all side effects go through injected
services and callbacks, so it is fully testable headlessly.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

from src.core.session_manager import SessionManager
from src.core.state_manager import (
    AppState,
    StateManager,
    allowed_transitions,
)
from src.voice import recognition_taxonomy
from src.voice.speech_output_service import SpeechOutputService
from src.voice.speech_recognition_service import SpeechRecognitionService

logger = logging.getLogger(__name__)

DEFAULT_ACTIVATION_RESPONSE = "At your service."
DEFAULT_GOODBYE_RESPONSE = "Goodbye."
DEFAULT_TIMEOUT_RESPONSE = "Going back to standby."
DEFAULT_NO_SPEECH_RESPONSE = "Sorry, I didn't catch that."
DEFAULT_UNKNOWN_COMMAND_RESPONSE = (
    "Sorry, I don't know how to handle that command yet."
)
DEFAULT_INVALID_DETAIL_RESPONSE = (
    "Sorry, I need a bit more detail for that."
)

_PUNCT_RE = re.compile(r"[.!?…,;:]+$")


def normalize_text(text: str) -> str:
    """Lowercase and strip surrounding whitespace/punctuation."""
    return _PUNCT_RE.sub("", text.strip().lower()).strip()


def build_acknowledgement(text: str) -> str:
    """Build the spoken acknowledgement for a recognized utterance.

    Questions (ending in ``?`` or starting with a wh-word) are reported
    with "I heard you ask ..."; everything else with "I heard you say ...".
    The echoed text keeps its original casing minus trailing punctuation,
    matching the Task 4 examples.
    """
    cleaned = _PUNCT_RE.sub("", text.strip())
    lowered_first = cleaned[:1].lower() + cleaned[1:] if cleaned else cleaned
    question = text.strip().endswith("?") or normalize_text(text).startswith(
        ("what ", "who ", "when ", "where ", "why ", "how ", "which ")
    )
    verb = "ask" if question else "say"
    return f"I heard you {verb} {lowered_first}."


class VoiceSessionController:
    """Drives one voice session at a time (no concurrent sessions)."""

    def __init__(
        self,
        state_manager: StateManager,
        session_manager: SessionManager,
        speech_recognition: SpeechRecognitionService,
        speech_output: SpeechOutputService,
        microphone_controller=None,
        activation_response: str = DEFAULT_ACTIVATION_RESPONSE,
        goodbye_response: str = DEFAULT_GOODBYE_RESPONSE,
        timeout_response: str = DEFAULT_TIMEOUT_RESPONSE,
        no_speech_response: str = DEFAULT_NO_SPEECH_RESPONSE,
        exit_phrase: str = "goodbye jarvis",
        wake_phrase: str = "hello jarvis",
        on_transcript: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_activity: Optional[Callable[[str], None]] = None,
        on_session_state: Optional[Callable[[bool], None]] = None,
        on_no_speech: Optional[Callable[[str], None]] = None,
        intent_recognizer=None,
        command_registry=None,
        history_recorder: Optional[Callable[[str], str, str], None] = None,
        ack_mode: str = "full",
        quick_ack_response: str = "Yes.",
        command_recovery=None,
    ) -> None:
        self._state = state_manager
        self._session = session_manager
        self._stt = speech_recognition
        self._tts = speech_output
        self._mic = microphone_controller
        self._activation_response = activation_response
        self._goodbye_response = goodbye_response
        self._timeout_response = timeout_response
        self._no_speech_response = no_speech_response
        self._exit_phrase = normalize_text(exit_phrase)
        self._wake_phrase = wake_phrase
        # Phrases that re-address Jarvis mid-session (never commands).
        self._wake_variants = {
            normalize_text(p)
            for p in (
                wake_phrase,
                "Hello Jarvis",
                "Hey Jarvis",
                "Hi Jarvis",
                "Okay Jarvis",
                "Ok Jarvis",
                "Jarvis",
            )
        }
        self._on_transcript = on_transcript
        self._on_response = on_response
        self._on_activity = on_activity
        self._on_session_state = on_session_state
        self._on_no_speech = on_no_speech
        self._history = history_recorder

        # Intent recognition + allowlisted command routing (Task 5).
        if intent_recognizer is None:
            from src.commands.intent import RuleBasedIntentRecognizer

            intent_recognizer = RuleBasedIntentRecognizer()
        if command_registry is None:
            from src.commands.handlers import create_default_registry

            command_registry = create_default_registry()
        self._intents = intent_recognizer
        self._registry = command_registry
        self._unknown_response = DEFAULT_UNKNOWN_COMMAND_RESPONSE
        self._invalid_detail_response = DEFAULT_INVALID_DETAIL_RESPONSE
        self._wake_ack_response = "Yes?"
        # V10.4 acknowledgement modes: "full" speaks the configured
        # activation response, "quick" a short phrase, "silent" nothing.
        mode = (ack_mode or "full").strip().lower()
        self._ack_mode = mode if mode in ("full", "quick", "silent") else "full"
        self._quick_ack_response = quick_ack_response or "Yes."
        # Optional misheard-command recovery (V10.4): consulted only
        # when primary intent recognition returns UNKNOWN.
        self._recovery = command_recovery
        # Per-interaction latency records (monotonic timestamps only).
        self._interaction_log: deque = deque(maxlen=100)
        self._current_interaction: Optional[dict] = None
        # V10.8 telemetry: correlation ids + per-phase timestamps.
        self._session_seq = 0
        self._command_seq = 0
        self._current_command: Optional[dict] = None
        # Explicit command-correlation scope: True only while a
        # transcript is being routed synchronously (set by
        # _begin_command, cleared when _handle_transcript returns).
        # Session-level events — no_speech, stt_exception, duplicate
        # transcripts — must NEVER stamp their outcome onto the last
        # command's record.
        self._command_open = False
        self._tts_meta: deque = deque()
        self._last_norm = ""
        self._last_norm_at = 0.0
        # V10.7 HUD: last recognized Intent (read-only diagnostics).
        self.last_intent = None
        self._last_unknown_speak = 0.0
        self._last_no_speech_speak = 0.0
        self._last_stt_error_speak = 0.0

        self._ending = False
        self._end_reason = None
        self._lock = threading.Lock()
        self._pending_confirm = None  # {"intent": Intent, "at": float}
        # Bounded per-utterance recognition diagnostics: transcript,
        # intent, outcome and failure reason — never raw audio.
        self._recognition_log: deque = deque(maxlen=50)

        # Wire service callbacks (they arrive on worker threads).
        self._stt.set_callbacks(
            on_transcript=self._handle_transcript,
            on_no_speech=self._handle_no_speech,
            on_error=self._handle_stt_error,
            on_speech_activity=self._handle_speech_activity,
        )
        self._tts.set_callbacks(
            on_started=self._handle_tts_started,
            on_finished=self._handle_tts_finished,
            on_error=self._handle_tts_error,
        )

    # ------------------------------------------------------------------
    @property
    def session_active(self) -> bool:
        return self._session.is_active and not self._ending

    # ------------------------------------------------------------------
    # Public API (called from GUI thread via Qt signals)
    # ------------------------------------------------------------------
    def recognition_log(self) -> list:
        """Recent per-utterance recognition outcomes (diagnostics)."""
        return [dict(e) for e in self._recognition_log]

    def interaction_timings(self) -> list:
        """Recent wake->ack->first-command latency records (V10.4)."""
        return [dict(e) for e in self._interaction_log]

    def _wake_detection_info(self) -> dict:
        """Best available wake-emission record for correlation."""
        info = {
            "monotonic": time.monotonic(),
            "keyword": None,
            "window_uid": None,
            "generation": None,
            "detection_method": None,
        }
        mic = self._mic
        if mic is None:
            return info
        getter = getattr(mic, "last_wake_detection", None)
        if callable(getter):
            try:
                last = getter()
            except Exception:  # noqa: BLE001
                last = None
            if isinstance(last, dict):
                info.update({k: last.get(k) for k in info})
                if last.get("monotonic"):
                    info["monotonic"] = float(last["monotonic"])
                return info
        detection = getattr(mic, "_detection", None)
        last = getattr(detection, "last_detection", None)
        if isinstance(last, dict):
            info.update({k: last.get(k) for k in info})
            if last.get("monotonic"):
                info["monotonic"] = float(last["monotonic"])
            return info
        mono_getter = getattr(mic, "last_wake_detection_monotonic", None)
        if callable(mono_getter):
            try:
                value = mono_getter()
            except Exception:  # noqa: BLE001
                value = None
            if value:
                info["monotonic"] = float(value)
        return info

    def _wake_detection_monotonic(self) -> float:
        """Best available monotonic timestamp for the wake emission."""
        value = self._wake_detection_info().get("monotonic")
        return float(value) if value else time.monotonic()

    def _begin_interaction(
        self,
        wake_detected_monotonic: Optional[float] = None,
        wake_info: Optional[dict] = None,
    ) -> None:
        self._session_seq += 1
        session_id = f"session-{self._session_seq}"
        self._command_seq = 0
        self._current_command = None
        self._command_open = False
        self._pending_confirm = None
        self._tts_meta.clear()
        info = wake_info if isinstance(wake_info, dict) else (
            self._wake_detection_info()
        )
        wake = (
            float(wake_detected_monotonic)
            if wake_detected_monotonic
            else float(info.get("monotonic") or time.monotonic())
        )
        self._current_interaction = {
            "session_id": session_id,
            "interaction_id": session_id,
            # Wake detection is the KWS emission/callback timestamp;
            # activation is when the session controller took ownership.
            "wake_detected_monotonic": wake,
            "wake_monotonic": wake,  # legacy alias
            # KWS-side correlation for the detection that opened this
            # session (window_uid is generation-scoped, see KWS logs).
            "wake_keyword": info.get("keyword"),
            "wake_window_uid": info.get("window_uid"),
            "wake_detection_generation": info.get("generation"),
            "wake_detection_method": info.get("detection_method"),
            "activated_monotonic": None,
            "ack_tts_started_monotonic": None,
            "ack_tts_finished_monotonic": None,
            "first_transcript_monotonic": None,
            "first_command_processing_started_monotonic": None,
            "first_command_processing_finished_monotonic": None,
            "first_response_tts_started_monotonic": None,
            "first_response_tts_finished_monotonic": None,
            "goodbye_detected_monotonic": None,
            "end_reason": None,
            "end_requested_monotonic": None,
            "farewell_tts_started_monotonic": None,
            "farewell_tts_finished_monotonic": None,
            "standby_restored_monotonic": None,
            # Legacy single-TTS fields now mean the FIRST spoken segment
            # (ack when present, otherwise the first response).
            "tts_started_monotonic": None,
            "tts_finished_monotonic": None,
            "ack_mode": self._ack_mode,
            "commands": [],
        }

    def _begin_command(self, text: str, now: float) -> dict:
        self._command_seq += 1
        rec = self._current_interaction
        session_id = (
            rec.get("session_id") if rec else f"session-{self._session_seq}"
        )
        command_id = f"{session_id}-cmd-{self._command_seq}"
        cmd = {
            "command_id": command_id,
            "utterance_id": command_id,
            "session_id": session_id,
            "transcript": text,
            "transcript_monotonic": now,
            "processing_started_monotonic": None,
            "processing_finished_monotonic": None,
            "processing_seconds": None,
            "intent": None,
            "outcome": None,
            "detail": None,
            "response_tts_started_monotonic": None,
            "response_tts_finished_monotonic": None,
            "response_playback_seconds": None,
        }
        self._current_command = cmd
        self._command_open = True
        if rec is not None:
            rec.setdefault("commands", []).append(cmd)
        return cmd

    def _command_processing_started(self) -> None:
        now = time.monotonic()
        cmd = self._current_command
        rec = self._current_interaction
        if cmd is not None and cmd.get("processing_started_monotonic") is None:
            cmd["processing_started_monotonic"] = now
        if (
            rec is not None
            and rec.get("first_command_processing_started_monotonic") is None
            and cmd is not None
        ):
            rec["first_command_processing_started_monotonic"] = now

    def _command_processing_finished(self) -> None:
        now = time.monotonic()
        cmd = self._current_command
        rec = self._current_interaction
        if cmd is not None and cmd.get("processing_finished_monotonic") is None:
            cmd["processing_finished_monotonic"] = now
        if (
            rec is not None
            and rec.get("first_command_processing_finished_monotonic") is None
            and cmd is not None
        ):
            rec["first_command_processing_finished_monotonic"] = now

    def _close_interaction(self) -> None:
        rec = self._current_interaction
        self._current_interaction = None
        self._current_command = None
        self._command_open = False
        if rec is None:
            return
        commands = rec.get("commands") or []
        rec["command_count"] = len(commands)
        for cmd in commands:
            p0 = cmd.get("processing_started_monotonic")
            p1 = cmd.get("processing_finished_monotonic")
            if p0 is not None and p1 is not None:
                cmd["processing_seconds"] = round(p1 - p0, 4)
            r0 = cmd.get("response_tts_started_monotonic")
            r1 = cmd.get("response_tts_finished_monotonic")
            if r0 is not None and r1 is not None:
                cmd["response_playback_seconds"] = round(r1 - r0, 4)
        first_proc = next(
            (c for c in commands if c.get("processing_started_monotonic")),
            None,
        )
        first_resp = next(
            (c for c in commands if c.get("response_tts_started_monotonic")),
            None,
        )
        if first_proc is not None:
            rec.setdefault(
                "first_command_processing_started_monotonic",
                first_proc.get("processing_started_monotonic"),
            )
            rec.setdefault(
                "first_command_processing_finished_monotonic",
                first_proc.get("processing_finished_monotonic"),
            )
        if first_resp is not None:
            rec.setdefault(
                "first_response_tts_started_monotonic",
                first_resp.get("response_tts_started_monotonic"),
            )
            rec.setdefault(
                "first_response_tts_finished_monotonic",
                first_resp.get("response_tts_finished_monotonic"),
            )
        wake = rec.get("wake_detected_monotonic") or rec.get("wake_monotonic")
        activated = rec.get("activated_monotonic")
        tts_started = rec.get("tts_started_monotonic")
        tts_finished = rec.get("tts_finished_monotonic")
        first_transcript = rec.get("first_transcript_monotonic")
        standby = rec.get("standby_restored_monotonic")
        if wake and activated:
            rec["wake_to_active_seconds"] = round(activated - wake, 4)
        if activated and tts_started:
            rec["active_to_tts_start_seconds"] = round(
                tts_started - activated, 4
            )
        if tts_started and tts_finished:
            rec["tts_playback_seconds"] = round(
                tts_finished - tts_started, 4
            )
        if wake and first_transcript:
            rec["wake_to_first_transcript_seconds"] = round(
                first_transcript - wake, 4
            )
        if wake and standby:
            rec["wake_to_standby_seconds"] = round(standby - wake, 4)
        ack0 = rec.get("ack_tts_started_monotonic")
        ack1 = rec.get("ack_tts_finished_monotonic")
        if ack0 is not None and ack1 is not None:
            rec["ack_playback_seconds"] = round(ack1 - ack0, 4)
        fare0 = rec.get("farewell_tts_started_monotonic")
        fare1 = rec.get("farewell_tts_finished_monotonic")
        if fare0 is not None and fare1 is not None:
            rec["farewell_playback_seconds"] = round(fare1 - fare0, 4)
        self._interaction_log.append(rec)
        logger.info(
            "Interaction timing: %s",
            {k: v for k, v in rec.items() if v is not None},
        )

    def _record_recognition(
        self,
        text: str,
        outcome: str,
        intent_name: Optional[str] = None,
        detail: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> None:
        """Append a structured recognition outcome and log it."""
        rec = getattr(self, "_current_interaction", None)
        # Correlate with a command ONLY inside its open scope: events
        # that never began a command (no_speech, stt_exception,
        # duplicate_ignored) must not overwrite the previous command's
        # outcome — each keeps its own final result.
        cmd = (
            self._current_command if self._command_open else None
        )
        # V10.8 Task 4: correlate with the STT-side attempt record when
        # the recognizer service exposes one (StubSTT fakes do not).
        stt_attempt = None
        try:
            last = getattr(self._stt, "last_attempt", None)
            if isinstance(last, dict):
                stt_attempt = last.get("attempt_id")
        except Exception:  # noqa: BLE001
            stt_attempt = None
        entry = {
            "wall": datetime.now().isoformat(timespec="milliseconds"),
            "monotonic": round(time.monotonic(), 3),
            "session_id": rec.get("session_id") if rec else None,
            "command_id": cmd.get("command_id") if cmd else None,
            "stt_attempt": stt_attempt,
            "text": text,
            "outcome": outcome,
            "category": recognition_taxonomy.category_for_outcome(
                outcome, detail),
            "intent": intent_name,
            "detail": detail,
        }
        if cmd is not None:
            if intent_name is not None:
                cmd["intent"] = intent_name
            cmd["outcome"] = outcome
            if detail is not None:
                cmd["detail"] = detail
        if meta:
            entry.update(meta)
        self._recognition_log.append(entry)
        logger.info(
            "Recognition outcome=%s intent=%s text=%r detail=%s",
            outcome, intent_name, text, detail,
        )
    def handle_wake_word(self, keyword: str = "") -> bool:
        """Activate a session on wake-word detection.

        Returns True when a new session was started; duplicate or
        concurrent activations are ignored.
        """
        with self._lock:
            if self._session.is_active or self._ending:
                logger.info("Wake word ignored: session already active")
                return False
            if self._state.state is AppState.STOPPED:
                return False

            # Capture the KWS emission record before suspend_detection
            # drops the detection service reference.
            wake_info = self._wake_detection_info()
            wake_detected = float(
                wake_info.get("monotonic") or time.monotonic()
            )
            self._session.activate()
            self._transition(AppState.ACTIVATED)
            self._begin_interaction(wake_detected, wake_info)
            if self._current_interaction is not None:
                self._current_interaction["activated_monotonic"] = (
                    time.monotonic()
                )
            if self._mic is not None:
                try:
                    # Stop wake-word inference but keep the frame queue
                    # flowing: the STT service consumes the same queue.
                    self._mic.suspend_detection()
                    self._mic.capture.resume()
                    flush = getattr(self._mic.capture, "flush", None)
                    if flush is not None:
                        flush()
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to hand off microphone to STT")
            self._stt.start()
            if self._ack_mode == "silent":
                # No spoken acknowledgement: straight to listening so the
                # user can issue the command immediately.
                self._transition(AppState.LISTENING)
                self._activity("listening")
            else:
                self._activity("speaking")
                ack = (
                    self._activation_response
                    if self._ack_mode == "full"
                    else self._quick_ack_response
                )
                self._speak(ack, role="ack")
                if not self._tts.is_running:
                    self._handle_tts_finished(ack)
            self._emit_session(True)
            logger.info(
                "Voice session started (wake=%r, ack_mode=%s)",
                keyword, self._ack_mode,
            )
            return True

    def tick(self) -> Optional[int]:
        """Periodic session housekeeping; call ~1/s from the GUI thread.

        Returns remaining seconds while a session is active, else None.
        Triggers the inactivity timeout when expired.
        """
        if not self._session.is_active:
            return None
        remaining = int(self._session.remaining_seconds + 0.999)
        if self._session.is_expired() and not self._ending:
            logger.info("Voice session inactivity timeout")
            self._begin_end(reason="timeout")
            self._speak(self._timeout_response, role="timeout_response")
            self._maybe_finalize_end()
            return 0
        return remaining

    def shutdown(self) -> None:
        """Stop all workers and leave a clean state."""
        with self._lock:
            self._ending = True
        self._stt.stop()
        self._tts.stop()
        if self._mic is not None:
            try:
                self._mic.resume_detection()
            except Exception:  # noqa: BLE001
                pass
        if self._session.is_active:
            self._session.deactivate()
        self._close_interaction()
        self._tts_meta.clear()
        try:
            if self._state.state is not AppState.STOPPED:
                self._transition(AppState.STANDBY)
        except Exception:  # noqa: BLE001
            pass
        self._emit_session(False)

    # ------------------------------------------------------------------
    # STT callbacks (worker thread)
    # ------------------------------------------------------------------
    def _handle_transcript(self, text: str) -> None:
        """STT transcript callback.

        The command-correlation scope opened by ``_begin_command`` is
        valid only for the synchronous routing of THIS transcript —
        the ``finally`` guarantees it closes on every return path so a
        later no-speech/error event can never stamp its outcome onto
        this command's record.
        """
        try:
            self._route_transcript(text)
        finally:
            self._command_open = False

    def _route_transcript(self, text: str) -> None:
        if not self._session.is_active or self._ending:
            return
        if not text or not text.strip():
            # Empty/whitespace transcript: never touch intent recognition,
            # command handlers, history or TTS. The session stays active
            # and LISTENING is preserved.
            logger.debug("Empty transcript ignored")
            return
        self._session.touch()
        if (
            self._current_interaction is not None
            and self._current_interaction.get("first_transcript_monotonic")
            is None
        ):
            self._current_interaction["first_transcript_monotonic"] = (
                time.monotonic()
            )
        if self._on_transcript is not None:
            try:
                self._on_transcript(text)
            except Exception:  # noqa: BLE001
                pass

        if self._history is not None:
            try:
                self._history(text, "voice_command", "acknowledged")
            except Exception:  # noqa: BLE001
                logger.debug("History recording failed", exc_info=True)

        normalized = normalize_text(text)

        # Duplicate transcript suppression: identical text within a few
        # seconds is almost always an STT artefact, not a new command.
        now = time.monotonic()
        if (
            normalized
            and normalized == self._last_norm
            and now - self._last_norm_at < self._DUPE_WINDOW
        ):
            logger.info("Duplicate transcript ignored: %r", text)
            self._record_recognition(text, "duplicate_ignored")
            return
        self._last_norm = normalized
        self._last_norm_at = now
        self._begin_command(text, now)

        if normalized == self._exit_phrase:
            logger.info("Exit phrase recognized: %r", text)
            self._record_recognition(text, "exit_session", "END_SESSION")
            self._command_processing_started()
            self._command_processing_finished()
            self._end_with_goodbye()
            return
        if self._is_wake_variant(normalized):
            # Repeated wake phrase inside a session: acknowledge briefly,
            # never restart or route it as a command. Fuzzy matching
            # tolerates STT mishearings like "Below Jarvis".
            logger.info("Wake phrase repeated inside session: %r", text)
            self._record_recognition(text, "wake_repeat")
            self._activity("speaking")
            self._speak_response(self._wake_ack_response)
            return

        try:
            self._route_command(text)
        finally:
            self._command_processing_finished()

    def _end_with_goodbye(self) -> None:
        """Speak the farewell and end the session."""
        now = time.monotonic()
        rec = self._current_interaction
        if rec is not None:
            rec["goodbye_detected_monotonic"] = now
            rec["end_reason"] = "goodbye"
            rec["end_requested_monotonic"] = (
                rec.get("end_requested_monotonic") or now
            )
        self._begin_end(reason="goodbye")
        self._activity("speaking")
        self._speak(self._goodbye_response, role="goodbye")
        self._maybe_finalize_end()

    _YES_WORDS = frozenset(
        {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
         "do it", "please do", "go ahead"}
    )
    _NO_WORDS = frozenset(
        {"no", "nope", "nah", "cancel", "don't", "do not", "never mind",
         "stop"}
    )
    _CLARIFY_CANCEL_WORDS = frozenset(
        {"cancel", "nevermind", "never mind", "forget it", "forget that",
         "no thanks", "stop", "nothing", "neither"}
    )
    _CONFIRM_WINDOW = 30.0
    # Intents whose pending confirmation can be cancelled by a spoken
    # power-cancel utterance ("cancel shutdown", "cancel sherdown").
    # Cancellation here only drops the pending confirmation — it never
    # dispatches the intent and can never confirm or execute it.
    _POWER_CONFIRM_INTENTS = frozenset(
        {"SLEEP_COMPUTER", "RESTART_COMPUTER", "SHUTDOWN_COMPUTER"}
    )
    _DUPE_WINDOW = 1.5
    _FRAGMENT_WORDS = frozenset({
        "open", "close", "launch", "start", "run", "switch", "find",
        "search", "show", "turn", "mute", "unmute", "minimize",
        "maximize", "restore", "focus", "play", "stop",
    })

    @staticmethod
    def _is_probably_noise(text: str) -> bool:
        """True for empty/tiny/non-speech fragments worth ignoring."""
        norm = normalize_text(text)
        if not norm:
            return True
        letters = sum(c.isalpha() for c in norm)
        if letters == 0:
            return True
        # Very short fragments (<=2 chars) are almost never commands;
        # pending confirmations are handled before this check.
        if len(norm.replace(" ", "")) <= 2:
            return True
        return False

    def _is_wake_variant(self, norm: str) -> bool:
        """Exact or fuzzy match against known wake phrases."""
        if not norm:
            return False
        if norm in self._wake_variants:
            return True
        # Only consider short phrases that still end in "jarvis" —
        # anything longer is a real command, not a wake re-trigger.
        if "jarvis" not in norm.split()[-1:]:
            return False
        if len(norm.split()) > 3:
            return False
        import difflib

        return any(
            difflib.SequenceMatcher(None, norm, v).ratio() >= 0.8
            for v in self._wake_variants
        )

    def _route_command(self, text: str) -> None:
        """Recognize intent, dispatch through the registry, respond."""
        self._command_processing_started()
        # A pending destructive confirmation takes priority: "yes"
        # re-dispatches the stored intent with _confirmed=True.
        pending = self._pending_confirm
        self._pending_confirm = None
        pending_fresh = False
        if pending is not None:
            norm = normalize_text(text)
            pending_fresh = (
                time.monotonic() - pending["at"] < self._CONFIRM_WINDOW
            )
            if not pending_fresh:
                logger.info(
                    "Confirmation expired for %s",
                    pending.get("intent").name
                    if pending.get("intent") is not None
                    else pending.get("kind", "?"),
                )
            if pending.get("kind") == "clarify" and pending_fresh:
                if norm in self._CLARIFY_CANCEL_WORDS:
                    # Explicit cancel: drop the pending selection and
                    # never execute the earlier intent.
                    logger.info("Clarification cancelled by user")
                    self._record_recognition(
                        text, "clarification_cancelled")
                    self._activity("speaking")
                    self._speak_response("Okay, never mind.")
                    return
                resolver = pending.get("resolver")
                if resolver is not None:
                    # Generic clarification (missing window target,
                    # ambiguous monitor, ambiguous window). The resolver
                    # re-interprets the reply; "found" re-dispatches the
                    # original intent with the entity filled in.
                    res = resolver(text)
                    status = getattr(res, "status", "missing")
                    if status == "found":
                        payload = (
                            getattr(res, "app", None)
                            or getattr(res, "monitor", None)
                            or (res.options[0] if getattr(res, "options", ()) else None)
                        )
                        if payload is not None:
                            pending["apply"](pending["intent"], payload)
                            self._dispatch_and_respond(
                                pending["intent"], raw_text=text
                            )
                            return
                    elif status == "ambiguous":
                        self._pending_confirm = {
                            "kind": "clarify",
                            "intent": pending["intent"],
                            "resolver": resolver,
                            "apply": pending["apply"],
                            "at": time.monotonic(),
                        }
                        self._record_recognition(
                            text, "entity_ambiguous", None,
                            "clarify_reply_ambiguous")
                        self._activity("speaking")
                        self._speak_response(
                            pending.get("ambiguous_message")
                            or "Which one did you mean?"
                        )
                        return
                    # Unrecognized/missing reply: fall through to normal
                    # routing (the pending clarification expires).
                else:
                    # Single-option clarification: a bare "yes" accepts
                    # the only candidate; otherwise try name/ordinal
                    # matching as usual.
                    apps = pending.get("apps", [])
                    if (
                        len(apps) == 1
                        and norm in self._YES_WORDS
                    ):
                        pick = apps[0]
                    else:
                        pick = self._match_app_option(norm, apps)
                    if pick is not None:
                        intent = pending["intent"]
                        intent.entities["_app_info"] = pick
                        self._dispatch_and_respond(intent, raw_text=text)
                        return
                    # Unrecognized reply: fall through to normal routing.
            elif pending_fresh and norm in self._YES_WORDS:
                intent = pending["intent"]
                intent.entities["_confirmed"] = True
                logger.info(
                    "Confirmation accepted for %s", intent.name
                )
                self._dispatch_and_respond(intent)
                return
            if pending_fresh and norm in self._NO_WORDS:
                logger.info(
                    "Confirmation rejected for %s",
                    pending["intent"].name,
                )
                self._record_recognition(text, "confirmation_rejected")
                self._activity("speaking")
                self._speak_response(
                    pending.get("decline_message")
                    or "Okay, I won't close it."
                )
                return
            if (
                pending_fresh
                and pending.get("kind") is None
                and getattr(pending.get("intent"), "name", None)
                in self._POWER_CONFIRM_INTENTS
            ):
                # A power-cancel utterance ("cancel shutdown",
                # "cancel sherdown") while Jarvis is asking about a
                # power action cancels the PENDING CONFIRMATION. It is
                # consumed here and never dispatched, so it can never
                # be misread as a fresh command or as a confirmation.
                cancel = self._intents.recognize(text)
                if cancel.name == "CANCEL_PENDING_POWER_ACTION":
                    logger.info(
                        "Power confirmation cancelled by %r (was %s)",
                        text,
                        pending["intent"].name,
                    )
                    self._record_recognition(
                        text, "confirmation_cancelled",
                        "CANCEL_PENDING_POWER_ACTION",
                    )
                    self._activity("speaking")
                    self._speak_response(
                        pending.get("decline_message")
                        or "Okay, cancelled."
                    )
                    return
            # Anything else: confirmation expires, route normally.

        # Skip empty/tiny/non-speech fragments silently — they are
        # almost always VAD noise or background speech, not commands.
        if self._is_probably_noise(text):
            logger.debug("Noise-like transcript ignored: %r", text)
            self._record_recognition(text, "noise_ignored", "UNKNOWN")
            return

        intent = self._intents.recognize(text)
        logger.info(
            "Command routed: text=%r intent=%s entities=%s",
            text,
            intent.name,
            intent.entities,
        )

        # A repeated destructive request while its confirmation is
        # still fresh re-arms the SAME pending operation (never a new
        # silent execution): re-ask instead of overwriting state.
        if (
            pending is not None
            and pending_fresh
            and pending.get("kind") is None
            and intent.name == pending["intent"].name
            and intent.name != "UNKNOWN"
        ):
            self._pending_confirm = {
                "intent": intent,
                "message": pending.get("message"),
                "decline_message": pending.get("decline_message"),
                "at": time.monotonic(),
            }
            self._activity("speaking")
            self._speak_response(
                pending.get("message") or "Should I proceed?"
            )
            return

        if intent.name == "END_SESSION":
            self._record_recognition(text, "exit_session", intent.name)
            self._end_with_goodbye()
            return

        if intent.name == "UNKNOWN":
            # V10.4: before giving up, try conservative contextual
            # recovery of likely misheard commands (move/open/switch/
            # close) against running apps and monitor vocabulary.
            if self._try_recovery(text):
                return
            self._record_recognition(text, "intent_unknown", intent.name)
            # Unsupported/irrelevant speech: never execute, never enter
            # EXECUTING. Speak a short clarification at most once per
            # interval and keep listening.
            now = time.monotonic()
            if now - self._last_unknown_speak >= self._UNKNOWN_SPEAK_INTERVAL:
                self._last_unknown_speak = now
                self._activity("speaking")
                # Bare command verbs ("Open.", "Switch to.") are
                # truncated fragments — ask for detail instead of the
                # full unknown-command speech.
                words = normalize_text(text).split()
                if len(words) == 1 and words[0] in self._FRAGMENT_WORDS:
                    self._speak_listening(self._invalid_detail_response)
                else:
                    self._speak_listening(self._unknown_response)
            return

        self._dispatch_and_respond(intent, raw_text=text)

    def _try_recovery(self, text: str) -> bool:
        """Attempt misheard-command recovery; True when handled.

        A confident recovery is dispatched through the normal registry
        (all safety/clarification rules still apply). An ambiguous
        recovery asks the user to clarify. Fields stay separated: the
        original transcript, the interpreted command, the matched app
        and monitor, the confidence and the match reason are all
        recorded — nothing is silently rewritten.
        """
        if self._recovery is None:
            return False
        try:
            result = self._recovery.recover(text)
        except Exception:  # noqa: BLE001
            logger.exception("Command recovery raised")
            return False
        if result is None or result.status == "none":
            return False

        meta = dict(result.meta or {})
        meta.update({
            "recovered_intent": (
                result.intent.name if result.intent is not None else None
            ),
            "recovery_confidence": result.confidence,
            "recovery_reason": result.reason,
        })
        if result.intent is not None:
            ent = result.intent.entities.setdefault("_recovery", meta)
            if isinstance(ent, dict):
                ent.update(meta)
        if result.status == "ambiguous" and result.clarify is not None:
            # Route through the existing generic clarification flow.
            from src.commands.intent import Intent

            intent = result.intent or Intent(
                name="UNKNOWN", entities={}, raw_text=text
            )
            intent.entities.setdefault("_recovery", meta)
            self._pending_confirm = {
                "kind": "clarify",
                "intent": intent,
                "resolver": result.clarify["resolver"],
                "apply": result.clarify["apply"],
                "ambiguous_message": result.clarify.get(
                    "ambiguous_message"
                ),
                "at": time.monotonic(),
            }
            self._activity("speaking")
            self._speak_response(
                result.clarify.get("message") or "Which one did you mean?"
            )
            self._record_recognition(
                text, "recovery_clarify", None, result.reason, meta=meta
            )
            return True
        if result.intent is not None:
            self._record_recognition(
                text, "recovered", result.intent.name,
                result.interpretation or result.reason, meta=meta,
            )
            self._dispatch_and_respond(result.intent, raw_text=text)
            return True
        self._record_recognition(
            text, "recovery_rejected", None, result.reason, meta=meta
        )
        return False

    def _dispatch_and_respond(self, intent, raw_text: str = "") -> None:
        """Dispatch ``intent`` and speak the appropriate response."""
        # The transcript handler runs on the STT worker thread while
        # timeout/goodbye/shutdown run on the GUI thread. A transcript
        # that passed the entry guard can still be mid-route when the
        # session ends — never execute a stale command, speak a stale
        # response, or resurrect the state machine from STANDBY.
        if not self._session.is_active or self._ending:
            logger.info(
                "Dropping stale dispatch of %s: session ended",
                getattr(intent, "name", "?"),
            )
            self._record_recognition(
                raw_text or getattr(intent, "raw_text", "") or "",
                "session_ended",
                getattr(intent, "name", None),
            )
            return
        self.last_intent = intent
        if intent.name == "END_SESSION":
            self._end_with_goodbye()
            return

        if intent.name == "UNKNOWN":
            self._activity("speaking")
            self._speak_response(self._unknown_response)
            return

        # Well-defined termination ordering: session end flips
        # ``_ending`` under ``self._lock`` (``_begin_end``,
        # ``shutdown``). Hold the same lock across the final liveness
        # check AND ``registry.dispatch`` itself, plus the outcome
        # record, so termination can never interleave mid-dispatch:
        # either the command executes fully and its outcome is
        # recorded before the session may end, or the end wins and
        # the dispatch is dropped as ``session_ended`` above.
        with self._lock:
            if not self._session.is_active or self._ending:
                logger.info(
                    "Dropping stale dispatch of %s: session ended",
                    getattr(intent, "name", "?"),
                )
                self._record_recognition(
                    raw_text or getattr(intent, "raw_text", "") or "",
                    "session_ended",
                    getattr(intent, "name", None),
                )
                return
            result = self._registry.dispatch(intent)
            logger.info(
                "Command result: intent=%s success=%s executed=%s "
                "confirmation=%s ambiguous=%s message=%r",
                result.intent,
                result.success,
                result.executed,
                result.confirmation_required,
                result.ambiguous,
                result.message,
            )
            # V10.8 Task 4: classify WHERE the attempt ended. Entity
            # problems (missing/ambiguous) are distinct from handler
            # failures; clarify payloads carry an explicit entity_issue
            # tag set by the command layer.
            clarify = result.data.get("clarify")
            entity_issue = (
                clarify.get("entity_issue")
                if isinstance(clarify, dict) else None
            )
            err = result.error or ""
            outcome = (
                "executed" if result.executed
                else "confirmation_requested" if result.confirmation_required
                else "entity_ambiguous"
                if (
                    result.ambiguous
                    or entity_issue == "ambiguous"
                    or err == "ambiguous_selection"
                )
                else "entity_missing"
                if (entity_issue == "missing" or err.startswith("missing_"))
                else "clarification_requested" if clarify is not None
                else "handler_failed" if not result.success
                else "validated"
            )
            meta = {"entity_issue": entity_issue} if entity_issue else None
            self._record_recognition(
                raw_text or getattr(intent, "raw_text", "") or "",
                outcome,
                intent.name,
                result.error or result.message,
                meta=meta,
            )
        if result.confirmation_required:
            # Potentially disruptive op: ask, remember, wait for "yes".
            self._pending_confirm = {
                "intent": intent,
                "message": result.message or "Should I proceed?",
                "decline_message": result.data.get("decline_message"),
                "at": time.monotonic(),
            }
            logger.info(
                "Confirmation requested for %s: %s",
                intent.name,
                result.message,
            )
            self._activity("speaking")
            self._speak_response(result.message or "Should I proceed?")
            return
        if result.ambiguous:
            # Several plausible apps: ask which one, remember options.
            self._pending_confirm = {
                "kind": "clarify",
                "intent": intent,
                "apps": list(result.data.get("apps", [])),
                "at": time.monotonic(),
            }
            self._activity("speaking")
            self._speak_response(result.message or "Which one did you mean?")
            return
        clarify = result.data.get("clarify")
        if clarify is not None:
            # Handler-requested clarification (missing target, ambiguous
            # monitor/window). Bounded by _CONFIRM_WINDOW like other
            # pending confirmations; the original intent is preserved.
            self._pending_confirm = {
                "kind": "clarify",
                "intent": intent,
                "resolver": clarify["resolver"],
                "apply": clarify["apply"],
                "ambiguous_message": clarify.get("ambiguous_message"),
                "at": time.monotonic(),
            }
            self._activity("speaking")
            self._speak_response(result.message or "Could you clarify?")
            return
        if result.executed:
            self._activity("speaking")
            after = result.data.get("after_speech")
            self._speak_response(
                result.message or self._unknown_response,
                on_finished=after if callable(after) else None,
            )
            return
        if not result.success:
            # Recognized but invalid arguments (missing name, bad folder,
            # out-of-range amount, denied permission, unregistered).
            self._activity("speaking")
            if result.error in ("unregistered_intent", "permission_denied"):
                self._speak_response(self._unknown_response)
            else:
                # Speak the handler's specific reason ("I couldn't find
                # X.", "X doesn't appear to be running.") when it has
                # one; fall back to the generic detail request.
                self._speak_response(
                    result.message or self._invalid_detail_response
                )
            return

        # Recognized and validated, but intentionally not executed yet —
        # fall back to the spoken acknowledgement.
        self._activity("speaking")
        self._speak_response(build_acknowledgement(raw_text or intent.raw_text))

    @staticmethod
    def _match_app_option(norm: str, apps: list):
        """Match a spoken reply to one of the ambiguous app options."""
        if not apps:
            return None
        from src.controllers.application_discovery import _normalize

        # Ordinal / numeric selection: "the first one", "2", "number two".
        ordinals = {
            "first": 0, "1": 0, "one": 0, "second": 1, "2": 1, "two": 1,
            "third": 2, "3": 2, "three": 2, "fourth": 3, "4": 3,
            "four": 3, "fifth": 4, "5": 4, "five": 4,
        }
        stripped = norm.strip(" .!?")
        if stripped in ordinals and ordinals[stripped] < len(apps):
            return apps[ordinals[stripped]]
        # Phrases like "the second one" / "number two please".
        for token in re.split(r"\s+", stripped):
            token = token.strip(".,!?")
            if token in ordinals and ordinals[token] < len(apps):
                return apps[ordinals[token]]
        # Exact normalized name or exe-stem match.
        for app in apps:
            if _normalize(app.name) == norm or _normalize(app.exe_name) == norm:
                return app
        # Unique substring match.
        hits = [
            a for a in apps
            if norm and (norm in _normalize(a.name)
                         or norm in _normalize(a.exe_name))
        ]
        if len(hits) == 1:
            return hits[0]
        return None

    _NO_SPEECH_SPEAK_INTERVAL = 20.0
    _UNKNOWN_SPEAK_INTERVAL = 20.0

    def _handle_no_speech(self, reason: str) -> None:
        """Empty/whitespace STT result: stay in LISTENING, stay silent.

        An utterance that produced no transcript is almost always VAD
        noise or silence — it must not enter PROCESSING/EXECUTING, must
        not dispatch UNKNOWN and must not generate TTS.
        """
        if not self._session.is_active or self._ending:
            return
        # A stalled or dead capture stream is a pipeline fault, not
        # user activity — it must never reset the inactivity timeout.
        if reason not in ("no_audio_frames", "capture_failed"):
            self._session.touch()
        logger.info("No speech captured (%s)", reason)
        self._record_recognition("", "no_speech", None, reason)
        if self._on_no_speech is not None:
            try:
                self._on_no_speech(reason)
            except Exception:  # noqa: BLE001
                pass
        # V10.5: voiced audio reached STT but decoded empty — the user
        # likely spoke a real command. Ask once per interval instead of
        # staying silently LISTENING. Never for no-audio/below-VAD
        # reasons (those are not failed commands), and never a replay.
        if reason == "vad_accepted_stt_empty":
            now = time.monotonic()
            if now - self._last_no_speech_speak >= (
                self._NO_SPEECH_SPEAK_INTERVAL
            ):
                self._last_no_speech_speak = now
                self._activity("speaking")
                self._speak_listening(self._no_speech_response)

    def _handle_speech_activity(self, active: bool) -> None:
        if not self._session.is_active or self._ending:
            return
        self._activity("listening" if active else "processing")

    _STT_ERROR_SPEAK_INTERVAL = 20.0
    _STT_ERROR_RESPONSE = "Sorry, I had trouble hearing that."

    def _handle_stt_error(self, message: str) -> None:
        """STT decode failure: record, recover, stay in LISTENING.

        V10.8 Task 4: an STT exception is NOT a recognition outcome —
        it is recorded as ``stt_exception`` (distinct from empty
        transcript and from handler failure), the session stays
        active, and at most one throttled clarification is spoken so
        the user can simply repeat the command.
        """
        logger.warning("STT error: %s", message)
        if not self._session.is_active or self._ending:
            return
        self._session.touch()
        self._record_recognition("", "stt_exception", None, message)
        now = time.monotonic()
        if now - self._last_stt_error_speak >= (
            self._STT_ERROR_SPEAK_INTERVAL
        ):
            self._last_stt_error_speak = now
            # Defensive: never leave the state machine in a non-
            # LISTENING state because of a decode failure.
            self._transition(AppState.LISTENING)
            self._activity("speaking")
            self._speak_listening(self._STT_ERROR_RESPONSE)

    def _handle_tts_error(self, message: str) -> None:
        logger.warning("TTS error: %s", message)

    # ------------------------------------------------------------------
    # TTS callbacks (worker thread)
    # ------------------------------------------------------------------
    def _handle_tts_started(self, text: str) -> None:
        now = time.monotonic()
        meta = self._tts_meta[0] if self._tts_meta else None
        if meta is not None:
            meta["started_monotonic"] = now
        rec = self._current_interaction
        if rec is None:
            return
        role = meta.get("role") if meta else None
        cmd = meta.get("command") if meta else None
        if rec.get("tts_started_monotonic") is None:
            rec["tts_started_monotonic"] = now
        if role == "ack":
            rec["ack_tts_started_monotonic"] = now
        elif role == "goodbye":
            rec["farewell_tts_started_monotonic"] = now
        else:
            if rec.get("first_response_tts_started_monotonic") is None:
                rec["first_response_tts_started_monotonic"] = now
            if cmd is not None:
                cmd["response_tts_started_monotonic"] = now

    def _handle_tts_finished(self, text: str) -> None:
        now = time.monotonic()
        meta = self._tts_meta.popleft() if self._tts_meta else None
        if meta is not None:
            meta["finished_monotonic"] = now
        post_speech = meta.get("on_finished") if meta else None
        rec = self._current_interaction
        if rec is not None:
            role = meta.get("role") if meta else None
            cmd = meta.get("command") if meta else None
            if rec.get("tts_finished_monotonic") is None:
                rec["tts_finished_monotonic"] = now
            if role == "ack":
                rec["ack_tts_finished_monotonic"] = now
            elif role == "goodbye":
                rec["farewell_tts_finished_monotonic"] = now
            else:
                if rec.get("first_response_tts_finished_monotonic") is None:
                    rec["first_response_tts_finished_monotonic"] = now
                if cmd is not None:
                    cmd["response_tts_finished_monotonic"] = now
        if self._on_response is not None:
            try:
                self._on_response(text)
            except Exception:  # noqa: BLE001
                pass
        if self._ending:
            # Finalize only after EVERY queued utterance has finished —
            # a farewell queued behind a still-playing response must
            # not restore standby (and wake-word detection) while
            # Jarvis is still speaking.
            if not self._tts_meta:
                self._finalize_end()
        elif self._session.is_active:
            # Jarvis finished speaking; listen for the next command.
            self._transition(AppState.LISTENING)
            self._activity("listening")
        # Deferred action hook (e.g. arm a power countdown): runs only
        # now that this utterance has finished playing, so the 15 s
        # window can never start while Jarvis is still talking. A
        # returned string is spoken as a follow-up response.
        if post_speech is not None:
            followup = None
            try:
                followup = post_speech()
            except Exception:  # noqa: BLE001
                logger.exception("Post-speech callback failed")
            if isinstance(followup, str) and followup.strip():
                if self._session.is_active and not self._ending:
                    self._speak_response(followup)
                else:
                    logger.info(
                        "Dropping post-speech follow-up: session ended"
                    )

    # ------------------------------------------------------------------
    def _begin_end(self, reason: str) -> None:
        """Stop listening and mark the session as ending."""
        self._end_reason = reason
        rec = self._current_interaction
        if rec is not None:
            now = time.monotonic()
            rec["end_reason"] = reason
            rec["end_requested_monotonic"] = (
                rec.get("end_requested_monotonic") or now
            )
            if reason == "goodbye":
                rec["goodbye_detected_monotonic"] = (
                    rec.get("goodbye_detected_monotonic") or now
                )
        with self._lock:
            self._ending = True
        self._stt.stop()
        logger.info("Voice session ending (%s)", reason)

    def _maybe_finalize_end(self) -> None:
        """Finalize a pending session end when no speech can complete it.

        Covers the cases where the farewell never produces a finish
        callback: ``speak()`` dropped it (queue full) or the playback
        worker is not running. When speech IS queued, meta entries
        remain and ``_handle_tts_finished`` finalizes after the last
        queued item drains.
        """
        if not self._ending:
            return
        if self._tts_meta and getattr(self._tts, "is_running", True):
            # Farewell still queued/playing on a live worker — its
            # finish callback will finalize once the queue drains.
            return
        self._finalize_end()

    def _finalize_end(self) -> None:
        """Return to standby once the farewell has been spoken."""
        # V10.3.3 lifecycle milestone: farewell TTS finished, handover
        # to standby wake-word detection begins now.
        logger.info(
            "Farewell TTS completed; standby handover starting "
            "(reason=%s)",
            self._end_reason,
        )
        with self._lock:
            self._ending = False
        self._session.deactivate()
        if self._mic is not None:
            try:
                self._mic.resume_detection()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to resume wake-word detection")
        self._transition(AppState.STANDBY)
        self._activity("standby")
        rec = self._current_interaction
        if rec is not None:
            rec["standby_restored_monotonic"] = time.monotonic()
        self._close_interaction()
        self._tts_meta.clear()
        self._emit_session(False)
        logger.info("Voice session ended; standby resumed")

    def _speak(
        self,
        text: str,
        *,
        role: str = "response",
        command: Optional[dict] = None,
        on_finished: Optional[Callable[[], Optional[str]]] = None,
    ) -> bool:
        """Queue TTS with telemetry metadata (FIFO matches playback).

        ``on_finished`` (optional) runs when THIS utterance finishes
        playing — used to defer destructive scheduling until the
        acknowledgement has been heard. It may return a follow-up
        string to speak, or None.
        """
        meta = {
            "role": role,
            # TTS attribution follows the same explicit scope: a
            # no-speech/clarification utterance spoken after a command
            # must not be recorded as that command's response playback.
            "command": (
                command if command is not None
                else (
                    self._current_command
                    if self._command_open else None
                )
            ),
            "text": text,
            "on_finished": on_finished,
            "queued_monotonic": time.monotonic(),
            "started_monotonic": None,
            "finished_monotonic": None,
        }
        self._tts_meta.append(meta)
        try:
            queued = self._tts.speak(text)
        except Exception:
            try:
                self._tts_meta.remove(meta)
            except ValueError:
                pass
            raise
        if queued is False:
            try:
                self._tts_meta.remove(meta)
            except ValueError:
                pass
        return bool(queued)

    def _speak_response(
        self,
        text: str,
        on_finished: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._transition(AppState.EXECUTING)
        self._speak(text, role="response", on_finished=on_finished)
        if not self._tts.is_running:
            # No playback worker (e.g. tests with a stub service): treat
            # the response as finished so the session keeps listening.
            self._handle_tts_finished(text)

    def _speak_listening(self, text: str) -> None:
        """Speak a clarification WITHOUT leaving LISTENING.

        Used for unsupported/invalid transcripts: the session stays in
        LISTENING (no PROCESSING/EXECUTING detour) while Jarvis says a
        short reply. ``_handle_tts_finished`` still normalizes the state
        back to LISTENING when playback completes.
        """
        self._transition(AppState.LISTENING)
        self._speak(text, role="response")
        if not self._tts.is_running:
            self._handle_tts_finished(text)

    # ------------------------------------------------------------------
    def _transition(self, target: AppState) -> None:
        """Transition to ``target`` along the shortest legal path."""
        start = self._state.state
        if start is target or start is AppState.STOPPED:
            return
        if target not in self._reachable(start):
            logger.debug(
                "State transition %s -> %s unavailable",
                start.name,
                target.name,
            )
            return
        path = self._shortest_path(start, target)
        for step in path:
            self._state.transition_to(step)

    @staticmethod
    def _reachable(start: AppState) -> set:
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in allowed_transitions(current):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    @staticmethod
    def _shortest_path(start: AppState, target: AppState) -> list:
        prev = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current is target:
                break
            for nxt in allowed_transitions(current):
                if nxt not in prev:
                    prev[nxt] = current
                    queue.append(nxt)
        path = []
        node = target
        while node is not None:
            path.append(node)
            node = prev.get(node)
        path.reverse()
        return path[1:]  # exclude the start state

    def _activity(self, activity: str) -> None:
        if self._on_activity is not None:
            try:
                self._on_activity(activity)
            except Exception:  # noqa: BLE001
                pass

    def _emit_session(self, active: bool) -> None:
        if self._on_session_state is not None:
            try:
                self._on_session_state(active)
            except Exception:  # noqa: BLE001
                pass
