"""Qt signal bridge for voice/audio workers.

Workers run on plain Python threads; this QObject lives on the GUI
thread so emitting its signals delivers them there safely (queued
connections).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class VoiceSignals(QObject):
    """Thread-safe signals emitted by microphone/wake-word workers."""

    wakeWordDetected = Signal(str)
    micStatusChanged = Signal(str, str)
    micError = Signal(str)
    audioStats = Signal(object)
    transcriptReceived = Signal(str)
    noSpeech = Signal(str)
    jarvisResponse = Signal(str)
    assistantActivity = Signal(str)
    sessionStateChanged = Signal(bool)
    sessionCountdown = Signal(int)
    sttStatusChanged = Signal(str)
    ttsStatusChanged = Signal(str)
    # V10.7 HUD: measured TTS amplitude envelope (levels, chunk_seconds),
    # emitted on the GUI thread via queued connection.
    ttsEnvelope = Signal(object, float)
