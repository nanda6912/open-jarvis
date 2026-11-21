"""Command recognition, registry and handlers for VoiceOS."""

from src.commands.intent import Intent, IntentRecognizer, RuleBasedIntentRecognizer
from src.commands.registry import (
    CommandRegistry,
    CommandResult,
    PERMISSION_DESTRUCTIVE,
    PERMISSION_SAFE,
    PERMISSION_SYSTEM,
)
from src.commands.handlers import create_default_registry

__all__ = [
    "CommandRegistry",
    "CommandResult",
    "Intent",
    "IntentRecognizer",
    "PERMISSION_DESTRUCTIVE",
    "PERMISSION_SAFE",
    "PERMISSION_SYSTEM",
    "RuleBasedIntentRecognizer",
    "create_default_registry",
]
