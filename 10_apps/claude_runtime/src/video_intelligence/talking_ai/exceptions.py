"""Shared error hierarchy for the Talking AI Intelligence package.
Every other talking_ai/*.py module raises one of these -- callers
only need to catch `TalkingAIError` (or a specific subclass) to handle
every failure mode this package can produce.
"""
from __future__ import annotations


class TalkingAIError(RuntimeError):
    """Base error for the Talking AI Intelligence package."""


class TalkingAIConfigError(TalkingAIError):
    """Raised when config/video_intelligence/talking_ai.yaml is
    missing or invalid."""


class InvalidTalkingAIEvidenceError(TalkingAIError):
    """Raised when a TalkingAIEvidence/SpeechSegment field carries a
    value outside its closed vocabulary."""


class TalkingAIValidationError(TalkingAIError):
    """Raised when a structural validation check fails clearly
    (unknown data is allowed; invalid data is not)."""


class KnowledgeBaseError(TalkingAIError):
    """Raised for knowledge-base read/write/consistency failures."""


class SerializationError(TalkingAIError):
    """Raised for save/load failures, including dna_id/pattern_id
    integrity mismatches on load."""


class TalkingAICliError(TalkingAIError):
    """Raised for CLI-level failures (bad arguments, bad input)."""
