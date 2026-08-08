"""Shared error hierarchy for the Talking AI Live Learning Adapter.
Every other live/*.py module raises one of these -- callers only need
to catch `LiveAdapterError` (or a specific subclass) to handle every
failure mode this package can produce.
"""
from __future__ import annotations


class LiveAdapterError(RuntimeError):
    """Base error for the Talking AI Live Learning Adapter."""


class LiveAdapterConfigError(LiveAdapterError):
    """Raised when config/video_intelligence/talking_ai_live.yaml is
    missing or invalid."""


class LiveAdapterValidationError(LiveAdapterError):
    """Raised when a structural validation check fails clearly
    (unknown data is allowed; invalid data is not)."""


class LiveAdapterCliError(LiveAdapterError):
    """Raised for CLI-level failures (bad arguments, bad input)."""
