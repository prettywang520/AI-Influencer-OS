"""Shared error hierarchy for the Instagram Reels Learning Adapter.
Every other instagram/*.py module raises one of these -- callers only
need to catch `AdapterError` (or a specific subclass) to handle every
failure mode this package can produce.
"""
from __future__ import annotations


class AdapterError(RuntimeError):
    """Base error for the Instagram Reels Learning Adapter."""


class AdapterConfigError(AdapterError):
    """Raised when config/video_intelligence/instagram_reels.yaml is
    missing or invalid."""


class AdapterValidationError(AdapterError):
    """Raised when a packet/evidence structural validation check
    fails clearly (unknown data is allowed; invalid data is not)."""


class AdapterMappingError(AdapterError):
    """Raised when Reel evidence cannot be mapped into the adapter's
    normalized schema (e.g. an unparseable source_description on
    evidence that claims to be reel-shaped)."""


class AdapterCliError(AdapterError):
    """Raised for CLI-level failures (bad arguments, bad input files)."""
