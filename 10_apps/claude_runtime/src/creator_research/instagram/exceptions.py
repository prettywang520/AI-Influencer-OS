"""Shared error hierarchy for the Instagram Research Connector. Every
other module in this package raises one of these -- callers (including
orchestrator.run_job(), which catches a bare `Exception` per section)
only ever need to catch `InstagramConnectorError` (or a specific
subclass) to handle every failure mode this package can produce.
"""
from __future__ import annotations


class InstagramConnectorError(RuntimeError):
    """Base error for the Instagram Research Connector."""


class InstagramConfigError(InstagramConnectorError):
    """Raised when config/creator_research/instagram.yaml is missing or invalid."""


class AccessLimitedError(InstagramConnectorError):
    """Raised when observed page state indicates content is not
    accessible (private account, geo-block, removed content, etc.).
    Never an attempt to circumvent the limitation."""


class AuthenticationRequiredError(InstagramConnectorError):
    """Raised when observed page state indicates a login wall.
    Credentials are never read, stored, or entered anywhere in this
    package -- this exception is the connector's only response."""


class RateLimitedError(InstagramConnectorError):
    """Raised when observed page state indicates Instagram is rate
    limiting requests ("try again later" style messaging)."""


class ChallengeDetectedError(InstagramConnectorError):
    """Raised when observed page state indicates a CAPTCHA, checkpoint,
    or other anti-bot challenge screen. Never solved or bypassed."""


class NavigationError(InstagramConnectorError):
    """Raised when the browser adapter fails to reach a target page
    after exhausting the configured retry budget."""


class SelectorNotFoundError(InstagramConnectorError):
    """Raised when none of a UI element's selector candidates matched
    and the caller requires the element to proceed."""


class CheckpointError(InstagramConnectorError):
    """Raised for checkpoint read/write failures."""


class CheckpointMismatchError(CheckpointError):
    """Raised when a loaded checkpoint's job_id/creator_id does not
    match the job being resumed -- never silently adopted."""


class RetryBudgetExhaustedError(InstagramConnectorError):
    """Raised when rate_limit.py's configured max_retries_per_navigation
    is exceeded. Always a stop, never an evasion attempt."""
