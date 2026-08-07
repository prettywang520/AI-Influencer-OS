"""Shared error hierarchy for the Creator Research Agent. Every other
module in this package raises one of these -- callers only ever need
to catch `ResearchError` (or a specific subclass) to handle every
failure mode this package can produce.
"""
from __future__ import annotations


class ResearchError(RuntimeError):
    """Base error for the Creator Research Agent."""


class InvalidTransitionError(ResearchError):
    """Raised when a job status transition is not allowed by state.py's
    transition table."""


class JobNotFoundError(ResearchError):
    """Raised when a queue operation references an unknown job_id."""


class DuplicateJobError(ResearchError):
    """Raised when an operation would create a second job with an
    already-used job_id (outside of enqueue()'s own idempotent
    re-enqueue path, which never raises this)."""


class ConnectorNotRegisteredError(ResearchError):
    """Raised when ConnectorRegistry.get() is called with an unknown
    connector name."""


class ConnectorNotImplementedError(ResearchError):
    """Raised when a BaseConnector method has not been overridden by a
    concrete subclass. Never a bare NotImplementedError, so callers
    only need to catch this package's own hierarchy."""


class QueueError(ResearchError):
    """Raised for queue read/write/consistency failures."""


class RetryLimitExceededError(QueueError):
    """Raised when retry() is called on a job that has already used
    its configured max_attempts."""


class PlannerError(ResearchError):
    """Raised for planning failures (e.g. an unrecognized section)."""


class SessionError(ResearchError):
    """Raised for ResearchSession construction/consistency failures."""


class ReportError(ResearchError):
    """Raised for report-building failures."""


class CliError(ResearchError):
    """Raised for CLI-level failures (bad arguments, bad input)."""


class ResearchConfigError(ResearchError):
    """Raised when a config/creator_research/*.yaml file is missing or
    invalid."""
