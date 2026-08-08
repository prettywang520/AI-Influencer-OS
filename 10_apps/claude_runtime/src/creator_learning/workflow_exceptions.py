"""Shared error hierarchy for the Creator Learning Workflow. Every
other workflow_*.py module raises one of these -- callers only need
to catch `WorkflowError` (or a specific subclass) to handle every
failure mode this layer can produce.
"""
from __future__ import annotations


class WorkflowError(RuntimeError):
    """Base error for the Creator Learning Workflow."""


class WorkflowConfigError(WorkflowError):
    """Raised when config/creator_learning/workflow.yaml is missing or
    invalid."""


class WorkflowNotFoundError(WorkflowError):
    """Raised when a checkpoint operation references an unknown
    workflow_id."""


class InvalidWorkflowTransitionError(WorkflowError):
    """Raised when a workflow state transition is not allowed by
    workflow_state.py's transition table."""


class WorkflowCheckpointError(WorkflowError):
    """Raised for checkpoint read/write/consistency failures."""


class WorkflowCheckpointMismatchError(WorkflowCheckpointError):
    """Raised when a loaded checkpoint's workflow_id/creator_id does
    not match what was requested."""


class WorkflowAlreadyTerminalError(WorkflowError):
    """Raised when resume()/retry()/cancel() is called on a workflow
    that has already reached COMPLETED or CANCELLED."""


class WorkflowRetryLimitExceededError(WorkflowError):
    """Raised when retry() is called on a workflow that has already
    used its configured max_retry_attempts."""


class WorkflowCliError(WorkflowError):
    """Raised for CLI-level failures (bad arguments, bad input)."""
