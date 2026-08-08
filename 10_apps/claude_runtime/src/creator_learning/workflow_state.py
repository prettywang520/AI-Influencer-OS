"""Workflow state machine. CREATED -> RESEARCHING -> LEARNING ->
KNOWLEDGE_UPDATED -> DNA_UPDATED -> REPORTING -> COMPLETED, with
FAILED and CANCELLED both reachable from any non-terminal state.
Every transition is validated here -- nothing else in this package
sets a WorkflowCheckpoint's `state` without going through
validate_transition() (mirrors creator_research/state.py exactly).
"""
from __future__ import annotations

from .workflow_exceptions import InvalidWorkflowTransitionError


class WorkflowState:
    CREATED = "created"
    RESEARCHING = "researching"
    LEARNING = "learning"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    DNA_UPDATED = "dna_updated"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    # Forward progression order. FAILED/CANCELLED are intentionally
    # excluded -- both are reachable from any non-terminal state but
    # have no fixed position in the forward sequence.
    ORDER = (
        CREATED,
        RESEARCHING,
        LEARNING,
        KNOWLEDGE_UPDATED,
        DNA_UPDATED,
        REPORTING,
        COMPLETED,
    )

    ALL = ORDER + (FAILED, CANCELLED)

    TERMINAL = (COMPLETED, FAILED, CANCELLED)


def _build_allowed_transitions() -> dict[str, frozenset[str]]:
    transitions: dict[str, frozenset[str]] = {}
    for index, state in enumerate(WorkflowState.ORDER):
        allowed = set()
        if state not in WorkflowState.TERMINAL and index + 1 < len(WorkflowState.ORDER):
            allowed.add(WorkflowState.ORDER[index + 1])
        if state not in WorkflowState.TERMINAL:
            allowed.add(WorkflowState.FAILED)
            allowed.add(WorkflowState.CANCELLED)
        transitions[state] = frozenset(allowed)
    transitions[WorkflowState.FAILED] = frozenset()
    transitions[WorkflowState.CANCELLED] = frozenset()
    return transitions


ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = _build_allowed_transitions()


def validate_transition(current: str, target: str) -> None:
    """Raises InvalidWorkflowTransitionError unless `target` is a
    direct successor of `current` in WorkflowState.ORDER, or `target`
    is FAILED/CANCELLED and `current` is non-terminal. Never allows a
    backward move or a skipped state -- resume()/retry() re-enter via
    WorkflowRunner._run_from(), not by force-setting state."""
    if current not in WorkflowState.ALL:
        raise InvalidWorkflowTransitionError(f"Unrecognized current state: {current!r}")
    if target not in WorkflowState.ALL:
        raise InvalidWorkflowTransitionError(f"Unrecognized target state: {target!r}")
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidWorkflowTransitionError(f"Cannot transition from {current!r} to {target!r}")
