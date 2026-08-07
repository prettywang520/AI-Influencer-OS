"""Research job state machine. QUEUED -> STARTING -> PROFILE -> GRID
-> CAPTIONS -> COMMENTS -> REPLIES -> REELS -> HIGHLIGHTS -> VISUAL ->
COMPLETE, with FAILED reachable from any non-terminal state. Every
transition is validated here -- nothing else in this package sets
`ResearchJob.status` without going through `validate_transition()`.
"""
from __future__ import annotations

from .exceptions import InvalidTransitionError


class JobStatus:
    QUEUED = "queued"
    STARTING = "starting"
    PROFILE = "profile"
    GRID = "grid"
    CAPTIONS = "captions"
    COMMENTS = "comments"
    REPLIES = "replies"
    REELS = "reels"
    HIGHLIGHTS = "highlights"
    VISUAL = "visual"
    COMPLETE = "complete"
    FAILED = "failed"

    # Forward progression order. FAILED is intentionally excluded --
    # it is reachable from any non-terminal state but has no fixed
    # position in the forward sequence.
    ORDER = (
        QUEUED,
        STARTING,
        PROFILE,
        GRID,
        CAPTIONS,
        COMMENTS,
        REPLIES,
        REELS,
        HIGHLIGHTS,
        VISUAL,
        COMPLETE,
    )

    ALL = ORDER + (FAILED,)

    TERMINAL = (COMPLETE, FAILED)

    # The 8 "working" states a research job passes through while
    # actively collecting evidence (excludes the QUEUED/STARTING
    # bookends and the COMPLETE/FAILED terminals).
    WORKING_STATES = (PROFILE, GRID, CAPTIONS, COMMENTS, REPLIES, REELS, HIGHLIGHTS, VISUAL)


def _build_allowed_transitions() -> dict[str, frozenset[str]]:
    transitions: dict[str, frozenset[str]] = {}
    for index, status in enumerate(JobStatus.ORDER):
        allowed = set()
        if status not in JobStatus.TERMINAL and index + 1 < len(JobStatus.ORDER):
            allowed.add(JobStatus.ORDER[index + 1])
        if status not in JobStatus.TERMINAL:
            allowed.add(JobStatus.FAILED)
        transitions[status] = frozenset(allowed)
    transitions[JobStatus.FAILED] = frozenset()
    return transitions


ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = _build_allowed_transitions()


def validate_transition(current: str, target: str) -> None:
    """Raises InvalidTransitionError unless `target` is a direct
    successor of `current` in JobStatus.ORDER, or `target` is FAILED
    and `current` is non-terminal. Never allows a backward move or a
    skipped state -- resuming re-enters via planner.resume_point(),
    not by force-setting status."""
    if current not in JobStatus.ALL:
        raise InvalidTransitionError(f"Unrecognized current status: {current!r}")
    if target not in JobStatus.ALL:
        raise InvalidTransitionError(f"Unrecognized target status: {target!r}")
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidTransitionError(f"Cannot transition from {current!r} to {target!r}")
