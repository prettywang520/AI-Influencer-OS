"""Coarse-timing bridge (task §4) -- the one place "preserve ordering
without inventing exact seconds... do not fake frame precision" is
enforced. `TalkingAIEvidence.timestamp_seconds`/`SpeechSegment.start_seconds`/
`end_seconds` stay required `float`s (evidence.py is not modified --
see the approved plan's Exploration Summary), so this module never
claims a placeholder second value is an observed one: ordinal-only
evidence gets monotonically increasing placeholder timestamps, capped
at `confidence="low"`, with an explicit note and (from adapter.py, once
per record) a warning -- the *order* is always real, the *seconds* are
only ever real when the caller actually supplied real seconds.

All functions return a *new* object via `dataclasses.replace()` --
never mutate a caller's object in place, and never hand-patch
`evidence_id`/`segment_id` (both are `init=False` fields that
`dataclasses.replace()` correctly recomputes via `__post_init__`,
since neither hash includes `confidence`, only `notes`/timing, which
this module *does* legitimately change).
"""
from __future__ import annotations

from dataclasses import replace

from src.video_intelligence.models import ConfidenceLevel, confidence_rank

from ..evidence import SpeechSegment, TalkingAIEvidence

APPROXIMATE_TIMING_NOTE = "approximate timing"
ORDINAL_PLACEHOLDER_NOTE = "ordinal placeholder timing (not observed seconds)"

DEFAULT_ORDINAL_START_SECONDS = 0.0
DEFAULT_ORDINAL_STEP_SECONDS = 1.0


def _append_note(existing: str, marker: str) -> str:
    if marker in existing:
        return existing
    return f"{existing}; {marker}" if existing else marker


def _cap_confidence_low(current: str) -> str:
    """Never allows a confidence above `low` for a timing-bridged
    item -- both "unknown" (bumped up, since an approximate/ordinal
    observation is more informative than no observation at all) and
    anything at or above "medium" (bumped down, since this module's
    own timing is never itself more than approximate) collapse to
    exactly `low`."""
    if confidence_rank(current) > confidence_rank(ConfidenceLevel.LOW):
        return ConfidenceLevel.LOW
    if current == ConfidenceLevel.UNKNOWN:
        return ConfidenceLevel.LOW
    return current


def mark_approximate_timing(item: TalkingAIEvidence) -> TalkingAIEvidence:
    """For a "near Xs" style observation -- the float is used as-is;
    only the confidence/notes are adjusted to honestly reflect that
    the exact instant is a caller-estimated approximation, not a
    precisely observed one."""
    return replace(
        item,
        confidence=_cap_confidence_low(item.confidence),
        notes=_append_note(item.notes, APPROXIMATE_TIMING_NOTE),
    )


def mark_approximate_speech_segment(segment: SpeechSegment) -> SpeechSegment:
    return replace(segment, confidence=_cap_confidence_low(segment.confidence))


def assign_ordinal_timestamps(
    items: list[TalkingAIEvidence],
    *,
    start_seconds: float = DEFAULT_ORDINAL_START_SECONDS,
    step_seconds: float = DEFAULT_ORDINAL_STEP_SECONDS,
) -> list[TalkingAIEvidence]:
    """Assigns strictly increasing placeholder timestamps preserving
    the given list order -- never claims this is real elapsed time.
    `step_seconds` must be positive so successive items are always
    strictly ordered, never tied."""
    if step_seconds <= 0:
        raise ValueError(f"step_seconds must be positive, got {step_seconds!r}")

    assigned: list[TalkingAIEvidence] = []
    for index, item in enumerate(items):
        assigned.append(
            replace(
                item,
                timestamp_seconds=start_seconds + index * step_seconds,
                confidence=ConfidenceLevel.LOW,
                notes=_append_note(item.notes, ORDINAL_PLACEHOLDER_NOTE),
            )
        )
    return assigned


def assign_ordinal_speech_segments(
    segments: list[SpeechSegment],
    *,
    start_seconds: float = DEFAULT_ORDINAL_START_SECONDS,
    step_seconds: float = DEFAULT_ORDINAL_STEP_SECONDS,
) -> list[SpeechSegment]:
    """Same ordinal-placeholder discipline as `assign_ordinal_timestamps()`,
    for speech segments -- `step_seconds` doubles as each placeholder
    segment's duration, since only relative order (not real duration)
    is known."""
    if step_seconds <= 0:
        raise ValueError(f"step_seconds must be positive, got {step_seconds!r}")

    assigned: list[SpeechSegment] = []
    for index, segment in enumerate(segments):
        segment_start = start_seconds + index * step_seconds
        assigned.append(
            replace(
                segment,
                start_seconds=segment_start,
                end_seconds=segment_start + step_seconds,
                confidence=ConfidenceLevel.LOW,
            )
        )
    return assigned
