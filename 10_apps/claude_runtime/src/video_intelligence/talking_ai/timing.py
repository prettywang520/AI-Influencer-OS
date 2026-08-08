"""Time-base infrastructure (§2): canonical seconds, sparse
observation support, no frame-by-frame requirement. Shared by every
domain analyzer -- one implementation, not nine copies. Produces no
DNA field of its own.
"""
from __future__ import annotations

from .evidence import SpeechSegment, TalkingAIEvidence

DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD = 0.15
DEFAULT_LIP_MOTION_INTENSITY_THRESHOLD = 0.15


def speech_active_intervals(segments: list[SpeechSegment]) -> list[tuple[float, float]]:
    """Merges SpeechSegment start/end into speech-on intervals,
    sorted and merged where overlapping/adjacent."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda segment: (segment.start_seconds, segment.end_seconds))
    merged: list[list[float]] = []
    for segment in ordered:
        if merged and segment.start_seconds <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], segment.end_seconds)
        else:
            merged.append([segment.start_seconds, segment.end_seconds])
    return [(start, end) for start, end in merged]


def mouth_motion_events(
    evidence: list[TalkingAIEvidence],
    *,
    open_ratio_threshold: float = DEFAULT_MOUTH_OPEN_RATIO_THRESHOLD,
    intensity_threshold: float = DEFAULT_LIP_MOTION_INTENSITY_THRESHOLD,
) -> list[float]:
    """Timestamps where mouth_open_ratio or lip_motion_intensity
    crosses the configured threshold. Only considers evidence that
    actually reports one of these two fields -- a missing field is
    never treated as "closed"/"zero motion.\""""
    events: list[float] = []
    for item in sorted(evidence, key=lambda item: item.timestamp_seconds):
        if item.mouth_open_ratio is not None and item.mouth_open_ratio >= open_ratio_threshold:
            events.append(item.timestamp_seconds)
        elif item.lip_motion_intensity is not None and item.lip_motion_intensity >= intensity_threshold:
            events.append(item.timestamp_seconds)
    return events


def nearest_event_after(target: float, candidates: list[float], *, max_window: float) -> float | None:
    """The nearest candidate at or after `target`, within max_window
    seconds. None if nothing qualifies -- the core latency-measurement
    primitive every onset/offset alignment metric uses."""
    best: float | None = None
    for candidate in candidates:
        if candidate < target:
            continue
        latency = candidate - target
        if latency > max_window:
            continue
        if best is None or latency < (best - target):
            best = candidate
    return best


def nearest_event_near(target: float, candidates: list[float], *, max_window: float) -> float | None:
    """The nearest candidate to `target` in either direction, within
    max_window seconds."""
    best: float | None = None
    best_distance: float | None = None
    for candidate in candidates:
        distance = abs(candidate - target)
        if distance > max_window:
            continue
        if best_distance is None or distance < best_distance:
            best = candidate
            best_distance = distance
    return best
