"""TalkingAIAnalyzerContext + shared timeline statistics
(intervals(), variability(), ratio_true(), score_from_ratio()) every
domain analyzer in this package reuses -- one implementation, not nine
copies. Confidence is computed via
video_intelligence.models.compute_confidence_level(), imported not
reimplemented -- genuinely different math from the parent package's
own tag-coverage analyzer.py (there is no notion of time, ordering, or
co-occurrence in tag-count scoring), but the same confidence
philosophy throughout this codebase.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from .evidence import SpeechSegment, TalkingAIEvidence
from .models import TalkingAIConfig, compute_confidence_level


@dataclass(slots=True)
class TalkingAIAnalyzerContext:
    video_id: str
    evidence: list[TalkingAIEvidence]
    speech_segments: list[SpeechSegment]
    config: TalkingAIConfig


def evidence_for_video(evidence: list[TalkingAIEvidence], video_id: str) -> list[TalkingAIEvidence]:
    return [item for item in evidence if item.video_id == video_id]


def segments_for_video(segments: list[SpeechSegment], video_id: str) -> list[SpeechSegment]:
    return [item for item in segments if item.video_id == video_id]


def sorted_by_timestamp(evidence: list[TalkingAIEvidence]) -> list[TalkingAIEvidence]:
    return sorted(evidence, key=lambda item: item.timestamp_seconds)


def intervals(timestamps: list[float]) -> list[float]:
    """Successive differences between sorted timestamps."""
    ordered = sorted(timestamps)
    return [later - earlier for earlier, later in zip(ordered, ordered[1:])]


def variability(values: list[float]) -> float | None:
    """Population standard deviation, guarded for <2 samples (None,
    never a fabricated 0.0 -- variability is meaningless with fewer
    than 2 observations)."""
    if len(values) < 2:
        return None
    return statistics.pstdev(values)


def ratio_true(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return sum(1 for flag in flags if flag) / len(flags)


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_from_ratio(evidenced_count: int, matched_count: int) -> float:
    """Score from an alignment/coverage hit rate -- 0.0 with no
    evidenced comparisons, matched_count/evidenced_count otherwise,
    clamped to [0, 1]."""
    if evidenced_count <= 0:
        return 0.0
    return clamp_score(matched_count / evidenced_count)


def confidence_for(context: TalkingAIAnalyzerContext, *, evidence_count: int, corroboration_count: int) -> str:
    return compute_confidence_level(evidence_count, corroboration_count, context.config.confidence_thresholds)


def corroboration_count(collected_by_values: list[str]) -> int:
    """Distinct independent sources -- the same "corroboration" notion
    every analyzer in this codebase already uses (a subset of
    evidence_count, never larger)."""
    return len(set(collected_by_values))
