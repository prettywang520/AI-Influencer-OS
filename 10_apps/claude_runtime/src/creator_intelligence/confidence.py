"""Confidence vocabulary and computation for the Creator Intelligence
Framework. Confidence is derived strictly from evidence quantity and
corroboration -- it is never assigned by an analyzer's own judgement
call, and zero evidence always yields UNKNOWN. This keeps "how sure
are we" auditable back to concrete Evidence records rather than a
free-floating opinion.
"""
from __future__ import annotations

from dataclasses import dataclass


class ConfidenceLevel:
    """Ordered confidence vocabulary, weakest to strongest."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"

    ORDER = (UNKNOWN, LOW, MEDIUM, HIGH, VERIFIED)


def confidence_rank(level: str) -> int:
    """Returns the ordinal position of a confidence level (higher is
    stronger). Raises ValueError for an unrecognized level."""
    try:
        return ConfidenceLevel.ORDER.index(level)
    except ValueError as exc:
        raise ValueError(f"Unrecognized confidence level: {level!r}") from exc


def min_confidence(levels: list[str]) -> str:
    """Weakest-link aggregation: never reports a combined confidence
    stronger than its weakest input. An empty list is UNKNOWN."""
    if not levels:
        return ConfidenceLevel.UNKNOWN
    return min(levels, key=confidence_rank)


@dataclass(slots=True)
class ConfidenceThresholds:
    min_evidence_for_medium: int
    min_evidence_for_high: int
    min_corroboration_for_verified: int


def compute_confidence(
    evidence_count: int,
    corroboration_count: int,
    thresholds: ConfidenceThresholds,
) -> str:
    """Derives a ConfidenceLevel from how much independent evidence
    backs a trait/claim. `corroboration_count` counts distinct,
    independently-sourced items that agree (a subset of
    evidence_count, never larger). Zero evidence is always UNKNOWN --
    confidence is never fabricated in the absence of evidence."""
    if evidence_count <= 0:
        return ConfidenceLevel.UNKNOWN
    if corroboration_count >= thresholds.min_corroboration_for_verified and (
        evidence_count >= thresholds.min_evidence_for_high
    ):
        return ConfidenceLevel.VERIFIED
    if evidence_count >= thresholds.min_evidence_for_high:
        return ConfidenceLevel.HIGH
    if evidence_count >= thresholds.min_evidence_for_medium:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
