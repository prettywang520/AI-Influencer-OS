"""Shared result vocabulary every analyzer returns: confidence
(derived strictly from evidence quantity/corroboration, never an
analyzer's own judgement -- zero evidence always yields UNKNOWN),
TraitScore (one analyzer's output), StoryBeat (a basis-free but
evidence-qualified narrative beat), and CTAObservation (one observed
call-to-action). Fresh reimplementation of
creator_intelligence.confidence/models' shape, not imported -- this
package stays fully decoupled from creator_intelligence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class ConfidenceLevel:
    """Ordered confidence vocabulary, weakest to strongest."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"

    ORDER = (UNKNOWN, LOW, MEDIUM, HIGH, VERIFIED)


def confidence_rank(level: str) -> int:
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


@dataclass(slots=True)
class VideoIntelligenceConfig:
    """Engine-wide policy loaded from config/video_intelligence/engine.yaml
    (see production_dna.load_engine_config()). Lives here, not in
    production_dna.py, so analyzer.py can depend on the type without a
    circular import (production_dna.py is the one file that imports
    every analyzer)."""

    schema_version: str
    confidence_min_evidence_for_medium: int
    confidence_min_evidence_for_high: int
    confidence_min_corroboration_for_verified: int
    excerpt_max_chars: int
    output_root: str = "output/video_intelligence"
    analyzer_weights: dict[str, float] = field(default_factory=dict)

    @property
    def confidence_thresholds(self) -> ConfidenceThresholds:
        return ConfidenceThresholds(
            min_evidence_for_medium=self.confidence_min_evidence_for_medium,
            min_evidence_for_high=self.confidence_min_evidence_for_high,
            min_corroboration_for_verified=self.confidence_min_corroboration_for_verified,
        )

    def resolved_output_root(self) -> Path:
        """
        models.py location: 10_apps/claude_runtime/src/video_intelligence/models.py
        parents[2] resolves to 10_apps/claude_runtime.
        """
        runtime_root = Path(__file__).resolve().parents[2]
        return runtime_root / self.output_root

    def weight_for(self, analyzer_name: str) -> float:
        return float(self.analyzer_weights.get(analyzer_name, 1.0))


def compute_confidence_level(
    evidence_count: int,
    corroboration_count: int,
    thresholds: ConfidenceThresholds,
) -> str:
    """Derives a ConfidenceLevel from how much independent evidence
    backs a trait. `corroboration_count` counts distinct,
    independently-sourced items that agree (a subset of
    evidence_count, never larger). Zero evidence is always UNKNOWN."""
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


@dataclass(slots=True)
class TraitScore:
    """One analyzer's scored output for a single trait category.
    `score` represents how well-evidenced the observed pattern is,
    never a subjective quality judgement of the creator."""

    trait_name: str
    score: float
    confidence: str = ConfidenceLevel.UNKNOWN
    evidence_ids: list[str] = field(default_factory=list)
    rationale: str = ""


class BeatType:
    """Closed vocabulary for a story beat's position in the narrative
    arc -- matches the task's STORY section literally."""

    BEGINNING = "beginning"
    MIDDLE = "middle"
    END = "end"
    PROBLEM = "problem"
    SOLUTION = "solution"
    PAYOFF = "payoff"

    ALL = (BEGINNING, MIDDLE, END, PROBLEM, SOLUTION, PAYOFF)


@dataclass(slots=True)
class StoryBeat:
    """One observed narrative beat, always evidence-qualified (same
    "never assert more than the evidence shows" discipline
    creator_intelligence.models.RelationshipClaim already established)."""

    beat_type: str
    description: str
    evidence_ids: list[str] = field(default_factory=list)


class CTAType:
    """Closed vocabulary for a call-to-action -- matches the task's
    CTA section literally."""

    FOLLOW = "follow"
    COMMENT = "comment"
    SHARE = "share"
    SAVE = "save"
    QUESTION = "question"

    ALL = (FOLLOW, COMMENT, SHARE, SAVE, QUESTION)


@dataclass(slots=True)
class CTAObservation:
    """One observed call-to-action and when it occurred."""

    cta_type: str
    timing_seconds: float | None = None
    evidence_ids: list[str] = field(default_factory=list)
