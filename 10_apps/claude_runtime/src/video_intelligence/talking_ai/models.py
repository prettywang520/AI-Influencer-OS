"""Shared result vocabulary + engine config for the Talking AI
Intelligence package. Confidence vocabulary (ConfidenceLevel,
ConfidenceThresholds, compute_confidence_level, min_confidence) is
imported directly from video_intelligence.models, never redefined --
the same "confidence derived strictly from evidence quantity/
corroboration, zero evidence always UNKNOWN" philosophy applies here
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Re-exported for convenience -- every talking_ai module imports
# confidence vocabulary from here rather than reaching into
# video_intelligence.models directly, so this module is the single
# place that documents the reuse.
from src.video_intelligence.models import (  # noqa: F401
    ConfidenceLevel,
    ConfidenceThresholds,
    compute_confidence_level,
    min_confidence,
)

# The 10 canonical trait names -- these are TalkingAIProductionDNA's
# own per-domain fields (§17). speech_sync.py alone produces two of
# them (speech_rhythm + pause_behavior); every other analyzer file
# produces exactly one.
TRAIT_FIELDS = (
    "speech_rhythm", "lip_sync", "pause_behavior", "blink", "gaze",
    "head_motion", "facial_motion", "gesture_sync", "subtitle_sync", "framing",
)

# The task's own naturalness-dimension spelling (§15/config) differs
# from the DNA field spelling for 5 of the 10 -- documented once, used
# everywhere (mirrors the "lip_sync" vs "lipsync" reconciliation
# Phase 12D.1 already had to make for its own priorities config).
NATURALNESS_DIMENSION_ALIASES: dict[str, str] = {
    "speech_rhythm": "speech_rhythm",
    "lip_sync": "lip_sync",
    "pause_behavior": "pause_behavior",
    "blink": "blink_variability",
    "gaze": "gaze_variability",
    "head_motion": "head_micro_motion",
    "facial_motion": "facial_micro_motion",
    "gesture_sync": "gesture_speech_sync",
    "subtitle_sync": "subtitle_sync",
    "framing": "camera_naturalness",
}


@dataclass(slots=True)
class TalkingAIConfig:
    """Loaded from config/video_intelligence/talking_ai.yaml (see
    workflow.load_talking_ai_config()). Lives here, not
    production_dna.py, so analyzer.py can depend on the type without a
    circular import -- mirrors VideoIntelligenceConfig's own placement
    reasoning in video_intelligence/models.py exactly."""

    version: str
    schema_version: str
    engine_version: str
    maximum_reasonable_sync_latency_seconds: float
    pause_min_seconds: float
    minimum_talking_videos: int
    recommended_talking_videos: int
    strong_sample: int
    naturalness_dimensions: tuple[str, ...]
    minimum_pattern_corroboration: int
    deidentify_source: bool = True
    preserve_verbatim_transcripts: bool = False
    output_root: str = "output/video_intelligence/talking_ai"
    # Not in the task's own suggested talking_ai.yaml shape, which has
    # no confidence section at all -- added here as a natural,
    # additive completion (same confidence-threshold shape every other
    # engine.yaml in this codebase already carries) since
    # compute_confidence_level() requires them.
    confidence_min_evidence_for_medium: int = 2
    confidence_min_evidence_for_high: int = 4
    confidence_min_corroboration_for_verified: int = 2

    def resolved_output_root(self) -> Path:
        """
        models.py location: 10_apps/claude_runtime/src/video_intelligence/talking_ai/models.py
        parents[3] resolves to 10_apps/claude_runtime.
        """
        runtime_root = Path(__file__).resolve().parents[3]
        return runtime_root / self.output_root

    @property
    def confidence_thresholds(self) -> ConfidenceThresholds:
        return ConfidenceThresholds(
            min_evidence_for_medium=self.confidence_min_evidence_for_medium,
            min_evidence_for_high=self.confidence_min_evidence_for_high,
            min_corroboration_for_verified=self.confidence_min_corroboration_for_verified,
        )


@dataclass(slots=True)
class TalkingAIMetricResult:
    """The common output shape every one of the 10 domain analyzers
    returns: a `trait_name` plus a dict of named metrics (each
    optional -- a metric with no evidence to compute it stays absent,
    never fabricated as 0.0), and the same score/confidence/
    sample_count/evidence_ids/warnings/rationale envelope every
    analyzer in this codebase already uses."""

    trait_name: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    score: float = 0.0
    confidence: str = ConfidenceLevel.UNKNOWN
    sample_count: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class TalkingNaturalnessResult:
    """§15's exact shape. Never a human/AI classifier -- there is no
    field anywhere in this class that could hold such a verdict."""

    dimension_scores: dict[str, float] = field(default_factory=dict)
    dimension_confidence: dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    confidence: str = ConfidenceLevel.UNKNOWN
    sample_count: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class RecommendedRange:
    """§18 -- only ever constructed once enough independent samples
    exist (see production_dna.py); never derived from a single video."""

    metric_name: str
    low: float
    high: float
    sample_count: int
    confidence: str = ConfidenceLevel.UNKNOWN
