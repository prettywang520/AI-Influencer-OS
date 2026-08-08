"""TalkingAIProductionDNA -- the top-level aggregate record for one
talking-AI video (task §17). A sibling of video_intelligence.VideoDNA,
never an extension of it (see the approved plan's Exploration
Summary) -- its own dataclass, own dna_id, own file.
build_talking_ai_dna() runs all 10 domain analyzers plus the
naturalness aggregator over one video's evidence, mirroring
video_intelligence.production_dna.build_video_dna() exactly.
derive_recommended_ranges() (§18) only ever derives a range once at
least `minimum_pattern_corroboration` independent videos supply a
given metric -- a single source video never becomes a universal rule.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .analyzer import TalkingAIAnalyzerContext
from .blink import BlinkAnalyzer
from .evidence import SpeechSegment, TalkingAIEvidence
from .facial_motion import FacialMotionAnalyzer
from .framing import FramingAnalyzer
from .gaze import GazeAnalyzer
from .gesture_sync import GestureSyncAnalyzer
from .head_motion import HeadMotionAnalyzer
from .lip_sync import LipSyncAnalyzer
from .models import (
    ConfidenceLevel,
    RecommendedRange,
    TRAIT_FIELDS,
    TalkingAIConfig,
    TalkingAIMetricResult,
    TalkingNaturalnessResult,
    compute_confidence_level,
    min_confidence,
)
from .naturalness import NaturalnessAnalyzer
from .speech_sync import SpeechSyncAnalyzer
from .subtitle_sync import SubtitleSyncAnalyzer


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(payload: dict[str, Any], *, length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def _metric_result_to_payload(result: TalkingAIMetricResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "trait_name": result.trait_name,
        "metrics": result.metrics,
        "score": result.score,
        "confidence": result.confidence,
        "evidence_ids": sorted(result.evidence_ids),
        "rationale": result.rationale,
    }


def _naturalness_to_payload(result: TalkingNaturalnessResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "dimension_scores": result.dimension_scores,
        "dimension_confidence": result.dimension_confidence,
        "score": result.score,
        "confidence": result.confidence,
        "evidence_ids": sorted(result.evidence_ids),
        "rationale": result.rationale,
    }


def _compute_dna_id(dna: "TalkingAIProductionDNA") -> str:
    """Content-hash over every field except `generated_at` and the id
    itself -- re-serializing an unchanged record at a different time
    never mints a new id (mirrors
    video_intelligence.production_dna._compute_dna_id() exactly)."""
    payload: dict = {
        "video_id": dna.video_id,
        "subject_label": dna.subject_label,
        "schema_version": dna.schema_version,
        "evidence_index": sorted(dna.evidence_index),
        "artifact_patterns": dict(sorted(dna.artifact_patterns.items())),
        "overall_confidence": dna.overall_confidence,
        "warnings": sorted(dna.warnings),
        "naturalness": _naturalness_to_payload(dna.naturalness),
    }
    for name in TRAIT_FIELDS:
        payload[name] = _metric_result_to_payload(getattr(dna, name))
    return _content_hash(payload)


@dataclass(slots=True)
class TalkingAIProductionDNA:
    """Top-level aggregated result for one observed talking-AI video.
    `subject_label` is an operator-chosen free text label, same rule
    every other *DNA record in this codebase already enforces."""

    video_id: str
    subject_label: str
    schema_version: str = "1.0"
    generated_at: str = ""
    evidence_index: list[str] = field(default_factory=list)

    speech_rhythm: TalkingAIMetricResult | None = None
    lip_sync: TalkingAIMetricResult | None = None
    pause_behavior: TalkingAIMetricResult | None = None
    blink: TalkingAIMetricResult | None = None
    gaze: TalkingAIMetricResult | None = None
    head_motion: TalkingAIMetricResult | None = None
    facial_motion: TalkingAIMetricResult | None = None
    gesture_sync: TalkingAIMetricResult | None = None
    subtitle_sync: TalkingAIMetricResult | None = None
    framing: TalkingAIMetricResult | None = None

    naturalness: TalkingNaturalnessResult | None = None

    # Tally of observed ArtifactType tags across all supplied evidence
    # (task §16) -- a raw count only, never an interpretive
    # "AI-generated" conclusion. One artifact tag never implies AI
    # generation on its own.
    artifact_patterns: dict[str, int] = field(default_factory=dict)

    overall_confidence: str = ConfidenceLevel.UNKNOWN
    warnings: list[str] = field(default_factory=list)

    dna_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.dna_id = _compute_dna_id(self)


def build_talking_ai_dna(
    video_id: str,
    subject_label: str,
    evidence: list[TalkingAIEvidence],
    speech_segments: list[SpeechSegment],
    config: TalkingAIConfig,
) -> TalkingAIProductionDNA:
    """Builds a TalkingAIProductionDNA record from operator-/test-
    supplied evidence only, scoped to `video_id`."""
    context = TalkingAIAnalyzerContext(
        video_id=video_id, evidence=evidence, speech_segments=speech_segments, config=config,
    )

    speech_analyzer = SpeechSyncAnalyzer()
    rhythm_result = speech_analyzer.analyze_speech_rhythm(context)
    pause_result = speech_analyzer.analyze_pause_behavior(context)
    lip_sync_result = LipSyncAnalyzer().analyze(context)
    blink_result = BlinkAnalyzer().analyze(context)
    gaze_result = GazeAnalyzer().analyze(context)
    head_motion_result = HeadMotionAnalyzer().analyze(context)
    facial_motion_result = FacialMotionAnalyzer().analyze(context)
    gesture_sync_result = GestureSyncAnalyzer().analyze(context)
    subtitle_sync_result = SubtitleSyncAnalyzer().analyze(context)
    framing_result = FramingAnalyzer().analyze(context)

    domain_results: dict[str, TalkingAIMetricResult] = {
        "speech_rhythm": rhythm_result,
        "lip_sync": lip_sync_result,
        "pause_behavior": pause_result,
        "blink": blink_result,
        "gaze": gaze_result,
        "head_motion": head_motion_result,
        "facial_motion": facial_motion_result,
        "gesture_sync": gesture_sync_result,
        "subtitle_sync": subtitle_sync_result,
        "framing": framing_result,
    }
    naturalness_result = NaturalnessAnalyzer().analyze(context, domain_results)

    own_evidence = [item for item in evidence if item.video_id == video_id]
    own_segments = [segment for segment in speech_segments if segment.video_id == video_id]

    artifact_tally: Counter[str] = Counter()
    for item in own_evidence:
        artifact_tally.update(item.artifact_tags)

    warnings: list[str] = list(naturalness_result.warnings)
    evidence_ids: set[str] = {item.evidence_id for item in own_evidence} | {
        segment.segment_id for segment in own_segments
    }
    for result in domain_results.values():
        warnings.extend(result.warnings)

    overall_confidence = min_confidence(
        [result.confidence for result in domain_results.values()] + [naturalness_result.confidence]
    )

    return TalkingAIProductionDNA(
        video_id=video_id,
        subject_label=subject_label,
        schema_version=config.schema_version,
        generated_at=_now_iso(),
        evidence_index=sorted(evidence_ids),
        speech_rhythm=rhythm_result,
        lip_sync=lip_sync_result,
        pause_behavior=pause_result,
        blink=blink_result,
        gaze=gaze_result,
        head_motion=head_motion_result,
        facial_motion=facial_motion_result,
        gesture_sync=gesture_sync_result,
        subtitle_sync=subtitle_sync_result,
        framing=framing_result,
        naturalness=naturalness_result,
        artifact_patterns=dict(artifact_tally),
        overall_confidence=overall_confidence,
        warnings=warnings,
    )


def derive_recommended_ranges(
    dnas: list[TalkingAIProductionDNA],
    config: TalkingAIConfig,
) -> list[RecommendedRange]:
    """§18 -- only ever derives a range for a metric once at least
    `minimum_pattern_corroboration` independent videos supplied a
    value for it. A single source video never becomes a universal
    rule; a metric supplied by fewer videos than the corroboration
    floor is silently skipped, not fabricated with a wide/low-
    confidence range."""
    values_by_metric: dict[str, list[float]] = {}
    for dna in dnas:
        for trait_name in TRAIT_FIELDS:
            result = getattr(dna, trait_name)
            if result is None:
                continue
            for metric_key, value in result.metrics.items():
                if value is None:
                    continue
                values_by_metric.setdefault(f"{trait_name}.{metric_key}", []).append(float(value))

    ranges: list[RecommendedRange] = []
    for metric_name, values in sorted(values_by_metric.items()):
        sample_count = len(values)
        if sample_count < config.minimum_pattern_corroboration:
            continue
        confidence = compute_confidence_level(sample_count, sample_count, config.confidence_thresholds)
        ranges.append(
            RecommendedRange(
                metric_name=metric_name,
                low=min(values),
                high=max(values),
                sample_count=sample_count,
                confidence=confidence,
            )
        )
    return ranges
