"""Common analyzer contract every one of the 13 domain analyzers
implements: it reads only `AnalyzerContext.evidence` (already in
memory, operator-supplied) and `AnalyzerContext.config`, and returns
an AnalyzerResult. No analyzer performs file I/O, subprocess calls, or
network access. Fresh reimplementation of
creator_intelligence/analyzer_base.py's shape, not imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .evidence import VideoEvidence
from .models import TraitScore, VideoIntelligenceConfig, compute_confidence_level

SCORE_MIN = 0.0
SCORE_MAX = 1.0


def clamp_score(value: float) -> float:
    if value < SCORE_MIN:
        return SCORE_MIN
    if value > SCORE_MAX:
        return SCORE_MAX
    return float(value)


@dataclass(slots=True)
class AnalyzerContext:
    video_id: str
    evidence: list[VideoEvidence]
    config: VideoIntelligenceConfig


@dataclass(slots=True)
class AnalyzerResult:
    trait_score: TraitScore
    warnings: list[str] = field(default_factory=list)


class Analyzer(Protocol):
    name: str

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult: ...


def evidence_for_video(evidence: list[VideoEvidence], video_id: str) -> list[VideoEvidence]:
    """Scopes a (possibly multi-video) evidence list down to just one
    video -- needed because, unlike creator_intelligence's per-creator
    workspace scoping, one video_intelligence evidence set may span
    many videos at once (the knowledge base ingests many)."""
    return [item for item in evidence if item.video_id == video_id]


def evidence_for_tags(evidence: list[VideoEvidence], tags: tuple[str, ...]) -> list[VideoEvidence]:
    """Returns evidence whose `tags` intersect `tags`. Analyzers use
    this to scope a video's evidence down to the subset relevant to
    their own trait domain (e.g. only evidence tagged "hook")."""
    tag_set = set(tags)
    return [item for item in evidence if tag_set.intersection(item.tags)]


def observed_vocabulary(evidence: list[VideoEvidence], vocabulary: tuple[str, ...]) -> list[str]:
    """Which controlled-vocabulary terms (e.g. hooks.yaml's recognized
    hook_type values) appear as exact tags among `evidence`, sorted.
    Used by the 5 yaml-configured analyzers (hook/pacing/camera/
    subtitles/speech) to surface qualitative detail through
    TraitScore.rationale without a bespoke typed field per analyzer."""
    vocabulary_set = set(vocabulary)
    observed: set[str] = set()
    for item in evidence:
        observed.update(vocabulary_set.intersection(item.tags))
    return sorted(observed)


def analyze_by_tag(
    context: AnalyzerContext,
    *,
    trait_name: str,
    tags: tuple[str, ...],
    rationale_with_evidence: str,
    rationale_without_evidence: str,
) -> AnalyzerResult:
    """Shared scoring math every trait analyzer in this package uses:
    scope this video's evidence by tag, derive confidence from
    evidence quantity/corroboration (never from an analyzer's own
    judgement), and derive score from evidence coverage -- NOT a
    quality judgement of the video or its creator. `score` approaches
    1.0 as more independent evidence accumulates and is 0.0 with none."""
    own_video_evidence = evidence_for_video(context.evidence, context.video_id)
    relevant = evidence_for_tags(own_video_evidence, tags)
    evidence_count = len(relevant)
    corroboration_count = len({item.collected_by for item in relevant})
    confidence = compute_confidence_level(evidence_count, corroboration_count, context.config.confidence_thresholds)
    score = clamp_score(evidence_count / (evidence_count + 2)) if evidence_count else 0.0
    rationale = rationale_with_evidence if evidence_count else rationale_without_evidence
    trait_score = TraitScore(
        trait_name=trait_name,
        score=score,
        confidence=confidence,
        evidence_ids=[item.evidence_id for item in relevant],
        rationale=rationale,
    )
    return AnalyzerResult(trait_score=trait_score, warnings=[])
