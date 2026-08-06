"""Common Analyzer contract. Every analyzer in this framework
(persona, visual realism, photography, human authenticity, caption,
reply, storytelling, reels, relationship) implements this exact
shape: it reads only `AnalyzerContext.evidence` (already in memory,
operator-supplied) and `AnalyzerContext.config`, and returns an
AnalyzerResult. No analyzer performs file I/O, subprocess calls, or
network access -- there is nothing to inject/fake in tests beyond
plain in-memory Evidence objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .config import CreatorIntelligenceConfig
from .confidence import compute_confidence
from .evidence import Evidence
from .models import TraitScore
from .scoring import clamp_score


@dataclass(slots=True)
class AnalyzerContext:
    evidence: list[Evidence]
    config: CreatorIntelligenceConfig


@dataclass(slots=True)
class AnalyzerResult:
    trait_score: TraitScore
    warnings: list[str] = field(default_factory=list)


class Analyzer(Protocol):
    name: str

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult: ...


def evidence_for_tags(evidence: list[Evidence], tags: tuple[str, ...]) -> list[Evidence]:
    """Returns evidence whose `tags` intersect `tags`. Analyzers use
    this to scope generic evidence lists down to the subset relevant
    to their own trait domain (e.g. only evidence tagged "caption")."""
    tag_set = set(tags)
    return [item for item in evidence if tag_set.intersection(item.tags)]


def analyze_by_tag(
    context: AnalyzerContext,
    *,
    trait_name: str,
    tags: tuple[str, ...],
    rationale_with_evidence: str,
    rationale_without_evidence: str,
) -> AnalyzerResult:
    """Shared scoring math used by every trait analyzer in this
    framework: scope evidence by tag, derive confidence from evidence
    quantity/corroboration (never from an analyzer's own judgement),
    and derive score from evidence coverage/consistency -- NOT from
    any subjective rating of the creator. `score` approaches 1.0 as
    more independent evidence accumulates and is 0.0 with none; it
    represents "how well-evidenced this observed pattern is," not a
    quality judgement of the person. This is what keeps every analyzer
    evidence-based rather than fabricating an opinion."""
    relevant = evidence_for_tags(context.evidence, tags)
    evidence_count = len(relevant)
    corroboration_count = len({item.collected_by for item in relevant})
    confidence = compute_confidence(evidence_count, corroboration_count, context.config.confidence_thresholds)
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
