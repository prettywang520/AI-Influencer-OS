"""Pure report dataclasses. Every field here is derived entirely from
an already-computed CreatorDNA + its cited Evidence +
LearningHistoryEntry list + StyleEvolutionRecord -- no new analysis,
no new confidence math happens in this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LearningReport:
    creator_id: str
    session_id: str
    generated_at: str
    evidence_count: int
    overall_confidence: str
    session_count: int
    confidence_trend: list[tuple[str, str]] = field(default_factory=list)
    gaps: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(slots=True)
class VisualReport:
    creator_id: str
    session_id: str
    visual_realism_score: float | None
    visual_realism_confidence: str | None
    visual_realism_rationale: str
    photography_score: float | None
    photography_confidence: str | None
    photography_rationale: str
    covered_tags: tuple[str, ...] = ()
    missing_tags: tuple[str, ...] = ()
    cited_excerpts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RelationshipReport:
    creator_id: str
    session_id: str
    relationships_score: float | None
    relationships_confidence: str | None
    claims: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class ReplyReport:
    creator_id: str
    session_id: str
    replies_score: float | None
    replies_confidence: str | None
    rationale: str
    language_mix_covered: tuple[str, ...] = ()
    language_mix_missing: tuple[str, ...] = ()
    cited_excerpts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CaptionReport:
    creator_id: str
    session_id: str
    captions_score: float | None
    captions_confidence: str | None
    rationale: str
    hashtag_frequency: dict[str, int] = field(default_factory=dict)
    mention_frequency: dict[str, int] = field(default_factory=dict)
    cited_excerpts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StyleReport:
    creator_id: str
    session_id: str
    is_baseline: bool
    new_relationship_claim_count: int
    changed_relationship_claim_count: int
    new_warning_count: int
    trait_summary: list[str] = field(default_factory=list)
