"""LearningHistoryEntry -- one immutable record per learning session,
appended (never rewritten) to a CreatorKnowledgeBase's
learning_history.json.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.creator_intelligence.models import CreatorDNA

# Mirrors creator_intelligence.models._TRAIT_FIELDS. Duplicated
# locally rather than importing a private name across packages,
# matching this codebase's established small-helper-duplication
# convention.
TRAIT_FIELDS = (
    "persona", "visual_realism", "photography", "human_authenticity",
    "relationships", "captions", "replies", "storytelling", "reels",
    "branding", "posting", "engagement", "growth",
)


@dataclass(slots=True)
class LearningHistoryEntry:
    session_id: str
    triggered_at: str
    evidence_count_before: int
    evidence_count_after: int
    new_evidence_count: int
    dna_id: str
    overall_confidence: str
    per_trait_confidence: dict[str, str] = field(default_factory=dict)
    per_trait_score: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> LearningHistoryEntry:
        return cls(
            session_id=payload["session_id"],
            triggered_at=payload["triggered_at"],
            evidence_count_before=payload["evidence_count_before"],
            evidence_count_after=payload["evidence_count_after"],
            new_evidence_count=payload["new_evidence_count"],
            dna_id=payload["dna_id"],
            overall_confidence=payload["overall_confidence"],
            per_trait_confidence=dict(payload.get("per_trait_confidence", {})),
            per_trait_score=dict(payload.get("per_trait_score", {})),
        )


def build_learning_history_entry(
    *,
    session_id: str,
    triggered_at: str,
    evidence_count_before: int,
    evidence_count_after: int,
    dna: CreatorDNA,
) -> LearningHistoryEntry:
    per_trait_confidence: dict[str, str] = {}
    per_trait_score: dict[str, float] = {}
    for trait_name in TRAIT_FIELDS:
        trait = getattr(dna, trait_name)
        if trait is not None:
            per_trait_confidence[trait_name] = trait.confidence
            per_trait_score[trait_name] = trait.score
    return LearningHistoryEntry(
        session_id=session_id,
        triggered_at=triggered_at,
        evidence_count_before=evidence_count_before,
        evidence_count_after=evidence_count_after,
        new_evidence_count=evidence_count_after - evidence_count_before,
        dna_id=dna.dna_id,
        overall_confidence=dna.overall_confidence,
        per_trait_confidence=per_trait_confidence,
        per_trait_score=per_trait_score,
    )
