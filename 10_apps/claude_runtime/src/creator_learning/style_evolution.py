"""Pure diff between two CreatorDNA snapshots -- "Style Evolution".
Never computes a subjective improved/declined judgement; only reports
what numerically/structurally changed, and lets the report layer
describe it factually. previous=None (first-ever session) yields an
all-new-baseline record, never a fabricated delta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.creator_intelligence.models import CreatorDNA, RelationshipClaim

# Mirrors creator_intelligence.models._TRAIT_FIELDS -- duplicated
# locally rather than importing a private name across packages.
TRAIT_FIELDS = (
    "persona", "visual_realism", "photography", "human_authenticity",
    "relationships", "captions", "replies", "storytelling", "reels",
    "branding", "posting", "engagement", "growth",
)


@dataclass(slots=True)
class TraitDelta:
    trait_name: str
    previous_score: float | None
    current_score: float | None
    score_delta: float | None
    previous_confidence: str | None
    current_confidence: str | None
    confidence_changed: bool


@dataclass(slots=True)
class StyleEvolutionRecord:
    is_baseline: bool
    trait_deltas: dict[str, TraitDelta] = field(default_factory=dict)
    new_relationship_claims: list[RelationshipClaim] = field(default_factory=list)
    changed_relationship_claims: list[RelationshipClaim] = field(default_factory=list)
    new_warnings: list[str] = field(default_factory=list)
    previous_overall_confidence: str | None = None
    current_overall_confidence: str = ""


def _claim_key(claim: RelationshipClaim) -> tuple[str, str]:
    return (claim.description, claim.basis)


def diff_creator_dna(previous: CreatorDNA | None, current: CreatorDNA) -> StyleEvolutionRecord:
    if previous is None:
        trait_deltas = {}
        for trait_name in TRAIT_FIELDS:
            trait = getattr(current, trait_name)
            trait_deltas[trait_name] = TraitDelta(
                trait_name=trait_name,
                previous_score=None,
                current_score=trait.score if trait else None,
                score_delta=None,
                previous_confidence=None,
                current_confidence=trait.confidence if trait else None,
                confidence_changed=False,
            )
        return StyleEvolutionRecord(
            is_baseline=True,
            trait_deltas=trait_deltas,
            new_relationship_claims=list(current.relationship_claims),
            changed_relationship_claims=[],
            new_warnings=list(current.warnings),
            previous_overall_confidence=None,
            current_overall_confidence=current.overall_confidence,
        )

    trait_deltas = {}
    for trait_name in TRAIT_FIELDS:
        prev_trait = getattr(previous, trait_name)
        curr_trait = getattr(current, trait_name)
        prev_score = prev_trait.score if prev_trait else None
        curr_score = curr_trait.score if curr_trait else None
        score_delta = (
            (curr_score - prev_score) if (prev_score is not None and curr_score is not None) else None
        )
        prev_conf = prev_trait.confidence if prev_trait else None
        curr_conf = curr_trait.confidence if curr_trait else None
        trait_deltas[trait_name] = TraitDelta(
            trait_name=trait_name,
            previous_score=prev_score,
            current_score=curr_score,
            score_delta=score_delta,
            previous_confidence=prev_conf,
            current_confidence=curr_conf,
            confidence_changed=(prev_conf != curr_conf),
        )

    prev_claims = {_claim_key(claim): claim for claim in previous.relationship_claims}
    curr_claims = {_claim_key(claim): claim for claim in current.relationship_claims}
    new_claims = [claim for key, claim in curr_claims.items() if key not in prev_claims]
    changed_claims = [
        claim
        for key, claim in curr_claims.items()
        if key in prev_claims and sorted(prev_claims[key].evidence_ids) != sorted(claim.evidence_ids)
    ]

    new_warnings = [warning for warning in current.warnings if warning not in previous.warnings]

    return StyleEvolutionRecord(
        is_baseline=False,
        trait_deltas=trait_deltas,
        new_relationship_claims=new_claims,
        changed_relationship_claims=changed_claims,
        new_warnings=new_warnings,
        previous_overall_confidence=previous.overall_confidence,
        current_overall_confidence=current.overall_confidence,
    )
