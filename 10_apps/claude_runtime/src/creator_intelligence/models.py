"""Typed result models for the Creator Intelligence Framework:
TraitScore (one analyzer's output), RelationshipClaim (a
basis-qualified relationship observation), and CreatorDNA (the
top-level aggregated record). Every score/confidence here traces back
to concrete Evidence IDs -- nothing is asserted without a citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import _hashing
from .confidence import ConfidenceLevel


class RelationshipBasis:
    """How a relationship-related observation is grounded. Encodes
    the framework's core rule: a childhood photo is evidence of a
    childhood-photo *narrative* only, never of a verified relationship
    fact, unless independently corroborated."""

    PRESENTED_NARRATIVE = "presented_narrative"
    VISUALLY_DEPICTED = "visually_depicted"
    VERIFIED = "verified"
    INFERRED = "inferred"
    UNKNOWN = "unknown"

    ALL = (PRESENTED_NARRATIVE, VISUALLY_DEPICTED, VERIFIED, INFERRED, UNKNOWN)


@dataclass(slots=True)
class TraitScore:
    """One analyzer's scored output for a single trait category."""

    trait_name: str
    score: float
    confidence: str = ConfidenceLevel.UNKNOWN
    evidence_ids: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class RelationshipClaim:
    """One relationship-related observation, always basis-qualified."""

    description: str
    basis: str
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CreatorDNA:
    """Top-level aggregated result: a structured, confidence-scored
    summary of a creator's observable content patterns, built entirely
    from cited Evidence. `subject_label` is an operator-chosen free
    text label -- this framework never requires or auto-populates a
    real account handle here."""

    subject_label: str
    schema_version: str = "1.0"
    generated_at: str = ""
    evidence_index: list[str] = field(default_factory=list)

    persona: TraitScore | None = None
    visual_realism: TraitScore | None = None
    photography: TraitScore | None = None
    human_authenticity: TraitScore | None = None
    relationships: TraitScore | None = None
    relationship_claims: list[RelationshipClaim] = field(default_factory=list)
    captions: TraitScore | None = None
    replies: TraitScore | None = None
    storytelling: TraitScore | None = None
    reels: TraitScore | None = None

    branding: TraitScore | None = None
    posting: TraitScore | None = None
    engagement: TraitScore | None = None
    growth: TraitScore | None = None

    overall_confidence: str = ConfidenceLevel.UNKNOWN
    warnings: list[str] = field(default_factory=list)

    dna_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.dna_id = _compute_dna_id(self)


_TRAIT_FIELDS = (
    "persona",
    "visual_realism",
    "photography",
    "human_authenticity",
    "relationships",
    "captions",
    "replies",
    "storytelling",
    "reels",
    "branding",
    "posting",
    "engagement",
    "growth",
)


def _trait_score_to_payload(trait: TraitScore | None) -> dict | None:
    if trait is None:
        return None
    return {
        "trait_name": trait.trait_name,
        "score": trait.score,
        "confidence": trait.confidence,
        "evidence_ids": sorted(trait.evidence_ids),
        "rationale": trait.rationale,
    }


def _compute_dna_id(dna: CreatorDNA) -> str:
    """Content-hash over every field except `generated_at` and the
    id itself, so re-serializing an unchanged record at a different
    time never mints a new id."""
    payload: dict = {
        "subject_label": dna.subject_label,
        "schema_version": dna.schema_version,
        "evidence_index": sorted(dna.evidence_index),
        "relationship_claims": sorted(
            (
                {
                    "description": c.description,
                    "basis": c.basis,
                    "evidence_ids": sorted(c.evidence_ids),
                }
                for c in dna.relationship_claims
            ),
            key=lambda c: (c["description"], c["basis"]),
        ),
        "overall_confidence": dna.overall_confidence,
        "warnings": sorted(dna.warnings),
    }
    for name in _TRAIT_FIELDS:
        payload[name] = _trait_score_to_payload(getattr(dna, name))
    return _hashing.content_hash(payload)
