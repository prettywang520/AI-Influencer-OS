"""Comparison-ready model. Reshapes a CreatorDNA into a typed
container two records could later be placed side-by-side into by a
future phase. This module performs NO actual comparison -- no diffing,
no similarity scoring, no ranking. That logic does not exist yet and
is explicitly out of scope for Phase 12A.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import CreatorDNA, RelationshipClaim, TraitScore

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


@dataclass(slots=True)
class ComparisonReadyProfile:
    """Same trait shape as CreatorDNA, carried as a standalone typed
    record so a future comparison phase can hold two of these
    side-by-side without depending on CreatorDNA's own identity
    fields (dna_id/generated_at), which are not comparison-relevant."""

    subject_label: str
    dna_id: str
    schema_version: str
    overall_confidence: str
    traits: dict[str, TraitScore | None] = field(default_factory=dict)
    relationship_claims: list[RelationshipClaim] = field(default_factory=list)


def build_comparison_ready_profile(dna: CreatorDNA) -> ComparisonReadyProfile:
    """Pure reshape of a CreatorDNA -- no comparison logic."""
    return ComparisonReadyProfile(
        subject_label=dna.subject_label,
        dna_id=dna.dna_id,
        schema_version=dna.schema_version,
        overall_confidence=dna.overall_confidence,
        traits={name: getattr(dna, name) for name in _TRAIT_FIELDS},
        relationship_claims=list(dna.relationship_claims),
    )
