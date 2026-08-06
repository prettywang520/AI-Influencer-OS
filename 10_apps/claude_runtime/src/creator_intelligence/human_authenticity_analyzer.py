"""Human Authenticity Analyzer -- scores evidence coverage for
narrative consistency, and extracts relationship-related observations
as basis-qualified RelationshipClaims. Every relationship claim states
its own evidentiary basis (presented narrative / visually depicted /
verified / inferred / unknown) and is never upgraded to VERIFIED
without distinct corroborating evidence. A childhood photo, for
example, is evidence of a childhood-photo narrative only.
"""
from __future__ import annotations

from collections import defaultdict

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags
from .models import RelationshipBasis, RelationshipClaim

TRAIT_NAME = "human_authenticity"
TAGS = ("human_authenticity",)
RELATIONSHIP_TAG = "relationship"

_BASIS_TAG_MAP = {
    "verified": RelationshipBasis.VERIFIED,
    "visually_depicted": RelationshipBasis.VISUALLY_DEPICTED,
    "presented_narrative": RelationshipBasis.PRESENTED_NARRATIVE,
    "inferred": RelationshipBasis.INFERRED,
}

MIN_CORROBORATION_FOR_VERIFIED = 2


def _basis_for(evidence_tags: list[str]) -> str:
    for tag, basis in _BASIS_TAG_MAP.items():
        if tag in evidence_tags:
            return basis
    return RelationshipBasis.UNKNOWN


class HumanAuthenticityAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence="Narrative-consistency pattern observed across cited evidence.",
            rationale_without_evidence="No human-authenticity-tagged evidence supplied; confidence is unknown.",
        )

    def extract_relationship_claims(self, context: AnalyzerContext) -> list[RelationshipClaim]:
        """Groups relationship-tagged evidence by `source_description`
        (treated as the claim's description) and assigns each group a
        basis. A claim only reaches VERIFIED when at least
        MIN_CORROBORATION_FOR_VERIFIED *independent* sources
        (distinct collected_by) each supplied "verified"-tagged
        evidence for it -- a single source's own assertion, however
        strongly tagged, is never sufficient on its own."""
        relevant = evidence_for_tags(context.evidence, (RELATIONSHIP_TAG,))
        grouped: dict[str, list] = defaultdict(list)
        for item in relevant:
            grouped[item.source_description].append(item)

        claims: list[RelationshipClaim] = []
        for description, items in grouped.items():
            bases = [_basis_for(item.tags) for item in items]
            verified_sources = {item.collected_by for item, basis in zip(items, bases) if basis == RelationshipBasis.VERIFIED}
            if len(verified_sources) >= MIN_CORROBORATION_FOR_VERIFIED:
                basis = RelationshipBasis.VERIFIED
            elif RelationshipBasis.VISUALLY_DEPICTED in bases:
                basis = RelationshipBasis.VISUALLY_DEPICTED
            elif RelationshipBasis.PRESENTED_NARRATIVE in bases:
                basis = RelationshipBasis.PRESENTED_NARRATIVE
            elif RelationshipBasis.INFERRED in bases:
                basis = RelationshipBasis.INFERRED
            else:
                basis = RelationshipBasis.UNKNOWN
            claims.append(
                RelationshipClaim(
                    description=description,
                    basis=basis,
                    evidence_ids=[item.evidence_id for item in items],
                )
            )
        return claims
