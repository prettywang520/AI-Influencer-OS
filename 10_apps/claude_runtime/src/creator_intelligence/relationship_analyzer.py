"""Relationship Analyzer -- scores overall evidence coverage for
named relationships and their basis distribution. This scores *how
well-evidenced* the relationship pattern is; it does not itself
construct the basis-qualified claims (that is
HumanAuthenticityAnalyzer.extract_relationship_claims()'s job) --
kept as two separate analyzers because "is there enough evidence to
say anything" (this file) and "what exactly can be said, and on what
basis" (human_authenticity_analyzer.py) are different questions with
different failure modes.
"""
from __future__ import annotations

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "relationships"
TAGS = ("relationship",)


class RelationshipAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence="Relationship-mention coverage observed across cited evidence.",
            rationale_without_evidence="No relationship-tagged evidence supplied; confidence is unknown.",
        )
