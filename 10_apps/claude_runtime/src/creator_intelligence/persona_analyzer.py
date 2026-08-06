"""Persona Analyzer -- scores evidence coverage for a creator's
observable tone, values, recurring themes, and self-presentation.
Describes patterns only; never asserts an internal motive or trait
the evidence doesn't directly show.
"""
from __future__ import annotations

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "persona"
TAGS = ("persona",)


class PersonaAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence="Persona tone/values pattern observed across cited evidence.",
            rationale_without_evidence="No persona-tagged evidence supplied; confidence is unknown.",
        )
