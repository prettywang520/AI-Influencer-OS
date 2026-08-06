"""Storytelling Analyzer -- scores evidence coverage for narrative
arc patterns, recurring motifs, and pacing across a creator's content.
"""
from __future__ import annotations

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "storytelling"
TAGS = ("storytelling",)


class StorytellingAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence="Narrative-arc pattern observed across cited evidence.",
            rationale_without_evidence="No storytelling-tagged evidence supplied; confidence is unknown.",
        )
