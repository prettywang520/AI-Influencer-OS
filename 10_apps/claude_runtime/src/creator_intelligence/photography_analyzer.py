"""Photography Analyzer -- scores evidence coverage for framing,
depth-of-field impression, setting variety, and cross-post
consistency. Descriptive only, same evidence-based scoring convention
as every other analyzer in this framework.
"""
from __future__ import annotations

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "photography"
TAGS = ("photography",)


class PhotographyAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence="Framing/setting pattern observed across cited evidence.",
            rationale_without_evidence="No photography-tagged evidence supplied; confidence is unknown.",
        )
