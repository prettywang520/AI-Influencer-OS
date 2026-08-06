"""Reels Analyzer -- scores evidence coverage for hook style, pacing,
editing cadence, and audio usage patterns observed in a creator's
reels.
"""
from __future__ import annotations

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "reels"
TAGS = ("reels",)


class ReelsAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence="Reels pacing/hook pattern observed across cited evidence.",
            rationale_without_evidence="No reels-tagged evidence supplied; confidence is unknown.",
        )
