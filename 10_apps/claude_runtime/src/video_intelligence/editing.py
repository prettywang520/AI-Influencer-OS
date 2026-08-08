"""Editing Analyzer -- scores evidence coverage for concrete editing
technique: shot duration, average cut, fast/slow cut, zoom, motion,
transition, jump cut. Complementary to pacing.py's overall tempo/
rhythm "meta" trait -- see docs/video_intelligence/dna.md.
"""
from __future__ import annotations

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "editing"
TAGS = ("editing", "shot_duration", "average_cut", "fast_cut", "slow_cut", "zoom", "motion", "transition", "jump_cut")


class EditingAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Editing technique pattern observed across cited evidence.",
            rationale_without_evidence="No editing-tagged evidence supplied; confidence is unknown.",
        )
