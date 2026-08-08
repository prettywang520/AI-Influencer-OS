"""Lip Sync Analyzer -- scores evidence coverage for mouth timing,
open/close ratio, pause, smile timing, blink timing, and overall
naturalness. No facial-landmark detection of any kind -- every
observation is operator-supplied, like every other VideoEvidence item
in this package.
"""
from __future__ import annotations

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "lipsync"
TAGS = ("lipsync", "mouth_timing", "open_ratio", "close_ratio", "lip_pause", "smile_timing", "blink_timing", "naturalness")


class LipsyncAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Lip sync / facial naturalness pattern observed across cited evidence.",
            rationale_without_evidence="No lipsync-tagged evidence supplied; confidence is unknown.",
        )
