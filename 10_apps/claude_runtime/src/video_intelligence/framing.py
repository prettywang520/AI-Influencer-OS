"""Framing Analyzer -- scores evidence coverage for shot composition
and subject placement specifically (rule of thirds, headroom, symmetry).
Deliberately split from camera.py's movement/equipment focus even
though the task's own CAMERA section also mentions "Composition" --
see docs/video_intelligence/dna.md for why this is a deliberate split,
not an accidental duplicate.
"""
from __future__ import annotations

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "framing"
TAGS = ("framing", "composition", "rule_of_thirds", "headroom", "subject_placement", "symmetry")


class FramingAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Composition/framing pattern observed across cited evidence.",
            rationale_without_evidence="No framing-tagged evidence supplied; confidence is unknown.",
        )
