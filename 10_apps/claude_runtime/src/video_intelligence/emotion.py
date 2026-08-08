"""Emotion Analyzer -- scores evidence coverage for primary emotion,
emotion changes, intensity, and authenticity.
"""
from __future__ import annotations

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "emotion"
TAGS = ("emotion", "primary_emotion", "emotion_change", "intensity", "authenticity")


class EmotionAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Emotion pattern observed across cited evidence.",
            rationale_without_evidence="No emotion-tagged evidence supplied; confidence is unknown.",
        )
