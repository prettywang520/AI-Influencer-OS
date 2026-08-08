"""Gesture Analyzer -- scores evidence coverage for hand movement,
finger pointing, head motion, eye contact, and body movement (walking/
standing/sitting).
"""
from __future__ import annotations

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "gesture"
TAGS = ("gesture", "hand_movement", "finger_pointing", "head_motion", "eye_contact", "body_movement", "walking", "standing", "sitting")


class GestureAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Gesture/body-movement pattern observed across cited evidence.",
            rationale_without_evidence="No gesture-tagged evidence supplied; confidence is unknown.",
        )
