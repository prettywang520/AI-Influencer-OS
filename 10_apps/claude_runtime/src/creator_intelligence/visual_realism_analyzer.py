"""Visual Realism Analyzer -- scores evidence coverage for lighting,
composition, and editing style. Deliberately never asserts a real
camera/device model as fact: any device-adjacent observation is
phrased as a described stylistic impression only. This constraint is
enforced by construction (CAMERA_MODEL_DISCLAIMER is always the
rationale prefix), not left to a future author's discretion.
"""
from __future__ import annotations

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "visual_realism"
TAGS = ("visual_realism",)

CAMERA_MODEL_DISCLAIMER = (
    "Describes stylistic impression only (lighting/composition/editing); "
    "does not assert an actual camera or device model."
)


class VisualRealismAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence=f"{CAMERA_MODEL_DISCLAIMER} Pattern observed across cited evidence.",
            rationale_without_evidence=f"{CAMERA_MODEL_DISCLAIMER} No evidence supplied; confidence is unknown.",
        )
