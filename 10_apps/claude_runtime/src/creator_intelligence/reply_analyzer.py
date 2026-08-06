"""Reply Analyzer -- scores evidence coverage for reply tone,
response length, emoji usage, and response cadence, based on evidence
*about* a creator's past replies. This analyzer never generates a
reply itself -- it has no method that produces reply text, and its
TraitScore carries only a rationale describing the observed pattern,
never a draft response. Real reply generation for Aiko's own account
remains owned exclusively by the existing production reply-generation
modules, which this module never imports.
"""
from __future__ import annotations

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "replies"
TAGS = ("reply",)


class ReplyAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence="Reply tone/cadence pattern observed across cited evidence.",
            rationale_without_evidence="No reply-tagged evidence supplied; confidence is unknown.",
        )
