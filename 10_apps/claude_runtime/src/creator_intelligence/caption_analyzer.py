"""Caption Analyzer -- scores evidence coverage for caption tone,
length pattern, hashtag usage, and call-to-action style. Stores only
short excerpts (never full-text reproduction), copyright-respecting by
construction: any evidence excerpt over the configured character cap
is flagged as a warning here, and rejected outright by validation.py.
"""
from __future__ import annotations

from .analyzer_base import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags

TRAIT_NAME = "captions"
TAGS = ("caption",)


class CaptionAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        result = analyze_by_tag(
            context,
            trait_name=TRAIT_NAME,
            tags=TAGS,
            rationale_with_evidence="Caption tone/length pattern observed across cited evidence.",
            rationale_without_evidence="No caption-tagged evidence supplied; confidence is unknown.",
        )
        cap = context.config.caption_excerpt_max_chars
        for item in evidence_for_tags(context.evidence, TAGS):
            if len(item.content_excerpt) > cap:
                result.warnings.append(
                    f"Evidence {item.evidence_id} excerpt exceeds {cap} chars; must be shortened before use."
                )
        return result
