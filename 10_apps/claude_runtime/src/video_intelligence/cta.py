"""CTA Analyzer -- scores evidence coverage for calls-to-action
(follow/comment/share/save/question) and extracts timed
CTAObservation records, one per CTA-typed evidence item.
"""
from __future__ import annotations

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags, evidence_for_video
from .models import CTAObservation, CTAType

TRAIT_NAME = "cta"
# Deliberately narrow: does NOT include CTAType.ALL's raw values
# ("follow"/"comment"/"share"/"save"/"question") as top-level scoping
# tags -- those words are generic enough to collide with unrelated
# domains (e.g. a hook's "question" hook_type). Evidence must carry
# an explicit "cta"/"cta_timing" domain tag to be considered CTA
# evidence at all; CTAType.ALL is only used *within* that already-
# scoped subset (see extract_cta_observations()) to pick a cta_type.
TAGS = ("cta", "cta_timing")


class CTAAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Call-to-action pattern observed across cited evidence.",
            rationale_without_evidence="No CTA-tagged evidence supplied; confidence is unknown.",
        )

    def extract_cta_observations(self, context: AnalyzerContext) -> list[CTAObservation]:
        own_video_evidence = evidence_for_video(context.evidence, context.video_id)
        relevant = evidence_for_tags(own_video_evidence, TAGS)
        observations: list[CTAObservation] = []
        for item in relevant:
            cta_type = next((cta for cta in CTAType.ALL if cta in item.tags), None)
            if cta_type is None:
                continue
            observations.append(
                CTAObservation(cta_type=cta_type, timing_seconds=item.timestamp_seconds, evidence_ids=[item.evidence_id])
            )
        return observations
