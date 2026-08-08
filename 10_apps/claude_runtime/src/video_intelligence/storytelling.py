"""Storytelling Analyzer -- scores evidence coverage for narrative
structure (beginning/middle/end, story arc, problem/solution/payoff)
and extracts evidence-qualified StoryBeat observations. Mirrors
creator_intelligence.human_authenticity_analyzer's
extract_relationship_claims() dual-output shape.
"""
from __future__ import annotations

from collections import defaultdict

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags, evidence_for_video
from .models import BeatType, StoryBeat

TRAIT_NAME = "storytelling"
# Deliberately narrow: does NOT include BeatType.ALL's raw values
# ("beginning"/"middle"/"end"/"problem"/"solution"/"payoff") as
# top-level scoping tags -- those words are generic enough to collide
# with unrelated domains. Evidence must carry an explicit
# "storytelling"/"story_arc" domain tag to be considered storytelling
# evidence at all; BeatType.ALL is only used *within* that already-
# scoped subset (see extract_story_beats()) to pick a beat_type.
TAGS = ("storytelling", "story_arc")


class StorytellingAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Narrative-structure pattern observed across cited evidence.",
            rationale_without_evidence="No storytelling-tagged evidence supplied; confidence is unknown.",
        )

    def extract_story_beats(self, context: AnalyzerContext) -> list[StoryBeat]:
        """Groups beat-tagged evidence by (beat_type, source_description)
        -- a beat's description is treated as its distinguishing
        identity, so repeated observations of the same beat accumulate
        evidence_ids rather than producing duplicate StoryBeats."""
        own_video_evidence = evidence_for_video(context.evidence, context.video_id)
        relevant = evidence_for_tags(own_video_evidence, TAGS)
        grouped: dict[tuple[str, str], list] = defaultdict(list)
        for item in relevant:
            beat_type = next((beat for beat in BeatType.ALL if beat in item.tags), None)
            if beat_type is None:
                continue
            grouped[(beat_type, item.source_description)].append(item)

        beats: list[StoryBeat] = []
        for (beat_type, description), items in grouped.items():
            beats.append(
                StoryBeat(beat_type=beat_type, description=description, evidence_ids=[item.evidence_id for item in items])
            )
        return beats
