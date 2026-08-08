"""SubtitleSyncAnalyzer -- DNA field subtitle_sync (task §13).
`evidence.py`'s schema (task's own §1) has no dedicated
highlight-timing field; `highlight_alignment` stays structurally None
(unavailable), never fabricated, mirroring facial_motion.py's own
honesty about schema limitations.
"""
from __future__ import annotations

from statistics import mean

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, ratio_true, score_from_ratio, sorted_by_timestamp
from .models import TalkingAIMetricResult
from .timing import nearest_event_after, speech_active_intervals

TRAIT_NAME = "subtitle_sync"


class SubtitleSyncAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        own_evidence = sorted_by_timestamp(evidence_for_video(context.evidence, context.video_id))
        own_segments = [segment for segment in context.speech_segments if segment.video_id == context.video_id]
        speech_intervals = speech_active_intervals(own_segments)

        visible_evidence = [item for item in own_evidence if item.subtitle_visible is not None]
        change_events = sorted(item.timestamp_seconds for item in own_evidence if item.subtitle_change)

        max_window = context.config.maximum_reasonable_sync_latency_seconds
        search_window = max(max_window * 4, 1.0)

        latencies: list[float] = []
        onset_hits = onset_checked = 0
        for start, _end in speech_intervals:
            onset_checked += 1
            nearest = nearest_event_after(start, change_events, max_window=search_window)
            if nearest is not None:
                latency = nearest - start
                latencies.append(latency)
                if latency <= max_window:
                    onset_hits += 1

        span = (change_events[-1] - change_events[0]) if len(change_events) >= 2 else None

        metrics = {
            "speech_subtitle_latency": mean(latencies) if latencies else None,
            "subtitle_segment_alignment": (onset_hits / onset_checked) if onset_checked else None,
            "highlight_alignment": None,  # not modeled in this evidence schema version
            "subtitle_density": (len(change_events) / span) if span else None,
            "subtitle_visible_ratio": ratio_true([item.subtitle_visible for item in visible_evidence]),
            "subtitle_pacing_naturalness": score_from_ratio(onset_checked, onset_hits) if onset_checked else None,
        }

        relevant_ids = sorted(
            {item.evidence_id for item in visible_evidence} | {item.evidence_id for item in own_evidence if item.subtitle_change}
        )
        confidence = confidence_for(context, evidence_count=len(relevant_ids), corroboration_count=len(relevant_ids))
        score = metrics["subtitle_pacing_naturalness"] or 0.0
        rationale = (
            f"Subtitle-speech synchronization observed across {len(relevant_ids)} evidence item(s)."
            if relevant_ids
            else "No subtitle evidence supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME, metrics=metrics, score=score, confidence=confidence,
            sample_count=len(relevant_ids), evidence_ids=relevant_ids, rationale=rationale,
        )
