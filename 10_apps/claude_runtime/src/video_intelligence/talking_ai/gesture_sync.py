"""GestureSyncAnalyzer -- DNA field gesture_sync (task §10). Hand/body
gesture timing relative to speech emphasis. Generalized pattern
descriptions (for the knowledge base) are built from counted
categories only -- unique source choreography is never preserved
verbatim.
"""
from __future__ import annotations

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, sorted_by_timestamp
from .models import TalkingAIMetricResult

TRAIT_NAME = "gesture_sync"

_GESTURE_MOTION_THRESHOLD = 0.15
_OVERACTIVITY_THRESHOLD = 0.70
_UNDERACTIVITY_THRESHOLD = 0.05
_ONSET_SEARCH_WINDOW_SECONDS = 0.5


def _gesture_intensity(item) -> float | None:
    values = [value for value in (item.left_hand_motion, item.right_hand_motion) if value is not None]
    return max(values) if values else None


class GestureSyncAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        own_evidence = sorted_by_timestamp(evidence_for_video(context.evidence, context.video_id))
        intensity_items = [(item, _gesture_intensity(item)) for item in own_evidence]
        intensity_items = [(item, value) for item, value in intensity_items if value is not None]
        gesture_events = [item.timestamp_seconds for item, value in intensity_items if value >= _GESTURE_MOTION_THRESHOLD]

        gesture_types = {item.gesture_type for item in own_evidence if item.gesture_type is not None}

        own_segments = [segment for segment in context.speech_segments if segment.video_id == context.video_id]
        emphasis_windows = [(segment.start_seconds, segment.end_seconds) for segment in own_segments if segment.emphasis]

        onset_hits = peak_hits = end_hits = 0
        for start, end in emphasis_windows:
            if any(start - _ONSET_SEARCH_WINDOW_SECONDS <= ts < start for ts in gesture_events):
                onset_hits += 1
            if any(start <= ts <= end for ts in gesture_events):
                peak_hits += 1
            if any(end < ts <= end + _ONSET_SEARCH_WINDOW_SECONDS for ts in gesture_events):
                end_hits += 1

        onset_ratio = (onset_hits / len(emphasis_windows)) if emphasis_windows else None
        peak_ratio = (peak_hits / len(emphasis_windows)) if emphasis_windows else None
        end_ratio = (end_hits / len(emphasis_windows)) if emphasis_windows else None
        coherence_terms = [ratio for ratio in (onset_ratio, peak_ratio, end_ratio) if ratio is not None]
        coherence = (sum(coherence_terms) / len(coherence_terms)) if coherence_terms else None

        span = (
            intensity_items[-1][0].timestamp_seconds - intensity_items[0][0].timestamp_seconds
            if len(intensity_items) >= 2 else None
        )
        overactive = sum(1 for _item, value in intensity_items if value >= _OVERACTIVITY_THRESHOLD)
        underactive = sum(1 for _item, value in intensity_items if value <= _UNDERACTIVITY_THRESHOLD)

        pause_windows = [
            (segment.start_seconds - segment.pause_before, segment.start_seconds)
            for segment in own_segments
            if segment.pause_before is not None and segment.pause_before >= context.config.pause_min_seconds
        ]
        pause_evidenced = [
            value for item, value in intensity_items if any(start <= item.timestamp_seconds <= end for start, end in pause_windows)
        ]

        intensity_series = [value for _item, value in intensity_items]
        repetition_cycles = sum(
            1 for i in range(1, len(intensity_series) - 1)
            if intensity_series[i] > intensity_series[i - 1] and intensity_series[i] > intensity_series[i + 1]
        )

        metrics = {
            "gesture_onset_before_emphasis": onset_ratio,
            "gesture_peak_alignment": peak_ratio,
            "gesture_end_alignment": end_ratio,
            "gesture_frequency": (len(gesture_events) / span) if span else None,
            "gesture_variety": float(len(gesture_types)) if gesture_types else None,
            "gesture_repetition": float(repetition_cycles) if intensity_series else None,
            "gesture_overactivity": (overactive / len(intensity_items)) if intensity_items else None,
            "gesture_underactivity": (underactive / len(intensity_items)) if intensity_items else None,
            "gesture_pause_behavior": (
                sum(1 for value in pause_evidenced if value < _GESTURE_MOTION_THRESHOLD) / len(pause_evidenced)
                if pause_evidenced else None
            ),
            "speech_gesture_coherence": coherence,
        }

        relevant_ids = [item.evidence_id for item, _value in intensity_items]
        confidence = confidence_for(context, evidence_count=len(relevant_ids), corroboration_count=len(relevant_ids))
        score = min(1.0, len(relevant_ids) / (len(relevant_ids) + 2)) if relevant_ids else 0.0
        rationale = (
            f"Gesture-speech synchronization observed across {len(relevant_ids)} evidence item(s)."
            if relevant_ids
            else "No gesture-motion evidence supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME, metrics=metrics, score=score, confidence=confidence,
            sample_count=len(relevant_ids), evidence_ids=relevant_ids, rationale=rationale,
        )
