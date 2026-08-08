"""BlinkAnalyzer -- DNA field blink (task §6). Detects patterns that
may look mechanically regular or unnaturally absent -- never
diagnoses a medical condition; only reports observed timing statistics.
"""
from __future__ import annotations

from statistics import mean

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, intervals, variability
from .models import TalkingAIMetricResult

TRAIT_NAME = "blink"

_DOUBLE_BLINK_WINDOW_SECONDS = 1.0
_LONG_NO_BLINK_THRESHOLD_SECONDS = 8.0
_BOUNDARY_PROXIMITY_SECONDS = 0.5


class BlinkAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        own_evidence = evidence_for_video(context.evidence, context.video_id)
        blink_evidence = [item for item in own_evidence if item.blink is not None]
        blink_timestamps = sorted(item.timestamp_seconds for item in blink_evidence if item.blink)

        blink_intervals = intervals(blink_timestamps)
        span = (blink_timestamps[-1] - blink_timestamps[0]) if len(blink_timestamps) >= 2 else None

        double_blinks = sum(1 for gap in blink_intervals if gap <= _DOUBLE_BLINK_WINDOW_SECONDS)
        long_no_blink = sum(1 for gap in blink_intervals if gap >= _LONG_NO_BLINK_THRESHOLD_SECONDS)

        own_segments = [segment for segment in context.speech_segments if segment.video_id == context.video_id]
        pause_windows = [
            (segment.start_seconds - segment.pause_before, segment.start_seconds)
            for segment in own_segments
            if segment.pause_before is not None and segment.pause_before >= context.config.pause_min_seconds
        ]
        boundaries = sorted({segment.start_seconds for segment in own_segments} | {segment.end_seconds for segment in own_segments})

        blinks_near_pause = sum(
            1 for ts in blink_timestamps if any(start <= ts <= end for start, end in pause_windows)
        )
        blinks_near_boundary = sum(
            1 for ts in blink_timestamps if any(abs(ts - boundary) <= _BOUNDARY_PROXIMITY_SECONDS for boundary in boundaries)
        )

        metrics = {
            "blink_count": float(len(blink_timestamps)),
            "blink_frequency": (len(blink_timestamps) / span) if span else None,
            "average_blink_interval": mean(blink_intervals) if blink_intervals else None,
            "blink_interval_variability": variability(blink_intervals),
            "double_blink_frequency": (double_blinks / len(blink_intervals)) if blink_intervals else None,
            "long_no_blink_interval_count": float(long_no_blink),
            "blink_timing_relative_to_pause_ratio": (
                blinks_near_pause / len(blink_timestamps) if blink_timestamps and pause_windows else None
            ),
            "blink_timing_relative_to_sentence_boundary_ratio": (
                blinks_near_boundary / len(blink_timestamps) if blink_timestamps and boundaries else None
            ),
        }

        evidence_count = len(blink_evidence)
        confidence = confidence_for(context, evidence_count=evidence_count, corroboration_count=evidence_count)
        score = min(1.0, evidence_count / (evidence_count + 2)) if evidence_count else 0.0
        rationale = (
            f"Blink pattern observed across {evidence_count} evidence item(s)."
            if evidence_count
            else "No blink evidence supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME, metrics=metrics, score=score, confidence=confidence,
            sample_count=evidence_count, evidence_ids=[item.evidence_id for item in blink_evidence], rationale=rationale,
        )
