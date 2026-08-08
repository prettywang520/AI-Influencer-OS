"""HeadMotionAnalyzer -- DNA field head_motion (task §8). Distinguishes
a large deliberate gesture from small micro-motion via a configured
intensity band -- never conflates the two.
"""
from __future__ import annotations

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, sorted_by_timestamp, variability
from .models import TalkingAIMetricResult

TRAIT_NAME = "head_motion"

_FREEZE_THRESHOLD = 0.05
_LARGE_GESTURE_THRESHOLD = 0.60


class HeadMotionAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        own_evidence = sorted_by_timestamp(evidence_for_video(context.evidence, context.video_id))

        yaw = [item.head_yaw for item in own_evidence if item.head_yaw is not None]
        pitch = [item.head_pitch for item in own_evidence if item.head_pitch is not None]
        roll = [item.head_roll for item in own_evidence if item.head_roll is not None]
        intensity_items = [item for item in own_evidence if item.head_motion_intensity is not None]
        intensity_series = [item.head_motion_intensity for item in intensity_items]

        micro_motion_count = sum(1 for value in intensity_series if _FREEZE_THRESHOLD < value < _LARGE_GESTURE_THRESHOLD)
        large_gesture_count = sum(1 for value in intensity_series if value >= _LARGE_GESTURE_THRESHOLD)

        # longest run of consecutive near-zero-intensity observations
        freeze_duration = 0.0
        run_start: float | None = None
        for item in intensity_items:
            if item.head_motion_intensity <= _FREEZE_THRESHOLD:
                if run_start is None:
                    run_start = item.timestamp_seconds
                freeze_duration = max(freeze_duration, item.timestamp_seconds - run_start)
            else:
                run_start = None

        successive_diffs = [abs(b - a) for a, b in zip(intensity_series, intensity_series[1:])]
        motion_smoothness = None if not successive_diffs else 1.0 - min(1.0, sum(successive_diffs) / len(successive_diffs))

        # coarse repetition proxy: count of local peaks (value higher
        # than both neighbors) in the intensity series
        repetition_cycles = sum(
            1 for i in range(1, len(intensity_series) - 1)
            if intensity_series[i] > intensity_series[i - 1] and intensity_series[i] > intensity_series[i + 1]
        )

        own_segments = [segment for segment in context.speech_segments if segment.video_id == context.video_id]
        emphasis_windows = [(segment.start_seconds, segment.end_seconds) for segment in own_segments if segment.emphasis]
        pause_windows = [
            (segment.start_seconds - segment.pause_before, segment.start_seconds)
            for segment in own_segments
            if segment.pause_before is not None and segment.pause_before >= context.config.pause_min_seconds
        ]

        def _motion_ratio_near(windows: list[tuple[float, float]]) -> float | None:
            if not windows or not intensity_items:
                return None
            near = [item for item in intensity_items if any(start <= item.timestamp_seconds <= end for start, end in windows)]
            if not near:
                return None
            return sum(1 for item in near if item.head_motion_intensity > _FREEZE_THRESHOLD) / len(near)

        metrics = {
            "yaw_range": (max(yaw) - min(yaw)) if yaw else None,
            "pitch_range": (max(pitch) - min(pitch)) if pitch else None,
            "roll_range": (max(roll) - min(roll)) if roll else None,
            "yaw_variability": variability(yaw),
            "pitch_variability": variability(pitch),
            "roll_variability": variability(roll),
            "micro_motion_ratio": (micro_motion_count / len(intensity_series)) if intensity_series else None,
            "large_gesture_ratio": (large_gesture_count / len(intensity_series)) if intensity_series else None,
            "motion_during_emphasis_ratio": _motion_ratio_near(emphasis_windows),
            "motion_during_pause_ratio": _motion_ratio_near(pause_windows),
            "head_freeze_duration": freeze_duration if intensity_items else None,
            "motion_smoothness": motion_smoothness,
            "motion_repetition_cycle_count": float(repetition_cycles) if intensity_series else None,
        }

        relevant_ids = [
            item.evidence_id for item in own_evidence
            if item.head_yaw is not None or item.head_pitch is not None
            or item.head_roll is not None or item.head_motion_intensity is not None
        ]
        confidence = confidence_for(context, evidence_count=len(relevant_ids), corroboration_count=len(relevant_ids))
        score = min(1.0, len(relevant_ids) / (len(relevant_ids) + 2)) if relevant_ids else 0.0
        rationale = (
            f"Head motion pattern observed across {len(relevant_ids)} evidence item(s)."
            if relevant_ids
            else "No head-motion evidence supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME, metrics=metrics, score=score, confidence=confidence,
            sample_count=len(relevant_ids), evidence_ids=relevant_ids, rationale=rationale,
        )
