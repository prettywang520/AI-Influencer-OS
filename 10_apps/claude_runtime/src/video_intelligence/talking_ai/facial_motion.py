"""FacialMotionAnalyzer -- DNA field facial_motion (task §9). Consumes
observations only -- no facial landmarks or computer vision required
or performed in this phase. `evidence.py`'s schema (task's own §1)
does not model eyebrow/cheek motion fields; metrics for those stay
structurally None (unavailable), never fabricated, rather than
inventing evidence fields beyond the approved schema.
"""
from __future__ import annotations

from statistics import mean

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, sorted_by_timestamp
from .evidence import FacialExpression
from .models import TalkingAIMetricResult
from .timing import speech_active_intervals

TRAIT_NAME = "facial_motion"

_FROZEN_INTENSITY_EPSILON = 0.05


class FacialMotionAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        own_evidence = sorted_by_timestamp(evidence_for_video(context.evidence, context.video_id))
        expression_evidence = [item for item in own_evidence if item.facial_expression is not None]
        smile_evidence = [item for item in own_evidence if item.smile_intensity is not None]
        smile_series = [item.smile_intensity for item in smile_evidence]

        # runs of consecutive same-expression observations
        runs: list[tuple[str, float, float]] = []
        for item in expression_evidence:
            if runs and runs[-1][0] == item.facial_expression:
                runs[-1] = (runs[-1][0], runs[-1][1], item.timestamp_seconds)
            else:
                runs.append((item.facial_expression, item.timestamp_seconds, item.timestamp_seconds))
        expression_durations = [end - start for _expr, start, end in runs]
        expression_change_count = max(0, len(runs) - 1)

        frozen_face_duration = max(
            (end - start for expr, start, end in runs if expr == FacialExpression.NEUTRAL), default=None,
        )

        smile_onsets = smile_offsets = 0
        was_smiling = False
        for item in smile_evidence:
            is_smiling = item.smile_intensity >= 0.3
            if is_smiling and not was_smiling:
                smile_onsets += 1
            elif was_smiling and not is_smiling:
                smile_offsets += 1
            was_smiling = is_smiling

        repeated_expression_cycles = sum(
            1 for i in range(1, len(smile_series) - 1)
            if smile_series[i] > smile_series[i - 1] and smile_series[i] > smile_series[i + 1]
        )

        own_segments = [segment for segment in context.speech_segments if segment.video_id == context.video_id]
        speech_intervals = speech_active_intervals(own_segments)
        change_timestamps = [run[1] for run in runs[1:]]
        changes_during_speech = sum(
            1 for ts in change_timestamps if any(start <= ts <= end for start, end in speech_intervals)
        )

        metrics = {
            "smile_onset_count": float(smile_onsets),
            "smile_offset_count": float(smile_offsets),
            "expression_change_count": float(expression_change_count),
            "expression_persistence": mean(expression_durations) if expression_durations else None,
            "frozen_face_duration": frozen_face_duration,
            "repeated_expression_cycle_count": float(repeated_expression_cycles) if smile_series else None,
            "expression_speech_timing_ratio": (
                changes_during_speech / len(change_timestamps) if change_timestamps and speech_intervals else None
            ),
            "eyebrow_movement": None,  # not modeled in this evidence schema version
            "cheek_movement": None,    # not modeled in this evidence schema version
        }

        relevant_ids = sorted({item.evidence_id for item in expression_evidence} | {item.evidence_id for item in smile_evidence})
        confidence = confidence_for(context, evidence_count=len(relevant_ids), corroboration_count=len(relevant_ids))
        score = min(1.0, len(relevant_ids) / (len(relevant_ids) + 2)) if relevant_ids else 0.0
        rationale = (
            f"Facial micro-motion observed across {len(relevant_ids)} evidence item(s) "
            "(eyebrow/cheek motion not modeled in this evidence schema version)."
            if relevant_ids
            else "No facial-expression evidence supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME, metrics=metrics, score=score, confidence=confidence,
            sample_count=len(relevant_ids), evidence_ids=relevant_ids, rationale=rationale,
        )
