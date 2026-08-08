"""FramingAnalyzer -- DNA field framing (task §14). Adapts (does not
import) video_intelligence.camera's config-vocabulary shape:
`camera_motion` (an evidence.py closed vocabulary, mirroring
camera.yaml's equipment_types) directly covers tripod/handheld/drift/
punch-in/tracking. `evidence.py`'s schema (task's own §1) has no
distance/shot-type field (close-up/chest-up/waist-up/full-body/
selfie/direct-to-camera) or face-position field (headroom, face-center
stability) -- those metrics stay structurally None (unavailable),
never fabricated, mirroring facial_motion.py's/subtitle_sync.py's own
honesty about schema limitations.
"""
from __future__ import annotations

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, sorted_by_timestamp
from .evidence import CameraMotionType
from .models import TalkingAIMetricResult
from .timing import speech_active_intervals

TRAIT_NAME = "framing"


class FramingAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        own_evidence = sorted_by_timestamp(evidence_for_video(context.evidence, context.video_id))
        motion_evidence = [item for item in own_evidence if item.camera_motion is not None]
        boundary_events = sorted(item.timestamp_seconds for item in own_evidence if item.shot_boundary)

        def _ratio_of(motion_type: str) -> float | None:
            if not motion_evidence:
                return None
            return sum(1 for item in motion_evidence if item.camera_motion == motion_type) / len(motion_evidence)

        own_segments = [segment for segment in context.speech_segments if segment.video_id == context.video_id]
        speech_intervals = speech_active_intervals(own_segments)
        shot_changes_during_speech = sum(
            1 for ts in boundary_events if any(start <= ts <= end for start, end in speech_intervals)
        )

        tripod_ratio = _ratio_of(CameraMotionType.STATIC)
        shot_boundary_count = len(boundary_events)
        framing_stability = (
            None if tripod_ratio is None
            else max(0.0, tripod_ratio - min(1.0, shot_boundary_count / max(1, len(motion_evidence))))
        )

        metrics = {
            "tripod_ratio": tripod_ratio,
            "handheld_ratio": _ratio_of(CameraMotionType.HANDHELD),
            "drift_ratio": _ratio_of(CameraMotionType.DRIFT),
            "punch_in_ratio": _ratio_of(CameraMotionType.PUNCH_IN),
            "tracking_ratio": _ratio_of(CameraMotionType.TRACKING),
            "shot_boundary_count": float(shot_boundary_count),
            "shot_change_during_speech_ratio": (
                shot_changes_during_speech / len(boundary_events) if boundary_events else None
            ),
            "framing_stability": framing_stability,
            "shot_distance_type": None,   # not modeled in this evidence schema version
            "headroom": None,             # not modeled in this evidence schema version
            "face_center_stability": None,  # not modeled in this evidence schema version
        }

        relevant_ids = sorted({item.evidence_id for item in motion_evidence} | {item.evidence_id for item in own_evidence if item.shot_boundary})
        confidence = confidence_for(context, evidence_count=len(relevant_ids), corroboration_count=len(relevant_ids))
        score = min(1.0, len(relevant_ids) / (len(relevant_ids) + 2)) if relevant_ids else 0.0
        rationale = (
            f"Talking-specific framing observed across {len(relevant_ids)} evidence item(s) "
            "(shot-distance/headroom/face-center metrics not modeled in this evidence schema version)."
            if relevant_ids
            else "No camera-motion evidence supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME, metrics=metrics, score=score, confidence=confidence,
            sample_count=len(relevant_ids), evidence_ids=relevant_ids, rationale=rationale,
        )
