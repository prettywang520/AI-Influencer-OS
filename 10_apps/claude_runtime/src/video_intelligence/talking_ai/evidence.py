"""TalkingAIEvidence and SpeechSegment -- the two evidence shapes this
package analyzes. TalkingAIEvidence is a *simultaneous snapshot* at a
timestamp (mouth/blink/gaze/head/gesture/subtitle state all
co-occurring), fundamentally different from
video_intelligence.evidence.VideoEvidence's single tagged excerpt --
judging synchronization/naturalness requires knowing what multiple
channels were doing at the same instant, not a flat list of
independently-tagged observations. `video_id` is reused directly from
video_intelligence.evidence.video_id_for() -- no second identity
scheme. Every field defaults to None/empty; nothing is ever
fabricated, and a None field is always "not observed," never "false."
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.video_intelligence.models import ConfidenceLevel

from .exceptions import InvalidTalkingAIEvidenceError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(payload: dict[str, Any], *, length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


class MouthShapeCategory:
    CLOSED = "closed"
    SLIGHTLY_OPEN = "slightly_open"
    OPEN = "open"
    WIDE_OPEN = "wide_open"
    ROUNDED = "rounded"
    OTHER = "other"

    ALL = (CLOSED, SLIGHTLY_OPEN, OPEN, WIDE_OPEN, ROUNDED, OTHER)


class GazeDirection:
    DIRECT_CAMERA = "direct_camera"
    AWAY_LEFT = "away_left"
    AWAY_RIGHT = "away_right"
    UP = "up"
    DOWN = "down"
    CLOSED_EYES = "closed_eyes"
    OTHER = "other"

    ALL = (DIRECT_CAMERA, AWAY_LEFT, AWAY_RIGHT, UP, DOWN, CLOSED_EYES, OTHER)


class GestureType:
    HAND_RAISE = "hand_raise"
    POINT = "point"
    OPEN_PALM = "open_palm"
    WAVE = "wave"
    CLASP = "clasp"
    STILL = "still"
    OTHER = "other"

    ALL = (HAND_RAISE, POINT, OPEN_PALM, WAVE, CLASP, STILL, OTHER)


class FacialExpression:
    NEUTRAL = "neutral"
    SMILE = "smile"
    SLIGHT_SMILE = "slight_smile"
    SURPRISE = "surprise"
    CONCERN = "concern"
    OTHER = "other"

    ALL = (NEUTRAL, SMILE, SLIGHT_SMILE, SURPRISE, CONCERN, OTHER)


class CameraMotionType:
    STATIC = "static"
    HANDHELD = "handheld"
    DRIFT = "drift"
    PAN = "pan"
    PUNCH_IN = "punch_in"
    ZOOM = "zoom"
    TRACKING = "tracking"
    OTHER = "other"

    ALL = (STATIC, HANDHELD, DRIFT, PAN, PUNCH_IN, ZOOM, TRACKING, OTHER)


class ArtifactType:
    """§16 -- observed artifact labels. Never a basis for an automatic
    AI-generation conclusion (naturalness.py enforces this)."""

    MOUTH_EDGE_INSTABILITY = "mouth_edge_instability"
    TEETH_INSTABILITY = "teeth_instability"
    LIP_TEXTURE_SHIFT = "lip_texture_shift"
    FACE_TEXTURE_TEMPORAL_SHIFT = "face_texture_temporal_shift"
    BLINK_ASYMMETRY = "blink_asymmetry"
    FROZEN_EYE_REGION = "frozen_eye_region"
    GESTURE_LOOP = "gesture_loop"
    HEAD_MOTION_LOOP = "head_motion_loop"
    BACKGROUND_TEMPORAL_INSTABILITY = "background_temporal_instability"
    HAIR_TEMPORAL_INSTABILITY = "hair_temporal_instability"
    ACCESSORY_TEMPORAL_INSTABILITY = "accessory_temporal_instability"

    ALL = (
        MOUTH_EDGE_INSTABILITY, TEETH_INSTABILITY, LIP_TEXTURE_SHIFT, FACE_TEXTURE_TEMPORAL_SHIFT,
        BLINK_ASYMMETRY, FROZEN_EYE_REGION, GESTURE_LOOP, HEAD_MOTION_LOOP,
        BACKGROUND_TEMPORAL_INSTABILITY, HAIR_TEMPORAL_INSTABILITY, ACCESSORY_TEMPORAL_INSTABILITY,
    )


class TalkingLanguage:
    CANTONESE = "cantonese"
    MANDARIN = "mandarin"
    ENGLISH = "english"
    JAPANESE = "japanese"
    MIXED = "mixed"
    OTHER = "other"

    ALL = (CANTONESE, MANDARIN, ENGLISH, JAPANESE, MIXED, OTHER)


def _validate_choice(value: str | None, allowed: tuple[str, ...], field_name: str) -> None:
    if value is not None and value not in allowed:
        raise InvalidTalkingAIEvidenceError(f"{field_name} must be one of {allowed}, got {value!r}")


@dataclass(slots=True)
class TalkingAIEvidence:
    """One simultaneous snapshot at a timestamp. Evidence may be
    sparse -- only the fields actually observed need be set; every
    other field stays None ("not observed"), never guessed as False/0."""

    video_id: str
    timestamp_seconds: float
    speech_active: bool | None = None
    mouth_open_ratio: float | None = None
    mouth_shape_category: str | None = None
    jaw_motion: float | None = None
    lip_motion_intensity: float | None = None
    blink: bool | None = None
    eye_contact: bool | None = None
    gaze_direction: str | None = None
    head_yaw: float | None = None
    head_pitch: float | None = None
    head_roll: float | None = None
    head_motion_intensity: float | None = None
    facial_expression: str | None = None
    smile_intensity: float | None = None
    left_hand_motion: float | None = None
    right_hand_motion: float | None = None
    gesture_type: str | None = None
    body_motion: float | None = None
    subtitle_visible: bool | None = None
    subtitle_change: bool | None = None
    camera_motion: str | None = None
    shot_boundary: bool | None = None
    artifact_tags: list[str] = field(default_factory=list)
    notes: str = ""
    collected_at: str = field(default_factory=_now_iso)
    collected_by: str = "operator"
    confidence: str = ConfidenceLevel.UNKNOWN
    evidence_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        _validate_choice(self.mouth_shape_category, MouthShapeCategory.ALL, "mouth_shape_category")
        _validate_choice(self.gaze_direction, GazeDirection.ALL, "gaze_direction")
        _validate_choice(self.gesture_type, GestureType.ALL, "gesture_type")
        _validate_choice(self.facial_expression, FacialExpression.ALL, "facial_expression")
        _validate_choice(self.camera_motion, CameraMotionType.ALL, "camera_motion")
        for tag in self.artifact_tags:
            if tag not in ArtifactType.ALL:
                raise InvalidTalkingAIEvidenceError(f"artifact_tags entry must be one of {ArtifactType.ALL}, got {tag!r}")
        self.evidence_id = _content_hash(
            {
                "video_id": self.video_id,
                "timestamp_seconds": self.timestamp_seconds,
                "speech_active": self.speech_active,
                "mouth_open_ratio": self.mouth_open_ratio,
                "mouth_shape_category": self.mouth_shape_category,
                "jaw_motion": self.jaw_motion,
                "lip_motion_intensity": self.lip_motion_intensity,
                "blink": self.blink,
                "eye_contact": self.eye_contact,
                "gaze_direction": self.gaze_direction,
                "head_yaw": self.head_yaw,
                "head_pitch": self.head_pitch,
                "head_roll": self.head_roll,
                "head_motion_intensity": self.head_motion_intensity,
                "facial_expression": self.facial_expression,
                "smile_intensity": self.smile_intensity,
                "left_hand_motion": self.left_hand_motion,
                "right_hand_motion": self.right_hand_motion,
                "gesture_type": self.gesture_type,
                "body_motion": self.body_motion,
                "subtitle_visible": self.subtitle_visible,
                "subtitle_change": self.subtitle_change,
                "camera_motion": self.camera_motion,
                "shot_boundary": self.shot_boundary,
                "artifact_tags": sorted(self.artifact_tags),
                "notes": self.notes,
                "collected_by": self.collected_by,
            }
        )


@dataclass(slots=True)
class SpeechSegment:
    """A coarser-grained, speech-level (not frame-level) fact.
    `text_optional` is never auto-transcribed -- always operator- (or,
    in a future phase, upstream-) supplied, exactly like every other
    text excerpt in this codebase."""

    video_id: str
    start_seconds: float
    end_seconds: float
    language: str | None = None
    text_optional: str | None = None
    speech_rate_optional: float | None = None
    energy: str | None = None
    pause_before: float | None = None
    pause_after: float | None = None
    question_like: bool | None = None
    emphasis: bool | None = None
    sentence_final_particle: bool | None = None
    code_switching: bool | None = None
    confidence: str = ConfidenceLevel.UNKNOWN
    metadata: dict = field(default_factory=dict)
    segment_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        _validate_choice(self.language, TalkingLanguage.ALL, "language")
        if self.end_seconds < self.start_seconds:
            raise InvalidTalkingAIEvidenceError(
                f"end_seconds ({self.end_seconds}) must not be before start_seconds ({self.start_seconds})"
            )
        self.segment_id = _content_hash(
            {
                "video_id": self.video_id,
                "start_seconds": self.start_seconds,
                "end_seconds": self.end_seconds,
                "language": self.language,
                "text_optional": self.text_optional,
                "speech_rate_optional": self.speech_rate_optional,
                "energy": self.energy,
                "pause_before": self.pause_before,
                "pause_after": self.pause_after,
                "question_like": self.question_like,
                "emphasis": self.emphasis,
                "sentence_final_particle": self.sentence_final_particle,
                "code_switching": self.code_switching,
            }
        )
