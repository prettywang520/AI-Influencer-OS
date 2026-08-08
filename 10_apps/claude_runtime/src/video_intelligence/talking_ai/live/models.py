"""LiveTalkingReelRecord (task §2) -- one Reel's live-research evidence
collection, before bridging into TalkingAIEvidence/SpeechSegment
(adapter.py owns that step). Also LiveAdapterConfig (loaded from
config/video_intelligence/talking_ai_live.yaml, see
workflow.load_talking_ai_live_config()) and
compute_live_completeness() (task §16).

`timeline_observations`/`speech_segments` are native, already-typed
TalkingAIEvidence/SpeechSegment objects reused directly from
talking_ai.evidence -- this package invents no new per-observation
fields. `camera_observations`/`subtitle_observations`/
`artifact_observations` are small, reel-level `list[str]` context
lists (never auto-merged into individual timeline items -- see
adapter.py's own docstring) used for sampling/completeness/reporting
only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.video_intelligence.evidence import VideoPlatform, video_id_for

from ..evidence import ArtifactType, CameraMotionType, SpeechSegment, TalkingAIEvidence, TalkingLanguage
from ..models import ConfidenceLevel
from ..production_dna import TalkingAIProductionDNA
from .exceptions import LiveAdapterValidationError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(payload: dict[str, Any], *, length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


class FramingDistanceTag:
    """Framing/scene-context vocabulary talking_ai.evidence's own
    CameraMotionType does not model (see Phase 12D.2's framing.py,
    which already documents `shot_distance_type` as structurally
    `None` for this exact reason). Defined here, at the live-adapter
    layer, rather than by modifying evidence.py -- these tags feed
    `camera_observations` (reel-level context for sampling/
    completeness/reporting) only, never an individual TalkingAIEvidence
    field."""

    DIRECT_TO_CAMERA = "direct_to_camera"
    CLOSE_UP = "close_up"
    CHEST_UP = "chest_up"
    WAIST_UP = "waist_up"
    SELFIE = "selfie"
    WALKING = "walking"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"

    ALL = (DIRECT_TO_CAMERA, CLOSE_UP, CHEST_UP, WAIST_UP, SELFIE, WALKING, INDOOR, OUTDOOR)


CAMERA_OBSERVATION_VOCABULARY = tuple(CameraMotionType.ALL) + FramingDistanceTag.ALL

COMPLETENESS_DOMAINS = (
    "speech", "mouth", "blink", "gaze", "head", "face", "gesture",
    "subtitle", "camera", "artifact", "timing",
)


@dataclass(slots=True)
class LiveAdapterConfig:
    """Loaded from config/video_intelligence/talking_ai_live.yaml (see
    workflow.load_talking_ai_live_config())."""

    version: str
    schema_version: str
    adapter_version: str
    source_platform: str
    minimum_talking_reels: int
    recommended_talking_reels: int
    strong_sample: int
    deterministic: bool = True
    allow_sparse_timeline: bool = True
    require_exact_timestamps: bool = False
    preserve_source_references: bool = True
    deidentify_source: bool = True
    preserve_verbatim_transcripts: bool = False
    preserve_creator_username: bool = False
    live_enabled_by_default: bool = False
    live_require_explicit_authorization: bool = True
    output_root: str = "output/video_intelligence/talking_ai/live"

    def resolved_output_root(self) -> Path:
        """
        models.py location: 10_apps/claude_runtime/src/video_intelligence/talking_ai/live/models.py
        parents[4] resolves to 10_apps/claude_runtime.
        """
        runtime_root = Path(__file__).resolve().parents[4]
        return runtime_root / self.output_root


def _validate_vocabulary(values: list[str], allowed: tuple[str, ...], field_name: str) -> None:
    for value in values:
        if value not in allowed:
            raise LiveAdapterValidationError(f"{field_name} entry must be one of {allowed}, got {value!r}")


def compute_live_completeness(record: "LiveTalkingReelRecord") -> dict[str, float]:
    """Binary evidence-coverage score per domain (task §16) -- 1.0 if
    *any* evidence exists for that domain, 0.0 otherwise. Completeness
    is coverage only, never a quality/naturalness/performance
    judgement, and is computable before any Talking AI analyzer runs."""
    timeline = record.timeline_observations
    segments = record.speech_segments

    has_mouth = any(
        item.mouth_open_ratio is not None or item.mouth_shape_category is not None
        or item.jaw_motion is not None or item.lip_motion_intensity is not None
        for item in timeline
    )
    has_blink = any(item.blink is not None for item in timeline)
    has_gaze = any(item.gaze_direction is not None or item.eye_contact is not None for item in timeline)
    has_head = any(
        item.head_yaw is not None or item.head_pitch is not None
        or item.head_roll is not None or item.head_motion_intensity is not None
        for item in timeline
    )
    has_face = any(item.facial_expression is not None or item.smile_intensity is not None for item in timeline)
    has_gesture = any(
        item.left_hand_motion is not None or item.right_hand_motion is not None
        or item.gesture_type is not None or item.body_motion is not None
        for item in timeline
    )
    has_subtitle = bool(record.subtitle_observations) or any(
        item.subtitle_visible is not None or item.subtitle_change is not None for item in timeline
    )
    has_camera = bool(record.camera_observations) or any(
        item.camera_motion is not None or item.shot_boundary is not None for item in timeline
    )
    has_artifact = bool(record.artifact_observations) or any(item.artifact_tags for item in timeline)
    has_timing = bool(timeline) or bool(segments)

    return {
        "speech": 1.0 if segments else 0.0,
        "mouth": 1.0 if has_mouth else 0.0,
        "blink": 1.0 if has_blink else 0.0,
        "gaze": 1.0 if has_gaze else 0.0,
        "head": 1.0 if has_head else 0.0,
        "face": 1.0 if has_face else 0.0,
        "gesture": 1.0 if has_gesture else 0.0,
        "subtitle": 1.0 if has_subtitle else 0.0,
        "camera": 1.0 if has_camera else 0.0,
        "artifact": 1.0 if has_artifact else 0.0,
        "timing": 1.0 if has_timing else 0.0,
    }


def _compute_record_id(record: "LiveTalkingReelRecord") -> str:
    """Content-hash excluding `published_at` and `collected_at` (both
    timestamps -- task's own explicit instruction) and excluding
    `completeness` (fully derived from the other hashed fields, so
    including it would be redundant, never a source of drift)."""
    payload: dict = {
        "reel_id": record.reel_id,
        "source_url": record.source_url,
        "source_evidence_ids": sorted(record.source_evidence_ids),
        "duration_seconds": record.duration_seconds,
        "language_observations": sorted(record.language_observations),
        "timeline_observation_ids": sorted(item.evidence_id for item in record.timeline_observations),
        "speech_segment_ids": sorted(segment.segment_id for segment in record.speech_segments),
        "subtitle_observations": sorted(record.subtitle_observations),
        "camera_observations": sorted(record.camera_observations),
        "artifact_observations": sorted(record.artifact_observations),
        "warnings": sorted(record.warnings),
    }
    return _content_hash(payload)


@dataclass(slots=True)
class LiveTalkingReelRecord:
    """One Reel's live-research evidence collection (task §2).
    `creator_label` is an operator-chosen free-text label (never
    required to be a real handle) used only to derive `reel_id` -- the
    same rule every other *DNA-adjacent record in this codebase already
    enforces. `record_id` is a stable content hash excluding
    timestamps; `reel_id` is the same deterministic identity scheme
    `video_intelligence.evidence.video_id_for()` already provides,
    reused directly (never a second identity scheme)."""

    source_url: str
    creator_label: str = ""
    source_evidence_ids: list[str] = field(default_factory=list)
    published_at: str | None = None
    duration_seconds: float | None = None
    language_observations: list[str] = field(default_factory=list)
    timeline_observations: list[TalkingAIEvidence] = field(default_factory=list)
    speech_segments: list[SpeechSegment] = field(default_factory=list)
    subtitle_observations: list[str] = field(default_factory=list)
    camera_observations: list[str] = field(default_factory=list)
    artifact_observations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    collected_at: str = field(default_factory=_now_iso)

    reel_id: str = field(default="", init=False)
    completeness: dict[str, float] = field(default_factory=dict, init=False)
    record_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.source_url or not self.source_url.strip():
            raise LiveAdapterValidationError("source_url must not be empty")
        _validate_vocabulary(self.language_observations, TalkingLanguage.ALL, "language_observations")
        _validate_vocabulary(self.camera_observations, CAMERA_OBSERVATION_VOCABULARY, "camera_observations")
        _validate_vocabulary(self.artifact_observations, ArtifactType.ALL, "artifact_observations")

        self.reel_id = video_id_for(VideoPlatform.INSTAGRAM_REELS, self.creator_label, self.source_url)
        self.completeness = compute_live_completeness(self)
        self.record_id = _compute_record_id(self)


@dataclass(slots=True)
class LiveTalkingLearningRecord:
    """One Reel's outcome after learn_live_talking_reel() -- mirrors
    video_intelligence.instagram.models.InstagramReelLearningRecord's
    own shape. `source_url`/`reel_id`/`record_id` are kept for local
    traceability only (task §19's explicit exception) -- nothing here
    is copied into TalkingAIProductionDNA/the knowledge base beyond
    what those already-unmodified, already-de-identified types produce
    on their own."""

    reel_id: str
    record_id: str
    source_url: str
    dna: TalkingAIProductionDNA | None = None
    completeness: dict[str, float] = field(default_factory=dict)
    patterns_touched: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    collected_at: str = field(default_factory=_now_iso)

    @property
    def overall_confidence(self) -> str:
        return self.dna.overall_confidence if self.dna is not None else ConfidenceLevel.UNKNOWN


@dataclass(slots=True)
class LiveBatchLearningReport:
    """Task's own §21/§25 batch report shape."""

    total_reels_discovered: int
    reels_sampled: int
    reels_analyzed: int
    evidence_completeness: dict[str, float] = field(default_factory=dict)
    production_dna_ids: list[str] = field(default_factory=list)
    knowledge_patterns_touched: int = 0
    warnings: list[str] = field(default_factory=list)
    insufficient_evidence_reel_ids: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=_now_iso)
