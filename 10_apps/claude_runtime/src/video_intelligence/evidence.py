"""VideoEvidence, VideoRecord, and their supporting vocabulary.

video_intelligence never scrapes, browses, or transcribes a video --
every VideoEvidence item is always operator-supplied or synthetic
(tests/demo), exactly like creator_intelligence.evidence.Evidence's
own boundary (reimplemented fresh here, not imported -- this package
is scoped around *a video*, not a creator, and stays fully decoupled
from creator_intelligence/creator_research/creator_learning). Two
fields have no analog there: `video_id` (one knowledge base spans
many videos, so evidence must self-identify which video it's about)
and `timestamp_seconds` (video evidence is inherently time-anchored).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(payload: dict[str, Any], *, length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


class VideoPlatform:
    """Open-ended-by-design closed vocabulary -- adding a future
    platform is a new member here, never an analyzer change (see
    docs/video_intelligence/architecture.md's Future Expansion section)."""

    INSTAGRAM_REELS = "instagram_reels"
    TIKTOK = "tiktok"
    YOUTUBE_SHORTS = "youtube_shorts"
    FACEBOOK_REELS = "facebook_reels"
    X_VIDEO = "x_video"
    OTHER = "other"

    ALL = (INSTAGRAM_REELS, TIKTOK, YOUTUBE_SHORTS, FACEBOOK_REELS, X_VIDEO, OTHER)


class VideoEvidenceType:
    """Closed set of allowed evidence sources -- all human-mediated.
    Mirrors creator_intelligence.evidence.EvidenceType exactly."""

    SCREENSHOT = "screenshot"
    MANUAL_NOTE = "manual_note"
    TEXT_EXCERPT = "text_excerpt"
    EXPORTED_DATA = "exported_data"
    OPERATOR_OBSERVATION = "operator_observation"

    ALL = (SCREENSHOT, MANUAL_NOTE, TEXT_EXCERPT, EXPORTED_DATA, OPERATOR_OBSERVATION)


class VideoIntelligenceError(RuntimeError):
    """Base error for the Video Intelligence OS."""


class InvalidVideoEvidenceTypeError(VideoIntelligenceError):
    """Raised when an evidence_type outside the allowed set is used."""


def video_id_for(platform: str, creator_label: str, reference: str) -> str:
    """Deterministic video identity, stable across re-runs -- no
    timestamp, no free-text field that could vary between calls.
    `creator_label` is an optional, operator-chosen free-text label,
    never required to be a real account handle -- same rule
    creator_intelligence already enforces."""
    return _content_hash({"platform": platform, "creator_label": creator_label, "reference": reference})


@dataclass(slots=True)
class VideoRecord:
    """One observed video's identity/metadata -- analogous to
    creator_intelligence.intake.CreatorProfile, but scoped to a single
    video rather than a creator's overall presence."""

    platform: str
    creator_label: str = ""
    reference: str = ""
    duration_seconds: float | None = None
    notes: str = ""
    created_at: str = field(default_factory=_now_iso)
    metadata: dict = field(default_factory=dict)
    video_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.platform not in VideoPlatform.ALL:
            raise VideoIntelligenceError(f"platform must be one of {VideoPlatform.ALL}, got {self.platform!r}")
        self.video_id = video_id_for(self.platform, self.creator_label, self.reference)


def _compute_evidence_id(
    video_id: str,
    evidence_type: str,
    source_description: str,
    content_excerpt: str,
    collected_by: str,
    timestamp_seconds: float | None,
    tags: tuple[str, ...],
) -> str:
    payload = {
        "video_id": video_id,
        "evidence_type": evidence_type,
        "source_description": source_description,
        "content_excerpt": content_excerpt,
        "collected_by": collected_by,
        "timestamp_seconds": timestamp_seconds,
        "tags": sorted(tags),
    }
    return _content_hash(payload)


@dataclass(slots=True)
class VideoEvidence:
    """One piece of operator-supplied evidence about one observed
    video. `evidence_id` is a stable content hash (excludes
    `collected_at`, so re-recording identical evidence at a different
    time does not mint a new ID). `content_excerpt` must stay short --
    this is a documented convention, not code-enforced here (no
    validation.py exists in this package; see docs/video_intelligence/
    architecture.md)."""

    video_id: str
    evidence_type: str
    source_description: str
    content_excerpt: str = ""
    timestamp_seconds: float | None = None
    collected_at: str = ""
    collected_by: str = "operator"
    tags: list[str] = field(default_factory=list)
    evidence_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.evidence_type not in VideoEvidenceType.ALL:
            raise InvalidVideoEvidenceTypeError(
                f"evidence_type must be one of {VideoEvidenceType.ALL}, got {self.evidence_type!r}"
            )
        self.evidence_id = _compute_evidence_id(
            self.video_id,
            self.evidence_type,
            self.source_description,
            self.content_excerpt,
            self.collected_by,
            self.timestamp_seconds,
            tuple(self.tags),
        )
