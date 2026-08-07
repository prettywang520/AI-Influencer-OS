"""Intermediate, full-fidelity records observed from Instagram pages,
before conversion into creator_intelligence.evidence.Evidence
(evidence_mapper.py owns that conversion). Every optional field
defaults to None/empty -- unknown values stay unknown, never
fabricated or guessed. Nothing here infers legal identity, age,
relationship status, private address, religion, or politics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AccessEventKind:
    """Vocabulary for observed access obstacles. New in this package
    -- confirmed no collision with src/social/instagram_models.py."""

    ACCESS_LIMITED = "access_limited"
    AUTHENTICATION_REQUIRED = "authentication_required"
    RATE_LIMITED = "rate_limited"
    CHALLENGE_DETECTED = "challenge_detected"

    ALL = (ACCESS_LIMITED, AUTHENTICATION_REQUIRED, RATE_LIMITED, CHALLENGE_DETECTED)


class RelationshipEvidenceBasis:
    """How a relationship/collaboration observation was grounded --
    matches the task's own §13 vocabulary. Distinct from (but mapped
    onto, in evidence_mapper.py) creator_intelligence.models.RelationshipBasis."""

    CAPTION_DECLARED = "caption_declared"
    TAGGED_ACCOUNT = "tagged_account"
    COLLABORATION_LABEL = "collaboration_label"
    VISIBLE_PUBLIC_CONTEXT = "visible_public_context"

    ALL = (CAPTION_DECLARED, TAGGED_ACCOUNT, COLLABORATION_LABEL, VISIBLE_PUBLIC_CONTEXT)


class VisualSamplingBucket:
    """Suggested deterministic sampling buckets for collect_visual_examples()
    (task §14). Selection only -- no visual-realism judgement is made here."""

    RECENT = "recent"
    HIGH_ENGAGEMENT_VISIBLE = "high_engagement_visible"
    SELFIE = "selfie"
    STREET = "street"
    HOME_DAILY = "home_daily"
    FRIENDS_SOCIAL = "friends_social"
    TRAVEL = "travel"
    NIGHT_FLASH = "night_flash"
    SOFT_FOCUS_CCD_CANDIDATE = "soft_focus_ccd_candidate"
    LUXURY = "luxury"
    ORDINARY_ENVIRONMENT = "ordinary_environment"

    ALL = (
        RECENT,
        HIGH_ENGAGEMENT_VISIBLE,
        SELFIE,
        STREET,
        HOME_DAILY,
        FRIENDS_SOCIAL,
        TRAVEL,
        NIGHT_FLASH,
        SOFT_FOCUS_CCD_CANDIDATE,
        LUXURY,
        ORDINARY_ENVIRONMENT,
    )


@dataclass(slots=True)
class AccessEvent:
    kind: str
    section: str
    detail: str = ""
    occurred_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class ProfileRecord:
    username: str
    display_name: str | None = None
    bio: str | None = None
    profile_url: str = ""
    post_count: int | None = None
    follower_count: int | None = None
    following_count: int | None = None
    category: str | None = None
    external_link: str | None = None
    verified: bool | None = None
    profile_image_reference: str | None = None
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class DiscoveredItem:
    source_url: str
    source_id: str | None = None
    content_type: str = "post"  # "post" | "reel"
    thumbnail_reference: str | None = None
    position: int = 0
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class PostRecord:
    post_url: str
    published_at: str | None = None
    caption_reference: str | None = None
    is_carousel: bool = False
    like_count: int | None = None
    comment_count: int | None = None
    tagged_accounts: list[str] = field(default_factory=list)
    collaboration_labels: list[str] = field(default_factory=list)
    location_label: str | None = None
    content_type: str = "post"
    visual_reference: str | None = None
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class CaptionObservation:
    caption_text: str
    post_reference: str
    published_at: str | None = None
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    language: str | None = None
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class CommentObservation:
    comment_text: str
    post_reference: str
    author_username: str | None = None
    timestamp_text: str | None = None
    thread_parent_id: str | None = None
    is_creator: bool = False
    comment_id: str | None = None
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class CreatorReplyObservation:
    parent_comment_text: str
    creator_reply_text: str
    post_reference: str
    language: str | None = None
    emoji: list[str] = field(default_factory=list)
    thread_depth: int = 1
    timestamp_text: str | None = None
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class ReelRecord:
    reel_url: str
    caption: str | None = None
    published_at: str | None = None
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    duration_seconds: float | None = None
    text_overlays: list[str] = field(default_factory=list)
    thumbnail_reference: str | None = None
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class HighlightRecord:
    highlight_title: str
    item_index: int
    content_type: str | None = None
    reference: str | None = None
    caption_text: str | None = None
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class RelationshipEvidenceRecord:
    description: str
    basis: str
    related_account: str | None = None
    post_reference: str | None = None
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class VisualExampleRecord:
    source_reference: str
    sampling_bucket: str
    screenshot_reference: str | None = None
    captured_at: str = field(default_factory=now_iso)
