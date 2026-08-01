from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


ReplyAction = Literal[
    "publish",
    "ignore",
    "hold_for_review",
    "escalate",
    "block_recommended",
    "clarification_required",
]


@dataclass(slots=True)
class CommentInput:
    request_id: str
    comment_id: str
    follower_id: str
    text: str
    platform: str
    content_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    follower_username: str | None = None
    parent_comment_id: str | None = None
    language_hint: str | None = None


@dataclass(slots=True)
class ContentContext:
    content_type: str
    topic: str | None = None
    caption: str | None = None
    location: str | None = None
    city: str | None = None
    country: str | None = None
    venue: str | None = None
    activity: str | None = None
    emotion: str | None = None
    story_stage: str | None = None
    ai_generated: bool = True
    sponsored: bool = False
    gifted: bool = False
    hosted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClassificationResult:
    primary_intent: str
    selected_library: str | None
    action: ReplyAction
    confidence: float
    detected_emotion: str = "neutral"
    reply_emotion: str = "friendly"
    secondary_intents: list[str] = field(default_factory=list)
    matched_triggers: list[str] = field(default_factory=list)
    moderation_reason: str | None = None


@dataclass(slots=True)
class ReplyCandidate:
    reply_id: str
    library: str
    reply_group: str
    text: str
    weight: int = 1
    emotion: str | None = None
    requires_context: list[str] = field(default_factory=list)
    source_data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    failed_rules: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    word_count: int = 0
    emoji_count: int = 0
    repeat_score: float = 0.0


@dataclass(slots=True)
class ReplyResult:
    request_id: str
    action: ReplyAction
    approved: bool
    reply_text: str | None
    classification: ClassificationResult
    validation: ValidationResult
    reply_id: str | None = None
    library: str | None = None
    reply_group: str | None = None
    language: str = "english"
    japanese_injected: bool = False
    selected_emojis: list[str] = field(default_factory=list)
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)