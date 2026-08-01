from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SessionStatus = Literal["logged_in", "logged_out", "unknown"]


class InstagramSessionError(RuntimeError):
    """Base error for all Instagram browser session infrastructure failures."""


class InstagramConfigError(InstagramSessionError):
    """Raised when config/social/instagram.yaml is missing or invalid."""


class ProfileDirectoryError(InstagramSessionError):
    """Raised when the persistent browser profile directory cannot be prepared."""


class LoginTimeoutError(InstagramSessionError):
    """Raised when manual login is not detected before the configured timeout."""


class SessionDetectionError(InstagramSessionError):
    """Raised when the browser page cannot be reached to detect session state."""


@dataclass(slots=True)
class ViewportConfig:
    width: int = 1280
    height: int = 900

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LoginDetectionConfig:
    logged_in_selectors: list[str] = field(default_factory=list)
    logged_out_selectors: list[str] = field(default_factory=list)
    logged_out_url_markers: list[str] = field(default_factory=list)
    pending_url_markers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InstagramSessionConfig:
    """
    Fully-resolved runtime configuration for the Instagram browser session.

    Built by instagram_session.load_config() from config/social/instagram.yaml.
    """

    base_url: str
    login_url: str
    profile_dir: Path
    headless: bool
    dry_run: bool
    locale: str
    timezone_id: str
    viewport: ViewportConfig
    navigation_timeout_ms: int
    login_wait_timeout_ms: int
    check_timeout_ms: int
    login_detection: LoginDetectionConfig
    screenshot_dir: Path
    screenshot_on_detection_failure: bool
    disallowed_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["profile_dir"] = str(self.profile_dir)
        data["screenshot_dir"] = str(self.screenshot_dir)
        return data


@dataclass(slots=True)
class SessionState:
    """
    Result of a single logged_in / logged_out detection pass.
    """

    status: SessionStatus
    checked_at: str
    current_url: str | None = None
    reason: str | None = None
    screenshot_path: str | None = None
    dry_run: bool = False

    @property
    def logged_in(self) -> bool:
        return self.status == "logged_in"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 8B — Instagram Comment Reader
# ---------------------------------------------------------------------------
#
# Read-only. This module never posts, replies, likes, follows, unfollows,
# hides, deletes, reports, or sends direct messages.


class CommentReaderError(InstagramSessionError):
    """Base error for the Instagram comment reader."""


class PostAccessError(CommentReaderError):
    """Raised when the target post/reel cannot be opened."""


class CommentContainerNotFoundError(CommentReaderError):
    """Raised when the comment container cannot be located on the page."""


class CommentReaderLoginLostError(CommentReaderError):
    """Raised when the session is not logged_in while reading comments."""


CommentRecordStatus = Literal["pending", "processed", "error", "skipped"]


@dataclass(slots=True)
class CommentSelectors:
    """
    Centralised CSS/Playwright selectors for the comment reader.

    All comment-content selectors here are untested placeholder
    defaults (unlike the Phase 8A login_detection selectors, which were
    validated against a real session) and should be tuned against a
    live post using the screenshots and diagnostic log this module
    produces on failure.
    """

    comment_container: list[str] = field(default_factory=list)
    comment_item: list[str] = field(default_factory=list)
    comment_username: list[str] = field(default_factory=list)
    comment_text: list[str] = field(default_factory=list)
    comment_timestamp: list[str] = field(default_factory=list)
    comment_permalink: list[str] = field(default_factory=list)
    reply_item: list[str] = field(default_factory=list)
    load_more_button: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommentReaderConfig:
    """
    Fully-resolved runtime configuration for the comment reader.

    Built by instagram_comment_reader.load_comment_reader_config()
    from config/social/instagram.yaml.
    """

    own_username: str
    selectors: CommentSelectors
    default_limit: int
    max_scroll_attempts: int
    scroll_pause_ms: int
    read_timeout_ms: int
    post_navigation_timeout_ms: int
    queue_dir: Path
    logs_dir: Path
    screenshots_dir: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["queue_dir"] = str(self.queue_dir)
        data["logs_dir"] = str(self.logs_dir)
        data["screenshots_dir"] = str(self.screenshots_dir)
        return data


@dataclass(slots=True)
class CommentRecord:
    """
    One discovered Instagram comment, as stored in comment_queue.json.
    """

    source: str
    stable_id: str
    status: CommentRecordStatus
    discovered_at: str
    post_url: str
    username: str | None
    comment_text: str | None
    timestamp_text: str | None
    parent_comment_id: str | None
    is_reply: bool
    error: str | None = None
    comment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommentReadResult:
    """
    Summary of one instagram_comment_reader run, as reported by the CLI.
    """

    login_status: str
    post_url: str
    comments_found: int
    new_comments_queued: int
    duplicates_skipped: int
    excluded_own_count: int
    queue_path: str
    screenshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 8C — Reply Brain + Safety Gate + Approval Queue
# ---------------------------------------------------------------------------
#
# Nothing in this section ever sends anything to Instagram. Approval
# and rejection only update local JSON.


class ReplyBrainError(InstagramSessionError):
    """Base error for reply classification and generation."""


class ReplyRulesConfigError(ReplyBrainError):
    """Raised when config/social/reply_rules.yaml is missing or invalid."""


class ReplyGenerationError(ReplyBrainError):
    """Raised when no reply candidate can be produced for a category."""


class ApprovalServiceError(InstagramSessionError):
    """Base error for the local approval queue service."""


class ApprovalNotFoundError(ApprovalServiceError):
    """Raised when an approval_id does not exist in the approval queue."""


ReplyClassification = Literal[
    "compliment",
    "emoji_only",
    "travel_question",
    "location_question",
    "food_question",
    "outfit_question",
    "camera_question",
    "hotel_question",
    "casual_chat",
    "flirting",
    "negative",
    "spam",
    "brand_collaboration",
    "sensitive",
    "unknown",
]

SafetyRoute = Literal["auto_eligible", "human_review_required", "never_reply"]

ApprovalStatus = Literal[
    "pending_approval",
    "approved",
    "rejected",
    "sent",
    "skipped",
]


@dataclass(slots=True)
class ReplyStyleConfig:
    """
    Aiko reply style rules: 5-25 words, warm, natural, max 2 emojis,
    lowercase English preferred, never mentions being AI.
    """

    minimum_words: int = 5
    maximum_words: int = 25
    maximum_emojis: int = 2
    banned_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplyCategoryConfig:
    """
    One classification category: which existing persona reply library
    (under 03_personas/aiko/reply_brain/) it reuses, plus any extra
    keywords/fallback replies for categories with no existing library.
    """

    name: str
    source_library: str | None = None
    keywords: list[str] = field(default_factory=list)
    fallback_replies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplyRulesConfig:
    """
    Fully-resolved runtime configuration for the Reply Brain.

    Built by reply_brain.load_reply_rules_config() from
    config/social/reply_rules.yaml.
    """

    reply_brain_root: Path
    style: ReplyStyleConfig
    emoji_pattern: str
    classification_priority: list[str]
    categories: dict[str, ReplyCategoryConfig]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply_brain_root": str(self.reply_brain_root),
            "style": self.style.to_dict(),
            "emoji_pattern": self.emoji_pattern,
            "classification_priority": list(self.classification_priority),
            "categories": {
                name: category.to_dict()
                for name, category in self.categories.items()
            },
        }


@dataclass(slots=True)
class ClassificationResult:
    """
    Result of classifying one comment's text.
    """

    classification: ReplyClassification
    matched_keyword: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplyStyleCheck:
    """
    Result of validating one candidate reply against ReplyStyleConfig.
    """

    passed: bool
    issues: list[str]
    word_count: int
    emoji_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ApprovalRecord:
    """
    One proposed (or skipped) reply awaiting human approval.

    Stored in output/community/queues/approval_queue.json.
    """

    approval_id: str
    stable_id: str
    comment_id: str | None
    post_url: str
    username: str | None
    comment_text: str | None
    classification: ReplyClassification
    proposed_reply: str | None
    safety_route: SafetyRoute
    status: ApprovalStatus
    created_at: str
    approved_at: str | None = None
    rejected_at: str | None = None
    sent_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PrepareRepliesResult:
    """
    Summary of one community_service --prepare-replies run.
    """

    comments_processed: int
    replies_proposed: int
    auto_eligible: int
    human_review_required: int
    skipped: int
    duplicates_skipped: int
    approval_queue_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Phase 8D — Instagram Reply Controller (Dry-Run Preview)
# ---------------------------------------------------------------------------
#
# Strict dry run: fills the reply textbox for a human to visually
# verify, then stops. Never presses Enter, never clicks Post/Send,
# never writes to approval_queue.json, and never likes, follows,
# unfollows, hides, deletes, reports, or sends direct messages.


class ReplyControllerError(InstagramSessionError):
    """Base error for the Instagram reply controller."""


class ApprovalRecordNotFoundError(ReplyControllerError):
    """Raised when the given approval_id does not exist in the approval queue."""


class ApprovalRecordNotApprovedError(ReplyControllerError):
    """Raised when the approval record's status is not 'approved'."""


class ApprovalRecordInvalidError(ReplyControllerError):
    """Raised when the approval record is missing a required field."""


class ReplyControllerLoginLostError(ReplyControllerError):
    """Raised when the session is not logged_in while previewing a reply."""


class ReplyPostAccessError(ReplyControllerError):
    """Raised when the target post cannot be opened."""


class CommentNotFoundError(ReplyControllerError):
    """Raised when the exact comment (by comment_id) cannot be located."""


class ReplyControlNotFoundError(ReplyControllerError):
    """Raised when the Reply control for the matched comment cannot be found."""


class ReplyTextboxNotFoundError(ReplyControllerError):
    """Raised when the reply textbox cannot be located after clicking Reply."""


@dataclass(slots=True)
class ReplyControllerSelectors:
    """
    Centralised CSS/Playwright selectors for the reply controller.

    comment_container / comment_item are reused from
    comment_reader.selectors (Phase 8B, validated against a real
    post). comment_permalink_template, reply_button, and
    reply_textbox are new to Phase 8D and are untested placeholder
    defaults — tune against a real approved record using the
    screenshot and diagnostic log this module writes on failure.
    """

    comment_container: list[str] = field(default_factory=list)
    comment_item: list[str] = field(default_factory=list)
    comment_permalink_template: str = "a[href*='/c/{comment_id}/']"
    reply_button: list[str] = field(default_factory=list)
    reply_textbox: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplyControllerConfig:
    """
    Fully-resolved runtime configuration for the reply controller.

    Built by instagram_reply_controller.load_reply_controller_config()
    from config/social/instagram.yaml.
    """

    selectors: ReplyControllerSelectors
    max_scroll_attempts: int
    scroll_pause_ms: int
    search_timeout_ms: int
    post_navigation_timeout_ms: int
    post_fill_wait_ms: int
    screenshots_dir: Path
    logs_dir: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["screenshots_dir"] = str(self.screenshots_dir)
        data["logs_dir"] = str(self.logs_dir)
        return data


@dataclass(slots=True)
class ReplyPreviewResult:
    """
    Summary of one instagram_reply_controller --dry-run run.

    sent is always False: Phase 8D never submits a reply.
    """

    login_status: str
    approval_id: str
    comment_found: bool
    reply_control_clicked: bool
    textbox_filled: bool
    sent: bool
    screenshot_path: str | None = None
    selector_used: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
