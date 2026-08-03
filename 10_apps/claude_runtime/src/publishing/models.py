from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

PlatformName = Literal["instagram", "threads"]

ContentType = Literal[
    "instagram_feed",
    "instagram_story",
    "instagram_reel",
    "threads_post",
]

CONTENT_TYPES: tuple[str, ...] = (
    "instagram_feed",
    "instagram_story",
    "instagram_reel",
    "threads_post",
)

PublishStatus = Literal[
    "draft",
    "validation_failed",
    "ready",
    "pending_approval",
    "approved",
    "scheduled",
    "publishing",
    "published",
    "failed",
    "rejected",
    "cancelled",
]

STATUS_VALUES: tuple[str, ...] = (
    "draft",
    "validation_failed",
    "ready",
    "pending_approval",
    "approved",
    "scheduled",
    "publishing",
    "published",
    "failed",
    "rejected",
    "cancelled",
)

# Jobs in these statuses represent an in-flight or completed live publish
# and must never be silently overwritten by --prepare --force or mutated
# by approval/scheduling.
TERMINAL_LOCKED_STATUSES: frozenset[str] = frozenset({"publishing", "published"})


def platform_for_content_type(content_type: str) -> str:
    return "threads" if content_type == "threads_post" else "instagram"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def app_root() -> Path:
    """
    models.py location: 10_apps/claude_runtime/src/publishing/models.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def date_output_root(production_date: str) -> Path:
    """Phase 9's per-day output directory: output/<date>/ (not configurable)."""
    return app_root() / "output" / production_date


def default_publisher_config_path() -> Path:
    return app_root() / "config" / "publishing" / "publisher.yaml"


@dataclass(slots=True)
class PublisherConfig:
    version: str = "1.0"
    default_persona_id: str = "aiko"
    timezone: str = "UTC"
    approval_required: bool = True
    dry_run_default: bool = False
    platform_limits: dict[str, dict[str, int]] = field(default_factory=dict)
    required_fields: dict[str, list[str]] = field(default_factory=dict)
    allowed_media_extensions: tuple[str, ...] = ()
    paths: dict[str, str] = field(default_factory=dict)

    def resolve_path(self, key: str, default: str) -> Path:
        return app_root() / self.paths.get(key, default)

    def queue_path(self) -> Path:
        queue_dir = self.resolve_path("queue_dir", "output/publishing/queues")
        filename = self.paths.get("queue_filename", "publish_queue.json")
        return queue_dir / filename

    def history_dir(self) -> Path:
        return self.resolve_path("history_dir", "output/publishing/history")

    def logs_dir(self) -> Path:
        return self.resolve_path("logs_dir", "output/publishing/logs")


def load_publisher_config(config_path: str | Path | None = None) -> PublisherConfig:
    path = Path(config_path) if config_path else default_publisher_config_path()

    if not path.exists():
        raise FileNotFoundError(f"Publisher config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    return PublisherConfig(
        version=str(data.get("version", "1.0")),
        default_persona_id=str(data.get("default_persona_id", "aiko")),
        timezone=str(data.get("timezone", "UTC")),
        approval_required=bool(data.get("approval_required", True)),
        dry_run_default=bool(data.get("dry_run_default", False)),
        platform_limits=data.get("platform_limits") or {},
        required_fields=data.get("required_fields") or {},
        allowed_media_extensions=tuple(
            str(ext).lower() for ext in (data.get("allowed_media_extensions") or [])
        ),
        paths=data.get("paths") or {},
    )


_KNOWN_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "job_id",
        "production_date",
        "persona_id",
        "platform",
        "content_type",
        "media_paths",
        "caption_path",
        "caption_text",
        "hashtags_path",
        "hashtags",
        "threads_text",
        "location",
        "alt_text",
        "scheduled_at",
        "status",
        "created_at",
        "updated_at",
        "approved_at",
        "published_at",
        "platform_post_id",
        "platform_url",
        "validation_errors",
        "error",
        "metadata",
    }
)


@dataclass(slots=True)
class PublishJob:
    job_id: str
    production_date: str
    persona_id: str
    platform: str
    content_type: str
    media_paths: list[str] = field(default_factory=list)
    caption_path: str | None = None
    caption_text: str | None = None
    hashtags_path: str | None = None
    hashtags: list[str] = field(default_factory=list)
    threads_text: str | None = None
    location: str | None = None
    alt_text: str | None = None
    scheduled_at: str | None = None
    status: str = "draft"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    approved_at: str | None = None
    published_at: str | None = None
    platform_post_id: str | None = None
    platform_url: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Unknown top-level keys encountered when loading a job from the queue
    # file (e.g. written by a newer version). Round-tripped verbatim by
    # to_dict()/from_dict() so nothing is ever silently dropped.
    extra: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "production_date": self.production_date,
            "persona_id": self.persona_id,
            "platform": self.platform,
            "content_type": self.content_type,
            "media_paths": list(self.media_paths),
            "caption_path": self.caption_path,
            "caption_text": self.caption_text,
            "hashtags_path": self.hashtags_path,
            "hashtags": list(self.hashtags),
            "threads_text": self.threads_text,
            "location": self.location,
            "alt_text": self.alt_text,
            "scheduled_at": self.scheduled_at,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "approved_at": self.approved_at,
            "published_at": self.published_at,
            "platform_post_id": self.platform_post_id,
            "platform_url": self.platform_url,
            "validation_errors": list(self.validation_errors),
            "error": self.error,
            "metadata": copy.deepcopy(self.metadata),
        }

        for key, value in self.extra.items():
            if key not in data:
                data[key] = value

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PublishJob":
        known = {key: value for key, value in data.items() if key in _KNOWN_FIELD_NAMES}
        extra = {key: value for key, value in data.items() if key not in _KNOWN_FIELD_NAMES}

        return cls(
            job_id=known["job_id"],
            production_date=known["production_date"],
            persona_id=known["persona_id"],
            platform=known["platform"],
            content_type=known["content_type"],
            media_paths=list(known.get("media_paths") or []),
            caption_path=known.get("caption_path"),
            caption_text=known.get("caption_text"),
            hashtags_path=known.get("hashtags_path"),
            hashtags=list(known.get("hashtags") or []),
            threads_text=known.get("threads_text"),
            location=known.get("location"),
            alt_text=known.get("alt_text"),
            scheduled_at=known.get("scheduled_at"),
            status=known.get("status", "draft"),
            created_at=known.get("created_at") or now_iso(),
            updated_at=known.get("updated_at") or now_iso(),
            approved_at=known.get("approved_at"),
            published_at=known.get("published_at"),
            platform_post_id=known.get("platform_post_id"),
            platform_url=known.get("platform_url"),
            validation_errors=list(known.get("validation_errors") or []),
            error=known.get("error"),
            metadata=dict(known.get("metadata") or {}),
            extra=extra,
        )
