from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ContentType = Literal["feed", "story", "reel"]


@dataclass(slots=True)
class ContentMoment:
    content_id: str
    content_type: ContentType
    title: str
    location: str
    real_life: str
    behavior: str
    emotion: str
    interaction: str
    body_motion: str
    camera_story: str
    daily_details: list[str]
    story_stage: str
    japanese_detail: str | None = None
    prompt: str | None = None
    caption: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReelScene:
    scene_number: int
    title: str
    behavior: str
    emotion: str
    interaction: str
    camera_story: str
    body_motion: str
    daily_details: list[str]
    duration_seconds: float = 2.5
    prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DailyContentPlan:
    date: str
    country: str
    city: str
    venue: str
    theme: str
    story_summary: str
    feed: ContentMoment
    stories: list[ContentMoment]
    reel_scenes: list[ReelScene]
    feed_caption: str | None = None
    story_captions: list[str] = field(default_factory=list)
    reel_caption: str | None = None
    hashtags: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)