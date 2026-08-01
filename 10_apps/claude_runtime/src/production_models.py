from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ProductionStatus = Literal[
    "planned",
    "ready_for_images",
    "images_in_progress",
    "images_complete",
    "ready_to_publish",
    "published",
    "failed",
]


@dataclass(slots=True)
class ImageProductionTask:
    task_id: str
    production_date: str
    content_id: str
    content_type: str
    title: str
    prompt_file: str
    expected_output_file: str
    status: str = "pending"
    provider: str = "chatgpt_manual"
    attempts: int = 0
    generated_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReelProductionTask:
    task_id: str
    production_date: str
    scene_number: int
    title: str
    duration_seconds: float
    prompt_file: str
    expected_output_file: str
    status: str = "pending"
    provider: str = "runway_manual"
    attempts: int = 0
    generated_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProductionManifest:
    production_id: str
    production_date: str
    created_at: str
    country: str
    city: str
    venue: str
    theme: str
    story_summary: str
    status: ProductionStatus
    feed_count: int
    story_count: int
    reel_scene_count: int
    image_tasks: list[ImageProductionTask] = field(default_factory=list)
    reel_tasks: list[ReelProductionTask] = field(default_factory=list)
    captions: dict[str, Any] = field(default_factory=dict)
    hashtags: list[str] = field(default_factory=list)
    quality_gate: dict[str, Any] = field(default_factory=dict)
    output_directory: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)