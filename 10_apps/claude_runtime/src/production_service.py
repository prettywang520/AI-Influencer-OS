from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .content_models import DailyContentPlan
from .content_service import ContentService, build_service as build_content_service


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
    """
    One still-image generation task.

    Feed: one task
    Stories: four tasks
    """

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
    """
    One Reel scene production task.

    The current workflow prepares five scene prompts for Runway
    or another video-generation platform.
    """

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
    """
    Complete production record for one AIKO content day.
    """

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

    image_tasks: list[ImageProductionTask] = field(
        default_factory=list
    )
    reel_tasks: list[ReelProductionTask] = field(
        default_factory=list
    )

    captions: dict[str, Any] = field(
        default_factory=dict
    )
    hashtags: list[str] = field(
        default_factory=list
    )

    quality_gate: dict[str, Any] = field(
        default_factory=dict
    )

    output_directory: str = ""
    errors: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["image_tasks"] = [
            task.to_dict()
            for task in self.image_tasks
        ]

        data["reel_tasks"] = [
            task.to_dict()
            for task in self.reel_tasks
        ]

        return data


class ProductionServiceError(RuntimeError):
    """Raised when daily production cannot be prepared safely."""


class ProductionService:
    """
    AIKO Production Service.

    Pipeline:

    target date
    -> ContentService
    -> DynamicContentPlanner
    -> prompts
    -> captions
    -> hashtags
    -> quality gate
    -> image production queue
    -> Reel production queue
    -> production manifest

    This version does not generate or publish media automatically.

    It prepares a complete production package for:

    - ChatGPT manual image generation
    - Runway manual Reel generation
    - later publishing automation
    """

    def __init__(
        self,
        *,
        content_service: ContentService,
        output_root: str | Path,
    ) -> None:
        self.content_service = content_service

        self.output_root = Path(
            output_root
        ).expanduser().resolve()

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _production_id(
        production_date: str,
    ) -> str:
        return f"aiko-production-{production_date}"

    def run(
        self,
        *,
        target_date: str | None = None,
        remember_plan: bool = True,
        overwrite: bool = True,
    ) -> ProductionManifest:
        """
        Generate and prepare one complete daily production package.
        """
        production_date = (
            target_date
            or date.today().isoformat()
        )

        self._validate_date(
            production_date
        )

        day_root = (
            self.output_root
            / production_date
        )

        manifest_path = (
            day_root
            / "production_manifest.json"
        )

        if (
            manifest_path.exists()
            and not overwrite
        ):
            raise ProductionServiceError(
                "Production manifest already exists: "
                f"{manifest_path}"
            )

        try:
            plan = self.content_service.generate(
                target_date=production_date,
                remember_plan=remember_plan,
            )

            self._prepare_directories(
                day_root
            )

            image_tasks = self._build_image_tasks(
                plan=plan,
                day_root=day_root,
            )

            reel_tasks = self._build_reel_tasks(
                plan=plan,
                day_root=day_root,
            )

            manifest = ProductionManifest(
                production_id=self._production_id(
                    production_date
                ),
                production_date=production_date,
                created_at=self._now(),
                country=plan.country,
                city=plan.city,
                venue=plan.venue,
                theme=plan.theme,
                story_summary=plan.story_summary,
                status="ready_for_images",
                feed_count=1,
                story_count=len(
                    plan.stories
                ),
                reel_scene_count=len(
                    plan.reel_scenes
                ),
                image_tasks=image_tasks,
                reel_tasks=reel_tasks,
                captions={
                    "feed": plan.feed_caption,
                    "stories": plan.story_captions,
                    "reel": plan.reel_caption,
                },
                hashtags=list(
                    plan.hashtags
                ),
                quality_gate=dict(
                    plan.validation
                ),
                output_directory=str(
                    day_root
                ),
            )

            self._write_manifest(
                manifest_path=manifest_path,
                manifest=manifest,
            )

            self._write_image_queue(
                day_root=day_root,
                tasks=image_tasks,
            )

            self._write_reel_queue(
                day_root=day_root,
                tasks=reel_tasks,
            )

            self._write_production_summary(
                day_root=day_root,
                manifest=manifest,
            )

            return manifest

        except Exception as exc:
            self._write_failure_record(
                day_root=day_root,
                production_date=production_date,
                error=exc,
            )

            raise

    @staticmethod
    def _validate_date(
        production_date: str,
    ) -> None:
        try:
            date.fromisoformat(
                production_date
            )
        except ValueError as exc:
            raise ProductionServiceError(
                "Production date must use "
                "YYYY-MM-DD format"
            ) from exc

    @staticmethod
    def _prepare_directories(
        day_root: Path,
    ) -> None:
        directories = [
            day_root,
            day_root / "prompts",
            day_root / "captions",
            day_root / "reels",
            day_root / "images",
            day_root / "images" / "feed",
            day_root / "images" / "stories",
            day_root / "videos",
            day_root / "videos" / "reel_scenes",
            day_root / "queues",
            day_root / "json",
            day_root / "logs",
            day_root / "archive",
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def _build_image_tasks(
        *,
        plan: DailyContentPlan,
        day_root: Path,
    ) -> list[ImageProductionTask]:
        tasks: list[ImageProductionTask] = []

        feed_prompt_path = (
            day_root
            / "prompts"
            / "feed_prompt.txt"
        )

        feed_output_path = (
            day_root
            / "images"
            / "feed"
            / "feed_01.png"
        )

        tasks.append(
            ImageProductionTask(
                task_id=(
                    f"{plan.date}-image-feed-01"
                ),
                production_date=plan.date,
                content_id=plan.feed.content_id,
                content_type="feed",
                title=plan.feed.title,
                prompt_file=str(
                    feed_prompt_path
                ),
                expected_output_file=str(
                    feed_output_path
                ),
            )
        )

        for index, story in enumerate(
            plan.stories,
            start=1,
        ):
            prompt_path = (
                day_root
                / "prompts"
                / f"story_{index}_prompt.txt"
            )

            output_path = (
                day_root
                / "images"
                / "stories"
                / f"story_{index:02d}.png"
            )

            tasks.append(
                ImageProductionTask(
                    task_id=(
                        f"{plan.date}-image-story-"
                        f"{index:02d}"
                    ),
                    production_date=plan.date,
                    content_id=story.content_id,
                    content_type="story",
                    title=story.title,
                    prompt_file=str(
                        prompt_path
                    ),
                    expected_output_file=str(
                        output_path
                    ),
                )
            )

        return tasks

    @staticmethod
    def _build_reel_tasks(
        *,
        plan: DailyContentPlan,
        day_root: Path,
    ) -> list[ReelProductionTask]:
        tasks: list[ReelProductionTask] = []

        for scene in plan.reel_scenes:
            prompt_path = (
                day_root
                / "reels"
                / (
                    f"scene_{scene.scene_number}"
                    "_prompt.txt"
                )
            )

            output_path = (
                day_root
                / "videos"
                / "reel_scenes"
                / (
                    f"scene_{scene.scene_number:02d}"
                    ".mp4"
                )
            )

            tasks.append(
                ReelProductionTask(
                    task_id=(
                        f"{plan.date}-reel-scene-"
                        f"{scene.scene_number:02d}"
                    ),
                    production_date=plan.date,
                    scene_number=scene.scene_number,
                    title=scene.title,
                    duration_seconds=(
                        scene.duration_seconds
                    ),
                    prompt_file=str(
                        prompt_path
                    ),
                    expected_output_file=str(
                        output_path
                    ),
                )
            )

        return tasks

    @staticmethod
    def _write_manifest(
        *,
        manifest_path: Path,
        manifest: ProductionManifest,
    ) -> None:
        temporary_path = (
            manifest_path.with_suffix(
                ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(
            manifest_path
        )

    @staticmethod
    def _write_json_atomic(
        *,
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(path)

    @classmethod
    def _write_image_queue(
        cls,
        *,
        day_root: Path,
        tasks: list[ImageProductionTask],
    ) -> None:
        queue_path = (
            day_root
            / "queues"
            / "image_queue.json"
        )

        payload = {
            "version": "1.0",
            "provider": "chatgpt_manual",
            "status": "pending",
            "task_count": len(tasks),
            "tasks": [
                task.to_dict()
                for task in tasks
            ],
        }

        cls._write_json_atomic(
            path=queue_path,
            payload=payload,
        )

    @classmethod
    def _write_reel_queue(
        cls,
        *,
        day_root: Path,
        tasks: list[ReelProductionTask],
    ) -> None:
        queue_path = (
            day_root
            / "queues"
            / "reel_queue.json"
        )

        payload = {
            "version": "1.0",
            "provider": "runway_manual",
            "status": "pending",
            "task_count": len(tasks),
            "tasks": [
                task.to_dict()
                for task in tasks
            ],
        }

        cls._write_json_atomic(
            path=queue_path,
            payload=payload,
        )

    @staticmethod
    def _write_production_summary(
        *,
        day_root: Path,
        manifest: ProductionManifest,
    ) -> None:
        feed_caption = (
            manifest.captions.get(
                "feed"
            )
            or ""
        )

        reel_caption = (
            manifest.captions.get(
                "reel"
            )
            or ""
        )

        summary = f"""
# AIKO Daily Production

## Production

- Production ID: {manifest.production_id}
- Date: {manifest.production_date}
- Status: {manifest.status}

## Location

- Country: {manifest.country}
- City: {manifest.city}
- Venue: {manifest.venue}
- Theme: {manifest.theme}

## Story

{manifest.story_summary}

## Content Package

- Feed: {manifest.feed_count}
- Stories: {manifest.story_count}
- Reel scenes: {manifest.reel_scene_count}
- Image tasks: {len(manifest.image_tasks)}
- Video tasks: {len(manifest.reel_tasks)}

## Quality Gate

- Passed: {manifest.quality_gate.get("passed", False)}
- Errors: {len(manifest.quality_gate.get("errors", []))}
- Warnings: {len(manifest.quality_gate.get("warnings", []))}

## Feed Caption

{feed_caption}

## Reel Caption

{reel_caption}

## Next Manual Actions

1. Open `queues/image_queue.json`.
2. Generate Feed and Story images in ChatGPT.
3. Save the images into the expected output paths.
4. Open `queues/reel_queue.json`.
5. Generate five Reel scenes in Runway.
6. Update the task status after each file is saved.
""".strip()

        (
            day_root
            / "production_summary.md"
        ).write_text(
            summary + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_failure_record(
        *,
        day_root: Path,
        production_date: str,
        error: Exception,
    ) -> None:
        logs_root = (
            day_root
            / "logs"
        )

        logs_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        failure_path = (
            logs_root
            / "production_failure.json"
        )

        payload = {
            "production_date": production_date,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        with failure_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def status(
        self,
        production_date: str,
    ) -> dict[str, Any]:
        """
        Read the current production manifest for one date.
        """
        manifest_path = (
            self.output_root
            / production_date
            / "production_manifest.json"
        )

        if not manifest_path.exists():
            return {
                "production_date": production_date,
                "exists": False,
                "status": "not_created",
            }

        try:
            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ProductionServiceError(
                f"Invalid production manifest: {exc}"
            ) from exc

        return {
            "production_date": production_date,
            "exists": True,
            "status": data.get(
                "status",
                "unknown",
            ),
            "country": data.get("country"),
            "city": data.get("city"),
            "venue": data.get("venue"),
            "theme": data.get("theme"),
            "image_tasks": len(
                data.get(
                    "image_tasks",
                    [],
                )
            ),
            "reel_tasks": len(
                data.get(
                    "reel_tasks",
                    [],
                )
            ),
            "quality_gate": data.get(
                "quality_gate",
                {},
            ),
        }


def build_production_service() -> ProductionService:
    """
    Build the full AIKO Production Service.

    Output:

    10_apps/claude_runtime/output/YYYY-MM-DD/
    """
    runtime_root = (
        Path(__file__).resolve().parents[1]
    )

    content_service = (
        build_content_service()
    )

    return ProductionService(
        content_service=content_service,
        output_root=(
            runtime_root
            / "output"
        ),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Production Service"
    )

    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Production date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Display production status only.",
    )

    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not update planning history.",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not replace an existing production manifest.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    service = build_production_service()

    if arguments.status:
        status = service.status(
            arguments.date
        )

        print()
        print("AIKO Production Status")
        print("----------------------")

        for key, value in status.items():
            print(
                f"{key:<18} {value}"
            )

        print()
        return

    manifest = service.run(
        target_date=arguments.date,
        remember_plan=(
            not arguments.no_history
        ),
        overwrite=(
            not arguments.no_overwrite
        ),
    )

    print()
    print("AIKO Production Service")
    print("-----------------------")
    print(
        f"production ID: {manifest.production_id}"
    )
    print(
        f"date:          {manifest.production_date}"
    )
    print(
        f"country:       {manifest.country}"
    )
    print(
        f"city:          {manifest.city}"
    )
    print(
        f"venue:         {manifest.venue}"
    )
    print(
        f"theme:         {manifest.theme}"
    )
    print(
        f"status:        {manifest.status}"
    )
    print(
        f"image tasks:   {len(manifest.image_tasks)}"
    )
    print(
        f"reel tasks:    {len(manifest.reel_tasks)}"
    )
    print(
        "quality gate:  "
        f"{'passed' if manifest.quality_gate.get('passed') else 'failed'}"
    )
    print(
        f"output:        {manifest.output_directory}"
    )
    print()


if __name__ == "__main__":
    main()