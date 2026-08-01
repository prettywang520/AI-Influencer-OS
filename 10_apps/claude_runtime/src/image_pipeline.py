from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


class ImagePipelineError(RuntimeError):
    """Raised when generated image files cannot be processed safely."""


@dataclass(slots=True)
class ImageMetadata:
    task_id: str
    content_type: str
    title: str
    source_file: str
    filename: str
    extension: str
    width: int
    height: int
    aspect_ratio: float
    colour_mode: str
    format: str
    size_bytes: int
    checksum_sha256: str
    thumbnail_file: str | None
    processed_at: str
    valid: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImageManifest:
    production_date: str
    status: str
    total_tasks: int
    completed: int
    missing: int
    invalid: int
    feed_count: int
    story_count: int
    thumbnails_created: int
    generated_at: str
    images: list[ImageMetadata] = field(default_factory=list)
    missing_tasks: list[dict[str, Any]] = field(default_factory=list)
    invalid_tasks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImagePipeline:
    """
    Validates and organises AIKO Feed and Story images.

    Pipeline:

    image_queue.json
    -> locate completed image files
    -> validate image
    -> read dimensions and metadata
    -> create thumbnail
    -> write per-image metadata
    -> write image_manifest.json
    -> update production_manifest.json
    """

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    def __init__(
        self,
        *,
        production_root: str | Path,
        thumbnail_size: tuple[int, int] = (360, 450),
    ) -> None:
        self.production_root = Path(
            production_root
        ).expanduser().resolve()

        self.thumbnail_size = thumbnail_size

        self.queue_path = (
            self.production_root
            / "queues"
            / "image_queue.json"
        )

        self.production_manifest_path = (
            self.production_root
            / "production_manifest.json"
        )

        self.image_manifest_path = (
            self.production_root
            / "image_manifest.json"
        )

        self.metadata_root = (
            self.production_root
            / "metadata"
            / "images"
        )

        self.thumbnail_root = (
            self.production_root
            / "thumbnails"
        )

        if not self.production_root.exists():
            raise ImagePipelineError(
                f"Production folder not found: {self.production_root}"
            )

        if not self.queue_path.exists():
            raise ImagePipelineError(
                f"Image queue not found: {self.queue_path}"
            )

        self.metadata_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.thumbnail_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except FileNotFoundError as exc:
            raise ImagePipelineError(
                f"JSON file not found: {path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ImagePipelineError(
                f"Invalid JSON in {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ImagePipelineError(
                f"Expected JSON object in {path}"
            )

        return data

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(path)

    def _resolve_path(
        self,
        value: str,
    ) -> Path:
        path = Path(value).expanduser()

        if path.is_absolute():
            return path.resolve()

        return (
            self.production_root
            / path
        ).resolve()

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _aspect_ratio(
        width: int,
        height: int,
    ) -> float:
        if height == 0:
            return 0.0

        return round(
            width / height,
            4,
        )

    @staticmethod
    def _aspect_warnings(
        *,
        content_type: str,
        width: int,
        height: int,
    ) -> list[str]:
        warnings: list[str] = []

        if width <= 0 or height <= 0:
            warnings.append(
                "invalid_dimensions"
            )
            return warnings

        ratio = width / height

        if content_type == "feed":
            target_ratio = 4 / 5

            if abs(ratio - target_ratio) > 0.05:
                warnings.append(
                    "feed_aspect_ratio_not_4_5"
                )

        if content_type == "story":
            target_ratio = 9 / 16

            if abs(ratio - target_ratio) > 0.04:
                warnings.append(
                    "story_aspect_ratio_not_9_16"
                )

        if width < 720 or height < 900:
            warnings.append(
                "image_resolution_low"
            )

        return warnings

    def _thumbnail_path(
        self,
        *,
        task: dict[str, Any],
        source_path: Path,
    ) -> Path:
        content_type = str(
            task.get("content_type", "image")
        )

        folder = (
            self.thumbnail_root
            / content_type
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        return (
            folder
            / f"{source_path.stem}_thumb.jpg"
        )

    def _create_thumbnail(
        self,
        *,
        source_path: Path,
        thumbnail_path: Path,
    ) -> None:
        try:
            with Image.open(source_path) as image:
                working = image.convert("RGB")
                working.thumbnail(
                    self.thumbnail_size,
                    Image.Resampling.LANCZOS,
                )

                working.save(
                    thumbnail_path,
                    format="JPEG",
                    quality=88,
                    optimize=True,
                )

        except (
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise ImagePipelineError(
                f"Unable to create thumbnail for "
                f"{source_path}: {exc}"
            ) from exc

    def _metadata_path(
        self,
        task_id: str,
    ) -> Path:
        safe_task_id = (
            task_id.replace("/", "_")
            .replace("\\", "_")
        )

        return (
            self.metadata_root
            / f"{safe_task_id}.json"
        )

    def _process_task(
        self,
        task: dict[str, Any],
        *,
        create_thumbnail: bool,
    ) -> ImageMetadata:
        task_id = str(
            task.get("task_id", "unknown")
        )

        content_type = str(
            task.get("content_type", "image")
        )

        source_path = self._resolve_path(
            str(
                task.get(
                    "expected_output_file",
                    "",
                )
            )
        )

        if not source_path.exists():
            raise ImagePipelineError(
                f"Image file not found: {source_path}"
            )

        if not source_path.is_file():
            raise ImagePipelineError(
                f"Image path is not a file: {source_path}"
            )

        if (
            source_path.suffix.lower()
            not in self.SUPPORTED_EXTENSIONS
        ):
            raise ImagePipelineError(
                f"Unsupported image extension: "
                f"{source_path.suffix}"
            )

        try:
            with Image.open(source_path) as image:
                width, height = image.size
                colour_mode = image.mode
                image_format = (
                    image.format or "unknown"
                )

                image.verify()

        except (
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise ImagePipelineError(
                f"Invalid image file {source_path}: {exc}"
            ) from exc

        warnings = self._aspect_warnings(
            content_type=content_type,
            width=width,
            height=height,
        )

        thumbnail_path: Path | None = None

        if create_thumbnail:
            thumbnail_path = self._thumbnail_path(
                task=task,
                source_path=source_path,
            )

            self._create_thumbnail(
                source_path=source_path,
                thumbnail_path=thumbnail_path,
            )

        metadata = ImageMetadata(
            task_id=task_id,
            content_type=content_type,
            title=str(
                task.get("title", "")
            ),
            source_file=str(source_path),
            filename=source_path.name,
            extension=source_path.suffix.lower(),
            width=width,
            height=height,
            aspect_ratio=self._aspect_ratio(
                width,
                height,
            ),
            colour_mode=colour_mode,
            format=image_format,
            size_bytes=source_path.stat().st_size,
            checksum_sha256=self._checksum(
                source_path
            ),
            thumbnail_file=(
                str(thumbnail_path)
                if thumbnail_path
                else None
            ),
            processed_at=self._now(),
            valid=True,
            warnings=warnings,
        )

        self._write_json(
            self._metadata_path(task_id),
            metadata.to_dict(),
        )

        return metadata

    def run(
        self,
        *,
        create_thumbnails: bool = True,
        require_all: bool = False,
    ) -> ImageManifest:
        queue = self._read_json(
            self.queue_path
        )

        tasks = queue.get("tasks")

        if not isinstance(tasks, list):
            raise ImagePipelineError(
                "image_queue.json must contain a tasks list"
            )

        images: list[ImageMetadata] = []
        missing_tasks: list[dict[str, Any]] = []
        invalid_tasks: list[dict[str, Any]] = []

        for task in tasks:
            if not isinstance(task, dict):
                continue

            source_path = self._resolve_path(
                str(
                    task.get(
                        "expected_output_file",
                        "",
                    )
                )
            )

            if not source_path.exists():
                missing_tasks.append(
                    {
                        "task_id": task.get(
                            "task_id"
                        ),
                        "expected_output_file": str(
                            source_path
                        ),
                    }
                )
                continue

            try:
                metadata = self._process_task(
                    task,
                    create_thumbnail=(
                        create_thumbnails
                    ),
                )

                images.append(metadata)

                task["status"] = "completed"
                task["processed_at"] = (
                    metadata.processed_at
                )
                task["image_metadata_file"] = str(
                    self._metadata_path(
                        metadata.task_id
                    )
                )
                task["thumbnail_file"] = (
                    metadata.thumbnail_file
                )

            except ImagePipelineError as exc:
                invalid_tasks.append(
                    {
                        "task_id": task.get(
                            "task_id"
                        ),
                        "error": str(exc),
                    }
                )

                task["status"] = "failed"
                task["error"] = str(exc)

        completed = len(images)
        missing = len(missing_tasks)
        invalid = len(invalid_tasks)
        total_tasks = len(tasks)

        if completed == total_tasks:
            status = "complete"
        elif completed > 0:
            status = "partial"
        else:
            status = "not_ready"

        if require_all and (
            missing > 0 or invalid > 0
        ):
            status = "failed"

        manifest = ImageManifest(
            production_date=self.production_root.name,
            status=status,
            total_tasks=total_tasks,
            completed=completed,
            missing=missing,
            invalid=invalid,
            feed_count=sum(
                1
                for image in images
                if image.content_type == "feed"
            ),
            story_count=sum(
                1
                for image in images
                if image.content_type == "story"
            ),
            thumbnails_created=sum(
                1
                for image in images
                if image.thumbnail_file
            ),
            generated_at=self._now(),
            images=images,
            missing_tasks=missing_tasks,
            invalid_tasks=invalid_tasks,
        )

        queue["tasks"] = tasks
        queue["status"] = (
            "completed"
            if status == "complete"
            else "in_progress"
            if completed > 0
            else "pending"
        )
        queue["updated_at"] = self._now()

        self._write_json(
            self.queue_path,
            queue,
        )

        self._write_json(
            self.image_manifest_path,
            manifest.to_dict(),
        )

        self._update_production_manifest(
            manifest
        )

        if require_all and status == "failed":
            raise ImagePipelineError(
                f"Image Pipeline incomplete: "
                f"{missing} missing, "
                f"{invalid} invalid"
            )

        return manifest

    def _update_production_manifest(
        self,
        image_manifest: ImageManifest,
    ) -> None:
        if not self.production_manifest_path.exists():
            return

        production_manifest = self._read_json(
            self.production_manifest_path
        )

        production_manifest[
            "image_pipeline"
        ] = {
            "status": image_manifest.status,
            "manifest_file": str(
                self.image_manifest_path
            ),
            "completed": image_manifest.completed,
            "missing": image_manifest.missing,
            "invalid": image_manifest.invalid,
            "thumbnails_created": (
                image_manifest.thumbnails_created
            ),
            "updated_at": self._now(),
        }

        if image_manifest.status == "complete":
            production_manifest[
                "status"
            ] = "images_complete"
        elif image_manifest.completed > 0:
            production_manifest[
                "status"
            ] = "images_in_progress"
        else:
            production_manifest[
                "status"
            ] = "ready_for_images"

        production_manifest[
            "updated_at"
        ] = self._now()

        self._write_json(
            self.production_manifest_path,
            production_manifest,
        )

    def status(self) -> dict[str, Any]:
        if not self.image_manifest_path.exists():
            return {
                "production_date": (
                    self.production_root.name
                ),
                "exists": False,
                "status": "not_processed",
            }

        manifest = self._read_json(
            self.image_manifest_path
        )

        return {
            "production_date": (
                self.production_root.name
            ),
            "exists": True,
            "status": manifest.get("status"),
            "total_tasks": manifest.get(
                "total_tasks"
            ),
            "completed": manifest.get(
                "completed"
            ),
            "missing": manifest.get(
                "missing"
            ),
            "invalid": manifest.get(
                "invalid"
            ),
            "feed_count": manifest.get(
                "feed_count"
            ),
            "story_count": manifest.get(
                "story_count"
            ),
            "thumbnails_created": (
                manifest.get(
                    "thumbnails_created"
                )
            ),
        }

    def print_status(self) -> None:
        status = self.status()

        print()
        print("AIKO Image Pipeline Status")
        print("--------------------------")

        for key, value in status.items():
            print(f"{key:<20} {value}")

        print()


def build_image_pipeline(
    production_date: str,
) -> ImagePipeline:
    runtime_root = (
        Path(__file__).resolve().parents[1]
    )

    return ImagePipeline(
        production_root=(
            runtime_root
            / "output"
            / production_date
        )
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Image Pipeline"
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Production date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Display Image Pipeline status.",
    )

    parser.add_argument(
        "--no-thumbnails",
        action="store_true",
        help="Skip thumbnail generation.",
    )

    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail unless every expected image exists and is valid.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    pipeline = build_image_pipeline(
        arguments.date
    )

    if arguments.status:
        pipeline.print_status()
        return

    manifest = pipeline.run(
        create_thumbnails=(
            not arguments.no_thumbnails
        ),
        require_all=arguments.require_all,
    )

    print()
    print("AIKO Image Pipeline")
    print("-------------------")
    print(
        f"production date:    "
        f"{manifest.production_date}"
    )
    print(
        f"status:             "
        f"{manifest.status}"
    )
    print(
        f"total tasks:        "
        f"{manifest.total_tasks}"
    )
    print(
        f"completed:          "
        f"{manifest.completed}"
    )
    print(
        f"missing:            "
        f"{manifest.missing}"
    )
    print(
        f"invalid:            "
        f"{manifest.invalid}"
    )
    print(
        f"feed images:        "
        f"{manifest.feed_count}"
    )
    print(
        f"story images:       "
        f"{manifest.story_count}"
    )
    print(
        f"thumbnails created: "
        f"{manifest.thumbnails_created}"
    )
    print(
        f"manifest:           "
        f"{pipeline.image_manifest_path}"
    )
    print()


if __name__ == "__main__":
    main()