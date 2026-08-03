from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .history_service import HistoryService, build_history_service
from .models import (
    TERMINAL_LOCKED_STATUSES,
    PublisherConfig,
    PublishJob,
    app_root,
    load_publisher_config,
    now_iso,
)
from .queue_service import QueueService, build_queue_service
from .validator import ValidationResult, validate_job, validate_story_batch

STORY_COUNT = 4


class PublisherServiceError(Exception):
    """Raised when Phase 9 source output for a date cannot be found."""


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _read_text_file(path: Path) -> str | None:
    if not path.is_file():
        return None

    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    return text or None


def _read_hashtags_file(path: Path) -> list[str]:
    if not path.is_file():
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    return [token for token in text.split() if token]


def _content_plan_location(content_plan: dict[str, Any] | None) -> str | None:
    if not content_plan:
        return None

    parts = [content_plan.get(key) for key in ("venue", "city", "country")]
    parts = [part for part in parts if part]
    return ", ".join(parts) if parts else None


def _resolve_task_media_path(task: dict[str, Any] | None) -> str | None:
    if not task or task.get("status") != "completed":
        return None

    path_str = task.get("expected_output_file")
    if not path_str:
        return None

    path = Path(path_str)
    if path.is_file() and path.stat().st_size > 0:
        return str(path)

    return None


def _feed_image_task(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None

    for task in manifest.get("image_tasks", []):
        if task.get("content_type") == "feed":
            return task

    return None


def _story_image_tasks(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not manifest:
        return []

    return [task for task in manifest.get("image_tasks", []) if task.get("content_type") == "story"]


def _build_feed_job(
    date: str,
    persona_id: str,
    date_root: Path,
    config: PublisherConfig,
    manifest: dict[str, Any] | None,
    content_plan: dict[str, Any] | None,
) -> PublishJob:
    media_path = _resolve_task_media_path(_feed_image_task(manifest))
    caption_path = date_root / config.paths.get("feed_caption_file", "captions/feed_caption.txt")
    hashtags_path = date_root / config.paths.get("feed_hashtags_file", "hashtags/feed_hashtags.txt")

    return PublishJob(
        job_id=f"{date}-instagram_feed",
        production_date=date,
        persona_id=persona_id,
        platform="instagram",
        content_type="instagram_feed",
        media_paths=[media_path] if media_path else [],
        caption_path=str(caption_path) if caption_path.is_file() else None,
        caption_text=_read_text_file(caption_path),
        hashtags_path=str(hashtags_path) if hashtags_path.is_file() else None,
        hashtags=_read_hashtags_file(hashtags_path),
        location=_content_plan_location(content_plan),
    )


def _build_story_jobs(
    date: str,
    persona_id: str,
    date_root: Path,
    config: PublisherConfig,
    manifest: dict[str, Any] | None,
    content_plan: dict[str, Any] | None,
) -> list[PublishJob]:
    story_tasks = _story_image_tasks(manifest)
    caption_template = config.paths.get("story_caption_template", "captions/story_{index}_caption.txt")
    location = _content_plan_location(content_plan)
    jobs = []

    for index in range(1, STORY_COUNT + 1):
        task = story_tasks[index - 1] if index - 1 < len(story_tasks) else None
        media_path = _resolve_task_media_path(task)
        caption_path = date_root / caption_template.format(index=index)

        jobs.append(
            PublishJob(
                job_id=f"{date}-instagram_story-{index}",
                production_date=date,
                persona_id=persona_id,
                platform="instagram",
                content_type="instagram_story",
                media_paths=[media_path] if media_path else [],
                caption_path=str(caption_path) if caption_path.is_file() else None,
                caption_text=_read_text_file(caption_path),
                location=location,
                metadata={"story_order": index},
            )
        )

    return jobs


def _build_reel_job(
    date: str,
    persona_id: str,
    date_root: Path,
    config: PublisherConfig,
    content_plan: dict[str, Any] | None,
) -> PublishJob | None:
    reel_video_path = date_root / config.paths.get("reel_final_video_file", "videos/reel_final.mp4")

    if not (reel_video_path.is_file() and reel_video_path.stat().st_size > 0):
        return None

    caption_path = date_root / config.paths.get("reel_caption_file", "captions/reel_caption.txt")
    hashtags_path = date_root / config.paths.get("reel_hashtags_file", "hashtags/reel_hashtags.txt")

    return PublishJob(
        job_id=f"{date}-instagram_reel",
        production_date=date,
        persona_id=persona_id,
        platform="instagram",
        content_type="instagram_reel",
        media_paths=[str(reel_video_path)],
        caption_path=str(caption_path) if caption_path.is_file() else None,
        caption_text=_read_text_file(caption_path),
        hashtags_path=str(hashtags_path) if hashtags_path.is_file() else None,
        hashtags=_read_hashtags_file(hashtags_path),
        location=_content_plan_location(content_plan),
    )


def _build_threads_job(
    date: str,
    persona_id: str,
    date_root: Path,
    config: PublisherConfig,
    content_plan: dict[str, Any] | None,
) -> PublishJob:
    text_path = date_root / config.paths.get("threads_text_file", "captions/threads_post.txt")
    hashtags_path = date_root / config.paths.get("threads_hashtags_file", "hashtags/threads_hashtags.txt")

    return PublishJob(
        job_id=f"{date}-threads_post",
        production_date=date,
        persona_id=persona_id,
        platform="threads",
        content_type="threads_post",
        media_paths=[],
        threads_text=_read_text_file(text_path),
        hashtags_path=str(hashtags_path) if hashtags_path.is_file() else None,
        hashtags=_read_hashtags_file(hashtags_path),
        location=_content_plan_location(content_plan),
    )


def validate_and_route(jobs: list[PublishJob], config: PublisherConfig) -> dict[str, ValidationResult]:
    """Validate every job in place, setting .status/.validation_errors, and return each result."""
    story_batch_results = validate_story_batch(jobs)
    results: dict[str, ValidationResult] = {}

    for job in jobs:
        result = validate_job(job, config)

        if job.job_id in story_batch_results:
            result = result.merge(story_batch_results[job.job_id])

        results[job.job_id] = result
        job.status = "pending_approval" if result.passed else "validation_failed"
        job.validation_errors = list(result.errors)
        job.touch()

    return results


@dataclass(slots=True)
class PrepareResult:
    production_date: str
    dry_run: bool
    jobs_built: list[PublishJob]
    jobs_created: list[str]
    duplicates_skipped: list[str]
    validation_failed: list[str]
    warnings: dict[str, list[str]] = field(default_factory=dict)
    missing_source_files: list[str] = field(default_factory=list)
    log_path: str | None = None


class PublisherService:
    def __init__(
        self,
        *,
        config: PublisherConfig | None = None,
        queue_service: QueueService | None = None,
        history_service: HistoryService | None = None,
        output_root: str | Path | None = None,
        logs_dir: str | Path | None = None,
    ) -> None:
        self.config = config or load_publisher_config()
        self.queue_service = queue_service or build_queue_service(self.config)
        self.history_service = history_service or build_history_service(self.config)
        self.output_root = Path(output_root) if output_root else app_root() / "output"
        self.logs_dir = Path(logs_dir) if logs_dir else self.config.logs_dir()

    def _date_root(self, date: str) -> Path:
        return self.output_root / date

    def build_jobs(self, date: str, persona_id: str | None = None) -> list[PublishJob]:
        persona = persona_id or self.config.default_persona_id
        date_root = self._date_root(date)

        if not date_root.is_dir():
            raise PublisherServiceError(
                f"No Phase 9 output found for date {date}: {date_root}"
            )

        manifest = _load_json(
            date_root / self.config.paths.get("production_manifest_file", "production_manifest.json")
        )
        content_plan = _load_json(
            date_root / self.config.paths.get("content_plan_file", "content_plan.json")
        )

        jobs = [_build_feed_job(date, persona, date_root, self.config, manifest, content_plan)]
        jobs.extend(_build_story_jobs(date, persona, date_root, self.config, manifest, content_plan))

        reel_job = _build_reel_job(date, persona, date_root, self.config, content_plan)
        if reel_job is not None:
            jobs.append(reel_job)

        jobs.append(_build_threads_job(date, persona, date_root, self.config, content_plan))
        return jobs

    def prepare(
        self,
        date: str,
        *,
        force: bool = False,
        dry_run: bool = False,
        persona_id: str | None = None,
    ) -> PrepareResult:
        started_at = now_iso()
        jobs = self.build_jobs(date, persona_id=persona_id)
        validation_results = validate_and_route(jobs, self.config)
        missing_source_files = [
            f"{job.job_id}: {error}"
            for job in jobs
            for error in job.validation_errors
        ]

        if dry_run:
            return PrepareResult(
                production_date=date,
                dry_run=True,
                jobs_built=jobs,
                jobs_created=[],
                duplicates_skipped=[],
                validation_failed=[job.job_id for job in jobs if job.status == "validation_failed"],
                warnings={
                    job_id: result.warnings
                    for job_id, result in validation_results.items()
                    if result.warnings
                },
                missing_source_files=missing_source_files,
            )

        existing_jobs = {job.job_id: job for job in self.queue_service.load_jobs()}
        final_jobs_by_id = dict(existing_jobs)
        jobs_created: list[str] = []
        duplicates_skipped: list[str] = []

        for job in jobs:
            existing = existing_jobs.get(job.job_id)

            if existing is not None and (not force or existing.status in TERMINAL_LOCKED_STATUSES):
                duplicates_skipped.append(job.job_id)
                continue

            final_jobs_by_id[job.job_id] = job
            jobs_created.append(job.job_id)
            self.history_service.record_prepared(production_date=job.production_date, job_id=job.job_id)
            self.history_service.record_validated(
                production_date=job.production_date,
                job_id=job.job_id,
                previous_status="draft",
                new_status=job.status,
                details={"errors": job.validation_errors} if job.validation_errors else None,
            )

        self.queue_service.save_jobs(list(final_jobs_by_id.values()))

        result = PrepareResult(
            production_date=date,
            dry_run=False,
            jobs_built=jobs,
            jobs_created=jobs_created,
            duplicates_skipped=duplicates_skipped,
            validation_failed=[
                job_id for job_id in jobs_created if final_jobs_by_id[job_id].status == "validation_failed"
            ],
            warnings={
                job_id: result_.warnings
                for job_id, result_ in validation_results.items()
                if result_.warnings and job_id in jobs_created
            },
            missing_source_files=missing_source_files,
        )
        result.log_path = str(self._write_log(started_at=started_at, result=result))
        return result

    def _write_log(self, *, started_at: str, result: PrepareResult, error: str | None = None) -> Path:
        log_dir = self.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        finished_at = now_iso()
        run_id = finished_at.replace(":", "").replace("-", "").replace("+", "").replace(".", "")
        log_path = log_dir / f"publisher_service_{result.production_date}_{run_id}.json"

        payload = {
            "started_at": started_at,
            "finished_at": finished_at,
            "command": "publisher_service --prepare",
            "production_date": result.production_date,
            "services_called": ["publisher_service", "validator", "queue_service", "history_service"],
            "jobs_created": result.jobs_created,
            "duplicates_skipped": result.duplicates_skipped,
            "validation_failed": result.validation_failed,
            "files_written": [str(self.queue_service.queue_path)],
            "result": "error" if error else "ok",
            "error": error,
        }

        temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary_path.replace(log_path)
        return log_path

    def write_error_log(self, *, started_at: str, production_date: str, error: str) -> Path:
        empty_result = PrepareResult(
            production_date=production_date,
            dry_run=False,
            jobs_built=[],
            jobs_created=[],
            duplicates_skipped=[],
            validation_failed=[],
        )
        return self._write_log(started_at=started_at, result=empty_result, error=error)


def _print_prepare_summary(result: PrepareResult) -> None:
    label = "DRY RUN" if result.dry_run else "PREPARE"
    print(f"[{label}] production_date={result.production_date}")
    print(f"  jobs built:          {len(result.jobs_built)}")

    if result.dry_run:
        for job in result.jobs_built:
            print(f"    {job.job_id:35s} {job.status:18s} media={len(job.media_paths)}")
    else:
        print(f"  jobs created/updated: {len(result.jobs_created)} ({', '.join(result.jobs_created) or '-'})")
        print(f"  duplicates skipped:   {len(result.duplicates_skipped)} ({', '.join(result.duplicates_skipped) or '-'})")

    print(f"  validation failed:   {len(result.validation_failed)} ({', '.join(result.validation_failed) or '-'})")

    if result.missing_source_files:
        print("  missing source files:")
        for entry in result.missing_source_files:
            print(f"    - {entry}")

    if result.log_path:
        print(f"  log written to:      {result.log_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIKO OS Publisher Service (Phase 10A)")
    parser.add_argument("--prepare", action="store_true", required=True, help="Prepare publish jobs from Phase 9 output")
    parser.add_argument("--date", required=True, help="Production date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate in memory only; write nothing")
    parser.add_argument("--force", action="store_true", help="Rebuild unpublished jobs for this date")
    parser.add_argument("--persona-id", default=None, help="Override the configured default persona_id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    service = PublisherService()
    started_at = now_iso()

    try:
        result = service.prepare(
            args.date,
            force=args.force,
            dry_run=args.dry_run,
            persona_id=args.persona_id,
        )
    except PublisherServiceError as exc:
        service.write_error_log(started_at=started_at, production_date=args.date, error=str(exc))
        raise SystemExit(str(exc)) from exc

    _print_prepare_summary(result)


if __name__ == "__main__":
    main()
