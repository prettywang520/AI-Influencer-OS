from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import PublishJob, PublisherConfig


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        return ValidationResult(
            passed=self.passed and other.passed,
            errors=[*self.errors, *other.errors],
            warnings=[*self.warnings, *other.warnings],
        )


def _is_nonzero_file(path_str: str | None) -> bool:
    if not path_str:
        return False

    path = Path(path_str)
    return path.is_file() and path.stat().st_size > 0


def _check_extension(path_str: str, config: PublisherConfig, errors: list[str], label: str) -> None:
    if not config.allowed_media_extensions:
        return

    extension = Path(path_str).suffix.lower()
    if extension not in config.allowed_media_extensions:
        errors.append(f"{label} has an unsupported media extension: {extension or '(none)'}")


def _validate_feed(job: PublishJob, config: PublisherConfig) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if len(job.media_paths) != 1:
        errors.append(
            f"Instagram Feed requires exactly one image, found {len(job.media_paths)}"
        )
    elif not _is_nonzero_file(job.media_paths[0]):
        errors.append(f"Feed image does not exist or is empty: {job.media_paths[0]}")
    else:
        _check_extension(job.media_paths[0], config, errors, "Feed image")

    if not job.caption_text or not job.caption_text.strip():
        errors.append("Feed caption is required and must not be empty")
    else:
        limit = config.platform_limits.get("instagram_feed", {}).get("caption_max_characters")
        if limit and len(job.caption_text) > limit:
            errors.append(
                f"Feed caption exceeds maximum characters ({len(job.caption_text)} > {limit})"
            )

    if not job.alt_text or not job.alt_text.strip():
        warnings.append("Feed alt text is recommended but missing")

    return ValidationResult(passed=not errors, errors=errors, warnings=warnings)


def _validate_story(job: PublishJob, config: PublisherConfig) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if len(job.media_paths) != 1:
        errors.append(
            f"Instagram Story requires exactly one image or video, found {len(job.media_paths)}"
        )
    elif not _is_nonzero_file(job.media_paths[0]):
        errors.append(f"Story media does not exist or is empty: {job.media_paths[0]}")
    else:
        _check_extension(job.media_paths[0], config, errors, "Story media")

    story_order = job.metadata.get("story_order")
    if story_order is None:
        errors.append("Story order is required in metadata")
    elif not isinstance(story_order, int) or isinstance(story_order, bool):
        errors.append("Story order metadata must be an integer")

    if job.caption_text is not None and not job.caption_text.strip():
        warnings.append("Story caption is present but empty")

    return ValidationResult(passed=not errors, errors=errors, warnings=warnings)


def _validate_reel(job: PublishJob, config: PublisherConfig) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if len(job.media_paths) != 1:
        errors.append(
            f"Instagram Reel requires exactly one final video, found {len(job.media_paths)}"
        )
    elif not _is_nonzero_file(job.media_paths[0]):
        errors.append(f"Reel video does not exist or is empty: {job.media_paths[0]}")
    else:
        _check_extension(job.media_paths[0], config, errors, "Reel video")

    if not job.caption_text or not job.caption_text.strip():
        errors.append("Reel caption is required and must not be empty")

    return ValidationResult(passed=not errors, errors=errors, warnings=warnings)


def _validate_threads(job: PublishJob, config: PublisherConfig) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not job.threads_text or not job.threads_text.strip():
        errors.append("Threads post text is required and must not be empty")
    else:
        limit = config.platform_limits.get("threads_post", {}).get("text_max_characters")
        if limit and len(job.threads_text) > limit:
            errors.append(
                f"Threads text exceeds maximum characters ({len(job.threads_text)} > {limit})"
            )

    for media_path in job.media_paths:
        if not _is_nonzero_file(media_path):
            errors.append(f"Threads media does not exist or is empty: {media_path}")
        else:
            _check_extension(media_path, config, errors, "Threads media")

    return ValidationResult(passed=not errors, errors=errors, warnings=warnings)


_VALIDATORS = {
    "instagram_feed": _validate_feed,
    "instagram_story": _validate_story,
    "instagram_reel": _validate_reel,
    "threads_post": _validate_threads,
}


def validate_job(job: PublishJob, config: PublisherConfig) -> ValidationResult:
    validator = _VALIDATORS.get(job.content_type)

    if validator is None:
        return ValidationResult(
            passed=False,
            errors=[f"Unknown publish content_type: {job.content_type}"],
        )

    return validator(job, config)


def validate_story_batch(jobs: list[PublishJob]) -> dict[str, ValidationResult]:
    """
    Cross-job check: story order (metadata['story_order']) must be unique
    among sibling instagram_story jobs. A single job's own validate_job()
    call cannot see its siblings, so this is run separately by the
    orchestrator across the full batch of story jobs for one production
    date and merged into each affected job's result.
    """
    orders: dict[int, list[str]] = {}

    for job in jobs:
        if job.content_type != "instagram_story":
            continue

        order = job.metadata.get("story_order")
        if isinstance(order, int) and not isinstance(order, bool):
            orders.setdefault(order, []).append(job.job_id)

    results: dict[str, ValidationResult] = {}

    for order, job_ids in orders.items():
        if len(job_ids) <= 1:
            continue

        for job_id in job_ids:
            others = ", ".join(sorted(jid for jid in job_ids if jid != job_id))
            results[job_id] = ValidationResult(
                passed=False,
                errors=[f"Duplicate story order {order}, also used by: {others}"],
            )

    return results
