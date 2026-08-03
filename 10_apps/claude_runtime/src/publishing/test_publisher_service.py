from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .history_service import HistoryService
from .models import PublisherConfig, app_root
from .publisher_service import PublisherService, PublisherServiceError
from .queue_service import QueueService

PUBLISHING_PACKAGE_DIR = Path(__file__).resolve().parent


def _test_config() -> PublisherConfig:
    return PublisherConfig(
        version="1.0",
        default_persona_id="aiko",
        timezone="UTC",
        approval_required=True,
        dry_run_default=False,
        platform_limits={
            "instagram_feed": {"caption_max_characters": 2200},
            "instagram_story": {"caption_max_characters": 2200},
            "instagram_reel": {"caption_max_characters": 2200},
            "threads_post": {"text_max_characters": 500},
        },
        required_fields={},
        allowed_media_extensions=(".png", ".jpg", ".jpeg", ".mp4", ".mov"),
        paths={
            "production_manifest_file": "production_manifest.json",
            "content_plan_file": "content_plan.json",
            "feed_caption_file": "captions/feed_caption.txt",
            "feed_hashtags_file": "hashtags/feed_hashtags.txt",
            "story_caption_template": "captions/story_{index}_caption.txt",
            "reel_caption_file": "captions/reel_caption.txt",
            "reel_hashtags_file": "hashtags/reel_hashtags.txt",
            "threads_text_file": "captions/threads_post.txt",
            "threads_hashtags_file": "hashtags/threads_hashtags.txt",
            "reel_final_video_file": "videos/reel_final.mp4",
        },
    )


def _write_fake_production_date(output_root: Path, date: str, *, with_reel_video: bool = False) -> Path:
    """Builds a minimal but realistic Phase 9 output/<date>/ directory."""
    date_root = output_root / date
    (date_root / "captions").mkdir(parents=True)
    (date_root / "hashtags").mkdir()
    (date_root / "images" / "feed").mkdir(parents=True)
    (date_root / "images" / "stories").mkdir(parents=True)

    feed_image = date_root / "images" / "feed" / "feed_01.png"
    feed_image.write_bytes(b"fake-feed-image-bytes")

    story_images = []
    for index in range(1, 5):
        story_image = date_root / "images" / "stories" / f"story_0{index}.png"
        story_image.write_bytes(b"fake-story-image-bytes")
        story_images.append(story_image)
        (date_root / "captions" / f"story_{index}_caption.txt").write_text(
            f"story {index} caption", encoding="utf-8"
        )

    (date_root / "captions" / "feed_caption.txt").write_text("feed caption text", encoding="utf-8")
    (date_root / "captions" / "reel_caption.txt").write_text("reel caption text", encoding="utf-8")
    (date_root / "captions" / "threads_post.txt").write_text("threads post text", encoding="utf-8")
    (date_root / "hashtags" / "feed_hashtags.txt").write_text("#a #b #c", encoding="utf-8")
    (date_root / "hashtags" / "threads_hashtags.txt").write_text("#a", encoding="utf-8")

    manifest = {
        "image_tasks": [
            {
                "content_type": "feed",
                "status": "completed",
                "expected_output_file": str(feed_image),
            },
            *[
                {
                    "content_type": "story",
                    "status": "completed",
                    "expected_output_file": str(path),
                }
                for path in story_images
            ],
        ]
    }
    (date_root / "production_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    content_plan = {"venue": "Test Venue", "city": "Test City", "country": "Testland"}
    (date_root / "content_plan.json").write_text(json.dumps(content_plan), encoding="utf-8")

    if with_reel_video:
        (date_root / "videos").mkdir(parents=True, exist_ok=True)
        (date_root / "videos" / "reel_final.mp4").write_bytes(b"fake-final-reel-bytes")

    return date_root


class PublisherServiceFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.output_root = self.temp_dir / "output"
        self.queue_service = QueueService(self.temp_dir / "queue.json")
        self.history_service = HistoryService(self.temp_dir / "history")
        self.config = _test_config()

    def _service(self) -> PublisherService:
        return PublisherService(
            config=self.config,
            queue_service=self.queue_service,
            history_service=self.history_service,
            output_root=self.output_root,
            logs_dir=self.temp_dir / "logs",
        )


class PrepareBuildsExpectedJobsTests(PublisherServiceFixture):
    def test_prepare_builds_feed_stories_and_threads_but_no_reel(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01", with_reel_video=False)
        service = self._service()
        result = service.prepare("2026-08-01")

        content_types = sorted(job.content_type for job in result.jobs_built)
        self.assertEqual(
            content_types,
            sorted(
                ["instagram_feed", "instagram_story", "instagram_story", "instagram_story", "instagram_story", "threads_post"]
            ),
        )
        self.assertEqual(len(result.jobs_created), 6)
        self.assertEqual(result.validation_failed, [])

    def test_reel_job_only_built_when_final_video_exists(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01", with_reel_video=True)
        service = self._service()
        result = service.prepare("2026-08-01")
        reel_jobs = [job for job in result.jobs_built if job.content_type == "instagram_reel"]
        self.assertEqual(len(reel_jobs), 1)
        self.assertTrue(reel_jobs[0].media_paths[0].endswith("reel_final.mp4"))

    def test_missing_production_date_raises(self) -> None:
        service = self._service()
        with self.assertRaises(PublisherServiceError):
            service.prepare("2099-12-31")

    def test_valid_jobs_become_pending_approval(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        result = service.prepare("2026-08-01")
        queued = self.queue_service.load_jobs()
        feed_job = next(job for job in queued if job.content_type == "instagram_feed")
        self.assertEqual(feed_job.status, "pending_approval")

    def test_missing_feed_image_causes_validation_failed(self) -> None:
        date_root = _write_fake_production_date(self.output_root, "2026-08-01")
        (date_root / "images" / "feed" / "feed_01.png").unlink()
        service = self._service()
        result = service.prepare("2026-08-01")
        self.assertIn("2026-08-01-instagram_feed", result.validation_failed)

    def test_persona_id_defaults_from_config(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        service.prepare("2026-08-01")
        job = self.queue_service.get("2026-08-01-instagram_feed")
        self.assertEqual(job.persona_id, "aiko")

    def test_persona_id_override_is_preserved(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        service.prepare("2026-08-01", persona_id="taiwan_creator")
        job = self.queue_service.get("2026-08-01-instagram_feed")
        self.assertEqual(job.persona_id, "taiwan_creator")


class PrepareIdempotencyTests(PublisherServiceFixture):
    def test_prepare_twice_does_not_duplicate_jobs(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        service.prepare("2026-08-01")
        second = service.prepare("2026-08-01")
        self.assertEqual(second.jobs_created, [])
        self.assertEqual(len(second.duplicates_skipped), 6)
        self.assertEqual(len(self.queue_service.load_jobs()), 6)

    def test_force_rebuilds_unpublished_jobs(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        service.prepare("2026-08-01")

        # Simulate a human editing the queued feed caption out-of-band.
        self.queue_service.update_job(
            "2026-08-01-instagram_feed", lambda job: setattr(job, "caption_text", "edited-by-hand")
        )

        second = service.prepare("2026-08-01", force=True)
        self.assertIn("2026-08-01-instagram_feed", second.jobs_created)
        job = self.queue_service.get("2026-08-01-instagram_feed")
        self.assertEqual(job.caption_text, "feed caption text")

    def test_force_does_not_overwrite_published_jobs(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        service.prepare("2026-08-01")
        self.queue_service.update_job("2026-08-01-instagram_feed", lambda job: setattr(job, "status", "published"))

        second = service.prepare("2026-08-01", force=True)
        self.assertIn("2026-08-01-instagram_feed", second.duplicates_skipped)
        self.assertEqual(self.queue_service.get("2026-08-01-instagram_feed").status, "published")

    def test_force_does_not_overwrite_publishing_jobs(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        service.prepare("2026-08-01")
        self.queue_service.update_job("2026-08-01-instagram_feed", lambda job: setattr(job, "status", "publishing"))

        second = service.prepare("2026-08-01", force=True)
        self.assertIn("2026-08-01-instagram_feed", second.duplicates_skipped)
        self.assertEqual(self.queue_service.get("2026-08-01-instagram_feed").status, "publishing")


class DryRunTests(PublisherServiceFixture):
    def test_dry_run_writes_nothing(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        result = service.prepare("2026-08-01", dry_run=True)

        self.assertTrue(result.dry_run)
        self.assertEqual(len(result.jobs_built), 6)
        self.assertFalse(self.queue_service.queue_path.exists())
        self.assertFalse((self.temp_dir / "history").exists())
        self.assertIsNone(result.log_path)

    def test_dry_run_does_not_prevent_a_later_real_prepare(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        service.prepare("2026-08-01", dry_run=True)
        real = service.prepare("2026-08-01")
        self.assertEqual(len(real.jobs_created), 6)


class LoggingTests(PublisherServiceFixture):
    def test_prepare_writes_structured_log(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        service = self._service()
        result = service.prepare("2026-08-01")

        self.assertIsNotNone(result.log_path)
        log_data = json.loads(Path(result.log_path).read_text(encoding="utf-8"))
        for field_name in (
            "started_at",
            "finished_at",
            "command",
            "production_date",
            "services_called",
            "jobs_created",
            "duplicates_skipped",
            "validation_failed",
            "files_written",
            "result",
            "error",
        ):
            self.assertIn(field_name, log_data)
        self.assertEqual(log_data["result"], "ok")


class SafetyTests(unittest.TestCase):
    """Static/structural safety guarantees required by Phase 10A."""

    def test_no_playwright_import_anywhere_in_publishing_package(self) -> None:
        offending_lines = []

        for path in sorted(PUBLISHING_PACKAGE_DIR.glob("*.py")):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip().lower()
                if stripped.startswith("import playwright") or stripped.startswith("from playwright"):
                    offending_lines.append(f"{path}:{line_number}: {line.strip()}")

        self.assertEqual(offending_lines, [], f"playwright imported at: {offending_lines}")

    def test_social_and_config_social_are_untouched_by_a_full_workflow(self) -> None:
        social_dir = app_root() / "src" / "social"
        config_social_dir = app_root() / "config" / "social"

        def _snapshot(directory: Path) -> dict[str, str]:
            snapshot = {}
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    snapshot[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            return snapshot

        before = {**_snapshot(social_dir), **_snapshot(config_social_dir)}

        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            output_root = temp_dir / "output"
            _write_fake_production_date(output_root, "2026-08-01", with_reel_video=True)

            service = PublisherService(
                config=_test_config(),
                queue_service=QueueService(temp_dir / "queue.json"),
                history_service=HistoryService(temp_dir / "history"),
                output_root=output_root,
                logs_dir=temp_dir / "logs",
            )
            service.prepare("2026-08-01")
            service.prepare("2026-08-01", force=True)
            service.prepare("2026-08-01", dry_run=True)

        after = {**_snapshot(social_dir), **_snapshot(config_social_dir)}
        self.assertEqual(before, after)

    def test_full_workflow_never_writes_under_the_real_repo_output_publishing_dir(self) -> None:
        real_publishing_dir = app_root() / "output" / "publishing"
        before_files = set(real_publishing_dir.rglob("*")) if real_publishing_dir.exists() else set()

        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            output_root = temp_dir / "output"
            _write_fake_production_date(output_root, "2026-08-01", with_reel_video=True)

            service = PublisherService(
                config=_test_config(),
                queue_service=QueueService(temp_dir / "queue.json"),
                history_service=HistoryService(temp_dir / "history"),
                output_root=output_root,
                logs_dir=temp_dir / "logs",
            )
            service.prepare("2026-08-01")

        after_files = set(real_publishing_dir.rglob("*")) if real_publishing_dir.exists() else set()
        self.assertEqual(before_files, after_files)


if __name__ == "__main__":
    unittest.main()
