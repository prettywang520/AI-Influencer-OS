from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .models import PublisherConfig, PublishJob
from .validator import validate_job, validate_story_batch


def _config(**overrides) -> PublisherConfig:
    defaults = dict(
        platform_limits={
            "instagram_feed": {"caption_max_characters": 2200},
            "instagram_story": {"caption_max_characters": 2200},
            "instagram_reel": {"caption_max_characters": 2200},
            "threads_post": {"text_max_characters": 20},
        },
        allowed_media_extensions=(".png", ".jpg", ".jpeg", ".mp4", ".mov"),
    )
    defaults.update(overrides)
    return PublisherConfig(**defaults)


class ValidatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)

    def _write_file(self, name: str, content: bytes = b"fake-bytes") -> Path:
        path = self.temp_dir / name
        path.write_bytes(content)
        return path

    def _feed_job(self, **overrides) -> PublishJob:
        defaults = dict(
            job_id="job-feed",
            production_date="2026-08-01",
            persona_id="aiko",
            platform="instagram",
            content_type="instagram_feed",
            caption_text="a lovely feed caption",
        )
        defaults.update(overrides)
        return PublishJob(**defaults)

    def _story_job(self, order: int, **overrides) -> PublishJob:
        defaults = dict(
            job_id=f"job-story-{order}",
            production_date="2026-08-01",
            persona_id="aiko",
            platform="instagram",
            content_type="instagram_story",
            metadata={"story_order": order},
        )
        defaults.update(overrides)
        return PublishJob(**defaults)


class FeedValidationTests(ValidatorTestCase):
    def test_valid_feed_job_passes(self) -> None:
        image = self._write_file("feed.png")
        job = self._feed_job(media_paths=[str(image)])
        result = validate_job(job, _config())
        self.assertTrue(result.passed)
        self.assertEqual(result.errors, [])

    def test_missing_feed_image_fails(self) -> None:
        missing_path = str(self.temp_dir / "does_not_exist.png")
        job = self._feed_job(media_paths=[missing_path])
        result = validate_job(job, _config())
        self.assertFalse(result.passed)
        self.assertTrue(any("does not exist" in error for error in result.errors))

    def test_zero_byte_feed_media_fails(self) -> None:
        image = self._write_file("empty.png", content=b"")
        job = self._feed_job(media_paths=[str(image)])
        result = validate_job(job, _config())
        self.assertFalse(result.passed)
        self.assertTrue(any("empty" in error for error in result.errors))

    def test_empty_caption_fails(self) -> None:
        image = self._write_file("feed.png")
        job = self._feed_job(media_paths=[str(image)], caption_text="   ")
        result = validate_job(job, _config())
        self.assertFalse(result.passed)

    def test_missing_alt_text_warns_but_still_passes(self) -> None:
        image = self._write_file("feed.png")
        job = self._feed_job(media_paths=[str(image)], alt_text=None)
        result = validate_job(job, _config())
        self.assertTrue(result.passed)
        self.assertTrue(any("alt text" in warning for warning in result.warnings))


class StoryValidationTests(ValidatorTestCase):
    def test_four_ordered_story_jobs_validate(self) -> None:
        jobs = []
        for order in range(1, 5):
            image = self._write_file(f"story_{order}.png")
            jobs.append(self._story_job(order, media_paths=[str(image)]))

        batch_results = validate_story_batch(jobs)
        self.assertEqual(batch_results, {})

        for job in jobs:
            result = validate_job(job, _config())
            self.assertTrue(result.passed, result.errors)

    def test_duplicate_story_order_fails(self) -> None:
        image_a = self._write_file("story_a.png")
        image_b = self._write_file("story_b.png")
        job_a = self._story_job(1, job_id="job-a", media_paths=[str(image_a)])
        job_b = self._story_job(1, job_id="job-b", media_paths=[str(image_b)])

        batch_results = validate_story_batch([job_a, job_b])
        self.assertIn("job-a", batch_results)
        self.assertIn("job-b", batch_results)
        self.assertFalse(batch_results["job-a"].passed)

    def test_missing_story_order_fails(self) -> None:
        image = self._write_file("story.png")
        job = self._story_job(1, media_paths=[str(image)], metadata={})
        result = validate_job(job, _config())
        self.assertFalse(result.passed)
        self.assertTrue(any("story order" in error.lower() for error in result.errors))

    def test_zero_byte_story_media_fails(self) -> None:
        image = self._write_file("story.png", content=b"")
        job = self._story_job(1, media_paths=[str(image)])
        result = validate_job(job, _config())
        self.assertFalse(result.passed)


class ReelValidationTests(ValidatorTestCase):
    def test_reel_requires_final_video(self) -> None:
        job = PublishJob(
            job_id="job-reel",
            production_date="2026-08-01",
            persona_id="aiko",
            platform="instagram",
            content_type="instagram_reel",
            caption_text="reel caption",
            media_paths=[],
        )
        result = validate_job(job, _config())
        self.assertFalse(result.passed)
        self.assertTrue(any("exactly one final video" in error for error in result.errors))

    def test_valid_reel_job_passes(self) -> None:
        video = self._write_file("reel_final.mp4")
        job = PublishJob(
            job_id="job-reel",
            production_date="2026-08-01",
            persona_id="aiko",
            platform="instagram",
            content_type="instagram_reel",
            caption_text="reel caption",
            media_paths=[str(video)],
        )
        result = validate_job(job, _config())
        self.assertTrue(result.passed)


class ThreadsValidationTests(ValidatorTestCase):
    def test_threads_character_limit(self) -> None:
        job = PublishJob(
            job_id="job-threads",
            production_date="2026-08-01",
            persona_id="aiko",
            platform="threads",
            content_type="threads_post",
            threads_text="x" * 21,
        )
        result = validate_job(job, _config())
        self.assertFalse(result.passed)
        self.assertTrue(any("exceeds maximum characters" in error for error in result.errors))

    def test_threads_within_limit_passes(self) -> None:
        job = PublishJob(
            job_id="job-threads",
            production_date="2026-08-01",
            persona_id="aiko",
            platform="threads",
            content_type="threads_post",
            threads_text="short text",
        )
        result = validate_job(job, _config())
        self.assertTrue(result.passed)

    def test_threads_empty_text_fails(self) -> None:
        job = PublishJob(
            job_id="job-threads",
            production_date="2026-08-01",
            persona_id="aiko",
            platform="threads",
            content_type="threads_post",
            threads_text="",
        )
        result = validate_job(job, _config())
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
