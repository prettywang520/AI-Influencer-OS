from __future__ import annotations

import unittest

from .models import (
    PublishJob,
    PublisherConfig,
    load_publisher_config,
    platform_for_content_type,
)


class PlatformMappingTests(unittest.TestCase):
    def test_threads_post_maps_to_threads_platform(self) -> None:
        self.assertEqual(platform_for_content_type("threads_post"), "threads")

    def test_instagram_content_types_map_to_instagram_platform(self) -> None:
        for content_type in ("instagram_feed", "instagram_story", "instagram_reel"):
            self.assertEqual(platform_for_content_type(content_type), "instagram")


class PublishJobRoundTripTests(unittest.TestCase):
    def _job(self, **overrides) -> PublishJob:
        defaults = dict(
            job_id="2026-08-01-instagram_feed",
            production_date="2026-08-01",
            persona_id="aiko",
            platform="instagram",
            content_type="instagram_feed",
        )
        defaults.update(overrides)
        return PublishJob(**defaults)

    def test_defaults(self) -> None:
        job = self._job()
        self.assertEqual(job.status, "draft")
        self.assertEqual(job.media_paths, [])
        self.assertEqual(job.hashtags, [])
        self.assertEqual(job.metadata, {})
        self.assertIsNotNone(job.created_at)
        self.assertIsNotNone(job.updated_at)

    def test_to_dict_from_dict_round_trip(self) -> None:
        job = self._job(
            media_paths=["/tmp/feed.png"],
            caption_text="hello",
            hashtags=["#a", "#b"],
            metadata={"story_order": 1},
        )
        restored = PublishJob.from_dict(job.to_dict())
        self.assertEqual(restored.job_id, job.job_id)
        self.assertEqual(restored.media_paths, job.media_paths)
        self.assertEqual(restored.metadata, job.metadata)

    def test_unknown_fields_are_preserved_across_round_trip(self) -> None:
        data = self._job().to_dict()
        data["future_field_v2"] = {"nested": True}

        job = PublishJob.from_dict(data)
        self.assertEqual(job.extra.get("future_field_v2"), {"nested": True})

        round_tripped = job.to_dict()
        self.assertEqual(round_tripped["future_field_v2"], {"nested": True})

    def test_known_fields_win_over_extra_on_write(self) -> None:
        job = self._job()
        job.extra["status"] = "this-should-never-surface"
        data = job.to_dict()
        self.assertEqual(data["status"], "draft")

    def test_touch_updates_updated_at(self) -> None:
        job = self._job()
        job.updated_at = "2000-01-01T00:00:00+00:00"
        job.touch()
        self.assertNotEqual(job.updated_at, "2000-01-01T00:00:00+00:00")


class PublisherConfigTests(unittest.TestCase):
    def test_load_real_repo_config(self) -> None:
        config = load_publisher_config()
        self.assertEqual(config.default_persona_id, "aiko")
        self.assertIn("instagram_feed", config.platform_limits)
        self.assertIn(".png", config.allowed_media_extensions)

    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_publisher_config("/nonexistent/path/publisher.yaml")

    def test_resolve_path_uses_app_root(self) -> None:
        config = PublisherConfig(paths={"queue_dir": "output/publishing/queues"})
        resolved = config.resolve_path("queue_dir", "fallback")
        self.assertTrue(str(resolved).endswith("output/publishing/queues"))


if __name__ == "__main__":
    unittest.main()
