from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .models import PublishJob
from .queue_service import PublishQueueError, QueueService


def _job(job_id: str, **overrides) -> PublishJob:
    defaults = dict(
        job_id=job_id,
        production_date="2026-08-01",
        persona_id="aiko",
        platform="instagram",
        content_type="instagram_feed",
    )
    defaults.update(overrides)
    return PublishJob(**defaults)


class QueueServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.queue_path = self.temp_dir / "queues" / "publish_queue.json"
        self.service = QueueService(self.queue_path)

    def test_missing_file_recovers_as_empty_queue_without_writing(self) -> None:
        data = self.service.load()
        self.assertEqual(data["jobs"], [])
        self.assertFalse(self.queue_path.exists())

    def test_save_and_load_round_trip(self) -> None:
        self.service.save_jobs([_job("a"), _job("b")])
        loaded = self.service.load_jobs()
        self.assertEqual({job.job_id for job in loaded}, {"a", "b"})

    def test_no_tmp_file_left_behind_after_atomic_write(self) -> None:
        self.service.save_jobs([_job("a")])
        tmp_path = self.queue_path.with_suffix(self.queue_path.suffix + ".tmp")
        self.assertTrue(self.queue_path.exists())
        self.assertFalse(tmp_path.exists())

    def test_deterministic_ordering(self) -> None:
        self.service.save_jobs(
            [
                _job("z-job", production_date="2026-08-02"),
                _job("a-job", production_date="2026-08-01"),
                _job("m-job", production_date="2026-08-01"),
            ]
        )
        loaded = self.service.load_jobs()
        self.assertEqual([job.job_id for job in loaded], ["a-job", "m-job", "z-job"])

    def test_save_jobs_rejects_duplicate_job_id(self) -> None:
        with self.assertRaises(PublishQueueError):
            self.service.save_jobs([_job("a"), _job("a")])

    def test_add_job_rejects_duplicate_job_id(self) -> None:
        self.service.add_job(_job("a"))
        with self.assertRaises(PublishQueueError):
            self.service.add_job(_job("a"))

    def test_upsert_jobs_replaces_existing(self) -> None:
        self.service.add_job(_job("a", caption_text="first"))
        self.service.upsert_jobs([_job("a", caption_text="second")])
        job = self.service.get("a")
        self.assertEqual(job.caption_text, "second")

    def test_malformed_json_raises_clear_error(self) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue_path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(PublishQueueError) as ctx:
            self.service.load()
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_non_object_json_raises_clear_error(self) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue_path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(PublishQueueError):
            self.service.load()

    def test_update_job_mutates_and_persists(self) -> None:
        self.service.add_job(_job("a", status="pending_approval"))
        updated = self.service.update_job("a", lambda job: setattr(job, "status", "approved"))
        self.assertEqual(updated.status, "approved")
        self.assertEqual(self.service.get("a").status, "approved")

    def test_update_job_missing_raises(self) -> None:
        with self.assertRaises(PublishQueueError):
            self.service.update_job("missing", lambda job: None)


if __name__ == "__main__":
    unittest.main()
