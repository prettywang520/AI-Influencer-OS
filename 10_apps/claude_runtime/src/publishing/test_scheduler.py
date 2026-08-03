from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .history_service import HistoryService
from .models import PublishJob
from .queue_service import QueueService
from .scheduler import PublishScheduler, SchedulerError


def _job(job_id: str, status: str, **overrides) -> PublishJob:
    defaults = dict(
        job_id=job_id,
        production_date="2026-08-01",
        persona_id="aiko",
        platform="instagram",
        content_type="instagram_feed",
        status=status,
    )
    defaults.update(overrides)
    return PublishJob(**defaults)


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.queue_service = QueueService(self.temp_dir / "queue.json")
        self.history_service = HistoryService(self.temp_dir / "history")
        self.scheduler = PublishScheduler(self.queue_service, self.history_service, "UTC")

    def _future_iso(self, hours: int = 2) -> str:
        return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()

    def test_only_approved_job_can_be_scheduled(self) -> None:
        self.queue_service.add_job(_job("a", "approved"))
        job = self.scheduler.schedule("a", self._future_iso())
        self.assertEqual(job.status, "scheduled")
        self.assertIsNotNone(job.scheduled_at)

    def test_pending_approval_job_cannot_be_scheduled(self) -> None:
        self.queue_service.add_job(_job("a", "pending_approval"))
        with self.assertRaises(SchedulerError):
            self.scheduler.schedule("a", self._future_iso())

    def test_rejected_job_cannot_be_scheduled(self) -> None:
        self.queue_service.add_job(_job("a", "rejected"))
        with self.assertRaises(SchedulerError):
            self.scheduler.schedule("a", self._future_iso())

    def test_draft_job_cannot_be_scheduled(self) -> None:
        self.queue_service.add_job(_job("a", "draft"))
        with self.assertRaises(SchedulerError):
            self.scheduler.schedule("a", self._future_iso())

    def test_published_job_cannot_be_rescheduled(self) -> None:
        self.queue_service.add_job(_job("a", "published"))
        with self.assertRaises(SchedulerError):
            self.scheduler.schedule("a", self._future_iso())

    def test_past_schedule_time_fails(self) -> None:
        self.queue_service.add_job(_job("a", "approved"))
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with self.assertRaises(SchedulerError):
            self.scheduler.schedule("a", past)

    def test_invalid_iso_datetime_fails(self) -> None:
        self.queue_service.add_job(_job("a", "approved"))
        with self.assertRaises(SchedulerError):
            self.scheduler.schedule("a", "not-a-datetime")

    def test_unknown_job_id_fails(self) -> None:
        with self.assertRaises(SchedulerError):
            self.scheduler.schedule("missing", self._future_iso())

    def test_schedule_records_history_event(self) -> None:
        self.queue_service.add_job(_job("a", "approved"))
        self.scheduler.schedule("a", self._future_iso())
        events = self.history_service.list_events("2026-08-01")
        self.assertEqual(events[-1]["event_type"], "scheduled")
        self.assertEqual(events[-1]["new_status"], "scheduled")

    def test_list_for_date_returns_only_scheduled_jobs(self) -> None:
        self.queue_service.save_jobs(
            [
                _job("a", "approved"),
                _job("b", "scheduled", scheduled_at=self._future_iso()),
                _job("c", "scheduled", scheduled_at=self._future_iso(), production_date="2026-08-02"),
            ]
        )
        scheduled = self.scheduler.list_for_date("2026-08-01")
        self.assertEqual([job.job_id for job in scheduled], ["b"])

    def test_naive_datetime_uses_configured_timezone(self) -> None:
        self.queue_service.add_job(_job("a", "approved"))
        naive_future = (datetime.now(timezone.utc) + timedelta(hours=3)).replace(tzinfo=None).isoformat()
        job = self.scheduler.schedule("a", naive_future)
        self.assertIsNotNone(job.scheduled_at)


if __name__ == "__main__":
    unittest.main()
