from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .approval_service import ApprovalError, PublishApprovalService
from .history_service import HistoryService
from .models import PublishJob
from .queue_service import QueueService


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


class ApprovalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.queue_service = QueueService(self.temp_dir / "queue.json")
        self.history_service = HistoryService(self.temp_dir / "history")
        self.service = PublishApprovalService(self.queue_service, self.history_service)

    def test_list_pending_returns_only_pending_approval_jobs(self) -> None:
        self.queue_service.save_jobs(
            [
                _job("a", "pending_approval"),
                _job("b", "draft"),
                _job("c", "approved"),
            ]
        )
        pending = self.service.list_pending()
        self.assertEqual([job.job_id for job in pending], ["a"])

    def test_approve_pending_job_succeeds(self) -> None:
        self.queue_service.add_job(_job("a", "pending_approval"))
        job = self.service.approve("a")
        self.assertEqual(job.status, "approved")
        self.assertIsNotNone(job.approved_at)
        self.assertEqual(self.queue_service.get("a").status, "approved")

    def test_reject_pending_job_succeeds(self) -> None:
        self.queue_service.add_job(_job("a", "pending_approval"))
        job = self.service.reject("a")
        self.assertEqual(job.status, "rejected")

    def test_only_pending_approval_can_be_approved(self) -> None:
        self.queue_service.add_job(_job("a", "draft"))
        with self.assertRaises(ApprovalError):
            self.service.approve("a")

    def test_only_pending_approval_can_be_rejected(self) -> None:
        self.queue_service.add_job(_job("a", "approved"))
        with self.assertRaises(ApprovalError):
            self.service.reject("a")

    def test_published_job_cannot_be_changed(self) -> None:
        self.queue_service.add_job(_job("a", "published"))
        with self.assertRaises(ApprovalError):
            self.service.approve("a")
        with self.assertRaises(ApprovalError):
            self.service.reject("a")
        self.assertEqual(self.queue_service.get("a").status, "published")

    def test_publishing_job_cannot_be_changed(self) -> None:
        self.queue_service.add_job(_job("a", "publishing"))
        with self.assertRaises(ApprovalError):
            self.service.approve("a")

    def test_approve_unknown_job_raises(self) -> None:
        with self.assertRaises(ApprovalError):
            self.service.approve("does-not-exist")

    def test_approve_records_history_event(self) -> None:
        self.queue_service.add_job(_job("a", "pending_approval"))
        self.service.approve("a")
        events = self.history_service.list_events("2026-08-01")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "approved")
        self.assertEqual(events[0]["previous_status"], "pending_approval")
        self.assertEqual(events[0]["new_status"], "approved")


if __name__ == "__main__":
    unittest.main()
