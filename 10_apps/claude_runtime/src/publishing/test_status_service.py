from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .history_service import HistoryService
from .models import PublisherConfig
from .publisher_service import PublisherService
from .queue_service import QueueService
from .status_service import build_status_report
from .test_publisher_service import _test_config, _write_fake_production_date


class StatusServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.output_root = self.temp_dir / "output"
        self.queue_service = QueueService(self.temp_dir / "queue.json")
        self.history_service = HistoryService(self.temp_dir / "history")
        self.config: PublisherConfig = _test_config()

        self.service = PublisherService(
            config=self.config,
            queue_service=self.queue_service,
            history_service=self.history_service,
            output_root=self.output_root,
            logs_dir=self.temp_dir / "logs",
        )

        # status_service.build_status_report() reads config/queue via its
        # own factories, so patch those to point at this test's tempdir
        # instances instead of the real repo paths.
        self._patchers = [
            patch("src.publishing.status_service.load_publisher_config", return_value=self.config),
            patch("src.publishing.status_service.build_queue_service", return_value=self.queue_service),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_report_before_any_prepare(self) -> None:
        report = build_status_report("2026-08-01")
        self.assertEqual(report.total_jobs, 0)
        self.assertIn("--prepare", report.next_recommended_command)

    def test_report_counts_pending_approval_after_prepare(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        self.service.prepare("2026-08-01")

        report = build_status_report("2026-08-01")
        self.assertEqual(report.total_jobs, 6)
        self.assertEqual(report.pending_approval, 6)
        self.assertEqual(report.validation_failed, 0)
        self.assertIn("approval_service", report.next_recommended_command)

    def test_report_flags_validation_failed_jobs(self) -> None:
        date_root = _write_fake_production_date(self.output_root, "2026-08-01")
        (date_root / "images" / "feed" / "feed_01.png").unlink()
        self.service.prepare("2026-08-01")

        report = build_status_report("2026-08-01")
        self.assertEqual(report.validation_failed, 1)
        self.assertTrue(any("instagram_feed" in entry for entry in report.missing_source_files))
        self.assertIn("--force", report.next_recommended_command)

    def test_report_recommends_scheduler_once_approved(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        self.service.prepare("2026-08-01")

        for job in self.queue_service.load_jobs():
            self.queue_service.update_job(job.job_id, lambda job: setattr(job, "status", "approved"))

        report = build_status_report("2026-08-01")
        self.assertEqual(report.approved, 6)
        self.assertIn("scheduler", report.next_recommended_command)

    def test_report_scopes_by_production_date(self) -> None:
        _write_fake_production_date(self.output_root, "2026-08-01")
        _write_fake_production_date(self.output_root, "2026-08-02")
        self.service.prepare("2026-08-01")
        self.service.prepare("2026-08-02")

        report = build_status_report("2026-08-01")
        self.assertEqual(report.total_jobs, 6)


if __name__ == "__main__":
    unittest.main()
