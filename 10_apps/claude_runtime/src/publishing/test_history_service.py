from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .history_service import HistoryService, HistoryServiceError


class HistoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.service = HistoryService(self.temp_dir / "history")

    def test_missing_file_recovers_as_empty_history(self) -> None:
        data = self.service.load("2026-08-01")
        self.assertEqual(data["events"], [])

    def test_record_event_appends_and_preserves_prior_events(self) -> None:
        self.service.record_event(
            production_date="2026-08-01",
            job_id="job-1",
            event_type="prepared",
            previous_status=None,
            new_status="draft",
        )
        self.service.record_event(
            production_date="2026-08-01",
            job_id="job-1",
            event_type="validated",
            previous_status="draft",
            new_status="pending_approval",
        )
        events = self.service.list_events("2026-08-01")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event_type"], "prepared")
        self.assertEqual(events[1]["event_type"], "validated")

    def test_recorded_event_has_required_fields(self) -> None:
        event = self.service.record_event(
            production_date="2026-08-01",
            job_id="job-1",
            event_type="approved",
            previous_status="pending_approval",
            new_status="approved",
            details={"actor": "human"},
        )
        for field_name in (
            "event_id",
            "job_id",
            "event_type",
            "created_at",
            "previous_status",
            "new_status",
            "details",
        ):
            self.assertIn(field_name, event)
        self.assertEqual(event["details"], {"actor": "human"})

    def test_unknown_event_type_raises(self) -> None:
        with self.assertRaises(HistoryServiceError):
            self.service.record_event(
                production_date="2026-08-01",
                job_id="job-1",
                event_type="not_a_real_event",
                previous_status=None,
                new_status=None,
            )

    def test_history_is_scoped_per_production_date(self) -> None:
        self.service.record_event(
            production_date="2026-08-01",
            job_id="job-1",
            event_type="prepared",
            previous_status=None,
            new_status="draft",
        )
        self.assertEqual(len(self.service.list_events("2026-08-01")), 1)
        self.assertEqual(len(self.service.list_events("2026-08-02")), 0)

    def test_malformed_json_raises_clear_error(self) -> None:
        history_dir = self.temp_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / "history_2026-08-01.json").write_text("{bad json", encoding="utf-8")
        with self.assertRaises(HistoryServiceError):
            self.service.load("2026-08-01")

    def test_convenience_wrappers_use_correct_event_types(self) -> None:
        self.service.record_prepared(production_date="2026-08-01", job_id="job-1")
        self.service.record_validated(
            production_date="2026-08-01", job_id="job-1", previous_status="draft", new_status="pending_approval"
        )
        self.service.record_approved(production_date="2026-08-01", job_id="job-1")
        self.service.record_scheduled(production_date="2026-08-01", job_id="job-1")
        events = self.service.list_events("2026-08-01")
        self.assertEqual(
            [event["event_type"] for event in events],
            ["prepared", "validated", "approved", "scheduled"],
        )


if __name__ == "__main__":
    unittest.main()
