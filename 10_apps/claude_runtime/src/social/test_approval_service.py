from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .approval_service import ApprovalService
from .instagram_models import ApprovalNotFoundError, ApprovalRecord, ApprovalServiceError

# All tests use a tempdir-backed queue. Nothing here touches Instagram.


def _record(
    stable_id: str,
    *,
    proposed_reply: str | None = "thank you so much 🤍",
    classification: str = "compliment",
    safety_route: str = "auto_eligible",
    status: str = "pending_approval",
) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=f"appr-{stable_id}",
        stable_id=stable_id,
        comment_id=None,
        post_url="https://www.instagram.com/p/ABC123/",
        username="alice",
        comment_text="nice photo!",
        classification=classification,
        proposed_reply=proposed_reply,
        safety_route=safety_route,
        status=status,
        created_at="2026-07-31T00:00:00Z",
    )


class QueuePersistenceTests(unittest.TestCase):
    def test_load_queue_returns_empty_shape_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            queue = service.load_queue()

            self.assertEqual(queue["version"], "1.0")
            self.assertEqual(queue["approvals"], [])

    def test_queue_schema_has_required_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records([_record("comment_id:1")])

            raw = json.loads(service.queue_path.read_text(encoding="utf-8"))
            self.assertIn("version", raw)
            self.assertIn("updated_at", raw)
            self.assertIn("approvals", raw)

    def test_corrupt_queue_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.queue_path.parent.mkdir(parents=True, exist_ok=True)
            service.queue_path.write_text("not json", encoding="utf-8")

            with self.assertRaises(ApprovalServiceError):
                service.load_queue()

    def test_record_round_trips_all_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records([_record("comment_id:1")])

            stored = service.get("appr-comment_id:1")

            required_fields = {
                "approval_id",
                "stable_id",
                "comment_id",
                "post_url",
                "username",
                "comment_text",
                "classification",
                "proposed_reply",
                "safety_route",
                "status",
                "created_at",
                "approved_at",
                "rejected_at",
                "sent_at",
                "error",
            }

            self.assertTrue(required_fields.issubset(stored.keys()))

    def test_never_reply_record_has_null_proposed_reply_and_skipped_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records(
                [
                    _record(
                        "comment_id:2",
                        proposed_reply=None,
                        classification="spam",
                        safety_route="never_reply",
                        status="skipped",
                    )
                ]
            )

            stored = service.get("appr-comment_id:2")
            self.assertIsNone(stored["proposed_reply"])
            self.assertEqual(stored["status"], "skipped")

    def test_initial_status_is_pending_approval_for_normal_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records([_record("comment_id:3")])

            stored = service.get("appr-comment_id:3")
            self.assertEqual(stored["status"], "pending_approval")


class DedupTests(unittest.TestCase):
    def test_never_creates_duplicate_approval_for_same_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)

            added_first = service.add_records([_record("comment_id:999")])
            added_second = service.add_records([_record("comment_id:999")])

            self.assertEqual(added_first, 1)
            self.assertEqual(added_second, 0)

            queue = service.load_queue()
            matching = [
                a for a in queue["approvals"] if a["stable_id"] == "comment_id:999"
            ]
            self.assertEqual(len(matching), 1)

    def test_dedup_persists_across_separate_service_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_run = ApprovalService(queue_dir=temp_dir)
            first_run.add_records([_record("comment_id:1000")])

            second_run = ApprovalService(queue_dir=temp_dir)
            added = second_run.add_records([_record("comment_id:1000")])

            self.assertEqual(added, 0)

    def test_mixed_batch_only_adds_new_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records([_record("a"), _record("b")])

            added = service.add_records([_record("a"), _record("b"), _record("c")])
            self.assertEqual(added, 1)

            queue = service.load_queue()
            self.assertEqual(len(queue["approvals"]), 3)


class ExistingIdsAndTextsTests(unittest.TestCase):
    def test_existing_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records([_record("a"), _record("b")])

            self.assertEqual(service.existing_stable_ids(), {"a", "b"})

    def test_existing_reply_texts_excludes_null_proposed_reply(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records(
                [
                    _record("a", proposed_reply="aww thank you 🤍"),
                    _record("b", proposed_reply=None, safety_route="never_reply"),
                ]
            )

            self.assertEqual(service.existing_reply_texts(), {"aww thank you 🤍"})


class ListPendingTests(unittest.TestCase):
    def test_list_pending_only_returns_pending_approval_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records(
                [
                    _record("a", status="pending_approval"),
                    _record("b", status="pending_approval"),
                    _record("c", proposed_reply=None, status="skipped"),
                ]
            )

            pending = service.list_pending()
            self.assertEqual({r["stable_id"] for r in pending}, {"a", "b"})


class ApproveRejectTests(unittest.TestCase):
    def test_approve_sets_status_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records([_record("a")])

            record = service.approve("appr-a")
            self.assertEqual(record["status"], "approved")
            self.assertIsNotNone(record["approved_at"])
            self.assertIsNone(record["rejected_at"])

    def test_reject_sets_status_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records([_record("a")])

            record = service.reject("appr-a")
            self.assertEqual(record["status"], "rejected")
            self.assertIsNotNone(record["rejected_at"])
            self.assertIsNone(record["approved_at"])

    def test_approve_unknown_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)

            with self.assertRaises(ApprovalNotFoundError):
                service.approve("appr-does-not-exist")

    def test_reject_unknown_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)

            with self.assertRaises(ApprovalNotFoundError):
                service.reject("appr-does-not-exist")

    def test_approve_persists_across_service_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = ApprovalService(queue_dir=temp_dir)
            first.add_records([_record("a")])
            first.approve("appr-a")

            second = ApprovalService(queue_dir=temp_dir)
            stored = second.get("appr-a")
            self.assertEqual(stored["status"], "approved")

    def test_approve_and_reject_never_write_outside_the_queue_file(self) -> None:
        """
        Approval and rejection must only update local JSON — no other
        file should be created or modified by these operations.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ApprovalService(queue_dir=temp_dir)
            service.add_records([_record("a"), _record("b")])

            before = set(Path(temp_dir).iterdir())
            service.approve("appr-a")
            service.reject("appr-b")
            after = set(Path(temp_dir).iterdir())

            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
