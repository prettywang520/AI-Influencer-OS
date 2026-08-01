from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .dm_approval_service import (
    DMApprovalNotFoundError,
    DMApprovalService,
    build_dm_approval_service,
    prepare_dm_replies,
)
from .dm_reply_brain import build_dm_reply_brain

# None of these tests access the real Instagram website.


def _pending_message(**overrides) -> dict:
    message = {
        "source": "instagram_dm",
        "stable_id": "message_id:1",
        "conversation_id": "conv-1",
        "thread_url": "https://www.instagram.com/direct/t/conv-1/",
        "username": "fan_one",
        "display_name": "Fan One",
        "message_id": "1",
        "message_text": "you are so beautiful, love this!",
        "timestamp_text": "2h",
        "message_type": "text",
        "status": "pending",
        "discovered_at": "2026-07-31T00:00:00Z",
        "classification": None,
        "proposed_reply": None,
        "safety_route": None,
        "error": None,
    }
    message.update(overrides)
    return message


def _write_dm_queue(path: Path, messages: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": "1.0", "updated_at": None, "messages": messages},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class DMApprovalServiceCrudTests(unittest.TestCase):
    def test_missing_queue_returns_empty_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DMApprovalService(queue_dir=temp_dir)
            queue = service.load_queue()

            self.assertEqual(queue["version"], "1.0")
            self.assertEqual(queue["approvals"], [])

    def test_approve_updates_status_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DMApprovalService(queue_dir=temp_dir)
            from .dm_approval_service import DMApprovalRecord

            service.add_records(
                [
                    DMApprovalRecord(
                        approval_id="appr-dm-1",
                        stable_id="message_id:1",
                        conversation_id="conv-1",
                        thread_url="https://x/t/conv-1/",
                        username="fan_one",
                        display_name="Fan One",
                        message_text="hi",
                        message_type="text",
                        classification="compliment",
                        proposed_reply="thank you 🤍",
                        safety_route="auto_eligible",
                        status="pending_approval",
                        created_at="2026-07-31T00:00:00Z",
                    )
                ]
            )

            record = service.approve("appr-dm-1")
            self.assertEqual(record["status"], "approved")
            self.assertIsNotNone(record["approved_at"])

            stored = service.get("appr-dm-1")
            self.assertEqual(stored["status"], "approved")

    def test_reject_updates_status_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DMApprovalService(queue_dir=temp_dir)
            from .dm_approval_service import DMApprovalRecord

            service.add_records(
                [
                    DMApprovalRecord(
                        approval_id="appr-dm-2",
                        stable_id="message_id:2",
                        conversation_id="conv-2",
                        thread_url="https://x/t/conv-2/",
                        username="fan_two",
                        display_name="Fan Two",
                        message_text="hi",
                        message_type="text",
                        classification="compliment",
                        proposed_reply="thank you 🤍",
                        safety_route="auto_eligible",
                        status="pending_approval",
                        created_at="2026-07-31T00:00:00Z",
                    )
                ]
            )

            record = service.reject("appr-dm-2")
            self.assertEqual(record["status"], "rejected")
            self.assertIsNotNone(record["rejected_at"])

    def test_approve_unknown_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DMApprovalService(queue_dir=temp_dir)

            with self.assertRaises(DMApprovalNotFoundError):
                service.approve("does-not-exist")

    def test_reject_unknown_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DMApprovalService(queue_dir=temp_dir)

            with self.assertRaises(DMApprovalNotFoundError):
                service.reject("does-not-exist")

    def test_atomic_write_leaves_no_tmp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DMApprovalService(queue_dir=temp_dir)
            from .dm_approval_service import DMApprovalRecord

            service.add_records(
                [
                    DMApprovalRecord(
                        approval_id="appr-dm-3",
                        stable_id="message_id:3",
                        conversation_id="conv-3",
                        thread_url="https://x/t/conv-3/",
                        username="fan_three",
                        display_name="Fan Three",
                        message_text="hi",
                        message_type="text",
                        classification="compliment",
                        proposed_reply="thank you 🤍",
                        safety_route="auto_eligible",
                        status="pending_approval",
                        created_at="2026-07-31T00:00:00Z",
                    )
                ]
            )

            leftover = list(Path(temp_dir).glob("*.tmp"))
            self.assertEqual(leftover, [])

            data = json.loads(service.queue_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["approvals"]), 1)

    def test_build_dm_approval_service_default_queue_dir(self) -> None:
        service = build_dm_approval_service()
        self.assertTrue(str(service.queue_path).endswith("dm_approval_queue.json"))


class PrepareDMRepliesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dm_queue_path = Path(self.temp_dir.name) / "dm_queue.json"
        self.approval_service = DMApprovalService(queue_dir=self.temp_dir.name)
        self.dm_reply_brain = build_dm_reply_brain(random_seed=5)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_auto_eligible_message_gets_pending_approval_reply(self) -> None:
        _write_dm_queue(self.dm_queue_path, [_pending_message()])

        result = prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        self.assertEqual(result.messages_processed, 1)
        self.assertEqual(result.replies_proposed, 1)
        self.assertEqual(result.auto_eligible, 1)
        self.assertEqual(result.skipped, 0)

        pending = self.approval_service.list_pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["classification"], "compliment")
        self.assertEqual(pending[0]["safety_route"], "auto_eligible")
        self.assertIsNotNone(pending[0]["proposed_reply"])
        self.assertEqual(pending[0]["status"], "pending_approval")

        updated_queue = json.loads(self.dm_queue_path.read_text(encoding="utf-8"))
        message = updated_queue["messages"][0]
        self.assertEqual(message["status"], "processed")
        self.assertEqual(message["classification"], "compliment")
        self.assertIsNotNone(message["proposed_reply"])

    def test_never_reply_message_is_skipped_with_null_reply(self) -> None:
        _write_dm_queue(
            self.dm_queue_path,
            [_pending_message(stable_id="message_id:2", message_text="send nudes")],
        )

        result = prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.replies_proposed, 0)

        queue = self.approval_service.load_queue()
        record = queue["approvals"][0]
        self.assertEqual(record["status"], "skipped")
        self.assertIsNone(record["proposed_reply"])
        self.assertEqual(record["safety_route"], "never_reply")

        updated_queue = json.loads(self.dm_queue_path.read_text(encoding="utf-8"))
        message = updated_queue["messages"][0]
        self.assertEqual(message["status"], "skipped")
        self.assertIsNone(message["proposed_reply"])

    def test_harassment_message_is_skipped_with_null_reply(self) -> None:
        _write_dm_queue(
            self.dm_queue_path,
            [
                _pending_message(
                    stable_id="message_id:3", message_text="kys, i hate you so much"
                )
            ],
        )

        prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        record = self.approval_service.load_queue()["approvals"][0]
        self.assertEqual(record["classification"], "harassment")
        self.assertEqual(record["status"], "skipped")
        self.assertIsNone(record["proposed_reply"])

    def test_brand_collaboration_routes_to_human_review_with_professional_draft(
        self,
    ) -> None:
        _write_dm_queue(
            self.dm_queue_path,
            [
                _pending_message(
                    stable_id="message_id:4",
                    message_text="we would love a paid collaboration with your account",
                )
            ],
        )

        prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        record = self.approval_service.load_queue()["approvals"][0]
        self.assertEqual(record["classification"], "brand_collaboration")
        self.assertEqual(record["safety_route"], "human_review_required")
        self.assertEqual(record["status"], "pending_approval")
        self.assertIsNotNone(record["proposed_reply"])

    def test_booking_request_routes_to_human_review_with_professional_draft(
        self,
    ) -> None:
        _write_dm_queue(
            self.dm_queue_path,
            [
                _pending_message(
                    stable_id="message_id:5",
                    message_text="hi, i'd like to book you for an event",
                )
            ],
        )

        prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        record = self.approval_service.load_queue()["approvals"][0]
        self.assertEqual(record["classification"], "booking_request")
        self.assertEqual(record["safety_route"], "human_review_required")
        self.assertIsNotNone(record["proposed_reply"])
        self.assertIn("team", record["proposed_reply"])

    def test_unknown_media_attachment_is_prepared_for_human_review(self) -> None:
        _write_dm_queue(
            self.dm_queue_path,
            [
                _pending_message(
                    stable_id="message_id:6",
                    message_text=None,
                    message_type="unknown",
                )
            ],
        )

        prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        record = self.approval_service.load_queue()["approvals"][0]
        self.assertEqual(record["classification"], "unknown")
        self.assertEqual(record["safety_route"], "human_review_required")
        self.assertEqual(record["message_type"], "unknown")

    def test_non_pending_messages_are_skipped_by_prepare(self) -> None:
        _write_dm_queue(
            self.dm_queue_path,
            [_pending_message(status="processed")],
        )

        result = prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        self.assertEqual(result.messages_processed, 0)
        self.assertEqual(len(self.approval_service.load_queue()["approvals"]), 0)

    def test_duplicate_prevention_across_repeated_prepare_runs(self) -> None:
        _write_dm_queue(self.dm_queue_path, [_pending_message()])

        prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        # Second run over the SAME (now "processed") queue file must
        # not create a second approval record for the same stable_id,
        # even if the message were somehow re-marked pending.
        queue_after_first = json.loads(self.dm_queue_path.read_text(encoding="utf-8"))
        queue_after_first["messages"][0]["status"] = "pending"
        self.dm_queue_path.write_text(
            json.dumps(queue_after_first, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result = prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        self.assertEqual(result.duplicates_skipped, 1)
        self.assertEqual(
            len(self.approval_service.load_queue()["approvals"]), 1
        )

    def test_missing_dm_queue_produces_zero_results(self) -> None:
        result = prepare_dm_replies(
            dm_queue_path=self.dm_queue_path,
            dm_approval_service=self.approval_service,
            dm_reply_brain=self.dm_reply_brain,
        )

        self.assertEqual(result.messages_processed, 0)
        self.assertEqual(result.replies_proposed, 0)


if __name__ == "__main__":
    unittest.main()
