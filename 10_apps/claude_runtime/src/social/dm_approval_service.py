from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dm_reply_brain import DMReplyBrain, build_dm_reply_brain, resolve_dm_safety_route
from .instagram_dm_reader import DMReaderError
from .instagram_models import InstagramSessionError

# ---------------------------------------------------------------------------
# Phase 8F — Instagram DM Approval Service
# ---------------------------------------------------------------------------
#
# Local-only CRUD over output/community/queues/dm_approval_queue.json,
# plus --prepare-replies orchestration (classify each pending DM,
# route it through the safety gate, propose an Aiko-style reply where
# allowed). Approval and rejection only ever update local JSON.
# Nothing in this module sends anything to Instagram — there is no
# send/dispatch capability anywhere in this file.

QUEUE_FILENAME = "dm_approval_queue.json"
DM_QUEUE_FILENAME = "dm_queue.json"


class DMApprovalServiceError(InstagramSessionError):
    """Base error for the local DM approval queue service."""


class DMApprovalNotFoundError(DMApprovalServiceError):
    """Raised when an approval_id does not exist in the DM approval queue."""


@dataclass(slots=True)
class DMApprovalRecord:
    """
    One proposed (or skipped) DM reply awaiting human approval.

    Stored in output/community/queues/dm_approval_queue.json.
    """

    approval_id: str
    stable_id: str
    conversation_id: str | None
    thread_url: str | None
    username: str | None
    display_name: str | None
    message_text: str | None
    message_type: str | None
    classification: str
    proposed_reply: str | None
    safety_route: str
    status: str
    created_at: str
    approved_at: str | None = None
    rejected_at: str | None = None
    sent_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DMPrepareRepliesResult:
    """Summary of one dm_approval_service --prepare-replies run."""

    messages_processed: int
    replies_proposed: int
    auto_eligible: int
    human_review_required: int
    skipped: int
    duplicates_skipped: int
    dm_approval_queue_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _runtime_root() -> Path:
    """
    dm_approval_service.py location:

    10_apps/claude_runtime/src/social/dm_approval_service.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_dm_queue_path() -> Path:
    return _runtime_root() / "output" / "community" / "queues" / DM_QUEUE_FILENAME


class DMApprovalService:
    """
    Local-only CRUD over output/community/queues/dm_approval_queue.json.

    Approval and rejection only ever update this local JSON file.
    Nothing in this class sends a DM or anything else to Instagram.
    """

    def __init__(self, *, queue_dir: str | Path) -> None:
        self.queue_dir = Path(queue_dir)

    @property
    def queue_path(self) -> Path:
        return self.queue_dir / QUEUE_FILENAME

    @staticmethod
    def _now() -> str:
        return _now()

    def load_queue(self) -> dict[str, Any]:
        path = self.queue_path

        if not path.exists():
            return {"version": "1.0", "updated_at": None, "approvals": []}

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise DMApprovalServiceError(
                f"Invalid JSON in DM approval queue: {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise DMApprovalServiceError(
                f"DM approval queue must be a JSON object: {path}"
            )

        data.setdefault("version", "1.0")
        data.setdefault("approvals", [])

        if not isinstance(data["approvals"], list):
            raise DMApprovalServiceError(
                f"DM approval queue 'approvals' must be a list: {path}"
            )

        return data

    def save_queue(self, queue: dict[str, Any]) -> None:
        path = self.queue_path
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_suffix(path.suffix + ".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(queue, file, ensure_ascii=False, indent=2)

        temporary_path.replace(path)

    def existing_stable_ids(self) -> set[str]:
        queue = self.load_queue()
        return {str(record.get("stable_id")) for record in queue["approvals"]}

    def existing_reply_texts(self) -> set[str]:
        queue = self.load_queue()
        return {
            record["proposed_reply"]
            for record in queue["approvals"]
            if record.get("proposed_reply")
        }

    def add_records(self, records: list[DMApprovalRecord]) -> int:
        """
        Append records, skipping any whose stable_id already exists.

        Never creates duplicate approval records for the same
        stable_id (requirement 19). Returns the number actually added.
        """
        queue = self.load_queue()
        existing_ids = {
            str(record.get("stable_id")) for record in queue["approvals"]
        }

        added = 0

        for record in records:
            if record.stable_id in existing_ids:
                continue

            queue["approvals"].append(record.to_dict())
            existing_ids.add(record.stable_id)
            added += 1

        if added:
            queue["updated_at"] = self._now()
            self.save_queue(queue)

        return added

    def list_pending(self) -> list[dict[str, Any]]:
        queue = self.load_queue()
        return [
            record
            for record in queue["approvals"]
            if record.get("status") == "pending_approval"
        ]

    @staticmethod
    def _find(queue: dict[str, Any], approval_id: str) -> dict[str, Any] | None:
        for record in queue["approvals"]:
            if record.get("approval_id") == approval_id:
                return record

        return None

    def get(self, approval_id: str) -> dict[str, Any] | None:
        return self._find(self.load_queue(), approval_id)

    def approve(self, approval_id: str) -> dict[str, Any]:
        queue = self.load_queue()
        record = self._find(queue, approval_id)

        if record is None:
            raise DMApprovalNotFoundError(
                f"No DM approval record found for: {approval_id}"
            )

        record["status"] = "approved"
        record["approved_at"] = self._now()

        queue["updated_at"] = self._now()
        self.save_queue(queue)

        return record

    def reject(self, approval_id: str) -> dict[str, Any]:
        queue = self.load_queue()
        record = self._find(queue, approval_id)

        if record is None:
            raise DMApprovalNotFoundError(
                f"No DM approval record found for: {approval_id}"
            )

        record["status"] = "rejected"
        record["rejected_at"] = self._now()

        queue["updated_at"] = self._now()
        self.save_queue(queue)

        return record


def build_dm_approval_service(
    *,
    queue_dir: str | Path | None = None,
) -> DMApprovalService:
    resolved_queue_dir = (
        Path(queue_dir) if queue_dir is not None else _runtime_root() / "output" / "community" / "queues"
    )

    return DMApprovalService(queue_dir=resolved_queue_dir)


# ---------------------------------------------------------------------------
# dm_queue.json read/write (message records get classification /
# proposed_reply / safety_route / status filled in by prepare_dm_replies)
# ---------------------------------------------------------------------------


def _load_dm_queue(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": "1.0", "updated_at": None, "messages": []}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise DMReaderError(f"Invalid JSON in DM queue: {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise DMReaderError(f"DM queue must be a JSON object: {path}")

    data.setdefault("messages", [])
    return data


def _save_dm_queue(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def _build_dm_approval_id(stable_id: str) -> str:
    return "appr-dm-" + stable_id.replace(":", "-")


def prepare_dm_replies(
    *,
    dm_queue_path: Path,
    dm_approval_service: DMApprovalService,
    dm_reply_brain: DMReplyBrain,
) -> DMPrepareRepliesResult:
    """
    Read pending DMs, classify each one, route it through the DM
    safety gate, propose an Aiko-style reply where a reply is allowed
    (never for sexual/harassment/spam — requirement 18), and append
    new records to the DM approval queue. Also writes classification/
    proposed_reply/safety_route/status back onto the dm_queue.json
    message record itself (requirement 9).

    Never sends anything to Instagram.
    """
    dm_queue = _load_dm_queue(dm_queue_path)

    existing_ids = dm_approval_service.existing_stable_ids()
    used_reply_texts = dm_approval_service.existing_reply_texts()

    messages_processed = 0
    replies_proposed = 0
    auto_eligible_count = 0
    human_review_count = 0
    skipped_count = 0
    duplicates_skipped = 0

    new_records: list[DMApprovalRecord] = []

    for message in dm_queue.get("messages", []):
        if message.get("status") != "pending":
            continue

        messages_processed += 1
        stable_id = str(message.get("stable_id"))

        if stable_id in existing_ids:
            duplicates_skipped += 1
            continue

        classification_result = dm_reply_brain.classify(
            message.get("message_text"),
            message_type=message.get("message_type"),
        )
        classification = classification_result.classification
        safety_route = resolve_dm_safety_route(classification)
        created_at = _now()

        common_fields = dict(
            approval_id=_build_dm_approval_id(stable_id),
            stable_id=stable_id,
            conversation_id=message.get("conversation_id"),
            thread_url=message.get("thread_url"),
            username=message.get("username"),
            display_name=message.get("display_name"),
            message_text=message.get("message_text"),
            message_type=message.get("message_type"),
            classification=classification,
            safety_route=safety_route,
            created_at=created_at,
        )

        if safety_route == "never_reply":
            record = DMApprovalRecord(
                **common_fields,
                proposed_reply=None,
                status="skipped",
            )

            skipped_count += 1
            message["status"] = "skipped"
            proposed_reply = None

        else:
            proposed_reply = dm_reply_brain.generate_reply(
                classification,
                used_reply_texts=used_reply_texts,
            )

            if proposed_reply:
                used_reply_texts.add(proposed_reply)

            record = DMApprovalRecord(
                **common_fields,
                proposed_reply=proposed_reply,
                status="pending_approval",
            )

            replies_proposed += 1

            if safety_route == "auto_eligible":
                auto_eligible_count += 1
            else:
                human_review_count += 1

            message["status"] = "processed"

        message["classification"] = classification
        message["proposed_reply"] = proposed_reply
        message["safety_route"] = safety_route

        new_records.append(record)
        existing_ids.add(stable_id)

    dm_approval_service.add_records(new_records)
    _save_dm_queue(dm_queue_path, dm_queue)

    return DMPrepareRepliesResult(
        messages_processed=messages_processed,
        replies_proposed=replies_proposed,
        auto_eligible=auto_eligible_count,
        human_review_required=human_review_count,
        skipped=skipped_count,
        duplicates_skipped=duplicates_skipped,
        dm_approval_queue_path=str(dm_approval_service.queue_path),
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Instagram DM Approval Service (Phase 8F)"
    )

    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument(
        "--prepare-replies",
        action="store_true",
        help="Classify pending DMs and propose replies.",
    )

    action.add_argument(
        "--list-pending",
        action="store_true",
        help="List DM approval records awaiting a decision.",
    )

    action.add_argument(
        "--approve",
        metavar="APPROVAL_ID",
        default=None,
        help="Approve one proposed DM reply (updates local JSON only).",
    )

    action.add_argument(
        "--reject",
        metavar="APPROVAL_ID",
        default=None,
        help="Reject one proposed DM reply (updates local JSON only).",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate reply_rules.yaml config file.",
    )

    return parser.parse_args(argv)


def _print_prepare_result(result: DMPrepareRepliesResult) -> None:
    print()
    print("AIKO DM Approval Service — Prepare Replies")
    print("---------------------------------------------")
    print(f"messages processed:    {result.messages_processed}")
    print(f"replies proposed:      {result.replies_proposed}")
    print(f"auto eligible:         {result.auto_eligible}")
    print(f"human review required: {result.human_review_required}")
    print(f"skipped:               {result.skipped}")
    print(f"duplicates skipped:    {result.duplicates_skipped}")
    print(f"dm approval queue path:{result.dm_approval_queue_path}")
    print()


def _print_pending(records: list[dict[str, Any]]) -> None:
    print()
    print(f"AIKO DM Approval Service — Pending Approvals ({len(records)})")
    print("--------------------------------------------------------")

    for record in records:
        print(
            f"[{record.get('approval_id')}] "
            f"{record.get('classification')} / {record.get('safety_route')}"
        )
        print(f"  from:     @{record.get('username')}")
        print(f"  message:  {record.get('message_text')}")
        print(f"  proposed: {record.get('proposed_reply')}")
        print()

    print()


def main() -> None:
    arguments = parse_arguments()
    dm_approval_service = build_dm_approval_service()

    if arguments.prepare_replies:
        dm_reply_brain = build_dm_reply_brain(config_path=arguments.config)

        result = prepare_dm_replies(
            dm_queue_path=_default_dm_queue_path(),
            dm_approval_service=dm_approval_service,
            dm_reply_brain=dm_reply_brain,
        )

        _print_prepare_result(result)
        return

    if arguments.list_pending:
        _print_pending(dm_approval_service.list_pending())
        return

    if arguments.approve:
        try:
            record = dm_approval_service.approve(arguments.approve)
        except InstagramSessionError as exc:
            print(f"[DMApprovalService] approve failed: {exc}")
            raise SystemExit(1)

        print(f"[DMApprovalService] approved {record['approval_id']}")
        return

    if arguments.reject:
        try:
            record = dm_approval_service.reject(arguments.reject)
        except InstagramSessionError as exc:
            print(f"[DMApprovalService] reject failed: {exc}")
            raise SystemExit(1)

        print(f"[DMApprovalService] rejected {record['approval_id']}")
        return


if __name__ == "__main__":
    main()
