from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .instagram_models import ApprovalNotFoundError, ApprovalRecord, ApprovalServiceError

QUEUE_FILENAME = "approval_queue.json"


class ApprovalService:
    """
    Local-only CRUD over output/community/queues/approval_queue.json.

    Approval and rejection only ever update this local JSON file.
    Nothing in this class sends anything to Instagram.
    """

    def __init__(self, *, queue_dir: str | Path) -> None:
        self.queue_dir = Path(queue_dir)

    @property
    def queue_path(self) -> Path:
        return self.queue_dir / QUEUE_FILENAME

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def load_queue(self) -> dict[str, Any]:
        path = self.queue_path

        if not path.exists():
            return {"version": "1.0", "updated_at": None, "approvals": []}

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ApprovalServiceError(
                f"Invalid JSON in approval queue: {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ApprovalServiceError(
                f"Approval queue must be a JSON object: {path}"
            )

        data.setdefault("version", "1.0")
        data.setdefault("approvals", [])

        if not isinstance(data["approvals"], list):
            raise ApprovalServiceError(
                f"Approval queue 'approvals' must be a list: {path}"
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

    def add_records(self, records: list[ApprovalRecord]) -> int:
        """
        Append records, skipping any whose stable_id already exists.

        Never creates duplicate approval records for the same
        stable_id. Returns the number of records actually added.
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
    def _find(
        queue: dict[str, Any],
        approval_id: str,
    ) -> dict[str, Any] | None:
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
            raise ApprovalNotFoundError(
                f"No approval record found for: {approval_id}"
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
            raise ApprovalNotFoundError(
                f"No approval record found for: {approval_id}"
            )

        record["status"] = "rejected"
        record["rejected_at"] = self._now()

        queue["updated_at"] = self._now()
        self.save_queue(queue)

        return record


def _runtime_root() -> Path:
    """
    approval_service.py location:

    10_apps/claude_runtime/src/social/approval_service.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def build_approval_service(
    *,
    queue_dir: str | Path | None = None,
) -> ApprovalService:
    resolved_queue_dir = (
        Path(queue_dir)
        if queue_dir is not None
        else _runtime_root() / "output" / "community" / "queues"
    )

    return ApprovalService(queue_dir=resolved_queue_dir)
