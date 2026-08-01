from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class ReplyQueue:
    def __init__(self, queue_path: str | Path) -> None:
        self.path = Path(queue_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

        if not self.path.exists():
            self._write([])

    def _read(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

        return data if isinstance(data, list) else []

    def _write(self, items: list[dict[str, Any]]) -> None:
        temporary_path = self.path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                items,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(self.path)

    def enqueue(self, payload: dict[str, Any]) -> str:
        with self._lock:
            items = self._read()

            queue_id = (
                payload.get("request_id")
                or f"reply-{len(items) + 1}"
            )

            items.append(
                {
                    "queue_id": queue_id,
                    "status": "pending",
                    "attempts": 0,
                    "created_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "updated_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "payload": payload,
                    "result": None,
                    "error": None,
                }
            )

            self._write(items)
            return str(queue_id)

    def next_pending(self) -> dict[str, Any] | None:
        with self._lock:
            items = self._read()

            for item in items:
                if item.get("status") == "pending":
                    item["status"] = "processing"
                    item["attempts"] = int(
                        item.get("attempts", 0)
                    ) + 1
                    item["updated_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()

                    self._write(items)
                    return item

        return None

    def complete(
        self,
        queue_id: str,
        result: dict[str, Any],
    ) -> None:
        self._update(
            queue_id=queue_id,
            status="completed",
            result=result,
            error=None,
        )

    def fail(
        self,
        queue_id: str,
        error: str,
        retry: bool = False,
    ) -> None:
        self._update(
            queue_id=queue_id,
            status="pending" if retry else "failed",
            result=None,
            error=error,
        )

    def _update(
        self,
        queue_id: str,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        with self._lock:
            items = self._read()

            for item in items:
                if item.get("queue_id") != queue_id:
                    continue

                item["status"] = status
                item["result"] = result
                item["error"] = error
                item["updated_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                break

            self._write(items)

    def list_items(
        self,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = self._read()

        if status is None:
            return items

        return [
            item
            for item in items
            if item.get("status") == status
        ]