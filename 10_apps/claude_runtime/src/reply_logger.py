from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class ReplyLogger:
    """
    Append-only JSON Lines logger for reply executions.

    Each line contains one complete reply event.
    """

    def __init__(self, log_path: str | Path) -> None:
        self.path = Path(log_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def write(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "timestamp": self._now(),
            "event_type": event_type,
            "payload": payload,
        }

        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def log_started(
        self,
        queue_id: str,
        request_id: str,
    ) -> None:
        self.write(
            "reply_processing_started",
            {
                "queue_id": queue_id,
                "request_id": request_id,
            },
        )

    def log_completed(
        self,
        queue_id: str,
        result: dict[str, Any],
    ) -> None:
        self.write(
            "reply_processing_completed",
            {
                "queue_id": queue_id,
                "result": result,
            },
        )

    def log_failed(
        self,
        queue_id: str,
        error: str,
        retry: bool,
    ) -> None:
        self.write(
            "reply_processing_failed",
            {
                "queue_id": queue_id,
                "error": error,
                "retry": retry,
            },
        )