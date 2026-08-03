from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import PublisherConfig, load_publisher_config, now_iso

HISTORY_VERSION = "1.0"

EVENT_TYPES: frozenset[str] = frozenset(
    {
        "prepared",
        "validated",
        "approved",
        "rejected",
        "scheduled",
        "cancelled",
        "failed",
        "published",
    }
)


class HistoryServiceError(Exception):
    """Raised for malformed history data or an unknown event_type."""


class HistoryService:
    """
    Append-safe, one-file-per-production-date history log under
    output/publishing/history/history_<date>.json. Prior events are never
    rewritten or removed — record_event() only appends.
    """

    def __init__(self, history_dir: str | Path) -> None:
        self.history_dir = Path(history_dir)

    def _path(self, production_date: str) -> Path:
        return self.history_dir / f"history_{production_date}.json"

    def load(self, production_date: str) -> dict[str, Any]:
        path = self._path(production_date)

        if not path.exists():
            return {"version": HISTORY_VERSION, "production_date": production_date, "events": []}

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise HistoryServiceError(
                f"Invalid JSON in publish history: {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise HistoryServiceError(f"Publish history must be a JSON object: {path}")

        data.setdefault("version", HISTORY_VERSION)
        data.setdefault("production_date", production_date)
        events = data.get("events")

        if not isinstance(events, list):
            raise HistoryServiceError(f"Publish history 'events' must be a list: {path}")

        return data

    def _atomic_write(self, production_date: str, data: dict[str, Any]) -> None:
        path = self._path(production_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        temporary_path.replace(path)

    def record_event(
        self,
        *,
        production_date: str,
        job_id: str,
        event_type: str,
        previous_status: str | None,
        new_status: str | None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise HistoryServiceError(f"Unknown history event_type: {event_type}")

        data = self.load(production_date)
        event = {
            "event_id": f"evt-{len(data['events']) + 1}-{uuid4().hex[:8]}",
            "job_id": job_id,
            "event_type": event_type,
            "created_at": now_iso(),
            "previous_status": previous_status,
            "new_status": new_status,
            "details": details or {},
        }
        data["events"].append(event)
        self._atomic_write(production_date, data)
        return event

    def list_events(self, production_date: str) -> list[dict[str, Any]]:
        return self.load(production_date)["events"]

    # Convenience wrappers. Phase 10A only ever calls record_prepared,
    # record_validated, record_approved, record_rejected and
    # record_scheduled — the rest exist so a future browser-publishing
    # phase can record real outcomes without changing this service.
    def record_prepared(self, *, production_date: str, job_id: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.record_event(
            production_date=production_date,
            job_id=job_id,
            event_type="prepared",
            previous_status=None,
            new_status="draft",
            details=details,
        )

    def record_validated(
        self, *, production_date: str, job_id: str, previous_status: str, new_status: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.record_event(
            production_date=production_date,
            job_id=job_id,
            event_type="validated",
            previous_status=previous_status,
            new_status=new_status,
            details=details,
        )

    def record_approved(self, *, production_date: str, job_id: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.record_event(
            production_date=production_date,
            job_id=job_id,
            event_type="approved",
            previous_status="pending_approval",
            new_status="approved",
            details=details,
        )

    def record_rejected(self, *, production_date: str, job_id: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.record_event(
            production_date=production_date,
            job_id=job_id,
            event_type="rejected",
            previous_status="pending_approval",
            new_status="rejected",
            details=details,
        )

    def record_scheduled(
        self, *, production_date: str, job_id: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.record_event(
            production_date=production_date,
            job_id=job_id,
            event_type="scheduled",
            previous_status="approved",
            new_status="scheduled",
            details=details,
        )

    def record_cancelled(
        self, *, production_date: str, job_id: str, previous_status: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.record_event(
            production_date=production_date,
            job_id=job_id,
            event_type="cancelled",
            previous_status=previous_status,
            new_status="cancelled",
            details=details,
        )

    def record_failed(
        self, *, production_date: str, job_id: str, previous_status: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.record_event(
            production_date=production_date,
            job_id=job_id,
            event_type="failed",
            previous_status=previous_status,
            new_status="failed",
            details=details,
        )

    def record_published(
        self, *, production_date: str, job_id: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.record_event(
            production_date=production_date,
            job_id=job_id,
            event_type="published",
            previous_status="publishing",
            new_status="published",
            details=details,
        )


def build_history_service(config: PublisherConfig | None = None) -> HistoryService:
    config = config or load_publisher_config()
    return HistoryService(config.history_dir())
