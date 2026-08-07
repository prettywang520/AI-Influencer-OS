"""Structured, per-run diagnostics. No field for a password, cookie,
auth header, session token, or private message exists on this
dataclass -- structurally impossible to log one, not merely a
convention (test_structural_safety.py greps the source for those
literal words as a second layer of defense). Screenshots-on-error are
saved as file-path *references* only (via navigator.py's
get_screenshot_reference()) -- never embedded secrets.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_connector_run_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(slots=True)
class DiagnosticsRecorder:
    connector_run_id: str
    job_id: str
    username: str
    sections_attempted: list[str] = field(default_factory=list)
    sections_completed: list[str] = field(default_factory=list)
    items_collected: dict[str, int] = field(default_factory=dict)
    duplicates_skipped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    access_limit_events: list[dict] = field(default_factory=list)
    rate_limit_events: list[dict] = field(default_factory=list)
    selector_failures: list[dict] = field(default_factory=list)
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None

    def record_section_attempt(self, section: str) -> None:
        if section not in self.sections_attempted:
            self.sections_attempted.append(section)

    def record_section_completed(self, section: str, item_count: int) -> None:
        if section not in self.sections_completed:
            self.sections_completed.append(section)
        self.items_collected[section] = self.items_collected.get(section, 0) + item_count

    def record_duplicates_skipped(self, section: str, count: int) -> None:
        if count:
            self.duplicates_skipped[section] = self.duplicates_skipped.get(section, 0) + count

    def record_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    def record_access_event(self, *, kind: str, section: str, detail: str = "") -> None:
        event = {"kind": kind, "section": section, "detail": detail, "occurred_at": _now_iso()}
        if kind == "rate_limited":
            self.rate_limit_events.append(event)
        else:
            self.access_limit_events.append(event)

    def record_selector_failure(self, *, section: str, element: str) -> None:
        self.selector_failures.append({"section": section, "element": element, "occurred_at": _now_iso()})

    def finish(self) -> None:
        self.finished_at = _now_iso()

    def to_dict(self) -> dict:
        return {
            "connector_run_id": self.connector_run_id,
            "job_id": self.job_id,
            "username": self.username,
            "sections_attempted": list(self.sections_attempted),
            "sections_completed": list(self.sections_completed),
            "items_collected": dict(self.items_collected),
            "duplicates_skipped": dict(self.duplicates_skipped),
            "warnings": list(self.warnings),
            "access_limit_events": list(self.access_limit_events),
            "rate_limit_events": list(self.rate_limit_events),
            "selector_failures": list(self.selector_failures),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        Path(tmp_name).replace(path)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


def save_diagnostics(diagnostics_dir: str | Path, recorder: DiagnosticsRecorder) -> Path:
    path = Path(diagnostics_dir) / f"{recorder.connector_run_id}.json"
    _atomic_write_text(path, json.dumps(recorder.to_dict(), indent=2, sort_keys=True))
    return path
