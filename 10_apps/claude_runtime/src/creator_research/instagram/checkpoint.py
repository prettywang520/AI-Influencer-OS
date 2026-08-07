"""Per-(job, section) checkpointed research state -- lets a section be
resumed without re-collecting or duplicating evidence. Atomic JSON
(temp-file + Path.replace()), the same convention used throughout
this codebase. Loading always revalidates identity before reuse: a
job_id/creator_id mismatch is a hard error, never silently adopted.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .exceptions import CheckpointError, CheckpointMismatchError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class InstagramCheckpoint:
    job_id: str
    creator_id: str
    connector_version: str
    section: str
    last_processed_source: str | None = None
    processed_source_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    scroll_round: int = 0
    updated_at: str = field(default_factory=_now_iso)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "creator_id": self.creator_id,
            "connector_version": self.connector_version,
            "section": self.section,
            "last_processed_source": self.last_processed_source,
            "processed_source_ids": list(self.processed_source_ids),
            "evidence_ids": list(self.evidence_ids),
            "scroll_round": self.scroll_round,
            "updated_at": self.updated_at,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InstagramCheckpoint":
        return cls(
            job_id=payload["job_id"],
            creator_id=payload["creator_id"],
            connector_version=payload.get("connector_version", ""),
            section=payload["section"],
            last_processed_source=payload.get("last_processed_source"),
            processed_source_ids=list(payload.get("processed_source_ids", [])),
            evidence_ids=list(payload.get("evidence_ids", [])),
            scroll_round=int(payload.get("scroll_round", 0)),
            updated_at=payload.get("updated_at", _now_iso()),
            warnings=list(payload.get("warnings", [])),
        )


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


def _checkpoint_path(checkpoint_dir: str | Path, job_id: str, section: str) -> Path:
    return Path(checkpoint_dir) / f"{job_id}_{section}.json"


def save_checkpoint(checkpoint_dir: str | Path, checkpoint: InstagramCheckpoint) -> Path:
    path = _checkpoint_path(checkpoint_dir, checkpoint.job_id, checkpoint.section)
    _atomic_write_text(path, json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True))
    return path


def load_checkpoint(checkpoint_dir: str | Path, job_id: str, section: str) -> InstagramCheckpoint | None:
    path = _checkpoint_path(checkpoint_dir, job_id, section)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Invalid JSON in {path}: {exc}") from exc
    return InstagramCheckpoint.from_dict(payload)


def validate_checkpoint(
    checkpoint: InstagramCheckpoint,
    *,
    job_id: str,
    creator_id: str,
    connector_version: str,
) -> list[str]:
    """Raises CheckpointMismatchError if job_id/creator_id don't
    match. Returns a list of non-fatal warnings (e.g. a
    connector_version drift -- selectors/parsing may have changed, but
    the creator/job identity is still valid, so resuming is still
    safe)."""
    if checkpoint.job_id != job_id or checkpoint.creator_id != creator_id:
        raise CheckpointMismatchError(
            f"checkpoint identity mismatch: stored job_id={checkpoint.job_id!r}/"
            f"creator_id={checkpoint.creator_id!r} vs requested job_id={job_id!r}/creator_id={creator_id!r}"
        )
    warnings: list[str] = []
    if checkpoint.connector_version != connector_version:
        warnings.append(
            f"checkpoint was written by connector_version={checkpoint.connector_version!r}, "
            f"resuming with connector_version={connector_version!r}"
        )
    return warnings
