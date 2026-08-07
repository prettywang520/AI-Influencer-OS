"""ResearchJob -- the unit of work the queue/orchestrator operate on.
`job_id` is a deterministic content hash so re-creating a job for the
same creator+sections+connector always yields the same id (the same
"same logical request -> same id" convention used throughout this
codebase, e.g. end_to_end_render_pipeline.py's pipeline_id).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .interfaces import ConnectorSection
from .state import JobStatus, validate_transition


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(payload: dict, *, length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def compute_job_id(
    creator_id: str,
    platform: str,
    username: str,
    requested_sections: tuple[str, ...],
    connector_name: str,
) -> str:
    """Deterministic across re-creation: excludes status, timestamps,
    priority, and metadata -- only the logical identity of "research
    this creator, these sections, with this connector" affects it."""
    return _content_hash(
        {
            "creator_id": creator_id,
            "platform": platform,
            "username": username,
            "requested_sections": sorted(requested_sections),
            "connector_name": connector_name,
        }
    )


@dataclass(slots=True)
class ResearchJob:
    creator_id: str
    platform: str
    username: str
    profile_url: str
    connector_name: str
    status: str = JobStatus.QUEUED
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    priority: int = 0
    requested_sections: tuple[str, ...] = ConnectorSection.ALL
    metadata: dict = field(default_factory=dict)
    job_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.job_id = compute_job_id(
            self.creator_id, self.platform, self.username, self.requested_sections, self.connector_name
        )

    def advance(self, target_status: str) -> None:
        """Validates and applies a state transition, bumping
        updated_at. Raises InvalidTransitionError via state.py's
        validate_transition() on any disallowed jump."""
        validate_transition(self.status, target_status)
        self.status = target_status
        self.updated_at = _now_iso()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "creator_id": self.creator_id,
            "platform": self.platform,
            "username": self.username,
            "profile_url": self.profile_url,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "priority": self.priority,
            "requested_sections": list(self.requested_sections),
            "connector_name": self.connector_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ResearchJob":
        job = cls(
            creator_id=payload["creator_id"],
            platform=payload["platform"],
            username=payload["username"],
            profile_url=payload.get("profile_url", ""),
            connector_name=payload["connector_name"],
            created_at=payload.get("created_at", _now_iso()),
            updated_at=payload.get("updated_at", _now_iso()),
            priority=payload.get("priority", 0),
            requested_sections=tuple(payload.get("requested_sections", ConnectorSection.ALL)),
            metadata=dict(payload.get("metadata", {})),
        )
        stored_status = payload.get("status", JobStatus.QUEUED)
        # Restore status directly (not via advance()) -- this is a
        # reload of already-validated history, not a new transition.
        job.status = stored_status
        return job
