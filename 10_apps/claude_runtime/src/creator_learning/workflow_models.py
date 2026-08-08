"""WorkflowCheckpoint -- the persisted record of one workflow run.
Pure data + to_dict()/from_dict(); persistence itself lives in
workflow_checkpoint.py.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .workflow_state import WorkflowState


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_workflow_id() -> str:
    """Fresh per run -- a workflow run is a repeatable, time-distinct
    event (re-learn the same creator next week), not a stable identity
    to dedupe against. Matches learning_session.py's own session_id
    convention, not jobs.py's deterministic-hash job_id convention."""
    return uuid.uuid4().hex[:16]


def new_resume_token() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(slots=True)
class WorkflowCheckpoint:
    workflow_id: str
    creator_id: str
    platform: str
    username: str
    profile_url: str
    connector_name: str
    requested_sections: tuple[str, ...] = ()
    research_job_id: str | None = None
    learning_session_id: str | None = None
    state: str = WorkflowState.CREATED
    last_completed_step: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    resume_token: str = field(default_factory=new_resume_token)
    resume_count: int = 0
    retry_count: int = 0
    restart_count: int = 0
    cancelled: bool = False
    step_history: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def record_transition(self, target_state: str) -> None:
        """Applies a validated state transition (validation happens
        in the caller via workflow_state.validate_transition() --
        this method only records the result), bumps updated_at,
        appends to step_history, mints a fresh resume_token, and -- if
        `target_state` is itself a forward-progression state (not
        FAILED/CANCELLED) -- records it as the new last_completed_step."""
        self.state = target_state
        self.updated_at = _now_iso()
        self.resume_token = new_resume_token()
        self.step_history.append({"state": target_state, "entered_at": self.updated_at})
        if target_state in WorkflowState.ORDER:
            self.last_completed_step = target_state

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "creator_id": self.creator_id,
            "platform": self.platform,
            "username": self.username,
            "profile_url": self.profile_url,
            "connector_name": self.connector_name,
            "requested_sections": list(self.requested_sections),
            "research_job_id": self.research_job_id,
            "learning_session_id": self.learning_session_id,
            "state": self.state,
            "last_completed_step": self.last_completed_step,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resume_token": self.resume_token,
            "resume_count": self.resume_count,
            "retry_count": self.retry_count,
            "restart_count": self.restart_count,
            "cancelled": self.cancelled,
            "step_history": list(self.step_history),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> WorkflowCheckpoint:
        return cls(
            workflow_id=payload["workflow_id"],
            creator_id=payload["creator_id"],
            platform=payload["platform"],
            username=payload["username"],
            profile_url=payload["profile_url"],
            connector_name=payload["connector_name"],
            requested_sections=tuple(payload.get("requested_sections", ())),
            research_job_id=payload.get("research_job_id"),
            learning_session_id=payload.get("learning_session_id"),
            state=payload.get("state", WorkflowState.CREATED),
            last_completed_step=payload.get("last_completed_step"),
            created_at=payload.get("created_at", _now_iso()),
            updated_at=payload.get("updated_at", _now_iso()),
            resume_token=payload.get("resume_token", new_resume_token()),
            resume_count=int(payload.get("resume_count", 0)),
            retry_count=int(payload.get("retry_count", 0)),
            restart_count=int(payload.get("restart_count", 0)),
            cancelled=bool(payload.get("cancelled", False)),
            step_history=list(payload.get("step_history", [])),
            warnings=list(payload.get("warnings", [])),
            errors=list(payload.get("errors", [])),
            metadata=dict(payload.get("metadata", {})),
        )
