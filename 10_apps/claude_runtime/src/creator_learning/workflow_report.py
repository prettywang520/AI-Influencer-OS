"""WorkflowReport -- the human-readable outcome of one workflow run,
persisted as workflow_report.json. build_workflow_report() is a pure
function (no I/O): it only reads the checkpoint/evidence-count/DNA
values it's given (mirrors creator_research/report.py's
build_report()).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.creator_intelligence.models import CreatorDNA

from .style_evolution import TRAIT_FIELDS
from .workflow_models import WorkflowCheckpoint
from .workflow_state import WorkflowState


@dataclass(slots=True)
class WorkflowReport:
    workflow_id: str
    creator_id: str
    state: str
    started_at: str
    completed_at: str | None
    duration_seconds: float
    steps: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    resume_count: int = 0
    retry_count: int = 0
    restart_count: int = 0
    evidence_statistics: dict = field(default_factory=dict)
    dna_statistics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "creator_id": self.creator_id,
            "state": self.state,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "steps": list(self.steps),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "resume_count": self.resume_count,
            "retry_count": self.retry_count,
            "restart_count": self.restart_count,
            "evidence_statistics": dict(self.evidence_statistics),
            "dna_statistics": dict(self.dna_statistics),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> WorkflowReport:
        return cls(
            workflow_id=payload["workflow_id"],
            creator_id=payload["creator_id"],
            state=payload["state"],
            started_at=payload["started_at"],
            completed_at=payload.get("completed_at"),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            steps=list(payload.get("steps", [])),
            warnings=list(payload.get("warnings", [])),
            errors=list(payload.get("errors", [])),
            resume_count=int(payload.get("resume_count", 0)),
            retry_count=int(payload.get("retry_count", 0)),
            restart_count=int(payload.get("restart_count", 0)),
            evidence_statistics=dict(payload.get("evidence_statistics", {})),
            dna_statistics=dict(payload.get("dna_statistics", {})),
        )


def _duration_seconds(created_at: str, updated_at: str) -> float:
    try:
        start = datetime.fromisoformat(created_at)
        end = datetime.fromisoformat(updated_at)
    except ValueError:
        return 0.0
    return max(0.0, (end - start).total_seconds())


def _dna_statistics(dna: CreatorDNA | None) -> dict:
    if dna is None:
        return {}
    populated_traits = [name for name in TRAIT_FIELDS if getattr(dna, name) is not None]
    return {
        "dna_id": dna.dna_id,
        "overall_confidence": dna.overall_confidence,
        "populated_traits": populated_traits,
        "relationship_claim_count": len(dna.relationship_claims),
    }


def build_workflow_report(
    checkpoint: WorkflowCheckpoint,
    *,
    evidence_count_before: int | None = None,
    evidence_count_after: int | None = None,
    new_evidence_count: int | None = None,
    dna: CreatorDNA | None = None,
) -> WorkflowReport:
    completed_at = checkpoint.updated_at if checkpoint.state in WorkflowState.TERMINAL else None
    evidence_statistics = {}
    if evidence_count_before is not None or evidence_count_after is not None:
        evidence_statistics = {
            "count_before": evidence_count_before,
            "count_after": evidence_count_after,
            "new_evidence_count": new_evidence_count,
        }
    return WorkflowReport(
        workflow_id=checkpoint.workflow_id,
        creator_id=checkpoint.creator_id,
        state=checkpoint.state,
        started_at=checkpoint.created_at,
        completed_at=completed_at,
        duration_seconds=_duration_seconds(checkpoint.created_at, checkpoint.updated_at),
        steps=list(checkpoint.step_history),
        warnings=list(checkpoint.warnings),
        errors=list(checkpoint.errors),
        resume_count=checkpoint.resume_count,
        retry_count=checkpoint.retry_count,
        restart_count=checkpoint.restart_count,
        evidence_statistics=evidence_statistics,
        dna_statistics=_dna_statistics(dna),
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


def save_workflow_report(path: str | Path, report: WorkflowReport) -> Path:
    path = Path(path)
    _atomic_write_text(path, json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return path


def load_workflow_report(path: str | Path) -> WorkflowReport | None:
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return WorkflowReport.from_dict(payload)
