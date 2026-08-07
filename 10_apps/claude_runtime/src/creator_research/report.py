"""ResearchReport -- the human-readable outcome of one research job
run. build_report() is a pure function (no I/O): it only reads the
job/progress/evidence_queue objects it's given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .evidence_queue import EvidenceQueue
from .jobs import ResearchJob
from .progress import ResearchProgress


@dataclass(slots=True)
class ResearchReport:
    job_id: str
    creator_id: str
    coverage: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    sections_completed: list[str] = field(default_factory=list)
    sections_failed: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""


def _duration_seconds(created_at: str, updated_at: str) -> float:
    try:
        start = datetime.fromisoformat(created_at)
        end = datetime.fromisoformat(updated_at)
    except ValueError:
        return 0.0
    return max(0.0, (end - start).total_seconds())


def build_report(
    job: ResearchJob,
    progress: ResearchProgress,
    evidence_queue: EvidenceQueue,
    *,
    sections_failed: list[str] | None = None,
) -> ResearchReport:
    coverage = evidence_queue.counts_by_section()
    sections_failed = list(sections_failed or [])
    missing_sections = [
        section
        for section in job.requested_sections
        if coverage.get(section, 0) == 0 and section not in sections_failed
    ]
    warnings = list(progress.warnings) + evidence_queue.warnings()
    total_items = sum(coverage.values())
    summary = (
        f"{job.status}: {total_items} evidence item(s) across {len(coverage)} section(s); "
        f"{len(sections_failed)} failed, {len(missing_sections)} missing"
    )
    return ResearchReport(
        job_id=job.job_id,
        creator_id=job.creator_id,
        coverage=coverage,
        duration_seconds=_duration_seconds(job.created_at, job.updated_at),
        sections_completed=[s for s in job.requested_sections if coverage.get(s, 0) > 0],
        sections_failed=sections_failed,
        missing_sections=missing_sections,
        warnings=warnings,
        summary=summary,
    )
