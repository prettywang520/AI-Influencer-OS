"""ResearchQueue -- file-backed job persistence, ordering, and
lifecycle operations (enqueue/dequeue/pause/resume/cancel/retry).
Atomic JSON writes (temp-file + Path.replace()), the same convention
used throughout this codebase.

Two lifecycle operations deliberately step outside state.py's normal
forward-only transition table, and are documented here rather than
hidden:
- pause()/resume() never change `status` at all (there is no PAUSED
  state in the fixed state machine) -- they toggle a
  `metadata["paused"]` flag instead, so a job's status always still
  reflects real collection progress. dequeue() skips paused jobs.
- retry() resets a FAILED job back to QUEUED directly (FAILED has no
  outgoing transitions in state.py, by design -- it's terminal for
  advance()), incrementing metadata["retry_count"] and raising
  RetryLimitExceededError once the configured max_attempts is used up.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from .exceptions import JobNotFoundError, QueueError, RetryLimitExceededError, ResearchConfigError
from .jobs import ResearchJob, _now_iso
from .state import JobStatus

DEFAULT_QUEUE_CONFIG_RELATIVE_PATH = Path("config") / "creator_research" / "queue.yaml"


def _runtime_root() -> Path:
    """
    queue.py location: 10_apps/claude_runtime/src/creator_research/queue.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_queue_config_path() -> Path:
    return _runtime_root() / DEFAULT_QUEUE_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class QueueConfig:
    schema_version: str
    default_queue_path: str
    max_attempts: int

    def resolved_default_queue_path(self) -> Path:
        return _runtime_root() / self.default_queue_path


def load_queue_config(config_path: str | Path | None = None) -> QueueConfig:
    path = Path(config_path) if config_path else default_queue_config_path()
    if not path.exists():
        raise ResearchConfigError(f"Queue config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ResearchConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ResearchConfigError(f"Queue config is empty or invalid: {path}")

    section = raw.get("queue") or {}
    return QueueConfig(
        schema_version=str(section.get("schema_version", "1.0")),
        default_queue_path=str(section.get("default_queue_path", "output/creator_research/queue.json")),
        max_attempts=int(section.get("max_attempts", 3)),
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


def _sort_key(job: ResearchJob) -> tuple:
    return (-job.priority, job.created_at, job.job_id)


class ResearchQueue:
    def __init__(self, queue_path: str | Path) -> None:
        self.path = Path(queue_path)
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[ResearchJob]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise QueueError(f"Invalid JSON in {self.path}: {exc}") from exc
        if not isinstance(raw, list):
            raise QueueError(f"Queue file must contain a JSON list: {self.path}")
        return [ResearchJob.from_dict(item) for item in raw]

    def _write(self, jobs: list[ResearchJob]) -> None:
        ordered = sorted(jobs, key=_sort_key)
        payload = [job.to_dict() for job in ordered]
        _atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True))

    def enqueue(self, job: ResearchJob) -> ResearchJob:
        """Idempotent: re-enqueuing a job whose deterministic job_id
        already exists returns the existing stored job unchanged
        rather than raising or duplicating it."""
        jobs = self._read()
        for existing in jobs:
            if existing.job_id == job.job_id:
                return existing
        jobs.append(job)
        self._write(jobs)
        return job

    def get(self, job_id: str) -> ResearchJob:
        for job in self._read():
            if job.job_id == job_id:
                return job
        raise JobNotFoundError(f"No job found with job_id={job_id!r}")

    def list_jobs(self, status: str | None = None) -> list[ResearchJob]:
        jobs = self._read()
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        return sorted(jobs, key=_sort_key)

    def dequeue(self) -> ResearchJob | None:
        """Returns the highest-priority, non-paused QUEUED job
        (ties broken by oldest created_at, then job_id) and advances
        it to STARTING. Returns None if nothing is eligible."""
        jobs = self._read()
        candidates = [
            job for job in jobs if job.status == JobStatus.QUEUED and not job.metadata.get("paused", False)
        ]
        if not candidates:
            return None
        candidates.sort(key=_sort_key)
        selected = candidates[0]
        selected.advance(JobStatus.STARTING)
        self._replace(jobs, selected)
        return selected

    def _replace(self, jobs: list[ResearchJob], updated: ResearchJob) -> None:
        new_jobs = [updated if job.job_id == updated.job_id else job for job in jobs]
        self._write(new_jobs)

    def pause(self, job_id: str) -> ResearchJob:
        job = self.get(job_id)
        job.metadata["paused"] = True
        self._replace(self._read(), job)
        return job

    def resume(self, job_id: str) -> ResearchJob:
        job = self.get(job_id)
        job.metadata["paused"] = False
        self._replace(self._read(), job)
        return job

    def cancel(self, job_id: str) -> ResearchJob:
        job = self.get(job_id)
        if job.status not in JobStatus.TERMINAL:
            job.advance(JobStatus.FAILED)
        job.metadata["cancelled"] = True
        self._replace(self._read(), job)
        return job

    def retry(self, job_id: str, *, max_attempts: int = 3) -> ResearchJob:
        job = self.get(job_id)
        if job.status != JobStatus.FAILED:
            raise QueueError(f"Only FAILED jobs can be retried (job_id={job_id!r} is {job.status!r})")
        retry_count = int(job.metadata.get("retry_count", 0))
        if retry_count >= max_attempts:
            raise RetryLimitExceededError(f"job_id={job_id!r} has exceeded max_attempts={max_attempts}")
        job.metadata["retry_count"] = retry_count + 1
        job.metadata["cancelled"] = False
        job.status = JobStatus.QUEUED
        job.updated_at = _now_iso()
        self._replace(self._read(), job)
        return job


def validate_queue(queue: "ResearchQueue") -> tuple[bool, list[str]]:
    """Structural check over a queue file: every job's status is a
    recognized JobStatus, and every *stored* job_id (read from the raw
    file, not re-derived by ResearchJob.from_dict()'s own
    auto-recompute) matches a fresh recompute from that job's own
    identity fields (tamper/corruption check, same pattern as
    creator_intelligence.serialization's dna_id check). Read-only --
    never writes."""
    errors: list[str] = []
    if not queue.path.exists():
        return True, []
    try:
        raw_entries = json.loads(queue.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"Invalid JSON in {queue.path}: {exc}"]
    if not isinstance(raw_entries, list):
        return False, [f"Queue file must contain a JSON list: {queue.path}"]

    seen_ids: set[str] = set()
    for entry in raw_entries:
        stored_job_id = entry.get("job_id", "")
        if entry.get("status") not in JobStatus.ALL:
            errors.append(f"job {stored_job_id!r} has unrecognized status {entry.get('status')!r}")
        recomputed = ResearchJob(
            creator_id=entry.get("creator_id", ""),
            platform=entry.get("platform", ""),
            username=entry.get("username", ""),
            profile_url=entry.get("profile_url", ""),
            connector_name=entry.get("connector_name", ""),
            requested_sections=tuple(entry.get("requested_sections", [])),
        )
        if recomputed.job_id != stored_job_id:
            errors.append(f"job_id mismatch: stored={stored_job_id!r} recomputed={recomputed.job_id!r}")
        if stored_job_id in seen_ids:
            errors.append(f"duplicate job_id in queue file: {stored_job_id!r}")
        seen_ids.add(stored_job_id)

    return not errors, errors
