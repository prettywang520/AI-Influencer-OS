from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .models import PublishJob, PublisherConfig, load_publisher_config, now_iso

QUEUE_VERSION = "1.0"


class PublishQueueError(Exception):
    """Raised for malformed queue data or invariant violations (dup job_id, missing job)."""


class QueueService:
    """
    Atomic JSON persistence for output/publishing/queues/publish_queue.json.

    Never writes anything on construction or on load() — a missing file is
    treated as an empty queue in memory only. Writes only happen when
    save_jobs()/add_job()/update_job() are explicitly called.
    """

    def __init__(self, queue_path: str | Path) -> None:
        self.queue_path = Path(queue_path)

    def load(self) -> dict[str, Any]:
        if not self.queue_path.exists():
            return {"version": QUEUE_VERSION, "updated_at": None, "jobs": []}

        try:
            with self.queue_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise PublishQueueError(
                f"Invalid JSON in publish queue: {self.queue_path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise PublishQueueError(
                f"Publish queue must be a JSON object: {self.queue_path}"
            )

        data.setdefault("version", QUEUE_VERSION)
        data.setdefault("updated_at", None)
        jobs = data.get("jobs")

        if not isinstance(jobs, list):
            raise PublishQueueError(
                f"Publish queue 'jobs' must be a list: {self.queue_path}"
            )

        return data

    def load_jobs(self) -> list[PublishJob]:
        return [PublishJob.from_dict(item) for item in self.load()["jobs"]]

    @staticmethod
    def _sort_key(job: PublishJob) -> tuple[str, str, str, str]:
        return (job.production_date, job.platform, job.content_type, job.job_id)

    def save_jobs(self, jobs: list[PublishJob]) -> None:
        seen: set[str] = set()

        for job in jobs:
            if job.job_id in seen:
                raise PublishQueueError(f"Duplicate job_id in queue: {job.job_id}")
            seen.add(job.job_id)

        ordered = sorted(jobs, key=self._sort_key)
        data = {
            "version": QUEUE_VERSION,
            "updated_at": now_iso(),
            "jobs": [job.to_dict() for job in ordered],
        }
        self._atomic_write(data)

    def _atomic_write(self, data: dict[str, Any]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.queue_path.with_suffix(self.queue_path.suffix + ".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        temporary_path.replace(self.queue_path)

    def get(self, job_id: str) -> PublishJob | None:
        for job in self.load_jobs():
            if job.job_id == job_id:
                return job
        return None

    def add_job(self, job: PublishJob) -> None:
        jobs = self.load_jobs()

        if any(existing.job_id == job.job_id for existing in jobs):
            raise PublishQueueError(f"job_id already exists in queue: {job.job_id}")

        jobs.append(job)
        self.save_jobs(jobs)

    def upsert_jobs(self, jobs_to_upsert: list[PublishJob]) -> None:
        """Insert new jobs or replace existing ones with the same job_id."""
        jobs_by_id = {job.job_id: job for job in self.load_jobs()}

        for job in jobs_to_upsert:
            jobs_by_id[job.job_id] = job

        self.save_jobs(list(jobs_by_id.values()))

    def update_job(self, job_id: str, mutator: Callable[[PublishJob], None]) -> PublishJob:
        jobs = self.load_jobs()

        for job in jobs:
            if job.job_id == job_id:
                mutator(job)
                job.touch()
                self.save_jobs(jobs)
                return job

        raise PublishQueueError(f"job_id not found in queue: {job_id}")


def build_queue_service(config: PublisherConfig | None = None) -> QueueService:
    config = config or load_publisher_config()
    return QueueService(config.queue_path())


def load_job_by_id(queue_path: str | Path, job_id: str) -> PublishJob | None:
    """
    Pure read-only lookup: loads the queue file and returns one job (or
    None), without ever exposing a writable QueueService to the caller.

    Intended for callers — such as the Phase 10B preview module — that
    must never be able to write to publish_queue.json. Those callers
    should call this function directly rather than constructing their
    own QueueService instance.
    """
    return QueueService(queue_path).get(job_id)
