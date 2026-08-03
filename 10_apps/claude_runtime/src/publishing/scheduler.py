from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from .history_service import HistoryService, build_history_service
from .models import TERMINAL_LOCKED_STATUSES, PublishJob, load_publisher_config, now_iso
from .queue_service import QueueService, build_queue_service


class SchedulerError(Exception):
    """Raised for an invalid schedule request (bad status, bad/past timestamp)."""


def _parse_at(at_value: str, timezone_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(at_value)
    except ValueError as exc:
        raise SchedulerError(f"Invalid ISO datetime: {at_value}") from exc

    if parsed.tzinfo is None:
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
        except Exception as exc:
            raise SchedulerError(f"Invalid configured timezone: {timezone_name}") from exc

    return parsed


class PublishScheduler:
    """
    Local-only scheduling of already-approved jobs. Setting scheduled_at
    only updates the queue JSON — there is no background process, no
    browser, and nothing here triggers an actual publish.
    """

    def __init__(
        self,
        queue_service: QueueService,
        history_service: HistoryService,
        timezone_name: str,
    ) -> None:
        self.queue_service = queue_service
        self.history_service = history_service
        self.timezone_name = timezone_name

    def schedule(self, job_id: str, at_value: str) -> PublishJob:
        jobs = self.queue_service.load_jobs()
        job = next((candidate for candidate in jobs if candidate.job_id == job_id), None)

        if job is None:
            raise SchedulerError(f"No publish job found for job_id: {job_id}")

        if job.status in TERMINAL_LOCKED_STATUSES:
            raise SchedulerError(f"Job {job_id} is {job.status} and can no longer be changed")

        if job.status != "approved":
            raise SchedulerError(
                f"Only approved jobs can be scheduled (job {job_id} is {job.status})"
            )

        at_datetime = _parse_at(at_value, self.timezone_name)
        now = datetime.now(at_datetime.tzinfo)

        if at_datetime <= now:
            raise SchedulerError(f"Scheduled time must be in the future: {at_value}")

        job.scheduled_at = at_datetime.isoformat()
        job.status = "scheduled"
        job.touch()
        self.queue_service.save_jobs(jobs)
        self.history_service.record_event(
            production_date=job.production_date,
            job_id=job.job_id,
            event_type="scheduled",
            previous_status="approved",
            new_status="scheduled",
        )
        return job

    def list_for_date(self, production_date: str) -> list[PublishJob]:
        return [
            job
            for job in self.queue_service.load_jobs()
            if job.production_date == production_date and job.status == "scheduled"
        ]


def build_scheduler() -> PublishScheduler:
    config = load_publisher_config()
    return PublishScheduler(
        queue_service=build_queue_service(config),
        history_service=build_history_service(config),
        timezone_name=config.timezone,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIKO OS Publish Scheduler")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--schedule", metavar="JOB_ID", help="Schedule an approved job")
    group.add_argument("--list", action="store_true", help="List scheduled jobs for a date")
    parser.add_argument("--at", help="ISO datetime to schedule at (required with --schedule)")
    parser.add_argument("--date", help="Production date YYYY-MM-DD (required with --list)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    scheduler = build_scheduler()

    if args.schedule:
        if not args.at:
            raise SystemExit("--at ISO_DATETIME is required with --schedule")

        job = scheduler.schedule(args.schedule, args.at)
        print(f"Scheduled {job.job_id} at {job.scheduled_at}")
        return

    if args.list:
        if not args.date:
            raise SystemExit("--date YYYY-MM-DD is required with --list")

        jobs = scheduler.list_for_date(args.date)
        print(f"Scheduled for {args.date}: {len(jobs)}")
        for job in jobs:
            print(f"  {job.job_id}  {job.content_type}  {job.scheduled_at}")
        return


if __name__ == "__main__":
    main()
