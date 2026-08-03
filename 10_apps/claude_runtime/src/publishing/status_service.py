from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from .models import PublishJob, load_publisher_config
from .queue_service import build_queue_service
from .validator import validate_job


@dataclass(slots=True)
class StatusReport:
    production_date: str
    total_jobs: int
    validation_failed: int
    pending_approval: int
    approved: int
    scheduled: int
    published: int
    failed: int
    missing_source_files: list[str] = field(default_factory=list)
    next_recommended_command: str = ""


def _missing_source_files(jobs: list[PublishJob], config) -> list[str]:
    missing: list[str] = []

    for job in jobs:
        result = validate_job(job, config)
        for error in result.errors:
            missing.append(f"{job.job_id}: {error}")

    return missing


def _next_recommended_command(date: str, counts: dict[str, int], total: int) -> str:
    if total == 0:
        return f"python3 -u -m src.publishing.publisher_service --prepare --date {date}"

    if counts["validation_failed"]:
        return (
            f"Fix missing/invalid source files, then: "
            f"python3 -u -m src.publishing.publisher_service --prepare --date {date} --force"
        )

    if counts["pending_approval"]:
        return "python3 -u -m src.publishing.approval_service --list-pending"

    if counts["approved"]:
        return "python3 -u -m src.publishing.scheduler --schedule JOB_ID --at ISO_DATETIME"

    if counts["scheduled"]:
        return f"python3 -u -m src.publishing.scheduler --list --date {date}"

    return "No action needed."


def build_status_report(date: str) -> StatusReport:
    config = load_publisher_config()
    queue_service = build_queue_service(config)
    jobs = [job for job in queue_service.load_jobs() if job.production_date == date]

    counts = {
        "validation_failed": sum(1 for job in jobs if job.status == "validation_failed"),
        "pending_approval": sum(1 for job in jobs if job.status == "pending_approval"),
        "approved": sum(1 for job in jobs if job.status == "approved"),
        "scheduled": sum(1 for job in jobs if job.status == "scheduled"),
        "published": sum(1 for job in jobs if job.status == "published"),
        "failed": sum(1 for job in jobs if job.status == "failed"),
    }

    return StatusReport(
        production_date=date,
        total_jobs=len(jobs),
        validation_failed=counts["validation_failed"],
        pending_approval=counts["pending_approval"],
        approved=counts["approved"],
        scheduled=counts["scheduled"],
        published=counts["published"],
        failed=counts["failed"],
        missing_source_files=_missing_source_files(jobs, config),
        next_recommended_command=_next_recommended_command(date, counts, len(jobs)),
    )


def _print_report(report: StatusReport) -> None:
    print(f"Production date:     {report.production_date}")
    print(f"Total jobs:          {report.total_jobs}")
    print(f"Validation failed:   {report.validation_failed}")
    print(f"Pending approval:    {report.pending_approval}")
    print(f"Approved:            {report.approved}")
    print(f"Scheduled:           {report.scheduled}")
    print(f"Published:           {report.published}")
    print(f"Failed:              {report.failed}")

    if report.missing_source_files:
        print("Missing source files:")
        for entry in report.missing_source_files:
            print(f"  - {entry}")
    else:
        print("Missing source files: none")

    print(f"Next recommended command:\n  {report.next_recommended_command}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIKO OS Publish Status Service")
    parser.add_argument("--date", required=True, help="Production date YYYY-MM-DD")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = build_status_report(args.date)
    _print_report(report)


if __name__ == "__main__":
    main()
