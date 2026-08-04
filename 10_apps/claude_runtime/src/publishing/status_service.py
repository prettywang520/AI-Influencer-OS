from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from .models import PublishJob, load_publisher_config
from .publish_recovery import _lock_is_clearable, list_stuck_jobs
from .queue_service import build_queue_service
from .validator import validate_job

MANUAL_CHECK_REQUIRED_ERROR = "publish_verification_inconclusive_manual_check_required"


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
    publishing: int
    failed_manual_check_required: int
    stale_locks: int
    inconsistent_published_records: int
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


def _count_stale_locks(config, job_ids: set[str]) -> int:
    locks_dir = config.locks_dir()

    if not locks_dir.is_dir():
        return 0

    count = 0
    for lock_path in locks_dir.glob("*.lock.json"):
        job_id = lock_path.name.removesuffix(".lock.json")
        if job_id not in job_ids:
            continue
        if _lock_is_clearable(lock_path, config):
            count += 1

    return count


def build_status_report(date: str) -> StatusReport:
    config = load_publisher_config()
    queue_service = build_queue_service(config)
    jobs = [job for job in queue_service.load_jobs() if job.production_date == date]
    job_ids = {job.job_id for job in jobs}

    counts = {
        "validation_failed": sum(1 for job in jobs if job.status == "validation_failed"),
        "pending_approval": sum(1 for job in jobs if job.status == "pending_approval"),
        "approved": sum(1 for job in jobs if job.status == "approved"),
        "scheduled": sum(1 for job in jobs if job.status == "scheduled"),
        "published": sum(1 for job in jobs if job.status == "published"),
        "failed": sum(1 for job in jobs if job.status == "failed"),
        "publishing": sum(1 for job in jobs if job.status == "publishing"),
    }

    failed_manual_check_required = sum(
        1 for job in jobs if job.status == "failed" and job.error == MANUAL_CHECK_REQUIRED_ERROR
    )
    inconsistent_published_records = sum(
        1 for job in jobs if job.status == "published" and (not job.published_at or not job.platform_url)
    )
    stale_locks = _count_stale_locks(config, job_ids)

    stuck_reports = [report for report in list_stuck_jobs(config) if report.production_date == date]

    next_recommended_command = _next_recommended_command(date, counts, len(jobs))
    if stuck_reports:
        next_recommended_command = (
            f"python3 -u -m src.publishing.publish_recovery --inspect {stuck_reports[0].job_id}"
        )

    return StatusReport(
        production_date=date,
        total_jobs=len(jobs),
        validation_failed=counts["validation_failed"],
        pending_approval=counts["pending_approval"],
        approved=counts["approved"],
        scheduled=counts["scheduled"],
        published=counts["published"],
        failed=counts["failed"],
        publishing=counts["publishing"],
        failed_manual_check_required=failed_manual_check_required,
        stale_locks=stale_locks,
        inconsistent_published_records=inconsistent_published_records,
        missing_source_files=_missing_source_files(jobs, config),
        next_recommended_command=next_recommended_command,
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
    print(f"Publishing:          {report.publishing}")
    print(f"Failed (manual check required): {report.failed_manual_check_required}")
    print(f"Stale locks:         {report.stale_locks}")
    print(f"Inconsistent published records: {report.inconsistent_published_records}")

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
