from __future__ import annotations

import argparse

from .history_service import HistoryService, build_history_service
from .models import TERMINAL_LOCKED_STATUSES, PublishJob, now_iso
from .queue_service import QueueService, build_queue_service


class ApprovalError(Exception):
    """Raised when an approval/rejection is attempted on an ineligible job."""


class PublishApprovalService:
    """
    Local-only approval workflow over the publish queue. Approval and
    rejection only ever modify output/publishing/queues/publish_queue.json
    and append an event to the per-date history file — nothing here
    publishes or contacts any platform.
    """

    def __init__(self, queue_service: QueueService, history_service: HistoryService) -> None:
        self.queue_service = queue_service
        self.history_service = history_service

    def list_pending(self) -> list[PublishJob]:
        return [job for job in self.queue_service.load_jobs() if job.status == "pending_approval"]

    def approve(self, job_id: str) -> PublishJob:
        return self._transition(job_id, new_status="approved", event_type="approved")

    def reject(self, job_id: str) -> PublishJob:
        return self._transition(job_id, new_status="rejected", event_type="rejected")

    def _transition(self, job_id: str, *, new_status: str, event_type: str) -> PublishJob:
        jobs = self.queue_service.load_jobs()
        job = next((candidate for candidate in jobs if candidate.job_id == job_id), None)

        if job is None:
            raise ApprovalError(f"No publish job found for job_id: {job_id}")

        if job.status in TERMINAL_LOCKED_STATUSES:
            raise ApprovalError(
                f"Job {job_id} is {job.status} and can no longer be changed"
            )

        if job.status != "pending_approval":
            raise ApprovalError(
                f"Only pending_approval jobs can be approved or rejected "
                f"(job {job_id} is {job.status})"
            )

        previous_status = job.status
        job.status = new_status

        if new_status == "approved":
            job.approved_at = now_iso()

        job.touch()
        self.queue_service.save_jobs(jobs)
        self.history_service.record_event(
            production_date=job.production_date,
            job_id=job.job_id,
            event_type=event_type,
            previous_status=previous_status,
            new_status=new_status,
        )
        return job


def build_approval_service() -> PublishApprovalService:
    return PublishApprovalService(
        queue_service=build_queue_service(),
        history_service=build_history_service(),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIKO OS Publish Approval Service")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-pending", action="store_true", help="List jobs awaiting approval")
    group.add_argument("--approve", metavar="JOB_ID", help="Approve a pending_approval job")
    group.add_argument("--reject", metavar="JOB_ID", help="Reject a pending_approval job")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    service = build_approval_service()

    if args.list_pending:
        pending = service.list_pending()
        print(f"Pending approval: {len(pending)}")
        for job in pending:
            print(f"  {job.job_id}  {job.content_type}  {job.production_date}")
        return

    if args.approve:
        job = service.approve(args.approve)
        print(f"Approved {job.job_id} (approved_at={job.approved_at})")
        return

    if args.reject:
        job = service.reject(args.reject)
        print(f"Rejected {job.job_id}")
        return


if __name__ == "__main__":
    main()
