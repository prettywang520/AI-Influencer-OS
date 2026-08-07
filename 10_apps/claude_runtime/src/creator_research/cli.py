"""CLI entry point for the Creator Research Agent. Every subcommand
operates on the local queue file only -- there is no flag anywhere on
this parser that accepts or constructs a Connector, so this CLI has no
way to actually run collection (see docs/creator_research/architecture.md).
No network access anywhere in this module.

Flags:
    --create-job --creator-id ID --platform P --username U --profile-url URL
                 [--priority N] [--sections a,b,c] [--connector-name NAME]
    --resume JOB_ID
    --cancel JOB_ID
    --status [JOB_ID]
    --validate
    [--queue PATH] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .exceptions import CliError, JobNotFoundError, QueueError, RetryLimitExceededError, ResearchError
from .interfaces import ConnectorSection
from .jobs import ResearchJob
from .queue import QueueConfig, ResearchQueue, load_queue_config, validate_queue
from .state import JobStatus

PREFIX = "[CreatorResearch]"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="creator_research.cli", description="Creator Research Agent job queue CLI")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create-job", action="store_true")
    action.add_argument("--resume", metavar="JOB_ID")
    action.add_argument("--cancel", metavar="JOB_ID")
    action.add_argument("--status", nargs="?", const="", metavar="JOB_ID")
    action.add_argument("--validate", action="store_true")

    parser.add_argument("--creator-id")
    parser.add_argument("--platform")
    parser.add_argument("--username")
    parser.add_argument("--profile-url")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--sections", help="comma-separated section names, default: all")
    parser.add_argument("--connector-name", default="unassigned")
    parser.add_argument("--queue", type=Path, default=None)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.create_job and (not args.creator_id or not args.platform or not args.username or not args.profile_url):
        parser.error("--create-job requires --creator-id, --platform, --username, and --profile-url")
    return args


def _resolve_queue(args: argparse.Namespace) -> tuple[ResearchQueue, QueueConfig]:
    queue_config = load_queue_config()
    queue_path = args.queue or queue_config.resolved_default_queue_path()
    return ResearchQueue(queue_path), queue_config


def _parse_sections(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ConnectorSection.ALL
    sections = tuple(s.strip() for s in raw.split(",") if s.strip())
    for section in sections:
        if section not in ConnectorSection.ALL:
            raise CliError(f"Unrecognized section: {section!r}")
    return sections


def _job_to_summary(job: ResearchJob) -> dict:
    return {
        "job_id": job.job_id,
        "creator_id": job.creator_id,
        "username": job.username,
        "status": job.status,
        "priority": job.priority,
        "requested_sections": list(job.requested_sections),
        "metadata": job.metadata,
    }


def _run_create_job(args: argparse.Namespace, queue: ResearchQueue) -> dict:
    sections = _parse_sections(args.sections)
    job = ResearchJob(
        creator_id=args.creator_id,
        platform=args.platform,
        username=args.username,
        profile_url=args.profile_url,
        connector_name=args.connector_name,
        priority=args.priority,
        requested_sections=sections,
    )
    stored = queue.enqueue(job)
    return _job_to_summary(stored)


def _run_resume(args: argparse.Namespace, queue: ResearchQueue, queue_config: QueueConfig) -> dict:
    job = queue.get(args.resume)
    if job.metadata.get("paused", False):
        job = queue.resume(args.resume)
    elif job.status == JobStatus.FAILED:
        job = queue.retry(args.resume, max_attempts=queue_config.max_attempts)
    else:
        raise CliError(f"job {args.resume!r} is not paused or failed (status={job.status!r}); nothing to resume")
    return _job_to_summary(job)


def _run_cancel(args: argparse.Namespace, queue: ResearchQueue) -> dict:
    job = queue.cancel(args.cancel)
    return _job_to_summary(job)


def _run_status(args: argparse.Namespace, queue: ResearchQueue) -> dict:
    if args.status:
        return _job_to_summary(queue.get(args.status))
    return {"jobs": [_job_to_summary(job) for job in queue.list_jobs()]}


def _run_validate(queue: ResearchQueue) -> dict:
    passed, errors = validate_queue(queue)
    return {"passed": passed, "errors": errors}


def _print_result(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"{PREFIX} result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        queue, queue_config = _resolve_queue(args)
        if args.create_job:
            result = _run_create_job(args, queue)
        elif args.resume:
            result = _run_resume(args, queue, queue_config)
        elif args.cancel:
            result = _run_cancel(args, queue)
        elif args.status is not None:
            result = _run_status(args, queue)
        elif args.validate:
            result = _run_validate(queue)
        else:  # pragma: no cover - argparse enforces one action
            raise CliError("no action given")
    except (CliError, JobNotFoundError, QueueError, RetryLimitExceededError, ResearchError) as exc:
        print(f"{PREFIX} Error: {exc}", file=sys.stderr)
        return 1

    _print_result(result, args.json)
    if isinstance(result, dict) and result.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
