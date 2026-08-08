"""CLI entry point for the Creator Learning Workflow. Mirrors
creator_research/cli.py's structure and, critically, its restraint:
there is no flag anywhere on this parser that can construct or
reference a live Connector, so `--start` always runs in zero-network
mode (optionally ingesting an existing intake evidence_bundle.json).
A real, network-capable Connector can only ever be supplied by a
Python caller of WorkflowRunner.start()/resume()/retry()/restart()
directly -- never from this CLI. No network access anywhere in this
module.

Flags:
    --start --creator-url URL [--intake-evidence-bundle PATH] [--json]
    --resume WORKFLOW_ID [--creator-id ID] [--json]
    --cancel WORKFLOW_ID [--creator-id ID] [--json]
    --status [WORKFLOW_ID] [--creator-id ID] [--json]
    --validate [--creator-id ID] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .workflow_checkpoint import list_all_checkpoints, list_checkpoints, validate_checkpoints
from .workflow_exceptions import WorkflowCliError, WorkflowError
from .workflow_models import WorkflowCheckpoint
from .workflow_runner import WorkflowRunner
from .workflow_state import WorkflowState

PREFIX = "[CreatorLearningWorkflow]"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="creator_learning.workflow_cli", description="Creator Learning Workflow CLI")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true")
    action.add_argument("--resume", metavar="WORKFLOW_ID")
    action.add_argument("--cancel", metavar="WORKFLOW_ID")
    action.add_argument("--status", nargs="?", const="", metavar="WORKFLOW_ID")
    action.add_argument("--validate", action="store_true")

    parser.add_argument("--creator-url")
    parser.add_argument("--creator-id")
    parser.add_argument("--intake-evidence-bundle", type=Path, default=None)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.start and not args.creator_url:
        parser.error("--start requires --creator-url")
    return args


def _checkpoint_to_summary(checkpoint: WorkflowCheckpoint) -> dict:
    return {
        "workflow_id": checkpoint.workflow_id,
        "creator_id": checkpoint.creator_id,
        "username": checkpoint.username,
        "state": checkpoint.state,
        "last_completed_step": checkpoint.last_completed_step,
        "resume_count": checkpoint.resume_count,
        "retry_count": checkpoint.retry_count,
        "restart_count": checkpoint.restart_count,
        "errors": checkpoint.errors,
    }


def _run_start(args: argparse.Namespace, runner: WorkflowRunner) -> dict:
    result = runner.start(args.creator_url, intake_evidence_bundle_path=args.intake_evidence_bundle)
    return _checkpoint_to_summary(result.checkpoint)


def _run_resume(args: argparse.Namespace, runner: WorkflowRunner) -> dict:
    checkpoint = runner.status(args.resume, args.creator_id)
    if checkpoint.state == WorkflowState.FAILED:
        result = runner.retry(args.resume, args.creator_id, intake_evidence_bundle_path=args.intake_evidence_bundle)
    else:
        result = runner.resume(args.resume, args.creator_id, intake_evidence_bundle_path=args.intake_evidence_bundle)
    return _checkpoint_to_summary(result.checkpoint)


def _run_cancel(args: argparse.Namespace, runner: WorkflowRunner) -> dict:
    result = runner.cancel(args.cancel, args.creator_id)
    return _checkpoint_to_summary(result.checkpoint)


def _run_status(args: argparse.Namespace, runner: WorkflowRunner) -> dict:
    if args.status:
        return _checkpoint_to_summary(runner.status(args.status, args.creator_id))
    root = runner.knowledge_base_root()
    checkpoints = list_checkpoints(root, args.creator_id) if args.creator_id else list_all_checkpoints(root)
    return {"workflows": [_checkpoint_to_summary(checkpoint) for checkpoint in checkpoints]}


def _run_validate(args: argparse.Namespace, runner: WorkflowRunner) -> dict:
    passed, errors = validate_checkpoints(runner.knowledge_base_root(), args.creator_id)
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
        runner = WorkflowRunner()
        if args.start:
            result = _run_start(args, runner)
        elif args.resume:
            result = _run_resume(args, runner)
        elif args.cancel:
            result = _run_cancel(args, runner)
        elif args.status is not None:
            result = _run_status(args, runner)
        elif args.validate:
            result = _run_validate(args, runner)
        else:  # pragma: no cover - argparse enforces one action
            raise WorkflowCliError("no action given")
    except (WorkflowCliError, WorkflowError) as exc:
        print(f"{PREFIX} Error: {exc}", file=sys.stderr)
        return 1

    _print_result(result, args.json)
    if isinstance(result, dict) and result.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
