"""CLI entry point for the Video Intelligence OS. Every subcommand
operates on local file paths the operator supplies -- there is no
network access, no browser, no scraping, and no automated evidence
collection anywhere in this module.

Subcommands:
    analyze --platform PLATFORM --evidence PATH [--creator-label LABEL]
             [--reference REF] [--no-knowledge-base] [--json]
    report --video-id ID [--json]
    patterns [--category CATEGORY] [--json]
    validate --evidence PATH [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evidence import InvalidVideoEvidenceTypeError, VideoEvidence, VideoIntelligenceError, VideoPlatform, VideoRecord
from .knowledge_base import KnowledgeBaseError
from .production_dna import VideoIntelligenceConfigError
from .report import build_pattern_report, render_pattern_report_markdown
from .serialization import VideoIntelligenceSerializationError
from .workflow import VideoIntelligenceWorkflow

PREFIX = "[VideoIntelligence]"


class CliError(RuntimeError):
    """Raised for CLI-level failures (bad arguments, bad input files)."""


def _load_evidence_bundle(path: Path, video_id: str) -> list[VideoEvidence]:
    if not path.exists():
        raise CliError(f"Evidence file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise CliError(f"Evidence file must contain a JSON list: {path}")

    evidence: list[VideoEvidence] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CliError(f"Evidence entry {index} is not an object")
        try:
            evidence.append(
                VideoEvidence(
                    video_id=item.get("video_id", video_id),
                    evidence_type=item["evidence_type"],
                    source_description=item.get("source_description", ""),
                    content_excerpt=item.get("content_excerpt", ""),
                    timestamp_seconds=item.get("timestamp_seconds"),
                    collected_at=item.get("collected_at", ""),
                    collected_by=item.get("collected_by", "operator"),
                    tags=list(item.get("tags", [])),
                )
            )
        except KeyError as exc:
            raise CliError(f"Evidence entry {index} missing required field: {exc}") from exc
        except InvalidVideoEvidenceTypeError as exc:
            raise CliError(f"Evidence entry {index} invalid: {exc}") from exc
    return evidence


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="video_intelligence", description="Video Intelligence OS CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one video from an evidence JSON file")
    analyze_parser.add_argument("--platform", required=True, choices=VideoPlatform.ALL)
    analyze_parser.add_argument("--evidence", required=True, type=Path)
    analyze_parser.add_argument("--creator-label", default="")
    analyze_parser.add_argument("--reference", default="")
    analyze_parser.add_argument("--no-knowledge-base", action="store_true")
    analyze_parser.add_argument("--json", action="store_true")

    report_parser = subparsers.add_parser("report", help="Print a saved video report")
    report_parser.add_argument("--video-id", required=True)
    report_parser.add_argument("--json", action="store_true")

    patterns_parser = subparsers.add_parser("patterns", help="Print the cross-video pattern library")
    patterns_parser.add_argument("--category", default=None)
    patterns_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Structurally validate an evidence JSON file")
    validate_parser.add_argument("--evidence", required=True, type=Path)
    validate_parser.add_argument("--video-id", default="")
    validate_parser.add_argument("--json", action="store_true")

    return parser.parse_args(argv)


def _run_analyze(args: argparse.Namespace) -> dict:
    record = VideoRecord(platform=args.platform, creator_label=args.creator_label, reference=args.reference)
    evidence = _load_evidence_bundle(args.evidence, record.video_id)
    workflow = VideoIntelligenceWorkflow()
    result = workflow.analyze_video(record, evidence, ingest_into_knowledge_base=not args.no_knowledge_base)
    return {
        "video_id": result.video_id,
        "dna_id": result.dna.dna_id,
        "overall_confidence": result.dna.overall_confidence,
        "evidence_count": len(result.dna.evidence_index),
        "patterns_touched": len(result.patterns),
    }


def _run_report(args: argparse.Namespace) -> dict:
    workflow = VideoIntelligenceWorkflow()
    markdown = workflow.load_report_markdown(args.video_id)
    if markdown is None:
        raise CliError(f"No report found for video_id={args.video_id!r}")
    return {"video_id": args.video_id, "report_markdown": markdown}


def _run_patterns(args: argparse.Namespace) -> dict:
    workflow = VideoIntelligenceWorkflow()
    patterns = workflow.knowledge_base().list_patterns(args.category)
    report = build_pattern_report(patterns)
    return {
        "total_patterns": report.total_patterns,
        "categories": report.categories,
        "top_patterns": report.top_patterns,
        "report_markdown": render_pattern_report_markdown(report),
    }


def _run_validate(args: argparse.Namespace) -> dict:
    evidence = _load_evidence_bundle(args.evidence, video_id=args.video_id)
    failed_checks: list[str] = []
    for index, item in enumerate(evidence):
        if not item.source_description.strip():
            failed_checks.append(f"entry {index}: source_description must not be empty")
        if not item.video_id:
            failed_checks.append(f"entry {index}: video_id must not be empty")
    return {"passed": not failed_checks, "evidence_count": len(evidence), "failed_checks": failed_checks}


def _print_result(command: str, result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"{PREFIX} {command} result:")
    for key, value in result.items():
        if key == "report_markdown":
            print(f"  {key}:\n{value}")
            continue
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        if args.command == "analyze":
            result = _run_analyze(args)
        elif args.command == "report":
            result = _run_report(args)
        elif args.command == "patterns":
            result = _run_patterns(args)
        elif args.command == "validate":
            result = _run_validate(args)
        else:  # pragma: no cover - argparse enforces valid subcommands
            raise CliError(f"Unknown command: {args.command}")
    except (
        CliError,
        VideoIntelligenceConfigError,
        VideoIntelligenceSerializationError,
        KnowledgeBaseError,
        VideoIntelligenceError,
    ) as exc:
        print(f"{PREFIX} Error: {exc}", file=sys.stderr)
        return 1

    _print_result(args.command, result, args.json)
    if isinstance(result, dict) and result.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
