"""CLI entry point for the Creator Intelligence Framework. Every
subcommand operates on local file paths the operator supplies --
there is no network access, no login, and no automated collection
anywhere in this module.

Subcommands:
    validate --evidence PATH [--config PATH] [--evidence-types-config PATH]
    build --evidence PATH --subject-label LABEL --output PATH [--config PATH] [--evidence-types-config PATH]
    show --dna PATH
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CreatorIntelligenceConfigError, load_framework_config
from .evidence import Evidence, EvidenceError
from .serialization import CreatorIntelligenceSerializationError, load_creator_dna, save_creator_dna
from .summary_builder import build_creator_dna
from .validation import validate_creator_dna, validate_evidence

PREFIX = "[CreatorIntelligence]"


class CliError(RuntimeError):
    """Raised for CLI-level failures (bad arguments, bad input files)."""


def _load_evidence_file(path: Path) -> list[Evidence]:
    if not path.exists():
        raise CliError(f"Evidence file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise CliError(f"Evidence file must contain a JSON list: {path}")

    evidence: list[Evidence] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CliError(f"Evidence entry {index} is not an object")
        try:
            evidence.append(
                Evidence(
                    evidence_type=item["evidence_type"],
                    source_description=item.get("source_description", ""),
                    content_excerpt=item.get("content_excerpt", ""),
                    collected_at=item.get("collected_at", ""),
                    collected_by=item.get("collected_by", "operator"),
                    tags=list(item.get("tags", [])),
                )
            )
        except KeyError as exc:
            raise CliError(f"Evidence entry {index} missing required field: {exc}") from exc
        except EvidenceError as exc:
            raise CliError(f"Evidence entry {index} invalid: {exc}") from exc
    return evidence


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="creator_intelligence", description="Creator Intelligence Framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate an evidence JSON file")
    validate_parser.add_argument("--evidence", required=True, type=Path)
    validate_parser.add_argument("--config", type=Path, default=None)
    validate_parser.add_argument("--evidence-types-config", type=Path, default=None)
    validate_parser.add_argument("--json", action="store_true")

    build_parser = subparsers.add_parser("build", help="Build a CreatorDNA from an evidence JSON file")
    build_parser.add_argument("--evidence", required=True, type=Path)
    build_parser.add_argument("--subject-label", required=True)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--config", type=Path, default=None)
    build_parser.add_argument("--evidence-types-config", type=Path, default=None)
    build_parser.add_argument("--force", action="store_true")
    build_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Load and print a saved CreatorDNA")
    show_parser.add_argument("--dna", required=True, type=Path)
    show_parser.add_argument("--json", action="store_true")

    return parser.parse_args(argv)


def _run_validate(args: argparse.Namespace) -> dict:
    config = load_framework_config(args.config, args.evidence_types_config)
    evidence = _load_evidence_file(args.evidence)
    results = [validate_evidence(item, config) for item in evidence]
    passed = all(result.passed for result in results)
    return {
        "passed": passed,
        "evidence_count": len(evidence),
        "failed_checks": [check for result in results for check in result.failed_checks],
    }


def _run_build(args: argparse.Namespace) -> dict:
    if args.output.exists() and not args.force:
        raise CliError(f"Output already exists (use --force to overwrite): {args.output}")
    config = load_framework_config(args.config, args.evidence_types_config)
    evidence = _load_evidence_file(args.evidence)
    dna = build_creator_dna(args.subject_label, evidence, config)
    dna_validation = validate_creator_dna(dna, config)
    if not dna_validation.passed:
        raise CliError(f"Built CreatorDNA failed validation: {dna_validation.failed_checks}")
    save_creator_dna(dna, args.output)
    return {
        "dna_id": dna.dna_id,
        "subject_label": dna.subject_label,
        "overall_confidence": dna.overall_confidence,
        "output": str(args.output),
    }


def _run_show(args: argparse.Namespace) -> dict:
    dna = load_creator_dna(args.dna)
    return {
        "dna_id": dna.dna_id,
        "subject_label": dna.subject_label,
        "schema_version": dna.schema_version,
        "overall_confidence": dna.overall_confidence,
        "evidence_count": len(dna.evidence_index),
        "warnings": dna.warnings,
    }


def _print_result(command: str, result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"{PREFIX} {command} result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        if args.command == "validate":
            result = _run_validate(args)
        elif args.command == "build":
            result = _run_build(args)
        elif args.command == "show":
            result = _run_show(args)
        else:  # pragma: no cover - argparse enforces valid subcommands
            raise CliError(f"Unknown command: {args.command}")
    except (CliError, CreatorIntelligenceConfigError, CreatorIntelligenceSerializationError, EvidenceError) as exc:
        print(f"{PREFIX} Error: {exc}", file=sys.stderr)
        return 1

    _print_result(args.command, result, args.json)
    if isinstance(result, dict) and result.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
