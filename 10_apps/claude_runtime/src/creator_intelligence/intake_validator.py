"""Validates a Creator Evidence Intake workspace (Phase 12A.1) --
structure, ids, links, and completeness. Read-only: never writes
anything to the workspace it inspects. This module never runs a
Creator DNA analyzer and never fetches anything from the network.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import intake_manifest as im
from .evidence import Evidence, EvidenceError, EvidenceType

PREFIX = "[CreatorIntelligence]"

_DATE_ISH_MIN_LENGTH = 8  # e.g. "2026-08-07" -- lenient, not a strict ISO parse


class IntakeValidatorError(RuntimeError):
    """Raised for CLI-level failures (bad --intake path, unreadable files)."""


@dataclass(slots=True)
class IntakeValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    completeness: dict = field(default_factory=dict)
    missing_targets: list[str] = field(default_factory=list)
    summary: str = ""


def _read_json(path: Path) -> tuple[object | None, str | None]:
    if not path.exists():
        return None, f"missing file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"malformed JSON in {path}: {exc}"


def _looks_like_date(value: object) -> bool:
    if value in (None, ""):
        return True  # absent/unknown dates are allowed, not flagged
    return isinstance(value, str) and len(value) >= _DATE_ISH_MIN_LENGTH and value[:4].isdigit()


def _is_json_serializable(value: object) -> bool:
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False


def _validate_local_path_safety(intake_dir: Path, local_path: str) -> str | None:
    if not local_path:
        return None
    intake_root = intake_dir.resolve()
    candidate = (intake_dir / local_path).resolve()
    if intake_root not in candidate.parents and candidate != intake_root:
        return f"local path escapes intake workspace: {local_path!r}"
    return None


def validate_intake_workspace(
    intake_dir: str | Path,
    intake_config: im.IntakeConfig,
    *,
    strict: bool = False,
) -> IntakeValidationResult:
    intake_dir = Path(intake_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not intake_dir.is_dir():
        return IntakeValidationResult(
            passed=False, errors=[f"intake directory does not exist: {intake_dir}"], summary="FAILED (0 check(s) run)"
        )

    profile, profile_err = _read_json(intake_dir / "creator_profile.json")
    if profile_err:
        errors.append(profile_err)
        profile = {}
    profile = profile or {}

    profile_creator_id = profile.get("creator_id", "")
    if not profile_creator_id:
        errors.append("creator_profile.json missing creator_id")
    if not profile.get("username"):
        errors.append("creator_profile.json missing username")
    profile_url = profile.get("profile_url", "")
    if not profile_url or not (profile_url.startswith("http://") or profile_url.startswith("https://")):
        errors.append(f"creator_profile.json profile_url is not a valid URL: {profile_url!r}")

    evidence_bundle, bundle_err = _read_json(intake_dir / "evidence_bundle.json")
    if bundle_err:
        errors.append(bundle_err)
        evidence_bundle = []
    evidence_bundle = evidence_bundle or []
    if not isinstance(evidence_bundle, list):
        errors.append("evidence_bundle.json must contain a JSON list")
        evidence_bundle = []

    seen_evidence_ids: set[str] = set()
    for index, item in enumerate(evidence_bundle):
        if not isinstance(item, dict):
            errors.append(f"evidence_bundle[{index}] is not an object")
            continue
        stored_id = item.get("evidence_id", "")
        try:
            reconstructed = Evidence(
                evidence_type=item.get("evidence_type", ""),
                source_description=item.get("source_description", ""),
                content_excerpt=item.get("content_excerpt", ""),
                collected_at=item.get("collected_at", ""),
                collected_by=item.get("collected_by", "operator"),
                tags=list(item.get("tags", [])),
            )
        except EvidenceError as exc:
            errors.append(f"evidence_bundle[{index}] invalid: {exc}")
            continue
        if reconstructed.evidence_id != stored_id:
            errors.append(
                f"evidence_bundle[{index}] evidence_id mismatch: stored={stored_id!r} "
                f"recomputed={reconstructed.evidence_id!r}"
            )
        if stored_id in seen_evidence_ids:
            errors.append(f"duplicate evidence_id in evidence_bundle.json: {stored_id!r}")
        seen_evidence_ids.add(stored_id)
        if item.get("evidence_type") not in EvidenceType.ALL:
            errors.append(f"evidence_bundle[{index}] has unsupported evidence_type: {item.get('evidence_type')!r}")

    source_index = im.load_source_index(intake_dir)
    seen_source_ids: set[str] = set()
    referenced_evidence_ids: set[str] = set()
    for entry in source_index:
        if entry.source_id in seen_source_ids:
            errors.append(f"duplicate source_id in source_index.json: {entry.source_id!r}")
        seen_source_ids.add(entry.source_id)
        if entry.creator_id and profile_creator_id and entry.creator_id != profile_creator_id:
            errors.append(f"source {entry.source_id!r} creator_id mismatch with profile")
        path_error = _validate_local_path_safety(intake_dir, entry.local_file)
        if path_error:
            errors.append(f"source {entry.source_id!r}: {path_error}")
        elif entry.local_file:
            resolved = (intake_dir / entry.local_file).resolve()
            if not resolved.exists():
                errors.append(f"source {entry.source_id!r} references a missing file: {entry.local_file}")
            elif resolved.is_file() and resolved.stat().st_size == 0:
                warnings.append(f"source {entry.source_id!r} references an empty file: {entry.local_file}")
        for evidence_id in entry.evidence_ids:
            referenced_evidence_ids.add(evidence_id)
            if evidence_id not in seen_evidence_ids:
                errors.append(f"source {entry.source_id!r} references unknown evidence_id {evidence_id!r}")
        if not _looks_like_date(entry.published_at):
            warnings.append(f"source {entry.source_id!r} has an unparseable published_at: {entry.published_at!r}")
        if not _looks_like_date(entry.captured_at):
            warnings.append(f"source {entry.source_id!r} has an unparseable captured_at: {entry.captured_at!r}")

    for evidence_id in seen_evidence_ids:
        if evidence_id not in referenced_evidence_ids:
            warnings.append(f"evidence_id {evidence_id!r} is not referenced by any source_index entry")

    errors.extend(_validate_captions(intake_dir))
    errors.extend(_validate_reply_pairs(intake_dir))
    errors.extend(_validate_reels(intake_dir))
    errors.extend(_validate_creator_mismatch(intake_dir, profile_creator_id))
    errors.extend(_validate_metadata_serializability(intake_dir))

    counts = im.count_categories(intake_dir)
    per_category, overall = im.compute_completeness(counts, intake_config)
    # target >= minimum always (see IntakeConfig), so "below target" is
    # the superset that also covers every "below minimum" category.
    missing_target = [
        name for name in im.CATEGORY_NAMES if counts.get(name, 0) < intake_config.target_for(name).target
    ]

    for name in missing_target:
        warnings.append(f"below recommended target for category {name!r}")
        if strict:
            errors.append(f"below target for category {name!r} (strict mode)")

    missing_targets = list(missing_target)

    manifest_path = intake_dir / "intake_manifest.json"
    if manifest_path.exists():
        try:
            manifest = im.load_intake_manifest(intake_dir)
        except im.IntakeManifestIntegrityError as exc:
            errors.append(str(exc))
        else:
            if manifest.evidence_count != len(evidence_bundle):
                errors.append(
                    f"manifest evidence_count {manifest.evidence_count} does not match "
                    f"evidence_bundle.json length {len(evidence_bundle)}"
                )
            if manifest.source_count != len(source_index):
                errors.append(
                    f"manifest source_count {manifest.source_count} does not match source_index.json length {len(source_index)}"
                )
            if manifest.counts_by_type != counts:
                errors.append("manifest counts_by_type does not match recomputed counts")

    passed = not errors
    summary = "PASSED" if passed else f"FAILED ({len(errors)} error(s), {len(warnings)} warning(s))"
    return IntakeValidationResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        completeness={"per_category": per_category, "overall": overall},
        missing_targets=sorted(set(missing_targets)),
        summary=summary,
    )


def _validate_captions(intake_dir: Path) -> list[str]:
    errors: list[str] = []
    directory = intake_dir / "captions"
    if not directory.is_dir():
        return errors
    for path in sorted(directory.glob("*.json")):
        payload, err = _read_json(path)
        if err:
            errors.append(err)
            continue
        if not isinstance(payload, dict) or not payload.get("caption_text"):
            errors.append(f"malformed caption record (missing caption_text): {path}")
    return errors


def _validate_reply_pairs(intake_dir: Path) -> list[str]:
    errors: list[str] = []
    directory = intake_dir / "replies"
    if not directory.is_dir():
        return errors
    for path in sorted(directory.glob("*.json")):
        payload, err = _read_json(path)
        if err:
            errors.append(err)
            continue
        if not isinstance(payload, dict) or not payload.get("creator_reply") or not payload.get("audience_comment"):
            errors.append(f"malformed reply pair (missing audience_comment/creator_reply): {path}")
    return errors


def _validate_reels(intake_dir: Path) -> list[str]:
    errors: list[str] = []
    directory = intake_dir / "reels"
    if not directory.is_dir():
        return errors
    for path in sorted(directory.glob("*.json")):
        payload, err = _read_json(path)
        if err:
            errors.append(err)
            continue
        if not isinstance(payload, dict):
            continue
        shot_notes = payload.get("shot_notes") or []
        last_index = None
        for shot in shot_notes:
            index = shot.get("shot_index")
            if last_index is not None and isinstance(index, int) and index < last_index:
                errors.append(f"reel {path.name} has out-of-order shot_index: {index} after {last_index}")
            if isinstance(index, int):
                last_index = index
    return errors


def _validate_creator_mismatch(intake_dir: Path, profile_creator_id: str) -> list[str]:
    errors: list[str] = []
    if not profile_creator_id:
        return errors
    for category in ("screenshots", "captions", "replies", "reels", "highlights", "relationships", "notes"):
        directory = intake_dir / category
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload, err = _read_json(path)
            if err or not isinstance(payload, dict):
                continue
            record_creator_id = payload.get("creator_id")
            if record_creator_id and record_creator_id != profile_creator_id:
                errors.append(f"{path} creator_id {record_creator_id!r} does not match profile creator_id")
    return errors


def _validate_metadata_serializability(intake_dir: Path) -> list[str]:
    errors: list[str] = []
    for category in ("screenshots", "captions", "replies", "reels", "highlights"):
        directory = intake_dir / category
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload, err = _read_json(path)
            if err or not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata")
            if metadata is not None and not _is_json_serializable(metadata):
                errors.append(f"{path} has non-JSON-serializable metadata")
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="creator_intelligence.intake_validator", description="Validate a Creator Evidence Intake workspace"
    )
    parser.add_argument("--intake", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _print_result(result: IntakeValidationResult, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "passed": result.passed,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "completeness": result.completeness,
                    "missing_targets": result.missing_targets,
                    "summary": result.summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(f"{PREFIX} intake validation: {result.summary}")
    for error in result.errors:
        print(f"  ERROR: {error}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")
    print(f"  overall completeness: {result.completeness.get('overall', 0.0):.2%}")
    if result.missing_targets:
        print(f"  missing targets: {', '.join(result.missing_targets)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        intake_config = im.load_intake_config(args.config)
    except im.IntakeConfigError as exc:
        print(f"{PREFIX} Error: {exc}", file=sys.stderr)
        return 1

    result = validate_intake_workspace(args.intake, intake_config, strict=args.strict)
    _print_result(result, args.json)
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
