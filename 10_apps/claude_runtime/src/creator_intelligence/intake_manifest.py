"""Source index, evidence targets, completeness scoring, and the
top-level IntakeManifest for a Creator Evidence Intake workspace
(Phase 12A.1). This module is intentionally independent of
`intake.py` -- it counts evidence by reading the raw JSON files an
intake workspace already has on disk, so `intake.py` can depend on
it (for `SourceIndex`/manifest rebuilding) without a circular import.

Completeness here measures whether enough evidence has been supplied
-- it is NOT a quality score, NOT a confidence score, and NOT a
creator score. It is never fed into `confidence.ConfidenceLevel` or
`TraitScore` computation anywhere.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import _hashing

DEFAULT_INTAKE_CONFIG_RELATIVE_PATH = Path("config") / "creator_intelligence" / "intake.yaml"

# Tags recognized as human-authenticity-relevant evidence, per the
# Phase 12A.1 spec's own §7 vocabulary. Applied to whatever `tags`/
# `manual_tags` a screenshot, highlight, or manual note carries.
HUMAN_AUTHENTICITY_TAGS = frozenset(
    {
        "childhood_or_old_photo",
        "family_depiction",
        "friend_depiction",
        "partner_declared",
        "pet",
        "school_memory",
        "birthday",
        "home",
        "work_context",
        "daily_dump",
        "behind_the_scenes",
        "unpolished_moment",
        "recurring_person",
        "recurring_location",
        "tagged_real_account",
        "collaboration",
        "memory_post",
        "old_travel_photo",
    }
)

_GRID_SOURCE_TYPES = frozenset({"grid_screenshot"})
_POST_SOURCE_TYPES = frozenset({"post_screenshot", "carousel_screenshot"})
_VISUAL_REALISM_SOURCE_TYPES = frozenset({"visual_realism_example"})

CATEGORY_NAMES = (
    "profile",
    "grid",
    "posts",
    "captions",
    "creator_replies",
    "reels",
    "highlights",
    "visual_realism",
    "human_authenticity",
    "relationships",
)


def _runtime_root() -> Path:
    """
    intake_manifest.py location:
    10_apps/claude_runtime/src/creator_intelligence/intake_manifest.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_intake_config_path() -> Path:
    return _runtime_root() / DEFAULT_INTAKE_CONFIG_RELATIVE_PATH


class IntakeManifestError(RuntimeError):
    """Base error for source index / manifest construction and I/O."""


class IntakeConfigError(IntakeManifestError):
    """Raised when config/creator_intelligence/intake.yaml is missing or invalid."""


class DuplicateSourceIdError(IntakeManifestError):
    """Raised when a SourceIndexEntry's source_id already exists in the index."""


class IntakeManifestIntegrityError(IntakeManifestError):
    """Raised when a loaded manifest's recomputed manifest_id does not
    match its stored value."""


@dataclass(slots=True)
class EvidenceTarget:
    minimum: int
    target: int


@dataclass(slots=True)
class IntakeConfig:
    schema_version: str
    targets: dict[str, EvidenceTarget] = field(default_factory=dict)
    package_directories: tuple[str, ...] = ()
    template_source_dir: str = ""

    def target_for(self, category: str) -> EvidenceTarget:
        return self.targets.get(category, EvidenceTarget(minimum=0, target=0))


def load_intake_config(config_path: str | Path | None = None) -> IntakeConfig:
    """Loads config/creator_intelligence/intake.yaml (or an alternate
    path). Raises IntakeConfigError if the file is missing or invalid."""
    path = Path(config_path) if config_path else default_intake_config_path()
    if not path.exists():
        raise IntakeConfigError(f"Intake config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise IntakeConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise IntakeConfigError(f"Intake config is empty or invalid: {path}")

    intake_section = raw.get("intake") or {}
    targets_section = raw.get("targets") or {}
    if not targets_section:
        raise IntakeConfigError(f"targets section must be non-empty in {path}")

    targets: dict[str, EvidenceTarget] = {}
    for name, spec in targets_section.items():
        minimum = int(spec.get("minimum", 0))
        target = int(spec.get("target", minimum))
        targets[str(name)] = EvidenceTarget(minimum=minimum, target=target)

    return IntakeConfig(
        schema_version=str(intake_section.get("schema_version", "1.0")),
        targets=targets,
        package_directories=tuple(raw.get("package_directories") or ()),
        template_source_dir=str(raw.get("template_source_dir", "")),
    )


@dataclass(slots=True)
class SourceIndexEntry:
    source_id: str
    creator_id: str
    source_type: str
    original_reference: str = ""
    local_file: str = ""
    published_at: str = ""
    captured_at: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    notes: str = ""


def _source_index_path(intake_dir: Path) -> Path:
    return Path(intake_dir) / "source_index.json"


def load_source_index(intake_dir: str | Path) -> list[SourceIndexEntry]:
    path = _source_index_path(Path(intake_dir))
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        SourceIndexEntry(
            source_id=item["source_id"],
            creator_id=item.get("creator_id", ""),
            source_type=item.get("source_type", ""),
            original_reference=item.get("original_reference", ""),
            local_file=item.get("local_file", ""),
            published_at=item.get("published_at", ""),
            captured_at=item.get("captured_at", ""),
            evidence_ids=list(item.get("evidence_ids", [])),
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        Path(tmp_name).replace(path)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


def save_source_index(intake_dir: str | Path, entries: list[SourceIndexEntry]) -> Path:
    path = _source_index_path(Path(intake_dir))
    ordered = sorted(entries, key=lambda e: e.source_id)
    payload = [
        {
            "source_id": e.source_id,
            "creator_id": e.creator_id,
            "source_type": e.source_type,
            "original_reference": e.original_reference,
            "local_file": e.local_file,
            "published_at": e.published_at,
            "captured_at": e.captured_at,
            "evidence_ids": list(e.evidence_ids),
            "notes": e.notes,
        }
        for e in ordered
    ]
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def add_source_entry(entries: list[SourceIndexEntry], new_entry: SourceIndexEntry) -> list[SourceIndexEntry]:
    """Returns a new, deterministically-ordered list with `new_entry`
    appended. Raises DuplicateSourceIdError if `new_entry.source_id`
    already exists."""
    if any(e.source_id == new_entry.source_id for e in entries):
        raise DuplicateSourceIdError(f"source_id already exists: {new_entry.source_id!r}")
    return sorted([*entries, new_entry], key=lambda e: e.source_id)


def compute_completeness(
    counts_by_category: dict[str, int], config: IntakeConfig
) -> tuple[dict[str, float], float]:
    """Returns (per_category completeness in [0.0, 1.0], overall
    unweighted mean). A category with target<=0 is treated as fully
    complete (nothing was asked for). This is a coverage measure only
    -- never a quality/confidence/creator score."""
    per_category: dict[str, float] = {}
    for name in CATEGORY_NAMES:
        target = config.target_for(name).target
        count = counts_by_category.get(name, 0)
        if target <= 0:
            per_category[name] = 1.0
        else:
            per_category[name] = min(1.0, count / target)
    overall = sum(per_category.values()) / len(per_category) if per_category else 0.0
    return per_category, overall


def missing_recommended_evidence(counts_by_category: dict[str, int], config: IntakeConfig) -> list[str]:
    """Categories below their configured `minimum` -- a harder floor
    than `target`."""
    missing = []
    for name in CATEGORY_NAMES:
        minimum = config.target_for(name).minimum
        if counts_by_category.get(name, 0) < minimum:
            missing.append(name)
    return missing


def _load_json_records(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def count_categories(intake_dir: str | Path) -> dict[str, int]:
    """Counts evidence per target category by reading the on-disk
    per-category directories directly -- transparent and trivially
    verifiable with `ls`."""
    intake_dir = Path(intake_dir)
    counts = {name: 0 for name in CATEGORY_NAMES}

    counts["profile"] = 1 if (intake_dir / "creator_profile.json").is_file() else 0

    screenshots = _load_json_records(intake_dir / "screenshots")
    for item in screenshots:
        source_type = item.get("source_type", "")
        if source_type in _GRID_SOURCE_TYPES:
            counts["grid"] += 1
        if source_type in _POST_SOURCE_TYPES:
            counts["posts"] += 1
        if source_type in _VISUAL_REALISM_SOURCE_TYPES or item.get("visual_annotations") is not None:
            counts["visual_realism"] += 1
        tags = set(item.get("manual_tags", [])) | set(item.get("tags", []))
        if tags.intersection(HUMAN_AUTHENTICITY_TAGS):
            counts["human_authenticity"] += 1

    counts["captions"] = len(_load_json_records(intake_dir / "captions"))
    counts["creator_replies"] = len(_load_json_records(intake_dir / "replies"))
    counts["reels"] = len(_load_json_records(intake_dir / "reels"))
    counts["highlights"] = len(_load_json_records(intake_dir / "highlights"))
    relationship_records = _load_json_records(intake_dir / "relationships")
    counts["relationships"] = len(relationship_records)
    # Every relationship annotation is inherently a human-authenticity
    # observation (it exists specifically to qualify a depicted/claimed
    # relationship's evidentiary basis), so it counts toward both targets.
    counts["human_authenticity"] += len(relationship_records)

    for item in _load_json_records(intake_dir / "highlights") + _load_json_records(intake_dir / "notes"):
        tags = set(item.get("manual_tags", [])) | set(item.get("tags", []))
        if tags.intersection(HUMAN_AUTHENTICITY_TAGS):
            counts["human_authenticity"] += 1

    return counts


@dataclass(slots=True)
class IntakeManifest:
    schema_version: str
    creator_id: str
    username: str
    platform: str
    profile_url: str
    created_at: str = ""
    updated_at: str = ""
    source_count: int = 0
    evidence_count: int = 0
    counts_by_type: dict[str, int] = field(default_factory=dict)
    completeness: dict[str, float] = field(default_factory=dict)
    overall_completeness: float = 0.0
    missing_recommended_evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    manifest_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.manifest_id = _compute_manifest_id(self)


def _compute_manifest_id(manifest: IntakeManifest) -> str:
    payload = {
        "schema_version": manifest.schema_version,
        "creator_id": manifest.creator_id,
        "username": manifest.username,
        "platform": manifest.platform,
        "profile_url": manifest.profile_url,
        "source_count": manifest.source_count,
        "evidence_count": manifest.evidence_count,
        "counts_by_type": dict(sorted(manifest.counts_by_type.items())),
        "completeness": {k: round(v, 6) for k, v in sorted(manifest.completeness.items())},
        "overall_completeness": round(manifest.overall_completeness, 6),
        "missing_recommended_evidence": sorted(manifest.missing_recommended_evidence),
        "warnings": sorted(manifest.warnings),
        "source_hashes": dict(sorted(manifest.source_hashes.items())),
        "metadata": manifest.metadata,
    }
    return _hashing.content_hash(payload)


def build_intake_manifest(
    *,
    creator_id: str,
    username: str,
    platform: str,
    profile_url: str,
    intake_dir: str | Path,
    intake_config: IntakeConfig,
    schema_version: str = "1.0",
    created_at: str = "",
    updated_at: str = "",
    source_hashes: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    metadata: dict | None = None,
) -> IntakeManifest:
    """Rebuilds the manifest from the intake workspace's *current*
    on-disk state -- counts, completeness, and missing targets are
    always freshly derived, never carried over from a prior manifest."""
    intake_dir = Path(intake_dir)
    counts = count_categories(intake_dir)
    per_category, overall = compute_completeness(counts, intake_config)
    missing = missing_recommended_evidence(counts, intake_config)

    source_index = load_source_index(intake_dir)
    evidence_bundle_path = intake_dir / "evidence_bundle.json"
    evidence_count = 0
    if evidence_bundle_path.is_file():
        evidence_count = len(json.loads(evidence_bundle_path.read_text(encoding="utf-8")))

    return IntakeManifest(
        schema_version=schema_version,
        creator_id=creator_id,
        username=username,
        platform=platform,
        profile_url=profile_url,
        created_at=created_at,
        updated_at=updated_at,
        source_count=len(source_index),
        evidence_count=evidence_count,
        counts_by_type=counts,
        completeness=per_category,
        overall_completeness=overall,
        missing_recommended_evidence=missing,
        warnings=list(warnings or []),
        source_hashes=dict(source_hashes or {}),
        metadata=dict(metadata or {}),
    )


def _manifest_path(intake_dir: Path) -> Path:
    return Path(intake_dir) / "intake_manifest.json"


def save_intake_manifest(intake_dir: str | Path, manifest: IntakeManifest) -> Path:
    path = _manifest_path(Path(intake_dir))
    payload = {
        "schema_version": manifest.schema_version,
        "manifest_id": manifest.manifest_id,
        "creator_id": manifest.creator_id,
        "username": manifest.username,
        "platform": manifest.platform,
        "profile_url": manifest.profile_url,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "source_count": manifest.source_count,
        "evidence_count": manifest.evidence_count,
        "counts_by_type": manifest.counts_by_type,
        "completeness": manifest.completeness,
        "overall_completeness": manifest.overall_completeness,
        "missing_recommended_evidence": manifest.missing_recommended_evidence,
        "warnings": manifest.warnings,
        "source_hashes": manifest.source_hashes,
        "metadata": manifest.metadata,
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_intake_manifest(intake_dir: str | Path) -> IntakeManifest:
    path = _manifest_path(Path(intake_dir))
    if not path.exists():
        raise IntakeManifestError(f"Intake manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = IntakeManifest(
        schema_version=payload.get("schema_version", "1.0"),
        creator_id=payload["creator_id"],
        username=payload.get("username", ""),
        platform=payload.get("platform", ""),
        profile_url=payload.get("profile_url", ""),
        created_at=payload.get("created_at", ""),
        updated_at=payload.get("updated_at", ""),
        source_count=payload.get("source_count", 0),
        evidence_count=payload.get("evidence_count", 0),
        counts_by_type=dict(payload.get("counts_by_type", {})),
        completeness=dict(payload.get("completeness", {})),
        overall_completeness=payload.get("overall_completeness", 0.0),
        missing_recommended_evidence=list(payload.get("missing_recommended_evidence", [])),
        warnings=list(payload.get("warnings", [])),
        source_hashes=dict(payload.get("source_hashes", {})),
        metadata=dict(payload.get("metadata", {})),
    )
    stored_id = payload.get("manifest_id")
    if stored_id is not None and stored_id != manifest.manifest_id:
        raise IntakeManifestIntegrityError(
            f"manifest_id integrity check failed: stored={stored_id!r} recomputed={manifest.manifest_id!r}"
        )
    return manifest


def sha256_file(path: str | Path, *, chunk_size: int = 65536) -> str:
    """Chunked SHA-256 of a file's contents. Used for source integrity/
    duplicate detection/source-unchanged proof -- never embeds binary
    data into JSON, only the resulting hex digest."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
