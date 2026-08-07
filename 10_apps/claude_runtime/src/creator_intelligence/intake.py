"""Creator Evidence Intake (Phase 12A.1) -- turns manually-gathered
observations about a public creator into well-formed
`evidence.Evidence` records, organized into a per-creator intake
workspace that Phase 12A.2 can later analyze. Every record type here
converts onto one of the five existing `EvidenceType` members with
free-form tags -- no change to `evidence.py`/`models.py`/
`serialization.py` was needed (see the Phase 12A.1 plan's scope
decision). This module performs NO web access, browser automation, or
login of any kind; every input is a local file the operator supplies
or a value passed on the CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import _hashing
from . import intake_manifest as im
from .evidence import Evidence, EvidenceError, EvidenceType
from .models import RelationshipBasis

PREFIX = "[CreatorIntelligence]"
MARKER_FILENAME = ".ai_influencer_creator_intake"


def _runtime_root() -> Path:
    """
    intake.py location: 10_apps/claude_runtime/src/creator_intelligence/intake.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IntakeError(RuntimeError):
    """Base error for the Creator Evidence Intake workflow."""


class IntakeWorkspaceError(IntakeError):
    """Raised for workspace creation/access failures."""


class UnsafeIntakeWorkspaceError(IntakeWorkspaceError):
    """Raised when --init targets an existing, non-empty directory
    that is not already an owned intake workspace."""


class IntakeWorkspaceOwnershipError(IntakeWorkspaceError):
    """Raised when an operation targets a directory without this
    workspace's ownership marker, or when --init would overwrite an
    existing owned workspace without --force."""


class IntakeRecordExistsError(IntakeError):
    """Raised when an ingested record's id already has a file on disk
    and force was not given."""


class UnsafeLocalPathError(IntakeError):
    """Raised when a record's local_path would resolve outside the
    intake workspace (path traversal guard)."""


class MissingLocalFileError(IntakeError):
    """Raised when a record references a local file that does not
    exist."""


class InvalidAnnotationValueError(IntakeError):
    """Raised when a VisualRealismAnnotation/RelationshipAnnotation
    field is outside its controlled vocabulary."""


class IntakeRequestError(IntakeError):
    """Raised for structurally invalid CLI input."""


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class TriState:
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"
    UNKNOWN = "unknown"
    ALL = (YES, NO, UNCLEAR, UNKNOWN)


class Intensity:
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    UNKNOWN = "unknown"
    ALL = (NONE, LIGHT, MODERATE, HEAVY, UNKNOWN)


class RelationshipRole:
    FRIEND_LIKE_CONTEXT = "friend_like_context"
    FAMILY_DEPICTION = "family_depiction"
    COLLABORATION = "collaboration"
    PUBLIC_CREATOR_INTERACTION = "public_creator_interaction"
    UNKNOWN_PERSON = "unknown_person"
    ALL = (FRIEND_LIKE_CONTEXT, FAMILY_DEPICTION, COLLABORATION, PUBLIC_CREATOR_INTERACTION, UNKNOWN_PERSON)


_TRI_STATE_FIELDS = (
    "skin_texture_visible",
    "pores_visible",
    "blemishes_visible",
    "sensor_noise",
    "grain",
    "bloom",
    "flash",
    "highlight_clipping",
    "ccd_like_aesthetic",
    "compact_camera_aesthetic",
    "film_like_aesthetic",
    "phone_processing_aesthetic",
    "ordinary_location",
    "luxury_location",
)
_INTENSITY_FIELDS = (
    "skin_smoothing",
    "makeup_level",
    "focus_softness",
    "motion_blur",
    "editing_strength",
    "background_clutter",
    "environment_realism",
)
_FREE_TEXT_FIELDS = ("white_balance", "shadow_detail", "dynamic_range", "color_cast", "resolution_feel")
_MAX_FREE_TEXT_CHARS = 120


def _stable_record_id(payload: dict, *, exclude: tuple[str, ...] = ()) -> str:
    trimmed = {k: v for k, v in payload.items() if k not in exclude}
    return _hashing.content_hash(trimmed)


def _strip_comment_keys(payload: dict) -> dict:
    """Strips `_comment`-prefixed keys (documentation-only, from the
    fill-in-the-blank templates under data/creator_intelligence/intake_templates/)."""
    return {k: v for k, v in payload.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def creator_id(platform: str, username: str) -> str:
    """Deterministic creator identity, stable across re-runs -- no
    timestamp, no free-text field that could vary between calls."""
    return _hashing.content_hash({"platform": platform, "username": username})


@dataclass(slots=True)
class CreatorProfile:
    platform: str
    username: str
    profile_url: str
    display_name: str | None = None
    bio: str | None = None
    category: str | None = None
    region: str | None = None
    languages: list[str] = field(default_factory=list)
    analysis_start_date: str | None = None
    analysis_end_date: str | None = None
    notes: str = ""
    metadata: dict = field(default_factory=dict)
    creator_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.creator_id = creator_id(self.platform, self.username)


@dataclass(slots=True)
class VisualRealismAnnotation:
    skin_texture_visible: str = TriState.UNKNOWN
    pores_visible: str = TriState.UNKNOWN
    blemishes_visible: str = TriState.UNKNOWN
    skin_smoothing: str = Intensity.UNKNOWN
    makeup_level: str = Intensity.UNKNOWN
    focus_softness: str = Intensity.UNKNOWN
    motion_blur: str = Intensity.UNKNOWN
    sensor_noise: str = TriState.UNKNOWN
    grain: str = TriState.UNKNOWN
    bloom: str = TriState.UNKNOWN
    flash: str = TriState.UNKNOWN
    white_balance: str = "unknown"
    highlight_clipping: str = TriState.UNKNOWN
    shadow_detail: str = "unknown"
    dynamic_range: str = "unknown"
    color_cast: str = "unknown"
    resolution_feel: str = "unknown"
    ccd_like_aesthetic: str = TriState.UNKNOWN
    compact_camera_aesthetic: str = TriState.UNKNOWN
    film_like_aesthetic: str = TriState.UNKNOWN
    phone_processing_aesthetic: str = TriState.UNKNOWN
    editing_strength: str = Intensity.UNKNOWN
    environment_realism: str = Intensity.UNKNOWN
    background_clutter: str = Intensity.UNKNOWN
    ordinary_location: str = TriState.UNKNOWN
    luxury_location: str = TriState.UNKNOWN
    notes: str = ""

    def __post_init__(self) -> None:
        for name in _TRI_STATE_FIELDS:
            value = getattr(self, name)
            if value not in TriState.ALL:
                raise InvalidAnnotationValueError(f"{name} must be one of {TriState.ALL}, got {value!r}")
        for name in _INTENSITY_FIELDS:
            value = getattr(self, name)
            if value not in Intensity.ALL:
                raise InvalidAnnotationValueError(f"{name} must be one of {Intensity.ALL}, got {value!r}")
        for name in _FREE_TEXT_FIELDS:
            value = getattr(self, name)
            if len(value) > _MAX_FREE_TEXT_CHARS:
                raise InvalidAnnotationValueError(f"{name} exceeds {_MAX_FREE_TEXT_CHARS} chars")


@dataclass(slots=True)
class RelationshipAnnotation:
    role: str
    basis: str
    confidence: str
    description: str = ""
    annotation_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.role not in RelationshipRole.ALL:
            raise InvalidAnnotationValueError(f"role must be one of {RelationshipRole.ALL}, got {self.role!r}")
        if self.basis not in RelationshipBasis.ALL:
            raise InvalidAnnotationValueError(f"basis must be one of {RelationshipBasis.ALL}, got {self.basis!r}")
        if self.confidence not in ("low", "medium", "high"):
            raise InvalidAnnotationValueError(f"confidence must be one of low/medium/high, got {self.confidence!r}")
        self.annotation_id = _stable_record_id(
            {"role": self.role, "basis": self.basis, "confidence": self.confidence, "description": self.description}
        )


@dataclass(slots=True)
class ScreenshotEvidence:
    creator_id: str
    source_type: str
    local_path: str = ""
    original_filename: str | None = None
    captured_at: str | None = None
    source_url: str | None = None
    post_date: str | None = None
    caption_reference: str | None = None
    manual_notes: str = ""
    visual_annotations: VisualRealismAnnotation | None = None
    relationship_annotations: list[RelationshipAnnotation] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    screenshot_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.screenshot_id = _stable_record_id(
            {
                "creator_id": self.creator_id,
                "source_type": self.source_type,
                "local_path": self.local_path,
                "original_filename": self.original_filename,
                "source_url": self.source_url,
                "post_date": self.post_date,
                "caption_reference": self.caption_reference,
                "manual_notes": self.manual_notes,
            }
        )


@dataclass(slots=True)
class CaptionRecord:
    creator_id: str
    post_reference: str | None
    caption_text: str
    published_at: str | None = None
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    language: str | None = None
    manual_tags: list[str] = field(default_factory=list)
    metrics_if_known: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    caption_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.caption_id = _stable_record_id(
            {
                "creator_id": self.creator_id,
                "post_reference": self.post_reference,
                "caption_text": self.caption_text,
                "hashtags": sorted(self.hashtags),
                "mentions": sorted(self.mentions),
            }
        )


@dataclass(slots=True)
class ReplyPair:
    creator_id: str
    post_reference: str | None
    audience_comment: str
    creator_reply: str
    comment_language: str | None = None
    reply_language: str | None = None
    comment_timestamp: str | None = None
    reply_timestamp: str | None = None
    emoji: list[str] = field(default_factory=list)
    thread_depth: int = 0
    audience_username_optional: str | None = None
    relationship_context: str | None = None
    manual_tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    reply_pair_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.reply_pair_id = _stable_record_id(
            {
                "creator_id": self.creator_id,
                "post_reference": self.post_reference,
                "audience_comment": self.audience_comment,
                "creator_reply": self.creator_reply,
                "thread_depth": self.thread_depth,
            }
        )


@dataclass(slots=True)
class ShotNote:
    shot_index: int
    start_seconds_optional: float | None = None
    end_seconds_optional: float | None = None
    subject_action: str = ""
    camera_angle: str = ""
    camera_motion: str = ""
    environment: str = ""
    emotion: str = ""
    interaction: str = ""
    text_overlay: str = ""
    notes: str = ""


@dataclass(slots=True)
class ReelEvidence:
    creator_id: str
    source_reference: str | None
    published_at: str | None = None
    duration_seconds_if_known: float | None = None
    views_if_known: int | None = None
    likes_if_known: int | None = None
    comments_if_known: int | None = None
    shot_notes: list[ShotNote] = field(default_factory=list)
    hook_notes: str = ""
    subtitle_notes: str = ""
    music_notes: str = ""
    camera_notes: str = ""
    story_arc_notes: str = ""
    manual_tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    reel_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.reel_id = _stable_record_id(
            {
                "creator_id": self.creator_id,
                "source_reference": self.source_reference,
                "hook_notes": self.hook_notes,
                "story_arc_notes": self.story_arc_notes,
            }
        )


@dataclass(slots=True)
class HighlightItem:
    creator_id: str
    highlight_name: str
    item_index: int
    source_reference: str | None = None
    content_type: str | None = None
    manual_summary: str = ""
    people_context: str | None = None
    location_context: str | None = None
    story_type: str | None = None
    memory_type: str | None = None
    metadata: dict = field(default_factory=dict)
    highlight_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.highlight_id = _stable_record_id(
            {"creator_id": self.creator_id, "highlight_name": self.highlight_name, "item_index": self.item_index}
        )


# ---------------------------------------------------------------------------
# Record -> Evidence conversion
# ---------------------------------------------------------------------------


def _capped_excerpt(text: str, max_chars: int) -> tuple[str, list[str]]:
    if len(text) <= max_chars:
        return text, []
    return text[:max_chars], [f"content truncated to {max_chars} chars (was {len(text)})"]


def to_evidence_from_screenshot(screenshot: ScreenshotEvidence, max_excerpt_chars: int) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(screenshot.manual_notes, max_excerpt_chars)
    tags = [screenshot.source_type]
    if screenshot.visual_annotations is not None:
        tags.append("visual_realism")
    evidence = Evidence(
        evidence_type=EvidenceType.SCREENSHOT,
        source_description=f"screenshot ({screenshot.source_type}) for creator {screenshot.creator_id}",
        content_excerpt=excerpt,
        collected_by="operator",
        tags=tags,
    )
    return evidence, warnings


def to_evidence_from_caption(caption: CaptionRecord, max_excerpt_chars: int) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(caption.caption_text, max_excerpt_chars)
    evidence = Evidence(
        evidence_type=EvidenceType.TEXT_EXCERPT,
        source_description=f"caption for post {caption.post_reference} (creator {caption.creator_id})",
        content_excerpt=excerpt,
        collected_by="operator",
        tags=["caption", *caption.manual_tags],
    )
    return evidence, warnings


def to_evidence_from_reply_pair(pair: ReplyPair, max_excerpt_chars: int) -> tuple[Evidence, list[str]]:
    combined = f"comment: {pair.audience_comment} | reply: {pair.creator_reply}"
    excerpt, warnings = _capped_excerpt(combined, max_excerpt_chars)
    evidence = Evidence(
        evidence_type=EvidenceType.TEXT_EXCERPT,
        source_description=f"reply pair for post {pair.post_reference} (creator {pair.creator_id})",
        content_excerpt=excerpt,
        collected_by="operator",
        tags=["reply", *pair.manual_tags],
    )
    return evidence, warnings


def to_evidence_from_reel(reel: ReelEvidence, max_excerpt_chars: int) -> tuple[Evidence, list[str]]:
    combined = " | ".join(filter(None, [reel.hook_notes, reel.story_arc_notes]))
    excerpt, warnings = _capped_excerpt(combined, max_excerpt_chars)
    evidence = Evidence(
        evidence_type=EvidenceType.OPERATOR_OBSERVATION,
        source_description=f"reel notes for {reel.source_reference} (creator {reel.creator_id})",
        content_excerpt=excerpt,
        collected_by="operator",
        tags=["reels", *reel.manual_tags],
    )
    return evidence, warnings


def to_evidence_from_highlight(highlight: HighlightItem, max_excerpt_chars: int) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(highlight.manual_summary, max_excerpt_chars)
    tags = ["storytelling"]
    if highlight.story_type:
        tags.append(highlight.story_type)
    evidence = Evidence(
        evidence_type=EvidenceType.OPERATOR_OBSERVATION,
        source_description=f"highlight '{highlight.highlight_name}' item {highlight.item_index} "
        f"(creator {highlight.creator_id})",
        content_excerpt=excerpt,
        collected_by="operator",
        tags=tags,
    )
    return evidence, warnings


def to_evidence_from_relationship_annotation(
    annotation: RelationshipAnnotation, creator_id_value: str, max_excerpt_chars: int
) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(annotation.description, max_excerpt_chars)
    evidence = Evidence(
        evidence_type=EvidenceType.OPERATOR_OBSERVATION,
        source_description=f"relationship annotation ({annotation.role}, basis={annotation.basis}) "
        f"for creator {creator_id_value}",
        content_excerpt=excerpt,
        collected_by="operator",
        tags=["relationship", "human_authenticity", annotation.role],
    )
    return evidence, warnings


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        Path(tmp_name).replace(path)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Workspace ownership
# ---------------------------------------------------------------------------


def _marker_path(intake_dir: Path) -> Path:
    return intake_dir / MARKER_FILENAME


def _verify_marker(intake_dir: Path) -> dict:
    marker_path = _marker_path(intake_dir)
    if not marker_path.is_file():
        raise IntakeWorkspaceOwnershipError(
            f"{intake_dir} is not an initialized intake workspace (missing {MARKER_FILENAME}); run --init first"
        )
    return json.loads(marker_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Workspace init
# ---------------------------------------------------------------------------


def init_intake_workspace(
    username: str,
    platform: str,
    profile_url: str,
    output_dir: str | Path,
    intake_config: im.IntakeConfig,
    *,
    display_name: str | None = None,
    bio: str | None = None,
    category: str | None = None,
    region: str | None = None,
    languages: list[str] | None = None,
    analysis_start_date: str | None = None,
    analysis_end_date: str | None = None,
    notes: str = "",
    metadata: dict | None = None,
    force: bool = False,
) -> CreatorProfile:
    """Creates (or, with force=True, re-initializes the profile/marker
    of) an intake workspace at output_dir. Never deletes existing
    evidence/source data even with force -- force only permits
    re-writing creator_profile.json and the ownership marker."""
    intake_dir = Path(output_dir)
    marker_exists = _marker_path(intake_dir).is_file()

    if intake_dir.exists() and intake_dir.is_dir():
        has_content = any(intake_dir.iterdir())
        if has_content and not marker_exists:
            raise UnsafeIntakeWorkspaceError(
                f"{intake_dir} already exists and is not an owned intake workspace; refusing to touch it"
            )
        if marker_exists and not force:
            raise IntakeWorkspaceOwnershipError(
                f"{intake_dir} is already an initialized intake workspace; use force to re-initialize"
            )

    intake_dir.mkdir(parents=True, exist_ok=True)
    for dirname in intake_config.package_directories:
        (intake_dir / dirname).mkdir(parents=True, exist_ok=True)

    profile = CreatorProfile(
        platform=platform,
        username=username,
        profile_url=profile_url,
        display_name=display_name,
        bio=bio,
        category=category,
        region=region,
        languages=list(languages or []),
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
        notes=notes,
        metadata=dict(metadata or {}),
    )
    _atomic_write_json(
        intake_dir / "creator_profile.json",
        {
            "creator_id": profile.creator_id,
            "platform": profile.platform,
            "username": profile.username,
            "profile_url": profile.profile_url,
            "display_name": profile.display_name,
            "bio": profile.bio,
            "category": profile.category,
            "region": profile.region,
            "languages": profile.languages,
            "analysis_start_date": profile.analysis_start_date,
            "analysis_end_date": profile.analysis_end_date,
            "notes": profile.notes,
            "metadata": profile.metadata,
        },
    )

    if not (intake_dir / "evidence_bundle.json").exists():
        _atomic_write_json(intake_dir / "evidence_bundle.json", [])
    if not (intake_dir / "source_index.json").exists():
        _atomic_write_json(intake_dir / "source_index.json", [])

    if intake_config.template_source_dir:
        template_source = _runtime_root() / intake_config.template_source_dir
        templates_dest = intake_dir / "_templates"
        if template_source.is_dir():
            templates_dest.mkdir(parents=True, exist_ok=True)
            for template_file in template_source.glob("*"):
                if template_file.is_file():
                    shutil.copy2(template_file, templates_dest / template_file.name)

    now = _now_iso()
    _atomic_write_json(
        _marker_path(intake_dir),
        {"creator_id": profile.creator_id, "username": username, "platform": platform, "created_at": now},
    )

    manifest = im.build_intake_manifest(
        creator_id=profile.creator_id,
        username=username,
        platform=platform,
        profile_url=profile_url,
        intake_dir=intake_dir,
        intake_config=intake_config,
        schema_version=intake_config.schema_version,
        created_at=now,
        updated_at=now,
    )
    im.save_intake_manifest(intake_dir, manifest)

    return profile


def _rebuild_manifest(intake_dir: Path, intake_config: im.IntakeConfig) -> im.IntakeManifest:
    marker = _verify_marker(intake_dir)
    existing_created_at = marker.get("created_at", _now_iso())
    try:
        previous = im.load_intake_manifest(intake_dir)
        created_at = previous.created_at or existing_created_at
    except im.IntakeManifestError:
        created_at = existing_created_at
    manifest = im.build_intake_manifest(
        creator_id=marker["creator_id"],
        username=marker.get("username", ""),
        platform=marker.get("platform", ""),
        profile_url=_load_profile(intake_dir).profile_url if (intake_dir / "creator_profile.json").is_file() else "",
        intake_dir=intake_dir,
        intake_config=intake_config,
        schema_version=intake_config.schema_version,
        created_at=created_at,
        updated_at=_now_iso(),
    )
    im.save_intake_manifest(intake_dir, manifest)
    return manifest


def _load_profile(intake_dir: Path) -> CreatorProfile:
    payload = json.loads((intake_dir / "creator_profile.json").read_text(encoding="utf-8"))
    profile = CreatorProfile(
        platform=payload["platform"],
        username=payload["username"],
        profile_url=payload["profile_url"],
        display_name=payload.get("display_name"),
        bio=payload.get("bio"),
        category=payload.get("category"),
        region=payload.get("region"),
        languages=list(payload.get("languages", [])),
        analysis_start_date=payload.get("analysis_start_date"),
        analysis_end_date=payload.get("analysis_end_date"),
        notes=payload.get("notes", ""),
        metadata=dict(payload.get("metadata", {})),
    )
    return profile


# ---------------------------------------------------------------------------
# Local path safety
# ---------------------------------------------------------------------------


def _resolve_local_path(intake_dir: Path, local_path: str) -> Path:
    intake_root = intake_dir.resolve()
    candidate = (intake_dir / local_path).resolve()
    if intake_root not in candidate.parents and candidate != intake_root:
        raise UnsafeLocalPathError(f"local_path {local_path!r} resolves outside the intake workspace")
    return candidate


# ---------------------------------------------------------------------------
# Evidence bundle helpers
# ---------------------------------------------------------------------------


def _evidence_to_dict(evidence: Evidence) -> dict:
    return {
        "evidence_id": evidence.evidence_id,
        "evidence_type": evidence.evidence_type,
        "source_description": evidence.source_description,
        "content_excerpt": evidence.content_excerpt,
        "collected_at": evidence.collected_at,
        "collected_by": evidence.collected_by,
        "tags": list(evidence.tags),
    }


def _append_evidence_bundle(intake_dir: Path, evidence: Evidence) -> None:
    path = intake_dir / "evidence_bundle.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if any(item.get("evidence_id") == evidence.evidence_id for item in existing):
        return
    existing.append(_evidence_to_dict(evidence))
    _atomic_write_json(path, existing)


# ---------------------------------------------------------------------------
# Generic ingestion
# ---------------------------------------------------------------------------


def _ingest(
    intake_dir: Path,
    intake_config: im.IntakeConfig,
    *,
    category_dir: str,
    record_id: str,
    record_payload: dict,
    evidence_items: list[Evidence],
    source_type: str,
    creator_id_value: str,
    original_reference: str = "",
    local_file: str = "",
    published_at: str = "",
    captured_at: str = "",
    notes: str = "",
    force: bool = False,
) -> im.IntakeManifest:
    _verify_marker(intake_dir)

    record_path = intake_dir / category_dir / f"{record_id}.json"
    if record_path.exists() and not force:
        raise IntakeRecordExistsError(f"{record_path} already exists (use force to overwrite)")
    _atomic_write_json(record_path, record_payload)

    for evidence in evidence_items:
        _append_evidence_bundle(intake_dir, evidence)

    entries = [e for e in im.load_source_index(intake_dir) if e.source_id != record_id]
    entry = im.SourceIndexEntry(
        source_id=record_id,
        creator_id=creator_id_value,
        source_type=source_type,
        original_reference=original_reference,
        local_file=local_file,
        published_at=published_at,
        captured_at=captured_at,
        evidence_ids=[e.evidence_id for e in evidence_items],
        notes=notes,
    )
    entries = im.add_source_entry(entries, entry)
    im.save_source_index(intake_dir, entries)

    return _rebuild_manifest(intake_dir, intake_config)


def add_screenshot(
    intake_dir: str | Path, screenshot: ScreenshotEvidence, intake_config: im.IntakeConfig, *, force: bool = False
) -> im.IntakeManifest:
    intake_dir = Path(intake_dir)
    _verify_marker(intake_dir)

    if screenshot.local_path:
        resolved = _resolve_local_path(intake_dir, screenshot.local_path)
        if not resolved.is_file():
            raise MissingLocalFileError(f"screenshot local_path does not exist: {resolved}")

    evidence, _warnings = to_evidence_from_screenshot(screenshot, _max_excerpt_chars())
    evidence_items = [evidence]

    for annotation in screenshot.relationship_annotations:
        rel_evidence, _rel_warnings = to_evidence_from_relationship_annotation(
            annotation, screenshot.creator_id, _max_excerpt_chars()
        )
        evidence_items.append(rel_evidence)
        _atomic_write_json(
            intake_dir / "relationships" / f"{annotation.annotation_id}.json",
            {
                "annotation_id": annotation.annotation_id,
                "role": annotation.role,
                "basis": annotation.basis,
                "confidence": annotation.confidence,
                "description": annotation.description,
                "source_screenshot_id": screenshot.screenshot_id,
                "evidence_id": rel_evidence.evidence_id,
            },
        )

    payload = {
        "screenshot_id": screenshot.screenshot_id,
        "creator_id": screenshot.creator_id,
        "source_type": screenshot.source_type,
        "local_path": screenshot.local_path,
        "original_filename": screenshot.original_filename,
        "captured_at": screenshot.captured_at,
        "source_url": screenshot.source_url,
        "post_date": screenshot.post_date,
        "caption_reference": screenshot.caption_reference,
        "manual_notes": screenshot.manual_notes,
        "visual_annotations": _annotation_to_dict(screenshot.visual_annotations),
        "relationship_annotations": [
            {"role": a.role, "basis": a.basis, "confidence": a.confidence, "description": a.description}
            for a in screenshot.relationship_annotations
        ],
        "metadata": screenshot.metadata,
        "evidence_id": evidence.evidence_id,
    }

    return _ingest(
        intake_dir,
        intake_config,
        category_dir="screenshots",
        record_id=screenshot.screenshot_id,
        record_payload=payload,
        evidence_items=evidence_items,
        source_type=screenshot.source_type,
        creator_id_value=screenshot.creator_id,
        original_reference=screenshot.source_url or "",
        local_file=screenshot.local_path,
        captured_at=screenshot.captured_at or "",
        notes=screenshot.manual_notes,
        force=force,
    )


def _annotation_to_dict(annotation: VisualRealismAnnotation | None) -> dict | None:
    if annotation is None:
        return None
    return {
        "skin_texture_visible": annotation.skin_texture_visible,
        "pores_visible": annotation.pores_visible,
        "blemishes_visible": annotation.blemishes_visible,
        "skin_smoothing": annotation.skin_smoothing,
        "makeup_level": annotation.makeup_level,
        "focus_softness": annotation.focus_softness,
        "motion_blur": annotation.motion_blur,
        "sensor_noise": annotation.sensor_noise,
        "grain": annotation.grain,
        "bloom": annotation.bloom,
        "flash": annotation.flash,
        "white_balance": annotation.white_balance,
        "highlight_clipping": annotation.highlight_clipping,
        "shadow_detail": annotation.shadow_detail,
        "dynamic_range": annotation.dynamic_range,
        "color_cast": annotation.color_cast,
        "resolution_feel": annotation.resolution_feel,
        "ccd_like_aesthetic": annotation.ccd_like_aesthetic,
        "compact_camera_aesthetic": annotation.compact_camera_aesthetic,
        "film_like_aesthetic": annotation.film_like_aesthetic,
        "phone_processing_aesthetic": annotation.phone_processing_aesthetic,
        "editing_strength": annotation.editing_strength,
        "environment_realism": annotation.environment_realism,
        "background_clutter": annotation.background_clutter,
        "ordinary_location": annotation.ordinary_location,
        "luxury_location": annotation.luxury_location,
        "notes": annotation.notes,
    }


def add_caption(
    intake_dir: str | Path, caption: CaptionRecord, intake_config: im.IntakeConfig, *, force: bool = False
) -> im.IntakeManifest:
    intake_dir = Path(intake_dir)
    evidence, _warnings = to_evidence_from_caption(caption, _max_excerpt_chars())
    payload = {
        "caption_id": caption.caption_id,
        "creator_id": caption.creator_id,
        "post_reference": caption.post_reference,
        "published_at": caption.published_at,
        "caption_text": caption.caption_text,
        "hashtags": caption.hashtags,
        "mentions": caption.mentions,
        "language": caption.language,
        "manual_tags": caption.manual_tags,
        "metrics_if_known": caption.metrics_if_known,
        "metadata": caption.metadata,
        "evidence_id": evidence.evidence_id,
    }
    return _ingest(
        intake_dir,
        intake_config,
        category_dir="captions",
        record_id=caption.caption_id,
        record_payload=payload,
        evidence_items=[evidence],
        source_type="caption",
        creator_id_value=caption.creator_id,
        original_reference=caption.post_reference or "",
        published_at=caption.published_at or "",
        force=force,
    )


def add_reply_pair(
    intake_dir: str | Path, pair: ReplyPair, intake_config: im.IntakeConfig, *, force: bool = False
) -> im.IntakeManifest:
    intake_dir = Path(intake_dir)
    evidence, _warnings = to_evidence_from_reply_pair(pair, _max_excerpt_chars())
    payload = {
        "reply_pair_id": pair.reply_pair_id,
        "creator_id": pair.creator_id,
        "post_reference": pair.post_reference,
        "audience_comment": pair.audience_comment,
        "creator_reply": pair.creator_reply,
        "comment_language": pair.comment_language,
        "reply_language": pair.reply_language,
        "comment_timestamp": pair.comment_timestamp,
        "reply_timestamp": pair.reply_timestamp,
        "emoji": pair.emoji,
        "thread_depth": pair.thread_depth,
        "audience_username_optional": pair.audience_username_optional,
        "relationship_context": pair.relationship_context,
        "manual_tags": pair.manual_tags,
        "metadata": pair.metadata,
        "evidence_id": evidence.evidence_id,
    }
    return _ingest(
        intake_dir,
        intake_config,
        category_dir="replies",
        record_id=pair.reply_pair_id,
        record_payload=payload,
        evidence_items=[evidence],
        source_type="reply_pair",
        creator_id_value=pair.creator_id,
        original_reference=pair.post_reference or "",
        published_at=pair.comment_timestamp or "",
        force=force,
    )


def add_reel_note(
    intake_dir: str | Path, reel: ReelEvidence, intake_config: im.IntakeConfig, *, force: bool = False
) -> im.IntakeManifest:
    intake_dir = Path(intake_dir)
    evidence, _warnings = to_evidence_from_reel(reel, _max_excerpt_chars())
    payload = {
        "reel_id": reel.reel_id,
        "creator_id": reel.creator_id,
        "source_reference": reel.source_reference,
        "published_at": reel.published_at,
        "duration_seconds_if_known": reel.duration_seconds_if_known,
        "views_if_known": reel.views_if_known,
        "likes_if_known": reel.likes_if_known,
        "comments_if_known": reel.comments_if_known,
        "shot_notes": [
            {
                "shot_index": s.shot_index,
                "start_seconds_optional": s.start_seconds_optional,
                "end_seconds_optional": s.end_seconds_optional,
                "subject_action": s.subject_action,
                "camera_angle": s.camera_angle,
                "camera_motion": s.camera_motion,
                "environment": s.environment,
                "emotion": s.emotion,
                "interaction": s.interaction,
                "text_overlay": s.text_overlay,
                "notes": s.notes,
            }
            for s in reel.shot_notes
        ],
        "hook_notes": reel.hook_notes,
        "subtitle_notes": reel.subtitle_notes,
        "music_notes": reel.music_notes,
        "camera_notes": reel.camera_notes,
        "story_arc_notes": reel.story_arc_notes,
        "manual_tags": reel.manual_tags,
        "metadata": reel.metadata,
        "evidence_id": evidence.evidence_id,
    }
    return _ingest(
        intake_dir,
        intake_config,
        category_dir="reels",
        record_id=reel.reel_id,
        record_payload=payload,
        evidence_items=[evidence],
        source_type="reel_note",
        creator_id_value=reel.creator_id,
        original_reference=reel.source_reference or "",
        published_at=reel.published_at or "",
        force=force,
    )


def add_highlight_note(
    intake_dir: str | Path, highlight: HighlightItem, intake_config: im.IntakeConfig, *, force: bool = False
) -> im.IntakeManifest:
    intake_dir = Path(intake_dir)
    evidence, _warnings = to_evidence_from_highlight(highlight, _max_excerpt_chars())
    payload = {
        "highlight_id": highlight.highlight_id,
        "creator_id": highlight.creator_id,
        "highlight_name": highlight.highlight_name,
        "item_index": highlight.item_index,
        "source_reference": highlight.source_reference,
        "content_type": highlight.content_type,
        "manual_summary": highlight.manual_summary,
        "people_context": highlight.people_context,
        "location_context": highlight.location_context,
        "story_type": highlight.story_type,
        "memory_type": highlight.memory_type,
        "manual_tags": [highlight.story_type] if highlight.story_type else [],
        "metadata": highlight.metadata,
        "evidence_id": evidence.evidence_id,
    }
    return _ingest(
        intake_dir,
        intake_config,
        category_dir="highlights",
        record_id=highlight.highlight_id,
        record_payload=payload,
        evidence_items=[evidence],
        source_type="highlight_note",
        creator_id_value=highlight.creator_id,
        original_reference=highlight.source_reference or "",
        force=force,
    )


def add_manual_note(
    intake_dir: str | Path,
    *,
    creator_id_value: str,
    source_description: str,
    content_excerpt: str = "",
    collected_by: str = "operator",
    collected_at: str = "",
    tags: list[str] | None = None,
    intake_config: im.IntakeConfig,
    force: bool = False,
) -> im.IntakeManifest:
    intake_dir = Path(intake_dir)
    excerpt, _warnings = _capped_excerpt(content_excerpt, _max_excerpt_chars())
    try:
        evidence = Evidence(
            evidence_type=EvidenceType.MANUAL_NOTE,
            source_description=source_description,
            content_excerpt=excerpt,
            collected_by=collected_by,
            collected_at=collected_at,
            tags=list(tags or []),
        )
    except EvidenceError as exc:
        raise IntakeRequestError(str(exc)) from exc

    payload = {
        "creator_id": creator_id_value,
        "source_description": source_description,
        "content_excerpt": excerpt,
        "collected_by": collected_by,
        "collected_at": collected_at,
        "tags": list(tags or []),
        "evidence_id": evidence.evidence_id,
    }
    return _ingest(
        intake_dir,
        intake_config,
        category_dir="notes",
        record_id=evidence.evidence_id,
        record_payload=payload,
        evidence_items=[evidence],
        source_type="manual_note",
        creator_id_value=creator_id_value,
        notes=source_description,
        force=force,
    )


def _max_excerpt_chars() -> int:
    """Reuses the CreatorDNA-side excerpt cap (framework.yaml) so
    intake evidence and analyzer-facing evidence share one policy.
    Falls back to a conservative default if the framework config
    cannot be loaded (e.g. in isolated tests that don't need it)."""
    try:
        from .config import load_framework_config

        return load_framework_config().caption_excerpt_max_chars
    except Exception:  # pragma: no cover - defensive fallback only
        return 280


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="creator_intelligence.intake", description="Creator Evidence Intake workspace tool"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--init", action="store_true")
    action.add_argument("--add-screenshot", type=Path)
    action.add_argument("--add-caption", type=Path)
    action.add_argument("--add-reply-pair", type=Path)
    action.add_argument("--add-reel-note", type=Path)
    action.add_argument("--add-highlight-note", type=Path)
    action.add_argument("--add-manual-note", type=Path)

    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--username")
    parser.add_argument("--platform")
    parser.add_argument("--profile-url")
    parser.add_argument("--display-name")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.init and (not args.username or not args.platform or not args.profile_url):
        parser.error("--init requires --username, --platform, and --profile-url")
    return args


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        raise IntakeRequestError(f"File not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IntakeRequestError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise IntakeRequestError(f"{path} must contain a JSON object")
    return _strip_comment_keys(raw)


def _run_init(args: argparse.Namespace, intake_config: im.IntakeConfig) -> dict:
    profile = init_intake_workspace(
        username=args.username,
        platform=args.platform,
        profile_url=args.profile_url,
        output_dir=args.output,
        intake_config=intake_config,
        display_name=args.display_name,
        force=args.force,
    )
    return {"creator_id": profile.creator_id, "username": profile.username, "output": str(args.output)}


def _run_add_screenshot(args: argparse.Namespace, intake_config: im.IntakeConfig) -> dict:
    raw = _load_json_file(args.add_screenshot)
    visual_raw = raw.pop("visual_annotations", None)
    visual = VisualRealismAnnotation(**_strip_comment_keys(visual_raw)) if visual_raw else None
    relationship_raw = raw.pop("relationship_annotations", []) or []
    relationships = [RelationshipAnnotation(**_strip_comment_keys(r)) for r in relationship_raw]
    creator_id_value = raw.pop("creator_id", None) or _creator_id_from_workspace(args.output)
    screenshot = ScreenshotEvidence(
        creator_id=creator_id_value,
        visual_annotations=visual,
        relationship_annotations=relationships,
        **{k: v for k, v in raw.items() if k in _screenshot_fields()},
    )
    manifest = add_screenshot(args.output, screenshot, intake_config, force=args.force)
    return {"screenshot_id": screenshot.screenshot_id, "manifest_id": manifest.manifest_id}


def _screenshot_fields() -> set[str]:
    return {
        "source_type",
        "local_path",
        "original_filename",
        "captured_at",
        "source_url",
        "post_date",
        "caption_reference",
        "manual_notes",
        "metadata",
    }


def _creator_id_from_workspace(output_dir: Path) -> str:
    marker = _verify_marker(Path(output_dir))
    return marker["creator_id"]


def _run_add_caption(args: argparse.Namespace, intake_config: im.IntakeConfig) -> dict:
    raw = _load_json_file(args.add_caption)
    creator_id_value = raw.pop("creator_id", None) or _creator_id_from_workspace(args.output)
    caption = CaptionRecord(creator_id=creator_id_value, **raw)
    manifest = add_caption(args.output, caption, intake_config, force=args.force)
    return {"caption_id": caption.caption_id, "manifest_id": manifest.manifest_id}


def _run_add_reply_pair(args: argparse.Namespace, intake_config: im.IntakeConfig) -> dict:
    raw = _load_json_file(args.add_reply_pair)
    creator_id_value = raw.pop("creator_id", None) or _creator_id_from_workspace(args.output)
    pair = ReplyPair(creator_id=creator_id_value, **raw)
    manifest = add_reply_pair(args.output, pair, intake_config, force=args.force)
    return {"reply_pair_id": pair.reply_pair_id, "manifest_id": manifest.manifest_id}


def _run_add_reel_note(args: argparse.Namespace, intake_config: im.IntakeConfig) -> dict:
    raw = _load_json_file(args.add_reel_note)
    creator_id_value = raw.pop("creator_id", None) or _creator_id_from_workspace(args.output)
    shot_notes_raw = raw.pop("shot_notes", []) or []
    shot_notes = [ShotNote(**_strip_comment_keys(s)) for s in shot_notes_raw]
    reel = ReelEvidence(creator_id=creator_id_value, shot_notes=shot_notes, **raw)
    manifest = add_reel_note(args.output, reel, intake_config, force=args.force)
    return {"reel_id": reel.reel_id, "manifest_id": manifest.manifest_id}


def _run_add_highlight_note(args: argparse.Namespace, intake_config: im.IntakeConfig) -> dict:
    raw = _load_json_file(args.add_highlight_note)
    creator_id_value = raw.pop("creator_id", None) or _creator_id_from_workspace(args.output)
    highlight = HighlightItem(creator_id=creator_id_value, **raw)
    manifest = add_highlight_note(args.output, highlight, intake_config, force=args.force)
    return {"highlight_id": highlight.highlight_id, "manifest_id": manifest.manifest_id}


def _run_add_manual_note(args: argparse.Namespace, intake_config: im.IntakeConfig) -> dict:
    raw = _load_json_file(args.add_manual_note)
    creator_id_value = raw.pop("creator_id", None) or _creator_id_from_workspace(args.output)
    manifest = add_manual_note(
        args.output,
        creator_id_value=creator_id_value,
        intake_config=intake_config,
        force=args.force,
        **raw,
    )
    return {"manifest_id": manifest.manifest_id}


def _print_result(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"{PREFIX} intake result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        intake_config = im.load_intake_config(args.config)
        if args.init:
            result = _run_init(args, intake_config)
        elif args.add_screenshot:
            result = _run_add_screenshot(args, intake_config)
        elif args.add_caption:
            result = _run_add_caption(args, intake_config)
        elif args.add_reply_pair:
            result = _run_add_reply_pair(args, intake_config)
        elif args.add_reel_note:
            result = _run_add_reel_note(args, intake_config)
        elif args.add_highlight_note:
            result = _run_add_highlight_note(args, intake_config)
        elif args.add_manual_note:
            result = _run_add_manual_note(args, intake_config)
        else:  # pragma: no cover - argparse enforces one action
            raise IntakeRequestError("no action given")
    except (IntakeError, im.IntakeManifestError, EvidenceError) as exc:
        print(f"{PREFIX} Error: {exc}", file=sys.stderr)
        return 1

    _print_result(result, args.json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
