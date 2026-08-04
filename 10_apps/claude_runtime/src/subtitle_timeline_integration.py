from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import subtitle_engine, timeline_engine

# Phase 11D.1 — Subtitle-Timeline Integration. A dedicated adapter that
# attaches an already-validated subtitle_engine.SubtitleDocument onto an
# already-validated timeline_engine.Timeline as one new subtitle
# TimelineTrack, using only both modules' existing public APIs. Never
# renders, never burns subtitles into video, never calls ffmpeg or
# ffprobe, never transcribes speech, never publishes or uploads, never
# modifies media files, never downloads fonts, and never writes the
# source timeline.json/subtitles.json files it reads.

DEFAULT_INTEGRATION_CONFIG_RELATIVE_PATH = Path("config") / "video" / "subtitle_timeline_integration.yaml"


def _runtime_root() -> Path:
    """
    subtitle_timeline_integration.py location:
    10_apps/claude_runtime/src/subtitle_timeline_integration.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SubtitleTimelineIntegrationError(RuntimeError):
    """Base error for the Phase 11D.1 Subtitle-Timeline Integration adapter."""


class TimelineLoadError(SubtitleTimelineIntegrationError):
    """Raised when --timeline cannot be loaded via timeline_engine.load_timeline()."""


class SubtitleLoadError(SubtitleTimelineIntegrationError):
    """Raised when --subtitles cannot be loaded via subtitle_engine.load_subtitle_document()."""


class SourceValidationError(SubtitleTimelineIntegrationError):
    """Raised when the loaded Timeline or SubtitleDocument fails its own
    engine's validate_*() check, or has an empty language."""


class TimelineIDMismatchError(SubtitleTimelineIntegrationError):
    """Raised when the subtitle document's timeline_id does not match the
    target Timeline's timeline_id and config.timeline_id_must_match is true."""


class SubtitleDurationExceedsTimelineError(SubtitleTimelineIntegrationError):
    """Raised when the subtitle document (or any individual cue) extends
    beyond the target Timeline's duration."""


class SubtitleTrackExistsError(SubtitleTimelineIntegrationError):
    """Raised when the target track_id already exists and replacement is
    not permitted, or an enabled subtitle track already exists and
    multiple subtitle tracks are not allowed."""


class SubtitleTrackMappingError(SubtitleTimelineIntegrationError):
    """Raised for a structural mapping failure: no valid track-order slot,
    a non-subtitle track occupying the target track_id, an empty subtitle
    document, or a cue with no planned position when one is required."""


class SubtitleStyleReferenceError(SubtitleTimelineIntegrationError):
    """Raised when a cue references a style_id not present in the
    subtitle document's styles."""


class IntegratedTimelineValidationError(SubtitleTimelineIntegrationError):
    """Raised when the Timeline produced by attaching subtitles fails
    timeline_engine.validate_timeline() or this module's own adapter-level
    subtitle-track checks."""


class IntegrationOutputExistsError(SubtitleTimelineIntegrationError):
    """Raised when --output already exists and --force was not given."""


class UnsafeIntegrationOutputError(SubtitleTimelineIntegrationError):
    """Raised when --output is a directory, or resolves to the same path
    as --timeline or --subtitles."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SubtitleTimelineIntegrationConfig:
    schema_version: str = "1.0"
    default_track_id: str = "subtitle-main"
    default_track_order_policy: str = "after_audio_before_overlay"
    allow_multiple_subtitle_tracks: bool = True
    fail_on_existing_target_track: bool = True
    timeline_id_must_match: bool = True
    duration_tolerance_seconds: float = 0.001
    preserve_base_timeline_id: bool = True
    generate_integrated_timeline_id: bool = True

    clip_type: str = "overlay"
    preserve_style_metadata: bool = True
    preserve_unknown_metadata: bool = True
    require_safe_area_position: bool = True

    overwrite_requires_force: bool = True
    atomic_write: bool = True


def default_integration_config_path() -> Path:
    return _runtime_root() / DEFAULT_INTEGRATION_CONFIG_RELATIVE_PATH


def load_integration_config(config_path: str | Path | None = None) -> SubtitleTimelineIntegrationConfig:
    """Load config/video/subtitle_timeline_integration.yaml (or an
    alternate path) into a SubtitleTimelineIntegrationConfig. Raises
    SubtitleTimelineIntegrationError if the file is missing or invalid."""
    path = Path(config_path) if config_path else default_integration_config_path()

    if not path.exists():
        raise SubtitleTimelineIntegrationError(f"Subtitle-timeline integration config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SubtitleTimelineIntegrationError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise SubtitleTimelineIntegrationError(f"Subtitle-timeline integration config is empty or invalid: {path}")

    integration_section = raw.get("integration") or {}
    mapping_section = raw.get("mapping") or {}
    output_section = raw.get("output") or {}

    return SubtitleTimelineIntegrationConfig(
        schema_version=str(integration_section.get("schema_version", "1.0")),
        default_track_id=str(integration_section.get("default_track_id", "subtitle-main")),
        default_track_order_policy=str(
            integration_section.get("default_track_order_policy", "after_audio_before_overlay")
        ),
        allow_multiple_subtitle_tracks=bool(integration_section.get("allow_multiple_subtitle_tracks", True)),
        fail_on_existing_target_track=bool(integration_section.get("fail_on_existing_target_track", True)),
        timeline_id_must_match=bool(integration_section.get("timeline_id_must_match", True)),
        duration_tolerance_seconds=float(integration_section.get("duration_tolerance_seconds", 0.001)),
        preserve_base_timeline_id=bool(integration_section.get("preserve_base_timeline_id", True)),
        generate_integrated_timeline_id=bool(integration_section.get("generate_integrated_timeline_id", True)),
        clip_type=str(mapping_section.get("clip_type", "overlay")),
        preserve_style_metadata=bool(mapping_section.get("preserve_style_metadata", True)),
        preserve_unknown_metadata=bool(mapping_section.get("preserve_unknown_metadata", True)),
        require_safe_area_position=bool(mapping_section.get("require_safe_area_position", True)),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        atomic_write=bool(output_section.get("atomic_write", True)),
    )


# ---------------------------------------------------------------------------
# Typed models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SubtitleCueMapping:
    cue_id: str = ""
    clip_id: str = ""
    text: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    duration_seconds: float = 0.0
    style_id: str = "default"
    speaker: str | None = None
    language: str = "en"
    enabled: bool = True


@dataclass(slots=True)
class SubtitleTrackMapping:
    track_id: str = ""
    source_document_id: str = ""
    source_language: str = "en"
    source_schema_version: str = "1.0"
    source_timeline_id: str | None = None
    default_style_id: str = "default"
    cue_count: int = 0
    cues: list[SubtitleCueMapping] = field(default_factory=list)


@dataclass(slots=True)
class IntegrationWarning:
    code: str = ""
    message: str = ""


@dataclass(slots=True)
class SubtitleTimelineIntegrationResult:
    integration_id: str = ""
    base_timeline_id: str = ""
    integrated_timeline_id: str = ""
    subtitle_document_id: str = ""
    track_id: str = ""
    track_enabled: bool = True
    cue_count: int = 0
    language: str = "en"
    replaced_existing_track: bool = False
    validation_passed: bool = False
    warnings: list[str] = field(default_factory=list)
    output_path: str | None = None
    source_timeline_unchanged: bool = True
    source_subtitle_unchanged: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Compatibility checks (requirement 8) — never duplicates timing/style
# checks already owned by timeline_engine.validate_timeline()/
# subtitle_engine.validate_subtitle_document(); only adds genuinely new
# cross-document concerns.
# ---------------------------------------------------------------------------


def validate_subtitle_timeline_compatibility(
    timeline: timeline_engine.Timeline,
    subtitle_document: subtitle_engine.SubtitleDocument,
    config: SubtitleTimelineIntegrationConfig,
) -> list[str]:
    """Read-only, non-raising compatibility check — used by --validate-only
    and directly by callers that want a soft report instead of an
    exception."""
    errors: list[str] = []
    tolerance = config.duration_tolerance_seconds

    if (
        config.timeline_id_must_match
        and subtitle_document.timeline_id is not None
        and subtitle_document.timeline_id != timeline.timeline_id
    ):
        errors.append(
            f"subtitle document timeline_id {subtitle_document.timeline_id!r} does not match "
            f"target timeline_id {timeline.timeline_id!r}"
        )

    if subtitle_document.duration_seconds > timeline.duration_seconds + tolerance:
        errors.append(
            f"subtitle document duration ({subtitle_document.duration_seconds:.3f}s) exceeds "
            f"timeline duration ({timeline.duration_seconds:.3f}s)"
        )

    for track in subtitle_document.tracks:
        for cue in track.cues:
            if cue.end_seconds > timeline.duration_seconds + tolerance:
                errors.append(
                    f"cue {cue.cue_id} end_seconds ({cue.end_seconds:.3f}s) exceeds timeline "
                    f"duration ({timeline.duration_seconds:.3f}s)"
                )

    if not subtitle_document.language:
        errors.append("subtitle document language is empty")

    return errors


def _enforce_compatibility(
    timeline: timeline_engine.Timeline,
    subtitle_document: subtitle_engine.SubtitleDocument,
    config: SubtitleTimelineIntegrationConfig,
) -> None:
    """Raising counterpart of validate_subtitle_timeline_compatibility(),
    used internally by run_integration() so each failure mode surfaces
    its own specific exception type."""
    tolerance = config.duration_tolerance_seconds

    if (
        config.timeline_id_must_match
        and subtitle_document.timeline_id is not None
        and subtitle_document.timeline_id != timeline.timeline_id
    ):
        raise TimelineIDMismatchError(
            f"subtitle document timeline_id {subtitle_document.timeline_id!r} does not match "
            f"target timeline_id {timeline.timeline_id!r}"
        )

    if subtitle_document.duration_seconds > timeline.duration_seconds + tolerance:
        raise SubtitleDurationExceedsTimelineError(
            f"subtitle document duration ({subtitle_document.duration_seconds:.3f}s) exceeds "
            f"timeline duration ({timeline.duration_seconds:.3f}s)"
        )

    for track in subtitle_document.tracks:
        for cue in track.cues:
            if cue.end_seconds > timeline.duration_seconds + tolerance:
                raise SubtitleDurationExceedsTimelineError(
                    f"cue {cue.cue_id} end_seconds ({cue.end_seconds:.3f}s) exceeds timeline "
                    f"duration ({timeline.duration_seconds:.3f}s)"
                )

    if not subtitle_document.language:
        raise SourceValidationError("subtitle document language is empty")


# ---------------------------------------------------------------------------
# Track ordering
# ---------------------------------------------------------------------------


def _next_subtitle_track_order(timeline: timeline_engine.Timeline) -> int:
    video_audio_orders = [
        track.order
        for track in timeline.tracks
        if track.track_type in (timeline_engine.TrackType.VIDEO, timeline_engine.TrackType.AUDIO)
    ]
    overlay_metadata_orders = [
        track.order
        for track in timeline.tracks
        if track.track_type in (timeline_engine.TrackType.OVERLAY, timeline_engine.TrackType.METADATA)
    ]
    used_orders = {track.order for track in timeline.tracks}

    candidate = max(video_audio_orders, default=-1) + 1
    while candidate in used_orders:
        candidate += 1

    if overlay_metadata_orders and candidate >= min(overlay_metadata_orders):
        raise SubtitleTrackMappingError(
            "Cannot place subtitle track after video/audio tracks but before existing "
            "overlay/metadata tracks: no order slot available"
        )
    return candidate


# ---------------------------------------------------------------------------
# Cue / track mapping (requirements 5-7)
# ---------------------------------------------------------------------------


def _style_snapshot(style: subtitle_engine.SubtitleStyle) -> dict[str, Any]:
    return {
        "style_id": style.style_id,
        "font_family": style.font_family,
        "font_size": style.font_size,
        "font_weight": style.font_weight,
        "italic": style.italic,
        "underline": style.underline,
        "primary_color": style.primary_color,
        "outline_color": style.outline_color,
        "background_color": style.background_color,
        "outline_width": style.outline_width,
        "shadow_depth": style.shadow_depth,
        "margin_left": style.margin_left,
        "margin_right": style.margin_right,
        "margin_vertical": style.margin_vertical,
        "alignment": style.alignment,
    }


def map_subtitle_cue(
    cue: subtitle_engine.SubtitleCue,
    subtitle_document: subtitle_engine.SubtitleDocument,
    subtitle_track: subtitle_engine.SubtitleTrack,
    config: SubtitleTimelineIntegrationConfig,
    *,
    track_id: str,
    enabled: bool,
) -> tuple[timeline_engine.TimelineClip, SubtitleCueMapping]:
    """
    Maps one SubtitleCue onto Timeline Engine's existing, reserved
    OverlayClip shape. No media source file is required or fabricated —
    source_path is always "" (schema-legal for OverlayClip). Style is
    never flattened to plain text: the full style snapshot lives in
    clip.metadata alongside position/alignment/speaker/line_break_mode/
    language/source references/the cue's own metadata.
    """
    style = subtitle_document.styles.get(cue.style_id)
    if style is None:
        raise SubtitleStyleReferenceError(f"Cue {cue.cue_id} references unresolved style_id {cue.style_id!r}")

    if cue.position is None and config.require_safe_area_position:
        raise SubtitleTrackMappingError(
            f"Cue {cue.cue_id} has no planned position and require_safe_area_position is enabled"
        )

    clip_metadata: dict[str, Any] = {
        "source_cue_id": cue.cue_id,
        "source_document_id": subtitle_document.document_id,
        "style_id": cue.style_id,
        "alignment": cue.alignment,
        "speaker": cue.speaker,
        "line_break_mode": cue.line_break_mode,
        "language": subtitle_track.language,
    }

    if config.preserve_style_metadata:
        clip_metadata["style"] = _style_snapshot(style)

    if cue.position is not None:
        clip_metadata["position"] = {
            "x": cue.position.x,
            "y": cue.position.y,
            "alignment": cue.position.alignment,
            "safe_area_enabled": cue.position.safe_area_enabled,
        }
    else:
        clip_metadata["position"] = None

    if config.preserve_unknown_metadata:
        clip_metadata["cue_metadata"] = dict(cue.metadata)

    clip_id = f"{track_id}_{cue.cue_id}"

    clip = timeline_engine.OverlayClip(
        clip_id=clip_id,
        track_id=track_id,
        source_path="",
        start=cue.start_seconds,
        end=cue.end_seconds,
        duration_seconds=cue.duration_seconds,
        source_in=0.0,
        source_out=cue.duration_seconds,
        enabled=enabled,
        content=cue.text,
        overlay_type="subtitle",
        metadata=clip_metadata,
    )

    cue_mapping = SubtitleCueMapping(
        cue_id=cue.cue_id,
        clip_id=clip_id,
        text=cue.text,
        start_seconds=cue.start_seconds,
        end_seconds=cue.end_seconds,
        duration_seconds=cue.duration_seconds,
        style_id=cue.style_id,
        speaker=cue.speaker,
        language=subtitle_track.language,
        enabled=enabled,
    )
    return clip, cue_mapping


def map_subtitle_track(
    subtitle_track: subtitle_engine.SubtitleTrack,
    subtitle_document: subtitle_engine.SubtitleDocument,
    config: SubtitleTimelineIntegrationConfig,
    *,
    track_id: str,
    enabled: bool,
    order: int,
) -> tuple[timeline_engine.TimelineTrack, SubtitleTrackMapping]:
    """Maps one SubtitleTrack (and all its cues) onto a new
    timeline_engine.TimelineTrack(track_type=SUBTITLE). TimelineTrack has
    no muted/locked fields -- both live in track.metadata rather than
    requiring a second Timeline Engine schema change."""
    clips: list[timeline_engine.TimelineClip] = []
    cue_mappings: list[SubtitleCueMapping] = []

    for cue in subtitle_track.cues:
        clip, cue_mapping = map_subtitle_cue(
            cue, subtitle_document, subtitle_track, config, track_id=track_id, enabled=enabled
        )
        clips.append(clip)
        cue_mappings.append(cue_mapping)

    track_metadata: dict[str, Any] = {
        "source_document_id": subtitle_document.document_id,
        "source_language": subtitle_track.language,
        "source_schema_version": subtitle_document.schema_version,
        "source_timeline_id": subtitle_document.timeline_id,
        "default_style_id": subtitle_track.default_style_id,
        "cue_count": len(clips),
        "integration_version": config.schema_version,
        "muted": False,
        "locked": False,
    }
    if config.preserve_unknown_metadata:
        track_metadata["source_track_metadata"] = dict(subtitle_track.metadata)

    new_track = timeline_engine.TimelineTrack(
        track_id=track_id,
        track_type=timeline_engine.TrackType.SUBTITLE,
        order=order,
        enabled=enabled,
        clips=clips,
        metadata=track_metadata,
    )

    mapping = SubtitleTrackMapping(
        track_id=track_id,
        source_document_id=subtitle_document.document_id,
        source_language=subtitle_track.language,
        source_schema_version=subtitle_document.schema_version,
        source_timeline_id=subtitle_document.timeline_id,
        default_style_id=subtitle_track.default_style_id,
        cue_count=len(clips),
        cues=cue_mappings,
    )
    return new_track, mapping


# ---------------------------------------------------------------------------
# Existing subtitle track handling (requirement 9)
# ---------------------------------------------------------------------------


def remove_existing_subtitle_track(
    timeline: timeline_engine.Timeline, track_id: str
) -> timeline_engine.TimelineTrack | None:
    """Removes and returns the subtitle track with the exact given
    track_id, if present. Never touches any other track."""
    for index, track in enumerate(timeline.tracks):
        if track.track_id == track_id and track.track_type == timeline_engine.TrackType.SUBTITLE:
            return timeline.tracks.pop(index)
    return None


def replace_subtitle_track(
    timeline: timeline_engine.Timeline, new_track: timeline_engine.TimelineTrack, *, track_id: str
) -> timeline_engine.TimelineTrack | None:
    """Replaces only the exact target track_id; every other track
    (video, audio, unrelated subtitle tracks) is left untouched."""
    removed = remove_existing_subtitle_track(timeline, track_id)
    timeline.tracks.append(new_track)
    return removed


# ---------------------------------------------------------------------------
# Attachment (requirement 4, 9) — deep copy, never mutates the caller's
# Timeline object in place.
# ---------------------------------------------------------------------------


def attach_subtitles_to_timeline(
    timeline: timeline_engine.Timeline,
    subtitle_document: subtitle_engine.SubtitleDocument,
    config: SubtitleTimelineIntegrationConfig,
    *,
    track_id: str | None = None,
    replace_existing: bool = False,
    disable_track: bool = False,
) -> tuple[timeline_engine.Timeline, SubtitleTrackMapping, list[IntegrationWarning]]:
    resolved_track_id = track_id or config.default_track_id
    result_timeline = copy.deepcopy(timeline)
    warnings: list[IntegrationWarning] = []

    conflicting_non_subtitle = next(
        (
            t
            for t in result_timeline.tracks
            if t.track_id == resolved_track_id and t.track_type != timeline_engine.TrackType.SUBTITLE
        ),
        None,
    )
    if conflicting_non_subtitle is not None:
        raise SubtitleTrackMappingError(
            f"track_id {resolved_track_id!r} is already used by a non-subtitle track "
            f"({conflicting_non_subtitle.track_type!r})"
        )

    existing = next(
        (
            t
            for t in result_timeline.tracks
            if t.track_id == resolved_track_id and t.track_type == timeline_engine.TrackType.SUBTITLE
        ),
        None,
    )
    effective_replace_existing = replace_existing or not config.fail_on_existing_target_track
    if existing is not None and not effective_replace_existing:
        raise SubtitleTrackExistsError(
            f"Subtitle track {resolved_track_id!r} already exists; pass replace_existing=True "
            "(or --replace-existing) to replace it."
        )

    other_enabled_subtitle_tracks = [
        t
        for t in result_timeline.tracks
        if t.track_type == timeline_engine.TrackType.SUBTITLE and t.track_id != resolved_track_id and t.enabled
    ]
    if other_enabled_subtitle_tracks and not config.allow_multiple_subtitle_tracks:
        raise SubtitleTrackExistsError(
            f"An enabled subtitle track already exists ({other_enabled_subtitle_tracks[0].track_id!r}) "
            "and allow_multiple_subtitle_tracks is disabled"
        )

    if subtitle_document.timeline_id is not None and subtitle_document.timeline_id != result_timeline.timeline_id:
        warnings.append(
            IntegrationWarning(
                code="timeline_id_mismatch",
                message=(
                    f"subtitle document timeline_id {subtitle_document.timeline_id!r} does not "
                    f"match target timeline_id {result_timeline.timeline_id!r}"
                ),
            )
        )

    if not subtitle_document.tracks or not subtitle_document.tracks[0].cues:
        raise SubtitleTrackMappingError("Subtitle document has no cues to attach")

    source_subtitle_track = subtitle_document.tracks[0]
    new_track, mapping = map_subtitle_track(
        source_subtitle_track,
        subtitle_document,
        config,
        track_id=resolved_track_id,
        enabled=not disable_track,
        order=_next_subtitle_track_order(result_timeline),
    )

    replaced = None
    if existing is not None:
        replaced = remove_existing_subtitle_track(result_timeline, resolved_track_id)
    result_timeline.tracks.append(new_track)

    if replaced is not None:
        warnings.append(
            IntegrationWarning(
                code="replaced_existing_track",
                message=f"Replaced existing subtitle track {resolved_track_id!r} ({len(replaced.clips)} cue(s))",
            )
        )

    return result_timeline, mapping, warnings


# ---------------------------------------------------------------------------
# Deterministic identity (requirement 10-11)
# ---------------------------------------------------------------------------


def _compute_integration_id(
    *,
    base_timeline_id: str,
    subtitle_document_id: str,
    track_id: str,
    replace_existing: bool,
    disable_track: bool,
    schema_version: str,
) -> str:
    payload = {
        "base_timeline_id": base_timeline_id,
        "subtitle_document_id": subtitle_document_id,
        "track_id": track_id,
        "replace_existing": replace_existing,
        "disable_track": disable_track,
        "schema_version": schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _compute_integrated_timeline_id(base_timeline_id: str, integration_id: str) -> str:
    payload = {"base_timeline_id": base_timeline_id, "integration_id": integration_id}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Post-integration validation (requirement 12)
# ---------------------------------------------------------------------------


def _validate_integrated_subtitle_tracks(
    timeline: timeline_engine.Timeline, config: SubtitleTimelineIntegrationConfig
) -> list[str]:
    """
    Adapter-level checks on top of timeline_engine.validate_timeline()'s
    own generic coverage (duration math, unique IDs, JSON-serializable
    metadata, per-track overlap policy already apply to every track type
    including subtitle). Adds the two things no generic Timeline check
    could know: mapped clip count matches the recorded cue_count, and
    every mapped clip's metadata really is JSON-serializable.
    """
    errors: list[str] = []
    for track in timeline.tracks:
        if track.track_type != timeline_engine.TrackType.SUBTITLE:
            continue

        recorded_cue_count = track.metadata.get("cue_count")
        if recorded_cue_count is not None and len(track.clips) != recorded_cue_count:
            errors.append(
                f"subtitle track {track.track_id}: clip count ({len(track.clips)}) does not match "
                f"recorded cue_count ({recorded_cue_count})"
            )

        for clip in track.clips:
            if clip.track_id != track.track_id:
                errors.append(
                    f"subtitle clip {clip.clip_id} references track_id {clip.track_id!r} but is "
                    f"stored under track {track.track_id!r}"
                )
            try:
                json.dumps(clip.metadata)
            except TypeError:
                errors.append(f"subtitle clip {clip.clip_id} metadata is not JSON-serializable")

    return errors


# ---------------------------------------------------------------------------
# Source loading (requirement 3) — no duplicated JSON parsing.
# ---------------------------------------------------------------------------


def _load_sources(
    timeline_path: Path, subtitle_path: Path
) -> tuple[timeline_engine.Timeline, subtitle_engine.SubtitleDocument]:
    try:
        timeline = timeline_engine.load_timeline(timeline_path)
    except timeline_engine.TimelineEngineError as exc:
        raise TimelineLoadError(f"Failed to load timeline {timeline_path}: {exc}") from exc

    try:
        subtitle_document = subtitle_engine.load_subtitle_document(subtitle_path)
    except subtitle_engine.SubtitleEngineError as exc:
        raise SubtitleLoadError(f"Failed to load subtitle document {subtitle_path}: {exc}") from exc

    return timeline, subtitle_document


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_integration(
    timeline_path: str | Path,
    subtitle_path: str | Path,
    config: SubtitleTimelineIntegrationConfig,
    *,
    output_path: str | Path | None = None,
    track_id: str | None = None,
    replace_existing: bool = False,
    disable_track: bool = False,
    force: bool = False,
) -> tuple[timeline_engine.Timeline, SubtitleTimelineIntegrationResult]:
    timeline_path = Path(timeline_path)
    subtitle_path = Path(subtitle_path)
    resolved_track_id = track_id or config.default_track_id

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.resolve() == timeline_path.resolve():
            raise UnsafeIntegrationOutputError("--output must not be the same path as --timeline")
        if output_path.resolve() == subtitle_path.resolve():
            raise UnsafeIntegrationOutputError("--output must not be the same path as --subtitles")

    timeline_before_hash = _hash_file(timeline_path)
    subtitle_before_hash = _hash_file(subtitle_path)

    timeline, subtitle_document = _load_sources(timeline_path, subtitle_path)

    timeline_result = timeline_engine.validate_timeline(timeline)
    if not timeline_result.passed:
        raise SourceValidationError(
            f"Timeline {timeline_path} failed validation: {'; '.join(timeline_result.errors)}"
        )

    subtitle_config = subtitle_engine.load_subtitle_config()
    subtitle_result = subtitle_engine.validate_subtitle_document(subtitle_document, subtitle_config)
    if not subtitle_result.passed:
        raise SourceValidationError(
            f"Subtitle document {subtitle_path} failed validation: {'; '.join(subtitle_result.errors)}"
        )

    _enforce_compatibility(timeline, subtitle_document, config)

    integrated_timeline, mapping, warnings = attach_subtitles_to_timeline(
        timeline,
        subtitle_document,
        config,
        track_id=resolved_track_id,
        replace_existing=replace_existing,
        disable_track=disable_track,
    )

    replaced_existing = any(w.code == "replaced_existing_track" for w in warnings)

    integration_id = _compute_integration_id(
        base_timeline_id=timeline.timeline_id,
        subtitle_document_id=subtitle_document.document_id,
        track_id=resolved_track_id,
        replace_existing=replace_existing,
        disable_track=disable_track,
        schema_version=config.schema_version,
    )
    integrated_timeline_id = _compute_integrated_timeline_id(timeline.timeline_id, integration_id)

    if config.generate_integrated_timeline_id:
        integrated_timeline.metadata["integrated_timeline_id"] = integrated_timeline_id

    integrations = integrated_timeline.metadata.setdefault("subtitle_integrations", [])
    integrations[:] = [entry for entry in integrations if entry.get("integration_id") != integration_id]
    integrations.append(
        {
            "integration_id": integration_id,
            "subtitle_document_id": subtitle_document.document_id,
            "track_id": resolved_track_id,
            "attached_at": _now_iso(),
            "enabled": not disable_track,
            "cue_count": mapping.cue_count,
            "language": mapping.source_language,
        }
    )

    post_validation = timeline_engine.validate_timeline(integrated_timeline)
    adapter_errors = _validate_integrated_subtitle_tracks(integrated_timeline, config)
    validation_passed = post_validation.passed and not adapter_errors
    if not validation_passed:
        all_errors = list(post_validation.errors) + adapter_errors
        raise IntegratedTimelineValidationError(
            f"Integrated timeline failed validation: {'; '.join(all_errors)}"
        )

    output_path_written: Path | None = None
    if output_path is not None:
        try:
            output_path_written = timeline_engine.save_timeline(integrated_timeline, output_path, force=force)
        except timeline_engine.TimelineOutputExistsError as exc:
            raise IntegrationOutputExistsError(str(exc)) from exc
        except timeline_engine.UnsafeTimelineOutputError as exc:
            raise UnsafeIntegrationOutputError(str(exc)) from exc

    timeline_after_hash = _hash_file(timeline_path)
    subtitle_after_hash = _hash_file(subtitle_path)

    result = SubtitleTimelineIntegrationResult(
        integration_id=integration_id,
        base_timeline_id=timeline.timeline_id,
        integrated_timeline_id=integrated_timeline_id,
        subtitle_document_id=subtitle_document.document_id,
        track_id=resolved_track_id,
        track_enabled=not disable_track,
        cue_count=mapping.cue_count,
        language=mapping.source_language,
        replaced_existing_track=replaced_existing,
        validation_passed=validation_passed,
        warnings=[w.message for w in warnings],
        output_path=str(output_path_written) if output_path_written else None,
        source_timeline_unchanged=(timeline_before_hash == timeline_after_hash),
        source_subtitle_unchanged=(subtitle_before_hash == subtitle_after_hash),
        error=None,
    )
    return integrated_timeline, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Subtitle-Timeline Integration (Phase 11D.1). Attaches a "
            "validated subtitle document to a validated Timeline as a new "
            "subtitle track. Never renders, never calls ffmpeg/ffprobe."
        )
    )

    parser.add_argument("--timeline", required=True, help="Path to the source timeline.json.")
    parser.add_argument("--subtitles", required=True, help="Path to the source subtitle document JSON.")
    parser.add_argument("--output", default=None, help="Path to write the integrated timeline JSON.")
    parser.add_argument(
        "--config", default=None, help="Path to an alternate config/video/subtitle_timeline_integration.yaml."
    )
    parser.add_argument("--track-id", dest="track_id", default=None)
    parser.add_argument("--replace-existing", dest="replace_existing", action="store_true")
    parser.add_argument("--disable-track", dest="disable_track", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument(
        "--validate-only",
        dest="validate_only",
        action="store_true",
        help="Validate compatibility only; never writes an output file.",
    )

    arguments = parser.parse_args(argv)

    if arguments.validate_only and arguments.output:
        parser.error("--output cannot be combined with --validate-only.")
    if not arguments.validate_only and not arguments.output:
        parser.error("--output is required unless --validate-only is given.")

    return arguments


def _print_result(result: SubtitleTimelineIntegrationResult) -> None:
    print()
    print("AIKO Subtitle-Timeline Integration (Phase 11D.1)")
    print("----------------------------------------------------")
    print(f"integration_id:          {result.integration_id}")
    print(f"base_timeline_id:        {result.base_timeline_id}")
    print(f"integrated_timeline_id:  {result.integrated_timeline_id}")
    print(f"subtitle_document_id:    {result.subtitle_document_id}")
    print(f"track_id:                {result.track_id}")
    print(f"track_enabled:           {result.track_enabled}")
    print(f"cue_count:               {result.cue_count}")
    print(f"language:                {result.language}")
    print(f"replaced_existing_track: {result.replaced_existing_track}")
    print(f"validation_passed:       {result.validation_passed}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    print(f"output_path:             {result.output_path}")
    print(f"source_timeline_unchanged: {result.source_timeline_unchanged}")
    print(f"source_subtitle_unchanged: {result.source_subtitle_unchanged}")
    print()


def _run_validate_only(arguments: argparse.Namespace, config: SubtitleTimelineIntegrationConfig) -> None:
    timeline_path = Path(arguments.timeline)
    subtitle_path = Path(arguments.subtitles)

    timeline, subtitle_document = _load_sources(timeline_path, subtitle_path)

    timeline_result = timeline_engine.validate_timeline(timeline)
    subtitle_config = subtitle_engine.load_subtitle_config()
    subtitle_result = subtitle_engine.validate_subtitle_document(subtitle_document, subtitle_config)
    compatibility_errors = validate_subtitle_timeline_compatibility(timeline, subtitle_document, config)

    all_errors = list(timeline_result.errors) + list(subtitle_result.errors) + compatibility_errors
    passed = not all_errors

    if arguments.as_json:
        print(json.dumps({"passed": passed, "errors": all_errors}, indent=2, sort_keys=True))
    else:
        print()
        print(f"Validation {'PASSED' if passed else 'FAILED'} ({len(all_errors)} error(s))")
        for error in all_errors:
            print(f"  error: {error}")
        print()

    if not passed:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_integration_config(arguments.config)

        if arguments.validate_only:
            _run_validate_only(arguments, config)
        else:
            _integrated_timeline, result = run_integration(
                arguments.timeline,
                arguments.subtitles,
                config,
                output_path=arguments.output,
                track_id=arguments.track_id,
                replace_existing=arguments.replace_existing,
                disable_track=arguments.disable_track,
                force=arguments.force,
            )

            if arguments.as_json:
                print(json.dumps(asdict(result), indent=2, sort_keys=True))
            else:
                _print_result(result)
    except SubtitleTimelineIntegrationError as exc:
        print(f"[SubtitleTimelineIntegration] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
