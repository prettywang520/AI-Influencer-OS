from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import timeline_engine

# Phase 11E — Overlay Plan Engine. A reusable, platform-neutral,
# renderer-independent planner: converts a validated Timeline's
# subtitle/overlay tracks into one deterministic, validated
# overlay_plan.json for a future render pipeline. This module only ever
# reads timeline_engine.load_timeline()/validate_timeline() — it never
# renders overlays, never burns subtitles into video, never calls
# ffmpeg or ffprobe, never inspects or modifies media files, never uses
# Playwright, never publishes or uploads, never transcribes speech, and
# never downloads fonts or assets. Asset/style references are carried
# through as opaque planning metadata only.

DEFAULT_OVERLAY_PLAN_CONFIG_RELATIVE_PATH = Path("config") / "video" / "overlay_plan.yaml"


def _runtime_root() -> Path:
    """
    overlay_plan_engine.py location:
    10_apps/claude_runtime/src/overlay_plan_engine.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OverlayPlanEngineError(RuntimeError):
    """Base error for the Phase 11E Overlay Plan Engine."""


class OverlayPlanConfigError(OverlayPlanEngineError):
    """Raised when config/video/overlay_plan.yaml is missing or invalid."""


class OverlayTimelineLoadError(OverlayPlanEngineError):
    """Raised when --timeline cannot be loaded via timeline_engine.load_timeline()."""


class OverlayTimelineValidationError(OverlayPlanEngineError):
    """Raised when the loaded Timeline fails timeline_engine.validate_timeline()."""


class OverlayMappingError(OverlayPlanEngineError):
    """Raised for a structural mapping failure: fail_on_no_overlays with zero
    discovered overlays, or a malformed metadata shape that makes mapping
    impossible."""


class UnsupportedOverlayTypeError(OverlayPlanEngineError):
    """Raised when a clip resolves to an overlay type outside OverlayType.ALL
    and config.allow_custom_overlay_types is false."""


class OverlayPositionError(OverlayPlanEngineError):
    """Reserved for direct-API callers of position helpers that want a raised
    failure rather than a soft validation error."""


class OverlaySafeAreaError(OverlayPlanEngineError):
    """Reserved for direct-API callers of safe-area helpers that want a
    raised failure rather than a soft validation error."""


class OverlayCollisionError(OverlayPlanEngineError):
    """Reserved for direct-API callers of collision helpers that want a
    raised failure rather than a soft validation error."""


class OverlayValidationError(OverlayPlanEngineError):
    """Reserved for callers that want validate_overlay_plan() failures
    raised as an exception rather than inspected via
    OverlayPlanValidationResult."""


class OverlayPlanJSONError(OverlayPlanEngineError):
    """Raised when overlay plan JSON is malformed or missing required fields."""


class OverlayPlanOutputExistsError(OverlayPlanEngineError):
    """Raised when --output already exists and --force was not given."""


class UnsafeOverlayPlanOutputError(OverlayPlanEngineError):
    """Raised when --output is a directory, or resolves to the same path as
    --timeline."""


# ---------------------------------------------------------------------------
# String-constant "enums"
# ---------------------------------------------------------------------------


class OverlayType:
    SUBTITLE = "subtitle"
    LOGO = "logo"
    WATERMARK = "watermark"
    LOCATION = "location"
    CTA = "cta"
    STICKER = "sticker"
    CUSTOM = "custom"
    ALL = (SUBTITLE, LOGO, WATERMARK, LOCATION, CTA, STICKER, CUSTOM)


class OverlayCollisionPolicy:
    WARNING = "warning"
    ERROR = "error"
    IGNORE = "ignore"
    ALL = (WARNING, ERROR, IGNORE)


class OverlaySeverity:
    WARNING = "warning"
    ERROR = "error"
    ALL = (WARNING, ERROR)


class CoordinateSpace:
    PIXEL = "pixel"
    NORMALIZED = "normalized"
    ALL = (PIXEL, NORMALIZED)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OverlayTypeConfig:
    layer_order: int = 100
    default_z_index: int = 50
    safe_area_policy: str = OverlaySeverity.WARNING
    collision_group: str = "custom"


@dataclass(slots=True)
class OverlayPlanConfig:
    schema_version: str = "1.0"
    planning_version: str = "11E"
    fail_on_no_overlays: bool = False
    include_disabled_default: bool = False
    allow_custom_overlay_types: bool = True
    validation_tolerance_seconds: float = 0.001
    unresolved_overlay_type_fallback: str = OverlayType.CUSTOM

    coordinate_space: str = CoordinateSpace.PIXEL
    width_fallback: int = 1080
    height_fallback: int = 1920

    safe_area_top: int = 160
    safe_area_bottom: int = 320
    safe_area_left: int = 80
    safe_area_right: int = 80
    title_safe_top: int = 200
    title_safe_bottom: int = 360
    title_safe_left: int = 100
    title_safe_right: int = 100
    action_safe_top: int = 220
    action_safe_bottom: int = 420
    action_safe_left: int = 120
    action_safe_right: int = 120

    overlay_types: dict[str, OverlayTypeConfig] = field(default_factory=dict)

    collision_enabled: bool = True
    treat_unknown_bounds_as_warning: bool = True
    fail_on_subtitle_cta_overlap: bool = True
    fail_on_duplicate_overlay_id: bool = True
    minimum_spacing_pixels: float = 16

    allowed_anchors: tuple[str, ...] = (
        "top_left", "top_center", "top_right",
        "center_left", "center", "center_right",
        "bottom_left", "bottom_center", "bottom_right",
    )
    default_anchor: str = "bottom_center"

    opacity_min: float = 0.0
    opacity_max: float = 1.0
    scale_min: float = 0.01
    scale_max: float = 10.0
    allow_negative_z_index: bool = False

    overwrite_requires_force: bool = True
    atomic_write: bool = True

    def type_config(self, overlay_type: str) -> OverlayTypeConfig:
        return self.overlay_types.get(overlay_type) or self.overlay_types.get(
            OverlayType.CUSTOM, OverlayTypeConfig()
        )


def default_overlay_plan_config_path() -> Path:
    return _runtime_root() / DEFAULT_OVERLAY_PLAN_CONFIG_RELATIVE_PATH


def load_overlay_plan_config(config_path: str | Path | None = None) -> OverlayPlanConfig:
    """Load config/video/overlay_plan.yaml (or an alternate path) into an
    OverlayPlanConfig. Raises OverlayPlanConfigError if the file is
    missing or invalid."""
    path = Path(config_path) if config_path else default_overlay_plan_config_path()

    if not path.exists():
        raise OverlayPlanConfigError(f"Overlay plan config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise OverlayPlanConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise OverlayPlanConfigError(f"Overlay plan config is empty or invalid: {path}")

    plan_section = raw.get("plan") or {}
    canvas_section = raw.get("canvas") or {}
    safe_area_section = raw.get("safe_area") or {}
    title_safe_section = safe_area_section.get("title_safe") or {}
    action_safe_section = safe_area_section.get("action_safe") or {}
    overlay_types_section = raw.get("overlay_types") or {}
    collision_section = raw.get("collision") or {}
    position_section = raw.get("position") or {}
    validation_section = raw.get("validation") or {}
    output_section = raw.get("output") or {}

    overlay_types: dict[str, OverlayTypeConfig] = {}
    for type_name, type_data in overlay_types_section.items():
        type_data = type_data or {}
        overlay_types[type_name] = OverlayTypeConfig(
            layer_order=int(type_data.get("layer_order", 100)),
            default_z_index=int(type_data.get("default_z_index", 50)),
            safe_area_policy=str(type_data.get("safe_area_policy", OverlaySeverity.WARNING)),
            collision_group=str(type_data.get("collision_group", "custom")),
        )
    if OverlayType.CUSTOM not in overlay_types:
        overlay_types[OverlayType.CUSTOM] = OverlayTypeConfig()

    allowed_anchors = position_section.get("allowed_anchors") or [
        "top_left", "top_center", "top_right",
        "center_left", "center", "center_right",
        "bottom_left", "bottom_center", "bottom_right",
    ]

    return OverlayPlanConfig(
        schema_version=str(plan_section.get("schema_version", "1.0")),
        planning_version=str(plan_section.get("planning_version", "11E")),
        fail_on_no_overlays=bool(plan_section.get("fail_on_no_overlays", False)),
        include_disabled_default=bool(plan_section.get("include_disabled_default", False)),
        allow_custom_overlay_types=bool(plan_section.get("allow_custom_overlay_types", True)),
        validation_tolerance_seconds=float(plan_section.get("validation_tolerance_seconds", 0.001)),
        unresolved_overlay_type_fallback=str(
            plan_section.get("unresolved_overlay_type_fallback", OverlayType.CUSTOM)
        ),
        coordinate_space=str(canvas_section.get("coordinate_space", CoordinateSpace.PIXEL)),
        width_fallback=int(canvas_section.get("width_fallback", 1080)),
        height_fallback=int(canvas_section.get("height_fallback", 1920)),
        safe_area_top=int(safe_area_section.get("top", 160)),
        safe_area_bottom=int(safe_area_section.get("bottom", 320)),
        safe_area_left=int(safe_area_section.get("left", 80)),
        safe_area_right=int(safe_area_section.get("right", 80)),
        title_safe_top=int(title_safe_section.get("top", 200)),
        title_safe_bottom=int(title_safe_section.get("bottom", 360)),
        title_safe_left=int(title_safe_section.get("left", 100)),
        title_safe_right=int(title_safe_section.get("right", 100)),
        action_safe_top=int(action_safe_section.get("top", 220)),
        action_safe_bottom=int(action_safe_section.get("bottom", 420)),
        action_safe_left=int(action_safe_section.get("left", 120)),
        action_safe_right=int(action_safe_section.get("right", 120)),
        overlay_types=overlay_types,
        collision_enabled=bool(collision_section.get("enabled", True)),
        treat_unknown_bounds_as_warning=bool(collision_section.get("treat_unknown_bounds_as_warning", True)),
        fail_on_subtitle_cta_overlap=bool(collision_section.get("fail_on_subtitle_cta_overlap", True)),
        fail_on_duplicate_overlay_id=bool(collision_section.get("fail_on_duplicate_overlay_id", True)),
        minimum_spacing_pixels=float(collision_section.get("minimum_spacing_pixels", 16)),
        allowed_anchors=tuple(str(a) for a in allowed_anchors),
        default_anchor=str(position_section.get("default_anchor", "bottom_center")),
        opacity_min=float(validation_section.get("opacity_min", 0.0)),
        opacity_max=float(validation_section.get("opacity_max", 1.0)),
        scale_min=float(validation_section.get("scale_min", 0.01)),
        scale_max=float(validation_section.get("scale_max", 10.0)),
        allow_negative_z_index=bool(validation_section.get("allow_negative_z_index", False)),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        atomic_write=bool(output_section.get("atomic_write", True)),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OverlayPosition:
    x: float | None = None
    y: float | None = None
    coordinate_space: str = CoordinateSpace.PIXEL
    anchor: str = "bottom_center"
    alignment: str = "bottom_center"
    width: float | None = None
    height: float | None = None


@dataclass(slots=True)
class OverlaySafeArea:
    inside_safe_area: bool = True
    violations: list[str] = field(default_factory=list)
    severity: str | None = None
    recommended_position: OverlayPosition | None = None


@dataclass(slots=True)
class OverlayStyleSnapshot:
    style_id: str | None = None
    font_family: str | None = None
    font_size: float | None = None
    font_weight: float | None = None
    italic: bool | None = None
    underline: bool | None = None
    primary_color: str | None = None
    outline_color: str | None = None
    background_color: str | None = None
    outline_width: float | None = None
    shadow_depth: float | None = None
    alignment: str | None = None
    position_x: float | None = None
    position_y: float | None = None
    safe_area_enabled: bool = True
    opacity: float = 1.0
    rotation_degrees: float = 0.0
    scale: float = 1.0
    asset_reference: str | None = None
    width: float | None = None
    height: float | None = None
    anchor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlannedOverlay:
    overlay_id: str = ""
    source_clip_id: str = ""
    source_track_id: str = ""
    overlay_type: str = OverlayType.CUSTOM
    custom_overlay_type: str | None = None
    enabled: bool = True
    timeline_start_seconds: float = 0.0
    timeline_end_seconds: float = 0.0
    duration_seconds: float = 0.0
    layer_id: str = ""
    layer_order: int = 0
    z_index: int = 0
    position: OverlayPosition = field(default_factory=OverlayPosition)
    alignment: str = "bottom_center"
    margin_left: float = 0.0
    margin_right: float = 0.0
    margin_top: float = 0.0
    margin_bottom: float = 0.0
    opacity: float = 1.0
    style_snapshot: OverlayStyleSnapshot = field(default_factory=OverlayStyleSnapshot)
    safe_area_status: OverlaySafeArea = field(default_factory=OverlaySafeArea)
    collision_group: str = "custom"
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    language: str | None = None
    speaker: str | None = None
    style_id: str | None = None
    source_document_id: str | None = None
    source_integration_id: str | None = None


@dataclass(slots=True)
class OverlayLayer:
    layer_id: str = ""
    overlay_type: str = OverlayType.CUSTOM
    order: int = 0
    default_z_index: int = 0
    enabled: bool = True
    overlay_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OverlayCollision:
    collision_id: str = ""
    overlay_ids: list[str] = field(default_factory=list)
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    collision_type: str = "spatial_overlap"
    severity: str = OverlaySeverity.WARNING
    details: str = ""
    suggested_resolution: str | None = None


@dataclass(slots=True)
class OverlayPlanWarning:
    code: str = ""
    message: str = ""
    overlay_id: str | None = None
    layer_id: str | None = None


@dataclass(slots=True)
class OverlayPlan:
    schema_version: str = "1.0"
    plan_id: str = ""
    created_at: str = ""
    timeline_id: str = ""
    integrated_timeline_id: str | None = None
    production_date: str | None = None
    duration_seconds: float = 0.0
    canvas_width: int = 1080
    canvas_height: int = 1920
    frame_rate: float = 30
    layers: list[OverlayLayer] = field(default_factory=list)
    overlays: list[PlannedOverlay] = field(default_factory=list)
    collisions: list[OverlayCollision] = field(default_factory=list)
    overlay_count: int = 0
    enabled_overlay_count: int = 0
    disabled_overlay_count: int = 0
    warnings: list[OverlayPlanWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OverlayPlanValidationResult:
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    collision_count: int = 0
    overlay_count: int = 0
    enabled_overlay_count: int = 0
    layer_count: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Overlay type resolution (requirement 5)
# ---------------------------------------------------------------------------


def resolve_overlay_type(
    clip: timeline_engine.TimelineClip, track: timeline_engine.TimelineTrack, config: OverlayPlanConfig
) -> tuple[str, str | None]:
    """
    Resolves an overlay type deterministically: clip.metadata["overlay_type"]
    -> clip.overlay_type (real OverlayClip) -> "subtitle" for a subtitle
    track -> config's configured fallback. A resolved string outside
    OverlayType.ALL is preserved verbatim as custom_overlay_type and
    reported as OverlayType.CUSTOM when config.allow_custom_overlay_types;
    otherwise raises UnsupportedOverlayTypeError. Never silently
    reinterprets one overlay type as another.
    """
    raw = clip.metadata.get("overlay_type")

    if raw is None and isinstance(clip, timeline_engine.OverlayClip):
        raw = clip.overlay_type
        if raw == "text":
            # OverlayClip's own dataclass default -- not a real signal
            # that this is specifically a "text" overlay type.
            raw = None

    if raw is None and track.track_type == timeline_engine.TrackType.SUBTITLE:
        raw = OverlayType.SUBTITLE

    if raw is None:
        raw = config.unresolved_overlay_type_fallback

    if raw in OverlayType.ALL:
        return raw, None

    if config.allow_custom_overlay_types:
        return OverlayType.CUSTOM, raw

    raise UnsupportedOverlayTypeError(
        f"Clip {clip.clip_id} resolved to unsupported overlay type {raw!r} and "
        "allow_custom_overlay_types is disabled"
    )


# ---------------------------------------------------------------------------
# Discovery (requirement 4)
# ---------------------------------------------------------------------------


def discover_overlay_clips(
    timeline: timeline_engine.Timeline, config: OverlayPlanConfig, *, include_disabled: bool = False
) -> list[tuple[timeline_engine.TimelineTrack, timeline_engine.TimelineClip]]:
    """
    Discovers overlay-source clips: every clip on a subtitle/overlay
    track, plus (narrow opt-in) any clip on a video/audio/metadata track
    whose metadata explicitly sets is_overlay_descriptor=True. Never
    reinterprets ordinary video/audio clips as overlays. Deterministic
    order: (track.order, clip.start, clip.clip_id).
    """
    pairs: list[tuple[timeline_engine.TimelineTrack, timeline_engine.TimelineClip]] = []

    for track in timeline.tracks:
        is_overlay_source_track = track.track_type in (
            timeline_engine.TrackType.SUBTITLE,
            timeline_engine.TrackType.OVERLAY,
        )
        for clip in track.clips:
            if not clip.enabled and not include_disabled:
                continue
            if is_overlay_source_track or clip.metadata.get("is_overlay_descriptor") is True:
                pairs.append((track, clip))

    pairs.sort(key=lambda pair: (pair[0].order, pair[1].start, pair[1].clip_id))
    return pairs


# ---------------------------------------------------------------------------
# Position / style extraction — duck-typed over clip.metadata, matching
# the shape src/subtitle_timeline_integration.py already writes.
# ---------------------------------------------------------------------------


def _extract_position(clip: timeline_engine.TimelineClip, config: OverlayPlanConfig) -> OverlayPosition:
    raw = clip.metadata.get("position") or {}
    return OverlayPosition(
        x=raw.get("x"),
        y=raw.get("y"),
        coordinate_space=str(raw.get("coordinate_space", config.coordinate_space)),
        anchor=str(raw.get("anchor", config.default_anchor)),
        alignment=str(raw.get("alignment", config.default_anchor)),
        width=raw.get("width"),
        height=raw.get("height"),
    )


def _extract_style_snapshot(clip: timeline_engine.TimelineClip, config: OverlayPlanConfig) -> OverlayStyleSnapshot:
    style_data = clip.metadata.get("style") or {}
    asset_data = clip.metadata.get("asset") or {}
    return OverlayStyleSnapshot(
        style_id=style_data.get("style_id") or clip.metadata.get("style_id"),
        font_family=style_data.get("font_family"),
        font_size=style_data.get("font_size"),
        font_weight=style_data.get("font_weight"),
        italic=style_data.get("italic"),
        underline=style_data.get("underline"),
        primary_color=style_data.get("primary_color"),
        outline_color=style_data.get("outline_color"),
        background_color=style_data.get("background_color"),
        outline_width=style_data.get("outline_width"),
        shadow_depth=style_data.get("shadow_depth"),
        alignment=style_data.get("alignment"),
        position_x=style_data.get("position_x"),
        position_y=style_data.get("position_y"),
        safe_area_enabled=bool(style_data.get("safe_area_enabled", True)),
        opacity=float(clip.metadata.get("opacity", 1.0)),
        rotation_degrees=float(clip.metadata.get("rotation_degrees", 0.0)),
        scale=float(clip.metadata.get("scale", 1.0)),
        asset_reference=asset_data.get("asset_reference") or clip.metadata.get("asset_reference"),
        width=asset_data.get("width"),
        height=asset_data.get("height"),
        anchor=asset_data.get("anchor"),
        metadata=dict(style_data.get("metadata") or {}),
    )


# ---------------------------------------------------------------------------
# Safe-area calculation (requirement 9) — planning metadata only, never
# repositions anything.
# ---------------------------------------------------------------------------


def calculate_safe_area_status(
    position: OverlayPosition,
    overlay_type: str,
    canvas_width: int,
    canvas_height: int,
    config: OverlayPlanConfig,
) -> OverlaySafeArea:
    if position.x is None or position.y is None:
        return OverlaySafeArea(inside_safe_area=True, violations=[], severity=None)

    if position.coordinate_space == CoordinateSpace.NORMALIZED:
        pixel_x = position.x * canvas_width
        pixel_y = position.y * canvas_height
    else:
        pixel_x, pixel_y = position.x, position.y

    type_config = config.type_config(overlay_type)

    top, bottom = config.safe_area_top, canvas_height - config.safe_area_bottom
    left, right = config.safe_area_left, canvas_width - config.safe_area_right

    violations: list[str] = []
    if pixel_y < top or pixel_y > bottom:
        violations.append(f"y={pixel_y} outside safe area [{top}, {bottom}]")
    if pixel_x < left or pixel_x > right:
        violations.append(f"x={pixel_x} outside safe area [{left}, {right}]")

    if overlay_type == OverlayType.CTA:
        action_top = config.action_safe_top
        action_bottom = canvas_height - config.action_safe_bottom
        action_left = config.action_safe_left
        action_right = canvas_width - config.action_safe_right
        if pixel_y < action_top or pixel_y > action_bottom or pixel_x < action_left or pixel_x > action_right:
            violations.append(
                f"position ({pixel_x}, {pixel_y}) outside action-safe zone "
                f"[{action_left}, {action_right}] x [{action_top}, {action_bottom}]"
            )

    inside = not violations
    severity = None if inside else type_config.safe_area_policy

    recommended: OverlayPosition | None = None
    if not inside:
        recommended = OverlayPosition(
            x=canvas_width / 2,
            y=canvas_height - config.safe_area_bottom,
            coordinate_space=CoordinateSpace.PIXEL,
            anchor=config.default_anchor,
            alignment=config.default_anchor,
        )

    return OverlaySafeArea(
        inside_safe_area=inside, violations=violations, severity=severity, recommended_position=recommended
    )


# ---------------------------------------------------------------------------
# Layering / z-index (requirement 10)
# ---------------------------------------------------------------------------


def calculate_overlay_layers(overlays: list[PlannedOverlay], config: OverlayPlanConfig) -> list[OverlayLayer]:
    """Deterministic layer grouping by overlay_type, ordered by
    config.overlay_types[type].layer_order (ties broken by type name).
    Each layer's overlay_ids are sorted by (timeline_start_seconds,
    overlay_id). Video/audio tracks are never represented here since
    `overlays` only ever contains discovered overlay-source clips."""
    by_type: dict[str, list[PlannedOverlay]] = {}
    for overlay in overlays:
        by_type.setdefault(overlay.overlay_type, []).append(overlay)

    ordered_types = sorted(by_type, key=lambda t: (config.type_config(t).layer_order, t))

    layers: list[OverlayLayer] = []
    for overlay_type in ordered_types:
        type_config = config.type_config(overlay_type)
        group = sorted(by_type[overlay_type], key=lambda o: (o.timeline_start_seconds, o.overlay_id))
        layers.append(
            OverlayLayer(
                layer_id=f"layer_{overlay_type}",
                overlay_type=overlay_type,
                order=type_config.layer_order,
                default_z_index=type_config.default_z_index,
                enabled=any(o.enabled for o in group),
                overlay_ids=[o.overlay_id for o in group],
            )
        )
    return layers


# ---------------------------------------------------------------------------
# Collision detection (requirement 12) — renderer-neutral, deterministic,
# never guesses missing bounds, never auto-repositions anything.
# ---------------------------------------------------------------------------


def _boxes_overlap(a: OverlayPosition, b: OverlayPosition, spacing: float) -> bool:
    a_left, a_right = a.x - a.width / 2, a.x + a.width / 2
    a_top, a_bottom = a.y - a.height / 2, a.y + a.height / 2
    b_left, b_right = b.x - b.width / 2, b.x + b.width / 2
    b_top, b_bottom = b.y - b.height / 2, b.y + b.height / 2
    return not (
        a_right + spacing <= b_left
        or b_right + spacing <= a_left
        or a_bottom + spacing <= b_top
        or b_bottom + spacing <= a_top
    )


def detect_overlay_collisions(overlays: list[PlannedOverlay], config: OverlayPlanConfig) -> list[OverlayCollision]:
    if not config.collision_enabled:
        return []

    tolerance = 1e-9
    active = [o for o in overlays if o.enabled]
    collisions: list[OverlayCollision] = []

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            time_overlap = (
                a.timeline_start_seconds < b.timeline_end_seconds - tolerance
                and b.timeline_start_seconds < a.timeline_end_seconds - tolerance
            )
            if not time_overlap:
                continue

            collision_id = f"collision_{a.overlay_id}_{b.overlay_id}"
            start = max(a.timeline_start_seconds, b.timeline_start_seconds)
            end = min(a.timeline_end_seconds, b.timeline_end_seconds)

            a_has_bounds = all(
                v is not None for v in (a.position.x, a.position.y, a.position.width, a.position.height)
            )
            b_has_bounds = all(
                v is not None for v in (b.position.x, b.position.y, b.position.width, b.position.height)
            )

            if not (a_has_bounds and b_has_bounds):
                if config.treat_unknown_bounds_as_warning:
                    collisions.append(
                        OverlayCollision(
                            collision_id=collision_id,
                            overlay_ids=[a.overlay_id, b.overlay_id],
                            start_seconds=start,
                            end_seconds=end,
                            collision_type="unknown_bounds",
                            severity=OverlaySeverity.WARNING,
                            details=(
                                f"{a.overlay_id} and/or {b.overlay_id} has no planned width/height; "
                                "spatial overlap cannot be determined"
                            ),
                        )
                    )
                continue

            if _boxes_overlap(a.position, b.position, config.minimum_spacing_pixels):
                collision_type = "spatial_overlap"
                severity = OverlaySeverity.WARNING
                overlay_type_pair = {a.overlay_type, b.overlay_type}
                if overlay_type_pair == {OverlayType.SUBTITLE, OverlayType.CTA} and config.fail_on_subtitle_cta_overlap:
                    collision_type = "cta_subtitle_conflict"
                    severity = OverlaySeverity.ERROR
                elif a.collision_group == b.collision_group == "branding":
                    collision_type = "layer_conflict"
                    severity = OverlaySeverity.WARNING

                collisions.append(
                    OverlayCollision(
                        collision_id=collision_id,
                        overlay_ids=[a.overlay_id, b.overlay_id],
                        start_seconds=start,
                        end_seconds=end,
                        collision_type=collision_type,
                        severity=severity,
                        details=f"{a.overlay_id} and {b.overlay_id} overlap spatially while both active",
                        suggested_resolution="reposition one overlay or adjust timing (metadata only)",
                    )
                )
            elif a.position.x == b.position.x and a.position.y == b.position.y and a.position.x is not None:
                collisions.append(
                    OverlayCollision(
                        collision_id=collision_id,
                        overlay_ids=[a.overlay_id, b.overlay_id],
                        start_seconds=start,
                        end_seconds=end,
                        collision_type="duplicate_position",
                        severity=OverlaySeverity.WARNING,
                        details=f"{a.overlay_id} and {b.overlay_id} share the exact same position while both active",
                    )
                )

    return sorted(collisions, key=lambda c: c.collision_id)


# ---------------------------------------------------------------------------
# Cue / clip mapping (requirements 6-8)
# ---------------------------------------------------------------------------


def map_timeline_overlay_clip(
    track: timeline_engine.TimelineTrack, clip: timeline_engine.TimelineClip, config: OverlayPlanConfig
) -> PlannedOverlay:
    """Maps one discovered TimelineClip into a typed PlannedOverlay. Pure
    planning -- never raises for position/style/safe-area problems (those
    are judged by validate_overlay_plan()); only overlay-type resolution
    can raise here (UnsupportedOverlayTypeError)."""
    overlay_type, custom_type = resolve_overlay_type(clip, track, config)
    type_config = config.type_config(overlay_type)

    position = _extract_position(clip, config)
    style_snapshot = _extract_style_snapshot(clip, config)

    raw_z_index = clip.metadata.get("z_index")
    z_index = int(raw_z_index) if raw_z_index is not None else type_config.default_z_index

    content = getattr(clip, "content", "") or clip.metadata.get("content", "")
    overlay_metadata = dict(clip.metadata.get("cue_metadata") or clip.metadata.get("custom_metadata") or {})

    return PlannedOverlay(
        overlay_id=f"overlay_{clip.clip_id}",
        source_clip_id=clip.clip_id,
        source_track_id=track.track_id,
        overlay_type=overlay_type,
        custom_overlay_type=custom_type,
        enabled=clip.enabled,
        timeline_start_seconds=clip.start,
        timeline_end_seconds=clip.end,
        duration_seconds=clip.duration_seconds,
        layer_id=f"layer_{overlay_type}",
        layer_order=type_config.layer_order,
        z_index=z_index,
        position=position,
        alignment=str(clip.metadata.get("alignment") or position.alignment),
        margin_left=float(clip.metadata.get("margin_left", 0.0)),
        margin_right=float(clip.metadata.get("margin_right", 0.0)),
        margin_top=float(clip.metadata.get("margin_top", 0.0)),
        margin_bottom=float(clip.metadata.get("margin_bottom", 0.0)),
        opacity=style_snapshot.opacity,
        style_snapshot=style_snapshot,
        safe_area_status=OverlaySafeArea(),
        collision_group=str(clip.metadata.get("collision_group") or type_config.collision_group),
        content=content,
        metadata=overlay_metadata,
        language=clip.metadata.get("language"),
        speaker=clip.metadata.get("speaker"),
        style_id=clip.metadata.get("style_id"),
        source_document_id=clip.metadata.get("source_document_id"),
        source_integration_id=clip.metadata.get("source_integration_id"),
    )


# ---------------------------------------------------------------------------
# Deterministic plan_id (requirement 14)
# ---------------------------------------------------------------------------


def _compute_plan_id(plan: OverlayPlan, *, include_disabled: bool) -> str:
    payload = {
        "timeline_id": plan.timeline_id,
        "integrated_timeline_id": plan.integrated_timeline_id,
        "include_disabled": include_disabled,
        "schema_version": plan.schema_version,
        "overlays": [
            {
                "overlay_id": o.overlay_id,
                "overlay_type": o.overlay_type,
                "custom_overlay_type": o.custom_overlay_type,
                "enabled": o.enabled,
                "timeline_start_seconds": round(o.timeline_start_seconds, 6),
                "timeline_end_seconds": round(o.timeline_end_seconds, 6),
                "layer_id": o.layer_id,
                "z_index": o.z_index,
                "position": {
                    "x": o.position.x,
                    "y": o.position.y,
                    "coordinate_space": o.position.coordinate_space,
                    "anchor": o.position.anchor,
                    "width": o.position.width,
                    "height": o.position.height,
                },
                "content": o.content,
                "style_snapshot": {
                    key: value for key, value in asdict(o.style_snapshot).items() if key != "metadata"
                },
            }
            for o in plan.overlays
        ],
        "layers": [
            {"layer_id": l.layer_id, "order": l.order, "default_z_index": l.default_z_index}
            for l in plan.layers
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Build (requirements 3, 4, 13, 18) — planning only, deep-copy/read-only
# over the loaded Timeline, never mutated or saved.
# ---------------------------------------------------------------------------


def build_overlay_plan(
    timeline_path: str | Path, config: OverlayPlanConfig, *, include_disabled: bool | None = None
) -> OverlayPlan:
    timeline_path = Path(timeline_path)
    effective_include_disabled = (
        config.include_disabled_default if include_disabled is None else include_disabled
    )

    try:
        timeline = timeline_engine.load_timeline(timeline_path)
    except timeline_engine.TimelineEngineError as exc:
        raise OverlayTimelineLoadError(f"Failed to load timeline {timeline_path}: {exc}") from exc

    timeline_result = timeline_engine.validate_timeline(timeline)
    if not timeline_result.passed:
        raise OverlayTimelineValidationError(
            f"Timeline {timeline_path} failed validation: {'; '.join(timeline_result.errors)}"
        )

    all_pairs = discover_overlay_clips(timeline, config, include_disabled=True)
    included_pairs = discover_overlay_clips(timeline, config, include_disabled=effective_include_disabled)
    ignored_disabled_count = len(all_pairs) - len(included_pairs)

    if not included_pairs and config.fail_on_no_overlays:
        raise OverlayMappingError(
            f"Timeline {timeline_path} has no subtitle/overlay clips and fail_on_no_overlays is enabled"
        )

    warnings: list[OverlayPlanWarning] = []
    overlays: list[PlannedOverlay] = []

    canvas_width = timeline.width or config.width_fallback
    canvas_height = timeline.height or config.height_fallback

    for track, clip in included_pairs:
        overlay = map_timeline_overlay_clip(track, clip, config)
        overlay.safe_area_status = calculate_safe_area_status(
            overlay.position, overlay.overlay_type, canvas_width, canvas_height, config
        )
        overlays.append(overlay)

    if not overlays:
        warnings.append(
            OverlayPlanWarning(code="no_overlays_found", message="Timeline has no active subtitle/overlay clips")
        )

    layers = calculate_overlay_layers(overlays, config)
    collisions = detect_overlay_collisions(overlays, config)

    enabled_count = sum(1 for o in overlays if o.enabled)
    disabled_count = len(overlays) - enabled_count

    plan = OverlayPlan(
        schema_version=config.schema_version,
        timeline_id=timeline.timeline_id,
        integrated_timeline_id=timeline.metadata.get("integrated_timeline_id"),
        production_date=timeline.production_date,
        duration_seconds=timeline.duration_seconds,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        frame_rate=timeline.fps,
        layers=layers,
        overlays=overlays,
        collisions=collisions,
        overlay_count=len(overlays),
        enabled_overlay_count=enabled_count,
        disabled_overlay_count=disabled_count,
        warnings=warnings,
        metadata={
            "source_timeline_schema_version": timeline.schema_version,
            "source_timeline_path": str(timeline_path),
            "configuration_version": config.schema_version,
            "planning_version": config.planning_version,
            "ignored_track_types": [
                timeline_engine.TrackType.VIDEO,
                timeline_engine.TrackType.AUDIO,
                timeline_engine.TrackType.METADATA,
            ],
            "ignored_disabled_count": ignored_disabled_count,
            "include_disabled": effective_include_disabled,
        },
    )
    plan.created_at = _now_iso()
    plan.plan_id = _compute_plan_id(plan, include_disabled=effective_include_disabled)
    return plan


# ---------------------------------------------------------------------------
# Validation (requirement 17) — all pass/fail policy lives here, not in
# build_overlay_plan().
# ---------------------------------------------------------------------------


def validate_overlay_plan(
    plan: OverlayPlan, config: OverlayPlanConfig, *, timeline: timeline_engine.Timeline | None = None
) -> OverlayPlanValidationResult:
    tolerance = config.validation_tolerance_seconds
    errors: list[str] = []
    warnings: list[str] = []

    layer_ids = [layer.layer_id for layer in plan.layers]
    if len(layer_ids) != len(set(layer_ids)):
        errors.append("plan.layers: duplicate layer_id values field: layer_id expected unique")

    overlay_ids = [overlay.overlay_id for overlay in plan.overlays]
    if len(overlay_ids) != len(set(overlay_ids)):
        errors.append("plan.overlays: duplicate overlay_id values field: overlay_id expected unique")

    layer_id_set = set(layer_ids)

    for overlay in plan.overlays:
        if overlay.layer_id not in layer_id_set:
            errors.append(
                f"overlay {overlay.overlay_id}: layer_id={overlay.layer_id!r} field: layer_id expected "
                "to reference a layer present in plan.layers"
            )

        if overlay.timeline_start_seconds < 0:
            errors.append(
                f"overlay {overlay.overlay_id}: timeline_start_seconds={overlay.timeline_start_seconds} "
                "field: timeline_start_seconds expected >= 0"
            )
        if overlay.timeline_end_seconds <= overlay.timeline_start_seconds:
            errors.append(
                f"overlay {overlay.overlay_id}: timeline_end_seconds={overlay.timeline_end_seconds} "
                f"field: timeline_end_seconds expected > timeline_start_seconds "
                f"({overlay.timeline_start_seconds})"
            )
        expected_duration = overlay.timeline_end_seconds - overlay.timeline_start_seconds
        if abs(overlay.duration_seconds - expected_duration) > tolerance:
            errors.append(
                f"overlay {overlay.overlay_id}: duration_seconds={overlay.duration_seconds} "
                f"field: duration_seconds expected {expected_duration}"
            )
        if overlay.timeline_end_seconds > plan.duration_seconds + tolerance:
            errors.append(
                f"overlay {overlay.overlay_id}: timeline_end_seconds={overlay.timeline_end_seconds} "
                f"field: timeline_end_seconds expected <= plan.duration_seconds ({plan.duration_seconds})"
            )

        position = overlay.position
        if position.coordinate_space not in CoordinateSpace.ALL:
            errors.append(
                f"overlay {overlay.overlay_id}: position.coordinate_space={position.coordinate_space!r} "
                f"field: coordinate_space expected one of {CoordinateSpace.ALL}"
            )
        elif position.coordinate_space == CoordinateSpace.NORMALIZED:
            if position.x is not None and not (0.0 <= position.x <= 1.0):
                errors.append(
                    f"overlay {overlay.overlay_id}: position.x={position.x} field: normalized x "
                    "expected within 0.0-1.0"
                )
            if position.y is not None and not (0.0 <= position.y <= 1.0):
                errors.append(
                    f"overlay {overlay.overlay_id}: position.y={position.y} field: normalized y "
                    "expected within 0.0-1.0"
                )
        else:
            if position.x is not None and not (0 <= position.x <= plan.canvas_width):
                errors.append(
                    f"overlay {overlay.overlay_id}: position.x={position.x} field: pixel x expected "
                    f"within 0-{plan.canvas_width}"
                )
            if position.y is not None and not (0 <= position.y <= plan.canvas_height):
                errors.append(
                    f"overlay {overlay.overlay_id}: position.y={position.y} field: pixel y expected "
                    f"within 0-{plan.canvas_height}"
                )

        if position.width is not None and position.width <= 0:
            errors.append(
                f"overlay {overlay.overlay_id}: position.width={position.width} field: width expected > 0"
            )
        if position.height is not None and position.height <= 0:
            errors.append(
                f"overlay {overlay.overlay_id}: position.height={position.height} field: height expected > 0"
            )
        if position.anchor not in config.allowed_anchors:
            errors.append(
                f"overlay {overlay.overlay_id}: position.anchor={position.anchor!r} field: anchor "
                f"expected one of {config.allowed_anchors}"
            )

        if not (config.opacity_min <= overlay.opacity <= config.opacity_max):
            errors.append(
                f"overlay {overlay.overlay_id}: opacity={overlay.opacity} field: opacity expected "
                f"within [{config.opacity_min}, {config.opacity_max}]"
            )
        if not (config.scale_min <= overlay.style_snapshot.scale <= config.scale_max):
            errors.append(
                f"overlay {overlay.overlay_id}: scale={overlay.style_snapshot.scale} field: scale "
                f"expected within [{config.scale_min}, {config.scale_max}]"
            )

        if overlay.z_index < 0 and not config.allow_negative_z_index:
            errors.append(
                f"overlay {overlay.overlay_id}: z_index={overlay.z_index} field: z_index expected >= 0 "
                "(allow_negative_z_index is false)"
            )

        if overlay.enabled and overlay.safe_area_status.severity == OverlaySeverity.ERROR:
            errors.append(
                f"overlay {overlay.overlay_id}: safe_area_status field: violates safe area with error "
                f"policy ({'; '.join(overlay.safe_area_status.violations)})"
            )
        elif overlay.safe_area_status.severity == OverlaySeverity.WARNING:
            warnings.append(
                f"overlay {overlay.overlay_id}: safe-area warning ({'; '.join(overlay.safe_area_status.violations)})"
            )

        if overlay.enabled and overlay.overlay_type == OverlayType.SUBTITLE and not overlay.content.strip():
            errors.append(
                f"overlay {overlay.overlay_id}: content={overlay.content!r} field: content expected "
                "non-empty for an active subtitle overlay"
            )

        try:
            json.dumps(asdict(overlay.style_snapshot))
            json.dumps(overlay.metadata)
        except TypeError:
            errors.append(
                f"overlay {overlay.overlay_id}: metadata/style_snapshot field: expected JSON-serializable"
            )

    for collision in plan.collisions:
        if collision.severity == OverlaySeverity.ERROR:
            errors.append(
                f"collision {collision.collision_id}: {collision.details} field: severity=error"
            )
        elif collision.severity == OverlaySeverity.WARNING:
            warnings.append(f"collision {collision.collision_id}: {collision.details}")

    if not plan.timeline_id:
        errors.append("plan.timeline_id: empty field: timeline_id expected non-empty source Timeline identity")

    if timeline is not None and abs(plan.duration_seconds - timeline.duration_seconds) > tolerance:
        errors.append(
            f"plan.duration_seconds={plan.duration_seconds} field: duration_seconds expected to match "
            f"source Timeline duration ({timeline.duration_seconds})"
        )

    passed = len(errors) == 0
    enabled_count = sum(1 for overlay in plan.overlays if overlay.enabled)
    summary = (
        f"Overlay plan {plan.plan_id or '(no id)'}: {'PASSED' if passed else 'FAILED'} "
        f"({len(errors)} error(s), {len(warnings)} warning(s))"
    )

    return OverlayPlanValidationResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        collision_count=len(plan.collisions),
        overlay_count=len(plan.overlays),
        enabled_overlay_count=enabled_count,
        layer_count=len(plan.layers),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# JSON serialization — stable, round-trip-safe (requirement 16)
# ---------------------------------------------------------------------------


def _position_to_dict(position: OverlayPosition) -> dict[str, Any]:
    return asdict(position)


def _position_from_dict(data: dict[str, Any] | None) -> OverlayPosition:
    if not data:
        return OverlayPosition()
    return OverlayPosition(
        x=data.get("x"),
        y=data.get("y"),
        coordinate_space=data.get("coordinate_space", CoordinateSpace.PIXEL),
        anchor=data.get("anchor", "bottom_center"),
        alignment=data.get("alignment", "bottom_center"),
        width=data.get("width"),
        height=data.get("height"),
    )


def _safe_area_to_dict(safe_area: OverlaySafeArea) -> dict[str, Any]:
    return {
        "inside_safe_area": safe_area.inside_safe_area,
        "violations": list(safe_area.violations),
        "severity": safe_area.severity,
        "recommended_position": (
            _position_to_dict(safe_area.recommended_position) if safe_area.recommended_position else None
        ),
    }


def _safe_area_from_dict(data: dict[str, Any] | None) -> OverlaySafeArea:
    data = data or {}
    return OverlaySafeArea(
        inside_safe_area=data.get("inside_safe_area", True),
        violations=list(data.get("violations", [])),
        severity=data.get("severity"),
        recommended_position=(
            _position_from_dict(data["recommended_position"]) if data.get("recommended_position") else None
        ),
    )


def _style_snapshot_to_dict(style: OverlayStyleSnapshot) -> dict[str, Any]:
    return asdict(style)


def _style_snapshot_from_dict(data: dict[str, Any] | None) -> OverlayStyleSnapshot:
    data = data or {}
    known = {f.name for f in dataclasses.fields(OverlayStyleSnapshot)}
    return OverlayStyleSnapshot(**{key: value for key, value in data.items() if key in known})


def _planned_overlay_to_dict(overlay: PlannedOverlay) -> dict[str, Any]:
    data = asdict(overlay)
    data["position"] = _position_to_dict(overlay.position)
    data["style_snapshot"] = _style_snapshot_to_dict(overlay.style_snapshot)
    data["safe_area_status"] = _safe_area_to_dict(overlay.safe_area_status)
    return data


def _planned_overlay_from_dict(data: dict[str, Any]) -> PlannedOverlay:
    known = {f.name for f in dataclasses.fields(PlannedOverlay)}
    filtered = {key: value for key, value in data.items() if key in known}
    filtered["position"] = _position_from_dict(filtered.get("position"))
    filtered["style_snapshot"] = _style_snapshot_from_dict(filtered.get("style_snapshot"))
    filtered["safe_area_status"] = _safe_area_from_dict(filtered.get("safe_area_status"))
    try:
        return PlannedOverlay(**filtered)
    except TypeError as exc:
        raise OverlayPlanJSONError(f"Malformed planned overlay in overlay plan JSON: {exc}") from exc


def _layer_to_dict(layer: OverlayLayer) -> dict[str, Any]:
    return asdict(layer)


def _layer_from_dict(data: dict[str, Any]) -> OverlayLayer:
    known = {f.name for f in dataclasses.fields(OverlayLayer)}
    try:
        return OverlayLayer(**{key: value for key, value in data.items() if key in known})
    except TypeError as exc:
        raise OverlayPlanJSONError(f"Malformed overlay layer in overlay plan JSON: {exc}") from exc


def _collision_to_dict(collision: OverlayCollision) -> dict[str, Any]:
    return asdict(collision)


def _collision_from_dict(data: dict[str, Any]) -> OverlayCollision:
    known = {f.name for f in dataclasses.fields(OverlayCollision)}
    return OverlayCollision(**{key: value for key, value in data.items() if key in known})


def _warning_to_dict(warning: OverlayPlanWarning) -> dict[str, Any]:
    return asdict(warning)


def _warning_from_dict(data: dict[str, Any]) -> OverlayPlanWarning:
    known = {f.name for f in dataclasses.fields(OverlayPlanWarning)}
    return OverlayPlanWarning(**{key: value for key, value in data.items() if key in known})


def overlay_plan_to_dict(plan: OverlayPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "plan_id": plan.plan_id,
        "created_at": plan.created_at,
        "timeline_id": plan.timeline_id,
        "integrated_timeline_id": plan.integrated_timeline_id,
        "production_date": plan.production_date,
        "duration_seconds": plan.duration_seconds,
        "canvas_width": plan.canvas_width,
        "canvas_height": plan.canvas_height,
        "frame_rate": plan.frame_rate,
        "layers": [_layer_to_dict(layer) for layer in plan.layers],
        "overlays": [_planned_overlay_to_dict(overlay) for overlay in plan.overlays],
        "collisions": [_collision_to_dict(collision) for collision in plan.collisions],
        "overlay_count": plan.overlay_count,
        "enabled_overlay_count": plan.enabled_overlay_count,
        "disabled_overlay_count": plan.disabled_overlay_count,
        "warnings": [_warning_to_dict(warning) for warning in plan.warnings],
        "metadata": plan.metadata,
    }


def overlay_plan_from_dict(data: dict[str, Any]) -> OverlayPlan:
    try:
        return OverlayPlan(
            schema_version=data.get("schema_version", "1.0"),
            plan_id=data.get("plan_id", ""),
            created_at=data.get("created_at", ""),
            timeline_id=data.get("timeline_id", ""),
            integrated_timeline_id=data.get("integrated_timeline_id"),
            production_date=data.get("production_date"),
            duration_seconds=data.get("duration_seconds", 0.0),
            canvas_width=data.get("canvas_width", 1080),
            canvas_height=data.get("canvas_height", 1920),
            frame_rate=data.get("frame_rate", 30),
            layers=[_layer_from_dict(layer) for layer in data.get("layers", [])],
            overlays=[_planned_overlay_from_dict(overlay) for overlay in data.get("overlays", [])],
            collisions=[_collision_from_dict(collision) for collision in data.get("collisions", [])],
            overlay_count=data.get("overlay_count", 0),
            enabled_overlay_count=data.get("enabled_overlay_count", 0),
            disabled_overlay_count=data.get("disabled_overlay_count", 0),
            warnings=[_warning_from_dict(warning) for warning in data.get("warnings", [])],
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise OverlayPlanJSONError(f"Overlay plan JSON missing required field: {exc}") from exc
    except TypeError as exc:
        raise OverlayPlanJSONError(f"Malformed overlay plan JSON: {exc}") from exc


def load_overlay_plan(path: str | Path) -> OverlayPlan:
    path = Path(path)
    if not path.is_file():
        raise OverlayPlanJSONError(f"Overlay plan file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OverlayPlanJSONError(f"Invalid overlay plan JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise OverlayPlanJSONError(f"Overlay plan JSON root must be an object: {path}")

    return overlay_plan_from_dict(data)


def save_overlay_plan(plan: OverlayPlan, path: str | Path, *, force: bool = False) -> Path:
    """Atomically writes overlay plan JSON via temp-file + Path.replace()
    so a reader never observes a partially-written file. Refuses to
    overwrite an existing file unless force=True."""
    path = Path(path)

    if path.exists():
        if path.is_dir():
            raise UnsafeOverlayPlanOutputError(f"Output path is a directory: {path}")
        if not force:
            raise OverlayPlanOutputExistsError(f"{path} already exists; pass force=True to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(overlay_plan_to_dict(plan), indent=2, sort_keys=True, ensure_ascii=False)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Overlay Plan Engine (Phase 11E). Converts a validated "
            "Timeline's subtitle/overlay tracks into a deterministic, "
            "validated overlay_plan.json. Never renders, never calls "
            "ffmpeg/ffprobe."
        )
    )

    parser.add_argument("--timeline", required=True, help="Path to the source timeline.json.")
    parser.add_argument("--output", default=None, help="Path to write the overlay plan JSON.")
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/overlay_plan.yaml.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument("--include-disabled", dest="include_disabled", action="store_true")
    parser.add_argument(
        "--validate-only",
        dest="validate_only",
        action="store_true",
        help="Validate the Timeline/derived plan only; never writes an output file.",
    )

    arguments = parser.parse_args(argv)

    if arguments.validate_only and arguments.output:
        parser.error("--output cannot be combined with --validate-only.")
    if not arguments.validate_only and not arguments.output:
        parser.error("--output is required unless --validate-only is given.")

    return arguments


def _print_build_summary(plan: OverlayPlan, output_path: Path | None) -> None:
    print()
    print("AIKO Overlay Plan Engine (Phase 11E)")
    print("----------------------------------------")
    print(f"plan_id:                 {plan.plan_id}")
    print(f"timeline_id:             {plan.timeline_id}")
    print(f"integrated_timeline_id:  {plan.integrated_timeline_id}")
    print(f"duration_seconds:        {plan.duration_seconds:.3f}")
    print(f"layers:                  {len(plan.layers)}")
    print(f"overlays:                {plan.overlay_count} (enabled={plan.enabled_overlay_count}, "
          f"disabled={plan.disabled_overlay_count})")
    print(f"collisions:              {len(plan.collisions)}")
    for warning in plan.warnings:
        print(f"warning: {warning.message}")
    print(f"output_path:             {output_path if output_path else '(not written)'}")
    print()


def _print_validation_summary(result: OverlayPlanValidationResult) -> None:
    print()
    print("AIKO Overlay Plan Engine — validation (Phase 11E)")
    print("--------------------------------------------------------")
    print(result.summary)
    for error in result.errors:
        print(f"  error:   {error}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_overlay_plan_config(arguments.config)
        timeline_path = Path(arguments.timeline)

        if not arguments.validate_only and arguments.output:
            output_path = Path(arguments.output)
            if output_path.resolve() == timeline_path.resolve():
                raise UnsafeOverlayPlanOutputError("--output must not be the same path as --timeline")

        plan = build_overlay_plan(timeline_path, config, include_disabled=arguments.include_disabled)

        if arguments.validate_only:
            timeline = timeline_engine.load_timeline(timeline_path)
            result = validate_overlay_plan(plan, config, timeline=timeline)
            if arguments.as_json:
                print(json.dumps(asdict(result), indent=2, sort_keys=True))
            else:
                _print_validation_summary(result)
            if not result.passed:
                raise SystemExit(1)
            return

        output_path_written: Path | None = None
        if arguments.output:
            output_path_written = save_overlay_plan(plan, arguments.output, force=arguments.force)

        if arguments.as_json:
            print(json.dumps(overlay_plan_to_dict(plan), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_build_summary(plan, output_path_written)
    except OverlayPlanEngineError as exc:
        print(f"[OverlayPlanEngine] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
