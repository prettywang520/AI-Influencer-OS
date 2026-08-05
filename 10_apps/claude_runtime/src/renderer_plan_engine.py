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

from . import overlay_plan_engine, timeline_engine

# Phase 11E.1 — Renderer Plan Engine. A reusable, platform-neutral
# adapter that combines an already-validated Timeline, an already-
# validated Overlay Plan, and (optional) descriptive music metadata
# into one deterministic renderer_plan.json describing HOW a future
# renderer should execute -- pass order, dependency graph, renderer
# hints, and opaque asset references. This module never renders
# anything, never calls ffmpeg or ffprobe, never opens, verifies, or
# downloads any asset/font, never uses Playwright, never transcribes
# speech, and never publishes or uploads. It only ever reads
# timeline_engine.load_timeline()/validate_timeline() and
# overlay_plan_engine.load_overlay_plan()/validate_overlay_plan() -- it
# never imports subtitle_engine, subtitle_timeline_integration,
# music_mixer, video_engine, or reel_builder.

DEFAULT_RENDERER_PLAN_CONFIG_RELATIVE_PATH = Path("config") / "video" / "renderer_plan.yaml"


def _runtime_root() -> Path:
    """
    renderer_plan_engine.py location:
    10_apps/claude_runtime/src/renderer_plan_engine.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RendererPlanEngineError(RuntimeError):
    """Base error for the Phase 11E.1 Renderer Plan Engine."""


class RendererPlanConfigError(RendererPlanEngineError):
    """Raised when config/video/renderer_plan.yaml is missing or invalid."""


class RendererTimelineLoadError(RendererPlanEngineError):
    """Raised when --timeline cannot be loaded via timeline_engine.load_timeline()."""


class RendererTimelineValidationError(RendererPlanEngineError):
    """Raised when the loaded Timeline fails timeline_engine.validate_timeline()."""


class RendererOverlayPlanLoadError(RendererPlanEngineError):
    """Raised when --overlay-plan cannot be loaded via overlay_plan_engine.load_overlay_plan()."""


class RendererOverlayPlanValidationError(RendererPlanEngineError):
    """Raised when the loaded Overlay Plan fails overlay_plan_engine.validate_overlay_plan()."""


class RendererMusicMetadataError(RendererPlanEngineError):
    """Raised when --music-plan is malformed or missing required fields."""


class RendererPlanMismatchError(RendererPlanEngineError):
    """Raised when the Overlay Plan's timeline_id does not match the target Timeline's timeline_id."""


class RendererPlanMappingError(RendererPlanEngineError):
    """Raised for a structural mapping failure while deriving tracks/assets/passes."""


class RendererPlanValidationError(RendererPlanEngineError):
    """Reserved for callers that want validate_renderer_plan() failures raised as
    an exception rather than inspected via RendererValidation."""


class RendererPlanJSONError(RendererPlanEngineError):
    """Raised when renderer plan JSON is malformed or missing required fields."""


class RendererPlanOutputExistsError(RendererPlanEngineError):
    """Raised when --output already exists and --force was not given."""


class UnsafeRendererPlanOutputError(RendererPlanEngineError):
    """Raised when --output is a directory, or resolves to the same path as
    --timeline or --overlay-plan."""


# ---------------------------------------------------------------------------
# String-constant "enums"
# ---------------------------------------------------------------------------


class RenderPassType:
    VIDEO = "video"
    SUBTITLE = "subtitle"
    OVERLAY = "overlay"
    MUSIC = "music"
    FINAL_ENCODE = "final_encode"
    CUSTOM = "custom"
    ALL = (VIDEO, SUBTITLE, OVERLAY, MUSIC, FINAL_ENCODE, CUSTOM)


class RendererHintType:
    """Recognized planning hints only -- never executed by this module."""

    DRAWTEXT = "drawtext"
    ASS = "ass"
    OVERLAY_PNG = "overlay_png"
    OVERLAY_ALPHA = "overlay_alpha"
    IMAGE_SEQUENCE = "image_sequence"
    COPY = "copy"
    TRANSCODE = "transcode"
    NORMALIZE = "normalize"
    ALL = (DRAWTEXT, ASS, OVERLAY_PNG, OVERLAY_ALPHA, IMAGE_SEQUENCE, COPY, TRANSCODE, NORMALIZE)


class AssetType:
    VIDEO = "video"
    AUDIO = "audio"
    MUSIC = "music"
    IMAGE = "image"
    FONT = "font"
    SUBTITLE_FILE = "subtitle_file"
    CUSTOM = "custom"
    ALL = (VIDEO, AUDIO, MUSIC, IMAGE, FONT, SUBTITLE_FILE, CUSTOM)


class RendererSeverity:
    WARNING = "warning"
    ERROR = "error"
    ALL = (WARNING, ERROR)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RendererPlanConfig:
    schema_version: str = "1.0"
    renderer_version: str = "11E.1"
    validation_tolerance_seconds: float = 0.001

    width_fallback: int = 1080
    height_fallback: int = 1920
    frame_rate_fallback: float = 30

    pass_order: tuple[str, ...] = (
        RenderPassType.VIDEO, RenderPassType.SUBTITLE, RenderPassType.OVERLAY,
        RenderPassType.MUSIC, RenderPassType.FINAL_ENCODE,
    )

    video_hint: str = RendererHintType.COPY
    subtitle_hint: str = RendererHintType.DRAWTEXT
    overlay_hint: str = RendererHintType.OVERLAY_PNG
    music_hint: str = RendererHintType.NORMALIZE

    output_container: str = "mp4"
    output_video_codec_hint: str = "h264"
    output_audio_codec_hint: str = "aac"

    allow_orphan_assets: bool = False
    allow_unknown_renderer_hints: bool = False

    overwrite_requires_force: bool = True
    atomic_write: bool = True


def default_renderer_plan_config_path() -> Path:
    return _runtime_root() / DEFAULT_RENDERER_PLAN_CONFIG_RELATIVE_PATH


def load_renderer_plan_config(config_path: str | Path | None = None) -> RendererPlanConfig:
    """Load config/video/renderer_plan.yaml (or an alternate path) into a
    RendererPlanConfig. Raises RendererPlanConfigError if the file is
    missing or invalid."""
    path = Path(config_path) if config_path else default_renderer_plan_config_path()

    if not path.exists():
        raise RendererPlanConfigError(f"Renderer plan config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise RendererPlanConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise RendererPlanConfigError(f"Renderer plan config is empty or invalid: {path}")

    plan_section = raw.get("renderer_plan") or {}
    canvas_section = raw.get("canvas") or {}
    ordering_section = raw.get("ordering") or {}
    hints_section = raw.get("hints") or {}
    output_profile_section = raw.get("output_profile") or {}
    validation_section = raw.get("validation") or {}
    output_section = raw.get("output") or {}

    pass_order = ordering_section.get("pass_order") or [
        RenderPassType.VIDEO, RenderPassType.SUBTITLE, RenderPassType.OVERLAY,
        RenderPassType.MUSIC, RenderPassType.FINAL_ENCODE,
    ]

    return RendererPlanConfig(
        schema_version=str(plan_section.get("schema_version", "1.0")),
        renderer_version=str(plan_section.get("renderer_version", "11E.1")),
        validation_tolerance_seconds=float(plan_section.get("validation_tolerance_seconds", 0.001)),
        width_fallback=int(canvas_section.get("width_fallback", 1080)),
        height_fallback=int(canvas_section.get("height_fallback", 1920)),
        frame_rate_fallback=float(canvas_section.get("frame_rate_fallback", 30)),
        pass_order=tuple(str(p) for p in pass_order),
        video_hint=str(hints_section.get("video_hint", RendererHintType.COPY)),
        subtitle_hint=str(hints_section.get("subtitle_hint", RendererHintType.DRAWTEXT)),
        overlay_hint=str(hints_section.get("overlay_hint", RendererHintType.OVERLAY_PNG)),
        music_hint=str(hints_section.get("music_hint", RendererHintType.NORMALIZE)),
        output_container=str(output_profile_section.get("container", "mp4")),
        output_video_codec_hint=str(output_profile_section.get("video_codec_hint", "h264")),
        output_audio_codec_hint=str(output_profile_section.get("audio_codec_hint", "aac")),
        allow_orphan_assets=bool(validation_section.get("allow_orphan_assets", False)),
        allow_unknown_renderer_hints=bool(validation_section.get("allow_unknown_renderer_hints", False)),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        atomic_write=bool(output_section.get("atomic_write", True)),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RendererAsset:
    asset_id: str = ""
    asset_type: str = AssetType.CUSTOM
    source_path: str = ""
    logical_role: str = ""
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RendererHint:
    hint_id: str = ""
    hint_type: str = RendererHintType.COPY
    target_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RendererVideoTrack:
    track_id: str = ""
    source_clip_ids: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    width: int = 1080
    height: int = 1920
    frame_rate: float = 30
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RendererAudioTrack:
    track_id: str = ""
    source_clip_ids: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RendererOverlayTrack:
    track_id: str = ""
    overlay_ids: list[str] = field(default_factory=list)
    layer_id: str = ""
    asset_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RendererSubtitleTrack:
    track_id: str = ""
    overlay_ids: list[str] = field(default_factory=list)
    language: str | None = None
    asset_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RendererMusicTrack:
    track_id: str = ""
    asset_id: str | None = None
    volume: float = 0.2
    mode: str = "loop"
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0
    ducking_mode: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RendererPass:
    pass_id: str = ""
    pass_type: str = RenderPassType.CUSTOM
    order: int = 0
    enabled: bool = True
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RendererValidation:
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pass_count: int = 0
    asset_count: int = 0
    track_count: int = 0
    summary: str = ""


@dataclass(slots=True)
class RendererPlanWarning:
    code: str = ""
    message: str = ""


@dataclass(slots=True)
class RendererPlan:
    schema_version: str = "1.0"
    renderer_plan_id: str = ""
    timeline_id: str = ""
    overlay_plan_id: str | None = None
    created_at: str = ""
    canvas_width: int = 1080
    canvas_height: int = 1920
    frame_rate: float = 30
    duration_seconds: float = 0.0
    video_tracks: list[RendererVideoTrack] = field(default_factory=list)
    audio_tracks: list[RendererAudioTrack] = field(default_factory=list)
    overlay_tracks: list[RendererOverlayTrack] = field(default_factory=list)
    subtitle_tracks: list[RendererSubtitleTrack] = field(default_factory=list)
    music_tracks: list[RendererMusicTrack] = field(default_factory=list)
    passes: list[RendererPass] = field(default_factory=list)
    assets: list[RendererAsset] = field(default_factory=list)
    hints: list[RendererHint] = field(default_factory=list)
    output_profile: dict[str, Any] = field(default_factory=dict)
    warnings: list[RendererPlanWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Asset registry — deterministic, deduplicated, never opens/verifies a
# file. Shared across video/audio/overlay/subtitle/music derivation.
# ---------------------------------------------------------------------------


class _AssetRegistry:
    def __init__(self) -> None:
        self.assets: list[RendererAsset] = []
        self._seen: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def get_or_create(self, asset_type: str, source_path: str, logical_role: str) -> str:
        key = (asset_type, source_path)
        if key in self._seen:
            return self._seen[key]
        index = self._counters.get(asset_type, 0) + 1
        self._counters[asset_type] = index
        asset_id = f"asset_{asset_type}_{index:04d}"
        self.assets.append(
            RendererAsset(asset_id=asset_id, asset_type=asset_type, source_path=source_path, logical_role=logical_role)
        )
        self._seen[key] = asset_id
        return asset_id


# ---------------------------------------------------------------------------
# Music metadata (optional, descriptive-only — never touches music_mixer.py
# or real ffprobe-derived duration data)
# ---------------------------------------------------------------------------


def load_music_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RendererMusicMetadataError(f"--music-plan file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RendererMusicMetadataError(f"Invalid music metadata JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RendererMusicMetadataError(f"Music metadata JSON root must be an object: {path}")
    if not data.get("music_path"):
        raise RendererMusicMetadataError(f"Music metadata at {path} is missing required 'music_path'")

    return data


def _derive_music_track(
    music_metadata: dict[str, Any] | None, registry: _AssetRegistry
) -> RendererMusicTrack | None:
    if music_metadata is None:
        return None

    asset_id = registry.get_or_create(AssetType.MUSIC, str(music_metadata["music_path"]), "background_music")
    return RendererMusicTrack(
        track_id="renderer_track_music",
        asset_id=asset_id,
        volume=float(music_metadata.get("volume", 0.2)),
        mode=str(music_metadata.get("mode", "loop")),
        fade_in_seconds=float(music_metadata.get("fade_in_seconds", 0.0)),
        fade_out_seconds=float(music_metadata.get("fade_out_seconds", 0.0)),
        ducking_mode=str(music_metadata.get("ducking_mode", "none")),
        metadata=dict(music_metadata.get("metadata") or {}),
    )


# ---------------------------------------------------------------------------
# Track / asset derivation from Timeline + Overlay Plan
# ---------------------------------------------------------------------------


def _derive_video_audio_tracks(
    timeline: timeline_engine.Timeline, registry: _AssetRegistry
) -> tuple[list[RendererVideoTrack], list[RendererAudioTrack]]:
    video_tracks: list[RendererVideoTrack] = []
    audio_tracks: list[RendererAudioTrack] = []

    for track in timeline.tracks:
        if track.track_type == timeline_engine.TrackType.VIDEO:
            asset_ids: list[str] = []
            clip_ids: list[str] = []
            duration = 0.0
            for clip in track.clips:
                if not clip.enabled:
                    continue
                asset_ids.append(registry.get_or_create(AssetType.VIDEO, clip.source_path, "scene_video"))
                clip_ids.append(clip.clip_id)
                duration += clip.duration_seconds
            video_tracks.append(
                RendererVideoTrack(
                    track_id=track.track_id, source_clip_ids=clip_ids, asset_ids=asset_ids,
                    duration_seconds=duration, width=timeline.width, height=timeline.height,
                    frame_rate=timeline.fps,
                )
            )
        elif track.track_type == timeline_engine.TrackType.AUDIO:
            asset_ids = []
            clip_ids = []
            duration = 0.0
            for clip in track.clips:
                if not clip.enabled:
                    continue
                asset_ids.append(registry.get_or_create(AssetType.AUDIO, clip.source_path, "background_audio"))
                clip_ids.append(clip.clip_id)
                duration += clip.duration_seconds
            audio_tracks.append(
                RendererAudioTrack(
                    track_id=track.track_id, source_clip_ids=clip_ids, asset_ids=asset_ids, duration_seconds=duration
                )
            )

    return video_tracks, audio_tracks


def _derive_overlay_subtitle_tracks(
    overlay_plan: overlay_plan_engine.OverlayPlan, registry: _AssetRegistry
) -> tuple[RendererSubtitleTrack | None, list[RendererOverlayTrack]]:
    subtitle_overlay_ids: list[str] = []
    subtitle_asset_ids: list[str] = []
    subtitle_language: str | None = None

    groups: dict[str, dict[str, Any]] = {}

    for overlay in overlay_plan.overlays:
        if not overlay.enabled:
            continue

        asset_id: str | None = None
        asset_reference = overlay.style_snapshot.asset_reference if overlay.style_snapshot else None
        if asset_reference:
            asset_id = registry.get_or_create(AssetType.IMAGE, asset_reference, overlay.overlay_type)

        if overlay.overlay_type == overlay_plan_engine.OverlayType.SUBTITLE:
            subtitle_overlay_ids.append(overlay.overlay_id)
            if asset_id:
                subtitle_asset_ids.append(asset_id)
            if overlay.language and subtitle_language is None:
                subtitle_language = overlay.language
        else:
            group = groups.setdefault(overlay.layer_id, {"overlay_ids": [], "asset_ids": []})
            group["overlay_ids"].append(overlay.overlay_id)
            if asset_id:
                group["asset_ids"].append(asset_id)

    subtitle_track = None
    if subtitle_overlay_ids:
        subtitle_track = RendererSubtitleTrack(
            track_id="renderer_track_subtitle",
            overlay_ids=subtitle_overlay_ids,
            asset_ids=subtitle_asset_ids,
            language=subtitle_language,
        )

    overlay_tracks = [
        RendererOverlayTrack(
            track_id=f"renderer_track_overlay_{layer_id}",
            overlay_ids=group["overlay_ids"],
            layer_id=layer_id,
            asset_ids=group["asset_ids"],
        )
        for layer_id, group in sorted(groups.items())
    ]

    return subtitle_track, overlay_tracks


# ---------------------------------------------------------------------------
# Passes & dependency graph (deterministic, linear)
# ---------------------------------------------------------------------------


def _compute_passes(
    *, has_video: bool, has_subtitle: bool, has_overlay: bool, has_music: bool, config: RendererPlanConfig
) -> tuple[list[RendererPass], list[RendererHint]]:
    presence = {
        RenderPassType.VIDEO: has_video,
        RenderPassType.SUBTITLE: has_subtitle,
        RenderPassType.OVERLAY: has_overlay,
        RenderPassType.MUSIC: has_music,
        RenderPassType.FINAL_ENCODE: True,
    }
    hint_by_pass_type = {
        RenderPassType.VIDEO: config.video_hint,
        RenderPassType.SUBTITLE: config.subtitle_hint,
        RenderPassType.OVERLAY: config.overlay_hint,
        RenderPassType.MUSIC: config.music_hint,
    }

    stages = list(config.pass_order)
    if RenderPassType.FINAL_ENCODE not in stages:
        stages.append(RenderPassType.FINAL_ENCODE)

    passes: list[RendererPass] = []
    hints: list[RendererHint] = []
    previous_pass_id: str | None = None
    order_index = 0

    for pass_type in stages:
        if not presence.get(pass_type, False):
            continue

        pass_id = f"pass_{pass_type}"
        dependencies = [previous_pass_id] if previous_pass_id else []
        passes.append(
            RendererPass(
                pass_id=pass_id, pass_type=pass_type, order=order_index, enabled=True,
                inputs=list(dependencies), outputs=[f"{pass_id}_output"], dependencies=dependencies,
            )
        )
        if pass_type in hint_by_pass_type:
            hints.append(
                RendererHint(hint_id=f"hint_{pass_id}", hint_type=hint_by_pass_type[pass_type], target_id=pass_id)
            )

        previous_pass_id = pass_id
        order_index += 1

    return passes, hints


# ---------------------------------------------------------------------------
# Deterministic renderer_plan_id
# ---------------------------------------------------------------------------


def _compute_renderer_plan_id(plan: RendererPlan, config: RendererPlanConfig) -> str:
    def _track_ref(track: Any) -> dict[str, Any]:
        return {
            "track_id": track.track_id,
            "asset_ids": list(getattr(track, "asset_ids", [])),
        }

    payload = {
        "timeline_id": plan.timeline_id,
        "overlay_plan_id": plan.overlay_plan_id,
        "schema_version": plan.schema_version,
        "renderer_version": config.renderer_version,
        "canvas": {"width": plan.canvas_width, "height": plan.canvas_height, "frame_rate": plan.frame_rate},
        "video_tracks": [_track_ref(t) for t in plan.video_tracks],
        "audio_tracks": [_track_ref(t) for t in plan.audio_tracks],
        "overlay_tracks": [_track_ref(t) for t in plan.overlay_tracks],
        "subtitle_tracks": [_track_ref(t) for t in plan.subtitle_tracks],
        "music_tracks": [_track_ref(t) for t in plan.music_tracks],
        "passes": [
            {"pass_id": p.pass_id, "pass_type": p.pass_type, "order": p.order, "dependencies": p.dependencies}
            for p in plan.passes
        ],
        "assets": [
            {"asset_id": a.asset_id, "asset_type": a.asset_type, "source_path": a.source_path}
            for a in plan.assets
        ],
        "hints": [{"hint_id": h.hint_id, "hint_type": h.hint_type, "target_id": h.target_id} for h in plan.hints],
        "output_profile": plan.output_profile,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Build — planning only.
# ---------------------------------------------------------------------------


def build_renderer_plan(
    timeline_path: str | Path,
    overlay_plan_path: str | Path,
    config: RendererPlanConfig,
    *,
    music_plan_path: str | Path | None = None,
) -> RendererPlan:
    timeline_path = Path(timeline_path)
    overlay_plan_path = Path(overlay_plan_path)

    try:
        timeline = timeline_engine.load_timeline(timeline_path)
    except timeline_engine.TimelineEngineError as exc:
        raise RendererTimelineLoadError(f"Failed to load timeline {timeline_path}: {exc}") from exc

    timeline_result = timeline_engine.validate_timeline(timeline)
    if not timeline_result.passed:
        raise RendererTimelineValidationError(
            f"Timeline {timeline_path} failed validation: {'; '.join(timeline_result.errors)}"
        )

    try:
        overlay_plan = overlay_plan_engine.load_overlay_plan(overlay_plan_path)
    except overlay_plan_engine.OverlayPlanEngineError as exc:
        raise RendererOverlayPlanLoadError(f"Failed to load overlay plan {overlay_plan_path}: {exc}") from exc

    overlay_plan_config = overlay_plan_engine.load_overlay_plan_config()
    overlay_result = overlay_plan_engine.validate_overlay_plan(overlay_plan, overlay_plan_config, timeline=timeline)
    if not overlay_result.passed:
        raise RendererOverlayPlanValidationError(
            f"Overlay plan {overlay_plan_path} failed validation: {'; '.join(overlay_result.errors)}"
        )

    if overlay_plan.timeline_id != timeline.timeline_id:
        raise RendererPlanMismatchError(
            f"Overlay plan timeline_id {overlay_plan.timeline_id!r} does not match target "
            f"timeline_id {timeline.timeline_id!r}"
        )

    music_metadata = load_music_metadata(music_plan_path) if music_plan_path else None

    registry = _AssetRegistry()
    video_tracks, audio_tracks = _derive_video_audio_tracks(timeline, registry)
    subtitle_track, overlay_tracks = _derive_overlay_subtitle_tracks(overlay_plan, registry)
    music_track = _derive_music_track(music_metadata, registry)

    subtitle_tracks = [subtitle_track] if subtitle_track else []
    music_tracks = [music_track] if music_track else []

    passes, hints = _compute_passes(
        has_video=bool(video_tracks),
        has_subtitle=bool(subtitle_tracks),
        has_overlay=bool(overlay_tracks),
        has_music=bool(music_tracks),
        config=config,
    )

    warnings: list[RendererPlanWarning] = []
    if not overlay_plan.overlays:
        warnings.append(
            RendererPlanWarning(
                code="empty_overlay_plan",
                message="Overlay plan has no overlays; only video/audio passes will run",
            )
        )

    plan = RendererPlan(
        schema_version=config.schema_version,
        timeline_id=timeline.timeline_id,
        overlay_plan_id=overlay_plan.plan_id,
        canvas_width=timeline.width or config.width_fallback,
        canvas_height=timeline.height or config.height_fallback,
        frame_rate=timeline.fps or config.frame_rate_fallback,
        duration_seconds=timeline.duration_seconds,
        video_tracks=video_tracks,
        audio_tracks=audio_tracks,
        overlay_tracks=overlay_tracks,
        subtitle_tracks=subtitle_tracks,
        music_tracks=music_tracks,
        passes=passes,
        assets=registry.assets,
        hints=hints,
        output_profile={
            "container": config.output_container,
            "video_codec_hint": config.output_video_codec_hint,
            "audio_codec_hint": config.output_audio_codec_hint,
        },
        warnings=warnings,
        metadata={
            "renderer_version": config.renderer_version,
            "source_timeline_path": str(timeline_path),
            "source_overlay_plan_path": str(overlay_plan_path),
            "source_music_plan_path": str(music_plan_path) if music_plan_path else None,
        },
    )
    plan.created_at = _now_iso()
    plan.renderer_plan_id = _compute_renderer_plan_id(plan, config)
    return plan


# ---------------------------------------------------------------------------
# Validation — all pass/fail policy lives here.
# ---------------------------------------------------------------------------


def _detect_pass_cycle(passes: list[RendererPass]) -> bool:
    graph = {p.pass_id: [d for d in p.dependencies if d] for p in passes}
    white, gray, black = 0, 1, 2
    color = {pass_id: white for pass_id in graph}

    def visit(node: str) -> bool:
        color[node] = gray
        for dep in graph.get(node, []):
            if dep not in color:
                continue
            if color[dep] == gray:
                return True
            if color[dep] == white and visit(dep):
                return True
        color[node] = black
        return False

    return any(color[pass_id] == white and visit(pass_id) for pass_id in graph)


def validate_renderer_plan(
    plan: RendererPlan,
    config: RendererPlanConfig,
    *,
    timeline: timeline_engine.Timeline | None = None,
    overlay_plan: overlay_plan_engine.OverlayPlan | None = None,
) -> RendererValidation:
    errors: list[str] = []
    warnings: list[str] = []

    all_tracks = (
        list(plan.video_tracks) + list(plan.audio_tracks) + list(plan.overlay_tracks)
        + list(plan.subtitle_tracks) + list(plan.music_tracks)
    )
    all_track_ids = [t.track_id for t in all_tracks]
    if len(all_track_ids) != len(set(all_track_ids)):
        errors.append("renderer plan tracks: duplicate track_id values field: track_id expected unique")
    track_id_set = set(all_track_ids)

    asset_ids = [a.asset_id for a in plan.assets]
    if len(asset_ids) != len(set(asset_ids)):
        errors.append("plan.assets: duplicate asset_id values field: asset_id expected unique")
    asset_id_set = set(asset_ids)

    pass_ids = [p.pass_id for p in plan.passes]
    if len(pass_ids) != len(set(pass_ids)):
        errors.append("plan.passes: duplicate pass_id values field: pass_id expected unique")
    pass_id_set = set(pass_ids)

    referenced_asset_ids: set[str] = set()
    for track in all_tracks:
        referenced_asset_ids.update(getattr(track, "asset_ids", []))
        asset_id = getattr(track, "asset_id", None)
        if asset_id:
            referenced_asset_ids.add(asset_id)

    orphan_assets = asset_id_set - referenced_asset_ids
    if orphan_assets and not config.allow_orphan_assets:
        for asset_id in sorted(orphan_assets):
            errors.append(f"asset {asset_id}: not referenced by any track field: expected to be referenced")

    for pass_ in plan.passes:
        if pass_.pass_type not in RenderPassType.ALL:
            errors.append(
                f"pass {pass_.pass_id}: pass_type={pass_.pass_type!r} field: pass_type expected one of "
                f"{RenderPassType.ALL}"
            )
        for dependency in pass_.dependencies:
            if dependency not in pass_id_set:
                errors.append(
                    f"pass {pass_.pass_id}: dependencies references unknown pass_id {dependency!r}"
                )

    if _detect_pass_cycle(plan.passes):
        errors.append("plan.passes: dependency cycle detected field: dependencies expected acyclic")

    all_outputs = [output for pass_ in plan.passes for output in pass_.outputs]
    if len(all_outputs) != len(set(all_outputs)):
        errors.append("plan.passes: duplicate outputs across passes field: outputs expected unique")

    for hint in plan.hints:
        if hint.hint_type not in RendererHintType.ALL and not config.allow_unknown_renderer_hints:
            errors.append(
                f"hint {hint.hint_id}: hint_type={hint.hint_type!r} field: hint_type expected one of "
                f"{RendererHintType.ALL}"
            )
        if hint.target_id not in pass_id_set and hint.target_id not in track_id_set:
            errors.append(
                f"hint {hint.hint_id}: target_id={hint.target_id!r} field: target_id expected to "
                "reference a known pass or track"
            )

    if not plan.timeline_id:
        errors.append("plan.timeline_id: empty field: timeline_id expected non-empty")
    if timeline is not None and plan.timeline_id != timeline.timeline_id:
        errors.append(
            f"plan.timeline_id={plan.timeline_id!r} field: timeline_id expected to match supplied "
            f"Timeline ({timeline.timeline_id!r})"
        )
    if overlay_plan is not None and plan.overlay_plan_id != overlay_plan.plan_id:
        errors.append(
            f"plan.overlay_plan_id={plan.overlay_plan_id!r} field: overlay_plan_id expected to match "
            f"supplied OverlayPlan ({overlay_plan.plan_id!r})"
        )

    if not plan.video_tracks:
        errors.append("plan.video_tracks: empty field: video_tracks expected at least one entry")

    for track in plan.music_tracks:
        if not track.asset_id:
            errors.append(f"music track {track.track_id}: asset_id missing field: expected non-empty")

    try:
        json.dumps(plan.metadata)
        for asset in plan.assets:
            json.dumps(asset.metadata)
        for pass_ in plan.passes:
            json.dumps(pass_.metadata)
    except TypeError:
        errors.append("plan metadata is not JSON-serializable")

    passed = len(errors) == 0
    summary = (
        f"Renderer plan {plan.renderer_plan_id or '(no id)'}: {'PASSED' if passed else 'FAILED'} "
        f"({len(errors)} error(s), {len(warnings)} warning(s))"
    )

    return RendererValidation(
        passed=passed,
        errors=errors,
        warnings=warnings,
        pass_count=len(plan.passes),
        asset_count=len(plan.assets),
        track_count=len(all_track_ids),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# JSON serialization — stable, round-trip-safe
# ---------------------------------------------------------------------------


def _dataclass_to_dict(instance: Any) -> dict[str, Any]:
    return asdict(instance)


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    filtered = {key: value for key, value in (data or {}).items() if key in known}
    try:
        return cls(**filtered)
    except TypeError as exc:
        raise RendererPlanJSONError(f"Malformed {cls.__name__} in renderer plan JSON: {exc}") from exc


def renderer_plan_to_dict(plan: RendererPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "renderer_plan_id": plan.renderer_plan_id,
        "timeline_id": plan.timeline_id,
        "overlay_plan_id": plan.overlay_plan_id,
        "created_at": plan.created_at,
        "canvas_width": plan.canvas_width,
        "canvas_height": plan.canvas_height,
        "frame_rate": plan.frame_rate,
        "duration_seconds": plan.duration_seconds,
        "video_tracks": [_dataclass_to_dict(t) for t in plan.video_tracks],
        "audio_tracks": [_dataclass_to_dict(t) for t in plan.audio_tracks],
        "overlay_tracks": [_dataclass_to_dict(t) for t in plan.overlay_tracks],
        "subtitle_tracks": [_dataclass_to_dict(t) for t in plan.subtitle_tracks],
        "music_tracks": [_dataclass_to_dict(t) for t in plan.music_tracks],
        "passes": [_dataclass_to_dict(p) for p in plan.passes],
        "assets": [_dataclass_to_dict(a) for a in plan.assets],
        "hints": [_dataclass_to_dict(h) for h in plan.hints],
        "output_profile": plan.output_profile,
        "warnings": [_dataclass_to_dict(w) for w in plan.warnings],
        "metadata": plan.metadata,
    }


def renderer_plan_from_dict(data: dict[str, Any]) -> RendererPlan:
    try:
        return RendererPlan(
            schema_version=data.get("schema_version", "1.0"),
            renderer_plan_id=data.get("renderer_plan_id", ""),
            timeline_id=data.get("timeline_id", ""),
            overlay_plan_id=data.get("overlay_plan_id"),
            created_at=data.get("created_at", ""),
            canvas_width=data.get("canvas_width", 1080),
            canvas_height=data.get("canvas_height", 1920),
            frame_rate=data.get("frame_rate", 30),
            duration_seconds=data.get("duration_seconds", 0.0),
            video_tracks=[_dataclass_from_dict(RendererVideoTrack, t) for t in data.get("video_tracks", [])],
            audio_tracks=[_dataclass_from_dict(RendererAudioTrack, t) for t in data.get("audio_tracks", [])],
            overlay_tracks=[_dataclass_from_dict(RendererOverlayTrack, t) for t in data.get("overlay_tracks", [])],
            subtitle_tracks=[_dataclass_from_dict(RendererSubtitleTrack, t) for t in data.get("subtitle_tracks", [])],
            music_tracks=[_dataclass_from_dict(RendererMusicTrack, t) for t in data.get("music_tracks", [])],
            passes=[_dataclass_from_dict(RendererPass, p) for p in data.get("passes", [])],
            assets=[_dataclass_from_dict(RendererAsset, a) for a in data.get("assets", [])],
            hints=[_dataclass_from_dict(RendererHint, h) for h in data.get("hints", [])],
            output_profile=data.get("output_profile", {}),
            warnings=[_dataclass_from_dict(RendererPlanWarning, w) for w in data.get("warnings", [])],
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise RendererPlanJSONError(f"Renderer plan JSON missing required field: {exc}") from exc
    except TypeError as exc:
        raise RendererPlanJSONError(f"Malformed renderer plan JSON: {exc}") from exc


def load_renderer_plan(path: str | Path) -> RendererPlan:
    path = Path(path)
    if not path.is_file():
        raise RendererPlanJSONError(f"Renderer plan file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RendererPlanJSONError(f"Invalid renderer plan JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RendererPlanJSONError(f"Renderer plan JSON root must be an object: {path}")

    return renderer_plan_from_dict(data)


def save_renderer_plan(plan: RendererPlan, path: str | Path, *, force: bool = False) -> Path:
    """Atomically writes renderer plan JSON via temp-file + Path.replace()
    so a reader never observes a partially-written file. Refuses to
    overwrite an existing file unless force=True."""
    path = Path(path)

    if path.exists():
        if path.is_dir():
            raise UnsafeRendererPlanOutputError(f"Output path is a directory: {path}")
        if not force:
            raise RendererPlanOutputExistsError(f"{path} already exists; pass force=True to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(renderer_plan_to_dict(plan), indent=2, sort_keys=True, ensure_ascii=False)
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
            "AIKO Renderer Plan Engine (Phase 11E.1). Combines a validated "
            "Timeline, a validated Overlay Plan, and optional music metadata "
            "into a deterministic renderer_plan.json. Never renders, never "
            "calls ffmpeg/ffprobe."
        )
    )

    parser.add_argument("--timeline", required=True, help="Path to the source timeline.json.")
    parser.add_argument("--overlay-plan", dest="overlay_plan", required=True, help="Path to the source overlay_plan.json.")
    parser.add_argument("--output", default=None, help="Path to write the renderer plan JSON.")
    parser.add_argument("--music-plan", dest="music_plan", default=None, help="Path to optional descriptive music metadata JSON.")
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/renderer_plan.yaml.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument(
        "--validate-only",
        dest="validate_only",
        action="store_true",
        help="Build and validate the renderer plan only; never writes an output file.",
    )

    arguments = parser.parse_args(argv)

    if arguments.validate_only and arguments.output:
        parser.error("--output cannot be combined with --validate-only.")
    if not arguments.validate_only and not arguments.output:
        parser.error("--output is required unless --validate-only is given.")

    return arguments


def _print_build_summary(plan: RendererPlan, output_path: Path | None) -> None:
    print()
    print("AIKO Renderer Plan Engine (Phase 11E.1)")
    print("-------------------------------------------")
    print(f"renderer_plan_id:  {plan.renderer_plan_id}")
    print(f"timeline_id:       {plan.timeline_id}")
    print(f"overlay_plan_id:   {plan.overlay_plan_id}")
    print(f"duration_seconds:  {plan.duration_seconds:.3f}")
    print(f"passes:            {[p.pass_id for p in plan.passes]}")
    print(f"assets:            {len(plan.assets)}")
    print(
        "tracks:            "
        f"video={len(plan.video_tracks)} audio={len(plan.audio_tracks)} "
        f"overlay={len(plan.overlay_tracks)} subtitle={len(plan.subtitle_tracks)} music={len(plan.music_tracks)}"
    )
    for warning in plan.warnings:
        print(f"warning: {warning.message}")
    print(f"output_path:       {output_path if output_path else '(not written)'}")
    print()


def _print_validation_summary(result: RendererValidation) -> None:
    print()
    print("AIKO Renderer Plan Engine — validation (Phase 11E.1)")
    print("----------------------------------------------------------")
    print(result.summary)
    for error in result.errors:
        print(f"  error:   {error}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_renderer_plan_config(arguments.config)
        timeline_path = Path(arguments.timeline)
        overlay_plan_path = Path(arguments.overlay_plan)

        if arguments.output:
            output_path = Path(arguments.output)
            if output_path.resolve() == timeline_path.resolve():
                raise UnsafeRendererPlanOutputError("--output must not be the same path as --timeline")
            if output_path.resolve() == overlay_plan_path.resolve():
                raise UnsafeRendererPlanOutputError("--output must not be the same path as --overlay-plan")

        plan = build_renderer_plan(
            timeline_path, overlay_plan_path, config, music_plan_path=arguments.music_plan
        )

        if arguments.validate_only:
            timeline = timeline_engine.load_timeline(timeline_path)
            overlay_plan = overlay_plan_engine.load_overlay_plan(overlay_plan_path)
            result = validate_renderer_plan(plan, config, timeline=timeline, overlay_plan=overlay_plan)
            if arguments.as_json:
                print(json.dumps(asdict(result), indent=2, sort_keys=True))
            else:
                _print_validation_summary(result)
            if not result.passed:
                raise SystemExit(1)
            return

        output_path_written: Path | None = None
        if arguments.output:
            output_path_written = save_renderer_plan(plan, arguments.output, force=arguments.force)

        if arguments.as_json:
            print(json.dumps(renderer_plan_to_dict(plan), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_build_summary(plan, output_path_written)
    except RendererPlanEngineError as exc:
        print(f"[RendererPlanEngine] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
