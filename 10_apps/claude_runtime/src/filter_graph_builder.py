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

from . import overlay_asset_resolver, overlay_plan_engine, renderer_plan_engine, subtitle_render_engine

# Phase 11F.2 — Filter Graph Builder. One reusable, execution-free
# planning module: reads an already-validated RendererPlan +
# OverlayPlan (and, optionally, a pre-built
# subtitle_render_engine.SubtitleRenderRequest per subtitle pass) and
# produces a typed FilterGraph describing WHAT filters would run, in
# WHAT order, wired through WHAT labels. This module never builds a
# raw ffmpeg filter string, never touches ffmpeg/ffprobe, never
# probes media, never executes anything, and never imports subprocess.
# FilterSpec.parameters store raw semantic values (text/color/
# position/timing) -- escaping into literal ffmpeg syntax is deferred
# to whichever engine eventually consumes a FilterGraph for real
# execution (a future integration phase, out of scope here). Nothing
# existing is rewired to call this module yet.

DEFAULT_FILTER_GRAPH_CONFIG_RELATIVE_PATH = Path("config") / "video" / "filter_graph.yaml"


def _runtime_root() -> Path:
    """
    filter_graph_builder.py location:
    10_apps/claude_runtime/src/filter_graph_builder.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FilterGraphBuilderError(RuntimeError):
    """Base error for the Phase 11F.2 Filter Graph Builder."""


class FilterGraphConfigError(FilterGraphBuilderError):
    """Raised when config/video/filter_graph.yaml is missing or invalid."""


class FilterGraphRendererPlanLoadError(FilterGraphBuilderError):
    """Raised when --renderer-plan cannot be loaded."""


class FilterGraphOverlayPlanLoadError(FilterGraphBuilderError):
    """Raised when --overlay-plan cannot be loaded."""


class FilterGraphPlanMismatchError(FilterGraphBuilderError):
    """Raised when renderer_plan.overlay_plan_id does not match the
    supplied OverlayPlan's plan_id."""


class UnsupportedFilterTypeError(FilterGraphBuilderError):
    """Raised when a filter_type outside FilterType.ALL is requested of
    a direct API helper (e.g. build_subtitle_filter_spec())."""


class FilterGraphStructureError(FilterGraphBuilderError):
    """Raised only for a genuine build-time structural blocker (e.g. an
    unrecognized pass_type) -- never for policy judgments, which live
    in validate_filter_graph()."""


class FilterGraphJSONError(FilterGraphBuilderError):
    """Raised when filter graph JSON is malformed or missing required fields."""


class FilterGraphOutputExistsError(FilterGraphBuilderError):
    """Raised when --output already exists and --force was not given."""


class UnsafeFilterGraphOutputError(FilterGraphBuilderError):
    """Raised when --output is a directory, or resolves to the same
    path as --renderer-plan or --overlay-plan."""


class FilterGraphOverlayAssetManifestRequiredError(FilterGraphBuilderError):
    """Raised when the renderer plan has active overlay clips but no
    overlay_asset_manifest was supplied -- never falls back to a fake
    placeholder label for a missing manifest."""


class FilterGraphOverlayAssetUnresolvedError(FilterGraphBuilderError):
    """Raised when an active, required overlay's asset is missing from
    the manifest or is not status == RESOLVED."""


# ---------------------------------------------------------------------------
# String-constant "enum"
# ---------------------------------------------------------------------------


class FilterType:
    DRAWTEXT = "drawtext"
    ASS = "ass"
    OVERLAY = "overlay"
    SCALE = "scale"
    ALPHA = "alpha"
    FORMAT = "format"
    SETPTS = "setpts"
    FADE = "fade"
    TRIM = "trim"
    CONCAT = "concat"
    ALL = (DRAWTEXT, ASS, OVERLAY, SCALE, ALPHA, FORMAT, SETPTS, FADE, TRIM, CONCAT)
    # "enable expression" is not its own filter type -- it's the
    # enable_expression field every FilterSpec can carry, validated as
    # its own cross-cutting rule in validate_filter_graph().


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FilterGraphConfig:
    schema_version: str = "1.0"
    builder_version: str = "11F.2"

    width_fallback: int = 1080
    height_fallback: int = 1920

    video_label_prefix: str = "v"
    final_video_label: str = "outv"
    initial_input_label: str = "0:v"

    supported_filter_types: tuple[str, ...] = FilterType.ALL

    z_order_min: int = 0
    z_order_max: int = 1000
    require_known_renderer_hints: bool = True

    overlay_require_all_resolved: bool = True

    overwrite_requires_force: bool = True
    atomic_write: bool = True


def default_filter_graph_config_path() -> Path:
    return _runtime_root() / DEFAULT_FILTER_GRAPH_CONFIG_RELATIVE_PATH


def load_filter_graph_config(config_path: str | Path | None = None) -> FilterGraphConfig:
    """Load config/video/filter_graph.yaml (or an alternate path) into a
    FilterGraphConfig. Raises FilterGraphConfigError if the file is
    missing or invalid."""
    path = Path(config_path) if config_path else default_filter_graph_config_path()

    if not path.exists():
        raise FilterGraphConfigError(f"Filter graph config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise FilterGraphConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise FilterGraphConfigError(f"Filter graph config is empty or invalid: {path}")

    filter_graph_section = raw.get("filter_graph") or {}
    canvas_section = raw.get("canvas") or {}
    labels_section = raw.get("labels") or {}
    validation_section = raw.get("validation") or {}
    overlays_section = raw.get("overlays") or {}
    output_section = raw.get("output") or {}
    supported_filter_types = raw.get("supported_filter_types") or list(FilterType.ALL)

    return FilterGraphConfig(
        schema_version=str(filter_graph_section.get("schema_version", "1.0")),
        builder_version=str(filter_graph_section.get("builder_version", "11F.2")),
        width_fallback=int(canvas_section.get("width_fallback", 1080)),
        height_fallback=int(canvas_section.get("height_fallback", 1920)),
        video_label_prefix=str(labels_section.get("video_label_prefix", "v")),
        final_video_label=str(labels_section.get("final_video_label", "outv")),
        initial_input_label=str(labels_section.get("initial_input_label", "0:v")),
        supported_filter_types=tuple(str(t) for t in supported_filter_types),
        z_order_min=int(validation_section.get("z_order_min", 0)),
        z_order_max=int(validation_section.get("z_order_max", 1000)),
        require_known_renderer_hints=bool(validation_section.get("require_known_renderer_hints", True)),
        overlay_require_all_resolved=bool(overlays_section.get("require_all_active_overlays_resolved", True)),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        atomic_write=bool(output_section.get("atomic_write", True)),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FilterSpec:
    filter_id: str = ""
    filter_type: str = ""
    label_in: list[str] = field(default_factory=list)
    label_out: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    enable_expression: str | None = None
    z_order: int | None = None


@dataclass(slots=True)
class ExtraInputSpec:
    """A real, ffmpeg-consumable extra -i input (Phase 11F.5) -- e.g. an
    overlay image whose resolved path was validated by
    overlay_asset_resolver.py. `label` is a raw numbered stream label
    like "1:v", never a synthetic placeholder."""
    label: str = ""
    resolved_path: str = ""
    asset_id: str = ""
    logical_role: str = ""


@dataclass(slots=True)
class FilterGraphPass:
    pass_id: str = ""
    pass_type: str = ""
    hint_type: str | None = None
    filters: list[FilterSpec] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FilterGraphValidation:
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    filter_count: int = 0
    label_count: int = 0
    pass_count: int = 0
    summary: str = ""


@dataclass(slots=True)
class FilterGraph:
    schema_version: str = "1.0"
    graph_id: str = ""
    renderer_plan_id: str = ""
    overlay_plan_id: str | None = None
    created_at: str = ""
    passes: list[FilterGraphPass] = field(default_factory=list)
    filters: list[FilterSpec] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    extra_inputs: list[ExtraInputSpec] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    validation: FilterGraphValidation = field(default_factory=FilterGraphValidation)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Label allocation — deterministic, sequential.
# ---------------------------------------------------------------------------


class _LabelAllocator:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._counter = 0

    def next(self) -> str:
        label = f"{self._prefix}{self._counter}"
        self._counter += 1
        return label


def _ordered_dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class _ExtraInputAllocator:
    """Allocates real, ffmpeg-consumable extra -i stream labels (e.g.
    "1:v") for overlay image assets, starting from `start_index`.
    Deduplicates by resolved_path so the same image reused across
    multiple overlay clips gets exactly one -i input, not one per
    usage."""

    def __init__(self, start_index: int, video_kind: str = "v") -> None:
        self._next_index = start_index
        self._video_kind = video_kind
        self._by_path: dict[str, ExtraInputSpec] = {}
        self.ordered: list[ExtraInputSpec] = []

    def allocate(self, resolved_path: str, asset_id: str, logical_role: str) -> str:
        existing = self._by_path.get(resolved_path)
        if existing is not None:
            return existing.label
        label = f"{self._next_index}:{self._video_kind}"
        self._next_index += 1
        spec = ExtraInputSpec(
            label=label, resolved_path=resolved_path, asset_id=asset_id, logical_role=logical_role,
        )
        self._by_path[resolved_path] = spec
        self.ordered.append(spec)
        return label


# ---------------------------------------------------------------------------
# Semantic-parameter extraction — shared by the RendererPlan pathway and
# the standalone build_subtitle_filter_spec() helper.
# ---------------------------------------------------------------------------


def _planned_overlay_to_params(overlay: overlay_plan_engine.PlannedOverlay) -> dict[str, Any]:
    style = overlay.style_snapshot
    position = overlay.position
    return {
        "text": overlay.content,
        "start_seconds": round(overlay.timeline_start_seconds, 6),
        "end_seconds": round(overlay.timeline_end_seconds, 6),
        "font_family": style.font_family,
        "font_color": style.primary_color,
        "outline_color": style.outline_color,
        "outline_width": style.outline_width,
        "x": position.x,
        "y": position.y,
        "anchor": position.anchor,
    }


def _cue_to_params(cue: subtitle_render_engine.DrawTextCue) -> dict[str, Any]:
    return {
        "text": cue.text,
        "start_seconds": round(cue.start_seconds, 6),
        "end_seconds": round(cue.end_seconds, 6),
        "font_family": cue.font_family,
        "font_color": cue.font_color,
        "outline_color": cue.outline_color,
        "outline_width": cue.outline_width,
        "x": cue.x,
        "y": cue.y,
        "anchor": cue.anchor,
    }


# ---------------------------------------------------------------------------
# Per-pass-type filter derivation — pure planning, no execution.
# ---------------------------------------------------------------------------


def _count_raw_video_inputs(renderer_plan: renderer_plan_engine.RendererPlan, hint_type: str | None) -> int:
    """
    Returns the number of raw numbered ffmpeg inputs (0:v, 1:v, ...)
    the video pass already claims -- mirrors _derive_video_pass_filters's
    own indexing exactly. Used so overlay image extra-inputs (Phase
    11F.5) never collide with a normalize/transcode pass's per-asset
    raw video inputs.
    """
    if hint_type in (renderer_plan_engine.RendererHintType.NORMALIZE, renderer_plan_engine.RendererHintType.TRANSCODE):
        video_track = renderer_plan.video_tracks[0] if renderer_plan.video_tracks else None
        if video_track is not None and video_track.asset_ids:
            return len(video_track.asset_ids)
    return 1


def _derive_video_pass_filters(
    renderer_plan: renderer_plan_engine.RendererPlan,
    hint_type: str | None,
    config: FilterGraphConfig,
    allocator: _LabelAllocator,
) -> tuple[list[FilterSpec], str]:
    """
    hint 'copy' (or anything but normalize/transcode) needs no filter
    graph at all -- mirrors video_engine.py's own lossless-copy case,
    where concatenation happens via the concat *demuxer*, not a
    filter. hint 'normalize'/'transcode' emits one scale+format-style
    FilterSpec per referenced video asset, then (only when there is
    more than one) a label-wiring-only 'concat' FilterSpec.
    """
    if hint_type not in (renderer_plan_engine.RendererHintType.NORMALIZE, renderer_plan_engine.RendererHintType.TRANSCODE):
        return [], config.initial_input_label

    video_track = renderer_plan.video_tracks[0] if renderer_plan.video_tracks else None
    if video_track is None or not video_track.asset_ids:
        return [], config.initial_input_label

    filters: list[FilterSpec] = []
    per_input_labels: list[str] = []
    for index, asset_id in enumerate(video_track.asset_ids):
        out_label = allocator.next()
        filters.append(
            FilterSpec(
                filter_id=f"filter_scale_{asset_id}",
                filter_type=FilterType.SCALE,
                label_in=[f"{index}:v"],
                label_out=[out_label],
                parameters={
                    "width": renderer_plan.canvas_width or config.width_fallback,
                    "height": renderer_plan.canvas_height or config.height_fallback,
                    "asset_id": asset_id,
                },
            )
        )
        per_input_labels.append(out_label)

    if len(per_input_labels) > 1:
        concat_label = allocator.next()
        filters.append(
            FilterSpec(
                filter_id="filter_concat_video",
                filter_type=FilterType.CONCAT,
                label_in=list(per_input_labels),
                label_out=[concat_label],
                parameters={"n": len(per_input_labels), "v": 1, "a": 0},
            )
        )
        return filters, concat_label

    return filters, per_input_labels[0]


def _derive_subtitle_pass_filters(
    renderer_plan: renderer_plan_engine.RendererPlan,
    overlay_plan: overlay_plan_engine.OverlayPlan,
    hint_type: str | None,
    label_in: str,
    allocator: _LabelAllocator,
    *,
    pass_id: str,
    subtitle_requests: dict[str, subtitle_render_engine.SubtitleRenderRequest] | None,
) -> tuple[list[FilterSpec], str]:
    if not renderer_plan.subtitle_tracks:
        return [], label_in
    subtitle_track = renderer_plan.subtitle_tracks[0]
    request = (subtitle_requests or {}).get(pass_id)

    if hint_type == renderer_plan_engine.RendererHintType.ASS:
        if request is not None and request.ass_path is not None:
            ass_path = str(request.ass_path)
            fontsdir = str(request.fontsdir) if request.fontsdir else None
        else:
            asset_by_id = {asset.asset_id: asset for asset in renderer_plan.assets}
            ass_asset = next(
                (
                    asset_by_id[asset_id]
                    for asset_id in subtitle_track.asset_ids
                    if asset_id in asset_by_id
                    and asset_by_id[asset_id].asset_type == renderer_plan_engine.AssetType.SUBTITLE_FILE
                ),
                None,
            )
            ass_path = ass_asset.source_path if ass_asset else None
            fontsdir = None

        out_label = allocator.next()
        return [
            FilterSpec(
                filter_id=f"filter_ass_{pass_id}",
                filter_type=FilterType.ASS,
                label_in=[label_in],
                label_out=[out_label],
                parameters={"ass_path": ass_path, "fontsdir": fontsdir},
            )
        ], out_label

    if hint_type == renderer_plan_engine.RendererHintType.DRAWTEXT:
        if request is not None and request.cues:
            cue_params = [_cue_to_params(cue) for cue in request.cues]
        else:
            overlay_by_id = {overlay.overlay_id: overlay for overlay in overlay_plan.overlays}
            cue_params = [
                _planned_overlay_to_params(overlay_by_id[overlay_id])
                for overlay_id in subtitle_track.overlay_ids
                if overlay_id in overlay_by_id and overlay_by_id[overlay_id].enabled
            ]

        filters: list[FilterSpec] = []
        current = label_in
        for index, params in enumerate(cue_params):
            out_label = allocator.next()
            filters.append(
                FilterSpec(
                    filter_id=f"filter_drawtext_{pass_id}_{index}",
                    filter_type=FilterType.DRAWTEXT,
                    label_in=[current],
                    label_out=[out_label],
                    parameters=params,
                    enable_expression=f"between(t,{params['start_seconds']},{params['end_seconds']})",
                )
            )
            current = out_label
        return filters, current

    # Unsupported/unknown hint -- no filters derived here; the hint
    # itself is flagged separately by validate_filter_graph() /
    # renderer_execution_engine.py's own pre-flight support check.
    return [], label_in


def _derive_overlay_pass_filters(
    renderer_plan: renderer_plan_engine.RendererPlan,
    overlay_plan: overlay_plan_engine.OverlayPlan,
    label_in: str,
    allocator: _LabelAllocator,
    *,
    overlay_asset_manifest: overlay_asset_resolver.OverlayAssetManifest | None = None,
    extra_input_start_index: int | None = None,
) -> tuple[list[FilterSpec], str, list[ExtraInputSpec], list[str]]:
    """
    Phase 11F.5: overlay filters are planned off a real, validated
    OverlayAssetManifest (Phase 11F.4) -- never off the raw, unresolved
    style_snapshot.asset_reference. Each resolved overlay asset gets a
    real numbered ffmpeg stream label (e.g. "1:v") via extra_input
    allocation, deduplicated by resolved path, instead of the old fake
    "overlay_asset:{overlay_id}" placeholder.
    """
    overlay_ids: list[str] = []
    for track in renderer_plan.overlay_tracks:
        overlay_ids.extend(track.overlay_ids)
    if not overlay_ids:
        return [], label_in, [], []

    overlay_by_id = {overlay.overlay_id: overlay for overlay in overlay_plan.overlays}
    active_overlays = [
        overlay_by_id[overlay_id] for overlay_id in overlay_ids
        if overlay_id in overlay_by_id and overlay_by_id[overlay_id].enabled
    ]
    active_overlays.sort(key=lambda overlay: (overlay.z_index, overlay.overlay_id))
    if not active_overlays:
        return [], label_in, [], []

    if overlay_asset_manifest is None:
        raise FilterGraphOverlayAssetManifestRequiredError(
            "Active overlay clip(s) "
            f"{[o.overlay_id for o in active_overlays]} require an overlay_asset_manifest "
            "(Phase 11F.4 OverlayAssetManifest); none was supplied."
        )

    asset_by_overlay_id: dict[str, overlay_asset_resolver.OverlayRenderAsset] = {}
    for asset in overlay_asset_manifest.assets:
        for overlay_id in asset.source_overlay_ids:
            asset_by_overlay_id[overlay_id] = asset

    input_allocator = _ExtraInputAllocator(extra_input_start_index if extra_input_start_index is not None else 1)
    skipped_overlay_ids: list[str] = []

    filters: list[FilterSpec] = []
    current = label_in
    for overlay in active_overlays:
        asset = asset_by_overlay_id.get(overlay.overlay_id)
        if asset is None or asset.status != overlay_asset_resolver.AssetResolutionStatus.RESOLVED:
            required = asset.required if asset is not None else True
            if not required:
                skipped_overlay_ids.append(overlay.overlay_id)
                continue
            status = asset.status if asset is not None else "not_in_manifest"
            raise FilterGraphOverlayAssetUnresolvedError(
                f"Overlay {overlay.overlay_id!r} requires a resolved asset but status is {status!r}"
            )

        image_label = input_allocator.allocate(asset.resolved_path, asset.asset_id, asset.logical_role)
        effective_overlay_label = image_label

        if overlay.opacity < 1.0:
            alpha_label = allocator.next()
            filters.append(
                FilterSpec(
                    filter_id=f"filter_alpha_{overlay.overlay_id}",
                    filter_type=FilterType.ALPHA,
                    label_in=[image_label],
                    label_out=[alpha_label],
                    parameters={"opacity": overlay.opacity},
                )
            )
            effective_overlay_label = alpha_label

        out_label = allocator.next()
        filters.append(
            FilterSpec(
                filter_id=f"filter_overlay_{overlay.overlay_id}",
                filter_type=FilterType.OVERLAY,
                label_in=[current, effective_overlay_label],
                label_out=[out_label],
                parameters={
                    "x": overlay.position.x,
                    "y": overlay.position.y,
                    "asset_id": asset.asset_id,
                    "resolved_path": asset.resolved_path,
                    "has_alpha": asset.has_alpha,
                    "logical_role": asset.logical_role,
                    "checksum": asset.checksum,
                },
                z_order=overlay.z_index,
            )
        )
        current = out_label

    # A soft-skipped overlay's asset is never fabricated or silently
    # dropped without a trace: skipped_overlay_ids is merged into
    # graph.metadata by build_filter_graph(), and whether that's
    # actually acceptable is a policy judgment left entirely to
    # validate_filter_graph() (config.overlay_require_all_resolved).
    return filters, current, input_allocator.ordered, skipped_overlay_ids


# ---------------------------------------------------------------------------
# Standalone helper — the explicit "SubtitleRenderRequest" input,
# reusable without any RendererPlan/OverlayPlan at all.
# ---------------------------------------------------------------------------


def build_subtitle_filter_spec(
    request: subtitle_render_engine.SubtitleRenderRequest,
    config: FilterGraphConfig,
    *,
    label_in: str | None = None,
) -> list[FilterSpec]:
    """Converts an already-built SubtitleRenderRequest directly into a
    chained list of structured FilterSpecs (one for ass, one per cue
    for drawtext) -- the concrete demonstration that this is ONE
    shared builder, reusable outside the RendererPlan/OverlayPlan
    pathway. Never escapes text/paths; parameters remain semantic."""
    allocator = _LabelAllocator(config.video_label_prefix)
    current = label_in or config.initial_input_label

    if request.mode == subtitle_render_engine.SubtitleRenderMode.ASS:
        out_label = allocator.next()
        return [
            FilterSpec(
                filter_id="filter_ass_standalone",
                filter_type=FilterType.ASS,
                label_in=[current],
                label_out=[out_label],
                parameters={
                    "ass_path": str(request.ass_path) if request.ass_path else None,
                    "fontsdir": str(request.fontsdir) if request.fontsdir else None,
                },
            )
        ]

    if request.mode == subtitle_render_engine.SubtitleRenderMode.DRAWTEXT:
        if not request.cues:
            raise FilterGraphStructureError("build_subtitle_filter_spec: drawtext mode requires at least one cue")
        filters: list[FilterSpec] = []
        for index, cue in enumerate(request.cues):
            out_label = allocator.next()
            params = _cue_to_params(cue)
            filters.append(
                FilterSpec(
                    filter_id=f"filter_drawtext_standalone_{index}",
                    filter_type=FilterType.DRAWTEXT,
                    label_in=[current],
                    label_out=[out_label],
                    parameters=params,
                    enable_expression=f"between(t,{params['start_seconds']},{params['end_seconds']})",
                )
            )
            current = out_label
        return filters

    raise UnsupportedFilterTypeError(f"Unsupported subtitle render mode: {request.mode!r}")


# ---------------------------------------------------------------------------
# Deterministic graph_id
# ---------------------------------------------------------------------------


def _compute_graph_id(graph: FilterGraph) -> str:
    payload = {
        "schema_version": graph.schema_version,
        "renderer_plan_id": graph.renderer_plan_id,
        "overlay_plan_id": graph.overlay_plan_id,
        "passes": [
            {
                "pass_id": p.pass_id,
                "pass_type": p.pass_type,
                "hint_type": p.hint_type,
                "dependencies": list(p.dependencies),
                "filter_ids": [f.filter_id for f in p.filters],
            }
            for p in graph.passes
        ],
        "filters": [
            {
                "filter_id": f.filter_id,
                "filter_type": f.filter_type,
                "label_in": list(f.label_in),
                "label_out": list(f.label_out),
                "parameters": f.parameters,
                "enable_expression": f.enable_expression,
                "z_order": f.z_order,
            }
            for f in graph.filters
        ],
        "labels": list(graph.labels),
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "extra_inputs": [
            {"label": e.label, "resolved_path": e.resolved_path, "asset_id": e.asset_id, "logical_role": e.logical_role}
            for e in graph.extra_inputs
        ],
        "dependencies": {key: list(value) for key, value in graph.dependencies.items()},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Build — planning only. Only raises for a genuine structural blocker.
# ---------------------------------------------------------------------------


def build_filter_graph(
    renderer_plan: renderer_plan_engine.RendererPlan,
    overlay_plan: overlay_plan_engine.OverlayPlan,
    config: FilterGraphConfig,
    *,
    subtitle_requests: dict[str, subtitle_render_engine.SubtitleRenderRequest] | None = None,
    overlay_asset_manifest: overlay_asset_resolver.OverlayAssetManifest | None = None,
) -> FilterGraph:
    if renderer_plan.overlay_plan_id != overlay_plan.plan_id:
        raise FilterGraphPlanMismatchError(
            f"Renderer plan overlay_plan_id {renderer_plan.overlay_plan_id!r} does not match "
            f"overlay plan_id {overlay_plan.plan_id!r}"
        )

    hint_by_pass_id = {hint.target_id: hint for hint in renderer_plan.hints}
    allocator = _LabelAllocator(config.video_label_prefix)
    current_label = config.initial_input_label
    all_filters: list[FilterSpec] = []
    graph_passes: list[FilterGraphPass] = []
    all_extra_inputs: list[ExtraInputSpec] = []
    all_skipped_overlay_ids: list[str] = []
    raw_video_input_count = 1

    for pass_ in renderer_plan.passes:
        if pass_.pass_type not in renderer_plan_engine.RenderPassType.ALL:
            raise FilterGraphStructureError(
                f"Pass {pass_.pass_id} has unrecognized pass_type {pass_.pass_type!r}"
            )

        hint = hint_by_pass_id.get(pass_.pass_id)
        hint_type = hint.hint_type if hint else None

        if pass_.pass_type == renderer_plan_engine.RenderPassType.VIDEO:
            filters, current_label = _derive_video_pass_filters(renderer_plan, hint_type, config, allocator)
            raw_video_input_count = _count_raw_video_inputs(renderer_plan, hint_type)
        elif pass_.pass_type == renderer_plan_engine.RenderPassType.SUBTITLE:
            filters, current_label = _derive_subtitle_pass_filters(
                renderer_plan, overlay_plan, hint_type, current_label, allocator,
                pass_id=pass_.pass_id, subtitle_requests=subtitle_requests,
            )
        elif pass_.pass_type == renderer_plan_engine.RenderPassType.OVERLAY:
            filters, current_label, extra_inputs, skipped_overlay_ids = _derive_overlay_pass_filters(
                renderer_plan, overlay_plan, current_label, allocator,
                overlay_asset_manifest=overlay_asset_manifest, extra_input_start_index=raw_video_input_count,
            )
            all_extra_inputs.extend(extra_inputs)
            all_skipped_overlay_ids.extend(skipped_overlay_ids)
        else:
            filters = []

        all_filters.extend(filters)
        graph_passes.append(
            FilterGraphPass(
                pass_id=pass_.pass_id, pass_type=pass_.pass_type, hint_type=hint_type,
                filters=list(filters), dependencies=[d for d in pass_.dependencies if d],
            )
        )

    if all_filters:
        all_filters[-1].label_out[-1] = config.final_video_label
        final_output_label = config.final_video_label
    else:
        final_output_label = current_label

    produced = {label for f in all_filters for label in f.label_out}
    inputs = _ordered_dedup([label for f in all_filters for label in f.label_in if label not in produced])
    labels = _ordered_dedup(inputs + [label for f in all_filters for label in f.label_out])
    outputs = [final_output_label]
    dependencies = {p.pass_id: list(p.dependencies) for p in graph_passes}

    graph = FilterGraph(
        schema_version=config.schema_version,
        renderer_plan_id=renderer_plan.renderer_plan_id,
        overlay_plan_id=overlay_plan.plan_id,
        passes=graph_passes,
        filters=all_filters,
        labels=labels,
        inputs=inputs,
        outputs=outputs,
        extra_inputs=all_extra_inputs,
        dependencies=dependencies,
        metadata={
            "timeline_id": renderer_plan.timeline_id,
            "canvas_width": renderer_plan.canvas_width,
            "canvas_height": renderer_plan.canvas_height,
            "builder_version": config.builder_version,
            "skipped_overlays": all_skipped_overlay_ids,
        },
    )
    graph.created_at = _now_iso()
    graph.graph_id = _compute_graph_id(graph)
    graph.validation = validate_filter_graph(graph, config)
    return graph


def build_filter_graph_from_filters(
    filters: list[FilterSpec],
    *,
    pass_id: str,
    pass_type: str,
    hint_type: str | None = None,
    renderer_plan_id: str = "",
    overlay_plan_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    config: FilterGraphConfig,
) -> FilterGraph:
    """
    Wraps an already-ordered, already-built filter list (e.g. from
    build_subtitle_filter_spec()) into a single-pass FilterGraph,
    computing labels/inputs/outputs/graph_id/validation with the same
    logic build_filter_graph() applies at the end of its own loop.
    Never reorders or mutates `filters`. Exists so callers that already
    have concrete FilterSpecs (outside the RendererPlan/OverlayPlan
    pathway) can still get the full typed FilterGraph structure without
    re-deriving anything build_filter_graph() itself does not need.
    """
    graph_pass = FilterGraphPass(
        pass_id=pass_id, pass_type=pass_type, hint_type=hint_type,
        filters=list(filters), dependencies=[],
    )

    produced = {label for f in filters for label in f.label_out}
    inputs = _ordered_dedup([label for f in filters for label in f.label_in if label not in produced])
    labels = _ordered_dedup(inputs + [label for f in filters for label in f.label_out])
    final_output_label = filters[-1].label_out[-1] if filters else config.initial_input_label
    outputs = [final_output_label]
    dependencies = {pass_id: []}

    graph = FilterGraph(
        schema_version=config.schema_version,
        renderer_plan_id=renderer_plan_id,
        overlay_plan_id=overlay_plan_id,
        passes=[graph_pass],
        filters=list(filters),
        labels=labels,
        inputs=inputs,
        outputs=outputs,
        dependencies=dependencies,
        metadata=dict(metadata or {}, builder_version=config.builder_version),
    )
    graph.created_at = _now_iso()
    graph.graph_id = _compute_graph_id(graph)
    graph.validation = validate_filter_graph(graph, config)
    return graph


# ---------------------------------------------------------------------------
# Validation — all pass/fail policy lives here, not in build_filter_graph().
# ---------------------------------------------------------------------------


def _detect_pass_cycle(passes: list[FilterGraphPass]) -> bool:
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


def validate_filter_graph(graph: FilterGraph, config: FilterGraphConfig) -> FilterGraphValidation:
    errors: list[str] = []
    warnings: list[str] = []

    filter_ids = [f.filter_id for f in graph.filters]
    if len(filter_ids) != len(set(filter_ids)):
        errors.append("graph.filters: duplicate filter_id values field: filter_id expected unique")

    if len(graph.labels) != len(set(graph.labels)):
        errors.append("graph.labels: duplicate label values field: labels expected unique")

    produced_labels = [label for f in graph.filters for label in f.label_out]
    if len(produced_labels) != len(set(produced_labels)):
        errors.append("graph.filters: duplicate produced label(s) field: label_out expected unique across filters")

    produced_set = set(produced_labels)
    valid_sources = produced_set | set(graph.inputs)
    for f in graph.filters:
        for label in f.label_in:
            if label not in valid_sources:
                errors.append(
                    f"filter {f.filter_id}: label_in={label!r} field: label not produced by any filter "
                    "and not declared in graph.inputs (missing label)"
                )

    extra_input_labels = [e.label for e in graph.extra_inputs]
    if len(extra_input_labels) != len(set(extra_input_labels)):
        errors.append("graph.extra_inputs: duplicate label values field: label expected unique")
    for extra_input in graph.extra_inputs:
        parts = extra_input.label.split(":", 1)
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] not in ("v", "a"):
            errors.append(
                f"graph.extra_inputs: label={extra_input.label!r} field: expected a raw numbered stream "
                "label like '1:v' (invalid extra input label)"
            )

    extra_input_label_set = set(extra_input_labels)
    for f in graph.filters:
        if f.filter_type == FilterType.OVERLAY and len(f.label_in) == 2:
            image_label = f.label_in[1]
            if image_label not in extra_input_label_set:
                errors.append(
                    f"filter {f.filter_id}: label_in={image_label!r} field: not declared in "
                    "graph.extra_inputs (overlay image input not registered)"
                )

    consumed_by_filters = {label for f in graph.filters for label in f.label_in}
    for extra_input in graph.extra_inputs:
        if extra_input.label not in consumed_by_filters:
            warnings.append(
                f"extra_input {extra_input.label!r} (asset_id={extra_input.asset_id!r}): produced but never "
                "consumed by any filter (unused extra input)"
            )

    skipped_overlays = graph.metadata.get("skipped_overlays") or []
    if skipped_overlays and config.overlay_require_all_resolved:
        errors.append(
            f"graph.metadata.skipped_overlays: {skipped_overlays} field: expected empty when "
            "overlay_require_all_resolved is true (unresolved optional overlay asset)"
        )

    for f in graph.filters:
        if f.filter_type not in config.supported_filter_types:
            errors.append(
                f"filter {f.filter_id}: filter_type={f.filter_type!r} field: expected one of "
                f"{config.supported_filter_types} (unsupported filter)"
            )

    for f in graph.filters:
        if f.enable_expression is not None:
            expr = f.enable_expression.strip()
            if not expr or expr.count("(") != expr.count(")"):
                errors.append(
                    f"filter {f.filter_id}: enable_expression={f.enable_expression!r} field: expected "
                    "non-empty with balanced parentheses (invalid enable expression)"
                )

    for f in graph.filters:
        if f.z_order is not None:
            if not isinstance(f.z_order, int) or isinstance(f.z_order, bool):
                errors.append(
                    f"filter {f.filter_id}: z_order={f.z_order!r} field: expected an integer (invalid z-order)"
                )
            elif not (config.z_order_min <= f.z_order <= config.z_order_max):
                errors.append(
                    f"filter {f.filter_id}: z_order={f.z_order} field: expected within "
                    f"[{config.z_order_min}, {config.z_order_max}] (invalid z-order)"
                )

    if config.require_known_renderer_hints:
        for p in graph.passes:
            if p.hint_type is not None and p.hint_type not in renderer_plan_engine.RendererHintType.ALL:
                errors.append(
                    f"pass {p.pass_id}: hint_type={p.hint_type!r} field: expected one of "
                    f"{renderer_plan_engine.RendererHintType.ALL} (unknown renderer hint)"
                )

    pass_id_set = {p.pass_id for p in graph.passes}
    for p in graph.passes:
        for dep in p.dependencies:
            if dep not in pass_id_set:
                errors.append(
                    f"pass {p.pass_id}: dependencies references unknown pass_id {dep!r} (broken dependency)"
                )
    for pass_id, deps in graph.dependencies.items():
        if pass_id not in pass_id_set:
            errors.append(
                f"graph.dependencies: key {pass_id!r} field: does not reference a known pass_id (broken dependency)"
            )
        for dep in deps:
            if dep not in pass_id_set:
                errors.append(
                    f"graph.dependencies[{pass_id!r}]: references unknown pass_id {dep!r} (broken dependency)"
                )

    if _detect_pass_cycle(graph.passes):
        errors.append("graph.passes: dependency cycle detected field: dependencies expected acyclic")

    consumed_labels = {label for f in graph.filters for label in f.label_in}
    output_set = set(graph.outputs)
    for label in sorted(produced_set):
        if label not in consumed_labels and label not in output_set:
            warnings.append(f"label {label!r}: produced but never consumed and not a declared output (unused output)")

    if len(graph.outputs) != 1:
        errors.append(
            f"graph.outputs: expected exactly 1 final output, found {len(graph.outputs)} (multiple final outputs)"
        )

    passed = len(errors) == 0
    summary = (
        f"Filter graph {graph.graph_id or '(no id)'}: {'PASSED' if passed else 'FAILED'} "
        f"({len(errors)} error(s), {len(warnings)} warning(s))"
    )

    return FilterGraphValidation(
        passed=passed,
        errors=errors,
        warnings=warnings,
        filter_count=len(graph.filters),
        label_count=len(graph.labels),
        pass_count=len(graph.passes),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# JSON serialization — stable, round-trip-safe
# ---------------------------------------------------------------------------


def _filter_spec_to_dict(spec: FilterSpec) -> dict[str, Any]:
    return asdict(spec)


def _filter_spec_from_dict(data: dict[str, Any]) -> FilterSpec:
    known = {f.name for f in dataclasses.fields(FilterSpec)}
    try:
        return FilterSpec(**{key: value for key, value in data.items() if key in known})
    except TypeError as exc:
        raise FilterGraphJSONError(f"Malformed FilterSpec in filter graph JSON: {exc}") from exc


def _pass_to_dict(pass_: FilterGraphPass) -> dict[str, Any]:
    return {
        "pass_id": pass_.pass_id,
        "pass_type": pass_.pass_type,
        "hint_type": pass_.hint_type,
        "filters": [_filter_spec_to_dict(f) for f in pass_.filters],
        "dependencies": list(pass_.dependencies),
    }


def _pass_from_dict(data: dict[str, Any]) -> FilterGraphPass:
    return FilterGraphPass(
        pass_id=data.get("pass_id", ""),
        pass_type=data.get("pass_type", ""),
        hint_type=data.get("hint_type"),
        filters=[_filter_spec_from_dict(f) for f in data.get("filters", [])],
        dependencies=list(data.get("dependencies", [])),
    )


def _validation_to_dict(validation: FilterGraphValidation) -> dict[str, Any]:
    return asdict(validation)


def _validation_from_dict(data: dict[str, Any] | None) -> FilterGraphValidation:
    known = {f.name for f in dataclasses.fields(FilterGraphValidation)}
    return FilterGraphValidation(**{key: value for key, value in (data or {}).items() if key in known})


def _extra_input_to_dict(extra_input: ExtraInputSpec) -> dict[str, Any]:
    return asdict(extra_input)


def _extra_input_from_dict(data: dict[str, Any]) -> ExtraInputSpec:
    known = {f.name for f in dataclasses.fields(ExtraInputSpec)}
    try:
        return ExtraInputSpec(**{key: value for key, value in data.items() if key in known})
    except TypeError as exc:
        raise FilterGraphJSONError(f"Malformed ExtraInputSpec in filter graph JSON: {exc}") from exc


def filter_graph_to_dict(graph: FilterGraph) -> dict[str, Any]:
    return {
        "schema_version": graph.schema_version,
        "graph_id": graph.graph_id,
        "renderer_plan_id": graph.renderer_plan_id,
        "overlay_plan_id": graph.overlay_plan_id,
        "created_at": graph.created_at,
        "passes": [_pass_to_dict(p) for p in graph.passes],
        "filters": [_filter_spec_to_dict(f) for f in graph.filters],
        "labels": list(graph.labels),
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "extra_inputs": [_extra_input_to_dict(e) for e in graph.extra_inputs],
        "dependencies": {key: list(value) for key, value in graph.dependencies.items()},
        "validation": _validation_to_dict(graph.validation),
        "metadata": graph.metadata,
    }


def filter_graph_from_dict(data: dict[str, Any]) -> FilterGraph:
    try:
        return FilterGraph(
            schema_version=data.get("schema_version", "1.0"),
            graph_id=data.get("graph_id", ""),
            renderer_plan_id=data.get("renderer_plan_id", ""),
            overlay_plan_id=data.get("overlay_plan_id"),
            created_at=data.get("created_at", ""),
            passes=[_pass_from_dict(p) for p in data.get("passes", [])],
            filters=[_filter_spec_from_dict(f) for f in data.get("filters", [])],
            labels=list(data.get("labels", [])),
            inputs=list(data.get("inputs", [])),
            outputs=list(data.get("outputs", [])),
            extra_inputs=[_extra_input_from_dict(e) for e in data.get("extra_inputs", [])],
            dependencies={key: list(value) for key, value in (data.get("dependencies") or {}).items()},
            validation=_validation_from_dict(data.get("validation")),
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise FilterGraphJSONError(f"Filter graph JSON missing required field: {exc}") from exc
    except (TypeError, AttributeError) as exc:
        raise FilterGraphJSONError(f"Malformed filter graph JSON: {exc}") from exc


def load_filter_graph(path: str | Path) -> FilterGraph:
    path = Path(path)
    if not path.is_file():
        raise FilterGraphJSONError(f"Filter graph file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FilterGraphJSONError(f"Invalid filter graph JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise FilterGraphJSONError(f"Filter graph JSON root must be an object: {path}")

    return filter_graph_from_dict(data)


def save_filter_graph(graph: FilterGraph, path: str | Path, *, force: bool = False) -> Path:
    """Atomically writes filter graph JSON via temp-file + Path.replace()
    so a reader never observes a partially-written file. Refuses to
    overwrite an existing file unless force=True."""
    path = Path(path)

    if path.exists():
        if path.is_dir():
            raise UnsafeFilterGraphOutputError(f"Output path is a directory: {path}")
        if not force:
            raise FilterGraphOutputExistsError(f"{path} already exists; pass force=True to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(filter_graph_to_dict(graph), indent=2, sort_keys=True, ensure_ascii=False)
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
            "AIKO Filter Graph Builder (Phase 11F.2). Builds a deterministic, "
            "execution-free FilterGraph from a validated RendererPlan + "
            "OverlayPlan. Never touches ffmpeg/ffprobe, never executes anything."
        )
    )

    parser.add_argument("--renderer-plan", dest="renderer_plan", required=True, help="Path to the source renderer_plan.json.")
    parser.add_argument("--overlay-plan", dest="overlay_plan", required=True, help="Path to the source overlay_plan.json.")
    parser.add_argument("--output", default=None, help="Path to write the filter graph JSON.")
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/filter_graph.yaml.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument(
        "--validate-only", dest="validate_only", action="store_true",
        help="Build and validate the filter graph only; never writes an output file.",
    )

    arguments = parser.parse_args(argv)

    if arguments.validate_only and arguments.output:
        parser.error("--output cannot be combined with --validate-only.")
    if not arguments.validate_only and not arguments.output:
        parser.error("--output is required unless --validate-only is given.")

    return arguments


def _print_build_summary(graph: FilterGraph, output_path: Path | None) -> None:
    print()
    print("AIKO Filter Graph Builder (Phase 11F.2)")
    print("-------------------------------------------")
    print(f"graph_id:          {graph.graph_id}")
    print(f"renderer_plan_id:  {graph.renderer_plan_id}")
    print(f"overlay_plan_id:   {graph.overlay_plan_id}")
    print(f"passes:            {[p.pass_id for p in graph.passes]}")
    print(f"filters:           {len(graph.filters)}")
    print(f"labels:            {graph.labels}")
    print(f"outputs:           {graph.outputs}")
    print(f"validation:        {graph.validation.summary}")
    print(f"output_path:       {output_path if output_path else '(not written)'}")
    print()


def _print_validation_summary(result: FilterGraphValidation) -> None:
    print()
    print("AIKO Filter Graph Builder — validation (Phase 11F.2)")
    print("------------------------------------------------------------")
    print(result.summary)
    for error in result.errors:
        print(f"  error:   {error}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_filter_graph_config(arguments.config)
        renderer_plan_path = Path(arguments.renderer_plan)
        overlay_plan_path = Path(arguments.overlay_plan)

        if arguments.output:
            output_path = Path(arguments.output)
            if output_path.resolve() == renderer_plan_path.resolve():
                raise UnsafeFilterGraphOutputError("--output must not be the same path as --renderer-plan")
            if output_path.resolve() == overlay_plan_path.resolve():
                raise UnsafeFilterGraphOutputError("--output must not be the same path as --overlay-plan")

        try:
            renderer_plan = renderer_plan_engine.load_renderer_plan(renderer_plan_path)
        except renderer_plan_engine.RendererPlanEngineError as exc:
            raise FilterGraphRendererPlanLoadError(f"Failed to load renderer plan {renderer_plan_path}: {exc}") from exc

        try:
            overlay_plan = overlay_plan_engine.load_overlay_plan(overlay_plan_path)
        except overlay_plan_engine.OverlayPlanEngineError as exc:
            raise FilterGraphOverlayPlanLoadError(f"Failed to load overlay plan {overlay_plan_path}: {exc}") from exc

        graph = build_filter_graph(renderer_plan, overlay_plan, config)

        if arguments.validate_only:
            result = graph.validation
            if arguments.as_json:
                print(json.dumps(asdict(result), indent=2, sort_keys=True))
            else:
                _print_validation_summary(result)
            if not result.passed:
                raise SystemExit(1)
            return

        output_path_written: Path | None = None
        if arguments.output:
            output_path_written = save_filter_graph(graph, arguments.output, force=arguments.force)

        if arguments.as_json:
            print(json.dumps(filter_graph_to_dict(graph), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_build_summary(graph, output_path_written)
    except FilterGraphBuilderError as exc:
        print(f"[FilterGraphBuilder] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
