from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import filter_graph_builder

# Phase 11F.3 — Filter Graph Serializer. Converts an already-validated,
# typed FilterGraph (Phase 11F.2) into real ffmpeg filter syntax: a
# -vf or -filter_complex expression string, plus all escaping and
# label serialization. This module never executes anything, never
# touches ffmpeg/ffprobe, never probes media, never reorders filters
# or passes, and never mutates the FilterGraph it is given. It is the
# ONLY place in this codebase that turns semantic filter parameters
# into literal ffmpeg filtergraph strings -- subtitle_render_engine.py
# no longer implements its own escaping/filter-string construction.

DEFAULT_FILTER_GRAPH_SERIALIZER_CONFIG_RELATIVE_PATH = Path("config") / "video" / "filter_graph_serializer.yaml"

_STREAM_LABEL_PATTERN = re.compile(r"^\d+:[va]$")
_ENABLE_BETWEEN_PATTERN = re.compile(r"^between\(([A-Za-z_][A-Za-z0-9_]*),([^,()]+),([^,()]+)\)$")

_ANCHOR_MAP: dict[str, tuple[str, str]] = {
    "top_left": ("left", "top"),
    "top_center": ("center", "top"),
    "top_right": ("right", "top"),
    "center_left": ("left", "center"),
    "center": ("center", "center"),
    "center_right": ("right", "center"),
    "bottom_left": ("left", "bottom"),
    "bottom_center": ("center", "bottom"),
    "bottom_right": ("right", "bottom"),
}

_GENERIC_FILTER_TYPES = (
    filter_graph_builder.FilterType.SCALE,
    filter_graph_builder.FilterType.FORMAT,
    filter_graph_builder.FilterType.SETPTS,
    filter_graph_builder.FilterType.TRIM,
    filter_graph_builder.FilterType.FADE,
    filter_graph_builder.FilterType.ALPHA,
    filter_graph_builder.FilterType.CONCAT,
)

_KNOWN_FILTER_TYPES = _GENERIC_FILTER_TYPES + (
    filter_graph_builder.FilterType.DRAWTEXT,
    filter_graph_builder.FilterType.ASS,
    filter_graph_builder.FilterType.OVERLAY,
)


def _runtime_root() -> Path:
    """
    filter_graph_serializer.py location:
    10_apps/claude_runtime/src/filter_graph_serializer.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FilterGraphSerializerError(RuntimeError):
    """Base error for the Phase 11F.3 Filter Graph Serializer."""


class FilterGraphSerializerConfigError(FilterGraphSerializerError):
    """Raised when config/video/filter_graph_serializer.yaml is missing or invalid."""


class FilterGraphNotValidatedError(FilterGraphSerializerError):
    """Raised when serialize_filter_graph() is given a FilterGraph whose
    validation.passed is not True -- the serializer never validates on
    the caller's behalf."""


class UnsupportedSerializedFilterError(FilterGraphSerializerError):
    """Raised when a filter_type cannot be serialized and no passthrough
    is configured."""


class FilterGraphLabelSerializationError(FilterGraphSerializerError):
    """Raised for an empty, missing, or invalid-character label."""


class FilterGraphExpressionSerializationError(FilterGraphSerializerError):
    """Raised for an unsupported, malformed, or unsafe enable expression."""


class DrawTextSerializationError(FilterGraphSerializerError):
    """Raised for an invalid drawtext FilterSpec (missing text/position,
    missing required font)."""


class ASSSerializationError(FilterGraphSerializerError):
    """Raised for an invalid ass/subtitles FilterSpec (missing path,
    disallowed fontsdir/force_style)."""


class UnsafeFilterValueError(FilterGraphSerializerError):
    """Raised when a filter parameter value contains an unsafe byte
    (e.g. a NUL character)."""


class MultipleFinalOutputsError(FilterGraphSerializerError):
    """Raised when the graph does not have exactly one final output."""


class FilterGraphSerializationJSONError(FilterGraphSerializerError):
    """Raised for a malformed/unsafe filter graph serialization JSON
    file, or an I/O failure while reading/writing one."""


# ---------------------------------------------------------------------------
# String-constant "enum"
# ---------------------------------------------------------------------------


class FilterGraphOutputMode:
    SIMPLE_VF = "simple_vf"
    FILTER_COMPLEX = "filter_complex"
    ALL = (SIMPLE_VF, FILTER_COMPLEX)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FilterGraphSerializerConfig:
    schema_version: str = "1.0"
    renderer: str = "ffmpeg"
    reject_unknown_filters: bool = True
    allow_custom_passthrough: bool = False
    prefer_simple_vf: bool = True
    force_filter_complex: bool = False

    allow_stream_labels: tuple[str, ...] = ("0:v", "0:a")
    generated_label_pattern: str = r"^[A-Za-z][A-Za-z0-9_:-]*$"
    final_output_label: str = "outv"
    require_single_final_video_output: bool = True

    drawtext_filter_name: str = "drawtext"
    drawtext_preserve_unicode: bool = True
    drawtext_preserve_explicit_line_breaks: bool = True
    drawtext_escape_percent: bool = True
    drawtext_escape_colon: bool = True
    drawtext_escape_apostrophe: bool = True
    drawtext_escape_backslash: bool = True
    drawtext_escape_comma: bool = True
    drawtext_escape_semicolon: bool = True
    drawtext_escape_brackets: bool = True
    drawtext_require_font_file: bool = True
    drawtext_box_enabled_default: bool = False
    drawtext_default_box_color: str = "#00000000"
    drawtext_anchor_margin_pixels: float = 40.0

    ass_filter_name: str = "subtitles"
    ass_allow_fontsdir: bool = True
    ass_allow_force_style: bool = False

    overlay_filter_name: str = "overlay"
    overlay_format_auto_when_alpha: bool = True

    enable_allowed_functions: tuple[str, ...] = ("between",)
    enable_time_variable: str = "t"
    enable_reject_arbitrary_expressions: bool = True

    write_diagnostics: bool = True
    diagnostic_suffix: str = "_filter_graph_serialization.json"
    atomic_write: bool = True


def default_filter_graph_serializer_config_path() -> Path:
    return _runtime_root() / DEFAULT_FILTER_GRAPH_SERIALIZER_CONFIG_RELATIVE_PATH


def load_filter_graph_serializer_config(config_path: str | Path | None = None) -> FilterGraphSerializerConfig:
    """Load config/video/filter_graph_serializer.yaml (or an alternate
    path) into a FilterGraphSerializerConfig. Raises
    FilterGraphSerializerConfigError if the file is missing or invalid."""
    path = Path(config_path) if config_path else default_filter_graph_serializer_config_path()

    if not path.exists():
        raise FilterGraphSerializerConfigError(f"Filter graph serializer config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise FilterGraphSerializerConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise FilterGraphSerializerConfigError(f"Filter graph serializer config is empty or invalid: {path}")

    serializer_section = raw.get("serializer") or {}
    labels_section = raw.get("labels") or {}
    drawtext_section = raw.get("drawtext") or {}
    ass_section = raw.get("ass") or {}
    overlay_section = raw.get("overlay") or {}
    enable_section = raw.get("enable") or {}
    output_section = raw.get("output") or {}

    return FilterGraphSerializerConfig(
        schema_version=str(serializer_section.get("schema_version", "1.0")),
        renderer=str(serializer_section.get("renderer", "ffmpeg")),
        reject_unknown_filters=bool(serializer_section.get("reject_unknown_filters", True)),
        allow_custom_passthrough=bool(serializer_section.get("allow_custom_passthrough", False)),
        prefer_simple_vf=bool(serializer_section.get("prefer_simple_vf", True)),
        force_filter_complex=bool(serializer_section.get("force_filter_complex", False)),
        allow_stream_labels=tuple(str(l) for l in (labels_section.get("allow_stream_labels") or ["0:v", "0:a"])),
        generated_label_pattern=str(labels_section.get("generated_label_pattern", r"^[A-Za-z][A-Za-z0-9_:-]*$")),
        final_output_label=str(labels_section.get("final_output_label", "outv")),
        require_single_final_video_output=bool(labels_section.get("require_single_final_video_output", True)),
        drawtext_filter_name=str(drawtext_section.get("filter_name", "drawtext")),
        drawtext_preserve_unicode=bool(drawtext_section.get("preserve_unicode", True)),
        drawtext_preserve_explicit_line_breaks=bool(drawtext_section.get("preserve_explicit_line_breaks", True)),
        drawtext_escape_percent=bool(drawtext_section.get("escape_percent", True)),
        drawtext_escape_colon=bool(drawtext_section.get("escape_colon", True)),
        drawtext_escape_apostrophe=bool(drawtext_section.get("escape_apostrophe", True)),
        drawtext_escape_backslash=bool(drawtext_section.get("escape_backslash", True)),
        drawtext_escape_comma=bool(drawtext_section.get("escape_comma", True)),
        drawtext_escape_semicolon=bool(drawtext_section.get("escape_semicolon", True)),
        drawtext_escape_brackets=bool(drawtext_section.get("escape_brackets", True)),
        drawtext_require_font_file=bool(drawtext_section.get("require_font_file", True)),
        drawtext_box_enabled_default=bool(drawtext_section.get("box_enabled_default", False)),
        drawtext_default_box_color=str(drawtext_section.get("default_box_color", "#00000000")),
        drawtext_anchor_margin_pixels=float(drawtext_section.get("anchor_margin_pixels", 40.0)),
        ass_filter_name=str(ass_section.get("filter_name", "subtitles")),
        ass_allow_fontsdir=bool(ass_section.get("allow_fontsdir", True)),
        ass_allow_force_style=bool(ass_section.get("allow_force_style", False)),
        overlay_filter_name=str(overlay_section.get("filter_name", "overlay")),
        overlay_format_auto_when_alpha=bool(overlay_section.get("format_auto_when_alpha", True)),
        enable_allowed_functions=tuple(str(f) for f in (enable_section.get("allowed_functions") or ["between"])),
        enable_time_variable=str(enable_section.get("allowed_time_variable", "t")),
        enable_reject_arbitrary_expressions=bool(enable_section.get("reject_arbitrary_expressions", True)),
        write_diagnostics=bool(output_section.get("write_diagnostics", True)),
        diagnostic_suffix=str(output_section.get("diagnostic_suffix", "_filter_graph_serialization.json")),
        atomic_write=bool(output_section.get("atomic_write", True)),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SerializedFilter:
    filter_id: str = ""
    filter_type: str = ""
    serialized: str = ""
    label_in: list[str] = field(default_factory=list)
    label_out: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SerializedLabel:
    label: str = ""
    kind: str = ""  # stream_input | generated | final_output


@dataclass(slots=True)
class SerializedExtraInput:
    """Mirrors filter_graph_builder.ExtraInputSpec -- a real extra -i
    input (e.g. an overlay image) a future renderer must feed to
    ffmpeg for filter_expression's stream labels to resolve. Never an
    argv-escaped string: this is a plain path, since a future command
    builder will place it as its own argv element, not inside a filter
    string."""
    label: str = ""
    resolved_path: str = ""
    asset_id: str = ""
    logical_role: str = ""


@dataclass(slots=True)
class FilterGraphSerializationWarning:
    code: str = ""
    message: str = ""


@dataclass(slots=True)
class SerializedFilterGraph:
    serialization_id: str = ""
    graph_id: str = ""
    renderer: str = "ffmpeg"
    output_mode: str = ""
    filter_expression: str = ""
    filter_argument_name: str = ""
    input_labels: list[str] = field(default_factory=list)
    output_labels: list[str] = field(default_factory=list)
    extra_inputs: list[SerializedExtraInput] = field(default_factory=list)
    final_output_label: str = ""
    filter_count: int = 0
    filters: list[SerializedFilter] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FilterGraphSerializationResult:
    serialized: SerializedFilterGraph = field(default_factory=SerializedFilterGraph)
    written_at: str = ""


# ---------------------------------------------------------------------------
# Escaping — the only place in this codebase that turns semantic values
# into ffmpeg filtergraph syntax for subtitle rendering.
# ---------------------------------------------------------------------------


def escape_filter_value(value: str, config: FilterGraphSerializerConfig) -> str:
    """
    Wraps `value` in single quotes for ffmpeg filtergraph syntax --
    the same convention already proven in
    video_engine.VideoEngine._escape_manifest_path(): single quotes
    make every character literal except the quote itself, escaped via
    ffmpeg's own '\\'' close-escape-reopen sequence. Used for path-like
    values (font files, ass files, fontsdir) and color values.
    """
    if "\x00" in value:
        raise UnsafeFilterValueError("Filter value contains a NUL byte")
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"


def escape_drawtext_text(text: str, config: FilterGraphSerializerConfig) -> str:
    """
    Backslash-escapes drawtext text per ffmpeg's own documented
    per-character escaping mechanism, individually config-toggled.
    Order matters: backslash first, so already-inserted escape
    backslashes are never re-escaped. Never translates, wraps, or
    rewrites the text -- the exact planned text is used. Explicit line
    breaks and Unicode/CJK/emoji pass through untouched.
    """
    if "\x00" in text:
        raise UnsafeFilterValueError("drawtext text contains a NUL byte")

    result = text
    if config.drawtext_escape_backslash:
        result = result.replace("\\", "\\\\")
    if config.drawtext_escape_colon:
        result = result.replace(":", "\\:")
    if config.drawtext_escape_apostrophe:
        result = result.replace("'", "\\'")
    if config.drawtext_escape_comma:
        result = result.replace(",", "\\,")
    if config.drawtext_escape_semicolon:
        result = result.replace(";", "\\;")
    if config.drawtext_escape_brackets:
        result = result.replace("[", "\\[").replace("]", "\\]")
    if config.drawtext_escape_percent:
        result = result.replace("%", "%%")
    return result


# ---------------------------------------------------------------------------
# Label serialization
# ---------------------------------------------------------------------------


def _validate_label(label: str, config: FilterGraphSerializerConfig) -> None:
    if not label:
        raise FilterGraphLabelSerializationError("Label must not be empty")
    if label in config.allow_stream_labels:
        return
    if _STREAM_LABEL_PATTERN.match(label):
        return
    if re.match(config.generated_label_pattern, label):
        return
    raise FilterGraphLabelSerializationError(
        f"Label {label!r} is not a recognized stream label and does not match "
        f"generated_label_pattern {config.generated_label_pattern!r}"
    )


def serialize_label(label: str, config: FilterGraphSerializerConfig) -> str:
    """Validates and returns a label unchanged -- this module never
    invents or rewrites a label to "repair" a graph."""
    _validate_label(label, config)
    return label


def classify_label(label: str, config: FilterGraphSerializerConfig, *, is_final_output: bool = False) -> SerializedLabel:
    _validate_label(label, config)
    if is_final_output:
        kind = "final_output"
    elif label in config.allow_stream_labels or _STREAM_LABEL_PATTERN.match(label):
        kind = "stream_input"
    else:
        kind = "generated"
    return SerializedLabel(label=label, kind=kind)


# ---------------------------------------------------------------------------
# Enable expressions — never evaluated, only validated and re-emitted.
# ---------------------------------------------------------------------------


def serialize_enable_expression(expression: str, config: FilterGraphSerializerConfig) -> str:
    if not expression or not expression.strip():
        raise FilterGraphExpressionSerializationError("Enable expression must not be empty")

    match = _ENABLE_BETWEEN_PATTERN.match(expression.strip())
    if not match:
        raise FilterGraphExpressionSerializationError(f"Unsupported or malformed enable expression: {expression!r}")

    variable, start_str, end_str = match.group(1), match.group(2), match.group(3)

    if "between" not in config.enable_allowed_functions:
        raise FilterGraphExpressionSerializationError("Function 'between' is not in enable.allowed_functions")
    if variable != config.enable_time_variable:
        raise FilterGraphExpressionSerializationError(
            f"Unsupported time variable {variable!r}; expected {config.enable_time_variable!r}"
        )

    try:
        start_val = float(start_str)
        end_val = float(end_str)
    except ValueError as exc:
        raise FilterGraphExpressionSerializationError(f"Enable expression has non-numeric timing: {expression!r}") from exc

    if math.isnan(start_val) or math.isnan(end_val) or math.isinf(start_val) or math.isinf(end_val):
        raise FilterGraphExpressionSerializationError(f"Enable expression has NaN/infinite timing: {expression!r}")
    if start_val < 0 or end_val < 0:
        raise FilterGraphExpressionSerializationError(f"Enable expression has negative timing: {expression!r}")
    if end_val <= start_val:
        raise FilterGraphExpressionSerializationError(f"Enable expression has reversed or zero-length timing: {expression!r}")

    return f"between({variable},{start_val},{end_val})"


# ---------------------------------------------------------------------------
# Anchor resolution — ffmpeg expression *syntax*, so it lives here (moved
# out of subtitle_render_engine.py, which no longer resolves anchors).
# ---------------------------------------------------------------------------


def _anchor_expression(anchor: str, margin: float) -> tuple[str, str]:
    if anchor not in _ANCHOR_MAP:
        raise DrawTextSerializationError(f"Unsupported anchor {anchor!r}; expected one of {sorted(_ANCHOR_MAP)}")
    x_key, y_key = _ANCHOR_MAP[anchor]
    x_expr = {"left": str(margin), "center": "(w-text_w)/2", "right": f"w-text_w-{margin}"}[x_key]
    y_expr = {"top": str(margin), "center": "(h-text_h)/2", "bottom": f"h-text_h-{margin}"}[y_key]
    return x_expr, y_expr


# ---------------------------------------------------------------------------
# Per-filter serialization
# ---------------------------------------------------------------------------


def serialize_drawtext_filter(spec: filter_graph_builder.FilterSpec, config: FilterGraphSerializerConfig) -> str:
    params = spec.parameters
    text = params.get("text")
    if not text:
        raise DrawTextSerializationError(f"drawtext filter {spec.filter_id}: 'text' parameter is required and must be non-empty")

    x = params.get("x")
    y = params.get("y")
    anchor = params.get("anchor")

    if x is not None and y is not None:
        if (not isinstance(x, (int, float)) or isinstance(x, bool)) or (not isinstance(y, (int, float)) or isinstance(y, bool)):
            raise DrawTextSerializationError(f"drawtext filter {spec.filter_id}: x/y must be numeric")
        x_expr, y_expr = str(x), str(y)
    elif x is not None or y is not None:
        raise DrawTextSerializationError(f"drawtext filter {spec.filter_id}: must specify both x and y, or neither (use anchor instead)")
    elif anchor:
        x_expr, y_expr = _anchor_expression(anchor, config.drawtext_anchor_margin_pixels)
    else:
        raise DrawTextSerializationError(f"drawtext filter {spec.filter_id}: no x/y position and no anchor; cannot place text")

    parts = [f"{config.drawtext_filter_name}=text={escape_drawtext_text(text, config)}"]

    font_file = params.get("font_file") or params.get("font_path")
    if font_file:
        parts.append(f"fontfile={escape_filter_value(str(font_file), config)}")
    elif config.drawtext_require_font_file:
        raise DrawTextSerializationError(f"drawtext filter {spec.filter_id}: font_file is required")

    font_size = params.get("font_size", 48)
    parts.append(f"fontsize={font_size}")

    font_color = params.get("font_color") or "#FFFFFF"
    parts.append(f"fontcolor={escape_filter_value(str(font_color), config)}")

    border_color = params.get("border_color") or params.get("outline_color") or "#000000"
    parts.append(f"bordercolor={escape_filter_value(str(border_color), config)}")

    border_width = params.get("border_width")
    if border_width is None:
        border_width = params.get("outline_width", 2)
    parts.append(f"borderw={border_width}")

    parts.append(f"x={x_expr}")
    parts.append(f"y={y_expr}")

    line_spacing = params.get("line_spacing")
    if line_spacing is not None:
        parts.append(f"line_spacing={line_spacing}")

    box_enabled = params.get("box_enabled")
    if box_enabled is None:
        box_enabled = config.drawtext_box_enabled_default
    if box_enabled:
        box_color = params.get("box_color") or config.drawtext_default_box_color
        parts.append("box=1")
        parts.append(f"boxcolor={escape_filter_value(str(box_color), config)}")

    opacity = params.get("opacity")
    if opacity is not None and opacity != 1.0:
        parts.append(f"alpha={opacity}")

    if spec.enable_expression:
        serialized_expr = serialize_enable_expression(spec.enable_expression, config)
        parts.append(f"enable={escape_filter_value(serialized_expr, config)}")

    return ":".join(parts)


def serialize_ass_filter(spec: filter_graph_builder.FilterSpec, config: FilterGraphSerializerConfig) -> str:
    params = spec.parameters
    ass_path = params.get("ass_path")
    if not ass_path:
        raise ASSSerializationError(f"ass filter {spec.filter_id}: 'ass_path' parameter is required")

    parts = [f"{config.ass_filter_name}=filename={escape_filter_value(str(ass_path), config)}"]

    fontsdir = params.get("fontsdir")
    if fontsdir:
        if not config.ass_allow_fontsdir:
            raise ASSSerializationError(f"ass filter {spec.filter_id}: fontsdir provided but ass.allow_fontsdir is false")
        parts.append(f"fontsdir={escape_filter_value(str(fontsdir), config)}")

    force_style = params.get("force_style")
    if force_style:
        if not config.ass_allow_force_style:
            raise ASSSerializationError(f"ass filter {spec.filter_id}: force_style provided but ass.allow_force_style is false")
        parts.append(f"force_style={escape_filter_value(str(force_style), config)}")

    return ":".join(parts)


def serialize_overlay_filter(spec: filter_graph_builder.FilterSpec, config: FilterGraphSerializerConfig) -> str:
    """
    Real overlay filter serialization (Phase 11F.5): label_in[1] must
    already be a genuine ffmpeg stream label (e.g. "1:v") registered in
    graph.extra_inputs by the Filter Graph Builder -- this function
    never resolves, reads, or opens the underlying image itself, it
    only turns already-resolved semantic parameters (from
    overlay_asset_resolver.OverlayRenderAsset, via the builder) into
    ffmpeg filter syntax. When the resolved asset has alpha, emits
    format=auto -- without it ffmpeg's overlay filter silently drops
    the overlay input's alpha channel and renders it opaque.
    """
    params = spec.parameters
    x = params.get("x", 0)
    y = params.get("y", 0)
    body = f"{config.overlay_filter_name}=x={x}:y={y}"
    if params.get("has_alpha") is True and config.overlay_format_auto_when_alpha:
        body += ":format=auto"
    if spec.enable_expression:
        serialized_expr = serialize_enable_expression(spec.enable_expression, config)
        body += f":enable={escape_filter_value(serialized_expr, config)}"
    return body


def _serialize_generic_filter(spec: filter_graph_builder.FilterSpec, config: FilterGraphSerializerConfig) -> str:
    params = spec.parameters
    filter_type = spec.filter_type

    if filter_type == filter_graph_builder.FilterType.SCALE:
        width = params.get("width", -1)
        height = params.get("height", -1)
        return f"scale={width}:{height}"

    if filter_type == filter_graph_builder.FilterType.FORMAT:
        pixel_format = params.get("pixel_format") or params.get("format") or "yuv420p"
        return f"format={pixel_format}"

    if filter_type == filter_graph_builder.FilterType.SETPTS:
        expr = params.get("expr") or "PTS-STARTPTS"
        return f"setpts={expr}"

    if filter_type == filter_graph_builder.FilterType.TRIM:
        pieces = []
        if params.get("start_seconds") is not None:
            pieces.append(f"start={params['start_seconds']}")
        if params.get("end_seconds") is not None:
            pieces.append(f"end={params['end_seconds']}")
        return "trim=" + ":".join(pieces) if pieces else "trim"

    if filter_type == filter_graph_builder.FilterType.FADE:
        fade_type = params.get("type", "in")
        start = params.get("start_seconds", 0)
        duration = params.get("duration_seconds", 1)
        return f"fade=t={fade_type}:st={start}:d={duration}"

    if filter_type == filter_graph_builder.FilterType.ALPHA:
        opacity = params.get("opacity", 1.0)
        return f"colorchannelmixer=aa={opacity}"

    if filter_type == filter_graph_builder.FilterType.CONCAT:
        n = params.get("n", 2)
        v = params.get("v", 1)
        a = params.get("a", 0)
        return f"concat=n={n}:v={v}:a={a}"

    raise UnsupportedSerializedFilterError(f"No generic serialization defined for filter_type {filter_type!r}")


def _serialize_passthrough(spec: filter_graph_builder.FilterSpec, config: FilterGraphSerializerConfig) -> str:
    return f"{spec.filter_type}"


def serialize_filter_spec(spec: filter_graph_builder.FilterSpec, config: FilterGraphSerializerConfig) -> SerializedFilter:
    if spec.filter_type == filter_graph_builder.FilterType.DRAWTEXT:
        body = serialize_drawtext_filter(spec, config)
    elif spec.filter_type == filter_graph_builder.FilterType.ASS:
        body = serialize_ass_filter(spec, config)
    elif spec.filter_type == filter_graph_builder.FilterType.OVERLAY:
        body = serialize_overlay_filter(spec, config)
    elif spec.filter_type in _GENERIC_FILTER_TYPES:
        body = _serialize_generic_filter(spec, config)
    elif config.allow_custom_passthrough:
        body = _serialize_passthrough(spec, config)
    elif config.reject_unknown_filters:
        raise UnsupportedSerializedFilterError(f"filter {spec.filter_id}: unsupported filter_type {spec.filter_type!r}")
    else:
        body = _serialize_passthrough(spec, config)

    return SerializedFilter(
        filter_id=spec.filter_id, filter_type=spec.filter_type, serialized=body,
        label_in=list(spec.label_in), label_out=list(spec.label_out),
    )


# ---------------------------------------------------------------------------
# Output-mode decision and expression assembly
# ---------------------------------------------------------------------------


def _decide_output_mode(graph: filter_graph_builder.FilterGraph, config: FilterGraphSerializerConfig) -> str:
    if config.force_filter_complex:
        return FilterGraphOutputMode.FILTER_COMPLEX

    has_multi_input_filter = any(len(f.label_in) > 1 for f in graph.filters)
    has_overlay_filter = any(f.filter_type == filter_graph_builder.FilterType.OVERLAY for f in graph.filters)

    stream_inputs = {label for label in graph.inputs if _STREAM_LABEL_PATTERN.match(label)}
    has_multiple_streams = len(stream_inputs) > 1

    consumption_counts: dict[str, int] = {}
    for f in graph.filters:
        for label in f.label_in:
            consumption_counts[label] = consumption_counts.get(label, 0) + 1
    has_branching = any(count > 1 for count in consumption_counts.values())

    if has_multi_input_filter or has_overlay_filter or has_multiple_streams or has_branching:
        return FilterGraphOutputMode.FILTER_COMPLEX

    if config.prefer_simple_vf:
        return FilterGraphOutputMode.SIMPLE_VF

    return FilterGraphOutputMode.FILTER_COMPLEX


def _assemble_expression(
    serialized_filters: list[SerializedFilter], output_mode: str, config: FilterGraphSerializerConfig
) -> str:
    if output_mode == FilterGraphOutputMode.SIMPLE_VF:
        return ",".join(f.serialized for f in serialized_filters)

    parts = []
    for f in serialized_filters:
        in_part = "".join(f"[{serialize_label(l, config)}]" for l in f.label_in)
        out_part = "".join(f"[{serialize_label(l, config)}]" for l in f.label_out)
        parts.append(f"{in_part}{f.serialized}{out_part}")
    return ";".join(parts)


# ---------------------------------------------------------------------------
# Top-level serialization
# ---------------------------------------------------------------------------


def _compute_serialization_id(result: SerializedFilterGraph, config: FilterGraphSerializerConfig) -> str:
    payload = {
        "graph_id": result.graph_id,
        "schema_version": config.schema_version,
        "renderer": result.renderer,
        "output_mode": result.output_mode,
        "filter_expression": result.filter_expression,
        "filter_argument_name": result.filter_argument_name,
        "extra_inputs": [
            {"label": e.label, "resolved_path": e.resolved_path, "asset_id": e.asset_id, "logical_role": e.logical_role}
            for e in result.extra_inputs
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validate_serialized_filter_graph(result: SerializedFilterGraph, config: FilterGraphSerializerConfig) -> list[str]:
    warnings: list[str] = []
    if not result.filter_expression:
        warnings.append("filter_expression is empty")
    if result.output_mode not in FilterGraphOutputMode.ALL:
        warnings.append(f"output_mode {result.output_mode!r} is not a recognized output mode")
    if not result.final_output_label:
        warnings.append("final_output_label is empty")
    if result.filter_argument_name not in ("-vf", "-filter_complex"):
        warnings.append(f"filter_argument_name {result.filter_argument_name!r} is unexpected")
    return warnings


def serialize_filter_graph(
    graph: filter_graph_builder.FilterGraph, config: FilterGraphSerializerConfig
) -> SerializedFilterGraph:
    """
    Converts an already-validated FilterGraph into ffmpeg filter
    syntax. Requires graph.validation.passed -- never validates on the
    caller's behalf. Never reorders/rebuilds the graph: pass order,
    filter order, label wiring, dependencies, enable expressions, and
    semantic parameters are all preserved exactly as given.
    """
    if not graph.validation.passed:
        raise FilterGraphNotValidatedError(
            f"FilterGraph {graph.graph_id or '(no id)'} has not passed validation: "
            f"{'; '.join(graph.validation.errors)}"
        )

    if config.require_single_final_video_output and len(graph.outputs) != 1:
        raise MultipleFinalOutputsError(f"Expected exactly 1 final output, found {len(graph.outputs)}")

    for label in graph.inputs:
        _validate_label(label, config)
    for f in graph.filters:
        for label in f.label_in:
            _validate_label(label, config)
        for label in f.label_out:
            _validate_label(label, config)
    for label in graph.outputs:
        _validate_label(label, config)
    for extra_input in graph.extra_inputs:
        _validate_label(extra_input.label, config)

    # Cross-reference check, independent of validate_filter_graph():
    # every label_in must be either produced by some filter's label_out
    # or declared as an external input. This module never "repairs" a
    # graph with a missing source -- it fails clearly instead.
    produced_labels = {label for f in graph.filters for label in f.label_out}
    declared_sources = produced_labels | set(graph.inputs)
    for f in graph.filters:
        for label in f.label_in:
            if label not in declared_sources:
                raise FilterGraphLabelSerializationError(
                    f"filter {f.filter_id}: label_in={label!r} is not produced by any filter "
                    "and not declared in graph.inputs (missing label)"
                )

    output_mode = _decide_output_mode(graph, config)
    serialized_filters = [serialize_filter_spec(f, config) for f in graph.filters]
    filter_expression = _assemble_expression(serialized_filters, output_mode, config)
    filter_argument_name = "-vf" if output_mode == FilterGraphOutputMode.SIMPLE_VF else "-filter_complex"

    final_output_label = graph.outputs[0] if graph.outputs else ""
    label_classifications = [
        asdict(classify_label(label, config, is_final_output=(label == final_output_label)))
        for label in graph.labels
    ]

    extra_inputs = [
        SerializedExtraInput(
            label=e.label, resolved_path=e.resolved_path, asset_id=e.asset_id, logical_role=e.logical_role,
        )
        for e in graph.extra_inputs
    ]

    result = SerializedFilterGraph(
        graph_id=graph.graph_id,
        renderer=config.renderer,
        output_mode=output_mode,
        filter_expression=filter_expression,
        filter_argument_name=filter_argument_name,
        input_labels=list(graph.inputs),
        output_labels=list(graph.outputs),
        extra_inputs=extra_inputs,
        final_output_label=final_output_label,
        filter_count=len(graph.filters),
        filters=serialized_filters,
        warnings=[],
        metadata={"schema_version": config.schema_version, "labels": label_classifications},
    )
    result.warnings = validate_serialized_filter_graph(result, config)

    consumed_labels = {label for f in serialized_filters for label in f.label_in}
    for extra_input in extra_inputs:
        if extra_input.label not in consumed_labels:
            result.warnings.append(
                f"extra_input {extra_input.label!r} (asset_id={extra_input.asset_id!r}): produced but never "
                "consumed by any serialized filter (unused extra input)"
            )

    result.serialization_id = _compute_serialization_id(result, config)
    return result


# ---------------------------------------------------------------------------
# JSON serialization / diagnostics — atomic write, never logs secrets.
# ---------------------------------------------------------------------------


def serialized_filter_graph_to_dict(result: SerializedFilterGraph) -> dict[str, Any]:
    return {
        "serialization_id": result.serialization_id,
        "graph_id": result.graph_id,
        "renderer": result.renderer,
        "output_mode": result.output_mode,
        "filter_expression": result.filter_expression,
        "filter_argument_name": result.filter_argument_name,
        "input_labels": list(result.input_labels),
        "output_labels": list(result.output_labels),
        "extra_inputs": [asdict(e) for e in result.extra_inputs],
        "final_output_label": result.final_output_label,
        "filter_count": result.filter_count,
        "filters": [asdict(f) for f in result.filters],
        "warnings": list(result.warnings),
        "metadata": result.metadata,
    }


def serialized_filter_graph_from_dict(data: dict[str, Any]) -> SerializedFilterGraph:
    try:
        known = {f.name for f in dataclasses.fields(SerializedFilter)}
        filters = [
            SerializedFilter(**{key: value for key, value in entry.items() if key in known})
            for entry in data.get("filters", [])
        ]
        extra_input_known = {f.name for f in dataclasses.fields(SerializedExtraInput)}
        extra_inputs = [
            SerializedExtraInput(**{key: value for key, value in entry.items() if key in extra_input_known})
            for entry in data.get("extra_inputs", [])
        ]
        return SerializedFilterGraph(
            serialization_id=data.get("serialization_id", ""),
            graph_id=data.get("graph_id", ""),
            renderer=data.get("renderer", "ffmpeg"),
            output_mode=data.get("output_mode", ""),
            filter_expression=data.get("filter_expression", ""),
            filter_argument_name=data.get("filter_argument_name", ""),
            input_labels=list(data.get("input_labels", [])),
            output_labels=list(data.get("output_labels", [])),
            extra_inputs=extra_inputs,
            final_output_label=data.get("final_output_label", ""),
            filter_count=data.get("filter_count", 0),
            filters=filters,
            warnings=list(data.get("warnings", [])),
            metadata=data.get("metadata", {}),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise FilterGraphSerializationJSONError(f"Malformed filter graph serialization JSON: {exc}") from exc


def load_filter_graph_serialization(path: str | Path) -> SerializedFilterGraph:
    path = Path(path)
    if not path.is_file():
        raise FilterGraphSerializationJSONError(f"Filter graph serialization file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FilterGraphSerializationJSONError(f"Invalid filter graph serialization JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise FilterGraphSerializationJSONError(f"Filter graph serialization JSON root must be an object: {path}")

    return serialized_filter_graph_from_dict(data)


def save_filter_graph_serialization(result: SerializedFilterGraph, path: str | Path, *, force: bool = False) -> Path:
    """Atomically writes serialization JSON via temp-file + Path.replace()
    so a reader never observes a partially-written file. Refuses to
    overwrite an existing file unless force=True. Never logs tokens,
    env vars, cookies, or font binary content -- only semantic filter
    parameters and syntax strings."""
    path = Path(path)

    if path.exists():
        if path.is_dir():
            raise FilterGraphSerializationJSONError(f"Output path is a directory: {path}")
        if not force:
            raise FilterGraphSerializationJSONError(f"{path} already exists; pass force=True to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(serialized_filter_graph_to_dict(result), indent=2, sort_keys=True, ensure_ascii=False)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
    return path


def write_filter_graph_serialization_diagnostics(
    result: SerializedFilterGraph, path: str | Path, *, force: bool = False
) -> FilterGraphSerializationResult:
    save_filter_graph_serialization(result, path, force=force)
    return FilterGraphSerializationResult(serialized=result, written_at=_now_iso())
