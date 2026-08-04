from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import timeline_engine

# Phase 11D — Subtitle Engine. A reusable, platform-neutral subtitle-
# track planner: creates, validates, serializes, and exports subtitle
# tracks (JSON/SRT/ASS) that can later be attached to Timeline Engine
# and rendered by a future overlay/render pipeline. This module never
# renders subtitles, never burns them into video, never calls ffmpeg or
# ffprobe, never transcribes speech, never publishes or uploads, never
# modifies media files, and never downloads fonts or subtitle assets.
# Only reads (never writes) timeline_engine.load_timeline()/
# validate_timeline() — Timeline Engine itself is untouched this phase.

DEFAULT_SUBTITLE_CONFIG_RELATIVE_PATH = Path("config") / "video" / "subtitles.yaml"


def _runtime_root() -> Path:
    """
    subtitle_engine.py location: 10_apps/claude_runtime/src/subtitle_engine.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SubtitleEngineError(RuntimeError):
    """Base error for the Phase 11D Subtitle Engine."""


class SubtitleConfigError(SubtitleEngineError):
    """Raised when config/video/subtitles.yaml is missing or invalid."""


class SubtitleInputError(SubtitleEngineError):
    """Raised for a malformed/unrecognized/incomplete subtitle source."""


class SubtitleJSONError(SubtitleEngineError):
    """Raised when subtitle document JSON is malformed or missing required fields."""


class SubtitleSRTError(SubtitleEngineError):
    """Raised when an imported .srt file is malformed."""


class SubtitleValidationError(SubtitleEngineError):
    """Raised when planning-time text cannot fit within configured line limits
    and overflow_policy is 'fail'."""


class SubtitleTimingError(SubtitleEngineError):
    """Reserved for callers that want a hard timing failure raised directly
    rather than inspected via SubtitleValidationResult."""


class SubtitleOverlapError(SubtitleEngineError):
    """Reserved for callers that want a hard overlap failure raised directly
    rather than inspected via SubtitleValidationResult."""


class SubtitleStyleError(SubtitleEngineError):
    """Raised when a cue/track references an unknown style_id or an unknown
    style preset name."""


class SubtitleOutputExistsError(SubtitleEngineError):
    """Raised when an export target already exists and force was not given."""


class UnsafeSubtitleOutputError(SubtitleEngineError):
    """Raised when an export target path is a directory."""


class TimelineCompatibilityError(SubtitleEngineError):
    """Raised for any --timeline failure: missing/invalid timeline file, a
    subtitle document duration exceeding the Timeline's duration, or a
    timeline_metadata source referencing an unknown subtitle track."""


# ---------------------------------------------------------------------------
# String-constant "enums"
# ---------------------------------------------------------------------------


class SubtitleFormat:
    JSON = "json"
    SRT = "srt"
    ASS = "ass"
    ALL = (JSON, SRT, ASS)


class SubtitleAlignment:
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    MIDDLE_LEFT = "middle_left"
    MIDDLE_CENTER = "middle_center"
    MIDDLE_RIGHT = "middle_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
    ALL = (
        TOP_LEFT, TOP_CENTER, TOP_RIGHT,
        MIDDLE_LEFT, MIDDLE_CENTER, MIDDLE_RIGHT,
        BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT,
    )


class LineBreakMode:
    AUTO = "auto"
    PRESERVE = "preserve"
    ALL = (AUTO, PRESERVE)


# ASS numpad alignment (1-9), matching SubtitleAlignment.ALL order.
_ASS_ALIGNMENT_NUMPAD: dict[str, int] = {
    SubtitleAlignment.BOTTOM_LEFT: 1,
    SubtitleAlignment.BOTTOM_CENTER: 2,
    SubtitleAlignment.BOTTOM_RIGHT: 3,
    SubtitleAlignment.MIDDLE_LEFT: 4,
    SubtitleAlignment.MIDDLE_CENTER: 5,
    SubtitleAlignment.MIDDLE_RIGHT: 6,
    SubtitleAlignment.TOP_LEFT: 7,
    SubtitleAlignment.TOP_CENTER: 8,
    SubtitleAlignment.TOP_RIGHT: 9,
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SubtitlePosition:
    x: float | None = None
    y: float | None = None
    alignment: str = SubtitleAlignment.BOTTOM_CENTER
    safe_area_enabled: bool = True


@dataclass(slots=True)
class SubtitleStyle:
    style_id: str = "default"
    font_family: str = "Arial"
    font_size: int = 54
    font_weight: int = 400
    italic: bool = False
    underline: bool = False
    primary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    background_color: str = "#00000000"
    outline_width: float = 3.0
    shadow_depth: float = 1.0
    margin_left: int = 0
    margin_right: int = 0
    margin_vertical: int = 0
    alignment: str = SubtitleAlignment.BOTTOM_CENTER
    position_x: float | None = None
    position_y: float | None = None
    safe_area_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubtitleCue:
    cue_id: str = ""
    track_id: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    duration_seconds: float = 0.0
    text: str = ""
    style_id: str = "default"
    position: SubtitlePosition | None = None
    alignment: str | None = None
    line_break_mode: str = LineBreakMode.AUTO
    speaker: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubtitleTrack:
    track_id: str = ""
    language: str = "en"
    enabled: bool = True
    default_style_id: str = "default"
    cues: list[SubtitleCue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubtitleWarning:
    code: str = ""
    message: str = ""
    track_id: str | None = None
    cue_id: str | None = None


@dataclass(slots=True)
class SubtitleDocument:
    schema_version: str = "1.0"
    document_id: str = ""
    created_at: str = ""
    language: str = "en"
    duration_seconds: float = 0.0
    tracks: list[SubtitleTrack] = field(default_factory=list)
    styles: dict[str, SubtitleStyle] = field(default_factory=dict)
    timeline_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[SubtitleWarning] = field(default_factory=list)


@dataclass(slots=True)
class SubtitleValidationResult:
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cue_count: int = 0
    track_count: int = 0
    duration_seconds: float = 0.0
    language: str = "en"
    summary: str = ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SubtitleConfig:
    schema_version: str = "1.0"
    default_language: str = "en"
    allow_overlaps: bool = False
    timing_tolerance_seconds: float = 0.001
    minimum_cue_duration_seconds: float = 0.30
    maximum_cue_duration_seconds: float = 7.00
    minimum_gap_seconds: float = 0.00
    frame_rate_snap_enabled: bool = False
    default_frame_rate: float = 30

    trim_whitespace: bool = True
    normalize_repeated_spaces: bool = True
    preserve_explicit_line_breaks: bool = True
    maximum_characters_per_line: int = 32
    maximum_lines_per_cue: int = 2
    overflow_policy: str = "warning"
    cjk_character_wrap: bool = True

    canvas_width: int = 1080
    canvas_height: int = 1920
    safe_area_top: int = 160
    safe_area_bottom: int = 320
    safe_area_left: int = 80
    safe_area_right: int = 80
    default_position_x: float = 540
    default_position_y: float = 1450

    styles: dict[str, SubtitleStyle] = field(default_factory=dict)

    overwrite_requires_force: bool = True
    atomic_write: bool = True
    srt_encoding: str = "utf-8"
    ass_encoding: str = "utf-8"


def default_subtitle_config_path() -> Path:
    return _runtime_root() / DEFAULT_SUBTITLE_CONFIG_RELATIVE_PATH


def load_subtitle_config(config_path: str | Path | None = None) -> SubtitleConfig:
    """Load config/video/subtitles.yaml (or an alternate path) into a
    SubtitleConfig. Raises SubtitleConfigError if the file is missing or
    invalid. Never touches ffmpeg/ffprobe/fonts/media."""
    path = Path(config_path) if config_path else default_subtitle_config_path()

    if not path.exists():
        raise SubtitleConfigError(f"Subtitle config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SubtitleConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise SubtitleConfigError(f"Subtitle config is empty or invalid: {path}")

    subtitle_section = raw.get("subtitle") or {}
    text_section = raw.get("text") or {}
    canvas_section = raw.get("canvas") or {}
    safe_area_section = canvas_section.get("safe_area") or {}
    default_position_section = canvas_section.get("default_position") or {}
    styles_section = raw.get("styles") or {}
    export_section = raw.get("export") or {}

    styles: dict[str, SubtitleStyle] = {}
    for style_id, style_data in styles_section.items():
        style_data = style_data or {}
        styles[style_id] = SubtitleStyle(
            style_id=style_id,
            font_family=str(style_data.get("font_family", "Arial")),
            font_size=int(style_data.get("font_size", 54)),
            font_weight=int(style_data.get("font_weight", 400)),
            italic=bool(style_data.get("italic", False)),
            underline=bool(style_data.get("underline", False)),
            primary_color=str(style_data.get("primary_color", "#FFFFFF")),
            outline_color=str(style_data.get("outline_color", "#000000")),
            background_color=str(style_data.get("background_color", "#00000000")),
            outline_width=float(style_data.get("outline_width", 3)),
            shadow_depth=float(style_data.get("shadow_depth", 1)),
            margin_left=int(style_data.get("margin_left", 0)),
            margin_right=int(style_data.get("margin_right", 0)),
            margin_vertical=int(style_data.get("margin_vertical", 0)),
            alignment=str(style_data.get("alignment", SubtitleAlignment.BOTTOM_CENTER)),
        )

    if "default" not in styles:
        styles["default"] = SubtitleStyle(style_id="default")

    return SubtitleConfig(
        schema_version=str(subtitle_section.get("schema_version", "1.0")),
        default_language=str(subtitle_section.get("default_language", "en")),
        allow_overlaps=bool(subtitle_section.get("allow_overlaps", False)),
        timing_tolerance_seconds=float(subtitle_section.get("timing_tolerance_seconds", 0.001)),
        minimum_cue_duration_seconds=float(subtitle_section.get("minimum_cue_duration_seconds", 0.30)),
        maximum_cue_duration_seconds=float(subtitle_section.get("maximum_cue_duration_seconds", 7.00)),
        minimum_gap_seconds=float(subtitle_section.get("minimum_gap_seconds", 0.00)),
        frame_rate_snap_enabled=bool(subtitle_section.get("frame_rate_snap_enabled", False)),
        default_frame_rate=float(subtitle_section.get("default_frame_rate", 30)),
        trim_whitespace=bool(text_section.get("trim_whitespace", True)),
        normalize_repeated_spaces=bool(text_section.get("normalize_repeated_spaces", True)),
        preserve_explicit_line_breaks=bool(text_section.get("preserve_explicit_line_breaks", True)),
        maximum_characters_per_line=int(text_section.get("maximum_characters_per_line", 32)),
        maximum_lines_per_cue=int(text_section.get("maximum_lines_per_cue", 2)),
        overflow_policy=str(text_section.get("overflow_policy", "warning")),
        cjk_character_wrap=bool(text_section.get("cjk_character_wrap", True)),
        canvas_width=int(canvas_section.get("width", 1080)),
        canvas_height=int(canvas_section.get("height", 1920)),
        safe_area_top=int(safe_area_section.get("top", 160)),
        safe_area_bottom=int(safe_area_section.get("bottom", 320)),
        safe_area_left=int(safe_area_section.get("left", 80)),
        safe_area_right=int(safe_area_section.get("right", 80)),
        default_position_x=float(default_position_section.get("x", 540)),
        default_position_y=float(default_position_section.get("y", 1450)),
        styles=styles,
        overwrite_requires_force=bool(export_section.get("overwrite_requires_force", True)),
        atomic_write=bool(export_section.get("atomic_write", True)),
        srt_encoding=str(export_section.get("srt_encoding", "utf-8")),
        ass_encoding=str(export_section.get("ass_encoding", "utf-8")),
    )


# ---------------------------------------------------------------------------
# Text normalization (requirement 6) — never translates, censors, or
# summarizes; only trims/collapses whitespace per config.
# ---------------------------------------------------------------------------


def normalize_text(text: str, config: SubtitleConfig) -> str:
    if config.preserve_explicit_line_breaks:
        lines = text.split("\n")
    else:
        lines = [text.replace("\n", " ")]

    processed: list[str] = []
    for line in lines:
        if config.trim_whitespace:
            line = line.strip()
        if config.normalize_repeated_spaces:
            line = re.sub(r"[ \t]+", " ", line)
        processed.append(line)
    return "\n".join(processed)


# ---------------------------------------------------------------------------
# Line wrapping (requirement 7) — deterministic planning only, never
# rewrites cue.text; writes the plan into cue.metadata["planned_lines"].
# ---------------------------------------------------------------------------


_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7A3),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
)


def _is_cjk_char(character: str) -> bool:
    code = ord(character)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _line_is_cjk(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    cjk_count = sum(1 for c in letters if _is_cjk_char(c))
    return cjk_count / len(letters) > 0.5


def _wrap_plain_line(line: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(line) <= max_chars:
        return [line]

    words = line.split(" ")
    wrapped: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            wrapped.append(current)
        if len(word) > max_chars:
            for start in range(0, len(word), max_chars):
                wrapped.append(word[start:start + max_chars])
            current = ""
        else:
            current = word
    if current:
        wrapped.append(current)
    return wrapped or [""]


def _wrap_cjk_line(line: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [line]
    return [line[start:start + max_chars] for start in range(0, len(line), max_chars)] or [""]


def plan_cue_line_wrap(
    text: str, config: SubtitleConfig, *, line_break_mode: str = LineBreakMode.AUTO
) -> tuple[list[str], list[SubtitleWarning]]:
    """
    Pure planning helper — never mutates cue.text. Explicit '\\n' breaks
    always stay separate lines. line_break_mode == PRESERVE disables
    auto-wrapping entirely (only explicit breaks are honored). Otherwise
    a line over maximum_characters_per_line wraps on whitespace, or —
    when cjk_character_wrap is enabled and the line is CJK-majority —
    character-by-character. Raises SubtitleValidationError when the
    result exceeds maximum_lines_per_cue and overflow_policy is 'fail';
    otherwise returns a SubtitleWarning for the caller to record.
    """
    explicit_lines = text.split("\n") if config.preserve_explicit_line_breaks else [text.replace("\n", " ")]

    planned: list[str] = []
    for line in explicit_lines:
        if line_break_mode == LineBreakMode.PRESERVE or len(line) <= config.maximum_characters_per_line:
            planned.append(line)
            continue
        if config.cjk_character_wrap and _line_is_cjk(line):
            planned.extend(_wrap_cjk_line(line, config.maximum_characters_per_line))
        else:
            planned.extend(_wrap_plain_line(line, config.maximum_characters_per_line))

    warnings: list[SubtitleWarning] = []
    if len(planned) > config.maximum_lines_per_cue:
        message = (
            f"text wraps to {len(planned)} line(s), exceeding "
            f"maximum_lines_per_cue={config.maximum_lines_per_cue}"
        )
        if config.overflow_policy == "fail":
            raise SubtitleValidationError(message)
        warnings.append(SubtitleWarning(code="line_overflow", message=message))

    return planned, warnings


# ---------------------------------------------------------------------------
# Safe-area / position planning (requirement 8) — planned metadata only.
# ---------------------------------------------------------------------------


def plan_cue_position(
    style: SubtitleStyle,
    config: SubtitleConfig,
    *,
    cue_position: SubtitlePosition | None = None,
    cue_alignment: str | None = None,
) -> SubtitlePosition:
    alignment = cue_alignment or style.alignment
    if cue_position is not None:
        return SubtitlePosition(
            x=cue_position.x,
            y=cue_position.y,
            alignment=cue_alignment or cue_position.alignment,
            safe_area_enabled=cue_position.safe_area_enabled,
        )
    if style.position_x is not None and style.position_y is not None:
        return SubtitlePosition(
            x=style.position_x, y=style.position_y, alignment=alignment, safe_area_enabled=style.safe_area_enabled
        )
    return SubtitlePosition(
        x=config.default_position_x,
        y=config.default_position_y,
        alignment=alignment,
        safe_area_enabled=style.safe_area_enabled,
    )


def validate_position_within_safe_area(position: SubtitlePosition, config: SubtitleConfig) -> list[str]:
    """Read-only check — returns warning strings, never raises. A planned
    position outside the configured safe area is a caution, not a hard
    failure, since no rendering happens in this phase."""
    if not position.safe_area_enabled or position.x is None or position.y is None:
        return []

    warnings: list[str] = []
    top = config.safe_area_top
    bottom = config.canvas_height - config.safe_area_bottom
    left = config.safe_area_left
    right = config.canvas_width - config.safe_area_right

    if position.y < top or position.y > bottom:
        warnings.append(f"position y={position.y} falls outside the vertical safe area [{top}, {bottom}]")
    if position.x < left or position.x > right:
        warnings.append(f"position x={position.x} falls outside the horizontal safe area [{left}, {right}]")
    return warnings


# ---------------------------------------------------------------------------
# Frame-rate snapping (requirement 4)
# ---------------------------------------------------------------------------


def snap_to_frame_rate(seconds: float, frame_rate: float) -> float:
    if frame_rate <= 0:
        return seconds
    frame_duration = 1.0 / frame_rate
    return round(seconds / frame_duration) * frame_duration


# ---------------------------------------------------------------------------
# Style validation
# ---------------------------------------------------------------------------


_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$")


def _validate_style(style: SubtitleStyle) -> list[str]:
    errors: list[str] = []
    for field_name, value in (
        ("primary_color", style.primary_color),
        ("outline_color", style.outline_color),
        ("background_color", style.background_color),
    ):
        if not _COLOR_PATTERN.match(value):
            errors.append(f"{field_name} {value!r} is not a valid #RRGGBB or #RRGGBBAA color")
    if style.font_size <= 0:
        errors.append(f"font_size must be > 0, got {style.font_size}")
    if style.margin_left < 0 or style.margin_right < 0 or style.margin_vertical < 0:
        errors.append("margins must be >= 0")
    if style.outline_width < 0:
        errors.append(f"outline_width must be >= 0, got {style.outline_width}")
    if style.shadow_depth < 0:
        errors.append(f"shadow_depth must be >= 0, got {style.shadow_depth}")
    if style.alignment not in SubtitleAlignment.ALL:
        errors.append(f"alignment {style.alignment!r} is not a recognized SubtitleAlignment value")
    return errors


# ---------------------------------------------------------------------------
# SRT parsing (input format C) — no speech recognition, purely a text/
# timestamp parser over an already-authored .srt file.
# ---------------------------------------------------------------------------


_SRT_TIMESTAMP_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
_SRT_ARROW_PATTERN = re.compile(r"^(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")


def _parse_srt_timestamp(raw: str) -> float:
    match = _SRT_TIMESTAMP_PATTERN.match(raw.strip())
    if not match:
        raise SubtitleSRTError(f"Invalid SRT timestamp: {raw!r}")
    hours, minutes, seconds, millis = (int(group) for group in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def parse_srt(content: str, config: SubtitleConfig, *, track_id: str = "track_srt_import") -> list[SubtitleCue]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block for block in re.split(r"\n\s*\n", normalized.strip()) if block.strip()]
    if not blocks:
        raise SubtitleSRTError("SRT source contains no subtitle blocks")

    cues: list[SubtitleCue] = []
    for index, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 2:
            raise SubtitleSRTError(
                f"Malformed SRT block {index}: expected an index, a timestamp line, and text"
            )

        timestamp_line = lines[1]
        text_lines = lines[2:]
        match = _SRT_ARROW_PATTERN.match(timestamp_line.strip())
        if not match:
            raise SubtitleSRTError(f"Malformed SRT timestamp line in block {index}: {timestamp_line!r}")

        start = _parse_srt_timestamp(match.group(1))
        end = _parse_srt_timestamp(match.group(2))
        if end <= start:
            raise SubtitleSRTError(f"SRT block {index} has end <= start ({end} <= {start})")

        text = normalize_text("\n".join(text_lines), config)
        if not text.strip():
            raise SubtitleSRTError(f"SRT block {index} has no text")

        cues.append(
            SubtitleCue(
                cue_id=f"cue_{index:04d}",
                track_id=track_id,
                start_seconds=start,
                end_seconds=end,
                duration_seconds=end - start,
                text=text,
                style_id="default",
            )
        )
    return cues


# ---------------------------------------------------------------------------
# Input formats A/B/D — plain entries JSON, existing plan JSON, Timeline
# metadata source. Never infers timing from audio/video.
# ---------------------------------------------------------------------------


def _build_document_from_entries(data: dict[str, Any], config: SubtitleConfig) -> SubtitleDocument:
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SubtitleInputError("Source JSON 'entries' must be a non-empty list")

    language = str(data.get("language") or config.default_language)
    default_style_name = str(data.get("style") or "default")
    if default_style_name not in config.styles:
        raise SubtitleStyleError(f"Unknown style preset: {default_style_name!r}")

    track = SubtitleTrack(track_id="track_subtitles", language=language, default_style_id=default_style_name)
    referenced_style_ids = {default_style_name}

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise SubtitleInputError(f"Entry {index} must be an object")

        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SubtitleInputError(f"Entry {index} is missing non-empty 'text'")

        start = entry.get("start_seconds")
        end = entry.get("end_seconds")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            raise SubtitleInputError(f"Entry {index} is missing a numeric 'start_seconds'")
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            raise SubtitleInputError(f"Entry {index} is missing a numeric 'end_seconds'")

        style_id = str(entry.get("style_id") or default_style_name)
        if style_id not in config.styles:
            raise SubtitleStyleError(f"Entry {index} references unknown style_id {style_id!r}")
        referenced_style_ids.add(style_id)

        track.cues.append(
            SubtitleCue(
                cue_id=f"cue_{index:04d}",
                track_id=track.track_id,
                start_seconds=float(start),
                end_seconds=float(end),
                duration_seconds=float(end) - float(start),
                text=text,
                style_id=style_id,
                speaker=entry.get("speaker"),
                metadata=dict(entry.get("metadata") or {}),
            )
        )

    styles = {style_id: config.styles[style_id] for style_id in sorted(referenced_style_ids)}
    return SubtitleDocument(schema_version=config.schema_version, language=language, tracks=[track], styles=styles)


def _detect_source_kind(path: Path, data: Any) -> str:
    if path.suffix.lower() == ".srt":
        return "srt"
    if isinstance(data, dict):
        if data.get("source_type") == "timeline_metadata":
            return "timeline_metadata"
        if "tracks" in data and "styles" in data:
            return "plan"
        if "entries" in data:
            return "entries"
    raise SubtitleInputError(
        "Unrecognized subtitle source: expected an 'entries', existing-plan, or "
        "'timeline_metadata' JSON object, or a .srt file"
    )


def load_subtitle_source(
    path: Path, config: SubtitleConfig, *, timeline: timeline_engine.Timeline | None = None
) -> SubtitleDocument:
    if path.suffix.lower() == ".srt":
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SubtitleInputError(f"Could not read {path}: {exc}") from exc
        cues = parse_srt(content, config)
        language = config.default_language
        track = SubtitleTrack(track_id="track_srt_import", language=language, default_style_id="default")
        track.cues = cues
        return SubtitleDocument(
            schema_version=config.schema_version,
            language=language,
            tracks=[track],
            styles={"default": config.styles.get("default", SubtitleStyle())},
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubtitleJSONError(f"Invalid subtitle source JSON in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SubtitleInputError(f"Subtitle source JSON root must be an object: {path}")

    kind = _detect_source_kind(path, raw)

    if kind == "entries":
        return _build_document_from_entries(raw, config)

    if kind == "plan":
        return document_from_dict(raw)

    # kind == "timeline_metadata"
    if timeline is None:
        raise SubtitleInputError("A timeline_metadata source requires --timeline to be supplied")

    track_id = raw.get("track_id")
    if not track_id:
        raise SubtitleInputError("timeline_metadata source must specify 'track_id'")

    matching = next(
        (
            t
            for t in timeline.tracks
            if t.track_id == track_id and t.track_type == timeline_engine.TrackType.SUBTITLE
        ),
        None,
    )
    if matching is None:
        raise TimelineCompatibilityError(f"Timeline has no subtitle track with track_id {track_id!r}")

    cues_data = matching.metadata.get("cues")
    if not isinstance(cues_data, list) or not cues_data:
        raise SubtitleInputError(f"Timeline subtitle track {track_id!r} metadata has no 'cues'")

    entries_payload = {"language": matching.metadata.get("language", config.default_language), "entries": cues_data}
    document = _build_document_from_entries(entries_payload, config)
    document.tracks[0].track_id = str(track_id)
    for cue in document.tracks[0].cues:
        cue.track_id = str(track_id)
    return document


# ---------------------------------------------------------------------------
# Deterministic document_id
# ---------------------------------------------------------------------------


def _compute_document_id(document: SubtitleDocument) -> str:
    payload = {
        "schema_version": document.schema_version,
        "language": document.language,
        "duration_seconds": round(document.duration_seconds, 6),
        "timeline_id": document.timeline_id,
        "tracks": [
            {
                "track_id": track.track_id,
                "language": track.language,
                "enabled": track.enabled,
                "default_style_id": track.default_style_id,
                "cues": [
                    {
                        "cue_id": cue.cue_id,
                        "start_seconds": round(cue.start_seconds, 6),
                        "end_seconds": round(cue.end_seconds, 6),
                        "text": cue.text,
                        "style_id": cue.style_id,
                        "speaker": cue.speaker,
                    }
                    for cue in track.cues
                ],
            }
            for track in document.tracks
        ],
        "styles": {
            style_id: {
                "font_family": style.font_family,
                "font_size": style.font_size,
                "primary_color": style.primary_color,
                "alignment": style.alignment,
            }
            for style_id, style in sorted(document.styles.items())
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Timeline compatibility (requirement 5) — read-only. This module never
# writes a Timeline back to disk; timeline.json is never modified.
# ---------------------------------------------------------------------------


def load_timeline_for_subtitles(timeline_path: str | Path) -> timeline_engine.Timeline:
    path = Path(timeline_path)
    if not path.is_file():
        raise TimelineCompatibilityError(f"--timeline file not found: {path}")

    try:
        timeline = timeline_engine.load_timeline(path)
    except timeline_engine.TimelineEngineError as exc:
        raise TimelineCompatibilityError(f"Failed to load timeline {path}: {exc}") from exc

    result = timeline_engine.validate_timeline(timeline)
    if not result.passed:
        raise TimelineCompatibilityError(f"Timeline {path} failed validation: {'; '.join(result.errors)}")

    return timeline


# ---------------------------------------------------------------------------
# Build — orchestrates input loading, normalization, wrapping, position
# planning, frame snapping, and (optional) Timeline compatibility.
# ---------------------------------------------------------------------------


def build_subtitle_document(
    input_path: str | Path,
    config: SubtitleConfig,
    *,
    language: str | None = None,
    timeline: timeline_engine.Timeline | None = None,
) -> SubtitleDocument:
    source_path = Path(input_path)
    if not source_path.is_file():
        raise SubtitleInputError(f"--input file not found: {source_path}")
    if source_path.stat().st_size == 0:
        raise SubtitleInputError(f"--input file is empty: {source_path}")

    document = load_subtitle_source(source_path, config, timeline=timeline)

    if language is not None:
        document.language = language
        for track in document.tracks:
            track.language = language

    for track in document.tracks:
        for cue in track.cues:
            cue.text = normalize_text(cue.text, config)

            if config.frame_rate_snap_enabled:
                start = snap_to_frame_rate(cue.start_seconds, config.default_frame_rate)
                end = snap_to_frame_rate(cue.end_seconds, config.default_frame_rate)
                cue.start_seconds = start
                cue.end_seconds = end
                cue.duration_seconds = end - start

            planned_lines, wrap_warnings = plan_cue_line_wrap(
                cue.text, config, line_break_mode=cue.line_break_mode
            )
            cue.metadata["planned_lines"] = planned_lines
            document.warnings.extend(
                SubtitleWarning(code=w.code, message=w.message, track_id=track.track_id, cue_id=cue.cue_id)
                for w in wrap_warnings
            )

            style = document.styles.get(cue.style_id)
            if style is None:
                raise SubtitleStyleError(f"Cue {cue.cue_id} references unknown style_id {cue.style_id!r}")

            position = plan_cue_position(style, config, cue_position=cue.position, cue_alignment=cue.alignment)
            cue.position = position
            document.warnings.extend(
                SubtitleWarning(code="unsafe_position", message=message, track_id=track.track_id, cue_id=cue.cue_id)
                for message in validate_position_within_safe_area(position, config)
            )

        track.cues.sort(key=lambda c: (c.start_seconds, c.cue_id))

    document.duration_seconds = max(
        (cue.end_seconds for track in document.tracks for cue in track.cues), default=0.0
    )

    if timeline is not None:
        document.timeline_id = timeline.timeline_id

        subtitle_track = next(
            (
                t
                for t in timeline.tracks
                if t.track_type == timeline_engine.TrackType.SUBTITLE and t.enabled
            ),
            None,
        )
        if subtitle_track is not None and document.tracks:
            document.tracks[0].track_id = subtitle_track.track_id
            for cue in document.tracks[0].cues:
                cue.track_id = subtitle_track.track_id

        if document.duration_seconds > timeline.duration_seconds + config.timing_tolerance_seconds:
            raise TimelineCompatibilityError(
                f"Subtitle document duration ({document.duration_seconds:.3f}s) exceeds "
                f"Timeline duration ({timeline.duration_seconds:.3f}s)"
            )

    document.created_at = _now_iso()
    document.document_id = _compute_document_id(document)
    return document


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_subtitle_document(
    document: SubtitleDocument, config: SubtitleConfig, *, timeline: timeline_engine.Timeline | None = None
) -> SubtitleValidationResult:
    """
    Validates a SubtitleDocument on its own merits — whether built by
    build_subtitle_document() or loaded from disk. Returns
    passed=False iff any hard check fails; never raises for a merely-
    invalid document (a Timeline-duration mismatch here is a soft error
    in the result, not a raised TimelineCompatibilityError — that is
    build_subtitle_document()'s job).
    """
    tolerance = config.timing_tolerance_seconds
    errors: list[str] = []
    warnings: list[str] = []

    if not document.tracks or not any(track.cues for track in document.tracks):
        errors.append("document has no cues")

    all_cue_ids: list[str] = []
    max_end = 0.0

    for track in document.tracks:
        if track.default_style_id not in document.styles:
            errors.append(
                f"track {track.track_id}: default_style_id {track.default_style_id!r} not in styles"
            )

        previous_start: float | None = None
        for cue in track.cues:
            all_cue_ids.append(cue.cue_id)

            if cue.start_seconds < 0:
                errors.append(f"cue {cue.cue_id}: start_seconds is negative ({cue.start_seconds})")
            if cue.end_seconds <= cue.start_seconds:
                errors.append(
                    f"cue {cue.cue_id}: end_seconds ({cue.end_seconds}) is not greater than "
                    f"start_seconds ({cue.start_seconds})"
                )

            expected_duration = cue.end_seconds - cue.start_seconds
            if abs(cue.duration_seconds - expected_duration) > tolerance:
                errors.append(
                    f"cue {cue.cue_id}: duration_seconds ({cue.duration_seconds}) does not match "
                    f"end_seconds - start_seconds ({expected_duration})"
                )

            if not cue.text or not cue.text.strip():
                errors.append(f"cue {cue.cue_id}: text is empty")

            if cue.style_id not in document.styles:
                errors.append(f"cue {cue.cue_id}: style_id {cue.style_id!r} not found in styles")

            if cue.duration_seconds < config.minimum_cue_duration_seconds - tolerance:
                errors.append(
                    f"cue {cue.cue_id}: duration_seconds ({cue.duration_seconds}) is below "
                    f"minimum_cue_duration_seconds ({config.minimum_cue_duration_seconds})"
                )
            if cue.duration_seconds > config.maximum_cue_duration_seconds + tolerance:
                errors.append(
                    f"cue {cue.cue_id}: duration_seconds ({cue.duration_seconds}) exceeds "
                    f"maximum_cue_duration_seconds ({config.maximum_cue_duration_seconds})"
                )

            if previous_start is not None and cue.start_seconds + tolerance < previous_start:
                errors.append(
                    f"cue {cue.cue_id}: cues in track {track.track_id} are not sorted by start_seconds"
                )
            previous_start = cue.start_seconds

            max_end = max(max_end, cue.end_seconds)

        sorted_cues = sorted(track.cues, key=lambda c: c.start_seconds)
        for previous, current in zip(sorted_cues, sorted_cues[1:]):
            gap = current.start_seconds - previous.end_seconds
            if gap < -tolerance:
                if not config.allow_overlaps:
                    errors.append(
                        f"cues {previous.cue_id} and {current.cue_id} overlap in track {track.track_id}"
                    )
            elif gap < config.minimum_gap_seconds - tolerance:
                errors.append(
                    f"gap between cues {previous.cue_id} and {current.cue_id} in track "
                    f"{track.track_id} ({gap:.3f}s) is below minimum_gap_seconds "
                    f"({config.minimum_gap_seconds})"
                )

    if len(all_cue_ids) != len(set(all_cue_ids)):
        errors.append("duplicate cue_id values across the document")

    for style_id, style in document.styles.items():
        errors.extend(f"style {style_id}: {message}" for message in _validate_style(style))

    if timeline is not None and max_end > timeline.duration_seconds + tolerance:
        errors.append(
            f"document duration ({max_end:.3f}s) exceeds Timeline duration "
            f"({timeline.duration_seconds:.3f}s)"
        )

    passed = len(errors) == 0
    summary = (
        f"Subtitle document {document.document_id or '(no id)'}: {'PASSED' if passed else 'FAILED'} "
        f"({len(errors)} error(s), {len(warnings)} warning(s))"
    )

    return SubtitleValidationResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        cue_count=len(all_cue_ids),
        track_count=len(document.tracks),
        duration_seconds=max_end,
        language=document.language,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# JSON serialization — stable, round-trip-safe (requirement 10)
# ---------------------------------------------------------------------------


def _position_to_dict(position: SubtitlePosition | None) -> dict[str, Any] | None:
    return asdict(position) if position is not None else None


def _position_from_dict(data: dict[str, Any] | None) -> SubtitlePosition | None:
    if not data:
        return None
    known = {f.name for f in dataclasses.fields(SubtitlePosition)}
    return SubtitlePosition(**{key: value for key, value in data.items() if key in known})


def _style_to_dict(style: SubtitleStyle) -> dict[str, Any]:
    return asdict(style)


def _style_from_dict(data: dict[str, Any]) -> SubtitleStyle:
    known = {f.name for f in dataclasses.fields(SubtitleStyle)}
    return SubtitleStyle(**{key: value for key, value in data.items() if key in known})


def _cue_to_dict(cue: SubtitleCue) -> dict[str, Any]:
    data = asdict(cue)
    data["position"] = _position_to_dict(cue.position)
    return data


def _cue_from_dict(data: dict[str, Any]) -> SubtitleCue:
    known = {f.name for f in dataclasses.fields(SubtitleCue)}
    filtered = {key: value for key, value in data.items() if key in known}
    filtered["position"] = _position_from_dict(filtered.get("position"))
    return SubtitleCue(**filtered)


def _track_to_dict(track: SubtitleTrack) -> dict[str, Any]:
    return {
        "track_id": track.track_id,
        "language": track.language,
        "enabled": track.enabled,
        "default_style_id": track.default_style_id,
        "cues": [_cue_to_dict(cue) for cue in track.cues],
        "metadata": track.metadata,
    }


def _track_from_dict(data: dict[str, Any]) -> SubtitleTrack:
    try:
        return SubtitleTrack(
            track_id=data["track_id"],
            language=data.get("language", "en"),
            enabled=data.get("enabled", True),
            default_style_id=data.get("default_style_id", "default"),
            cues=[_cue_from_dict(cue) for cue in data.get("cues", [])],
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise SubtitleJSONError(f"Subtitle track JSON missing required field: {exc}") from exc


def _warning_to_dict(warning: SubtitleWarning) -> dict[str, Any]:
    return asdict(warning)


def _warning_from_dict(data: dict[str, Any]) -> SubtitleWarning:
    known = {f.name for f in dataclasses.fields(SubtitleWarning)}
    return SubtitleWarning(**{key: value for key, value in data.items() if key in known})


def document_to_dict(document: SubtitleDocument) -> dict[str, Any]:
    return {
        "schema_version": document.schema_version,
        "document_id": document.document_id,
        "created_at": document.created_at,
        "language": document.language,
        "duration_seconds": document.duration_seconds,
        "tracks": [_track_to_dict(track) for track in document.tracks],
        "styles": {style_id: _style_to_dict(style) for style_id, style in document.styles.items()},
        "timeline_id": document.timeline_id,
        "metadata": document.metadata,
        "warnings": [_warning_to_dict(warning) for warning in document.warnings],
    }


def document_from_dict(data: dict[str, Any]) -> SubtitleDocument:
    try:
        return SubtitleDocument(
            schema_version=data.get("schema_version", "1.0"),
            document_id=data.get("document_id", ""),
            created_at=data.get("created_at", ""),
            language=data.get("language", "en"),
            duration_seconds=data.get("duration_seconds", 0.0),
            tracks=[_track_from_dict(track) for track in data.get("tracks", [])],
            styles={
                style_id: _style_from_dict(style_data)
                for style_id, style_data in (data.get("styles") or {}).items()
            },
            timeline_id=data.get("timeline_id"),
            metadata=data.get("metadata", {}),
            warnings=[_warning_from_dict(warning) for warning in data.get("warnings", [])],
        )
    except KeyError as exc:
        raise SubtitleJSONError(f"Subtitle document JSON missing required field: {exc}") from exc
    except TypeError as exc:
        raise SubtitleJSONError(f"Malformed subtitle document JSON: {exc}") from exc


def load_subtitle_document(path: str | Path) -> SubtitleDocument:
    path = Path(path)
    if not path.is_file():
        raise SubtitleInputError(f"Subtitle document file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SubtitleJSONError(f"Invalid subtitle JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SubtitleJSONError(f"Subtitle document JSON root must be an object: {path}")

    return document_from_dict(data)


def save_subtitle_document(document: SubtitleDocument, path: str | Path, *, force: bool = False) -> Path:
    """Atomically writes document JSON via temp-file + Path.replace() so a
    reader never observes a partially-written file. Refuses to overwrite
    an existing file unless force=True."""
    path = Path(path)

    if path.exists():
        if path.is_dir():
            raise UnsafeSubtitleOutputError(f"Output path is a directory: {path}")
        if not force:
            raise SubtitleOutputExistsError(f"{path} already exists; pass force=True to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(document_to_dict(document), indent=2, sort_keys=True, ensure_ascii=False)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
    return path


def _write_export_file(path: Path, content: str, encoding: str, *, force: bool) -> Path:
    if path.exists():
        if path.is_dir():
            raise UnsafeSubtitleOutputError(f"Output path is a directory: {path}")
        if not force:
            raise SubtitleOutputExistsError(f"{path} already exists; pass force=True to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding=encoding)
    temporary_path.replace(path)
    return path


# ---------------------------------------------------------------------------
# SRT export (requirement 11)
# ---------------------------------------------------------------------------


def _format_srt_timestamp(seconds: float) -> str:
    total_millis = round(seconds * 1000)
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_srt(document: SubtitleDocument, config: SubtitleConfig) -> tuple[str, list[str]]:
    all_cues = sorted(
        (cue for track in document.tracks if track.enabled for cue in track.cues),
        key=lambda c: (c.start_seconds, c.cue_id),
    )

    lines: list[str] = []
    for index, cue in enumerate(all_cues, start=1):
        lines.append(str(index))
        lines.append(f"{_format_srt_timestamp(cue.start_seconds)} --> {_format_srt_timestamp(cue.end_seconds)}")
        lines.append(cue.text)
        lines.append("")

    content = "\n".join(lines).rstrip("\n") + "\n"
    warnings = ["SRT format does not carry style metadata (font/color/position); style information is lost."]
    return content, warnings


# ---------------------------------------------------------------------------
# ASS export (requirement 12) — pure string construction, never invokes
# a renderer or subprocess.
# ---------------------------------------------------------------------------


def _hex_color_to_ass(color: str) -> str:
    hex_part = color.lstrip("#")
    if len(hex_part) == 6:
        red, green, blue = hex_part[0:2], hex_part[2:4], hex_part[4:6]
        alpha = "00"
    elif len(hex_part) == 8:
        red, green, blue, alpha = hex_part[0:2], hex_part[2:4], hex_part[4:6], hex_part[6:8]
    else:
        raise SubtitleStyleError(f"Invalid color value: {color!r}")
    return f"&H{alpha}{blue}{green}{red}&".upper()


def _escape_ass_text(text: str) -> str:
    escaped = text.replace("{", "\\{").replace("}", "\\}")
    return escaped.replace("\n", "\\N")


def _format_ass_timestamp(seconds: float) -> str:
    total_centis = round(seconds * 100)
    hours, remainder = divmod(total_centis, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def export_ass(document: SubtitleDocument, config: SubtitleConfig) -> str:
    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {config.canvas_width}",
        f"PlayResY: {config.canvas_height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, Alignment, MarginL, MarginR, MarginV, Outline, Shadow",
    ]

    for style_id, style in sorted(document.styles.items()):
        lines.append(
            "Style: "
            + ",".join(
                [
                    style_id,
                    style.font_family,
                    str(style.font_size),
                    _hex_color_to_ass(style.primary_color),
                    _hex_color_to_ass(style.outline_color),
                    _hex_color_to_ass(style.background_color),
                    "-1" if style.font_weight >= 600 else "0",
                    "-1" if style.italic else "0",
                    "-1" if style.underline else "0",
                    str(_ASS_ALIGNMENT_NUMPAD.get(style.alignment, 2)),
                    str(style.margin_left),
                    str(style.margin_right),
                    str(style.margin_vertical),
                    str(style.outline_width),
                    str(style.shadow_depth),
                ]
            )
        )

    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    all_cues = sorted(
        (cue for track in document.tracks if track.enabled for cue in track.cues),
        key=lambda c: (c.start_seconds, c.cue_id),
    )
    for cue in all_cues:
        lines.append(
            "Dialogue: 0,"
            f"{_format_ass_timestamp(cue.start_seconds)},{_format_ass_timestamp(cue.end_seconds)},"
            f"{cue.style_id},{cue.speaker or ''},0,0,0,,{_escape_ass_text(cue.text)}"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Subtitle Engine (Phase 11D). Creates, validates, and exports "
            "platform-neutral subtitle tracks. Never renders, never calls "
            "ffmpeg/ffprobe, never transcribes audio."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", metavar="SOURCE", help="Build a subtitle document from a source file.")
    mode.add_argument(
        "--validate", metavar="SUBTITLES_JSON", help="Validate an existing subtitle document JSON file."
    )

    parser.add_argument("--config", default=None, help="Path to an alternate config/video/subtitles.yaml.")
    parser.add_argument(
        "--output", default=None, help="Write the built subtitle document JSON to this path (--input only)."
    )
    parser.add_argument(
        "--export-srt", dest="export_srt", default=None, help="Write an SRT export to this path (--input only)."
    )
    parser.add_argument(
        "--export-ass", dest="export_ass", default=None, help="Write an ASS export to this path (--input only)."
    )
    parser.add_argument(
        "--timeline", default=None, help="Path to a validated timeline.json for compatibility checks."
    )
    parser.add_argument("--language", default=None, help="Override the document/track language (--input only).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")

    return parser.parse_args(argv)


def _print_build_summary(
    document: SubtitleDocument, output_path: Path | None, srt_path: Path | None, ass_path: Path | None
) -> None:
    print()
    print("AIKO Subtitle Engine (Phase 11D)")
    print("-----------------------------------")
    print(f"document_id:       {document.document_id}")
    print(f"language:          {document.language}")
    print(f"tracks:            {len(document.tracks)}")
    print(f"cues:              {sum(len(track.cues) for track in document.tracks)}")
    print(f"duration_seconds:  {document.duration_seconds:.3f}")
    print(f"timeline_id:       {document.timeline_id}")
    for warning in document.warnings:
        print(f"warning: {warning.message}")
    print(f"json output:       {output_path if output_path else '(not written)'}")
    print(f"srt output:        {srt_path if srt_path else '(not written)'}")
    print(f"ass output:        {ass_path if ass_path else '(not written)'}")
    print()


def _print_validation_summary(result: SubtitleValidationResult) -> None:
    print()
    print("AIKO Subtitle Engine — validation (Phase 11D)")
    print("--------------------------------------------------")
    print(result.summary)
    for error in result.errors:
        print(f"  error:   {error}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_subtitle_config(arguments.config)
        timeline = load_timeline_for_subtitles(arguments.timeline) if arguments.timeline else None

        if arguments.input:
            document = build_subtitle_document(
                arguments.input, config, language=arguments.language, timeline=timeline
            )

            output_path: Path | None = None
            if arguments.output:
                output_path = save_subtitle_document(document, arguments.output, force=arguments.force)

            srt_path: Path | None = None
            if arguments.export_srt:
                srt_content, srt_warnings = export_srt(document, config)
                document.warnings.extend(SubtitleWarning(code="srt_style_loss", message=m) for m in srt_warnings)
                srt_path = _write_export_file(
                    Path(arguments.export_srt), srt_content, config.srt_encoding, force=arguments.force
                )

            ass_path: Path | None = None
            if arguments.export_ass:
                ass_content = export_ass(document, config)
                ass_path = _write_export_file(
                    Path(arguments.export_ass), ass_content, config.ass_encoding, force=arguments.force
                )

            if arguments.as_json:
                print(json.dumps(document_to_dict(document), indent=2, sort_keys=True, ensure_ascii=False))
            else:
                _print_build_summary(document, output_path, srt_path, ass_path)
        else:
            document = load_subtitle_document(arguments.validate)
            result = validate_subtitle_document(document, config, timeline=timeline)

            if arguments.as_json:
                print(json.dumps(asdict(result), indent=2, sort_keys=True, ensure_ascii=False))
            else:
                _print_validation_summary(result)

            if not result.passed:
                raise SystemExit(1)
    except SubtitleEngineError as exc:
        print(f"[SubtitleEngine] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
