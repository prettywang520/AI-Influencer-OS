from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

if TYPE_CHECKING:
    from . import filter_graph_builder, filter_graph_serializer

# NOTE: filter_graph_builder.py imports this module (for
# SubtitleRenderRequest/DrawTextCue/SubtitleRenderMode type references
# and build_subtitle_filter_spec(), added in Phase 11F.2). Importing it
# back at module load time here would create a circular import, so
# filter_graph_builder/filter_graph_serializer are imported lazily
# inside build_subtitle_render_plan() instead -- by the time that
# function actually runs, both modules are fully initialized. Type
# hints below are annotation-only (see `from __future__ import
# annotations`) and never require the real module at import time.

# Phase 11F.1 — Subtitle Render Engine. The first module in this
# pipeline permitted to burn real subtitles into video. Standalone and
# renderer-neutral: it accepts a validated subtitle render request
# (ASS file OR drawtext cues + resolved style/position metadata),
# resolves fonts from local config only (no downloads, no system-wide
# font search unless explicitly configured), builds one safe ffmpeg
# command, executes it exactly once via an injectable runner, verifies
# the output, and returns structured diagnostics. This module never
# rebuilds a Renderer Plan, never modifies a Timeline or Overlay Plan,
# never generates or translates subtitle text, never downloads fonts,
# never parses/rewrites .ass file content, and never publishes or
# uploads anything.
#
# It intentionally does NOT reuse video_engine.VideoEngine: that
# class's ConcatPlan/execute_plan() are shaped entirely around N-input
# concatenation (lossless_copy / normalized_render / rejected) and
# have no representation for a single-input subtitle filter graph.
# Rather than bend that model, this module owns its own small
# ProcessResult/SubprocessRunner/default_runner trio and command
# builder -- the same convention video_engine.py/music_mixer.py/
# media_inspector.py each already follow independently.
#
# Phase 11F.3 — filter ordering, graph labels, filter dependencies,
# drawtext cue ordering, and ASS/drawtext filter-node structure are no
# longer decided here: they come from filter_graph_builder.py's typed
# FilterGraph, converted into real ffmpeg filter syntax by
# filter_graph_serializer.py. This module keeps: request validation,
# font/subtitle-asset resolution, building the complete ffmpeg
# command, execution, output verification, and diagnostics.

DEFAULT_SUBTITLE_RENDER_CONFIG_RELATIVE_PATH = Path("config") / "video" / "subtitle_render.yaml"


def _runtime_root() -> Path:
    """
    subtitle_render_engine.py location:
    10_apps/claude_runtime/src/subtitle_render_engine.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SubtitleRenderEngineError(RuntimeError):
    """Base error for the Phase 11F.1 Subtitle Render Engine."""


class SubtitleRenderConfigError(SubtitleRenderEngineError):
    """Raised when config/video/subtitle_render.yaml is missing or invalid."""


class SubtitleRenderRequestError(SubtitleRenderEngineError):
    """Raised for a structurally invalid subtitle render request."""


class UnsupportedSubtitleRenderModeError(SubtitleRenderEngineError):
    """Raised when a requested mode is outside SubtitleRenderMode.ALL or
    not enabled by config.supported_modes."""


class SubtitleAssetNotFoundError(SubtitleRenderEngineError):
    """Raised when the input video or .ass subtitle file does not exist
    or is not a regular file."""


class SubtitleAssetEmptyError(SubtitleRenderEngineError):
    """Raised when the input video or .ass subtitle file is zero bytes."""


class UnsupportedSubtitleAssetError(SubtitleRenderEngineError):
    """Raised when the .ass file's extension is not in config's
    supported extensions."""


class SubtitleFontResolutionError(SubtitleRenderEngineError):
    """Raised when a required font cannot be resolved via explicit
    path, family_map, or fallback_font_path."""


class SubtitleDrawTextCueError(SubtitleRenderEngineError):
    """Raised for an invalid drawtext cue: bad timing, missing/invalid
    position, or an unsupported anchor."""


class SubtitleFilterGraphError(SubtitleRenderEngineError):
    """Raised for a structural filter-graph construction failure."""


class UnsafeSubtitleRenderOutputError(SubtitleRenderEngineError):
    """Raised when the output path is a directory or resolves to the
    same path as an input asset."""


class SubtitleRenderOutputExistsError(SubtitleRenderEngineError):
    """Raised when the output already exists and force was not given."""


class FFmpegNotFoundError(SubtitleRenderEngineError):
    """Raised when the configured ffmpeg binary cannot be found/executed."""


class FFmpegExecutionError(SubtitleRenderEngineError):
    """Raised when ffmpeg exits non-zero."""


class FFmpegTimeoutError(SubtitleRenderEngineError):
    """Raised when ffmpeg does not finish within the configured timeout."""


class SubtitleRenderVerificationError(SubtitleRenderEngineError):
    """Raised when the output file is missing or zero bytes after a
    reported-successful run."""


class SubtitleTemporaryFileError(SubtitleRenderEngineError):
    """Raised when a temporary/diagnostic file cannot be safely written
    or cleaned up."""


class SubtitleRenderFilterGraphError(SubtitleRenderEngineError):
    """Raised when filter_graph_builder.py planning/validation or
    filter_graph_serializer.py serialization fails while building a
    subtitle render plan. The original error message is preserved."""


# ---------------------------------------------------------------------------
# String-constant "enum"
# ---------------------------------------------------------------------------


class SubtitleRenderMode:
    ASS = "ass"
    DRAWTEXT = "drawtext"
    ALL = (ASS, DRAWTEXT)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SubtitleRenderConfig:
    schema_version: str = "1.0"
    supported_modes: tuple[str, ...] = (SubtitleRenderMode.ASS, SubtitleRenderMode.DRAWTEXT)
    default_mode: str = SubtitleRenderMode.ASS
    # Reserved for future use -- this module never implements automatic
    # ASS<->drawtext fallback regardless of this flag's value; no code
    # path ever reassigns a request's mode.
    allow_mode_fallback: bool = False

    ffmpeg_binary: str = "ffmpeg"
    ffmpeg_timeout_seconds: int = 300
    ffmpeg_loglevel: str = "error"
    ffmpeg_hide_banner: bool = True

    video_codec: str = "libx264"
    video_preset: str = "medium"
    video_crf: int = 18
    video_pixel_format: str = "yuv420p"
    copy_audio: bool = True
    audio_codec_when_reencode_required: str = "aac"
    audio_bitrate: str = "192k"
    faststart: bool = True

    ass_supported_extensions: tuple[str, ...] = (".ass",)
    ass_allow_fontconfig_resolution: bool = False
    ass_require_fontsdir_when_custom_fonts: bool = True

    drawtext_require_font_file: bool = True
    drawtext_use_textfile_mode: bool = False
    drawtext_box_enabled_default: bool = False
    drawtext_default_box_color: str = "#00000000"
    drawtext_strict_canvas_bounds: bool = True
    drawtext_anchor_margin_pixels: float = 40.0

    font_supported_extensions: tuple[str, ...] = (".ttf", ".otf", ".ttc")
    font_fallback_path: str | None = None
    font_family_map: dict[str, str] = field(default_factory=dict)

    overwrite_requires_force: bool = True
    verify_non_zero_bytes: bool = True

    temporary_directory_name: str = ".subtitle_render_tmp"
    cleanup_on_success: bool = True
    preserve_on_failure: bool = True

    diagnostics_write_log: bool = True
    diagnostics_log_filename_suffix: str = "_subtitle_render_log.json"


def default_subtitle_render_config_path() -> Path:
    return _runtime_root() / DEFAULT_SUBTITLE_RENDER_CONFIG_RELATIVE_PATH


def load_subtitle_render_config(config_path: str | Path | None = None) -> SubtitleRenderConfig:
    """Load config/video/subtitle_render.yaml (or an alternate path)
    into a SubtitleRenderConfig. Raises SubtitleRenderConfigError if the
    file is missing or invalid."""
    path = Path(config_path) if config_path else default_subtitle_render_config_path()

    if not path.exists():
        raise SubtitleRenderConfigError(f"Subtitle render config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SubtitleRenderConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise SubtitleRenderConfigError(f"Subtitle render config is empty or invalid: {path}")

    subtitle_render_section = raw.get("subtitle_render") or {}
    ffmpeg_section = raw.get("ffmpeg") or {}
    video_section = raw.get("video") or {}
    ass_section = raw.get("ass") or {}
    drawtext_section = raw.get("drawtext") or {}
    fonts_section = raw.get("fonts") or {}
    output_section = raw.get("output") or {}
    temporary_section = raw.get("temporary") or {}
    diagnostics_section = raw.get("diagnostics") or {}

    supported_modes = subtitle_render_section.get("supported_modes") or [
        SubtitleRenderMode.ASS,
        SubtitleRenderMode.DRAWTEXT,
    ]

    return SubtitleRenderConfig(
        schema_version=str(subtitle_render_section.get("schema_version", "1.0")),
        supported_modes=tuple(str(m) for m in supported_modes),
        default_mode=str(subtitle_render_section.get("default_mode", SubtitleRenderMode.ASS)),
        allow_mode_fallback=bool(subtitle_render_section.get("allow_mode_fallback", False)),
        ffmpeg_binary=str(ffmpeg_section.get("binary", "ffmpeg")),
        ffmpeg_timeout_seconds=int(ffmpeg_section.get("timeout_seconds", 300)),
        ffmpeg_loglevel=str(ffmpeg_section.get("loglevel", "error")),
        ffmpeg_hide_banner=bool(ffmpeg_section.get("hide_banner", True)),
        video_codec=str(video_section.get("codec", "libx264")),
        video_preset=str(video_section.get("preset", "medium")),
        video_crf=int(video_section.get("crf", 18)),
        video_pixel_format=str(video_section.get("pixel_format", "yuv420p")),
        copy_audio=bool(video_section.get("copy_audio", True)),
        audio_codec_when_reencode_required=str(video_section.get("audio_codec_when_reencode_required", "aac")),
        audio_bitrate=str(video_section.get("audio_bitrate", "192k")),
        faststart=bool(video_section.get("faststart", True)),
        ass_supported_extensions=tuple(str(e) for e in (ass_section.get("supported_extensions") or [".ass"])),
        ass_allow_fontconfig_resolution=bool(ass_section.get("allow_fontconfig_resolution", False)),
        ass_require_fontsdir_when_custom_fonts=bool(
            ass_section.get("require_fontsdir_when_custom_fonts", True)
        ),
        drawtext_require_font_file=bool(drawtext_section.get("require_font_file", True)),
        drawtext_use_textfile_mode=bool(drawtext_section.get("use_textfile_mode", False)),
        drawtext_box_enabled_default=bool(drawtext_section.get("box_enabled_default", False)),
        drawtext_default_box_color=str(drawtext_section.get("default_box_color", "#00000000")),
        drawtext_strict_canvas_bounds=bool(drawtext_section.get("strict_canvas_bounds", True)),
        drawtext_anchor_margin_pixels=float(drawtext_section.get("anchor_margin_pixels", 40.0)),
        font_supported_extensions=tuple(
            str(e) for e in (fonts_section.get("supported_extensions") or [".ttf", ".otf", ".ttc"])
        ),
        font_fallback_path=fonts_section.get("fallback_font_path"),
        font_family_map=dict(fonts_section.get("family_map") or {}),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        verify_non_zero_bytes=bool(output_section.get("verify_non_zero_bytes", True)),
        temporary_directory_name=str(temporary_section.get("directory_name", ".subtitle_render_tmp")),
        cleanup_on_success=bool(temporary_section.get("cleanup_on_success", True)),
        preserve_on_failure=bool(temporary_section.get("preserve_on_failure", True)),
        diagnostics_write_log=bool(diagnostics_section.get("write_log", True)),
        diagnostics_log_filename_suffix=str(
            diagnostics_section.get("log_filename_suffix", "_subtitle_render_log.json")
        ),
    )


# ---------------------------------------------------------------------------
# Runner (injectable, mockable) -- module-owned, never imported from or
# into video_engine.py/music_mixer.py/media_inspector.py.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProcessResult:
    """Minimal duck-typed stand-in for subprocess.CompletedProcess, used
    so fake runners in tests don't need to construct a real one."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


SubprocessRunner = Callable[..., ProcessResult]


def default_runner(command: list[str], *, timeout: int) -> ProcessResult:
    """
    Real subprocess runner: shell=False, check=False, command passed as
    list[str], text mode, captured stdout/stderr, configurable timeout.
    Never logs environment variables or secrets. Raises
    FFmpegNotFoundError/FFmpegTimeoutError; never retries.
    """
    try:
        completed = subprocess.run(
            command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(
            f"Executable not found: {command[0]!r}. Install ffmpeg to enable real subtitle rendering."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegTimeoutError(f"Command timed out after {timeout}s: {' '.join(command)}") from exc

    return ProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ResolvedFont:
    font_path: str | None = None
    source: str = ""  # explicit | family_map | fallback
    font_family: str | None = None


@dataclass(slots=True)
class DrawTextCue:
    text: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    font_family: str | None = None
    font_path: str | None = None
    font_size: float = 48.0
    font_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: float = 2.0
    x: float | None = None
    y: float | None = None
    anchor: str | None = None
    box_enabled: bool | None = None
    box_color: str | None = None
    opacity: float = 1.0


@dataclass(slots=True)
class SubtitleRenderAsset:
    asset_type: str = ""  # ass | drawtext
    path: str = ""


@dataclass(slots=True)
class SubtitleRenderWarning:
    code: str = ""
    message: str = ""


@dataclass(slots=True)
class SubtitleRenderRequest:
    mode: str
    video_path: Path
    output_path: Path
    force: bool = False
    ass_path: Path | None = None
    fontsdir: Path | None = None
    cues: list[DrawTextCue] = field(default_factory=list)
    canvas_width: int | None = None
    canvas_height: int | None = None


@dataclass(slots=True)
class SubtitleRenderPlan:
    render_id: str = ""
    mode: str = ""
    command: list[str] = field(default_factory=list)
    input_video: str = ""
    output_video: str = ""
    subtitle_asset: SubtitleRenderAsset | None = None
    font_resolution: list[ResolvedFont] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Phase 11F.3 — additive FilterGraph/serialization diagnostics.
    filter_graph_id: str = ""
    filter_graph_validation_passed: bool = False
    serialization_id: str = ""
    output_mode: str = ""


@dataclass(slots=True)
class SubtitleRenderResult:
    render_id: str = ""
    mode: str = ""
    input_video: str = ""
    subtitle_asset: str | None = None
    output_video: str = ""
    font_resolution: list[dict[str, Any]] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    output_exists: bool = False
    output_size_bytes: int | None = None
    audio_preserved: bool | None = None
    warnings: list[str] = field(default_factory=list)
    temporary_files: list[str] = field(default_factory=list)
    cleanup_result: str = "retained"
    error: str | None = None
    # Phase 11F.3 — additive FilterGraph/serialization diagnostics.
    filter_graph_id: str = ""
    filter_graph_validation_passed: bool = False
    serialization_id: str = ""
    output_mode: str = ""


# ---------------------------------------------------------------------------
# Font resolution — config-driven only, no downloads, no system-wide
# search unless explicitly configured via family_map/fallback_font_path.
# ---------------------------------------------------------------------------


def _validate_font_file(path: Path, config: SubtitleRenderConfig) -> None:
    if not path.exists():
        raise SubtitleFontResolutionError(f"Font file not found: {path}")
    if not path.is_file():
        raise SubtitleFontResolutionError(f"Font path is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise SubtitleFontResolutionError(f"Font file is zero bytes: {path}")
    supported = tuple(ext.lower() for ext in config.font_supported_extensions)
    if path.suffix.lower() not in supported:
        raise SubtitleFontResolutionError(
            f"Font file extension {path.suffix!r} not in supported extensions {config.font_supported_extensions}: {path}"
        )


def resolve_font(
    *, explicit_font_path: str | Path | None, font_family: str | None, config: SubtitleRenderConfig
) -> ResolvedFont:
    """
    Resolves a font path deterministically: explicit path -> configured
    family_map[font_family] -> configured fallback_font_path -> raise.
    Never downloads a font, never mutates a font cache, never searches
    the system font directory unless a caller-supplied path/family_map
    entry happens to point there.
    """
    if explicit_font_path:
        path = Path(explicit_font_path)
        _validate_font_file(path, config)
        return ResolvedFont(font_path=str(path), source="explicit", font_family=font_family)

    if font_family and font_family in config.font_family_map:
        mapped = Path(config.font_family_map[font_family])
        _validate_font_file(mapped, config)
        return ResolvedFont(font_path=str(mapped), source="family_map", font_family=font_family)

    if config.font_fallback_path:
        fallback = Path(config.font_fallback_path)
        _validate_font_file(fallback, config)
        return ResolvedFont(font_path=str(fallback), source="fallback", font_family=font_family)

    raise SubtitleFontResolutionError(
        f"Could not resolve a font for family {font_family!r}: no explicit font path, no "
        "fonts.family_map entry, and no fonts.fallback_font_path configured"
    )


def _resolve_cue_font(cue: DrawTextCue, config: SubtitleRenderConfig) -> ResolvedFont | None:
    try:
        return resolve_font(explicit_font_path=cue.font_path, font_family=cue.font_family, config=config)
    except SubtitleFontResolutionError:
        if config.drawtext_require_font_file:
            raise
        return None


# ---------------------------------------------------------------------------
# Request construction — all pass/fail policy for inputs lives here.
# ---------------------------------------------------------------------------


def build_subtitle_render_request(
    *,
    mode: str,
    video_path: str | Path,
    output_path: str | Path,
    config: SubtitleRenderConfig,
    force: bool = False,
    ass_path: str | Path | None = None,
    fontsdir: str | Path | None = None,
    cues: list[DrawTextCue] | None = None,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
) -> SubtitleRenderRequest:
    if mode not in SubtitleRenderMode.ALL:
        raise UnsupportedSubtitleRenderModeError(f"mode={mode!r} expected one of {SubtitleRenderMode.ALL}")
    if mode not in config.supported_modes:
        raise UnsupportedSubtitleRenderModeError(
            f"mode={mode!r} is not enabled by config.supported_modes={config.supported_modes}"
        )

    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise SubtitleAssetNotFoundError(f"Input video not found: {video_path}")
    if not video_path.is_file():
        raise SubtitleAssetNotFoundError(f"Input video is not a regular file: {video_path}")
    if video_path.stat().st_size == 0:
        raise SubtitleAssetEmptyError(f"Input video is zero bytes: {video_path}")

    resolved_ass_path: Path | None = None
    cues = list(cues or [])

    if mode == SubtitleRenderMode.ASS:
        if not ass_path:
            raise SubtitleRenderRequestError("ASS mode requires ass_path")
        resolved_ass_path = Path(ass_path)
        if not resolved_ass_path.exists():
            raise SubtitleAssetNotFoundError(f"ASS subtitle file not found: {resolved_ass_path}")
        if not resolved_ass_path.is_file():
            raise SubtitleAssetNotFoundError(f"ASS subtitle path is not a regular file: {resolved_ass_path}")
        if resolved_ass_path.stat().st_size == 0:
            raise SubtitleAssetEmptyError(f"ASS subtitle file is zero bytes: {resolved_ass_path}")
        supported = tuple(ext.lower() for ext in config.ass_supported_extensions)
        if resolved_ass_path.suffix.lower() not in supported:
            raise UnsupportedSubtitleAssetError(
                f"ASS subtitle file extension {resolved_ass_path.suffix!r} not in supported "
                f"extensions {config.ass_supported_extensions}: {resolved_ass_path}"
            )
    else:
        if not cues:
            raise SubtitleRenderRequestError("drawtext mode requires at least one cue")

    resolved_fontsdir: Path | None = None
    if fontsdir is not None:
        resolved_fontsdir = Path(fontsdir)
        if not resolved_fontsdir.is_dir():
            raise SubtitleFontResolutionError(f"fontsdir is not a directory: {resolved_fontsdir}")

    resolved_video = video_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_video:
        raise UnsafeSubtitleRenderOutputError("output_path must not be the same path as video_path")
    if resolved_ass_path is not None and resolved_output == resolved_ass_path.resolve():
        raise UnsafeSubtitleRenderOutputError("output_path must not be the same path as ass_path")

    asset_paths = [resolved_video]
    if resolved_ass_path is not None:
        asset_paths.append(resolved_ass_path.resolve())
    if len(asset_paths) != len(set(asset_paths)):
        raise SubtitleRenderRequestError("Duplicate resolved asset paths in request")

    if output_path.exists():
        if output_path.is_dir():
            raise UnsafeSubtitleRenderOutputError(f"output_path is a directory: {output_path}")
        if not force:
            raise SubtitleRenderOutputExistsError(f"{output_path} already exists; pass force=True to overwrite.")

    parent = output_path.parent
    if parent.exists() and not parent.is_dir():
        raise UnsafeSubtitleRenderOutputError(f"Output parent exists but is not a directory: {parent}")

    return SubtitleRenderRequest(
        mode=mode,
        video_path=video_path,
        output_path=output_path,
        force=force,
        ass_path=resolved_ass_path,
        fontsdir=resolved_fontsdir,
        cues=cues,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


# ---------------------------------------------------------------------------
# Command construction — the filter portion (-vf/-filter_complex) comes
# from an already-serialized FilterGraph (Phase 11F.3); everything else
# (input, codec, audio, faststart, output path) is unchanged from 11F.1.
# ---------------------------------------------------------------------------


def build_subtitle_ffmpeg_command(
    request: SubtitleRenderRequest,
    config: SubtitleRenderConfig,
    *,
    serialized: filter_graph_serializer.SerializedFilterGraph,
) -> list[str]:
    command = [config.ffmpeg_binary]

    if config.ffmpeg_hide_banner:
        command.append("-hide_banner")

    command.extend(["-v", config.ffmpeg_loglevel])
    command.append("-y" if request.force else "-n")
    command.extend(["-i", str(request.video_path)])

    command.extend([serialized.filter_argument_name, serialized.filter_expression])

    command.extend(["-c:v", config.video_codec])
    command.extend(["-preset", config.video_preset])
    command.extend(["-crf", str(config.video_crf)])
    command.extend(["-pix_fmt", config.video_pixel_format])

    if config.copy_audio:
        command.extend(["-c:a", "copy"])
    else:
        command.extend(["-c:a", config.audio_codec_when_reencode_required])
        command.extend(["-b:a", config.audio_bitrate])

    if config.faststart:
        command.extend(["-movflags", "+faststart"])

    command.append(str(request.output_path))
    return command


# ---------------------------------------------------------------------------
# Deterministic render_id
# ---------------------------------------------------------------------------


def _compute_render_id(request: SubtitleRenderRequest, resolved_fonts: list[ResolvedFont]) -> str:
    payload = {
        "mode": request.mode,
        "video_path": str(request.video_path),
        "output_path": str(request.output_path),
        "ass_path": str(request.ass_path) if request.ass_path else None,
        "fontsdir": str(request.fontsdir) if request.fontsdir else None,
        "canvas_width": request.canvas_width,
        "canvas_height": request.canvas_height,
        "cues": [
            {
                "text": cue.text,
                "start_seconds": round(cue.start_seconds, 6),
                "end_seconds": round(cue.end_seconds, 6),
                "font_family": cue.font_family,
                "font_path": cue.font_path,
                "font_size": cue.font_size,
                "font_color": cue.font_color,
                "outline_color": cue.outline_color,
                "outline_width": cue.outline_width,
                "x": cue.x,
                "y": cue.y,
                "anchor": cue.anchor,
                "box_enabled": cue.box_enabled,
                "box_color": cue.box_color,
                "opacity": cue.opacity,
            }
            for cue in request.cues
        ],
        "font_resolution": [
            {"font_path": f.font_path, "source": f.source, "font_family": f.font_family} for f in resolved_fonts
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Planning — never executes ffmpeg.
# ---------------------------------------------------------------------------


def build_subtitle_render_plan(
    request: SubtitleRenderRequest,
    config: SubtitleRenderConfig,
    *,
    filter_graph_builder_callable=None,
    filter_graph_validator_callable=None,
    filter_graph_serializer_callable=None,
    serializer_config: filter_graph_serializer.FilterGraphSerializerConfig | None = None,
) -> SubtitleRenderPlan:
    """
    Phase 11F.3 flow: SubtitleRenderRequest -> filter_graph_builder's
    semantic FilterSpecs -> (font resolution injected here, the only
    filesystem-touching step) -> typed FilterGraph (built+validated) ->
    filter_graph_serializer's real ffmpeg filter syntax -> ffmpeg
    command. Filter ordering/labels/dependencies/drawtext cue ordering/
    ASS-drawtext filter-node structure are no longer decided in this
    module. The four *_callable/serializer_config parameters are
    additive dependency-injection hooks (default to the real
    implementations) for tests that need to observe/replace a stage.
    """
    from . import filter_graph_builder, filter_graph_serializer  # local: avoids a circular import at module load time

    build_filters = filter_graph_builder_callable or filter_graph_builder.build_subtitle_filter_spec
    build_graph = filter_graph_validator_callable or filter_graph_builder.build_filter_graph_from_filters
    serialize = filter_graph_serializer_callable or filter_graph_serializer.serialize_filter_graph
    effective_serializer_config = serializer_config or filter_graph_serializer.load_filter_graph_serializer_config()

    fg_config = filter_graph_builder.load_filter_graph_config()

    try:
        filter_specs = build_filters(request, fg_config)
    except filter_graph_builder.FilterGraphBuilderError as exc:
        raise SubtitleRenderFilterGraphError(f"Failed to build filter graph specs: {exc}") from exc

    resolved_fonts: list[ResolvedFont | None] = []
    if request.mode == SubtitleRenderMode.DRAWTEXT:
        for index, cue in enumerate(request.cues):
            resolved_font = _resolve_cue_font(cue, config)
            resolved_fonts.append(resolved_font)
            if index < len(filter_specs) and resolved_font is not None and resolved_font.font_path:
                filter_specs[index].parameters["font_file"] = resolved_font.font_path

    known_fonts = [f for f in resolved_fonts if f is not None]
    render_id = _compute_render_id(request, known_fonts)

    try:
        graph = build_graph(
            filter_specs, pass_id="pass_subtitle", pass_type="subtitle", hint_type=request.mode,
            config=fg_config,
        )
    except filter_graph_builder.FilterGraphBuilderError as exc:
        raise SubtitleRenderFilterGraphError(f"Failed to build filter graph: {exc}") from exc

    if not graph.validation.passed:
        raise SubtitleRenderFilterGraphError(
            f"Filter graph {graph.graph_id} failed validation: {'; '.join(graph.validation.errors)}"
        )

    try:
        serialized = serialize(graph, effective_serializer_config)
    except filter_graph_serializer.FilterGraphSerializerError as exc:
        raise SubtitleRenderFilterGraphError(f"Failed to serialize filter graph: {exc}") from exc

    command = build_subtitle_ffmpeg_command(request, config, serialized=serialized)

    if request.ass_path is not None:
        subtitle_asset = SubtitleRenderAsset(asset_type="ass", path=str(request.ass_path))
    elif request.cues:
        subtitle_asset = SubtitleRenderAsset(asset_type="drawtext", path=f"{len(request.cues)} cue(s)")
    else:
        subtitle_asset = None

    return SubtitleRenderPlan(
        render_id=render_id,
        mode=request.mode,
        command=command,
        input_video=str(request.video_path),
        output_video=str(request.output_path),
        subtitle_asset=subtitle_asset,
        font_resolution=known_fonts,
        warnings=[],
        filter_graph_id=graph.graph_id,
        filter_graph_validation_passed=graph.validation.passed,
        serialization_id=serialized.serialization_id,
        output_mode=serialized.output_mode,
    )


# ---------------------------------------------------------------------------
# Output verification — read-only.
# ---------------------------------------------------------------------------


def verify_subtitle_render_output(
    output_path: Path, input_path: Path, *, inspector: Callable[[Path], Any] | None = None
) -> tuple[bool, int]:
    """Read-only check: returns (exists, size_bytes). size_bytes is 0
    when the file does not exist. If `inspector` is supplied (duck-typed
    e.g. wrapping media_inspector.inspect_file), it is called for a
    deeper check -- never required, never invoked with real ffprobe by
    default. Never modifies output_path."""
    if not output_path.is_file():
        return False, 0

    if output_path.resolve() == Path(input_path).resolve():
        raise UnsafeSubtitleRenderOutputError("Output path resolved to the same file as the input video")

    if inspector is not None:
        inspector(output_path)

    return True, output_path.stat().st_size


# ---------------------------------------------------------------------------
# Execution — runs the built command exactly once via the injected
# runner. No automatic retry, no hidden fallback command.
# ---------------------------------------------------------------------------


def execute_subtitle_render_plan(
    request: SubtitleRenderRequest,
    config: SubtitleRenderConfig,
    *,
    runner: SubprocessRunner = default_runner,
    inspector: Callable[[Path], Any] | None = None,
    filter_graph_builder_callable=None,
    filter_graph_validator_callable=None,
    filter_graph_serializer_callable=None,
    serializer_config: filter_graph_serializer.FilterGraphSerializerConfig | None = None,
) -> SubtitleRenderResult:
    plan = build_subtitle_render_plan(
        request, config,
        filter_graph_builder_callable=filter_graph_builder_callable,
        filter_graph_validator_callable=filter_graph_validator_callable,
        filter_graph_serializer_callable=filter_graph_serializer_callable,
        serializer_config=serializer_config,
    )

    if request.output_path.exists():
        if request.output_path.is_dir():
            raise UnsafeSubtitleRenderOutputError(f"output_path is a directory: {request.output_path}")
        if not request.force:
            raise SubtitleRenderOutputExistsError(
                f"{request.output_path} already exists; pass force=True to overwrite."
            )

    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _now_iso()
    start_monotonic = time.monotonic()

    process_result = runner(plan.command, timeout=config.ffmpeg_timeout_seconds)

    finished_at = _now_iso()
    duration_seconds = time.monotonic() - start_monotonic

    if process_result.returncode != 0:
        raise FFmpegExecutionError(
            f"ffmpeg exited {process_result.returncode}: {process_result.stderr.strip()[-2000:]}"
        )

    output_exists, output_size_bytes = verify_subtitle_render_output(
        request.output_path, request.video_path, inspector=inspector
    )

    if not output_exists:
        raise SubtitleRenderVerificationError(
            f"ffmpeg reported success but no output file exists: {request.output_path}"
        )
    if config.verify_non_zero_bytes and output_size_bytes == 0:
        raise SubtitleRenderVerificationError(f"Output file is zero bytes: {request.output_path}")

    audio_preserved: bool | None = None
    if inspector is not None:
        info = inspector(request.output_path)
        audio_preserved = bool(getattr(info, "has_audio", None))

    result = SubtitleRenderResult(
        render_id=plan.render_id,
        mode=plan.mode,
        input_video=plan.input_video,
        subtitle_asset=plan.subtitle_asset.path if plan.subtitle_asset else None,
        output_video=plan.output_video,
        font_resolution=[asdict(f) for f in plan.font_resolution],
        command=plan.command,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        return_code=process_result.returncode,
        stdout=process_result.stdout,
        stderr=process_result.stderr,
        output_exists=output_exists,
        output_size_bytes=output_size_bytes,
        audio_preserved=audio_preserved,
        warnings=list(plan.warnings),
        temporary_files=[],
        cleanup_result="retained",
        error=None,
        filter_graph_id=plan.filter_graph_id,
        filter_graph_validation_passed=plan.filter_graph_validation_passed,
        serialization_id=plan.serialization_id,
        output_mode=plan.output_mode,
    )

    if config.diagnostics_write_log:
        log_path = request.output_path.with_name(
            f"{request.output_path.stem}{config.diagnostics_log_filename_suffix}"
        )
        save_subtitle_render_log(result, log_path)

    return result


# ---------------------------------------------------------------------------
# Diagnostics log — atomic write, never logs secrets/env vars/font
# binary contents.
# ---------------------------------------------------------------------------


def subtitle_render_result_to_dict(result: SubtitleRenderResult) -> dict[str, Any]:
    return asdict(result)


def save_subtitle_render_log(result: SubtitleRenderResult, path: Path) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(subtitle_render_result_to_dict(result), indent=2, sort_keys=True, ensure_ascii=False)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise SubtitleTemporaryFileError(f"Failed to write subtitle render log {path}: {exc}") from exc
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_drawtext_cues(path: Path) -> list[DrawTextCue]:
    if not path.is_file():
        raise SubtitleRenderRequestError(f"--drawtext-json file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubtitleRenderRequestError(f"Invalid drawtext cues JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise SubtitleRenderRequestError(f"Drawtext cues JSON root must be a list: {path}")

    known = {f.name for f in dataclasses.fields(DrawTextCue)}
    cues: list[DrawTextCue] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise SubtitleRenderRequestError(f"Each drawtext cue must be an object: {path}")
        try:
            cues.append(DrawTextCue(**{key: value for key, value in entry.items() if key in known}))
        except TypeError as exc:
            raise SubtitleRenderRequestError(f"Malformed drawtext cue in {path}: {exc}") from exc
    return cues


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Subtitle Render Engine (Phase 11F.1). Burns ASS or drawtext "
            "subtitles into a video via a safe, injectable ffmpeg command. "
            "Defaults to real execution; pass --dry-run to only build and print "
            "the command."
        )
    )

    parser.add_argument("--video", required=True, help="Path to the source video.")
    parser.add_argument("--output", required=True, help="Path to write the subtitle-rendered video.")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--ass", default=None, help="Path to a .ass subtitle file to burn in.")
    mode_group.add_argument(
        "--drawtext-json", dest="drawtext_json", default=None,
        help="Path to a JSON file containing a list of drawtext cue objects.",
    )

    parser.add_argument("--fontsdir", default=None, help="Optional fonts directory for ASS mode.")
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/subtitle_render.yaml.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")

    return parser.parse_args(argv)


def _print_plan(plan: SubtitleRenderPlan) -> None:
    print()
    print("AIKO Subtitle Render Engine — dry run (Phase 11F.1)")
    print("----------------------------------------------------------")
    print(f"render_id:     {plan.render_id}")
    print(f"mode:          {plan.mode}")
    print(f"input_video:   {plan.input_video}")
    print(f"output_video:  {plan.output_video}")
    print(f"command:       {' '.join(plan.command)}")
    print()


def _print_result(result: SubtitleRenderResult) -> None:
    print()
    print("AIKO Subtitle Render Engine (Phase 11F.1)")
    print("----------------------------------------------")
    print(f"render_id:      {result.render_id}")
    print(f"mode:           {result.mode}")
    print(f"output_video:   {result.output_video}")
    print(f"output_exists:  {result.output_exists}")
    print(f"return_code:    {result.return_code}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_subtitle_render_config(arguments.config)
        mode = SubtitleRenderMode.ASS if arguments.ass else SubtitleRenderMode.DRAWTEXT
        cues = _load_drawtext_cues(Path(arguments.drawtext_json)) if arguments.drawtext_json else None

        request = build_subtitle_render_request(
            mode=mode,
            video_path=arguments.video,
            output_path=arguments.output,
            config=config,
            force=arguments.force,
            ass_path=arguments.ass,
            fontsdir=arguments.fontsdir,
            cues=cues,
        )

        if arguments.dry_run:
            plan = build_subtitle_render_plan(request, config)
            if arguments.as_json:
                print(json.dumps(asdict(plan), indent=2, sort_keys=True, ensure_ascii=False))
            else:
                _print_plan(plan)
            return

        result = execute_subtitle_render_plan(request, config)
        if arguments.as_json:
            print(json.dumps(subtitle_render_result_to_dict(result), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_result(result)
    except SubtitleRenderEngineError as exc:
        print(f"[SubtitleRenderEngine] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
