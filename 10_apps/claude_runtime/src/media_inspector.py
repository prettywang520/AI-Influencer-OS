from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

# Phase 11A.1 — Media Inspector. Purely local, read-only ffprobe metadata
# extraction. Never renders, concatenates, transcodes, normalizes,
# publishes, or uploads anything, and never modifies an inspected media
# file — the only file this module can ever write is an explicit
# --output report path supplied by the caller. It never imports
# Playwright, InstagramSession, or anything under src.publishing/
# src.social, and never invokes ffmpeg (only ffprobe).
DISALLOWED_ACTIONS = (
    "render",
    "concatenate",
    "transcode",
    "normalize_media",
    "publish",
    "upload",
    "send",
    "share",
    "modify_media",
)

DEFAULT_INSPECTOR_CONFIG_RELATIVE_PATH = Path("config") / "media" / "inspector.yaml"

# Records which version of THIS module's normalization/parsing logic
# produced a given MediaInfo — not ffprobe's own version (no
# -show_program_version flag is requested; see requirement 2's exact
# command list). Bump this if the MediaInfo schema shape changes.
MEDIA_INSPECTOR_SCHEMA_VERSION = "1.0"


def _runtime_root() -> Path:
    """
    media_inspector.py location: 10_apps/claude_runtime/src/media_inspector.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MediaInspectorError(RuntimeError):
    """Base error for the Phase 11A.1 Media Inspector."""


class MediaInspectorConfigError(MediaInspectorError):
    """Raised when config/media/inspector.yaml is missing or invalid."""


class MediaFileNotFoundError(MediaInspectorError):
    """Raised when the given path does not exist."""


class MediaFileEmptyError(MediaInspectorError):
    """Raised when the given file is zero bytes."""


class UnsupportedMediaExtensionError(MediaInspectorError):
    """Raised when the given file's extension is not a configured video/audio extension."""


class FFprobeNotFoundError(MediaInspectorError):
    """Raised when the configured ffprobe binary cannot be found/executed."""


class FFprobeExecutionError(MediaInspectorError):
    """Raised when ffprobe exits non-zero."""


class FFprobeTimeoutError(MediaInspectorError):
    """Raised when ffprobe does not finish within the configured timeout."""


class FFprobeOutputError(MediaInspectorError):
    """Raised when ffprobe's stdout is not valid/parseable top-level JSON."""


class ReportAlreadyExistsError(MediaInspectorError):
    """Raised when --output already exists and --force was not given."""


class UnsafeReportPathError(MediaInspectorError):
    """Raised when --output would resolve to the same path as the inspected input."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InspectorConfig:
    ffprobe_binary: str = "ffprobe"
    timeout_seconds: int = 30

    video_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv"})
    )
    audio_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac"})
    )

    container_stream_duration_warning_seconds: float = 0.5
    variable_fps_tolerance: float = 0.05
    follow_directory_symlinks: bool = False

    report_schema_version: str = "1.0"
    overwrite_requires_force: bool = True

    @property
    def supported_extensions(self) -> frozenset[str]:
        return self.video_extensions | self.audio_extensions

    def media_category(self, extension: str) -> str | None:
        lowered = extension.lower()
        if lowered in self.video_extensions:
            return "video"
        if lowered in self.audio_extensions:
            return "audio"
        return None


def default_inspector_config_path() -> Path:
    return _runtime_root() / DEFAULT_INSPECTOR_CONFIG_RELATIVE_PATH


def load_inspector_config(config_path: str | Path | None = None) -> InspectorConfig:
    path = Path(config_path) if config_path else default_inspector_config_path()

    if not path.exists():
        raise MediaInspectorConfigError(f"Media inspector config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise MediaInspectorConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise MediaInspectorConfigError(f"Media inspector config is empty or invalid: {path}")

    ffprobe_section = raw.get("ffprobe") or {}
    extensions_section = raw.get("supported_extensions") or {}
    inspection_section = raw.get("inspection") or {}
    report_section = raw.get("report") or {}

    def _extension_set(values: Any) -> frozenset[str]:
        return frozenset(str(item).lower() for item in (values or []))

    return InspectorConfig(
        ffprobe_binary=str(ffprobe_section.get("binary", "ffprobe")),
        timeout_seconds=int(ffprobe_section.get("timeout_seconds", 30)),
        video_extensions=_extension_set(extensions_section.get("video")),
        audio_extensions=_extension_set(extensions_section.get("audio")),
        container_stream_duration_warning_seconds=float(
            inspection_section.get("container_stream_duration_warning_seconds", 0.5)
        ),
        variable_fps_tolerance=float(
            inspection_section.get("variable_fps_tolerance", 0.05)
        ),
        follow_directory_symlinks=bool(
            inspection_section.get("follow_directory_symlinks", False)
        ),
        report_schema_version=str(report_section.get("schema_version", "1.0")),
        overwrite_requires_force=bool(
            report_section.get("overwrite_requires_force", True)
        ),
    )


# ---------------------------------------------------------------------------
# File validation (pure, before any subprocess call)
# ---------------------------------------------------------------------------


def validate_media_path(path: str | Path, config: InspectorConfig) -> Path:
    """
    Raise a clear MediaInspectorError subclass unless path is a readable,
    non-empty, supported-extension regular file. Returns the resolved
    absolute path. Never touches ffprobe.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise MediaFileNotFoundError(f"File not found: {resolved}")

    if resolved.is_dir():
        raise MediaInspectorError(
            f"Expected a file but found a directory: {resolved} "
            "(use --directory to inspect a directory of media files)"
        )

    if not resolved.is_file():
        raise MediaInspectorError(f"Not a regular file: {resolved}")

    if resolved.stat().st_size == 0:
        raise MediaFileEmptyError(f"File is zero bytes: {resolved}")

    extension = resolved.suffix.lower()
    if config.media_category(extension) is None:
        raise UnsupportedMediaExtensionError(
            f"Unsupported media extension {extension or '(none)'!r} for {resolved}. "
            f"Supported: {sorted(config.supported_extensions)}"
        )

    return resolved


# ---------------------------------------------------------------------------
# Subprocess runner (mockable)
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
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise FFprobeNotFoundError(
            f"ffprobe executable not found: {command[0]!r}. "
            "Install ffmpeg/ffprobe to enable real media inspection."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFprobeTimeoutError(
            f"ffprobe timed out after {timeout}s: {' '.join(command)}"
        ) from exc

    return ProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def run_ffprobe(
    path: Path, config: InspectorConfig, *, runner: SubprocessRunner = default_runner
) -> dict[str, Any]:
    """
    Invoke ffprobe (never ffmpeg) via an injectable runner and return the
    parsed top-level JSON document. Command is always passed as a list —
    no shell string interpretation is ever used anywhere in this module.
    """
    command = [
        config.ffprobe_binary,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(path),
    ]

    result = runner(command, timeout=config.timeout_seconds)

    if result.returncode != 0:
        raise FFprobeExecutionError(
            f"ffprobe exited {result.returncode} for {path}: {result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFprobeOutputError(
            f"ffprobe returned unparseable top-level JSON for {path}: {exc}"
        ) from exc

    if not isinstance(data, dict) or "streams" not in data:
        raise FFprobeOutputError(
            f"ffprobe JSON for {path} is missing a top-level 'streams' key."
        )

    return data


# ---------------------------------------------------------------------------
# Safe parsing helpers
# ---------------------------------------------------------------------------


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _safe_rational(raw: Any) -> float | None:
    """
    Parses ffprobe rational strings like "30000/1001" or "30/1". Returns
    None (never raises) for "N/A", "0/0", a zero denominator, or any
    other unparseable value.
    """
    if raw is None:
        return None

    text = str(raw).strip()
    if not text or text.upper() == "N/A":
        return None

    if "/" in text:
        numerator_str, _, denominator_str = text.partition("/")
        numerator = _safe_float(numerator_str)
        denominator = _safe_float(denominator_str)
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    return _safe_float(text)


_safe_duration = _safe_float
_safe_bitrate = _safe_int
_safe_frame_count = _safe_int


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InspectionWarning:
    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VideoStreamInfo:
    stream_index: int
    codec_name: str | None = None
    codec_long_name: str | None = None
    profile: str | None = None
    codec_tag: str | None = None
    width: int | None = None
    height: int | None = None
    coded_width: int | None = None
    coded_height: int | None = None
    display_aspect_ratio: str | None = None
    sample_aspect_ratio: str | None = None
    pixel_format: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_range: str | None = None
    field_order: str | None = None
    fps: float | None = None
    average_fps: float | None = None
    time_base: str | None = None
    duration_seconds: float | None = None
    bitrate: int | None = None
    frame_count: int | None = None
    rotation_degrees: int | None = None
    is_attached_picture: bool = False
    language: str | None = None
    disposition_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AudioStreamInfo:
    stream_index: int
    codec_name: str | None = None
    codec_long_name: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    sample_format: str | None = None
    bitrate: int | None = None
    duration_seconds: float | None = None
    language: str | None = None
    disposition_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChapterInfo:
    chapter_id: Any
    start_seconds: float | None = None
    end_seconds: float | None = None
    title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MediaInfo:
    path: str
    filename: str
    extension: str
    file_size_bytes: int
    format_name: str | None
    format_long_name: str | None
    duration_seconds: float | None
    overall_bitrate: int | None
    start_time_seconds: float | None
    stream_count: int
    video_streams: list[VideoStreamInfo]
    audio_streams: list[AudioStreamInfo]
    chapters: list[ChapterInfo]
    has_video: bool
    has_audio: bool
    primary_video_stream_index: int | None
    primary_audio_stream_index: int | None
    warnings: list[InspectionWarning]
    raw_probe_version: str = MEDIA_INSPECTOR_SCHEMA_VERSION

    # -- derived properties (requirement 8): never stored, always
    # -- recomputed from the fields above; no rendering or file access.

    @property
    def primary_video(self) -> VideoStreamInfo | None:
        if self.primary_video_stream_index is None:
            return None
        return next(
            (s for s in self.video_streams if s.stream_index == self.primary_video_stream_index),
            None,
        )

    @property
    def primary_audio(self) -> AudioStreamInfo | None:
        if self.primary_audio_stream_index is None:
            return None
        return next(
            (s for s in self.audio_streams if s.stream_index == self.primary_audio_stream_index),
            None,
        )

    @property
    def resolution(self) -> str | None:
        video = self.primary_video
        if video is None or video.width is None or video.height is None:
            return None
        return f"{video.width}x{video.height}"

    @property
    def display_resolution_after_rotation(self) -> str | None:
        video = self.primary_video
        if video is None or video.width is None or video.height is None:
            return None
        width, height = video.width, video.height
        if video.rotation_degrees in (90, 270):
            width, height = height, width
        return f"{width}x{height}"

    @property
    def fps(self) -> float | None:
        video = self.primary_video
        if video is None:
            return None
        return video.fps if video.fps is not None else video.average_fps

    @property
    def codec_summary(self) -> str | None:
        video = self.primary_video
        if video is None:
            return None
        parts = [video.codec_name or "unknown"]
        if video.profile:
            parts.append(f"({video.profile})")
        if self.resolution:
            parts.append(self.resolution)
        if self.fps:
            parts.append(f"{self.fps:g}fps")
        return " ".join(parts)

    @property
    def audio_summary(self) -> str | None:
        audio = self.primary_audio
        if audio is None:
            return None
        parts = [audio.codec_name or "unknown"]
        if audio.sample_rate:
            parts.append(f"{audio.sample_rate}Hz")
        if audio.channel_layout:
            parts.append(audio.channel_layout)
        return " ".join(parts)

    @property
    def aspect_ratio_float(self) -> float | None:
        video = self.primary_video
        if video is None or not video.width or not video.height:
            return None
        width, height = video.width, video.height
        if video.rotation_degrees in (90, 270):
            width, height = height, width
        return width / height if height else None

    @property
    def is_vertical(self) -> bool | None:
        ratio = self.aspect_ratio_float
        return None if ratio is None else ratio < 1.0

    @property
    def is_horizontal(self) -> bool | None:
        ratio = self.aspect_ratio_float
        return None if ratio is None else ratio > 1.0

    @property
    def is_square(self) -> bool | None:
        ratio = self.aspect_ratio_float
        return None if ratio is None else math.isclose(ratio, 1.0, abs_tol=0.01)

    @property
    def duration_text(self) -> str | None:
        if self.duration_seconds is None:
            return None
        return _format_duration(self.duration_seconds)

    @property
    def file_size_text(self) -> str:
        return _format_file_size(self.file_size_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "extension": self.extension,
            "file_size_bytes": self.file_size_bytes,
            "file_size_text": self.file_size_text,
            "format_name": self.format_name,
            "format_long_name": self.format_long_name,
            "duration_seconds": self.duration_seconds,
            "duration_text": self.duration_text,
            "overall_bitrate": self.overall_bitrate,
            "start_time_seconds": self.start_time_seconds,
            "stream_count": self.stream_count,
            "video_streams": [s.to_dict() for s in self.video_streams],
            "audio_streams": [s.to_dict() for s in self.audio_streams],
            "chapters": [c.to_dict() for c in self.chapters],
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "primary_video_stream_index": self.primary_video_stream_index,
            "primary_audio_stream_index": self.primary_audio_stream_index,
            "resolution": self.resolution,
            "display_resolution_after_rotation": self.display_resolution_after_rotation,
            "fps": self.fps,
            "codec_summary": self.codec_summary,
            "audio_summary": self.audio_summary,
            "is_vertical": self.is_vertical,
            "is_horizontal": self.is_horizontal,
            "is_square": self.is_square,
            "aspect_ratio_float": self.aspect_ratio_float,
            "warnings": [w.to_dict() for w in self.warnings],
            "raw_probe_version": self.raw_probe_version,
        }


def _format_duration(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        seconds = 0.0
    total_milliseconds = round(seconds * 1000)
    hours, remainder_ms = divmod(total_milliseconds, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, milliseconds = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# Rotation detection
# ---------------------------------------------------------------------------


def _normalize_rotation(raw_degrees: float) -> int:
    snapped = round(raw_degrees / 90.0) * 90
    return int(snapped % 360)


def _extract_rotation(stream: dict[str, Any]) -> tuple[int | None, list[InspectionWarning]]:
    """
    Priority: (1) side_data_list Display Matrix rotation — ffmpeg's
    modern, authoritative source; (2) legacy tags.rotate. Never touches
    the file — this is metadata extraction only, no physical rotation.
    """
    warnings: list[InspectionWarning] = []

    display_matrix_raw: float | None = None
    for side_data in stream.get("side_data_list") or []:
        if side_data.get("side_data_type") == "Display Matrix":
            display_matrix_raw = _safe_float(side_data.get("rotation"))
            if display_matrix_raw is not None:
                break

    tags_raw = _safe_float((stream.get("tags") or {}).get("rotate"))

    display_matrix_normalized = (
        _normalize_rotation(display_matrix_raw) if display_matrix_raw is not None else None
    )
    tags_normalized = _normalize_rotation(tags_raw) if tags_raw is not None else None

    if display_matrix_normalized is not None and tags_normalized is not None:
        if display_matrix_normalized != tags_normalized:
            warnings.append(
                InspectionWarning(
                    code="contradictory_rotation_metadata",
                    message=(
                        "Display Matrix rotation "
                        f"({display_matrix_normalized}) disagrees with "
                        f"tags.rotate ({tags_normalized}); using Display "
                        "Matrix (higher priority)."
                    ),
                    field="rotation_degrees",
                )
            )
        return display_matrix_normalized, warnings

    if display_matrix_normalized is not None:
        return display_matrix_normalized, warnings

    if tags_normalized is not None:
        return tags_normalized, warnings

    return None, warnings


# ---------------------------------------------------------------------------
# Primary stream selection
# ---------------------------------------------------------------------------


def _is_attached_picture(stream: dict[str, Any]) -> bool:
    return (stream.get("disposition") or {}).get("attached_pic") == 1


def _is_default_disposition(stream: dict[str, Any]) -> bool:
    return (stream.get("disposition") or {}).get("default") == 1


def select_primary_stream_index(streams: list[dict[str, Any]], codec_type: str) -> int | None:
    """
    Deterministic priority: (1) disposition.default == 1; (2) first
    non-attached-picture stream (video only — attached cover art must not
    win over a real video stream when one exists); (3) lowest index.
    """
    candidates = [s for s in streams if s.get("codec_type") == codec_type]
    if not candidates:
        return None

    if codec_type == "video":
        non_attached = [s for s in candidates if not _is_attached_picture(s)]
        pool = non_attached if non_attached else candidates
    else:
        pool = candidates

    default_candidates = [s for s in pool if _is_default_disposition(s)]
    chosen_pool = default_candidates if default_candidates else pool

    return min(int(s.get("index", 0)) for s in chosen_pool)


# ---------------------------------------------------------------------------
# Stream / chapter builders
# ---------------------------------------------------------------------------


def _build_video_stream(stream: dict[str, Any], warnings: list[InspectionWarning]) -> VideoStreamInfo:
    index = int(stream.get("index", 0))
    tags = stream.get("tags") or {}

    r_frame_rate_raw = stream.get("r_frame_rate")
    avg_frame_rate_raw = stream.get("avg_frame_rate")
    fps = _safe_rational(r_frame_rate_raw)
    average_fps = _safe_rational(avg_frame_rate_raw)

    if r_frame_rate_raw is not None and fps is None:
        warnings.append(
            InspectionWarning(
                code="invalid_rational_value",
                message=f"stream {index}: could not parse r_frame_rate={r_frame_rate_raw!r}",
                field="fps",
            )
        )

    rotation_degrees, rotation_warnings = _extract_rotation(stream)
    warnings.extend(rotation_warnings)

    return VideoStreamInfo(
        stream_index=index,
        codec_name=stream.get("codec_name"),
        codec_long_name=stream.get("codec_long_name"),
        profile=stream.get("profile"),
        codec_tag=stream.get("codec_tag_string") or stream.get("codec_tag"),
        width=_safe_int(stream.get("width")),
        height=_safe_int(stream.get("height")),
        coded_width=_safe_int(stream.get("coded_width")),
        coded_height=_safe_int(stream.get("coded_height")),
        display_aspect_ratio=stream.get("display_aspect_ratio"),
        sample_aspect_ratio=stream.get("sample_aspect_ratio"),
        pixel_format=stream.get("pix_fmt"),
        color_space=stream.get("color_space"),
        color_transfer=stream.get("color_transfer"),
        color_primaries=stream.get("color_primaries"),
        color_range=stream.get("color_range"),
        field_order=stream.get("field_order"),
        fps=fps,
        average_fps=average_fps,
        time_base=stream.get("time_base"),
        duration_seconds=_safe_duration(stream.get("duration")),
        bitrate=_safe_bitrate(stream.get("bit_rate")),
        frame_count=_safe_frame_count(stream.get("nb_frames")),
        rotation_degrees=rotation_degrees,
        is_attached_picture=_is_attached_picture(stream),
        language=tags.get("language"),
        disposition_default=_is_default_disposition(stream),
    )


def _build_audio_stream(stream: dict[str, Any]) -> AudioStreamInfo:
    tags = stream.get("tags") or {}
    return AudioStreamInfo(
        stream_index=int(stream.get("index", 0)),
        codec_name=stream.get("codec_name"),
        codec_long_name=stream.get("codec_long_name"),
        sample_rate=_safe_int(stream.get("sample_rate")),
        channels=_safe_int(stream.get("channels")),
        channel_layout=stream.get("channel_layout"),
        sample_format=stream.get("sample_fmt"),
        bitrate=_safe_bitrate(stream.get("bit_rate")),
        duration_seconds=_safe_duration(stream.get("duration")),
        language=tags.get("language"),
        disposition_default=_is_default_disposition(stream),
    )


def _build_chapter(chapter: dict[str, Any]) -> ChapterInfo:
    tags = chapter.get("tags") or {}
    return ChapterInfo(
        chapter_id=chapter.get("id"),
        start_seconds=_safe_duration(chapter.get("start_time")),
        end_seconds=_safe_duration(chapter.get("end_time")),
        title=tags.get("title"),
    )


# ---------------------------------------------------------------------------
# Orchestration: single-file inspection
# ---------------------------------------------------------------------------


def build_media_info(
    resolved_path: Path,
    probe_data: dict[str, Any],
    config: InspectorConfig,
) -> MediaInfo:
    """
    Pure transform: probe_data (already-parsed ffprobe JSON) + filesystem
    facts about resolved_path -> a fully normalized MediaInfo. No
    subprocess call happens here — run_ffprobe() already ran.
    """
    warnings: list[InspectionWarning] = []

    raw_streams = probe_data.get("streams") or []
    format_section = probe_data.get("format") or {}
    chapters_raw = probe_data.get("chapters") or []

    video_streams = [
        _build_video_stream(s, warnings) for s in raw_streams if s.get("codec_type") == "video"
    ]
    audio_streams = [_build_audio_stream(s) for s in raw_streams if s.get("codec_type") == "audio"]
    chapters = [_build_chapter(c) for c in chapters_raw]

    has_video = len(video_streams) > 0
    has_audio = len(audio_streams) > 0

    primary_video_index = select_primary_stream_index(raw_streams, "video")
    primary_audio_index = select_primary_stream_index(raw_streams, "audio")

    format_duration = _safe_duration(format_section.get("duration"))
    overall_bitrate = _safe_bitrate(format_section.get("bit_rate"))
    start_time = _safe_duration(format_section.get("start_time"))

    extension = resolved_path.suffix.lower()
    media_category = config.media_category(extension)

    if media_category == "video" and not has_video:
        warnings.append(
            InspectionWarning(code="no_video_stream", message="No video stream found.")
        )

    if not has_audio:
        warnings.append(
            InspectionWarning(code="no_audio_stream", message="No audio stream found.")
        )

    if len(video_streams) > 1:
        warnings.append(
            InspectionWarning(
                code="multiple_video_streams",
                message=f"Found {len(video_streams)} video streams; using the primary one.",
            )
        )

    if len(audio_streams) > 1:
        warnings.append(
            InspectionWarning(
                code="multiple_audio_streams",
                message=f"Found {len(audio_streams)} audio streams; using the primary one.",
            )
        )

    if any(v.is_attached_picture for v in video_streams):
        warnings.append(
            InspectionWarning(
                code="attached_picture_present",
                message="One or more video streams are attached pictures (cover art).",
            )
        )

    if format_duration is None:
        warnings.append(
            InspectionWarning(
                code="missing_duration", message="Container duration is missing.", field="duration_seconds"
            )
        )

    if overall_bitrate is None:
        warnings.append(
            InspectionWarning(
                code="missing_bitrate", message="Overall bitrate is missing.", field="overall_bitrate"
            )
        )

    primary_video = next((v for v in video_streams if v.stream_index == primary_video_index), None)

    if primary_video is not None:
        if primary_video.fps and primary_video.average_fps:
            if primary_video.fps > 0:
                relative_diff = abs(primary_video.fps - primary_video.average_fps) / primary_video.fps
                if relative_diff > config.variable_fps_tolerance:
                    warnings.append(
                        InspectionWarning(
                            code="variable_frame_rate_suspected",
                            message=(
                                f"r_frame_rate ({primary_video.fps:g}) and "
                                f"avg_frame_rate ({primary_video.average_fps:g}) differ "
                                "by more than the configured tolerance."
                            ),
                            field="fps",
                        )
                    )

        if primary_video.duration_seconds is not None and format_duration is not None:
            if abs(primary_video.duration_seconds - format_duration) > (
                config.container_stream_duration_warning_seconds
            ):
                warnings.append(
                    InspectionWarning(
                        code="container_stream_duration_mismatch",
                        message=(
                            f"Container duration ({format_duration:g}s) differs from "
                            f"primary video stream duration ({primary_video.duration_seconds:g}s)."
                        ),
                        field="duration_seconds",
                    )
                )

        unsupported_color_markers = {None, "", "unknown", "reserved"}
        color_fields = (
            primary_video.color_space,
            primary_video.color_transfer,
            primary_video.color_primaries,
        )
        if all((c in unsupported_color_markers) for c in color_fields):
            warnings.append(
                InspectionWarning(
                    code="unsupported_color_metadata",
                    message="No usable color space/transfer/primaries metadata found.",
                )
            )

    return MediaInfo(
        path=str(resolved_path),
        filename=resolved_path.name,
        extension=extension,
        file_size_bytes=resolved_path.stat().st_size,
        format_name=format_section.get("format_name"),
        format_long_name=format_section.get("format_long_name"),
        duration_seconds=format_duration,
        overall_bitrate=overall_bitrate,
        start_time_seconds=start_time,
        stream_count=len(raw_streams),
        video_streams=video_streams,
        audio_streams=audio_streams,
        chapters=chapters,
        has_video=has_video,
        has_audio=has_audio,
        primary_video_stream_index=primary_video_index,
        primary_audio_stream_index=primary_audio_index,
        warnings=warnings,
    )


def inspect_file(
    path: str | Path,
    config: InspectorConfig,
    *,
    runner: SubprocessRunner = default_runner,
) -> MediaInfo:
    """
    Validate path, run ffprobe (never ffmpeg), and return a fully
    normalized MediaInfo. Read-only: never modifies path.
    """
    resolved_path = validate_media_path(path, config)
    probe_data = run_ffprobe(resolved_path, config, runner=runner)
    return build_media_info(resolved_path, probe_data, config)


# ---------------------------------------------------------------------------
# Directory inspection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FileInspectionResult:
    path: str
    success: bool
    media: MediaInfo | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "success": self.success,
            "media": self.media.to_dict() if self.media is not None else None,
            "error": self.error,
        }


@dataclass(slots=True)
class DirectoryInspectionResult:
    directory: str
    recursive: bool
    total_files: int
    successful: int
    failed: int
    results: list[FileInspectionResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "recursive": self.recursive,
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "results": [r.to_dict() for r in self.results],
        }


def _iter_candidate_files(directory: Path, *, recursive: bool, follow_symlinks: bool) -> list[Path]:
    if recursive:
        candidates = [
            entry
            for entry in directory.rglob("*")
            if entry.is_file() or (follow_symlinks and entry.is_symlink() and entry.is_file())
        ]
    else:
        candidates = [entry for entry in directory.iterdir() if entry.is_file()]

    return sorted(candidates, key=lambda entry: str(entry))


def inspect_directory(
    directory: str | Path,
    config: InspectorConfig,
    *,
    recursive: bool = False,
    runner: SubprocessRunner = default_runner,
) -> DirectoryInspectionResult:
    """
    Inspect every supported media file directly under (or, with
    recursive=True, anywhere beneath) directory, in deterministic sorted
    order. Unsupported files are silently skipped. One file's failure
    never aborts inspection of the others.
    """
    resolved_directory = Path(directory).expanduser().resolve()

    if not resolved_directory.is_dir():
        raise MediaInspectorError(f"Not a directory: {resolved_directory}")

    candidates = _iter_candidate_files(
        resolved_directory,
        recursive=recursive,
        follow_symlinks=config.follow_directory_symlinks,
    )

    supported = [
        entry for entry in candidates if config.media_category(entry.suffix.lower()) is not None
    ]

    results: list[FileInspectionResult] = []
    for entry in supported:
        try:
            media = inspect_file(entry, config, runner=runner)
            results.append(FileInspectionResult(path=str(entry), success=True, media=media, error=None))
        except MediaInspectorError as exc:
            results.append(FileInspectionResult(path=str(entry), success=False, media=None, error=str(exc)))

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    return DirectoryInspectionResult(
        directory=str(resolved_directory),
        recursive=recursive,
        total_files=len(results),
        successful=successful,
        failed=failed,
        results=results,
    )


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def build_single_file_envelope(media: MediaInfo, config: InspectorConfig) -> dict[str, Any]:
    return {
        "schema_version": config.report_schema_version,
        "inspected_at": _now_iso(),
        "media": media.to_dict(),
    }


def build_directory_envelope(
    result: DirectoryInspectionResult, config: InspectorConfig
) -> dict[str, Any]:
    return {
        "schema_version": config.report_schema_version,
        "inspected_at": _now_iso(),
        **result.to_dict(),
    }


def write_report(
    payload: dict[str, Any],
    output_path: str | Path,
    *,
    force: bool,
    input_path: str | Path | None = None,
) -> Path:
    resolved_output = Path(output_path).expanduser().resolve()

    if input_path is not None:
        resolved_input = Path(input_path).expanduser().resolve()
        if resolved_output == resolved_input:
            raise UnsafeReportPathError(
                f"--output path must not equal the inspected input path: {resolved_output}"
            )

    if resolved_output.exists() and not force:
        raise ReportAlreadyExistsError(
            f"{resolved_output} already exists; pass --force to overwrite."
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = resolved_output.with_suffix(resolved_output.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(resolved_output)

    return resolved_output


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------


def format_human_report(media: MediaInfo) -> str:
    lines: list[str] = []
    lines.append(f"File:             {media.path}")
    lines.append(f"Size:             {media.file_size_text} ({media.file_size_bytes} bytes)")
    lines.append(f"Container:        {media.format_name or 'unknown'} ({media.format_long_name or '?'})")
    lines.append(f"Duration:         {media.duration_text or 'unknown'}")
    lines.append(f"Overall bitrate:  {media.overall_bitrate if media.overall_bitrate is not None else 'unknown'}")
    lines.append("")

    video = media.primary_video
    if video is not None:
        lines.append("Primary video:")
        lines.append(f"  codec:              {video.codec_name}")
        lines.append(f"  resolution:         {media.resolution}")
        lines.append(f"  display resolution: {media.display_resolution_after_rotation}")
        lines.append(f"  aspect ratio:       {media.aspect_ratio_float}")
        lines.append(f"  fps:                {media.fps}")
        lines.append(f"  pixel format:       {video.pixel_format}")
        lines.append(f"  rotation:           {video.rotation_degrees}")
        lines.append(f"  bitrate:            {video.bitrate}")
        lines.append(f"  frame count:        {video.frame_count}")
        lines.append(
            "  color:              "
            f"space={video.color_space} transfer={video.color_transfer} "
            f"primaries={video.color_primaries} range={video.color_range}"
        )
        lines.append("")

    audio = media.primary_audio
    if audio is not None:
        lines.append("Primary audio:")
        lines.append(f"  codec:           {audio.codec_name}")
        lines.append(f"  sample rate:     {audio.sample_rate}")
        lines.append(f"  channels:        {audio.channels}")
        lines.append(f"  channel layout:  {audio.channel_layout}")
        lines.append(f"  bitrate:         {audio.bitrate}")
        lines.append("")

    lines.append("Streams:")
    lines.append(f"  video count: {len(media.video_streams)}")
    lines.append(f"  audio count: {len(media.audio_streams)}")
    lines.append("")

    lines.append(f"Warnings ({len(media.warnings)}):")
    if media.warnings:
        for warning in media.warnings:
            lines.append(f"  - [{warning.code}] {warning.message}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Media Inspector (Phase 11A.1). Read-only ffprobe "
            "metadata extraction for local video/audio files. Never "
            "renders, transcodes, or modifies any media file."
        )
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Path to a single media file to inspect.",
    )
    parser.add_argument(
        "--directory",
        default=None,
        help="Path to a directory of media files to inspect.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="With --directory, recurse into subdirectories.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable JSON envelope instead of a human report.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the JSON envelope to this path. Never written unless supplied.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing --output report file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate config/media/inspector.yaml.",
    )

    arguments = parser.parse_args(argv)

    if bool(arguments.file_path) == bool(arguments.directory):
        parser.error("Provide exactly one of FILE_PATH or --directory.")

    return arguments


def _run(arguments: argparse.Namespace, *, runner: SubprocessRunner = default_runner) -> int:
    try:
        config = load_inspector_config(arguments.config)
    except MediaInspectorConfigError as exc:
        print(f"[MediaInspector] {exc}")
        return 1

    if arguments.directory:
        try:
            result = inspect_directory(
                arguments.directory, config, recursive=arguments.recursive, runner=runner
            )
        except MediaInspectorError as exc:
            print(f"[MediaInspector] {exc}")
            return 1

        envelope = build_directory_envelope(result, config)

        if arguments.output:
            try:
                written_path = write_report(envelope, arguments.output, force=arguments.force)
            except MediaInspectorError as exc:
                print(f"[MediaInspector] {exc}")
                return 1
            print(f"Report written: {written_path}")

        if arguments.json:
            print(json.dumps(envelope, ensure_ascii=False, indent=2))
        else:
            print(f"Directory:   {result.directory}")
            print(f"Recursive:   {result.recursive}")
            print(f"Total files: {result.total_files}")
            print(f"Successful:  {result.successful}")
            print(f"Failed:      {result.failed}")
            for file_result in result.results:
                status = "OK" if file_result.success else f"FAILED: {file_result.error}"
                print(f"  {file_result.path}: {status}")

        if result.total_files == 0:
            return 1
        if result.successful == 0:
            return 1
        return 0

    try:
        media = inspect_file(arguments.file_path, config, runner=runner)
    except MediaInspectorError as exc:
        print(f"[MediaInspector] {exc}")
        return 1

    envelope = build_single_file_envelope(media, config)

    if arguments.output:
        try:
            written_path = write_report(
                envelope, arguments.output, force=arguments.force, input_path=media.path
            )
        except MediaInspectorError as exc:
            print(f"[MediaInspector] {exc}")
            return 1
        print(f"Report written: {written_path}")

    if arguments.json:
        print(json.dumps(envelope, ensure_ascii=False, indent=2))
    else:
        print(format_human_report(media))

    return 0


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    exit_code = _run(arguments)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
