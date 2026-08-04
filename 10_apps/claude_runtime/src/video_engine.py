from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

# Phase 11A.3 — FFmpeg Video Engine. A reusable, local ffmpeg core:
# command construction, execution, concat planning, manifest generation,
# and output verification. This module owns ffmpeg/ffprobe executable
# configuration and process execution ONLY. It never publishes, never
# uploads, never imports Playwright/InstagramSession, never implements
# Instagram-specific validation, captions, hashtags, music selection,
# subtitle generation, logo/outro design, or content planning.
DISALLOWED_ACTIONS = (
    "publish",
    "upload",
    "share",
    "send",
    "post",
    "add_music",
    "add_subtitles",
    "add_logo",
    "add_watermark",
    "add_transition",
)

DEFAULT_ENGINE_CONFIG_RELATIVE_PATH = Path("config") / "video" / "engine.yaml"

PLAN_LOSSLESS_COPY = "lossless_copy"
PLAN_NORMALIZED_RENDER = "normalized_render"
PLAN_REJECTED = "rejected"


def _runtime_root() -> Path:
    """
    video_engine.py location: 10_apps/claude_runtime/src/video_engine.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VideoEngineError(RuntimeError):
    """Base error for the Phase 11A.3 FFmpeg Video Engine."""


class VideoEngineConfigError(VideoEngineError):
    """Raised when config/video/engine.yaml is missing or invalid."""


class VideoInputError(VideoEngineError):
    """Raised when an input file fails path-safety validation."""


class DuplicateVideoInputError(VideoEngineError):
    """Raised when two inputs resolve to the same real file."""


class UnsafeVideoOutputError(VideoEngineError):
    """Raised when the output path is unsafe (equals an input, unwritable parent, etc.)."""


class VideoOutputExistsError(VideoEngineError):
    """Raised when the output already exists and force was not given."""


class VideoPlanRejectedError(VideoEngineError):
    """Raised when execute_plan() is called on a REJECTED ConcatPlan."""


class FFmpegNotFoundError(VideoEngineError):
    """Raised when the configured ffmpeg/ffprobe binary cannot be found/executed."""


class FFmpegExecutionError(VideoEngineError):
    """Raised when ffmpeg exits non-zero."""


class FFmpegTimeoutError(VideoEngineError):
    """Raised when ffmpeg does not finish within the configured timeout."""


class VideoOutputVerificationError(VideoEngineError):
    """Raised when the output file is missing or zero bytes after a reported-successful run."""


class ConcatManifestError(VideoEngineError):
    """Raised when the concat manifest cannot be safely written."""


class UnsupportedAudioLayoutError(VideoEngineError):
    """Raised when inputs have an audio layout this engine cannot safely concatenate (e.g. mixed presence)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VideoEngineConfig:
    ffmpeg_binary: str = "ffmpeg"
    ffmpeg_timeout_seconds: int = 300
    ffmpeg_loglevel: str = "error"
    ffmpeg_hide_banner: bool = True

    ffprobe_binary: str = "ffprobe"
    ffprobe_timeout_seconds: int = 30

    overwrite_requires_force: bool = True
    verify_non_zero_bytes: bool = True
    temporary_directory_name: str = ".video_engine_tmp"
    cleanup_temporary_files: bool = True

    default_video_codec: str = "libx264"
    default_pixel_format: str = "yuv420p"
    default_preset: str = "medium"
    default_crf: int = 18
    default_video_bitrate: str = "10M"
    default_fps: float = 30
    default_width: int = 1080
    default_height: int = 1920
    faststart: bool = True

    default_audio_codec: str = "aac"
    default_audio_bitrate: str = "192k"
    default_sample_rate: int = 48000
    default_channels: int = 2
    # Reserved for a future phase — never applied in Phase 11A.3. See
    # config/video/engine.yaml's comment on this key.
    insert_silence_when_missing: bool = False

    prefer_stream_copy: bool = True
    normalize_when_needed: bool = True
    require_sequential_inputs: bool = True
    fps_tolerance: float = 0.05


def default_engine_config_path() -> Path:
    return _runtime_root() / DEFAULT_ENGINE_CONFIG_RELATIVE_PATH


def load_engine_config(config_path: str | Path | None = None) -> VideoEngineConfig:
    """Load config/video/engine.yaml (or an alternate path) into a VideoEngineConfig.
    Raises VideoEngineConfigError if the file is missing or invalid."""
    path = Path(config_path) if config_path else default_engine_config_path()

    if not path.exists():
        raise VideoEngineConfigError(f"Video engine config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise VideoEngineConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise VideoEngineConfigError(f"Video engine config is empty or invalid: {path}")

    ffmpeg_section = raw.get("ffmpeg") or {}
    ffprobe_section = raw.get("ffprobe") or {}
    output_section = raw.get("output") or {}
    video_section = raw.get("video") or {}
    audio_section = raw.get("audio") or {}
    concat_section = raw.get("concat") or {}

    return VideoEngineConfig(
        ffmpeg_binary=str(ffmpeg_section.get("binary", "ffmpeg")),
        ffmpeg_timeout_seconds=int(ffmpeg_section.get("timeout_seconds", 300)),
        ffmpeg_loglevel=str(ffmpeg_section.get("loglevel", "error")),
        ffmpeg_hide_banner=bool(ffmpeg_section.get("hide_banner", True)),
        ffprobe_binary=str(ffprobe_section.get("binary", "ffprobe")),
        ffprobe_timeout_seconds=int(ffprobe_section.get("timeout_seconds", 30)),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        verify_non_zero_bytes=bool(output_section.get("verify_non_zero_bytes", True)),
        temporary_directory_name=str(
            output_section.get("temporary_directory_name", ".video_engine_tmp")
        ),
        cleanup_temporary_files=bool(output_section.get("cleanup_temporary_files", True)),
        default_video_codec=str(video_section.get("default_codec", "libx264")),
        default_pixel_format=str(video_section.get("default_pixel_format", "yuv420p")),
        default_preset=str(video_section.get("default_preset", "medium")),
        default_crf=int(video_section.get("default_crf", 18)),
        default_video_bitrate=str(video_section.get("default_bitrate", "10M")),
        default_fps=float(video_section.get("default_fps", 30)),
        default_width=int(video_section.get("default_width", 1080)),
        default_height=int(video_section.get("default_height", 1920)),
        faststart=bool(video_section.get("faststart", True)),
        default_audio_codec=str(audio_section.get("default_codec", "aac")),
        default_audio_bitrate=str(audio_section.get("default_bitrate", "192k")),
        default_sample_rate=int(audio_section.get("default_sample_rate", 48000)),
        default_channels=int(audio_section.get("default_channels", 2)),
        insert_silence_when_missing=bool(audio_section.get("insert_silence_when_missing", False)),
        prefer_stream_copy=bool(concat_section.get("prefer_stream_copy", True)),
        normalize_when_needed=bool(concat_section.get("normalize_when_needed", True)),
        require_sequential_inputs=bool(concat_section.get("require_sequential_inputs", True)),
        fps_tolerance=float(concat_section.get("fps_tolerance", 0.05)),
    )


# ---------------------------------------------------------------------------
# Runner (injectable, mockable)
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
    Never logs environment variables or any secret — only the command
    list (which never contains credentials) and captured output are ever
    touched. Raises FFmpegNotFoundError/FFmpegTimeoutError; never retries.
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
            f"Executable not found: {command[0]!r}. Install ffmpeg/ffprobe to enable real rendering."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegTimeoutError(
            f"Command timed out after {timeout}s: {' '.join(command)}"
        ) from exc

    return ProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VideoStreamSpec:
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec_name: str | None = None
    pixel_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AudioStreamSpec:
    present: bool = False
    codec_name: str | None = None
    sample_rate: int | None = None
    channels: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VideoInput:
    path: Path
    video: VideoStreamSpec = field(default_factory=VideoStreamSpec)
    audio: AudioStreamSpec = field(default_factory=AudioStreamSpec)
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "video": self.video.to_dict(),
            "audio": self.audio.to_dict(),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class VideoOutputSpec:
    path: Path
    video_codec: str | None = None
    pixel_format: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    video_bitrate: str | None = None
    crf: int | None = None
    preset: str | None = None
    audio_codec: str | None = None
    audio_bitrate: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    faststart: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class ConcatPlan:
    plan_type: str
    inputs: list[VideoInput]
    output: VideoOutputSpec
    reasons: list[str] = field(default_factory=list)
    target_video: VideoStreamSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_type": self.plan_type,
            "inputs": [i.to_dict() for i in self.inputs],
            "output": self.output.to_dict(),
            "reasons": list(self.reasons),
            "target_video": self.target_video.to_dict() if self.target_video else None,
        }


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    started_at: str
    finished_at: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VideoEngineResult:
    plan_type: str
    input_files: list[str]
    output_file: str
    command: list[str]
    started_at: str
    finished_at: str
    duration_seconds: float
    return_code: int
    stdout: str
    stderr: str
    output_exists: bool
    output_size_bytes: int | None
    manifest_path: str | None
    temporary_files: list[str]
    cleanup_result: str
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# VideoEngine
# ---------------------------------------------------------------------------


class VideoEngine:
    """
    Reusable, local ffmpeg core: concat planning, command construction,
    execution via an injectable runner, and output verification. Owns no
    Instagram-specific validation, publishing, upload, caption, hashtag,
    music, subtitle, logo, or content-planning logic — callers (e.g.
    src/reel_builder.py) own that and hand this class typed VideoInput/
    VideoOutputSpec objects.
    """

    def __init__(self, config: VideoEngineConfig, *, runner: SubprocessRunner = default_runner) -> None:
        self.config = config
        self.runner = runner

    # -- path safety --------------------------------------------------------

    def _resolve_and_check_input(self, video_input: VideoInput) -> Path:
        path = video_input.path

        if not path.exists():
            raise VideoInputError(f"Input file not found: {path}")

        if not path.is_file():
            raise VideoInputError(f"Input is not a regular file: {path}")

        if path.stat().st_size == 0:
            raise VideoInputError(f"Input file is zero bytes: {path}")

        return path.resolve()

    def validate_inputs(self, inputs: list[VideoInput], output: VideoOutputSpec) -> list[Path]:
        """
        Validate every input and the output path per Phase 11A.3's
        path-safety rules: inputs exist, are regular files, are non-zero
        bytes; duplicate *resolved* input paths are rejected (this is
        also what makes a symlink input safe — it is compared by its
        real target); the output must not resolve to any input; the
        output's parent must already be writable or not yet exist.
        Returns the resolved input paths in the same order as `inputs`.
        Raises VideoInputError/DuplicateVideoInputError/
        UnsafeVideoOutputError. Never touches ffmpeg/ffprobe.
        """
        if not inputs:
            raise VideoInputError("At least one video input is required.")

        resolved_inputs: list[Path] = []
        seen: set[Path] = set()
        duplicates: set[Path] = set()

        for video_input in inputs:
            resolved = self._resolve_and_check_input(video_input)
            if resolved in seen:
                duplicates.add(resolved)
            seen.add(resolved)
            resolved_inputs.append(resolved)

        if duplicates:
            raise DuplicateVideoInputError(
                f"Duplicate resolved input path(s): {sorted(str(d) for d in duplicates)}"
            )

        output_resolved = output.path.resolve()
        if output_resolved in resolved_inputs:
            raise UnsafeVideoOutputError(
                f"Output path must not be one of the input files: {output.path}"
            )

        parent = output.path.parent
        if parent.exists() and not parent.is_dir():
            raise UnsafeVideoOutputError(f"Output parent exists but is not a directory: {parent}")

        return resolved_inputs

    # -- planning -------------------------------------------------------------

    def build_concat_plan(self, inputs: list[VideoInput], output: VideoOutputSpec) -> ConcatPlan:
        """
        Pure planning — never executes ffmpeg. Deterministically decides
        LOSSLESS_COPY / NORMALIZED_RENDER / REJECTED for concatenating
        `inputs` (in the given order) into `output`, returning the
        reasons behind that decision. Validates inputs first (see
        validate_inputs).
        """
        self.validate_inputs(inputs, output)

        video_specs = [i.video for i in inputs]
        audio_specs = [i.audio for i in inputs]

        audio_presence = {a.present for a in audio_specs}

        if len(audio_presence) > 1:
            # Phase 11A.3 never synthesizes silence for clips missing
            # audio — a mixed layout always fails clearly, regardless of
            # config.insert_silence_when_missing (reserved, unused).
            return ConcatPlan(
                plan_type=PLAN_REJECTED,
                inputs=inputs,
                output=output,
                reasons=["audio_presence_mismatch"],
            )

        reasons: list[str] = []

        known_resolutions = {
            (v.width, v.height) for v in video_specs if v.width is not None and v.height is not None
        }
        if len(known_resolutions) > 1:
            reasons.append("resolution_mismatch")

        known_fps = [v.fps for v in video_specs if v.fps is not None]
        if len(known_fps) > 1:
            reference = known_fps[0]
            if any(abs(value - reference) > self.config.fps_tolerance for value in known_fps[1:]):
                reasons.append("fps_mismatch")

        known_codecs = {v.codec_name for v in video_specs if v.codec_name is not None}
        if len(known_codecs) > 1:
            reasons.append("codec_mismatch")

        known_pixel_formats = {v.pixel_format for v in video_specs if v.pixel_format is not None}
        if len(known_pixel_formats) > 1:
            reasons.append("pixel_format_mismatch")

        has_audio = next(iter(audio_presence)) if audio_presence else False
        if has_audio:
            known_audio_codecs = {a.codec_name for a in audio_specs if a.codec_name is not None}
            if len(known_audio_codecs) > 1:
                reasons.append("audio_codec_mismatch")

        target_video = video_specs[0] if reasons else None

        if not reasons:
            return ConcatPlan(
                plan_type=PLAN_LOSSLESS_COPY,
                inputs=inputs,
                output=output,
                reasons=["compatible_for_stream_copy"],
            )

        if not self.config.normalize_when_needed:
            return ConcatPlan(
                plan_type=PLAN_REJECTED,
                inputs=inputs,
                output=output,
                reasons=[*reasons, "normalization_disabled"],
            )

        return ConcatPlan(
            plan_type=PLAN_NORMALIZED_RENDER,
            inputs=inputs,
            output=output,
            reasons=reasons,
            target_video=target_video,
        )

    # -- command construction --------------------------------------------

    def build_lossless_concat_command(
        self, plan: ConcatPlan, manifest_path: Path, *, force: bool
    ) -> list[str]:
        """
        Builds the ffmpeg concat-demuxer command for a LOSSLESS_COPY plan:
        `ffmpeg [-hide_banner] -v LEVEL {-y|-n} -f concat -safe 0 -i
        MANIFEST -c copy [-movflags +faststart] OUTPUT`. Does not execute
        anything.
        """
        command = [self.config.ffmpeg_binary]

        if self.config.ffmpeg_hide_banner:
            command.append("-hide_banner")

        command.extend(["-v", self.config.ffmpeg_loglevel])
        command.append("-y" if force else "-n")
        command.extend(["-f", "concat", "-safe", "0", "-i", str(manifest_path), "-c", "copy"])

        faststart = plan.output.faststart if plan.output.faststart is not None else self.config.faststart
        if faststart:
            command.extend(["-movflags", "+faststart"])

        command.append(str(plan.output.path))
        return command

    def build_normalized_concat_command(self, plan: ConcatPlan, *, force: bool) -> list[str]:
        """
        Builds the ffmpeg filter_complex command for a NORMALIZED_RENDER
        plan: one -i per input (order preserved), scale/fps/pixel-format
        normalization per stream, then a single concat filter, encoded
        with the configured (or output-spec-overridden) codec/CRF/preset/
        bitrate/audio settings. Does not execute anything.
        """
        command = [self.config.ffmpeg_binary]

        if self.config.ffmpeg_hide_banner:
            command.append("-hide_banner")

        command.extend(["-v", self.config.ffmpeg_loglevel])
        command.append("-y" if force else "-n")

        for video_input in plan.inputs:
            command.extend(["-i", str(video_input.path)])

        width = plan.output.width or (plan.target_video.width if plan.target_video else None) or self.config.default_width
        height = plan.output.height or (plan.target_video.height if plan.target_video else None) or self.config.default_height
        fps = plan.output.fps or self.config.default_fps
        pixel_format = plan.output.pixel_format or self.config.default_pixel_format

        has_audio = all(i.audio.present for i in plan.inputs) if plan.inputs else False

        video_labels: list[str] = []
        filter_parts: list[str] = []

        for index, _video_input in enumerate(plan.inputs):
            video_label = f"v{index}"
            filter_parts.append(
                f"[{index}:v]scale={width}:{height},fps={fps},setsar=1,format={pixel_format}[{video_label}]"
            )
            video_labels.append(f"[{video_label}]")

        stream_count = len(plan.inputs)

        if has_audio:
            audio_labels = [f"[{index}:a]" for index in range(stream_count)]
            concat_inputs = "".join(f"{v}{a}" for v, a in zip(video_labels, audio_labels))
            filter_parts.append(f"{concat_inputs}concat=n={stream_count}:v=1:a=1[outv][outa]")
        else:
            concat_inputs = "".join(video_labels)
            filter_parts.append(f"{concat_inputs}concat=n={stream_count}:v=1:a=0[outv]")

        command.extend(["-filter_complex", ";".join(filter_parts)])
        command.extend(["-map", "[outv]"])
        if has_audio:
            command.extend(["-map", "[outa]"])

        command.extend(["-c:v", plan.output.video_codec or self.config.default_video_codec])
        command.extend(["-preset", plan.output.preset or self.config.default_preset])

        crf = plan.output.crf if plan.output.crf is not None else self.config.default_crf
        if crf is not None:
            command.extend(["-crf", str(crf)])

        bitrate = plan.output.video_bitrate or self.config.default_video_bitrate
        if bitrate:
            command.extend(["-b:v", bitrate])

        if has_audio:
            command.extend(["-c:a", plan.output.audio_codec or self.config.default_audio_codec])
            command.extend(["-ar", str(plan.output.sample_rate or self.config.default_sample_rate)])
            command.extend(["-ac", str(plan.output.channels or self.config.default_channels)])

        faststart = plan.output.faststart if plan.output.faststart is not None else self.config.faststart
        if faststart:
            command.extend(["-movflags", "+faststart"])

        command.append(str(plan.output.path))
        return command

    # -- concat manifest ------------------------------------------------------

    @staticmethod
    def _escape_manifest_path(path: Path) -> str:
        # ffmpeg's concat demuxer file format: each line is
        # `file '<path>'`, with literal single quotes inside the path
        # escaped as '\''. Unicode/space characters need no extra
        # handling — only the single-quote delimiter is special here.
        return str(path).replace("'", "'\\''")

    def write_concat_manifest(self, inputs: list[VideoInput], manifest_path: Path) -> Path:
        """
        Atomically writes an ffmpeg concat-demuxer manifest listing every
        input path (in order), one `file '<path>'` line per input, safely
        escaped. Refuses if manifest_path resolves to one of the input
        paths. Written via temp-file + Path.replace() so a reader never
        observes a partially-written manifest.
        """
        resolved_inputs = {video_input.path.resolve() for video_input in inputs}
        if manifest_path.resolve() in resolved_inputs:
            raise ConcatManifestError(
                f"Concat manifest path must not equal an input media path: {manifest_path}"
            )

        lines = [f"file '{self._escape_manifest_path(video_input.path)}'" for video_input in inputs]
        content = "\n".join(lines) + "\n"

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(manifest_path)
        return manifest_path

    # -- cleanup ----------------------------------------------------------

    def cleanup_temporary_files(self, paths: list[Path]) -> str:
        """
        Removes each path in `paths` if config.cleanup_temporary_files is
        true; otherwise leaves them in place. Returns "cleaned" or
        "retained". Missing files are ignored (not an error).
        """
        if not self.config.cleanup_temporary_files:
            return "retained"

        for path in paths:
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass

        return "cleaned"

    # -- output verification ------------------------------------------------

    def verify_output(self, output_path: Path) -> tuple[bool, int]:
        """Read-only check: returns (exists, size_bytes). size_bytes is 0
        when the file does not exist. Never modifies output_path."""
        if not output_path.is_file():
            return False, 0
        return True, output_path.stat().st_size

    # -- execution ----------------------------------------------------------

    def execute_plan(self, plan: ConcatPlan, *, force: bool = False) -> VideoEngineResult:
        """
        Executes a ConcatPlan exactly once: validates the plan/inputs,
        constructs the appropriate ffmpeg command (lossless concat demuxer
        or normalized filter_complex), runs it via the injected runner,
        verifies the output, cleans up temporary files, and returns a
        VideoEngineResult. Never retries, never invokes a fallback
        command, never deletes a pre-existing output before a confirmed
        successful run. Raises VideoPlanRejectedError/VideoOutputExistsError/
        FFmpegNotFoundError/FFmpegExecutionError/FFmpegTimeoutError/
        VideoOutputVerificationError on failure.
        """
        if plan.plan_type == PLAN_REJECTED:
            raise VideoPlanRejectedError(f"Plan was rejected: {plan.reasons}")

        self.validate_inputs(plan.inputs, plan.output)

        if plan.output.path.exists() and force is False and self.config.overwrite_requires_force:
            raise VideoOutputExistsError(
                f"{plan.output.path} already exists; pass force=True to overwrite."
            )

        temp_dir = plan.output.path.parent / self.config.temporary_directory_name
        temporary_files: list[Path] = []
        manifest_path: Path | None = None

        if plan.plan_type == PLAN_LOSSLESS_COPY:
            temp_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = temp_dir / f"{plan.output.path.stem}_concat_manifest.txt"
            self.write_concat_manifest(plan.inputs, manifest_path)
            temporary_files.append(manifest_path)
            command = self.build_lossless_concat_command(plan, manifest_path, force=force)
        else:
            command = self.build_normalized_concat_command(plan, force=force)

        started_at = _now_iso()
        start_monotonic = time.monotonic()

        try:
            process_result = self.runner(command, timeout=self.config.ffmpeg_timeout_seconds)
        except VideoEngineError:
            self.cleanup_temporary_files(temporary_files)
            raise

        finished_at = _now_iso()
        duration_seconds = time.monotonic() - start_monotonic

        if process_result.returncode != 0:
            self.cleanup_temporary_files(temporary_files)
            raise FFmpegExecutionError(
                f"ffmpeg exited {process_result.returncode}: {process_result.stderr.strip()[-2000:]}"
            )

        output_exists, output_size_bytes = self.verify_output(plan.output.path)
        cleanup_result = self.cleanup_temporary_files(temporary_files)

        if not output_exists:
            raise VideoOutputVerificationError(
                f"ffmpeg reported success but no output file exists: {plan.output.path}"
            )

        if self.config.verify_non_zero_bytes and output_size_bytes == 0:
            raise VideoOutputVerificationError(f"Output file is zero bytes: {plan.output.path}")

        return VideoEngineResult(
            plan_type=plan.plan_type,
            input_files=[str(i.path) for i in plan.inputs],
            output_file=str(plan.output.path),
            command=command,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            return_code=process_result.returncode,
            stdout=process_result.stdout,
            stderr=process_result.stderr,
            output_exists=output_exists,
            output_size_bytes=output_size_bytes,
            manifest_path=str(manifest_path) if manifest_path else None,
            temporary_files=[str(p) for p in temporary_files],
            cleanup_result=cleanup_result,
            warnings=[],
            error=None,
        )
