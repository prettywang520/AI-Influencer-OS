from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from . import media_inspector

# Phase 11B — Music Mixer Engine. A standalone, local audio-mixing
# engine: combines one rendered video with one background-music file
# into a new video. Handles background music only. Never publishes,
# never uploads, never imports Playwright/InstagramSession, never
# implements subtitles, logos, transitions, text overlays, voice
# generation, music recommendation/selection, or music downloading. This
# module is intentionally standalone — it does not import
# src/video_engine.py (see the Phase 11B plan's Reel Builder integration
# recommendation for how a future phase could wire the two together).
DISALLOWED_ACTIONS = (
    "publish",
    "upload",
    "share",
    "send",
    "post",
    "download_music",
    "select_music",
    "add_subtitles",
    "add_logo",
    "add_transition",
    "generate_voice",
)

DEFAULT_MIXER_CONFIG_RELATIVE_PATH = Path("config") / "audio" / "music_mixer.yaml"


class MusicMode:
    """Music duration-handling modes. Plain string constants (not
    enum.Enum) so they serialize to JSON with no extra handling — same
    idiom as video_engine.PLAN_LOSSLESS_COPY etc."""

    TRIM = "trim"
    LOOP = "loop"

    ALL = (TRIM, LOOP)


class DuckingMode:
    """Music-volume attenuation modes. FIXED is a single constant
    multiplier applied whenever source audio is mixed in — this is
    simple fixed attenuation only, never speech-aware sidechain ducking,
    and there is no voice-activity detection anywhere in this module."""

    NONE = "none"
    FIXED = "fixed"

    ALL = (NONE, FIXED)


def _runtime_root() -> Path:
    """
    music_mixer.py location: 10_apps/claude_runtime/src/music_mixer.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MusicMixerError(RuntimeError):
    """Base error for the Phase 11B Music Mixer Engine."""


class MusicMixerConfigError(MusicMixerError):
    """Raised when config/audio/music_mixer.yaml is missing/invalid, or uses an unimplemented setting."""


class MusicInputError(MusicMixerError):
    """
    General-purpose input-validation error. Covers video-side problems
    (missing/empty/unsupported/no video stream) as well as any
    video-vs-music path collision — the music-specific subclasses below
    cover only the music-file-specific cases named in Phase 11B's own
    exception list.
    """


class MusicFileNotFoundError(MusicInputError):
    """Raised when the music file does not exist."""


class MusicFileEmptyError(MusicInputError):
    """Raised when the music file is zero bytes."""


class UnsupportedMusicExtensionError(MusicInputError):
    """Raised when the music file's extension is not configured as supported."""


class UnsafeMusicOutputError(MusicMixerError):
    """Raised when the output path is unsafe (equals an input, unwritable parent, symlink collision, etc.)."""


class MusicOutputExistsError(MusicMixerError):
    """Raised when the output already exists and --force was not given."""


class MusicTooShortError(MusicMixerError):
    """Raised in trim mode when the music is shorter than the video."""


class MusicDurationError(MusicMixerError):
    """Raised when video/music duration metadata is missing or otherwise unusable."""


class MusicMixPlanError(MusicMixerError):
    """Raised when mix parameters (volumes/fades/modes) are invalid, or metadata is mixed/inconsistent."""


class FFmpegNotFoundError(MusicMixerError):
    """Raised when the configured ffmpeg binary cannot be found/executed."""


class FFmpegExecutionError(MusicMixerError):
    """Raised when ffmpeg exits non-zero."""


class FFmpegTimeoutError(MusicMixerError):
    """Raised when ffmpeg does not finish within the configured timeout."""


class MusicOutputVerificationError(MusicMixerError):
    """Raised when the output file is missing or zero bytes after a reported-successful run."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MusicMixerConfig:
    ffmpeg_binary: str = "ffmpeg"
    ffmpeg_timeout_seconds: int = 300

    media_inspector_config_path: str | None = None

    video_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".mp4", ".mov", ".m4v"})
    )
    music_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac"})
    )
    output_extensions: frozenset[str] = field(default_factory=lambda: frozenset({".mp4"}))

    default_music_mode: str = MusicMode.LOOP
    default_music_volume: float = 0.20
    default_source_audio_volume: float = 1.00
    default_ducking_mode: str = DuckingMode.NONE
    default_fade_in_seconds: float = 0.50
    default_fade_out_seconds: float = 1.00
    preserve_source_audio: bool = True
    trim_to_video_duration: bool = True
    shortest: bool = True

    fixed_music_multiplier: float = 0.55

    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    copy_video_stream: bool = True
    overwrite_requires_force: bool = True
    verify_non_zero_bytes: bool = True
    faststart: bool = True

    write_log: bool = True
    log_filename_suffix: str = "_music_mix_log.json"
    diagnostics_directory: str | None = None


def default_mixer_config_path() -> Path:
    return _runtime_root() / DEFAULT_MIXER_CONFIG_RELATIVE_PATH


def _validate_mix_parameters(
    *,
    music_volume: float,
    source_audio_volume: float,
    fade_in_seconds: float,
    fade_out_seconds: float,
    music_mode: str,
    ducking_mode: str,
    error_cls: type[Exception],
) -> None:
    """
    Shared validation for mix parameters, regardless of whether they came
    from config.yaml defaults or a CLI/request override. Raises
    error_cls (MusicMixerConfigError for config loading,
    MusicMixPlanError for request resolution) with a clear message.
    """
    if music_volume < 0:
        raise error_cls(f"music_volume must be >= 0, got {music_volume!r}")

    if source_audio_volume < 0:
        raise error_cls(f"source_audio_volume must be >= 0, got {source_audio_volume!r}")

    if fade_in_seconds < 0:
        raise error_cls(f"fade_in_seconds must be >= 0, got {fade_in_seconds!r}")

    if fade_out_seconds < 0:
        raise error_cls(f"fade_out_seconds must be >= 0, got {fade_out_seconds!r}")

    if music_mode not in MusicMode.ALL:
        raise error_cls(f"music_mode must be one of {MusicMode.ALL}, got {music_mode!r}")

    if ducking_mode not in DuckingMode.ALL:
        raise error_cls(f"ducking_mode must be one of {DuckingMode.ALL}, got {ducking_mode!r}")


def load_music_mixer_config(config_path: str | Path | None = None) -> MusicMixerConfig:
    """Load config/audio/music_mixer.yaml (or an alternate path) into a
    MusicMixerConfig. Raises MusicMixerConfigError if the file is
    missing/invalid, or if its own default mix parameters are invalid."""
    path = Path(config_path) if config_path else default_mixer_config_path()

    if not path.exists():
        raise MusicMixerConfigError(f"Music mixer config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise MusicMixerConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise MusicMixerConfigError(f"Music mixer config is empty or invalid: {path}")

    ffmpeg_section = raw.get("ffmpeg") or {}
    extensions_section = raw.get("supported_extensions") or {}
    mix_section = raw.get("mix") or {}
    ducking_section = raw.get("ducking") or {}
    output_section = raw.get("output") or {}
    diagnostics_section = raw.get("diagnostics") or {}

    def _extension_set(values: Any) -> frozenset[str]:
        return frozenset(str(item).lower() for item in (values or []))

    if not bool(output_section.get("copy_video_stream", True)):
        raise MusicMixerConfigError(
            "output.copy_video_stream=false is not implemented in Phase 11B "
            "(no fallback video codec is configured); leave it true."
        )

    config = MusicMixerConfig(
        ffmpeg_binary=str(ffmpeg_section.get("binary", "ffmpeg")),
        ffmpeg_timeout_seconds=int(ffmpeg_section.get("timeout_seconds", 300)),
        media_inspector_config_path=raw.get("media_inspector_config_path"),
        video_extensions=_extension_set(extensions_section.get("video")),
        music_extensions=_extension_set(extensions_section.get("music")),
        output_extensions=_extension_set(extensions_section.get("output")),
        default_music_mode=str(mix_section.get("music_mode", MusicMode.LOOP)),
        default_music_volume=float(mix_section.get("music_volume", 0.20)),
        default_source_audio_volume=float(mix_section.get("source_audio_volume", 1.00)),
        default_ducking_mode=str(mix_section.get("ducking_mode", DuckingMode.NONE)),
        default_fade_in_seconds=float(mix_section.get("fade_in_seconds", 0.50)),
        default_fade_out_seconds=float(mix_section.get("fade_out_seconds", 1.00)),
        preserve_source_audio=bool(mix_section.get("preserve_source_audio", True)),
        trim_to_video_duration=bool(mix_section.get("trim_to_video_duration", True)),
        shortest=bool(mix_section.get("shortest", True)),
        fixed_music_multiplier=float(ducking_section.get("fixed_music_multiplier", 0.55)),
        audio_codec=str(output_section.get("audio_codec", "aac")),
        audio_bitrate=str(output_section.get("audio_bitrate", "192k")),
        audio_sample_rate=int(output_section.get("audio_sample_rate", 48000)),
        audio_channels=int(output_section.get("audio_channels", 2)),
        copy_video_stream=True,
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        verify_non_zero_bytes=bool(output_section.get("verify_non_zero_bytes", True)),
        faststart=bool(output_section.get("faststart", True)),
        write_log=bool(diagnostics_section.get("write_log", True)),
        log_filename_suffix=str(
            diagnostics_section.get("log_filename_suffix", "_music_mix_log.json")
        ),
        diagnostics_directory=diagnostics_section.get("directory"),
    )

    _validate_mix_parameters(
        music_volume=config.default_music_volume,
        source_audio_volume=config.default_source_audio_volume,
        fade_in_seconds=config.default_fade_in_seconds,
        fade_out_seconds=config.default_fade_out_seconds,
        music_mode=config.default_music_mode,
        ducking_mode=config.default_ducking_mode,
        error_cls=MusicMixerConfigError,
    )

    return config


# ---------------------------------------------------------------------------
# Runner (independent copy — Music Mixer does not import video_engine.py)
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
    Never logs environment variables or any secret. Raises
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
            f"Executable not found: {command[0]!r}. Install ffmpeg/ffprobe to enable real mixing."
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
class MusicMixRequest:
    """
    One requested mix operation. video_path/music_path/output_path/force
    are required; every other field is Optional — None means "use the
    configured default."
    """

    video_path: Path
    music_path: Path
    output_path: Path
    force: bool = False
    music_volume: float | None = None
    source_audio_volume: float | None = None
    fade_in_seconds: float | None = None
    fade_out_seconds: float | None = None
    music_mode: str | None = None
    ducking_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["video_path"] = str(self.video_path)
        data["music_path"] = str(self.music_path)
        data["output_path"] = str(self.output_path)
        return data


@dataclass(slots=True)
class AudioMixDecision:
    """Which of Phase 11B's four source-audio-handling cases applies.
    case is one of "mix_source_and_music" (A), "music_only_no_source_audio"
    (B), or "music_only_discarded_source" (C). Case D (mixed/invalid
    metadata) never reaches this dataclass — it raises MusicDurationError
    during planning instead."""

    case: str
    use_amix: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MusicMixPlan:
    video_path: Path
    music_path: Path
    output_path: Path
    video_duration_seconds: float
    music_duration_seconds: float
    video_has_audio: bool
    music_mode: str
    music_loop_required: bool
    music_trim_required: bool
    fade_in_seconds: float
    fade_out_seconds: float
    music_volume: float
    source_audio_volume: float
    ducking_mode: str
    audio_decision: AudioMixDecision
    command: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "music_path": str(self.music_path),
            "output_path": str(self.output_path),
            "video_duration_seconds": self.video_duration_seconds,
            "music_duration_seconds": self.music_duration_seconds,
            "video_has_audio": self.video_has_audio,
            "music_mode": self.music_mode,
            "music_loop_required": self.music_loop_required,
            "music_trim_required": self.music_trim_required,
            "fade_in_seconds": self.fade_in_seconds,
            "fade_out_seconds": self.fade_out_seconds,
            "music_volume": self.music_volume,
            "source_audio_volume": self.source_audio_volume,
            "ducking_mode": self.ducking_mode,
            "audio_decision": self.audio_decision.to_dict(),
            "command": self.command,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class MusicMixResult:
    started_at: str
    finished_at: str
    duration_seconds: float
    video_path: str
    music_path: str
    output_path: str
    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    output_exists: bool
    output_size_bytes: int | None
    plan: MusicMixPlan
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["plan"] = self.plan.to_dict()
        return data


# ---------------------------------------------------------------------------
# Path / input validation (pure, before any ffprobe/ffmpeg call)
# ---------------------------------------------------------------------------


def _validate_paths(request: MusicMixRequest, config: MusicMixerConfig) -> tuple[Path, Path, Path]:
    """
    Validates every path-safety rule in requirement 2 before any
    Media Inspector or ffmpeg call happens. Returns the resolved
    (video, music, output) paths. Never modifies any file.
    """
    video_path = request.video_path
    music_path = request.music_path
    output_path = request.output_path

    if not video_path.exists():
        raise MusicInputError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise MusicInputError(f"Video path is not a regular file: {video_path}")
    if video_path.stat().st_size == 0:
        raise MusicInputError(f"Video file is zero bytes: {video_path}")
    if video_path.suffix.lower() not in config.video_extensions:
        raise MusicInputError(
            f"Unsupported video extension {video_path.suffix.lower()!r} for {video_path}. "
            f"Supported: {sorted(config.video_extensions)}"
        )

    if not music_path.exists():
        raise MusicFileNotFoundError(f"Music file not found: {music_path}")
    if not music_path.is_file():
        raise MusicInputError(f"Music path is not a regular file: {music_path}")
    if music_path.stat().st_size == 0:
        raise MusicFileEmptyError(f"Music file is zero bytes: {music_path}")
    if music_path.suffix.lower() not in config.music_extensions:
        raise UnsupportedMusicExtensionError(
            f"Unsupported music extension {music_path.suffix.lower()!r} for {music_path}. "
            f"Supported: {sorted(config.music_extensions)}"
        )

    resolved_video = video_path.resolve()
    resolved_music = music_path.resolve()

    if resolved_video == resolved_music:
        raise MusicInputError(
            f"Music input must not be the same file as the video input: {video_path}"
        )

    resolved_output = output_path.resolve()

    if resolved_output in (resolved_video, resolved_music):
        raise UnsafeMusicOutputError(
            f"Output path must not equal either input path: {output_path}"
        )

    parent = output_path.parent
    if parent.exists() and not parent.is_dir():
        raise UnsafeMusicOutputError(f"Output parent exists but is not a directory: {parent}")

    if output_path.exists() and not request.force and config.overwrite_requires_force:
        raise MusicOutputExistsError(f"{output_path} already exists; pass --force to overwrite.")

    return resolved_video, resolved_music, resolved_output


# ---------------------------------------------------------------------------
# Planning (pure, no execution)
# ---------------------------------------------------------------------------


def build_music_mix_plan(
    request: MusicMixRequest,
    video_info: media_inspector.MediaInfo,
    music_info: media_inspector.MediaInfo,
    config: MusicMixerConfig,
) -> MusicMixPlan:
    """
    Pure planning step — never executes ffmpeg. Decides whether the
    music must loop or trim, which of the four source-audio-handling
    cases applies, resolves fade-in/out placement (clamping with a
    warning rather than raising when they would overlap the whole
    video), and builds the exact ffmpeg command. video_info/music_info
    must already be built (e.g. via media_inspector.inspect_file()) —
    this function performs no I/O itself.
    """
    music_volume = request.music_volume if request.music_volume is not None else config.default_music_volume
    source_audio_volume = (
        request.source_audio_volume
        if request.source_audio_volume is not None
        else config.default_source_audio_volume
    )
    fade_in_seconds = (
        request.fade_in_seconds if request.fade_in_seconds is not None else config.default_fade_in_seconds
    )
    fade_out_seconds = (
        request.fade_out_seconds if request.fade_out_seconds is not None else config.default_fade_out_seconds
    )
    music_mode = request.music_mode or config.default_music_mode
    ducking_mode = request.ducking_mode or config.default_ducking_mode

    _validate_mix_parameters(
        music_volume=music_volume,
        source_audio_volume=source_audio_volume,
        fade_in_seconds=fade_in_seconds,
        fade_out_seconds=fade_out_seconds,
        music_mode=music_mode,
        ducking_mode=ducking_mode,
        error_cls=MusicMixPlanError,
    )

    video_duration = video_info.duration_seconds
    music_duration = music_info.duration_seconds

    if video_duration is None or video_duration <= 0:
        raise MusicDurationError(f"Video duration is missing or invalid: {video_duration!r}")
    if music_duration is None or music_duration <= 0:
        raise MusicDurationError(f"Music duration is missing or invalid: {music_duration!r}")

    music_loop_required = False
    music_trim_required = False

    if music_mode == MusicMode.TRIM:
        if music_duration < video_duration:
            raise MusicTooShortError(
                f"Music duration ({music_duration:g}s) is shorter than the video "
                f"({video_duration:g}s) and music_mode=trim; refusing to guess."
            )
        if music_duration > video_duration:
            music_trim_required = True
    else:  # MusicMode.LOOP
        if music_duration < video_duration:
            music_loop_required = True
        elif music_duration > video_duration:
            music_trim_required = True

    if not config.preserve_source_audio:
        audio_decision = AudioMixDecision(case="music_only_discarded_source", use_amix=False)
    elif not video_info.has_audio:
        audio_decision = AudioMixDecision(case="music_only_no_source_audio", use_amix=False)
    else:
        audio_decision = AudioMixDecision(case="mix_source_and_music", use_amix=True)

    warnings: list[str] = []

    fade_in = max(fade_in_seconds, 0.0)
    fade_out = max(fade_out_seconds, 0.0)

    if fade_in + fade_out > video_duration:
        warnings.append(
            "fades_clamped_to_video_duration: fade_in_seconds + fade_out_seconds "
            "exceeded the video duration."
        )
        if fade_in > video_duration:
            fade_in = video_duration
        fade_out = max(video_duration - fade_in, 0.0)

    effective_music_volume = music_volume
    if ducking_mode == DuckingMode.FIXED and audio_decision.use_amix:
        effective_music_volume = music_volume * config.fixed_music_multiplier

    command = _build_command(
        video_path=request.video_path,
        music_path=request.music_path,
        output_path=request.output_path,
        video_duration=video_duration,
        music_loop_required=music_loop_required,
        music_trim_required=music_trim_required,
        effective_music_volume=effective_music_volume,
        source_audio_volume=source_audio_volume,
        fade_in=fade_in,
        fade_out=fade_out,
        audio_decision=audio_decision,
        config=config,
        force=request.force,
    )

    return MusicMixPlan(
        video_path=request.video_path,
        music_path=request.music_path,
        output_path=request.output_path,
        video_duration_seconds=video_duration,
        music_duration_seconds=music_duration,
        video_has_audio=video_info.has_audio,
        music_mode=music_mode,
        music_loop_required=music_loop_required,
        music_trim_required=music_trim_required,
        fade_in_seconds=fade_in,
        fade_out_seconds=fade_out,
        music_volume=music_volume,
        source_audio_volume=source_audio_volume,
        ducking_mode=ducking_mode,
        audio_decision=audio_decision,
        command=command,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def _build_command(
    *,
    video_path: Path,
    music_path: Path,
    output_path: Path,
    video_duration: float,
    music_loop_required: bool,
    music_trim_required: bool,
    effective_music_volume: float,
    source_audio_volume: float,
    fade_in: float,
    fade_out: float,
    audio_decision: AudioMixDecision,
    config: MusicMixerConfig,
    force: bool,
) -> list[str]:
    """
    Builds the exact ffmpeg command for one music-mix plan. Never
    executes anything. Video is always input 0, music always input 1
    (deterministic order); the video stream is always stream-copied
    (Music Mixer only ever has one video input, so this is always safe);
    only the audio is filtered/re-encoded.
    """
    command = [config.ffmpeg_binary, "-y" if force else "-n"]
    command.extend(["-i", str(video_path)])

    if music_loop_required:
        command.extend(["-stream_loop", "-1"])
    command.extend(["-i", str(music_path)])

    music_filter_parts: list[str] = ["[1:a]"]
    stage_labels: list[str] = []

    if music_trim_required or music_loop_required:
        stage_labels.append(f"atrim=0:{video_duration}")
        stage_labels.append("asetpts=PTS-STARTPTS")

    stage_labels.append(f"volume={effective_music_volume}")

    if fade_in > 0:
        stage_labels.append(f"afade=t=in:st=0:d={fade_in}")

    if fade_out > 0:
        fade_out_start = max(video_duration - fade_out, 0.0)
        stage_labels.append(f"afade=t=out:st={fade_out_start}:d={fade_out}")

    music_filter = "".join(music_filter_parts) + ",".join(stage_labels)

    if audio_decision.use_amix:
        filter_complex = (
            f"{music_filter}[music];"
            f"[0:a]volume={source_audio_volume}[src];"
            f"[src][music]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
    else:
        filter_complex = f"{music_filter}[aout]"

    command.extend(["-filter_complex", filter_complex])
    command.extend(["-map", "0:v", "-map", "[aout]"])
    command.extend(["-c:v", "copy"])
    command.extend(["-c:a", config.audio_codec])
    command.extend(["-b:a", config.audio_bitrate])
    command.extend(["-ar", str(config.audio_sample_rate)])
    command.extend(["-ac", str(config.audio_channels)])

    if config.shortest:
        command.append("-shortest")

    if config.faststart:
        command.extend(["-movflags", "+faststart"])

    command.append(str(output_path))
    return command


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_music_mix_plan(
    plan: MusicMixPlan,
    config: MusicMixerConfig,
    *,
    runner: SubprocessRunner = default_runner,
) -> MusicMixResult:
    """
    Executes plan.command exactly once via the injected runner. Never
    retries, never invokes a fallback command, never modifies the input
    files. Raises FFmpegExecutionError on non-zero exit,
    MusicOutputVerificationError if the output is missing or (per
    config.verify_non_zero_bytes) zero bytes afterward.
    """
    started_at = _now_iso()
    start_monotonic = time.monotonic()

    process_result = runner(plan.command, timeout=config.ffmpeg_timeout_seconds)

    finished_at = _now_iso()
    duration_seconds = time.monotonic() - start_monotonic

    if process_result.returncode != 0:
        raise FFmpegExecutionError(
            f"ffmpeg exited {process_result.returncode}: {process_result.stderr.strip()[-2000:]}"
        )

    output_exists = plan.output_path.is_file()
    output_size_bytes = plan.output_path.stat().st_size if output_exists else 0

    if not output_exists:
        raise MusicOutputVerificationError(
            f"ffmpeg reported success but no output file exists: {plan.output_path}"
        )

    if config.verify_non_zero_bytes and output_size_bytes == 0:
        raise MusicOutputVerificationError(f"Output file is zero bytes: {plan.output_path}")

    return MusicMixResult(
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        video_path=str(plan.video_path),
        music_path=str(plan.music_path),
        output_path=str(plan.output_path),
        command=plan.command,
        return_code=process_result.returncode,
        stdout=process_result.stdout,
        stderr=process_result.stderr,
        output_exists=output_exists,
        output_size_bytes=output_size_bytes,
        plan=plan,
        warnings=list(plan.warnings),
        error=None,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnostic_log_path(output_path: Path, config: MusicMixerConfig) -> Path:
    directory = Path(config.diagnostics_directory) if config.diagnostics_directory else output_path.parent
    return directory / f"{output_path.stem}{config.log_filename_suffix}"


def _write_diagnostic_log(
    *,
    config: MusicMixerConfig,
    started_at: str,
    finished_at: str,
    video_path: Path,
    music_path: Path,
    output_path: Path,
    plan: MusicMixPlan | None,
    command: list[str] | None,
    return_code: int | None,
    output_size_bytes: int | None,
    warnings: list[str],
    result: str,
    error: str | None,
) -> Path | None:
    if not config.write_log:
        return None

    log_path = _diagnostic_log_path(output_path, config)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "started_at": started_at,
        "finished_at": finished_at,
        "video_path": str(video_path),
        "music_path": str(music_path),
        "output_path": str(output_path),
        "video_duration_seconds": plan.video_duration_seconds if plan else None,
        "music_duration_seconds": plan.music_duration_seconds if plan else None,
        "video_has_audio": plan.video_has_audio if plan else None,
        "music_mode": plan.music_mode if plan else None,
        "music_loop_required": plan.music_loop_required if plan else None,
        "music_trim_required": plan.music_trim_required if plan else None,
        "music_volume": plan.music_volume if plan else None,
        "source_audio_volume": plan.source_audio_volume if plan else None,
        "ducking_mode": plan.ducking_mode if plan else None,
        "fade_in_seconds": plan.fade_in_seconds if plan else None,
        "fade_out_seconds": plan.fade_out_seconds if plan else None,
        "command": command,
        "return_code": return_code,
        "output_size_bytes": output_size_bytes,
        "warnings": warnings,
        "result": result,
        "error": error,
        # Never included: cookies, passwords, session tokens, environment
        # variables, unrelated filesystem contents, publishing queue
        # contents, captions, or DM data.
    }

    temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(log_path)
    return log_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def mix_music(
    request: MusicMixRequest,
    config: MusicMixerConfig,
    *,
    runner: SubprocessRunner = default_runner,
) -> MusicMixResult:
    """
    Full orchestration: validates paths, inspects the video/music files
    via Media Inspector (never re-implementing ffprobe parsing), builds
    a MusicMixPlan, executes it, writes the diagnostic log (on both
    success and failure), and returns the MusicMixResult. Never
    publishes, uploads, or modifies either input file.
    """
    started_at = _now_iso()

    _validate_paths(request, config)

    inspector_config = media_inspector.load_inspector_config(config.media_inspector_config_path)

    try:
        video_info = media_inspector.inspect_file(request.video_path, inspector_config, runner=runner)
        if not video_info.has_video:
            raise MusicInputError(f"Video file contains no video stream: {request.video_path}")

        music_info = media_inspector.inspect_file(request.music_path, inspector_config, runner=runner)
        if not music_info.has_audio:
            raise MusicInputError(f"Music file contains no audio stream: {request.music_path}")

        plan = build_music_mix_plan(request, video_info, music_info, config)
        result = execute_music_mix_plan(plan, config, runner=runner)

    except (MusicMixerError, media_inspector.MediaInspectorError) as exc:
        _write_diagnostic_log(
            config=config,
            started_at=started_at,
            finished_at=_now_iso(),
            video_path=request.video_path,
            music_path=request.music_path,
            output_path=request.output_path,
            plan=None,
            command=None,
            return_code=None,
            output_size_bytes=None,
            warnings=[],
            result="failed",
            error=str(exc),
        )
        raise

    _write_diagnostic_log(
        config=config,
        started_at=result.started_at,
        finished_at=result.finished_at,
        video_path=request.video_path,
        music_path=request.music_path,
        output_path=request.output_path,
        plan=plan,
        command=result.command,
        return_code=result.return_code,
        output_size_bytes=result.output_size_bytes,
        warnings=result.warnings,
        result="success",
        error=None,
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Music Mixer Engine (Phase 11B). Combines one rendered "
            "video with one background-music file into a new video. "
            "Handles background music only — no automatic selection, "
            "no downloading, no publishing."
        )
    )

    parser.add_argument("--video", required=True, help="Path to the rendered video file.")
    parser.add_argument("--music", required=True, help="Path to the background-music file.")
    parser.add_argument("--output", required=True, help="Path to write the mixed output video.")
    parser.add_argument("--config", default=None, help="Path to an alternate config/audio/music_mixer.yaml.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    parser.add_argument("--music-volume", type=float, default=None, help="Override configured music volume.")
    parser.add_argument(
        "--source-audio-volume", type=float, default=None, help="Override configured source audio volume."
    )
    parser.add_argument("--fade-in-seconds", type=float, default=None, help="Override configured fade-in duration.")
    parser.add_argument(
        "--fade-out-seconds", type=float, default=None, help="Override configured fade-out duration."
    )
    parser.add_argument("--music-mode", choices=MusicMode.ALL, default=None, help="Music duration-handling mode.")
    parser.add_argument("--ducking", choices=DuckingMode.ALL, default=None, help="Music-volume ducking mode.")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable JSON result.")

    return parser.parse_args(argv)


def _print_result(result: MusicMixResult) -> None:
    print()
    print(f"video path:              {result.video_path}")
    print(f"music path:               {result.music_path}")
    print(f"output path:              {result.output_path}")
    print(f"music mode:               {result.plan.music_mode}")
    print(f"music loop required:      {result.plan.music_loop_required}")
    print(f"music trim required:      {result.plan.music_trim_required}")
    print(f"audio decision:           {result.plan.audio_decision.case}")
    print(f"music volume:             {result.plan.music_volume}")
    print(f"ducking mode:             {result.plan.ducking_mode}")
    print(f"fade in/out (s):          {result.plan.fade_in_seconds}/{result.plan.fade_out_seconds}")
    print(f"return code:              {result.return_code}")
    print(f"output size (bytes):      {result.output_size_bytes}")
    print(f"duration (s):             {result.duration_seconds:.2f}")
    print(f"warnings:                 {result.warnings}")
    print()


def _run(arguments: argparse.Namespace, *, runner: SubprocessRunner = default_runner) -> int:
    try:
        config = load_music_mixer_config(arguments.config)
    except MusicMixerConfigError as exc:
        print(f"[MusicMixer] {exc}")
        return 1

    request = MusicMixRequest(
        video_path=Path(arguments.video),
        music_path=Path(arguments.music),
        output_path=Path(arguments.output),
        force=arguments.force,
        music_volume=arguments.music_volume,
        source_audio_volume=arguments.source_audio_volume,
        fade_in_seconds=arguments.fade_in_seconds,
        fade_out_seconds=arguments.fade_out_seconds,
        music_mode=arguments.music_mode,
        ducking_mode=arguments.ducking,
    )

    try:
        result = mix_music(request, config, runner=runner)
    except (MusicMixerError, media_inspector.MediaInspectorError) as exc:
        print(f"[MusicMixer] {exc}")
        return 1

    if arguments.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_result(result)

    return 0


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    exit_code = _run(arguments)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
