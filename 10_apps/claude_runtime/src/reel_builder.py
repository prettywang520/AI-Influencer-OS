from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

# Phase 11A — Reel Builder Engine. Purely local: reads Reel scene clips
# already produced by Phase 9 for one production date and concatenates
# them into output/<date>/videos/reel_final.mp4 with ffmpeg. This module
# never publishes anything, never opens a browser, never imports
# Playwright, and never imports anything from src/publishing/ or
# src/social/ — it only creates a video file on local disk.
DISALLOWED_ACTIONS = (
    "publish",
    "share",
    "post",
    "upload_to_instagram",
    "send",
)

DEFAULT_BUILDER_CONFIG_RELATIVE_PATH = Path("config") / "reels" / "builder.yaml"

# Accepts an optional "scene_" prefix so this module works against both
# the literal "01.mp4".."05.mp4" naming this phase's spec illustrates and
# the actual filenames src/production_service.py generates today
# ("scene_01.mp4".."scene_05.mp4") — both forms share one numbering
# namespace, so a stray "01.mp4" alongside "scene_01.mp4" is correctly
# caught as a duplicate index 1, not silently accepted as two different
# clips.
SCENE_FILENAME_PATTERN = re.compile(r"^(?:scene_)?(\d{2})\.mp4$")


def _runtime_root() -> Path:
    """
    reel_builder.py location: 10_apps/claude_runtime/src/reel_builder.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReelBuilderError(RuntimeError):
    """Base error for the Phase 11A Reel Builder Engine."""


class ReelBuilderConfigError(ReelBuilderError):
    """Raised when config/reels/builder.yaml is missing/invalid, or uses an unimplemented setting."""


class SceneDirectoryNotFoundError(ReelBuilderError):
    """Raised when output/<date>/videos/reel_scenes/ does not exist."""


class MissingSceneError(ReelBuilderError):
    """Raised when a required scene index (1..scene_count) has no matching clip file."""


class DuplicateSceneError(ReelBuilderError):
    """Raised when a scene index is claimed by more than one clip file."""


class UnrecognizedSceneFileError(ReelBuilderError):
    """Raised when a .mp4 file in the scenes directory does not match the sequential-numbering pattern."""


class EmptySceneFileError(ReelBuilderError):
    """Raised when a scene clip exists but is zero bytes."""


class MixedResolutionError(ReelBuilderError):
    """Raised when scene clips have different resolutions and normalize_resolution is false."""


class MixedFpsError(ReelBuilderError):
    """Raised when scene clips have different frame rates and normalize_fps is false."""


class ReelAlreadyExistsError(ReelBuilderError):
    """Raised when reel_final.mp4 already exists and --force was not given."""


class FfprobeError(ReelBuilderError):
    """Raised when ffprobe cannot be run, times out, or returns unparseable output."""


class FfmpegRenderError(ReelBuilderError):
    """Raised when the ffmpeg render subprocess exits non-zero or times out."""


class ReelBuildOutputMissingError(ReelBuilderError):
    """Raised when ffmpeg exits 0 but reel_final.mp4 was not actually created."""


class ReelBuildEmptyOutputError(ReelBuilderError):
    """Raised when reel_final.mp4 was created but is zero bytes."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BuilderConfig:
    scene_count: int = 5

    transition: str = "none"
    transition_duration: float = 0.35
    fps: int = 30
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    video_bitrate: str = "10M"
    music_volume: float = 0.20
    normalize_resolution: bool = True
    normalize_fps: bool = True

    reel_scenes_dir_template: str = "output/{date}/videos/reel_scenes"
    reel_final_file_template: str = "output/{date}/videos/reel_final.mp4"
    build_log_file_template: str = "output/{date}/videos/build_log.json"

    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    timeout_seconds: int = 600

    def reel_scenes_dir(self, date: str, *, root: str | Path | None = None) -> Path:
        base = Path(root) if root is not None else _runtime_root()
        return base / self.reel_scenes_dir_template.format(date=date)

    def reel_final_path(self, date: str, *, root: str | Path | None = None) -> Path:
        base = Path(root) if root is not None else _runtime_root()
        return base / self.reel_final_file_template.format(date=date)

    def build_log_path(self, date: str, *, root: str | Path | None = None) -> Path:
        base = Path(root) if root is not None else _runtime_root()
        return base / self.build_log_file_template.format(date=date)


def default_builder_config_path() -> Path:
    return _runtime_root() / DEFAULT_BUILDER_CONFIG_RELATIVE_PATH


def load_builder_config(config_path: str | Path | None = None) -> BuilderConfig:
    path = Path(config_path) if config_path else default_builder_config_path()

    if not path.exists():
        raise ReelBuilderConfigError(f"Reel builder config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ReelBuilderConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise ReelBuilderConfigError(f"Reel builder config is empty or invalid: {path}")

    rendering = raw.get("rendering") or {}
    paths_section = raw.get("paths") or {}
    ffmpeg_section = raw.get("ffmpeg") or {}

    transition = str(rendering.get("transition", "none"))
    if transition != "none":
        raise ReelBuilderConfigError(
            f"rendering.transition={transition!r} is not implemented in Phase 11A; "
            "only 'none' (hard-cut concatenation) is supported."
        )

    config = BuilderConfig(
        scene_count=int(raw.get("scene_count", 5)),
        transition=transition,
        transition_duration=float(rendering.get("transition_duration", 0.35)),
        fps=int(rendering.get("fps", 30)),
        video_codec=str(rendering.get("video_codec", "libx264")),
        audio_codec=str(rendering.get("audio_codec", "aac")),
        video_bitrate=str(rendering.get("video_bitrate", "10M")),
        music_volume=float(rendering.get("music_volume", 0.20)),
        normalize_resolution=bool(rendering.get("normalize_resolution", True)),
        normalize_fps=bool(rendering.get("normalize_fps", True)),
        reel_scenes_dir_template=str(
            paths_section.get("reel_scenes_dir_template", "output/{date}/videos/reel_scenes")
        ),
        reel_final_file_template=str(
            paths_section.get("reel_final_file_template", "output/{date}/videos/reel_final.mp4")
        ),
        build_log_file_template=str(
            paths_section.get("build_log_file_template", "output/{date}/videos/build_log.json")
        ),
        ffmpeg_binary=str(ffmpeg_section.get("ffmpeg_binary", "ffmpeg")),
        ffprobe_binary=str(ffmpeg_section.get("ffprobe_binary", "ffprobe")),
        timeout_seconds=int(ffmpeg_section.get("timeout_seconds", 600)),
    )

    return config


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------


def validate_date(date: str) -> None:
    try:
        date_cls.fromisoformat(date)
    except ValueError as exc:
        raise ReelBuilderError(f"--date must be in YYYY-MM-DD format, got: {date!r}") from exc


# ---------------------------------------------------------------------------
# Scene discovery (pure, no ffmpeg)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SceneClip:
    index: int
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "path": str(self.path)}


def discover_scene_clips(scenes_dir: Path, scene_count: int) -> list[SceneClip]:
    """
    Scan scenes_dir for scene_NN.mp4 / NN.mp4 files, validate the set is
    complete, sequential, and duplicate-free, and return them ordered by
    index. Pure filesystem check — never invokes ffmpeg/ffprobe.
    """
    if not scenes_dir.is_dir():
        raise SceneDirectoryNotFoundError(f"Reel scenes directory not found: {scenes_dir}")

    by_index: dict[int, list[Path]] = {}
    unrecognized: list[Path] = []

    for entry in sorted(scenes_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".mp4":
            continue

        match = SCENE_FILENAME_PATTERN.match(entry.name)
        if match is None:
            unrecognized.append(entry)
            continue

        index = int(match.group(1))
        by_index.setdefault(index, []).append(entry)

    if unrecognized:
        names = ", ".join(sorted(path.name for path in unrecognized))
        raise UnrecognizedSceneFileError(
            f"Found .mp4 file(s) in {scenes_dir} that do not match the "
            f"sequential scene-numbering pattern (e.g. scene_01.mp4): {names}"
        )

    duplicates = {index: paths for index, paths in by_index.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(
            f"index {index:02d}: {', '.join(p.name for p in paths)}"
            for index, paths in sorted(duplicates.items())
        )
        raise DuplicateSceneError(f"Duplicate scene index(es) found in {scenes_dir}: {details}")

    missing = [index for index in range(1, scene_count + 1) if index not in by_index]
    if missing:
        missing_str = ", ".join(f"{index:02d}" for index in missing)
        raise MissingSceneError(
            f"Missing scene clip(s) in {scenes_dir} for index(es): {missing_str} "
            f"(expected {scene_count} sequential clips, 01..{scene_count:02d})"
        )

    extra = sorted(index for index in by_index if index > scene_count)
    if extra:
        extra_str = ", ".join(f"{index:02d}" for index in extra)
        raise UnrecognizedSceneFileError(
            f"Found scene clip(s) beyond the configured scene_count={scene_count} "
            f"in {scenes_dir}: index(es) {extra_str}"
        )

    clips: list[SceneClip] = []
    for index in range(1, scene_count + 1):
        clip_path = by_index[index][0]

        if not clip_path.is_file():
            raise EmptySceneFileError(f"Scene clip does not exist: {clip_path}")

        if clip_path.stat().st_size == 0:
            raise EmptySceneFileError(f"Scene clip is zero bytes: {clip_path}")

        clips.append(SceneClip(index=index, path=clip_path))

    return clips


# ---------------------------------------------------------------------------
# ffprobe wrapper (mockable via an injectable runner)
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
        raise ReelBuilderError(f"Executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ReelBuilderError(f"Command timed out after {timeout}s: {' '.join(command)}") from exc

    return ProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


@dataclass(slots=True)
class VideoProbe:
    path: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    codec_name: str
    has_audio: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_frame_rate(raw: str) -> float:
    if "/" in raw:
        numerator_str, _, denominator_str = raw.partition("/")
        try:
            numerator = float(numerator_str)
            denominator = float(denominator_str)
            return numerator / denominator if denominator else 0.0
        except ValueError:
            return 0.0

    try:
        return float(raw)
    except ValueError:
        return 0.0


def probe_video(
    path: Path,
    config: BuilderConfig,
    *,
    runner: SubprocessRunner = default_runner,
) -> VideoProbe:
    command = [
        config.ffprobe_binary,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    result = runner(command, timeout=config.timeout_seconds)

    if result.returncode != 0:
        raise FfprobeError(
            f"ffprobe failed for {path} (exit {result.returncode}): {result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FfprobeError(f"ffprobe returned unparseable JSON for {path}: {exc}") from exc

    streams = data.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    if video_stream is None:
        raise FfprobeError(f"No video stream found in {path}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration_raw = (data.get("format") or {}).get("duration") or video_stream.get("duration")
    try:
        duration_seconds = float(duration_raw)
    except (TypeError, ValueError):
        raise FfprobeError(f"Could not determine duration for {path}")

    return VideoProbe(
        path=str(path),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=_parse_frame_rate(str(video_stream.get("r_frame_rate", "0/1"))),
        duration_seconds=duration_seconds,
        codec_name=str(video_stream.get("codec_name", "")),
        has_audio=has_audio,
    )


# ---------------------------------------------------------------------------
# Clip-set consistency validation (pure, given probes already collected)
# ---------------------------------------------------------------------------


def validate_clip_consistency(probes: list[VideoProbe], config: BuilderConfig) -> bool:
    """
    Returns True if the clips require normalization (mixed
    resolution/fps/codec that a re-encode will unify), False if a
    lossless "-c copy" concat is possible. Raises MixedResolutionError/
    MixedFpsError if a mismatch exists that the config does not permit
    normalizing away.
    """
    resolutions = {(probe.width, probe.height) for probe in probes}
    frame_rates = {round(probe.fps, 3) for probe in probes}
    codecs = {probe.codec_name for probe in probes}

    if len(resolutions) > 1 and not config.normalize_resolution:
        raise MixedResolutionError(
            f"Scene clips have mixed resolutions {sorted(resolutions)} and "
            "normalize_resolution is false; refusing to guess."
        )

    if len(frame_rates) > 1 and not config.normalize_fps:
        raise MixedFpsError(
            f"Scene clips have mixed frame rates {sorted(frame_rates)} and "
            "normalize_fps is false; refusing to guess."
        )

    return len(resolutions) > 1 or len(frame_rates) > 1 or len(codecs) > 1


# ---------------------------------------------------------------------------
# ffmpeg command construction
# ---------------------------------------------------------------------------


def _escape_concat_path(path: Path) -> str:
    # ffmpeg's concat demuxer file format: each line is `file '<path>'`,
    # with literal single quotes inside the path escaped as '\''.
    return str(path).replace("'", "'\\''")


def build_concat_list_content(clips: list[SceneClip]) -> str:
    lines = [f"file '{_escape_concat_path(clip.path)}'" for clip in clips]
    return "\n".join(lines) + "\n"


def build_ffmpeg_command(
    *,
    clips: list[SceneClip],
    output_path: Path,
    config: BuilderConfig,
    concat_list_path: Path,
    normalize: bool,
    force: bool,
    target_width: int | None = None,
    target_height: int | None = None,
) -> list[str]:
    overwrite_flag = "-y" if force else "-n"

    if not normalize:
        return [
            config.ffmpeg_binary,
            overwrite_flag,
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",
            str(output_path),
        ]

    if not target_width or not target_height:
        raise ReelBuilderError(
            "target_width/target_height are required to normalize clips to a "
            "common resolution."
        )

    command = [config.ffmpeg_binary, overwrite_flag]
    for clip in clips:
        command.extend(["-i", str(clip.path)])

    video_labels = []
    audio_labels = []
    filter_parts = []

    for stream_index in range(len(clips)):
        video_label = f"v{stream_index}"
        filter_parts.append(
            f"[{stream_index}:v]scale={target_width}:{target_height},"
            f"fps={config.fps},setsar=1[{video_label}]"
        )
        video_labels.append(f"[{video_label}]")
        audio_labels.append(f"[{stream_index}:a]")

    concat_inputs = "".join(
        f"{v}{a}" for v, a in zip(video_labels, audio_labels)
    )
    filter_parts.append(
        f"{concat_inputs}concat=n={len(clips)}:v=1:a=1[outv][outa]"
    )

    command.extend(["-filter_complex", ";".join(filter_parts)])
    command.extend(["-map", "[outv]", "-map", "[outa]"])
    command.extend(["-c:v", config.video_codec, "-b:v", config.video_bitrate])
    command.extend(["-c:a", config.audio_codec])
    command.append(str(output_path))

    return command


# ---------------------------------------------------------------------------
# Diagnostics / build log
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BuildResult:
    production_date: str
    scene_clips: list[str]
    clip_durations: list[float]
    render_duration_seconds: float
    ffmpeg_command: list[str]
    output_path: str
    output_size_bytes: int
    normalized: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_build_log(
    *,
    log_path: Path,
    started_at: str,
    production_date: str,
    input_clips: list[dict[str, Any]],
    clip_durations: list[float],
    render_duration_seconds: float | None,
    ffmpeg_command: list[str] | None,
    output_size_bytes: int | None,
    warnings: list[str],
    errors: list[str],
    result: str,
) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    finished_at = _now_iso()

    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "production_date": production_date,
        "input_clips": input_clips,
        "clip_durations": clip_durations,
        "render_duration_seconds": render_duration_seconds,
        "ffmpeg_command": ffmpeg_command,
        "output_size_bytes": output_size_bytes,
        "warnings": warnings,
        "errors": errors,
        "result": result,
    }

    temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(log_path)
    return log_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_reel(
    date: str,
    config: BuilderConfig,
    *,
    force: bool = False,
    runner: SubprocessRunner = default_runner,
    root: str | Path | None = None,
) -> BuildResult:
    """
    Build output/<date>/videos/reel_final.mp4 from the Reel scene clips
    for that production date. Never publishes anything, never touches
    Instagram, never uses Playwright.

    root defaults to the real app root (10_apps/claude_runtime), what the
    CLI uses; tests pass an injected temp directory instead, the same
    testability pattern instagram_reel_preview.py's output_root uses.
    """
    started_at = _now_iso()
    validate_date(date)

    scenes_dir = config.reel_scenes_dir(date, root=root)
    output_path = config.reel_final_path(date, root=root)
    log_path = config.build_log_path(date, root=root)

    if output_path.exists() and not force:
        raise ReelAlreadyExistsError(
            f"{output_path} already exists; pass --force to overwrite."
        )

    try:
        clips = discover_scene_clips(scenes_dir, config.scene_count)

        probes = [probe_video(clip.path, config, runner=runner) for clip in clips]
        normalize = validate_clip_consistency(probes, config)

        warnings: list[str] = []
        if not all(probe.has_audio for probe in probes):
            warnings.append(
                "one_or_more_clips_missing_audio_stream_output_audio_may_be_incomplete"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with _temporary_concat_list(clips) as concat_list_path:
            command = build_ffmpeg_command(
                clips=clips,
                output_path=output_path,
                config=config,
                concat_list_path=concat_list_path,
                normalize=normalize,
                force=force,
                # When normalizing, every clip is scaled to the first
                # clip's resolution — there is no separately configured
                # target resolution, so the first clip acts as the
                # reference frame, consistent with concatenating in
                # scene-number order.
                target_width=probes[0].width if normalize else None,
                target_height=probes[0].height if normalize else None,
            )

            render_start = time.monotonic()
            result = runner(command, timeout=config.timeout_seconds)
            render_duration_seconds = time.monotonic() - render_start

            if result.returncode != 0:
                raise FfmpegRenderError(
                    f"ffmpeg exited {result.returncode}: {result.stderr.strip()[-2000:]}"
                )

        if not output_path.is_file():
            raise ReelBuildOutputMissingError(
                f"ffmpeg reported success but no output file exists: {output_path}"
            )

        output_size_bytes = output_path.stat().st_size

        if output_size_bytes == 0:
            raise ReelBuildEmptyOutputError(f"Output file is zero bytes: {output_path}")

        build_result = BuildResult(
            production_date=date,
            scene_clips=[str(clip.path) for clip in clips],
            clip_durations=[probe.duration_seconds for probe in probes],
            render_duration_seconds=render_duration_seconds,
            ffmpeg_command=command,
            output_path=str(output_path),
            output_size_bytes=output_size_bytes,
            normalized=normalize,
            warnings=warnings,
        )

        _write_build_log(
            log_path=log_path,
            started_at=started_at,
            production_date=date,
            input_clips=[clip.to_dict() for clip in clips],
            clip_durations=build_result.clip_durations,
            render_duration_seconds=render_duration_seconds,
            ffmpeg_command=command,
            output_size_bytes=output_size_bytes,
            warnings=warnings,
            errors=[],
            result="success",
        )

        return build_result

    except ReelBuilderError as exc:
        _write_build_log(
            log_path=log_path,
            started_at=started_at,
            production_date=date,
            input_clips=[],
            clip_durations=[],
            render_duration_seconds=None,
            ffmpeg_command=None,
            output_size_bytes=None,
            warnings=[],
            errors=[str(exc)],
            result="failed",
        )
        raise


@contextmanager
def _temporary_concat_list(clips: list[SceneClip]):
    with tempfile.TemporaryDirectory() as temp_dir_name:
        concat_list_path = Path(temp_dir_name) / "concat_list.txt"
        concat_list_path.write_text(build_concat_list_content(clips), encoding="utf-8")
        yield concat_list_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Reel Builder Engine (Phase 11A). Concatenates the "
            "generated Reel scene clips for one production date into "
            "output/<date>/videos/reel_final.mp4. Never publishes."
        )
    )

    parser.add_argument("--date", required=True, help="Production date YYYY-MM-DD.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate config/reels/builder.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing reel_final.mp4.",
    )

    return parser.parse_args(argv)


def _print_result(result: BuildResult) -> None:
    print()
    print("AIKO Reel Builder Engine (Phase 11A)")
    print("-------------------------------------")
    print(f"production date:        {result.production_date}")
    print(f"scene clips:             {result.scene_clips}")
    print(f"clip durations (s):      {result.clip_durations}")
    print(f"normalized:              {result.normalized}")
    print(f"render duration (s):     {result.render_duration_seconds:.2f}")
    print(f"ffmpeg command:          {' '.join(result.ffmpeg_command)}")
    print(f"output path:             {result.output_path}")
    print(f"output size (bytes):     {result.output_size_bytes}")
    print(f"warnings:                {result.warnings}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_builder_config(arguments.config)
        result = build_reel(arguments.date, config, force=arguments.force)
    except ReelBuilderError as exc:
        print(f"[ReelBuilder] {exc}")
        raise SystemExit(1) from exc

    _print_result(result)


if __name__ == "__main__":
    main()
