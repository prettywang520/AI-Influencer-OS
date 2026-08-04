from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from . import media_inspector, music_mixer, timeline_engine, video_engine

# Phase 11A — Reel Builder Engine. Purely local: reads Reel scene clips
# already produced by Phase 9 for one production date and concatenates
# them into output/<date>/videos/reel_final.mp4 with ffmpeg. This module
# never publishes anything, never opens a browser, never imports
# Playwright, and never imports anything from src/publishing/ or
# src/social/ — it only creates a video file on local disk.
#
# Phase 11A.3 refactor: command construction, execution, output
# verification, concat manifest generation, and temporary-file cleanup
# are now delegated to src/video_engine.py's VideoEngine (a generic,
# reusable ffmpeg core). Reel Builder keeps everything Instagram-Reel-
# specific: scene discovery/numbering validation, and the
# resolution/FPS/audio-presence compatibility POLICY (independent
# normalize_resolution/normalize_fps flags — a finer-grained control than
# Video Engine's own single normalize_when_needed flag, so this module
# constructs its own ConcatPlan from that policy decision rather than
# re-deriving it through engine.build_concat_plan()).
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


class MixedAudioPresenceError(ReelBuilderError):
    """
    Raised when some scene clips have an audio stream and others do not.
    Phase 11A.3 does not synthesize silence, so a mixed layout always
    fails clearly rather than risking a desynchronized/invalid render.
    """


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


class MusicMixIntegrationError(ReelBuilderError):
    """
    Raised when --music is supplied and the injected Music Mixer
    callable raises (music_mixer.MusicMixerError or
    media_inspector.MediaInspectorError). The intermediate
    reel_without_music.mp4 is always preserved when this happens — see
    build_reel()'s music-enabled flow.
    """


class TimelineIntegrationError(ReelBuilderError):
    """
    Raised for any --timeline failure: a missing/empty/malformed
    timeline file, a Timeline that fails validate_timeline(), a
    disallowed construct (active transition, non-unit playback_rate,
    gap, overlap, duplicate/missing source file), or a duration-
    reconciliation mismatch against the actual probed media. No
    VideoEngine or Music Mixer call is ever made when this is raised,
    and directory scene discovery is never used as a fallback — mirrors
    how MusicMixIntegrationError wraps every music-stage failure.
    """


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

    # Phase 11B.1 — Reel Builder Music Integration. Only read when
    # --music is supplied; otherwise the original Phase 11A.3 behavior
    # is unaffected by any of these fields.
    integration_enabled: bool = True
    intermediate_filename: str = "reel_without_music.mp4"
    final_filename: str = "reel_final.mp4"
    keep_intermediate_default: bool = False
    preserve_intermediate_on_music_failure: bool = True
    cleanup_intermediate_on_success: bool = True

    # Phase 11C.1 — Timeline -> Reel Builder Integration. Only read when
    # --timeline is supplied; otherwise the original behavior is
    # unaffected by any of these fields.
    timeline_integration_enabled: bool = True
    timeline_production_date_must_match: bool = True
    timeline_duration_tolerance_seconds: float = 0.05
    timeline_allow_trimmed_clips: bool = True
    timeline_reject_active_transitions: bool = True
    timeline_reject_non_unit_playback_rate: bool = True
    timeline_reject_gaps: bool = True
    timeline_reject_overlaps: bool = True

    def reel_scenes_dir(self, date: str, *, root: str | Path | None = None) -> Path:
        base = Path(root) if root is not None else _runtime_root()
        return base / self.reel_scenes_dir_template.format(date=date)

    def reel_final_path(self, date: str, *, root: str | Path | None = None) -> Path:
        base = Path(root) if root is not None else _runtime_root()
        return base / self.reel_final_file_template.format(date=date)

    def reel_without_music_path(self, date: str, *, root: str | Path | None = None) -> Path:
        """Only meaningful when --music is supplied. Same directory as
        reel_final_path(), different filename (config.intermediate_filename)."""
        return self.reel_final_path(date, root=root).parent / self.intermediate_filename

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
    integration_section = raw.get("integration") or {}
    timeline_integration_section = raw.get("timeline_integration") or {}

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
        integration_enabled=bool(integration_section.get("enabled", True)),
        intermediate_filename=str(
            integration_section.get("intermediate_filename", "reel_without_music.mp4")
        ),
        final_filename=str(integration_section.get("final_filename", "reel_final.mp4")),
        keep_intermediate_default=bool(integration_section.get("keep_intermediate_default", False)),
        preserve_intermediate_on_music_failure=bool(
            integration_section.get("preserve_intermediate_on_music_failure", True)
        ),
        cleanup_intermediate_on_success=bool(
            integration_section.get("cleanup_intermediate_on_success", True)
        ),
        timeline_integration_enabled=bool(timeline_integration_section.get("enabled", True)),
        timeline_production_date_must_match=bool(
            timeline_integration_section.get("production_date_must_match", True)
        ),
        timeline_duration_tolerance_seconds=float(
            timeline_integration_section.get("duration_tolerance_seconds", 0.05)
        ),
        timeline_allow_trimmed_clips=bool(
            timeline_integration_section.get("allow_trimmed_clips", True)
        ),
        timeline_reject_active_transitions=bool(
            timeline_integration_section.get("reject_active_transitions", True)
        ),
        timeline_reject_non_unit_playback_rate=bool(
            timeline_integration_section.get("reject_non_unit_playback_rate", True)
        ),
        timeline_reject_gaps=bool(timeline_integration_section.get("reject_gaps", True)),
        timeline_reject_overlaps=bool(timeline_integration_section.get("reject_overlaps", True)),
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
# Video Engine adapters (Phase 11A.3)
# ---------------------------------------------------------------------------


def _video_engine_config_from(config: BuilderConfig) -> video_engine.VideoEngineConfig:
    """Adapts BuilderConfig's existing fields into a VideoEngineConfig —
    Reel Builder does not load config/video/engine.yaml at all, keeping
    its own CLI/config surface unchanged."""
    return video_engine.VideoEngineConfig(
        ffmpeg_binary=config.ffmpeg_binary,
        ffmpeg_timeout_seconds=config.timeout_seconds,
        ffprobe_binary=config.ffprobe_binary,
        ffprobe_timeout_seconds=config.timeout_seconds,
        default_video_codec=config.video_codec,
        default_video_bitrate=config.video_bitrate,
        default_fps=config.fps,
        default_audio_codec=config.audio_codec,
        faststart=True,
        normalize_when_needed=True,
        cleanup_temporary_files=True,
    )


def _video_input_from_probe(clip: SceneClip, probe: VideoProbe) -> video_engine.VideoInput:
    return video_engine.VideoInput(
        path=clip.path,
        video=video_engine.VideoStreamSpec(
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
            codec_name=probe.codec_name,
        ),
        audio=video_engine.AudioStreamSpec(present=probe.has_audio),
        duration_seconds=probe.duration_seconds,
    )


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
    """
    Compatibility wrapper (unchanged signature/behavior from Phase 11A):
    builds the ffmpeg command for this clip set via VideoEngine's command
    builders, without executing anything. Audio is always assumed present
    for every clip here — this function has no way to know real
    per-clip audio presence (it only receives clips/paths, not probes),
    matching Phase 11A's own original unconditional behavior.
    """
    engine_config = _video_engine_config_from(config)
    engine = video_engine.VideoEngine(engine_config)

    inputs = [
        video_engine.VideoInput(
            path=clip.path,
            video=video_engine.VideoStreamSpec(width=target_width, height=target_height, fps=config.fps),
            audio=video_engine.AudioStreamSpec(present=True),
        )
        for clip in clips
    ]
    output = video_engine.VideoOutputSpec(
        path=output_path,
        video_codec=config.video_codec,
        video_bitrate=config.video_bitrate,
        fps=config.fps,
        width=target_width,
        height=target_height,
        audio_codec=config.audio_codec,
    )

    if not normalize:
        plan = video_engine.ConcatPlan(
            plan_type=video_engine.PLAN_LOSSLESS_COPY, inputs=inputs, output=output
        )
        return engine.build_lossless_concat_command(plan, concat_list_path, force=force)

    if not target_width or not target_height:
        raise ReelBuilderError(
            "target_width/target_height are required to normalize clips to a "
            "common resolution."
        )

    plan = video_engine.ConcatPlan(
        plan_type=video_engine.PLAN_NORMALIZED_RENDER,
        inputs=inputs,
        output=output,
        target_video=video_engine.VideoStreamSpec(width=target_width, height=target_height),
    )
    return engine.build_normalized_concat_command(plan, force=force)


# ---------------------------------------------------------------------------
# Timeline integration (Phase 11C.1)
#
# Consumes an already-built, already-validated timeline_engine.Timeline as
# the authoritative scene order and timing plan, in place of directory
# scene discovery. Timeline Engine never inspects real media, so every
# clip's planned duration is reconciled here against a real probe_video()
# call before anything is handed to Video Engine. Never falls back to
# discover_scene_clips() if any of this fails.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TimelineSceneInput:
    clip: "timeline_engine.VideoClip"
    path: Path
    source_in: float
    source_out: float
    actual_duration: float
    is_full_clip: bool


# Used for the base timeline_engine.validate_timeline() structural check
# only (unique IDs, duration math, valid ranges, at least one video
# clip, etc). Gap/overlap policy is deliberately NOT enforced here —
# config.timeline_reject_gaps/timeline_reject_overlaps in
# _select_primary_video_clips() are the actual authority for that, so
# those flags stay meaningful independent of Timeline Engine's own
# (always-on-by-default) gap/overlap checks.
_TIMELINE_STRUCTURAL_VALIDATION_CONFIG = timeline_engine.TimelineConfig(
    allow_video_gaps=True, allow_audio_overlap=True
)


def _load_and_validate_timeline(
    timeline_path: Path,
    date: str,
    config: BuilderConfig,
    *,
    loader: Callable[..., Any] | None,
    validator: Callable[..., Any] | None,
) -> "timeline_engine.Timeline":
    """
    Loads and validates a timeline.json, enforcing production_date
    matching per config. Raises TimelineIntegrationError for every
    failure mode — missing/empty file, load failure, failed validation,
    or a production_date mismatch. Never falls back to directory scene
    discovery.
    """
    load_fn = loader or timeline_engine.load_timeline
    validate_fn = validator or timeline_engine.validate_timeline

    if not timeline_path.is_file():
        raise TimelineIntegrationError(f"--timeline file not found: {timeline_path}")
    if timeline_path.stat().st_size == 0:
        raise TimelineIntegrationError(f"--timeline file is empty: {timeline_path}")

    try:
        timeline = load_fn(timeline_path)
    except timeline_engine.TimelineEngineError as exc:
        raise TimelineIntegrationError(f"Failed to load timeline {timeline_path}: {exc}") from exc

    result = validate_fn(timeline, _TIMELINE_STRUCTURAL_VALIDATION_CONFIG)
    if not result.passed:
        raise TimelineIntegrationError(
            f"Timeline {timeline_path} failed validation: {'; '.join(result.errors)}"
        )

    if (
        config.timeline_production_date_must_match
        and timeline.production_date is not None
        and timeline.production_date != date
    ):
        raise TimelineIntegrationError(
            f"Timeline production_date {timeline.production_date!r} does not match --date {date!r}"
        )

    return timeline


def _select_primary_video_clips(
    timeline: "timeline_engine.Timeline", config: BuilderConfig
) -> tuple[list["timeline_engine.VideoClip"], list[str], int]:
    """
    Picks the lowest-`order` enabled video track as authoritative, in
    timeline (start-ordered) sequence. Disabled clips are ignored.
    Non-primary tracks (music, subtitle, overlay, metadata) are reported
    back as ignored — subtitle/overlay explicitly flagged unsupported.
    Rejects active transitions, non-unit playback_rate, gaps, overlaps
    (per config flags), a missing/zero-byte source file, and duplicate
    *resolved* source paths.
    """
    video_tracks = [
        track for track in timeline.tracks if track.track_type == timeline_engine.TrackType.VIDEO
    ]
    enabled_video_tracks = [track for track in video_tracks if track.enabled]
    if not enabled_video_tracks:
        raise TimelineIntegrationError("Timeline has no enabled video track")

    primary_track = min(enabled_video_tracks, key=lambda track: track.order)

    ignored_tracks: list[str] = []
    for track in timeline.tracks:
        if track.track_id == primary_track.track_id:
            continue
        label = f"{track.track_id} ({track.track_type})"
        if track.track_type in (timeline_engine.TrackType.SUBTITLE, timeline_engine.TrackType.OVERLAY):
            label += " - unsupported in Phase 11C.1"
        ignored_tracks.append(label)

    all_clips = sorted(primary_track.clips, key=lambda clip: clip.start)
    enabled_clips = [clip for clip in all_clips if clip.enabled]
    ignored_disabled_count = len(all_clips) - len(enabled_clips)

    if not enabled_clips:
        raise TimelineIntegrationError("Timeline's primary video track has no enabled clips")

    for clip in enabled_clips:
        if not isinstance(clip, timeline_engine.VideoClip):
            raise TimelineIntegrationError(
                f"Non-video clip {clip.clip_id} found on the primary video track"
            )
        if config.timeline_reject_active_transitions and (
            clip.transition_in != "none" or clip.transition_out != "none"
        ):
            raise TimelineIntegrationError(
                f"Clip {clip.clip_id} uses an active transition (transition_in="
                f"{clip.transition_in!r}, transition_out={clip.transition_out!r}); transitions "
                "are not supported in Phase 11C.1"
            )
        if config.timeline_reject_non_unit_playback_rate and abs(clip.playback_rate - 1.0) > 1e-9:
            raise TimelineIntegrationError(
                f"Clip {clip.clip_id} has playback_rate={clip.playback_rate}; only 1.0 is "
                "supported in Phase 11C.1"
            )

    # Gap/overlap adjacency is checked across ALL clips (including
    # disabled ones), not just enabled_clips: a disabled clip still
    # occupies its authored timeslot, so skipping it is not a gap. Only
    # a genuinely empty span between two clips is flagged.
    tolerance = config.timeline_duration_tolerance_seconds
    for previous, current in zip(all_clips, all_clips[1:]):
        gap = current.start - previous.end
        if gap > tolerance and config.timeline_reject_gaps:
            raise TimelineIntegrationError(
                f"Gap between clips {previous.clip_id} and {current.clip_id} on the timeline "
                "video track"
            )
        if gap < -tolerance and config.timeline_reject_overlaps:
            raise TimelineIntegrationError(
                f"Overlap between clips {previous.clip_id} and {current.clip_id} on the "
                "timeline video track"
            )

    resolved_seen: dict[Path, str] = {}
    for clip in enabled_clips:
        source_path = Path(clip.source_path)
        if not source_path.is_file():
            raise TimelineIntegrationError(
                f"Clip {clip.clip_id} source_path does not resolve to a regular file: {source_path}"
            )
        if source_path.stat().st_size == 0:
            raise TimelineIntegrationError(f"Clip {clip.clip_id} source_path is zero bytes: {source_path}")
        resolved = source_path.resolve()
        if resolved in resolved_seen:
            raise TimelineIntegrationError(
                f"Clips {resolved_seen[resolved]} and {clip.clip_id} resolve to the same source "
                f"file: {resolved}"
            )
        resolved_seen[resolved] = clip.clip_id

    return enabled_clips, ignored_tracks, ignored_disabled_count


def _reconcile_clip_durations(
    clips: list["timeline_engine.VideoClip"],
    config: BuilderConfig,
    runner: SubprocessRunner,
) -> tuple[list[_TimelineSceneInput], list[VideoProbe], int]:
    """
    Probes each clip's real source file (via the existing probe_video()
    — no new Media Inspector dependency) and reconciles it against the
    timeline's planned source_in/source_out/duration_seconds. Raises
    TimelineIntegrationError, naming the exact clip and the planned vs.
    actual numbers, for: negative source_in, source_out <= source_in,
    planned duration not matching source_out - source_in, or source_out
    exceeding the actual media duration — all beyond
    config.timeline_duration_tolerance_seconds.
    """
    tolerance = config.timeline_duration_tolerance_seconds
    scene_inputs: list[_TimelineSceneInput] = []
    probes: list[VideoProbe] = []
    trimmed_count = 0

    for clip in clips:
        source_path = Path(clip.source_path)
        probe = probe_video(source_path, config, runner=runner)
        actual_duration = probe.duration_seconds

        source_in = clip.source_in
        source_out = clip.source_out if clip.source_out is not None else clip.duration_seconds

        if source_in < 0:
            raise TimelineIntegrationError(f"Clip {clip.clip_id} has a negative source_in: {source_in}")
        if source_out <= source_in:
            raise TimelineIntegrationError(
                f"Clip {clip.clip_id} source_out ({source_out}) must be greater than "
                f"source_in ({source_in})"
            )

        planned_duration = source_out - source_in
        if abs(planned_duration - clip.duration_seconds) > tolerance:
            raise TimelineIntegrationError(
                f"Clip {clip.clip_id} planned duration ({clip.duration_seconds:.3f}s) does not "
                f"match source_out - source_in ({planned_duration:.3f}s) beyond tolerance "
                f"{tolerance}s"
            )

        if source_out - actual_duration > tolerance:
            raise TimelineIntegrationError(
                f"Clip {clip.clip_id} source_out ({source_out:.3f}s) exceeds the actual media "
                f"duration ({actual_duration:.3f}s) of {source_path} beyond tolerance {tolerance}s"
            )

        is_full_clip = source_in <= tolerance and (actual_duration - source_out) <= tolerance
        if not is_full_clip and not config.timeline_allow_trimmed_clips:
            raise TimelineIntegrationError(
                f"Clip {clip.clip_id} requires trimming but timeline_integration."
                "allow_trimmed_clips is disabled"
            )
        if not is_full_clip:
            trimmed_count += 1

        scene_inputs.append(
            _TimelineSceneInput(
                clip=clip,
                path=source_path,
                source_in=source_in,
                source_out=source_out,
                actual_duration=actual_duration,
                is_full_clip=is_full_clip,
            )
        )
        probes.append(probe)

    return scene_inputs, probes, trimmed_count


def _find_timeline_music_clip(timeline: "timeline_engine.Timeline") -> "timeline_engine.AudioClip | None":
    """First enabled AudioClip on any audio track — purely descriptive;
    never used to auto-enable music mixing."""
    for track in timeline.tracks:
        if track.track_type != timeline_engine.TrackType.AUDIO:
            continue
        for clip in track.clips:
            if isinstance(clip, timeline_engine.AudioClip) and clip.enabled:
                return clip
    return None


def _inject_trim_flags(command: list[str], trims: dict[str, tuple[float, float]]) -> list[str]:
    """
    Inserts per-input `-ss <source_in> -t <duration>` immediately before
    the matching `-i <path>` pair, for ffmpeg input-seeking-based
    trimming. trims maps a VideoInput's exact command-line path string
    to (source_in_seconds, duration_seconds). Because the normalized-
    render command always re-encodes (never stream-copies), input
    seeking here lands on an exact re-encoded frame boundary rather than
    the nearest keyframe — the "safe non-copy path" Phase 11C.1 requires
    for trimmed clips. video_engine.py itself is not modified for this —
    see the Phase 11C.1 plan for why command-level post-processing was
    chosen instead.
    """
    result: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == "-i" and index + 1 < len(command) and command[index + 1] in trims:
            source_in, duration = trims[command[index + 1]]
            result.extend(["-ss", f"{source_in:.6f}", "-t", f"{duration:.6f}"])
        result.append(token)
        index += 1
    return result


def _execute_normalized_render_with_trim(
    engine: "video_engine.VideoEngine",
    plan: "video_engine.ConcatPlan",
    trims: dict[str, tuple[float, float]],
    *,
    force: bool,
) -> "video_engine.VideoEngineResult":
    """
    Mirrors VideoEngine.execute_plan()'s NORMALIZED_RENDER path exactly
    (same validation/overwrite/verification/error semantics — reuses its
    already-public validate_inputs()/build_normalized_concat_command()/
    verify_output() methods directly) but injects per-input trim flags
    into the constructed command before running it. video_engine.py
    itself is not modified.
    """
    engine.validate_inputs(plan.inputs, plan.output)

    if plan.output.path.exists() and force is False and engine.config.overwrite_requires_force:
        raise video_engine.VideoOutputExistsError(
            f"{plan.output.path} already exists; pass force=True to overwrite."
        )

    command = engine.build_normalized_concat_command(plan, force=force)
    command = _inject_trim_flags(command, trims)

    started_at = _now_iso()
    start_monotonic = time.monotonic()
    process_result = engine.runner(command, timeout=engine.config.ffmpeg_timeout_seconds)
    finished_at = _now_iso()
    duration_seconds = time.monotonic() - start_monotonic

    if process_result.returncode != 0:
        raise video_engine.FFmpegExecutionError(
            f"ffmpeg exited {process_result.returncode}: {process_result.stderr.strip()[-2000:]}"
        )

    output_exists, output_size_bytes = engine.verify_output(plan.output.path)
    if not output_exists:
        raise video_engine.VideoOutputVerificationError(
            f"ffmpeg reported success but no output file exists: {plan.output.path}"
        )
    if engine.config.verify_non_zero_bytes and output_size_bytes == 0:
        raise video_engine.VideoOutputVerificationError(f"Output file is zero bytes: {plan.output.path}")

    return video_engine.VideoEngineResult(
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
        manifest_path=None,
        temporary_files=[],
        cleanup_result="not_applicable",
        warnings=[],
        error=None,
    )


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
    # Phase 11B.1 — always None for a no-music build (existing callers
    # unaffected); a dict shaped per build_log.json's music_mix field
    # when --music was supplied.
    music_mix: dict[str, Any] | None = None
    # Phase 11C.1 — always None for a build without --timeline (existing
    # callers unaffected); a dict shaped per build_log.json's timeline
    # field when --timeline was supplied.
    timeline: dict[str, Any] | None = None

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
    engine_plan_type: str | None = None,
    engine_reasons: list[str] | None = None,
    manifest_path: str | None = None,
    temporary_files: list[str] | None = None,
    cleanup_result: str | None = None,
    music_mix: dict[str, Any] | None = None,
    timeline: dict[str, Any] | None = None,
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
        # Phase 11A.3 additions — optional, additive only. Never removes
        # any field a Phase 11A consumer of this log already relies on.
        "engine_plan_type": engine_plan_type,
        "engine_reasons": engine_reasons or [],
        "manifest_path": manifest_path,
        "temporary_files": temporary_files or [],
        "cleanup_result": cleanup_result,
        # Phase 11B.1 addition — always present. {"enabled": false} for
        # every build that did not request --music, so this never
        # changes the log shape for existing (no-music) callers.
        "music_mix": music_mix if music_mix is not None else {"enabled": False},
        # Phase 11C.1 addition — always present. {"enabled": false} for
        # every build that did not request --timeline, so this never
        # changes the log shape for existing (no-timeline) callers.
        "timeline": timeline if timeline is not None else {"enabled": False},
    }

    temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(log_path)
    return log_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _expected_music_diagnostic_log_path(
    output_path: Path, music_config: "music_mixer.MusicMixerConfig"
) -> str | None:
    """
    Mirrors music_mixer._diagnostic_log_path()'s own arithmetic (pure
    path math over already-public MusicMixerConfig fields — not
    validation or command-construction logic) so build_log.json can
    record where Music Mixer wrote its own diagnostic log, without
    modifying music_mixer.py to expose it directly.
    """
    if not music_config.write_log:
        return None
    directory = (
        Path(music_config.diagnostics_directory)
        if music_config.diagnostics_directory
        else output_path.parent
    )
    return str(directory / f"{output_path.stem}{music_config.log_filename_suffix}")


def build_reel(
    date: str,
    config: BuilderConfig,
    *,
    force: bool = False,
    runner: SubprocessRunner = default_runner,
    root: str | Path | None = None,
    music_path: str | Path | None = None,
    music_overrides: dict[str, Any] | None = None,
    keep_intermediate: bool | None = None,
    music_mixer_callable: Callable[..., Any] | None = None,
    timeline_path: str | Path | None = None,
    timeline_loader_callable: Callable[..., Any] | None = None,
    timeline_validator_callable: Callable[..., Any] | None = None,
) -> BuildResult:
    """
    Build output/<date>/videos/reel_final.mp4 from the Reel scene clips
    for that production date. Never publishes anything, never touches
    Instagram, never uses Playwright.

    root defaults to the real app root (10_apps/claude_runtime), what the
    CLI uses; tests pass an injected temp directory instead, the same
    testability pattern instagram_reel_preview.py's output_root uses.

    Phase 11B.1: when music_path is None (the default), this function's
    behavior is byte-for-byte identical to Phase 11A.3 — no intermediate
    file is created and Music Mixer is never imported or invoked. When
    music_path is given, scenes are rendered to an intermediate
    reel_without_music.mp4 first, then music_mixer_callable (defaults to
    music_mixer.mix_music — the real public API, never re-implemented
    here) combines it with music_path into reel_final.mp4.
    music_overrides is a plain dict of MusicMixRequest override fields
    (music_volume/source_audio_volume/fade_in_seconds/fade_out_seconds/
    music_mode/ducking_mode); any key absent or None falls back to Music
    Mixer's own configured defaults.

    Phase 11C.1: when timeline_path is None (the default), this
    function's scene-acquisition stage is byte-for-byte identical to
    Phase 11B.1 — directory scene discovery via discover_scene_clips(),
    Timeline Engine is never imported or invoked. When timeline_path is
    given, it replaces directory scene discovery entirely: the timeline
    is loaded and validated (timeline_loader_callable/
    timeline_validator_callable default to timeline_engine.load_timeline/
    validate_timeline — real public APIs, injectable for tests), its
    primary video track's enabled clips define scene order, and each
    clip's planned duration is reconciled against a real probe_video()
    call before any render planning happens. A trimmed clip always
    forces a NORMALIZED_RENDER plan. Never falls back to directory scene
    discovery if any of this fails.
    """
    started_at = _now_iso()
    validate_date(date)

    music_enabled = music_path is not None
    music_mixer_fn = music_mixer_callable or music_mixer.mix_music
    overrides = music_overrides or {}

    timeline_enabled = timeline_path is not None
    timeline_path_obj = Path(timeline_path) if timeline_enabled else None

    scenes_dir = config.reel_scenes_dir(date, root=root)
    final_output_path = config.reel_final_path(date, root=root)
    intermediate_path = config.reel_without_music_path(date, root=root) if music_enabled else None
    log_path = config.build_log_path(date, root=root)

    render_target_path = intermediate_path if music_enabled else final_output_path

    if not force:
        if final_output_path.exists():
            raise ReelAlreadyExistsError(
                f"{final_output_path} already exists; pass --force to overwrite."
            )
        if music_enabled and intermediate_path.exists():
            raise ReelAlreadyExistsError(
                f"{intermediate_path} already exists; pass --force to overwrite."
            )

    music_mix_log_info: dict[str, Any] | None = {"enabled": True} if music_enabled else None
    timeline_log_info: dict[str, Any] = (
        {"enabled": True, "timeline_path": str(timeline_path_obj)} if timeline_enabled else {"enabled": False}
    )

    try:
        if timeline_enabled and not config.timeline_integration_enabled:
            raise TimelineIntegrationError(
                "--timeline was supplied but config timeline_integration.enabled is false"
            )

        timeline_warnings: list[str] = []
        trimmed_count = 0

        if timeline_enabled:
            timeline_obj = _load_and_validate_timeline(
                timeline_path_obj,
                date,
                config,
                loader=timeline_loader_callable,
                validator=timeline_validator_callable,
            )
            timeline_clips, ignored_tracks, ignored_disabled_count = _select_primary_video_clips(
                timeline_obj, config
            )
            scene_inputs, probes, trimmed_count = _reconcile_clip_durations(
                timeline_clips, config, runner
            )
            clips = [
                SceneClip(
                    index=(scene_input.clip.scene_number or position),
                    path=scene_input.path,
                )
                for position, scene_input in enumerate(scene_inputs, start=1)
            ]

            timeline_music_clip = _find_timeline_music_clip(timeline_obj)
            if timeline_music_clip is not None and not music_enabled:
                timeline_warnings.append(
                    "Timeline includes a music track "
                    f"({timeline_music_clip.source_path}) but --music was not supplied; "
                    "automatic music mixing is never performed — pass --music explicitly."
                )
            elif (
                timeline_music_clip is not None
                and music_enabled
                and Path(music_path).resolve() != Path(timeline_music_clip.source_path).resolve()
            ):
                timeline_warnings.append(
                    f"--music ({music_path}) differs from the timeline's music clip "
                    f"({timeline_music_clip.source_path}); the CLI --music path is authoritative."
                )

            planned_duration_seconds = sum(scene_input.clip.duration_seconds for scene_input in scene_inputs)
            reconciled_duration_seconds = sum(
                scene_input.source_out - scene_input.source_in for scene_input in scene_inputs
            )

            timeline_log_info = {
                "enabled": True,
                "timeline_path": str(timeline_path_obj),
                "timeline_id": timeline_obj.timeline_id,
                "schema_version": timeline_obj.schema_version,
                "validation_passed": True,
                "production_date_match": (
                    timeline_obj.production_date is None or timeline_obj.production_date == date
                ),
                "video_clip_count": len(scene_inputs),
                "ignored_disabled_clips": ignored_disabled_count,
                "ignored_tracks": ignored_tracks,
                "planned_duration_seconds": planned_duration_seconds,
                "reconciled_duration_seconds": reconciled_duration_seconds,
                "duration_tolerance_seconds": config.timeline_duration_tolerance_seconds,
                "trimmed_clip_count": trimmed_count,
                "warnings": list(timeline_warnings),
            }
        else:
            clips = discover_scene_clips(scenes_dir, config.scene_count)
            probes = [probe_video(clip.path, config, runner=runner) for clip in clips]
            scene_inputs = None

        media_incompatible = validate_clip_consistency(probes, config)
        normalize = media_incompatible or (timeline_enabled and trimmed_count > 0)

        audio_flags = {probe.has_audio for probe in probes}
        if len(audio_flags) > 1:
            raise MixedAudioPresenceError(
                "Scene clips have mixed audio presence (some have an audio "
                "stream, some do not); refusing to guess — silence "
                "synthesis is not implemented in Phase 11A.3."
            )

        warnings: list[str] = list(timeline_warnings)
        if not all(probe.has_audio for probe in probes):
            warnings.append(
                "one_or_more_clips_missing_audio_stream_output_audio_may_be_incomplete"
            )

        render_target_path.parent.mkdir(parents=True, exist_ok=True)

        engine_config = _video_engine_config_from(config)
        engine = video_engine.VideoEngine(engine_config, runner=runner)

        inputs = [_video_input_from_probe(clip, probe) for clip, probe in zip(clips, probes)]
        output_spec = video_engine.VideoOutputSpec(
            path=render_target_path,
            video_codec=config.video_codec,
            video_bitrate=config.video_bitrate,
            fps=config.fps,
            width=probes[0].width if normalize else None,
            height=probes[0].height if normalize else None,
            audio_codec=config.audio_codec,
        )

        # Reel Builder owns this plan-type decision itself (via
        # validate_clip_consistency()'s independent normalize_resolution/
        # normalize_fps flags, plus — in timeline mode — whether any clip
        # needs trimming) rather than re-deriving it through
        # engine.build_concat_plan() — see the module docstring for why.
        if timeline_enabled:
            reasons = [
                "timeline_trim_required" if trimmed_count > 0 else "timeline_full_clip",
                "timeline_duration_reconciled",
            ]
            if media_incompatible:
                reasons.append("timeline_media_incompatible")
            if normalize:
                reasons.append("timeline_requires_normalized_render")
        else:
            reasons = (
                ["reel_builder_resolution_or_fps_mismatch_normalized"]
                if normalize
                else ["reel_builder_compatible_for_stream_copy"]
            )

        plan = video_engine.ConcatPlan(
            plan_type=(
                video_engine.PLAN_NORMALIZED_RENDER if normalize else video_engine.PLAN_LOSSLESS_COPY
            ),
            inputs=inputs,
            output=output_spec,
            reasons=reasons,
            target_video=(
                video_engine.VideoStreamSpec(width=probes[0].width, height=probes[0].height)
                if normalize
                else None
            ),
        )

        try:
            if timeline_enabled and trimmed_count > 0:
                trims = {
                    str(scene_input.path): (
                        scene_input.source_in,
                        scene_input.source_out - scene_input.source_in,
                    )
                    for scene_input in scene_inputs
                    if not scene_input.is_full_clip
                }
                engine_result = _execute_normalized_render_with_trim(engine, plan, trims, force=force)
            else:
                engine_result = engine.execute_plan(plan, force=force)
        except video_engine.FFmpegExecutionError as exc:
            raise FfmpegRenderError(str(exc)) from exc
        except video_engine.VideoOutputVerificationError as exc:
            if not render_target_path.is_file():
                raise ReelBuildOutputMissingError(str(exc)) from exc
            raise ReelBuildEmptyOutputError(str(exc)) from exc

        result_output_path = render_target_path
        result_output_size = engine_result.output_size_bytes

        if music_enabled:
            music_request = music_mixer.MusicMixRequest(
                video_path=intermediate_path,
                music_path=Path(music_path),
                output_path=final_output_path,
                force=force,
                music_volume=overrides.get("music_volume"),
                source_audio_volume=overrides.get("source_audio_volume"),
                fade_in_seconds=overrides.get("fade_in_seconds"),
                fade_out_seconds=overrides.get("fade_out_seconds"),
                music_mode=overrides.get("music_mode"),
                ducking_mode=overrides.get("ducking_mode"),
            )
            music_config = music_mixer.load_music_mixer_config()

            try:
                mix_result = music_mixer_fn(music_request, music_config, runner=runner)
            except (music_mixer.MusicMixerError, media_inspector.MediaInspectorError) as exc:
                music_mix_log_info.update(
                    {
                        "status": "failed",
                        "intermediate_path": str(intermediate_path),
                        "intermediate_retained": True,
                        "error": str(exc),
                    }
                )
                raise MusicMixIntegrationError(str(exc)) from exc

            keep_intermediate_effective = (
                keep_intermediate if keep_intermediate is not None else config.keep_intermediate_default
            )

            if keep_intermediate_effective or not config.cleanup_intermediate_on_success:
                intermediate_cleanup = "retained"
            else:
                try:
                    intermediate_path.unlink()
                    intermediate_cleanup = "cleaned"
                except OSError:
                    intermediate_cleanup = "retained"

            music_mix_log_info.update(
                {
                    "music_path": str(music_path),
                    "music_filename": Path(music_path).name,
                    "music_volume": mix_result.plan.music_volume,
                    "source_audio_volume": mix_result.plan.source_audio_volume,
                    "music_mode": mix_result.plan.music_mode,
                    "ducking_mode": mix_result.plan.ducking_mode,
                    "fade_in_seconds": mix_result.plan.fade_in_seconds,
                    "fade_out_seconds": mix_result.plan.fade_out_seconds,
                    "intermediate_path": str(intermediate_path),
                    "intermediate_retained": intermediate_cleanup == "retained",
                    "status": "success",
                    "mix_duration_seconds": mix_result.duration_seconds,
                    "diagnostic_log": _expected_music_diagnostic_log_path(
                        final_output_path, music_config
                    ),
                    "warnings": mix_result.warnings,
                }
            )

            result_output_path = final_output_path
            result_output_size = mix_result.output_size_bytes

        clip_durations = (
            [scene_input.source_out - scene_input.source_in for scene_input in scene_inputs]
            if timeline_enabled
            else [probe.duration_seconds for probe in probes]
        )

        build_result = BuildResult(
            production_date=date,
            scene_clips=[str(clip.path) for clip in clips],
            clip_durations=clip_durations,
            render_duration_seconds=engine_result.duration_seconds,
            ffmpeg_command=engine_result.command,
            output_path=str(result_output_path),
            output_size_bytes=result_output_size,
            normalized=normalize,
            warnings=warnings,
            music_mix=music_mix_log_info,
            timeline=timeline_log_info if timeline_enabled else None,
        )

        _write_build_log(
            log_path=log_path,
            started_at=started_at,
            production_date=date,
            input_clips=[clip.to_dict() for clip in clips],
            clip_durations=build_result.clip_durations,
            render_duration_seconds=engine_result.duration_seconds,
            ffmpeg_command=engine_result.command,
            output_size_bytes=result_output_size,
            warnings=warnings,
            errors=[],
            result="success",
            engine_plan_type=plan.plan_type,
            engine_reasons=plan.reasons,
            manifest_path=engine_result.manifest_path,
            temporary_files=engine_result.temporary_files,
            cleanup_result=engine_result.cleanup_result,
            music_mix=music_mix_log_info,
            timeline=timeline_log_info,
        )

        return build_result

    except ReelBuilderError as exc:
        if isinstance(exc, MusicMixIntegrationError):
            errors = ["music mix failed", str(exc)]
        elif isinstance(exc, TimelineIntegrationError):
            errors = ["timeline integration failed", str(exc)]
            timeline_log_info = {
                **timeline_log_info,
                "enabled": True,
                "validation_passed": False,
                "error": str(exc),
            }
        else:
            errors = [str(exc)]

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
            errors=errors,
            result="failed",
            music_mix=music_mix_log_info,
            timeline=timeline_log_info,
        )
        raise


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
        help="Overwrite an existing reel_final.mp4 (and intermediate, if --music is used).",
    )

    # Phase 11B.1 — optional Music Mixer integration. Absent by default,
    # so the original Phase 11A.3 CLI/behavior is unaffected unless
    # --music is explicitly supplied. There is no default/automatic
    # music selection anywhere — a music file must always be named
    # explicitly.
    parser.add_argument(
        "--music",
        default=None,
        help="Path to a background-music file. Optional — omit for the original no-music behavior.",
    )
    parser.add_argument("--music-volume", dest="music_volume", type=float, default=None)
    parser.add_argument("--source-audio-volume", dest="source_audio_volume", type=float, default=None)
    parser.add_argument("--music-mode", dest="music_mode", choices=music_mixer.MusicMode.ALL, default=None)
    parser.add_argument("--ducking", dest="ducking_mode", choices=music_mixer.DuckingMode.ALL, default=None)
    parser.add_argument("--fade-in-seconds", dest="fade_in_seconds", type=float, default=None)
    parser.add_argument("--fade-out-seconds", dest="fade_out_seconds", type=float, default=None)
    parser.add_argument(
        "--keep-intermediate",
        dest="keep_intermediate",
        action="store_true",
        help="Retain reel_without_music.mp4 after a successful music mix (requires --music).",
    )

    # Phase 11C.1 — optional Timeline Engine integration. Absent by
    # default, so behavior is unaffected unless --timeline is explicitly
    # supplied. There is no implicit timeline auto-discovery anywhere.
    parser.add_argument(
        "--timeline",
        default=None,
        help=(
            "Path to a validated timeline.json produced by Timeline Engine "
            "(Phase 11C). Optional — omit for the original scene-directory-"
            "discovery behavior."
        ),
    )

    arguments = parser.parse_args(argv)

    music_specific_flags_set = any(
        (
            arguments.music_volume is not None,
            arguments.source_audio_volume is not None,
            arguments.music_mode is not None,
            arguments.ducking_mode is not None,
            arguments.fade_in_seconds is not None,
            arguments.fade_out_seconds is not None,
            arguments.keep_intermediate,
        )
    )

    if music_specific_flags_set and not arguments.music:
        parser.error(
            "--music-volume/--source-audio-volume/--music-mode/--ducking/"
            "--fade-in-seconds/--fade-out-seconds/--keep-intermediate require --music."
        )

    return arguments


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
    if result.music_mix is not None:
        print(f"music mix:               {result.music_mix}")
    if result.timeline is not None:
        print(f"timeline:                {result.timeline}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    music_overrides = {
        "music_volume": arguments.music_volume,
        "source_audio_volume": arguments.source_audio_volume,
        "fade_in_seconds": arguments.fade_in_seconds,
        "fade_out_seconds": arguments.fade_out_seconds,
        "music_mode": arguments.music_mode,
        "ducking_mode": arguments.ducking_mode,
    }

    # store_true always yields a concrete bool (never None), so only
    # ever pass an explicit True through — otherwise config's own
    # keep_intermediate_default would be permanently overridden by an
    # implicit False on every run.
    keep_intermediate = True if arguments.keep_intermediate else None

    try:
        config = load_builder_config(arguments.config)
        result = build_reel(
            arguments.date,
            config,
            force=arguments.force,
            music_path=arguments.music,
            music_overrides=music_overrides,
            keep_intermediate=keep_intermediate,
            timeline_path=arguments.timeline,
        )
    except ReelBuilderError as exc:
        print(f"[ReelBuilder] {exc}")
        raise SystemExit(1) from exc

    _print_result(result)


if __name__ == "__main__":
    main()
