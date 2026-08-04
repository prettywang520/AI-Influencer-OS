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

# Phase 11C — Timeline Engine. A reusable, platform-neutral timeline
# model and planner for short-form video editing: describes ordered
# clips, exact start/end times, trim ranges, track structure,
# background-music placement, and room for a future subtitle/logo/outro
# track. Plans and validates timelines only. This module never invokes
# ffmpeg or ffprobe, never renders anything, never inspects real media
# (no src/media_inspector.py import either), and never publishes or
# uploads. Scene durations come only from a sidecar file,
# production_manifest.json, or a configured default — never from
# probing the video file itself.

DEFAULT_TIMELINE_CONFIG_RELATIVE_PATH = Path("config") / "video" / "timeline.yaml"

SCENE_FILENAME_PATTERN = re.compile(r"^(?:scene_)?(\d{2,})(\.[A-Za-z0-9]+)$", re.IGNORECASE)


def _runtime_root() -> Path:
    """
    timeline_engine.py location: 10_apps/claude_runtime/src/timeline_engine.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TimelineEngineError(RuntimeError):
    """Base error for the Phase 11C Timeline Engine."""


class TimelineConfigError(TimelineEngineError):
    """Raised when config/video/timeline.yaml is missing or invalid."""


class TimelineInputError(TimelineEngineError):
    """Raised when a scenes directory or other required input is invalid."""


class SceneDiscoveryError(TimelineEngineError):
    """Raised for scene-discovery integrity problems other than a
    duplicate/missing scene number (zero-byte file, resolved-path
    duplicate, unsupported extension when rejection is enabled, no
    scenes found)."""


class DuplicateSceneNumberError(TimelineEngineError):
    """Raised when two differently-named scene files share one scene number."""


class MissingSceneNumberError(TimelineEngineError):
    """Raised when the discovered scene numbers have a gap and --keep-gaps
    was not requested."""


class SceneDurationError(TimelineEngineError):
    """Raised when no configured duration source resolves a scene's
    duration, or a source that IS present is invalid (e.g. a malformed
    sidecar file)."""


class TimelineDurationLimitError(TimelineEngineError):
    """Raised when a timeline exceeds a supplied duration limit and
    trimming the final clip is disabled (or cannot fit)."""


class TimelineValidationError(TimelineEngineError):
    """Reserved for callers that want validate_timeline() failures raised
    as an exception rather than inspected via TimelineValidationResult."""


class TimelineFileNotFoundError(TimelineEngineError):
    """Raised when load_timeline() is given a path that does not exist."""


class TimelineJSONError(TimelineEngineError):
    """Raised when timeline JSON is malformed or missing required fields."""


class TimelineOutputExistsError(TimelineEngineError):
    """Raised when save_timeline() targets an existing file without force=True."""


class UnsafeTimelineOutputError(TimelineEngineError):
    """Raised when save_timeline() targets a path that is a directory."""


# ---------------------------------------------------------------------------
# String-constant "enums"
# ---------------------------------------------------------------------------


class TrackType:
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    OVERLAY = "overlay"
    METADATA = "metadata"
    ALL = (VIDEO, AUDIO, SUBTITLE, OVERLAY, METADATA)


class ClipType:
    VIDEO = "video"
    AUDIO = "audio"
    OVERLAY = "overlay"
    ALL = (VIDEO, AUDIO, OVERLAY)


DURATION_SOURCE_SIDECAR = "sidecar"
DURATION_SOURCE_MANIFEST = "manifest"
DURATION_SOURCE_DEFAULT = "default"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TimelineConfig:
    schema_version: str = "1.0"

    fps: float = 30
    width: int = 1080
    height: int = 1920
    validation_tolerance_seconds: float = 0.01

    accept_scene_prefixed_names: bool = True
    accept_plain_numbered_names: bool = True
    require_sequential_numbers: bool = True
    supported_extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v")
    reject_unsupported_extensions: bool = False

    duration_priority: tuple[str, ...] = (
        DURATION_SOURCE_SIDECAR,
        DURATION_SOURCE_MANIFEST,
        DURATION_SOURCE_DEFAULT,
    )
    allow_default_duration: bool = True
    default_scene_duration_seconds: float = 2.5

    duration_limit_policy: str = "fail"
    allow_trim_final_clip: bool = False

    music_loop: bool = True
    music_volume: float = 0.20
    music_fade_in_seconds: float = 0.50
    music_fade_out_seconds: float = 1.00
    music_ducking_mode: str = "none"

    allow_video_gaps: bool = False
    allow_audio_overlap: bool = False

    overwrite_requires_force: bool = True


def default_timeline_config_path() -> Path:
    return _runtime_root() / DEFAULT_TIMELINE_CONFIG_RELATIVE_PATH


def load_timeline_config(config_path: str | Path | None = None) -> TimelineConfig:
    """Load config/video/timeline.yaml (or an alternate path) into a
    TimelineConfig. Raises TimelineConfigError if the file is missing or
    invalid. Never touches ffmpeg/ffprobe/media."""
    path = Path(config_path) if config_path else default_timeline_config_path()

    if not path.exists():
        raise TimelineConfigError(f"Timeline config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise TimelineConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise TimelineConfigError(f"Timeline config is empty or invalid: {path}")

    timeline_section = raw.get("timeline") or {}
    scene_discovery_section = raw.get("scene_discovery") or {}
    duration_source_section = raw.get("duration_source") or {}
    duration_limit_section = raw.get("duration_limit") or {}
    music_section = raw.get("music") or {}
    validation_section = raw.get("validation") or {}
    output_section = raw.get("output") or {}

    extensions = scene_discovery_section.get("supported_extensions") or [".mp4", ".mov", ".m4v"]
    priority = duration_source_section.get("priority") or [
        DURATION_SOURCE_SIDECAR,
        DURATION_SOURCE_MANIFEST,
        DURATION_SOURCE_DEFAULT,
    ]

    return TimelineConfig(
        schema_version=str(raw.get("schema_version", "1.0")),
        fps=float(timeline_section.get("fps", 30)),
        width=int(timeline_section.get("width", 1080)),
        height=int(timeline_section.get("height", 1920)),
        validation_tolerance_seconds=float(timeline_section.get("validation_tolerance_seconds", 0.01)),
        accept_scene_prefixed_names=bool(scene_discovery_section.get("accept_scene_prefixed_names", True)),
        accept_plain_numbered_names=bool(scene_discovery_section.get("accept_plain_numbered_names", True)),
        require_sequential_numbers=bool(scene_discovery_section.get("require_sequential_numbers", True)),
        supported_extensions=tuple(str(ext).lower() for ext in extensions),
        reject_unsupported_extensions=bool(
            scene_discovery_section.get("reject_unsupported_extensions", False)
        ),
        duration_priority=tuple(str(source) for source in priority),
        allow_default_duration=bool(duration_source_section.get("allow_default_duration", True)),
        default_scene_duration_seconds=float(
            duration_source_section.get("default_scene_duration_seconds", 2.5)
        ),
        duration_limit_policy=str(duration_limit_section.get("policy", "fail")),
        allow_trim_final_clip=bool(duration_limit_section.get("allow_trim_final_clip", False)),
        music_loop=bool(music_section.get("loop", True)),
        music_volume=float(music_section.get("volume", 0.20)),
        music_fade_in_seconds=float(music_section.get("fade_in_seconds", 0.50)),
        music_fade_out_seconds=float(music_section.get("fade_out_seconds", 1.00)),
        music_ducking_mode=str(music_section.get("ducking_mode", "none")),
        allow_video_gaps=bool(validation_section.get("allow_video_gaps", False)),
        allow_audio_overlap=bool(validation_section.get("allow_audio_overlap", False)),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TimelineClip:
    clip_id: str = ""
    track_id: str = ""
    clip_type: str = ClipType.VIDEO
    source_path: str = ""
    start: float = 0.0
    end: float = 0.0
    duration_seconds: float = 0.0
    source_in: float = 0.0
    source_out: float | None = None
    playback_rate: float = 1.0
    transition_in: str = "none"
    transition_out: str = "none"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoClip(TimelineClip):
    clip_type: str = ClipType.VIDEO
    scene_number: int | None = None


@dataclass(slots=True)
class AudioClip(TimelineClip):
    clip_type: str = ClipType.AUDIO
    volume: float = 1.0
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0
    loop: bool = False
    ducking_mode: str = "none"


@dataclass(slots=True)
class OverlayClip(TimelineClip):
    clip_type: str = ClipType.OVERLAY
    overlay_type: str = "text"
    content: str = ""


_CLIP_CLASS_BY_TYPE: dict[str, type[TimelineClip]] = {
    ClipType.VIDEO: VideoClip,
    ClipType.AUDIO: AudioClip,
    ClipType.OVERLAY: OverlayClip,
}


@dataclass(slots=True)
class TimelineTrack:
    track_id: str = ""
    track_type: str = TrackType.VIDEO
    order: int = 0
    enabled: bool = True
    clips: list[TimelineClip] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TimelineMarker:
    marker_id: str = ""
    time_seconds: float = 0.0
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Timeline:
    timeline_id: str = ""
    schema_version: str = "1.0"
    production_date: str | None = None
    created_at: str = ""
    duration_seconds: float = 0.0
    fps: float = 30
    width: int = 1080
    height: int = 1920
    tracks: list[TimelineTrack] = field(default_factory=list)
    markers: list[TimelineMarker] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TimelineValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Scene discovery — no ffmpeg, no ffprobe, no media inspection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SceneFile:
    index: int
    path: Path


def discover_scenes(
    directory: str | Path, config: TimelineConfig, *, keep_gaps: bool = False
) -> list[SceneFile]:
    """
    Pure filesystem/filename discovery of scene clips — never opens the
    file to probe its contents. Rejects zero-byte files, duplicate scene
    numbers (two differently-spelled filenames sharing one number), and
    two different scene numbers whose files resolve to the same real
    path. Gaps in the numbering raise MissingSceneNumberError unless
    keep_gaps is True.
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise TimelineInputError(f"Scenes directory not found: {directory}")

    by_number: dict[int, Path] = {}
    resolved_by_number: dict[int, Path] = {}
    resolved_seen: dict[Path, int] = {}

    for entry in sorted(directory.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue

        extension = entry.suffix.lower()
        if extension not in config.supported_extensions:
            if config.reject_unsupported_extensions:
                raise SceneDiscoveryError(f"Unsupported scene file extension: {entry.name}")
            continue

        match = SCENE_FILENAME_PATTERN.match(entry.name)
        if not match:
            continue

        is_prefixed = entry.name.lower().startswith("scene_")
        if is_prefixed and not config.accept_scene_prefixed_names:
            continue
        if not is_prefixed and not config.accept_plain_numbered_names:
            continue

        scene_number = int(match.group(1))

        if entry.stat().st_size == 0:
            raise SceneDiscoveryError(f"Scene file is zero bytes: {entry.name}")

        if scene_number in by_number:
            raise DuplicateSceneNumberError(
                f"Scene number {scene_number:02d} appears more than once: "
                f"{by_number[scene_number].name!r} and {entry.name!r}"
            )
        by_number[scene_number] = entry

        resolved = entry.resolve()
        if resolved in resolved_seen:
            other_number = resolved_seen[resolved]
            raise SceneDiscoveryError(
                f"Scenes {other_number:02d} and {scene_number:02d} resolve to the same "
                f"real file: {resolved}"
            )
        resolved_seen[resolved] = scene_number
        resolved_by_number[scene_number] = resolved

    if not by_number:
        raise SceneDiscoveryError(f"No scene files found in {directory}")

    ordered_numbers = sorted(by_number)

    if config.require_sequential_numbers and not keep_gaps:
        expected = ordered_numbers[0]
        for number in ordered_numbers:
            if number != expected:
                raise MissingSceneNumberError(
                    f"Missing scene number {expected:02d} (found {number:02d} next)"
                )
            expected += 1

    return [SceneFile(index=number, path=by_number[number]) for number in ordered_numbers]


# ---------------------------------------------------------------------------
# Duration resolution — no ffprobe, no media inspection
# ---------------------------------------------------------------------------


def _load_production_manifest(scenes_dir: Path) -> dict[str, Any] | None:
    """
    Reads output/<date>/production_manifest.json given a scenes directory
    at output/<date>/videos/reel_scenes (Phase 9's layout). Returns None
    if the file does not exist or cannot be parsed — the manifest is an
    optional, shared duration source, so its absence/corruption simply
    means this source is unavailable, not a hard failure.
    """
    manifest_path = scenes_dir.parent.parent / "production_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _duration_from_sidecar(scene_path: Path) -> float | None:
    sidecar_path = scene_path.with_suffix(".json")
    if not sidecar_path.is_file():
        return None

    try:
        with sidecar_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneDurationError(f"Invalid sidecar JSON for {scene_path.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise SceneDurationError(f"Sidecar for {scene_path.name} must contain a JSON object")

    value = data.get("duration_seconds")
    if not _is_positive_number(value):
        raise SceneDurationError(
            f"Sidecar duration_seconds for {scene_path.name} is missing or invalid: {value!r}"
        )
    return float(value)


def _duration_from_manifest(manifest: dict[str, Any] | None, scene_number: int) -> float | None:
    if manifest is None:
        return None

    reel_tasks = manifest.get("reel_tasks")
    if not isinstance(reel_tasks, list):
        return None

    for entry in reel_tasks:
        if not isinstance(entry, dict):
            continue
        if entry.get("scene_number") != scene_number:
            continue
        value = entry.get("duration_seconds")
        return float(value) if _is_positive_number(value) else None

    return None


def resolve_scene_duration(
    scene: SceneFile, config: TimelineConfig, manifest: dict[str, Any] | None
) -> tuple[float, str]:
    """
    Walks config.duration_priority in order. A sidecar file that EXISTS
    but is invalid is a hard SceneDurationError immediately (not a
    silent fallthrough) — an ABSENT sidecar simply falls through to the
    next source. A bad manifest entry for this one scene is treated as
    "no manifest duration for this scene" rather than aborting
    resolution for other scenes.
    """
    for source in config.duration_priority:
        if source == DURATION_SOURCE_SIDECAR:
            value = _duration_from_sidecar(scene.path)
            if value is not None:
                return value, DURATION_SOURCE_SIDECAR
        elif source == DURATION_SOURCE_MANIFEST:
            value = _duration_from_manifest(manifest, scene.index)
            if value is not None:
                return value, DURATION_SOURCE_MANIFEST
        elif source == DURATION_SOURCE_DEFAULT:
            if config.allow_default_duration:
                return config.default_scene_duration_seconds, DURATION_SOURCE_DEFAULT

    raise SceneDurationError(
        f"No duration source resolved scene {scene.index:02d} ({scene.path.name}); "
        f"tried: {list(config.duration_priority)}"
    )


# ---------------------------------------------------------------------------
# Deterministic timeline_id
# ---------------------------------------------------------------------------


def _compute_timeline_id(
    *,
    production_date: str | None,
    scene_filenames: list[str],
    scene_durations: list[float],
    music_path: str | None,
    schema_version: str,
) -> str:
    payload = {
        "production_date": production_date,
        "scene_filenames": list(scene_filenames),
        "scene_durations": [round(value, 6) for value in scene_durations],
        "music_path": music_path,
        "schema_version": schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Timeline construction — no rendering, no ffmpeg, no ffprobe
# ---------------------------------------------------------------------------


def build_timeline_from_scenes(
    scenes_dir: str | Path,
    config: TimelineConfig,
    *,
    production_date: str | None = None,
    music_path: str | Path | None = None,
    music_overrides: dict[str, Any] | None = None,
    duration_limit_seconds: float | None = None,
    keep_gaps: bool = False,
) -> Timeline:
    """
    Discovers scene clips, resolves each one's duration, and lays them
    out sequentially on a single video track (no overlap, no
    transitions, playback_rate always 1.0). Optionally appends one music
    track spanning the resulting duration. Never calls ffmpeg/ffprobe,
    never inspects real media — this is planning only.
    """
    scenes_dir = Path(scenes_dir)
    scenes = discover_scenes(scenes_dir, config, keep_gaps=keep_gaps)
    manifest = _load_production_manifest(scenes_dir)

    video_track = TimelineTrack(track_id="track_video", track_type=TrackType.VIDEO, order=0)

    running_total = 0.0
    duration_sources: dict[str, str] = {}
    warnings: list[str] = []
    scene_filenames: list[str] = []
    scene_durations: list[float] = []

    for scene in scenes:
        duration, source = resolve_scene_duration(scene, config, manifest)
        if source == DURATION_SOURCE_DEFAULT:
            warnings.append(
                f"scene {scene.index:02d}: using default duration {duration}s "
                f"(no sidecar/manifest entry found)"
            )

        start = running_total
        end = start + duration
        video_track.clips.append(
            VideoClip(
                clip_id=f"clip_video_{scene.index:02d}",
                track_id=video_track.track_id,
                source_path=str(scene.path),
                start=start,
                end=end,
                duration_seconds=duration,
                source_in=0.0,
                source_out=duration,
                scene_number=scene.index,
            )
        )
        running_total = end
        duration_sources[f"{scene.index:02d}"] = source
        scene_filenames.append(scene.path.name)
        scene_durations.append(duration)

    total_duration = running_total

    if duration_limit_seconds is not None:
        overshoot = total_duration - duration_limit_seconds
        if overshoot > config.validation_tolerance_seconds:
            if not config.allow_trim_final_clip or not video_track.clips:
                raise TimelineDurationLimitError(
                    f"Timeline duration {total_duration:.3f}s exceeds limit "
                    f"{duration_limit_seconds:.3f}s (allow_trim_final_clip is disabled)"
                )

            final_clip = video_track.clips[-1]
            trimmed_duration = final_clip.duration_seconds - overshoot
            if trimmed_duration <= 0:
                raise TimelineDurationLimitError(
                    f"Cannot trim final clip (scene {final_clip.scene_number:02d}, "
                    f"{final_clip.duration_seconds:.3f}s) by {overshoot:.3f}s to fit "
                    f"duration limit {duration_limit_seconds:.3f}s"
                )

            final_clip.end -= overshoot
            final_clip.duration_seconds = trimmed_duration
            final_clip.source_out = final_clip.source_in + trimmed_duration
            total_duration = duration_limit_seconds
            scene_durations[-1] = trimmed_duration
            warnings.append(
                f"trimmed final clip (scene {final_clip.scene_number:02d}) by "
                f"{overshoot:.3f}s to fit duration limit {duration_limit_seconds:.3f}s"
            )

    tracks: list[TimelineTrack] = [video_track]

    music_path_str: str | None = None
    if music_path is not None:
        overrides = music_overrides or {}
        music_path_obj = Path(music_path)
        music_path_str = str(music_path_obj)

        audio_track = TimelineTrack(track_id="track_audio_music", track_type=TrackType.AUDIO, order=1)
        audio_track.clips.append(
            AudioClip(
                clip_id="clip_audio_music",
                track_id=audio_track.track_id,
                source_path=music_path_str,
                start=0.0,
                end=total_duration,
                duration_seconds=total_duration,
                source_in=0.0,
                source_out=total_duration,
                volume=float(overrides.get("volume", config.music_volume)),
                fade_in_seconds=float(overrides.get("fade_in_seconds", config.music_fade_in_seconds)),
                fade_out_seconds=float(overrides.get("fade_out_seconds", config.music_fade_out_seconds)),
                loop=config.music_loop,
                ducking_mode=str(overrides.get("ducking_mode", config.music_ducking_mode)),
            )
        )
        tracks.append(audio_track)

    metadata: dict[str, Any] = {
        "scene_duration_sources": duration_sources,
        "build_warnings": warnings,
    }

    timeline_id = _compute_timeline_id(
        production_date=production_date,
        scene_filenames=scene_filenames,
        scene_durations=scene_durations,
        music_path=music_path_str,
        schema_version=config.schema_version,
    )

    return Timeline(
        timeline_id=timeline_id,
        schema_version=config.schema_version,
        production_date=production_date,
        created_at=_now_iso(),
        duration_seconds=total_duration,
        fps=config.fps,
        width=config.width,
        height=config.height,
        tracks=tracks,
        markers=[],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_timeline(
    timeline: Timeline, config: TimelineConfig | None = None
) -> TimelineValidationResult:
    """
    Validates a Timeline on its own merits — whether hand-constructed or
    loaded from disk, not just "does this look like something
    build_timeline_from_scenes() would have produced." Returns a
    TimelineValidationResult with `passed=False` iff any hard check
    fails; never raises for a merely-invalid timeline.
    """
    cfg = config or TimelineConfig()
    tolerance = cfg.validation_tolerance_seconds
    errors: list[str] = []
    warnings: list[str] = []

    track_ids = [track.track_id for track in timeline.tracks]
    if len(track_ids) != len(set(track_ids)):
        errors.append("duplicate track_id values in timeline.tracks")

    track_orders = [track.order for track in timeline.tracks]
    if len(track_orders) != len(set(track_orders)):
        errors.append("duplicate track order values in timeline.tracks")

    all_clip_ids: list[str] = []
    video_tracks_with_enabled_clips = 0
    max_enabled_end = 0.0

    for track in timeline.tracks:
        for clip in track.clips:
            all_clip_ids.append(clip.clip_id)

            if clip.track_id != track.track_id:
                errors.append(
                    f"clip {clip.clip_id} references track_id {clip.track_id!r} but is "
                    f"stored under track {track.track_id!r}"
                )
            if clip.start < 0 or clip.end < 0:
                errors.append(f"clip {clip.clip_id} has a negative start/end ({clip.start}, {clip.end})")
            if clip.end <= clip.start:
                errors.append(f"clip {clip.clip_id} end ({clip.end}) is not greater than start ({clip.start})")

            expected_duration = clip.end - clip.start
            if abs(clip.duration_seconds - expected_duration) > tolerance:
                errors.append(
                    f"clip {clip.clip_id} duration_seconds ({clip.duration_seconds}) does not "
                    f"match end-start ({expected_duration})"
                )

            if clip.source_out is not None and clip.source_out <= clip.source_in:
                errors.append(
                    f"clip {clip.clip_id} source_out ({clip.source_out}) is not greater than "
                    f"source_in ({clip.source_in})"
                )

            if clip.playback_rate <= 0:
                errors.append(f"clip {clip.clip_id} playback_rate must be > 0, got {clip.playback_rate}")

            if clip.transition_in != "none":
                errors.append(
                    f"clip {clip.clip_id} transition_in {clip.transition_in!r} is not "
                    f"supported in this phase (must be 'none')"
                )
            if clip.transition_out != "none":
                errors.append(
                    f"clip {clip.clip_id} transition_out {clip.transition_out!r} is not "
                    f"supported in this phase (must be 'none')"
                )

            if not clip.source_path:
                errors.append(f"clip {clip.clip_id} has an empty source_path")

            try:
                json.dumps(clip.metadata)
            except TypeError:
                errors.append(f"clip {clip.clip_id} metadata is not JSON-serializable")

            if isinstance(clip, AudioClip):
                if clip.volume < 0:
                    errors.append(f"clip {clip.clip_id} volume must be >= 0, got {clip.volume}")
                if clip.fade_in_seconds < 0 or clip.fade_out_seconds < 0:
                    errors.append(f"clip {clip.clip_id} fade values must be >= 0")
                elif clip.fade_in_seconds + clip.fade_out_seconds > clip.duration_seconds + tolerance:
                    errors.append(
                        f"clip {clip.clip_id} combined fade duration exceeds clip duration"
                    )

            if clip.enabled:
                max_enabled_end = max(max_enabled_end, clip.end)

        enabled_clips = sorted((c for c in track.clips if c.enabled), key=lambda c: c.start)
        for previous, current in zip(enabled_clips, enabled_clips[1:]):
            gap = current.start - previous.end
            if gap < -tolerance:
                if not (track.track_type == TrackType.AUDIO and cfg.allow_audio_overlap):
                    errors.append(
                        f"clips {previous.clip_id} and {current.clip_id} overlap on "
                        f"track {track.track_id}"
                    )
            elif gap > tolerance and track.track_type == TrackType.VIDEO and not cfg.allow_video_gaps:
                errors.append(
                    f"gap between clips {previous.clip_id} and {current.clip_id} on "
                    f"video track {track.track_id}"
                )

        if track.track_type == TrackType.VIDEO and enabled_clips:
            video_tracks_with_enabled_clips += 1

    if len(all_clip_ids) != len(set(all_clip_ids)):
        errors.append("duplicate clip_id values across timeline")

    if video_tracks_with_enabled_clips == 0:
        errors.append("timeline has no video track with at least one enabled clip")

    if abs(timeline.duration_seconds - max_enabled_end) > tolerance:
        errors.append(
            f"timeline.duration_seconds ({timeline.duration_seconds}) does not match the "
            f"maximum enabled clip end ({max_enabled_end})"
        )

    video_clips_in_order = [
        clip
        for track in timeline.tracks
        if track.track_type == TrackType.VIDEO
        for clip in sorted(track.clips, key=lambda c: c.start)
        if isinstance(clip, VideoClip) and clip.enabled
    ]
    scene_numbers = [clip.scene_number for clip in video_clips_in_order if clip.scene_number is not None]
    if len(scene_numbers) != len(set(scene_numbers)):
        errors.append("duplicate scene_number values among video clips")
    elif scene_numbers != sorted(scene_numbers):
        errors.append("scene_number values are not increasing in timeline order")

    passed = len(errors) == 0
    summary = (
        f"Timeline {timeline.timeline_id or '(no id)'}: {'PASSED' if passed else 'FAILED'} "
        f"({len(errors)} error(s), {len(warnings)} warning(s))"
    )

    return TimelineValidationResult(passed=passed, errors=errors, warnings=warnings, summary=summary)


# ---------------------------------------------------------------------------
# JSON serialization — stable, round-trip-safe
# ---------------------------------------------------------------------------


def _clip_to_dict(clip: TimelineClip) -> dict[str, Any]:
    return asdict(clip)


def _clip_from_dict(data: dict[str, Any]) -> TimelineClip:
    clip_type = data.get("clip_type")
    clip_class = _CLIP_CLASS_BY_TYPE.get(clip_type)
    if clip_class is None:
        raise TimelineJSONError(f"Unknown clip_type in timeline JSON: {clip_type!r}")

    known_fields = {f.name for f in dataclasses.fields(clip_class)}
    filtered = {key: value for key, value in data.items() if key in known_fields}
    try:
        return clip_class(**filtered)
    except TypeError as exc:
        raise TimelineJSONError(f"Malformed clip in timeline JSON: {exc}") from exc


def _track_to_dict(track: TimelineTrack) -> dict[str, Any]:
    return {
        "track_id": track.track_id,
        "track_type": track.track_type,
        "order": track.order,
        "enabled": track.enabled,
        "clips": [_clip_to_dict(clip) for clip in track.clips],
        "metadata": track.metadata,
    }


def _track_from_dict(data: dict[str, Any]) -> TimelineTrack:
    try:
        return TimelineTrack(
            track_id=data["track_id"],
            track_type=data["track_type"],
            order=data["order"],
            enabled=data.get("enabled", True),
            clips=[_clip_from_dict(clip) for clip in data.get("clips", [])],
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise TimelineJSONError(f"Timeline track JSON missing required field: {exc}") from exc


def timeline_to_dict(timeline: Timeline) -> dict[str, Any]:
    return {
        "timeline_id": timeline.timeline_id,
        "schema_version": timeline.schema_version,
        "production_date": timeline.production_date,
        "created_at": timeline.created_at,
        "duration_seconds": timeline.duration_seconds,
        "fps": timeline.fps,
        "width": timeline.width,
        "height": timeline.height,
        "tracks": [_track_to_dict(track) for track in timeline.tracks],
        "markers": [asdict(marker) for marker in timeline.markers],
        "metadata": timeline.metadata,
    }


def timeline_from_dict(data: dict[str, Any]) -> Timeline:
    try:
        return Timeline(
            timeline_id=data["timeline_id"],
            schema_version=data.get("schema_version", "1.0"),
            production_date=data.get("production_date"),
            created_at=data.get("created_at", ""),
            duration_seconds=data["duration_seconds"],
            fps=data.get("fps", 30),
            width=data.get("width", 1080),
            height=data.get("height", 1920),
            tracks=[_track_from_dict(track) for track in data.get("tracks", [])],
            markers=[TimelineMarker(**marker) for marker in data.get("markers", [])],
            metadata=data.get("metadata", {}),
        )
    except KeyError as exc:
        raise TimelineJSONError(f"Timeline JSON missing required field: {exc}") from exc
    except TypeError as exc:
        raise TimelineJSONError(f"Malformed timeline JSON: {exc}") from exc


def load_timeline(path: str | Path) -> Timeline:
    path = Path(path)
    if not path.is_file():
        raise TimelineFileNotFoundError(f"Timeline file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise TimelineJSONError(f"Invalid timeline JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise TimelineJSONError(f"Timeline JSON root must be an object: {path}")

    return timeline_from_dict(data)


def save_timeline(timeline: Timeline, path: str | Path, *, force: bool = False) -> Path:
    """Atomically writes timeline JSON via temp-file + Path.replace() so a
    reader never observes a partially-written file. Refuses to overwrite
    an existing file unless force=True."""
    path = Path(path)

    if path.exists():
        if path.is_dir():
            raise UnsafeTimelineOutputError(f"Output path is a directory: {path}")
        if not force:
            raise TimelineOutputExistsError(f"{path} already exists; pass force=True to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(timeline_to_dict(timeline), indent=2, sort_keys=True)
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
            "AIKO Timeline Engine (Phase 11C). Plans and validates "
            "platform-neutral video timelines. Never calls ffmpeg/ffprobe, "
            "never renders, never publishes."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-scenes", metavar="DIR", help="Build a timeline from a scenes directory.")
    mode.add_argument("--validate", metavar="TIMELINE_JSON", help="Validate an existing timeline JSON file.")

    parser.add_argument("--config", default=None, help="Path to an alternate config/video/timeline.yaml.")
    parser.add_argument("--output", default=None, help="Write the built timeline JSON to this path (--from-scenes only).")
    parser.add_argument("--production-date", dest="production_date", default=None)
    parser.add_argument("--duration-limit", dest="duration_limit", type=float, default=None)
    parser.add_argument("--music", default=None, help="Path to a background-music file.")
    parser.add_argument("--music-volume", dest="music_volume", type=float, default=None)
    parser.add_argument("--keep-gaps", dest="keep_gaps", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")

    return parser.parse_args(argv)


def _print_build_summary(timeline: Timeline, output_path: Path | None) -> None:
    video_track = next((t for t in timeline.tracks if t.track_type == TrackType.VIDEO), None)
    audio_track = next((t for t in timeline.tracks if t.track_type == TrackType.AUDIO), None)

    print()
    print("AIKO Timeline Engine (Phase 11C)")
    print("---------------------------------")
    print(f"timeline_id:       {timeline.timeline_id}")
    print(f"production_date:   {timeline.production_date}")
    print(f"scenes:            {len(video_track.clips) if video_track else 0}")
    print(f"tracks:            {len(timeline.tracks)}")
    print(f"duration_seconds:  {timeline.duration_seconds:.3f}")
    print(f"has_music_track:   {audio_track is not None}")
    for scene_key, source in timeline.metadata.get("scene_duration_sources", {}).items():
        print(f"  scene {scene_key} duration_source: {source}")
    for warning in timeline.metadata.get("build_warnings", []):
        print(f"warning: {warning}")
    print(f"output:            {output_path if output_path else '(not written — no --output given)'}")
    print()


def _print_validation_summary(result: TimelineValidationResult) -> None:
    print()
    print("AIKO Timeline Engine — validation (Phase 11C)")
    print("-----------------------------------------------")
    print(result.summary)
    for error in result.errors:
        print(f"  error:   {error}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_timeline_config(arguments.config)

        if arguments.from_scenes:
            music_overrides = {}
            if arguments.music_volume is not None:
                music_overrides["volume"] = arguments.music_volume

            timeline = build_timeline_from_scenes(
                arguments.from_scenes,
                config,
                production_date=arguments.production_date,
                music_path=arguments.music,
                music_overrides=music_overrides or None,
                duration_limit_seconds=arguments.duration_limit,
                keep_gaps=arguments.keep_gaps,
            )

            output_path: Path | None = None
            if arguments.output:
                output_path = save_timeline(timeline, arguments.output, force=arguments.force)

            if arguments.as_json:
                print(json.dumps(timeline_to_dict(timeline), indent=2, sort_keys=True))
            else:
                _print_build_summary(timeline, output_path)
        else:
            timeline = load_timeline(arguments.validate)
            result = validate_timeline(timeline, config)

            if arguments.as_json:
                print(json.dumps(asdict(result), indent=2, sort_keys=True))
            else:
                _print_validation_summary(result)

            if not result.passed:
                raise SystemExit(1)
    except TimelineEngineError as exc:
        print(f"[TimelineEngine] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
