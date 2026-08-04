from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .media_inspector import (
    InspectorConfig,
    MediaInfo,
    MediaInspectorError,
    SubprocessRunner,
    default_runner,
    inspect_file,
    load_inspector_config,
    write_report,
)

# Phase 11A.2 — Video Validator. Purely local validation: takes the
# MediaInfo Media Inspector already produces and scores it against a
# named, config-driven profile. Never renders, transcodes, publishes, or
# uploads anything, and never modifies the inspected file — the only
# file this module can ever write is an explicit --output report path.
# It never duplicates ffprobe parsing (every technical fact comes from
# media_inspector.inspect_file()); the one exception is
# detect_moov_atom_position() below, which uses a completely different,
# non-ffprobe technique (raw ISO-BMFF box header reads) for the one
# signal MediaInfo does not carry.
DISALLOWED_ACTIONS = (
    "render",
    "transcode",
    "concatenate",
    "normalize_media",
    "publish",
    "upload",
    "send",
    "share",
    "modify_media",
)

DEFAULT_VALIDATOR_CONFIG_RELATIVE_PATH = Path("config") / "video" / "validator.yaml"
DEFAULT_PROFILE_NAME = "instagram_reel"

SEVERITY_PASS = "pass"
SEVERITY_WARNING = "warning"
SEVERITY_FAIL = "fail"


def _runtime_root() -> Path:
    """
    video_validator.py location: 10_apps/claude_runtime/src/video_validator.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VideoValidatorError(RuntimeError):
    """Base error for the Phase 11A.2 Video Validator."""


class VideoValidatorConfigError(VideoValidatorError):
    """Raised when config/video/validator.yaml is missing/invalid."""


class UnknownProfileError(VideoValidatorError):
    """Raised when --profile names a profile not present in validator.yaml."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _parse_target_ratio(text: str) -> float:
    """Parses a "W:H" string (e.g. "9:16") into a float. Never raises on
    malformed input — falls back to 1.0 (square) rather than crashing a
    config load over a typo; callers are expected to sanity-check their
    own config."""
    if ":" in text:
        width_str, _, height_str = text.partition(":")
        try:
            width = float(width_str)
            height = float(height_str)
            if height:
                return width / height
        except ValueError:
            pass
    try:
        return float(text)
    except ValueError:
        return 1.0


@dataclass(slots=True)
class ProfileConfig:
    name: str

    duration_min_seconds: float
    duration_max_seconds: float
    duration_penalty: int

    resolution_min_width: int
    resolution_min_height: int
    resolution_penalty: int

    aspect_ratio_target: float
    aspect_ratio_tolerance: float
    aspect_ratio_penalty: int

    fps_allowed: tuple[float, ...]
    fps_tolerance: float
    fps_penalty: int

    video_codec_allowed: tuple[str, ...]
    video_codec_penalty: int

    container_allowed_substrings: tuple[str, ...]
    container_penalty: int

    bitrate_min_bps: int | None
    bitrate_max_bps: int | None
    bitrate_penalty: int

    audio_required: bool
    audio_codec_allowed: tuple[str, ...]
    audio_penalty: int

    sample_rate_allowed: tuple[int, ...]
    sample_rate_penalty: int

    channels_allowed: tuple[int, ...]
    channels_penalty: int

    rotation_allowed: tuple[int, ...]
    rotation_penalty: int

    moov_required_position: str
    moov_applicable_extensions: tuple[str, ...]
    moov_penalty: int

    vertical_required: bool
    vertical_penalty: int


@dataclass(slots=True)
class ValidatorConfig:
    media_inspector_config_path: str | None
    starting_score: int
    minimum_score: int
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)


def default_validator_config_path() -> Path:
    return _runtime_root() / DEFAULT_VALIDATOR_CONFIG_RELATIVE_PATH


def _build_profile_config(name: str, raw: dict[str, Any]) -> ProfileConfig:
    duration = raw.get("duration") or {}
    resolution = raw.get("resolution") or {}
    aspect_ratio = raw.get("aspect_ratio") or {}
    fps = raw.get("fps") or {}
    video_codec = raw.get("video_codec") or {}
    container = raw.get("container") or {}
    bitrate = raw.get("bitrate") or {}
    audio = raw.get("audio") or {}
    sample_rate = raw.get("audio_sample_rate") or {}
    channels = raw.get("audio_channels") or {}
    rotation = raw.get("rotation") or {}
    moov_atom = raw.get("moov_atom") or {}
    vertical = raw.get("vertical_orientation") or {}

    return ProfileConfig(
        name=name,
        duration_min_seconds=float(duration.get("min_seconds", 0)),
        duration_max_seconds=float(duration.get("max_seconds", 10_800)),
        duration_penalty=int(duration.get("penalty", 0)),
        resolution_min_width=int(resolution.get("min_width", 0)),
        resolution_min_height=int(resolution.get("min_height", 0)),
        resolution_penalty=int(resolution.get("penalty", 0)),
        aspect_ratio_target=_parse_target_ratio(str(aspect_ratio.get("target", "1:1"))),
        aspect_ratio_tolerance=float(aspect_ratio.get("tolerance", 0.05)),
        aspect_ratio_penalty=int(aspect_ratio.get("penalty", 0)),
        fps_allowed=tuple(float(v) for v in (fps.get("allowed") or [])),
        fps_tolerance=float(fps.get("tolerance", 0.05)),
        fps_penalty=int(fps.get("penalty", 0)),
        video_codec_allowed=tuple(str(v).lower() for v in (video_codec.get("allowed") or [])),
        video_codec_penalty=int(video_codec.get("penalty", 0)),
        container_allowed_substrings=tuple(
            str(v).lower() for v in (container.get("allowed_format_name_substrings") or [])
        ),
        container_penalty=int(container.get("penalty", 0)),
        bitrate_min_bps=(
            int(bitrate["min_bps"]) if bitrate.get("min_bps") is not None else None
        ),
        bitrate_max_bps=(
            int(bitrate["max_bps"]) if bitrate.get("max_bps") is not None else None
        ),
        bitrate_penalty=int(bitrate.get("penalty", 0)),
        audio_required=bool(audio.get("required", False)),
        audio_codec_allowed=tuple(str(v).lower() for v in (audio.get("allowed_codecs") or [])),
        audio_penalty=int(audio.get("penalty", 0)),
        sample_rate_allowed=tuple(int(v) for v in (sample_rate.get("allowed") or [])),
        sample_rate_penalty=int(sample_rate.get("penalty", 0)),
        channels_allowed=tuple(int(v) for v in (channels.get("allowed") or [])),
        channels_penalty=int(channels.get("penalty", 0)),
        rotation_allowed=tuple(int(v) for v in (rotation.get("allowed") or [])),
        rotation_penalty=int(rotation.get("penalty", 0)),
        moov_required_position=str(moov_atom.get("required_position", "start")),
        moov_applicable_extensions=tuple(
            str(v).lower() for v in (moov_atom.get("applicable_extensions") or [])
        ),
        moov_penalty=int(moov_atom.get("penalty", 0)),
        vertical_required=bool(vertical.get("required", False)),
        vertical_penalty=int(vertical.get("penalty", 0)),
    )


def load_validator_config(config_path: str | Path | None = None) -> ValidatorConfig:
    path = Path(config_path) if config_path else default_validator_config_path()

    if not path.exists():
        raise VideoValidatorConfigError(f"Video validator config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise VideoValidatorConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise VideoValidatorConfigError(f"Video validator config is empty or invalid: {path}")

    media_inspector_section = raw.get("media_inspector") or {}
    scoring_section = raw.get("scoring") or {}
    profiles_section = raw.get("profiles") or {}

    profiles = {
        str(name): _build_profile_config(str(name), raw_profile or {})
        for name, raw_profile in profiles_section.items()
    }

    return ValidatorConfig(
        media_inspector_config_path=media_inspector_section.get("config_path"),
        starting_score=int(scoring_section.get("starting_score", 100)),
        minimum_score=int(scoring_section.get("minimum_score", 0)),
        profiles=profiles,
    )


# ---------------------------------------------------------------------------
# Moov atom detection (raw ISO-BMFF box walk — not ffprobe)
# ---------------------------------------------------------------------------


def detect_moov_atom_position(path: str | Path, *, max_boxes: int = 64) -> str:
    """
    Reads top-level ISO-BMFF ("MP4-family") box headers to determine
    whether the 'moov' (metadata) box appears before or after the 'mdat'
    (media data) box. Read-only, bounded (at most max_boxes header reads,
    each 8-16 bytes), never raises. Returns "start", "end", or "unknown".

    This is deliberately NOT ffprobe-based — ffprobe's
    -show_format/-show_streams/-show_chapters JSON never reports moov
    atom position, so there is nothing to reuse from Media Inspector for
    this one signal.
    """
    resolved = Path(path)

    try:
        with resolved.open("rb") as file:
            offset = 0

            for _ in range(max_boxes):
                file.seek(offset)
                header = file.read(8)

                if len(header) < 8:
                    return "unknown"

                size = int.from_bytes(header[0:4], "big")
                box_type = header[4:8].decode("ascii", errors="replace")

                if box_type == "moov":
                    return "start"

                if box_type == "mdat":
                    return "end"

                if size == 1:
                    extended = file.read(8)
                    if len(extended) < 8:
                        return "unknown"
                    size = int.from_bytes(extended, "big")
                elif size == 0:
                    # Box extends to EOF and is neither moov nor mdat —
                    # nothing more to walk.
                    return "unknown"

                if size < 8:
                    return "unknown"

                offset += size

            return "unknown"

    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuleResult:
    rule: str
    severity: str
    message: str
    penalty: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationResult:
    path: str
    profile: str
    passed: bool
    score: int
    failed_checks: list[RuleResult]
    warnings: list[RuleResult]
    checks: list[RuleResult]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "profile": self.profile,
            "passed": self.passed,
            "score": self.score,
            "failed_checks": [c.to_dict() for c in self.failed_checks],
            "warnings": [c.to_dict() for c in self.warnings],
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Individual rules — each (MediaInfo, ProfileConfig) -> RuleResult | None
# (None means "not applicable to this file", e.g. moov_atom on a .webm)
# ---------------------------------------------------------------------------


def _rule_duration(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    duration = media.duration_seconds

    if duration is None:
        return RuleResult("duration", SEVERITY_FAIL, "Duration is unknown.", profile.duration_penalty)

    if duration < profile.duration_min_seconds or duration > profile.duration_max_seconds:
        return RuleResult(
            "duration",
            SEVERITY_FAIL,
            f"Duration {duration:g}s is outside the allowed range "
            f"[{profile.duration_min_seconds:g}, {profile.duration_max_seconds:g}]s.",
            profile.duration_penalty,
        )

    return RuleResult("duration", SEVERITY_PASS, f"Duration {duration:g}s is within range.")


def _rule_resolution(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    video = media.primary_video

    if video is None or video.width is None or video.height is None:
        return RuleResult(
            "resolution", SEVERITY_FAIL, "No video stream/resolution found.", profile.resolution_penalty
        )

    if video.width < profile.resolution_min_width or video.height < profile.resolution_min_height:
        return RuleResult(
            "resolution",
            SEVERITY_FAIL,
            f"Resolution {video.width}x{video.height} is below the minimum "
            f"{profile.resolution_min_width}x{profile.resolution_min_height}.",
            profile.resolution_penalty,
        )

    return RuleResult(
        "resolution", SEVERITY_PASS, f"Resolution {video.width}x{video.height} meets the minimum."
    )


def _rule_aspect_ratio(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    ratio = media.aspect_ratio_float

    if ratio is None:
        return RuleResult(
            "aspect_ratio", SEVERITY_FAIL, "Aspect ratio could not be determined.", profile.aspect_ratio_penalty
        )

    if abs(ratio - profile.aspect_ratio_target) > profile.aspect_ratio_tolerance:
        return RuleResult(
            "aspect_ratio",
            SEVERITY_FAIL,
            f"Aspect ratio {ratio:.4f} is outside tolerance of target "
            f"{profile.aspect_ratio_target:.4f} (±{profile.aspect_ratio_tolerance}).",
            profile.aspect_ratio_penalty,
        )

    return RuleResult("aspect_ratio", SEVERITY_PASS, f"Aspect ratio {ratio:.4f} matches target.")


def _rule_fps(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    fps = media.fps

    if fps is None:
        return RuleResult("fps", SEVERITY_FAIL, "FPS could not be determined.", profile.fps_penalty)

    if not profile.fps_allowed:
        return RuleResult("fps", SEVERITY_PASS, f"FPS {fps:g} (no allowed-list configured).")

    if any(abs(fps - allowed) <= profile.fps_tolerance for allowed in profile.fps_allowed):
        return RuleResult("fps", SEVERITY_PASS, f"FPS {fps:g} is an allowed frame rate.")

    return RuleResult(
        "fps",
        SEVERITY_FAIL,
        f"FPS {fps:g} does not match any allowed frame rate {profile.fps_allowed}.",
        profile.fps_penalty,
    )


def _rule_video_codec(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    video = media.primary_video
    codec = (video.codec_name or "").lower() if video is not None else ""

    if not profile.video_codec_allowed:
        return RuleResult("video_codec", SEVERITY_PASS, f"Video codec {codec!r} (no allowed-list configured).")

    if codec in profile.video_codec_allowed:
        return RuleResult("video_codec", SEVERITY_PASS, f"Video codec {codec!r} is allowed.")

    return RuleResult(
        "video_codec",
        SEVERITY_FAIL,
        f"Video codec {codec!r} is not in the allowed list {profile.video_codec_allowed}.",
        profile.video_codec_penalty,
    )


def _rule_container(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    format_name = (media.format_name or "").lower()

    if not profile.container_allowed_substrings:
        return RuleResult("container", SEVERITY_PASS, f"Container {format_name!r} (no allowed-list configured).")

    if any(substring in format_name for substring in profile.container_allowed_substrings):
        return RuleResult("container", SEVERITY_PASS, f"Container {format_name!r} is allowed.")

    return RuleResult(
        "container",
        SEVERITY_FAIL,
        f"Container {format_name!r} does not contain any allowed substring "
        f"{profile.container_allowed_substrings}.",
        profile.container_penalty,
    )


def _rule_bitrate(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    bitrate = media.overall_bitrate

    if bitrate is None:
        return RuleResult(
            "bitrate", SEVERITY_WARNING, "Overall bitrate is unknown.", profile.bitrate_penalty
        )

    min_bps = profile.bitrate_min_bps
    max_bps = profile.bitrate_max_bps

    if (min_bps is not None and bitrate < min_bps) or (max_bps is not None and bitrate > max_bps):
        return RuleResult(
            "bitrate",
            SEVERITY_WARNING,
            f"Bitrate {bitrate} bps is outside the recommended range "
            f"[{min_bps}, {max_bps}] bps.",
            profile.bitrate_penalty,
        )

    return RuleResult("bitrate", SEVERITY_PASS, f"Bitrate {bitrate} bps is within the recommended range.")


def _rule_audio(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    if not media.has_audio:
        if profile.audio_required:
            return RuleResult("audio", SEVERITY_FAIL, "Missing audio stream.", profile.audio_penalty)
        return RuleResult("audio", SEVERITY_PASS, "No audio stream (not required by this profile).")

    audio = media.primary_audio
    codec = (audio.codec_name or "").lower() if audio is not None else ""

    if not profile.audio_codec_allowed:
        return RuleResult("audio", SEVERITY_PASS, f"Audio codec {codec!r} (no allowed-list configured).")

    if codec in profile.audio_codec_allowed:
        return RuleResult("audio", SEVERITY_PASS, f"Audio codec {codec!r} is allowed.")

    return RuleResult(
        "audio",
        SEVERITY_FAIL,
        f"Audio codec {codec!r} is not in the allowed list {profile.audio_codec_allowed}.",
        profile.audio_penalty,
    )


def _rule_audio_sample_rate(media: MediaInfo, profile: ProfileConfig) -> RuleResult | None:
    if not media.has_audio:
        return None  # covered by the audio rule

    audio = media.primary_audio
    sample_rate = audio.sample_rate if audio is not None else None

    if sample_rate is None:
        return RuleResult("audio_sample_rate", SEVERITY_WARNING, "Sample rate is unknown.", profile.sample_rate_penalty)

    if not profile.sample_rate_allowed or sample_rate in profile.sample_rate_allowed:
        return RuleResult("audio_sample_rate", SEVERITY_PASS, f"Sample rate {sample_rate}Hz is allowed.")

    return RuleResult(
        "audio_sample_rate",
        SEVERITY_WARNING,
        f"Sample rate {sample_rate}Hz is not in the allowed list {profile.sample_rate_allowed}.",
        profile.sample_rate_penalty,
    )


def _rule_audio_channels(media: MediaInfo, profile: ProfileConfig) -> RuleResult | None:
    if not media.has_audio:
        return None  # covered by the audio rule

    audio = media.primary_audio
    channels = audio.channels if audio is not None else None

    if channels is None:
        return RuleResult("audio_channels", SEVERITY_WARNING, "Channel count is unknown.", profile.channels_penalty)

    if not profile.channels_allowed or channels in profile.channels_allowed:
        return RuleResult("audio_channels", SEVERITY_PASS, f"Channel count {channels} is allowed.")

    return RuleResult(
        "audio_channels",
        SEVERITY_WARNING,
        f"Channel count {channels} is not in the allowed list {profile.channels_allowed}.",
        profile.channels_penalty,
    )


def _rule_rotation(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    video = media.primary_video
    rotation = video.rotation_degrees if video is not None else None
    rotation_value = rotation if rotation is not None else 0

    if not profile.rotation_allowed or rotation_value in profile.rotation_allowed:
        return RuleResult("rotation", SEVERITY_PASS, f"Rotation {rotation_value} degrees is allowed.")

    return RuleResult(
        "rotation",
        SEVERITY_WARNING,
        f"Rotation {rotation_value} degrees is not in the allowed list "
        f"{profile.rotation_allowed}; verify it displays correctly.",
        profile.rotation_penalty,
    )


def _rule_moov_atom(media: MediaInfo, profile: ProfileConfig) -> RuleResult | None:
    if media.extension not in profile.moov_applicable_extensions:
        return None  # not an ISO-BMFF/QuickTime container

    position = detect_moov_atom_position(media.path)

    if position == "unknown":
        return RuleResult(
            "moov_atom", SEVERITY_WARNING, "Could not determine moov atom position.", profile.moov_penalty
        )

    if position == profile.moov_required_position:
        return RuleResult("moov_atom", SEVERITY_PASS, f"moov atom is at the {position}.")

    return RuleResult(
        "moov_atom",
        SEVERITY_WARNING,
        f"moov atom is at the {position}, not the recommended "
        f"{profile.moov_required_position} (consider re-muxing with faststart).",
        profile.moov_penalty,
    )


def _rule_vertical_orientation(media: MediaInfo, profile: ProfileConfig) -> RuleResult:
    if not profile.vertical_required:
        return RuleResult("vertical_orientation", SEVERITY_PASS, "Vertical orientation not required.")

    if media.is_vertical:
        return RuleResult("vertical_orientation", SEVERITY_PASS, "Media is vertical.")

    return RuleResult(
        "vertical_orientation",
        SEVERITY_FAIL,
        "Media is not vertical, but this profile requires vertical orientation.",
        profile.vertical_penalty,
    )


_RULES = (
    _rule_duration,
    _rule_resolution,
    _rule_aspect_ratio,
    _rule_fps,
    _rule_video_codec,
    _rule_container,
    _rule_bitrate,
    _rule_audio,
    _rule_audio_sample_rate,
    _rule_audio_channels,
    _rule_rotation,
    _rule_moov_atom,
    _rule_vertical_orientation,
)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def validate_media(media: MediaInfo, profile: ProfileConfig, *, starting_score: int = 100, minimum_score: int = 0) -> ValidationResult:
    """
    Pure: runs every applicable rule against an already-built MediaInfo.
    Never touches ffprobe, never opens a browser, never mutates media —
    the only I/O here is detect_moov_atom_position()'s bounded, read-only
    header read for the one rule that needs it.
    """
    checks: list[RuleResult] = []

    for rule_function in _RULES:
        result = rule_function(media, profile)
        if result is not None:
            checks.append(result)

    failed_checks = [c for c in checks if c.severity == SEVERITY_FAIL]
    warnings = [c for c in checks if c.severity == SEVERITY_WARNING]
    passed_count = sum(1 for c in checks if c.severity == SEVERITY_PASS)

    score = starting_score - sum(c.penalty for c in checks if c.severity != SEVERITY_PASS)
    score = max(score, minimum_score)

    passed = len(failed_checks) == 0

    readiness = "READY" if passed else "NOT READY"
    summary = (
        f"{passed_count}/{len(checks)} checks passed, {len(warnings)} warning(s), "
        f"{len(failed_checks)} failure(s) — {readiness} for {profile.name}"
    )

    return ValidationResult(
        path=media.path,
        profile=profile.name,
        passed=passed,
        score=score,
        failed_checks=failed_checks,
        warnings=warnings,
        checks=checks,
        summary=summary,
    )


def validate_file(
    path: str | Path,
    config: ValidatorConfig,
    profile_name: str = DEFAULT_PROFILE_NAME,
    *,
    runner: SubprocessRunner = default_runner,
    media_inspector_config: InspectorConfig | None = None,
) -> ValidationResult:
    """
    Validate a file on disk: inspect it via Media Inspector
    (media_inspector.inspect_file — never a re-implementation), then
    score the resulting MediaInfo against the named profile.
    """
    if profile_name not in config.profiles:
        raise UnknownProfileError(
            f"Unknown profile {profile_name!r}. Configured profiles: {sorted(config.profiles)}"
        )

    inspector_config = media_inspector_config or load_inspector_config(config.media_inspector_config_path)
    media = inspect_file(path, inspector_config, runner=runner)

    return validate_media(
        media,
        config.profiles[profile_name],
        starting_score=config.starting_score,
        minimum_score=config.minimum_score,
    )


# ---------------------------------------------------------------------------
# JSON envelope / report writing
# ---------------------------------------------------------------------------


def build_result_envelope(result: ValidationResult) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "validated_at": _now_iso(),
        "result": result.to_dict(),
    }


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------


def format_human_report(result: ValidationResult) -> str:
    lines: list[str] = []
    lines.append(f"File:    {result.path}")
    lines.append(f"Profile: {result.profile}")
    lines.append(f"Passed:  {result.passed}")
    lines.append(f"Score:   {result.score}")
    lines.append("")
    lines.append("Checks:")
    for check in result.checks:
        lines.append(f"  [{check.severity.upper():7s}] {check.rule}: {check.message}")
    lines.append("")
    lines.append(result.summary)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Video Validator (Phase 11A.2). Validates whether a "
            "rendered video is suitable for publishing under a named "
            "profile. Never renders, publishes, or modifies media."
        )
    )

    parser.add_argument("file", help="Path to the video file to validate.")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable JSON envelope.")
    parser.add_argument("--output", default=None, help="Write the JSON envelope to this path.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --output report file.")
    parser.add_argument(
        "--profile", default=DEFAULT_PROFILE_NAME, help=f"Validation profile (default: {DEFAULT_PROFILE_NAME})."
    )
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/validator.yaml.")
    parser.add_argument(
        "--media-inspector-config", default=None, help="Path to an alternate config/media/inspector.yaml."
    )

    return parser.parse_args(argv)


def _run(arguments: argparse.Namespace, *, runner: SubprocessRunner = default_runner) -> int:
    try:
        config = load_validator_config(arguments.config)
        media_inspector_config = (
            load_inspector_config(arguments.media_inspector_config)
            if arguments.media_inspector_config
            else None
        )
        result = validate_file(
            arguments.file,
            config,
            arguments.profile,
            runner=runner,
            media_inspector_config=media_inspector_config,
        )
    except (VideoValidatorError, MediaInspectorError) as exc:
        print(f"[VideoValidator] {exc}")
        return 1

    envelope = build_result_envelope(result)

    if arguments.output:
        try:
            written_path = write_report(
                envelope, arguments.output, force=arguments.force, input_path=result.path
            )
        except MediaInspectorError as exc:
            print(f"[VideoValidator] {exc}")
            return 1
        print(f"Report written: {written_path}")

    if arguments.json:
        print(json.dumps(envelope, ensure_ascii=False, indent=2))
    else:
        print(format_human_report(result))

    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    exit_code = _run(arguments)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
