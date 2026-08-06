from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import (
    media_inspector,
    music_mixer,
    overlay_asset_resolver,
    overlay_plan_engine,
    overlay_render_engine,
    renderer_plan_engine,
    subtitle_render_engine,
    timeline_engine,
    video_engine,
)

# Phase 11F — Renderer Execution Engine. The first module in this
# pipeline permitted to actually invoke ffmpeg/ffprobe -- but it does
# so exclusively by orchestrating already-built, already-tested engines
# (video_engine.py, music_mixer.py, media_inspector.py,
# subtitle_render_engine.py, overlay_render_engine.py), never by
# constructing a raw ffmpeg command itself. It reads an
# already-validated renderer_plan.json (+ the timeline.json and
# overlay_plan.json it references) and executes the "video"/"music"
# passes for real, plus (Phase 11F.1) "subtitle" passes whose renderer
# hint is "ass" or "drawtext" -- delegated to subtitle_render_engine.py
# -- and (Phase 11F.6) "overlay" passes whose renderer hint is
# "overlay_png" or "overlay_alpha" -- delegated to
# overlay_render_engine.py, after this module resolves the plan's real
# OverlayAssetManifest via overlay_asset_resolver.py (the same
# "prepare real inputs before any pass runs" role _probe_video_assets()
# already plays for video). A subtitle/overlay pass with any other
# hint is detected but never executed here: by default this fails
# loudly rather than silently producing a video missing its planned
# subtitles/overlays; --allow-partial-execution is required to proceed
# anyway, and even then those passes are only ever recorded as
# "skipped", never faked. --dry-run is the implicit CLI default --
# ffmpeg is only actually invoked when --execute is explicitly given.
# This module never publishes, uploads, downloads fonts/assets, or
# modifies the source renderer_plan.json/timeline.json/overlay_plan.json
# files it reads.

DEFAULT_RENDERER_EXECUTION_CONFIG_RELATIVE_PATH = Path("config") / "video" / "renderer_execution.yaml"


def _runtime_root() -> Path:
    """
    renderer_execution_engine.py location:
    10_apps/claude_runtime/src/renderer_execution_engine.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RendererExecutionEngineError(RuntimeError):
    """Base error for the Phase 11F Renderer Execution Engine."""


class RendererExecutionConfigError(RendererExecutionEngineError):
    """Raised when config/video/renderer_execution.yaml is missing or invalid."""


class RendererExecutionPlanLoadError(RendererExecutionEngineError):
    """Raised when --renderer-plan cannot be loaded or fails validation."""


class RendererExecutionTimelineLoadError(RendererExecutionEngineError):
    """Raised when --timeline cannot be loaded or fails validation."""


class RendererExecutionPlanMismatchError(RendererExecutionEngineError):
    """Raised when the renderer plan's timeline_id does not match the
    target Timeline's timeline_id."""


class RendererExecutionUnsupportedPassError(RendererExecutionEngineError):
    """Raised when the plan contains a subtitle/overlay pass and
    --allow-partial-execution was not given."""


class RendererExecutionProbeError(RendererExecutionEngineError):
    """Raised when media_inspector.inspect_file() fails for a video asset."""


class RendererExecutionVideoError(RendererExecutionEngineError):
    """Raised when video_engine.py planning/execution/verification fails."""


class RendererExecutionMusicError(RendererExecutionEngineError):
    """Raised when music_mixer.py execution fails."""


class RendererExecutionSubtitleError(RendererExecutionEngineError):
    """Raised when subtitle_render_engine.py planning/execution fails,
    or a subtitle pass has no resolvable subtitle asset (missing
    subtitle_file asset for an ass hint, or an empty overlay set for a
    drawtext hint)."""


class RendererExecutionOverlayError(RendererExecutionEngineError):
    """Raised when overlay_asset_resolver.py asset resolution or
    overlay_render_engine.py planning/execution fails for an overlay
    pass."""


class RendererExecutionOutputExistsError(RendererExecutionEngineError):
    """Raised when --output already exists and --force was not given."""


class UnsafeRendererExecutionOutputError(RendererExecutionEngineError):
    """Raised when --output is a directory, or resolves to the same path
    as --renderer-plan or --timeline."""


class RendererExecutionLogError(RendererExecutionEngineError):
    """Raised when the execution log cannot be written."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RendererExecutionConfig:
    schema_version: str = "1.0"
    dry_run_default: bool = True
    allow_partial_execution_default: bool = False
    intermediate_filename_suffix: str = "_video_only"
    subtitle_intermediate_filename_suffix: str = "_subtitled"
    overlay_intermediate_filename_suffix: str = "_overlaid"
    log_filename_suffix: str = "_renderer_execution_log.json"
    overwrite_requires_force: bool = True
    atomic_write: bool = True


def default_renderer_execution_config_path() -> Path:
    return _runtime_root() / DEFAULT_RENDERER_EXECUTION_CONFIG_RELATIVE_PATH


def load_renderer_execution_config(config_path: str | Path | None = None) -> RendererExecutionConfig:
    """Load config/video/renderer_execution.yaml (or an alternate path)
    into a RendererExecutionConfig. Owns NO ffmpeg/ffprobe binary
    settings -- those remain owned by video_engine.py/media_inspector.py/
    music_mixer.py's own configs. Raises RendererExecutionConfigError if
    the file is missing or invalid."""
    path = Path(config_path) if config_path else default_renderer_execution_config_path()

    if not path.exists():
        raise RendererExecutionConfigError(f"Renderer execution config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise RendererExecutionConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise RendererExecutionConfigError(f"Renderer execution config is empty or invalid: {path}")

    execution_section = raw.get("execution") or {}
    output_section = raw.get("output") or {}

    return RendererExecutionConfig(
        schema_version=str(execution_section.get("schema_version", "1.0")),
        dry_run_default=bool(execution_section.get("dry_run_default", True)),
        allow_partial_execution_default=bool(execution_section.get("allow_partial_execution_default", False)),
        intermediate_filename_suffix=str(execution_section.get("intermediate_filename_suffix", "_video_only")),
        subtitle_intermediate_filename_suffix=str(
            execution_section.get("subtitle_intermediate_filename_suffix", "_subtitled")
        ),
        overlay_intermediate_filename_suffix=str(
            execution_section.get("overlay_intermediate_filename_suffix", "_overlaid")
        ),
        log_filename_suffix=str(execution_section.get("log_filename_suffix", "_renderer_execution_log.json")),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        atomic_write=bool(output_section.get("atomic_write", True)),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PassExecutionResult:
    pass_id: str = ""
    pass_type: str = ""
    status: str = ""  # executed | skipped | verified | dry_run
    command: list[str] | None = None
    output_path: str | None = None
    duration_seconds: float | None = None
    return_code: int | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    # Set only for subtitle passes: path to subtitle_render_engine.py's
    # own written diagnostic log for this pass, if any.
    diagnostic_log: str | None = None


@dataclass(slots=True)
class RendererExecutionResult:
    renderer_plan_id: str = ""
    timeline_id: str = ""
    dry_run: bool = True
    started_at: str = ""
    finished_at: str = ""
    passes: list[PassExecutionResult] = field(default_factory=list)
    final_output_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    result: str = ""  # success | failed | partial | dry_run


def renderer_execution_result_to_dict(result: RendererExecutionResult) -> dict[str, Any]:
    return {
        "renderer_plan_id": result.renderer_plan_id,
        "timeline_id": result.timeline_id,
        "dry_run": result.dry_run,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "passes": [asdict(p) for p in result.passes],
        "final_output_path": result.final_output_path,
        "warnings": result.warnings,
        "errors": result.errors,
        "result": result.result,
    }


def _save_execution_log(result: RendererExecutionResult, path: Path) -> Path:
    """Atomic temp-file + Path.replace() write, same convention as every
    prior save_*() function in this repo."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(renderer_execution_result_to_dict(result), indent=2, sort_keys=True, ensure_ascii=False)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise RendererExecutionLogError(f"Failed to write execution log {path}: {exc}") from exc
    return path


# ---------------------------------------------------------------------------
# Video pass — real ffprobe via media_inspector.py, real ffmpeg command
# construction/execution via video_engine.py. No new ffmpeg logic here.
# ---------------------------------------------------------------------------


def _video_input_from_media_info(source_path: str, info: media_inspector.MediaInfo) -> video_engine.VideoInput:
    """Adapter mirroring reel_builder._video_input_from_probe()'s own
    shape, sourced from media_inspector.MediaInfo instead of Reel
    Builder's own ffprobe wrapper."""
    video_stream = info.primary_video
    return video_engine.VideoInput(
        path=Path(source_path),
        video=video_engine.VideoStreamSpec(
            width=video_stream.width if video_stream else None,
            height=video_stream.height if video_stream else None,
            fps=video_stream.fps if video_stream else None,
            codec_name=video_stream.codec_name if video_stream else None,
        ),
        audio=video_engine.AudioStreamSpec(present=info.has_audio),
        duration_seconds=info.duration_seconds,
    )


def _probe_video_assets(
    plan: renderer_plan_engine.RendererPlan,
    inspector_config: media_inspector.InspectorConfig,
    *,
    runner: Any,
) -> dict[str, media_inspector.MediaInfo]:
    asset_by_id = {asset.asset_id: asset for asset in plan.assets}
    video_asset_ids = {
        asset_id for track in plan.video_tracks for asset_id in track.asset_ids
    }

    runner_kwargs = {"runner": runner} if runner is not None else {}

    probes: dict[str, media_inspector.MediaInfo] = {}
    for asset_id in sorted(video_asset_ids):
        asset = asset_by_id[asset_id]
        try:
            probes[asset_id] = media_inspector.inspect_file(asset.source_path, inspector_config, **runner_kwargs)
        except media_inspector.MediaInspectorError as exc:
            raise RendererExecutionProbeError(
                f"Failed to probe video asset {asset_id} ({asset.source_path}): {exc}"
            ) from exc

    return probes


def _execute_video_pass(
    pass_: renderer_plan_engine.RendererPass,
    plan: renderer_plan_engine.RendererPlan,
    probes: dict[str, media_inspector.MediaInfo],
    engine_config: video_engine.VideoEngineConfig,
    *,
    output_path: Path,
    dry_run: bool,
    force: bool,
    runner: Any,
) -> PassExecutionResult:
    asset_by_id = {asset.asset_id: asset for asset in plan.assets}
    video_track = plan.video_tracks[0]

    inputs = [
        _video_input_from_media_info(asset_by_id[asset_id].source_path, probes[asset_id])
        for asset_id in video_track.asset_ids
    ]

    engine = (
        video_engine.VideoEngine(engine_config, runner=runner) if runner is not None else video_engine.VideoEngine(engine_config)
    )
    output_spec = video_engine.VideoOutputSpec(path=output_path)

    try:
        concat_plan = engine.build_concat_plan(inputs, output_spec)
    except video_engine.VideoEngineError as exc:
        raise RendererExecutionVideoError(f"Failed to build video concat plan: {exc}") from exc

    if concat_plan.plan_type == video_engine.PLAN_REJECTED:
        raise RendererExecutionVideoError(f"Video pass plan was rejected: {concat_plan.reasons}")

    if dry_run:
        if concat_plan.plan_type == video_engine.PLAN_LOSSLESS_COPY:
            manifest_preview_path = (
                output_path.parent / engine_config.temporary_directory_name
                / f"{output_path.stem}_concat_manifest.txt"
            )
            command = engine.build_lossless_concat_command(concat_plan, manifest_preview_path, force=force)
        else:
            command = engine.build_normalized_concat_command(concat_plan, force=force)
        return PassExecutionResult(
            pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="dry_run",
            command=command, output_path=str(output_path),
        )

    try:
        result = engine.execute_plan(concat_plan, force=force)
    except video_engine.VideoEngineError as exc:
        raise RendererExecutionVideoError(f"Video pass execution failed: {exc}") from exc

    return PassExecutionResult(
        pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="executed",
        command=result.command, output_path=result.output_file,
        duration_seconds=result.duration_seconds, return_code=result.return_code,
        warnings=list(result.warnings), error=result.error,
    )


# ---------------------------------------------------------------------------
# Music pass — real execution via music_mixer.py. No new ffmpeg logic.
# ---------------------------------------------------------------------------


def _execute_music_pass(
    pass_: renderer_plan_engine.RendererPass,
    plan: renderer_plan_engine.RendererPlan,
    *,
    video_pass_output: Path,
    final_output_path: Path,
    music_config: "music_mixer.MusicMixerConfig",
    dry_run: bool,
    force: bool,
    runner: Any,
) -> PassExecutionResult:
    asset_by_id = {asset.asset_id: asset for asset in plan.assets}
    music_track = plan.music_tracks[0]
    music_asset = asset_by_id[music_track.asset_id]

    request = music_mixer.MusicMixRequest(
        video_path=video_pass_output,
        music_path=Path(music_asset.source_path),
        output_path=final_output_path,
        force=force,
        music_volume=music_track.volume,
        fade_in_seconds=music_track.fade_in_seconds,
        fade_out_seconds=music_track.fade_out_seconds,
        music_mode=music_track.mode,
        ducking_mode=music_track.ducking_mode,
    )

    if dry_run:
        # The real ffmpeg command depends on probing the video pass's
        # real output, which does not exist yet in dry-run -- only the
        # request parameters can be shown honestly here.
        return PassExecutionResult(
            pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="dry_run",
            command=None, output_path=str(final_output_path),
            warnings=[
                "music pass command depends on the video pass's real output; "
                "not resolved in dry-run"
            ],
        )

    runner_kwargs = {"runner": runner} if runner is not None else {}
    try:
        result = music_mixer.mix_music(request, music_config, **runner_kwargs)
    except (music_mixer.MusicMixerError, media_inspector.MediaInspectorError) as exc:
        # music_mixer.mix_music() lets Media Inspector's own exceptions
        # propagate as-is rather than wrapping them (the same contract
        # reel_builder.py's build_reel() already accounts for) -- both
        # exception families are caught here.
        raise RendererExecutionMusicError(f"Music pass execution failed: {exc}") from exc

    return PassExecutionResult(
        pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="executed",
        command=result.command, output_path=result.output_path,
        duration_seconds=result.duration_seconds, return_code=result.return_code,
        warnings=list(result.warnings), error=result.error,
    )


# ---------------------------------------------------------------------------
# Subtitle pass — real execution via subtitle_render_engine.py, the
# first phase permitted to burn subtitles into video (Phase 11F.1).
# Only "ass"/"drawtext" renderer hints are executable here; any other
# hint is treated as unsupported by the caller (see the pre-flight
# check and pass-loop dispatch in execute_renderer_plan()).
# ---------------------------------------------------------------------------


def _find_hint_for_pass(
    plan: renderer_plan_engine.RendererPlan, pass_id: str
) -> renderer_plan_engine.RendererHint | None:
    return next((hint for hint in plan.hints if hint.target_id == pass_id), None)


def _planned_overlay_to_drawtext_cue(
    overlay: overlay_plan_engine.PlannedOverlay,
) -> subtitle_render_engine.DrawTextCue:
    """Adapter: an Overlay Plan Engine PlannedOverlay -> a Subtitle
    Render Engine DrawTextCue. Read-only over the already-planned
    overlay -- never recalculates position/style, only reshapes it."""
    style = overlay.style_snapshot
    position = overlay.position
    return subtitle_render_engine.DrawTextCue(
        text=overlay.content,
        start_seconds=overlay.timeline_start_seconds,
        end_seconds=overlay.timeline_end_seconds,
        font_family=style.font_family,
        font_size=style.font_size if style.font_size is not None else 48.0,
        font_color=style.primary_color or "#FFFFFF",
        outline_color=style.outline_color or "#000000",
        outline_width=style.outline_width if style.outline_width is not None else 2.0,
        x=position.x,
        y=position.y,
        anchor=position.anchor if position.x is None or position.y is None else None,
        opacity=style.opacity,
    )


def _execute_subtitle_pass(
    pass_: renderer_plan_engine.RendererPass,
    plan: renderer_plan_engine.RendererPlan,
    hint: renderer_plan_engine.RendererHint,
    overlay_plan: overlay_plan_engine.OverlayPlan,
    subtitle_render_config: subtitle_render_engine.SubtitleRenderConfig,
    *,
    video_input: Path | None,
    output_path: Path,
    canvas_width: int,
    canvas_height: int,
    dry_run: bool,
    force: bool,
    runner: Any,
) -> PassExecutionResult:
    if dry_run and (video_input is None or not video_input.exists()):
        # The real subtitle command depends on reading the preceding
        # pass's real output, which does not exist yet in dry-run --
        # mirrors _execute_music_pass()'s own dry-run honesty limit:
        # only the request parameters can be shown, not the resolved
        # ffmpeg command.
        return PassExecutionResult(
            pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="dry_run",
            command=None, output_path=str(output_path),
            warnings=[
                "subtitle pass command depends on a preceding pass's real output; "
                "not resolved in dry-run"
            ],
        )

    if video_input is None:
        raise RendererExecutionSubtitleError(
            f"Pass {pass_.pass_id} has no video input available (no preceding video/subtitle pass output)"
        )

    if not plan.subtitle_tracks:
        raise RendererExecutionSubtitleError(f"Pass {pass_.pass_id} is a subtitle pass but plan has no subtitle_tracks")
    subtitle_track = plan.subtitle_tracks[0]

    if hint.hint_type == renderer_plan_engine.RendererHintType.ASS:
        asset_by_id = {asset.asset_id: asset for asset in plan.assets}
        ass_asset = next(
            (
                asset_by_id[asset_id]
                for asset_id in subtitle_track.asset_ids
                if asset_id in asset_by_id
                and asset_by_id[asset_id].asset_type == renderer_plan_engine.AssetType.SUBTITLE_FILE
            ),
            None,
        )
        if ass_asset is None:
            raise RendererExecutionSubtitleError(
                f"Pass {pass_.pass_id} has an 'ass' hint but no subtitle_file asset was found "
                "among the subtitle track's asset_ids"
            )
        request_kwargs: dict[str, Any] = {
            "mode": subtitle_render_engine.SubtitleRenderMode.ASS,
            "ass_path": ass_asset.source_path,
        }
    elif hint.hint_type == renderer_plan_engine.RendererHintType.DRAWTEXT:
        overlay_by_id = {overlay.overlay_id: overlay for overlay in overlay_plan.overlays}
        cues = [
            _planned_overlay_to_drawtext_cue(overlay_by_id[overlay_id])
            for overlay_id in subtitle_track.overlay_ids
            if overlay_id in overlay_by_id and overlay_by_id[overlay_id].enabled
        ]
        if not cues:
            raise RendererExecutionSubtitleError(
                f"Pass {pass_.pass_id} has a 'drawtext' hint but no enabled subtitle overlays were found"
            )
        request_kwargs = {"mode": subtitle_render_engine.SubtitleRenderMode.DRAWTEXT, "cues": cues}
    else:
        raise RendererExecutionSubtitleError(
            f"Pass {pass_.pass_id} has an unsupported subtitle hint {hint.hint_type!r}"
        )

    try:
        request = subtitle_render_engine.build_subtitle_render_request(
            video_path=video_input, output_path=output_path, force=force,
            canvas_width=canvas_width, canvas_height=canvas_height,
            config=subtitle_render_config, **request_kwargs,
        )
    except subtitle_render_engine.SubtitleRenderEngineError as exc:
        raise RendererExecutionSubtitleError(f"Failed to build subtitle render request: {exc}") from exc

    if dry_run:
        subtitle_plan = subtitle_render_engine.build_subtitle_render_plan(request, subtitle_render_config)
        return PassExecutionResult(
            pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="dry_run",
            command=subtitle_plan.command, output_path=str(output_path),
        )

    runner_kwargs = {"runner": runner} if runner is not None else {}
    try:
        subtitle_result = subtitle_render_engine.execute_subtitle_render_plan(
            request, subtitle_render_config, **runner_kwargs
        )
    except subtitle_render_engine.SubtitleRenderEngineError as exc:
        raise RendererExecutionSubtitleError(f"Subtitle pass execution failed: {exc}") from exc

    diagnostic_log = None
    if subtitle_render_config.diagnostics_write_log:
        diagnostic_log = str(
            output_path.with_name(f"{output_path.stem}{subtitle_render_config.diagnostics_log_filename_suffix}")
        )

    return PassExecutionResult(
        pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="executed",
        command=subtitle_result.command, output_path=subtitle_result.output_video,
        duration_seconds=subtitle_result.duration_seconds, return_code=subtitle_result.return_code,
        warnings=list(subtitle_result.warnings), error=subtitle_result.error,
        diagnostic_log=diagnostic_log,
    )


# ---------------------------------------------------------------------------
# Overlay pass — real execution via overlay_render_engine.py (Phase
# 11F.6), the first phase permitted to burn overlay compositing into
# video. Only "overlay_png"/"overlay_alpha" renderer hints are
# executable here; any other hint is treated as unsupported by the
# caller (see the pre-flight check and pass-loop dispatch in
# execute_renderer_plan()).
# ---------------------------------------------------------------------------


def _resolve_overlay_assets(
    plan: renderer_plan_engine.RendererPlan,
    overlay_plan: overlay_plan_engine.OverlayPlan,
    overlay_asset_config: overlay_asset_resolver.OverlayAssetResolverConfig,
    *,
    inspector: Any,
) -> overlay_asset_resolver.OverlayAssetManifest:
    """
    "Prepare inputs" for the overlay pass: resolves every overlay asset
    reference in the plan into a real, validated OverlayAssetManifest --
    the same orchestration-layer role _probe_video_assets() already
    plays for video assets. Raises RendererExecutionOverlayError on a
    hard resolution failure; a soft missing/invalid optional asset is
    governed entirely by overlay_asset_config's own policy (unchanged),
    never a second competing policy layer here.
    """
    try:
        references = overlay_asset_resolver.collect_overlay_asset_references(
            overlay_plan=overlay_plan, renderer_plan=plan, config=overlay_asset_config,
        )
        result = overlay_asset_resolver.resolve_overlay_assets(references, overlay_asset_config, inspector=inspector)
        if result.errors:
            raise RendererExecutionOverlayError(
                f"Overlay asset resolution failed: {'; '.join(result.errors)}"
            )
        manifest = overlay_asset_resolver.build_overlay_asset_manifest(
            result, references, overlay_asset_config, source_plan_ids=[overlay_plan.plan_id, plan.renderer_plan_id],
        )
    except overlay_asset_resolver.OverlayAssetResolverError as exc:
        raise RendererExecutionOverlayError(f"Failed to resolve overlay assets: {exc}") from exc

    return manifest


def _execute_overlay_pass(
    pass_: renderer_plan_engine.RendererPass,
    plan: renderer_plan_engine.RendererPlan,
    overlay_plan: overlay_plan_engine.OverlayPlan,
    overlay_asset_manifest: overlay_asset_resolver.OverlayAssetManifest,
    overlay_render_config: overlay_render_engine.OverlayRenderConfig,
    *,
    video_input: Path | None,
    output_path: Path,
    canvas_width: int,
    canvas_height: int,
    dry_run: bool,
    force: bool,
    runner: Any,
) -> PassExecutionResult:
    if dry_run and (video_input is None or not video_input.exists()):
        # The real overlay command depends on reading the preceding
        # pass's real output, which does not exist yet in dry-run --
        # mirrors _execute_subtitle_pass()'s own dry-run honesty limit:
        # only the request parameters can be shown, not the resolved
        # ffmpeg command.
        return PassExecutionResult(
            pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="dry_run",
            command=None, output_path=str(output_path),
            warnings=[
                "overlay pass command depends on a preceding pass's real output; "
                "not resolved in dry-run"
            ],
        )

    if video_input is None:
        raise RendererExecutionOverlayError(
            f"Pass {pass_.pass_id} has no video input available (no preceding video/subtitle pass output)"
        )

    try:
        request = overlay_render_engine.build_overlay_render_request(
            video_path=video_input, output_path=output_path, renderer_plan=plan, overlay_plan=overlay_plan,
            overlay_asset_manifest=overlay_asset_manifest, config=overlay_render_config, force=force,
            canvas_width=canvas_width, canvas_height=canvas_height,
        )
    except overlay_render_engine.OverlayRenderEngineError as exc:
        raise RendererExecutionOverlayError(f"Failed to build overlay render request: {exc}") from exc

    if dry_run:
        try:
            overlay_plan_result = overlay_render_engine.build_overlay_render_plan(request, overlay_render_config)
        except overlay_render_engine.OverlayRenderEngineError as exc:
            raise RendererExecutionOverlayError(f"Failed to build overlay render plan: {exc}") from exc
        return PassExecutionResult(
            pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="dry_run",
            command=overlay_plan_result.command, output_path=str(output_path),
        )

    runner_kwargs = {"runner": runner} if runner is not None else {}
    try:
        overlay_result = overlay_render_engine.execute_overlay_render_plan(
            request, overlay_render_config, **runner_kwargs
        )
    except overlay_render_engine.OverlayRenderEngineError as exc:
        raise RendererExecutionOverlayError(f"Overlay pass execution failed: {exc}") from exc

    return PassExecutionResult(
        pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="executed",
        command=overlay_result.command, output_path=overlay_result.output_video,
        duration_seconds=overlay_result.duration_seconds, return_code=overlay_result.return_code,
        warnings=list(overlay_result.warnings), error=overlay_result.error,
    )


# ---------------------------------------------------------------------------
# final_encode pass — verification only, no new ffmpeg call.
# ---------------------------------------------------------------------------


def _execute_final_encode_pass(
    pass_: renderer_plan_engine.RendererPass,
    *,
    final_output_path: Path,
    engine_config: video_engine.VideoEngineConfig,
    dry_run: bool,
) -> PassExecutionResult:
    if dry_run:
        return PassExecutionResult(
            pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="dry_run",
            output_path=str(final_output_path),
        )

    engine = video_engine.VideoEngine(engine_config)
    exists, size_bytes = engine.verify_output(final_output_path)
    if not exists:
        raise RendererExecutionVideoError(f"final_encode verification failed: output missing: {final_output_path}")
    if size_bytes == 0:
        raise RendererExecutionVideoError(f"final_encode verification failed: output is zero bytes: {final_output_path}")

    return PassExecutionResult(
        pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="verified",
        output_path=str(final_output_path), return_code=0,
    )


# ---------------------------------------------------------------------------
# Per-pass execution seam (Phase 11F.7) — the same prologue/dispatch
# logic execute_renderer_plan() itself uses, factored into reusable
# pieces so a caller needing per-pass control (e.g.
# end_to_end_render_pipeline.py's resume/retry logic) can drive passes
# one at a time without duplicating any ffmpeg/adapter logic. Never
# used to change execute_renderer_plan()'s own external behavior --
# its own full test suite is the proof of that.
# ---------------------------------------------------------------------------


def _pass_is_supported(pass_: renderer_plan_engine.RendererPass, plan: renderer_plan_engine.RendererPlan) -> bool:
    """Whether `pass_` can actually be executed by execute_single_pass():
    an overlay pass only when its hint is overlay_png/overlay_alpha, a
    subtitle pass only when its hint is ass/drawtext, every other pass
    type unconditionally. Pure policy check, never executes anything."""
    if pass_.pass_type == renderer_plan_engine.RenderPassType.OVERLAY:
        hint = _find_hint_for_pass(plan, pass_.pass_id)
        return hint is not None and hint.hint_type in (
            renderer_plan_engine.RendererHintType.OVERLAY_PNG,
            renderer_plan_engine.RendererHintType.OVERLAY_ALPHA,
        )
    if pass_.pass_type == renderer_plan_engine.RenderPassType.SUBTITLE:
        hint = _find_hint_for_pass(plan, pass_.pass_id)
        return hint is not None and hint.hint_type in (
            renderer_plan_engine.RendererHintType.ASS,
            renderer_plan_engine.RendererHintType.DRAWTEXT,
        )
    return True


def is_pass_supported(pass_: renderer_plan_engine.RendererPass, plan: renderer_plan_engine.RendererPlan) -> bool:
    """Public one-line wrapper around _pass_is_supported() -- lets a
    per-pass caller (e.g. end_to_end_render_pipeline.py) run the exact
    same up-front "will this pass actually execute" pre-flight check
    execute_renderer_plan() itself runs, before touching any media."""
    return _pass_is_supported(pass_, plan)


def _load_and_cross_check_plans(
    renderer_plan_path: Path, timeline_path: Path, overlay_plan_path: Path,
) -> tuple[renderer_plan_engine.RendererPlan, timeline_engine.Timeline, overlay_plan_engine.OverlayPlan]:
    """Loads+validates renderer_plan.json/timeline.json/overlay_plan.json
    and cross-checks their identities match -- the exact prologue both
    execute_renderer_plan() and prepare_execution_context() need, in
    the same order, raising the same errors."""
    try:
        plan = renderer_plan_engine.load_renderer_plan(renderer_plan_path)
    except renderer_plan_engine.RendererPlanEngineError as exc:
        raise RendererExecutionPlanLoadError(f"Failed to load renderer plan {renderer_plan_path}: {exc}") from exc

    renderer_plan_config = renderer_plan_engine.load_renderer_plan_config()
    plan_result = renderer_plan_engine.validate_renderer_plan(plan, renderer_plan_config)
    if not plan_result.passed:
        raise RendererExecutionPlanLoadError(
            f"Renderer plan {renderer_plan_path} failed validation: {'; '.join(plan_result.errors)}"
        )

    try:
        timeline = timeline_engine.load_timeline(timeline_path)
    except timeline_engine.TimelineEngineError as exc:
        raise RendererExecutionTimelineLoadError(f"Failed to load timeline {timeline_path}: {exc}") from exc

    timeline_result = timeline_engine.validate_timeline(timeline)
    if not timeline_result.passed:
        raise RendererExecutionTimelineLoadError(
            f"Timeline {timeline_path} failed validation: {'; '.join(timeline_result.errors)}"
        )

    if plan.timeline_id != timeline.timeline_id:
        raise RendererExecutionPlanMismatchError(
            f"Renderer plan timeline_id {plan.timeline_id!r} does not match timeline_id "
            f"{timeline.timeline_id!r}"
        )

    try:
        overlay_plan = overlay_plan_engine.load_overlay_plan(overlay_plan_path)
    except overlay_plan_engine.OverlayPlanEngineError as exc:
        raise RendererExecutionPlanLoadError(f"Failed to load overlay plan {overlay_plan_path}: {exc}") from exc

    overlay_plan_config = overlay_plan_engine.load_overlay_plan_config()
    overlay_plan_result = overlay_plan_engine.validate_overlay_plan(overlay_plan, overlay_plan_config, timeline=timeline)
    if not overlay_plan_result.passed:
        raise RendererExecutionPlanLoadError(
            f"Overlay plan {overlay_plan_path} failed validation: {'; '.join(overlay_plan_result.errors)}"
        )

    if plan.overlay_plan_id != overlay_plan.plan_id:
        raise RendererExecutionPlanMismatchError(
            f"Renderer plan overlay_plan_id {plan.overlay_plan_id!r} does not match overlay "
            f"plan_id {overlay_plan.plan_id!r}"
        )

    return plan, timeline, overlay_plan


@dataclass(slots=True)
class RendererExecutionContext:
    """Bundles everything a per-pass caller needs: the loaded+validated
    plan/timeline/overlay_plan (identity cross-checked), real video
    probes, the resolved OverlayAssetManifest (None when the plan has
    no supported overlay pass), and every engine config
    execute_single_pass() needs to dispatch a pass. Phase 11F.7's
    end_to_end_render_pipeline.py is the first caller that needs this
    outside execute_renderer_plan()'s own one-shot use."""

    plan: renderer_plan_engine.RendererPlan
    timeline: timeline_engine.Timeline
    overlay_plan: overlay_plan_engine.OverlayPlan
    probes: dict[str, media_inspector.MediaInfo]
    overlay_asset_manifest: overlay_asset_resolver.OverlayAssetManifest | None
    engine_config: video_engine.VideoEngineConfig
    inspector_config: media_inspector.InspectorConfig
    music_config: "music_mixer.MusicMixerConfig"
    subtitle_render_config: subtitle_render_engine.SubtitleRenderConfig
    overlay_render_config: overlay_render_engine.OverlayRenderConfig


def prepare_execution_context(
    renderer_plan_path: str | Path,
    timeline_path: str | Path,
    overlay_plan_path: str | Path,
    *,
    runner: Any = None,
    subtitle_render_config: subtitle_render_engine.SubtitleRenderConfig | None = None,
    overlay_render_config: overlay_render_engine.OverlayRenderConfig | None = None,
    overlay_asset_config: overlay_asset_resolver.OverlayAssetResolverConfig | None = None,
    overlay_asset_inspector: Any = None,
) -> RendererExecutionContext:
    """
    The load/validate/cross-check/probe/resolve-overlay-assets prologue
    execute_renderer_plan() itself runs, factored out so a per-pass
    caller can run it exactly once and then drive execute_single_pass()
    directly per pass. Raises the exact same errors
    execute_renderer_plan() already raises for each of these steps.
    """
    renderer_plan_path = Path(renderer_plan_path)
    timeline_path = Path(timeline_path)
    overlay_plan_path = Path(overlay_plan_path)

    plan, timeline, overlay_plan = _load_and_cross_check_plans(renderer_plan_path, timeline_path, overlay_plan_path)

    engine_config = video_engine.load_engine_config()
    inspector_config = media_inspector.load_inspector_config()
    music_config = music_mixer.load_music_mixer_config()
    effective_subtitle_render_config = subtitle_render_config or subtitle_render_engine.load_subtitle_render_config()
    effective_overlay_render_config = overlay_render_config or overlay_render_engine.load_overlay_render_config()
    effective_overlay_asset_config = overlay_asset_config or overlay_asset_resolver.load_overlay_asset_config()

    probes = _probe_video_assets(plan, inspector_config, runner=runner)

    overlay_asset_manifest: overlay_asset_resolver.OverlayAssetManifest | None = None
    has_supported_overlay_pass = any(
        p.pass_type == renderer_plan_engine.RenderPassType.OVERLAY and _pass_is_supported(p, plan)
        for p in plan.passes
    )
    if has_supported_overlay_pass:
        overlay_asset_manifest = _resolve_overlay_assets(
            plan, overlay_plan, effective_overlay_asset_config, inspector=overlay_asset_inspector,
        )

    return RendererExecutionContext(
        plan=plan, timeline=timeline, overlay_plan=overlay_plan, probes=probes,
        overlay_asset_manifest=overlay_asset_manifest, engine_config=engine_config,
        inspector_config=inspector_config, music_config=music_config,
        subtitle_render_config=effective_subtitle_render_config,
        overlay_render_config=effective_overlay_render_config,
    )


def execute_single_pass(
    pass_: renderer_plan_engine.RendererPass,
    context: RendererExecutionContext,
    *,
    video_input: Path | None,
    output_path: Path,
    dry_run: bool,
    force: bool,
    runner: Any,
) -> PassExecutionResult:
    """
    Executes exactly one pass, dispatching to the same private
    _execute_video_pass()/_execute_subtitle_pass()/_execute_overlay_pass()/
    _execute_music_pass()/_execute_final_encode_pass() helpers
    execute_renderer_plan()'s own loop calls -- zero duplicated ffmpeg/
    adapter logic. Raises RendererExecutionUnsupportedPassError for a
    subtitle/overlay pass whose hint isn't executable (a per-pass
    caller naturally discovers this the moment it reaches that pass,
    e.g. via a dry-run fingerprint check).
    """
    plan = context.plan

    if pass_.pass_type == renderer_plan_engine.RenderPassType.VIDEO:
        return _execute_video_pass(
            pass_, plan, context.probes, context.engine_config,
            output_path=output_path, dry_run=dry_run, force=force, runner=runner,
        )

    if pass_.pass_type == renderer_plan_engine.RenderPassType.SUBTITLE:
        if not _pass_is_supported(pass_, plan):
            raise RendererExecutionUnsupportedPassError(
                f"Pass {pass_.pass_id} (subtitle) is not executable (renderer hint is not 'ass'/'drawtext')."
            )
        hint = _find_hint_for_pass(plan, pass_.pass_id)
        return _execute_subtitle_pass(
            pass_, plan, hint, context.overlay_plan, context.subtitle_render_config,
            video_input=video_input, output_path=output_path,
            canvas_width=plan.canvas_width, canvas_height=plan.canvas_height,
            dry_run=dry_run, force=force, runner=runner,
        )

    if pass_.pass_type == renderer_plan_engine.RenderPassType.OVERLAY:
        if not _pass_is_supported(pass_, plan):
            raise RendererExecutionUnsupportedPassError(
                f"Pass {pass_.pass_id} (overlay) is not executable "
                "(renderer hint is not 'overlay_png'/'overlay_alpha')."
            )
        return _execute_overlay_pass(
            pass_, plan, context.overlay_plan, context.overlay_asset_manifest, context.overlay_render_config,
            video_input=video_input, output_path=output_path,
            canvas_width=plan.canvas_width, canvas_height=plan.canvas_height,
            dry_run=dry_run, force=force, runner=runner,
        )

    if pass_.pass_type == renderer_plan_engine.RenderPassType.MUSIC:
        return _execute_music_pass(
            pass_, plan, video_pass_output=video_input, final_output_path=output_path,
            music_config=context.music_config, dry_run=dry_run, force=force, runner=runner,
        )

    if pass_.pass_type == renderer_plan_engine.RenderPassType.FINAL_ENCODE:
        return _execute_final_encode_pass(
            pass_, final_output_path=output_path, engine_config=context.engine_config, dry_run=dry_run,
        )

    raise RendererExecutionUnsupportedPassError(
        f"Pass {pass_.pass_id} has an unrecognized pass_type {pass_.pass_type!r}"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def execute_renderer_plan(
    renderer_plan_path: str | Path,
    timeline_path: str | Path,
    overlay_plan_path: str | Path,
    config: RendererExecutionConfig,
    *,
    output_path: str | Path,
    dry_run: bool | None = None,
    allow_partial_execution: bool | None = None,
    force: bool = False,
    runner: Any = None,
    subtitle_render_config: subtitle_render_engine.SubtitleRenderConfig | None = None,
    overlay_render_config: overlay_render_engine.OverlayRenderConfig | None = None,
    overlay_asset_config: overlay_asset_resolver.OverlayAssetResolverConfig | None = None,
    overlay_asset_inspector: Any = None,
) -> RendererExecutionResult:
    """
    runner=None (the default) lets each delegated engine
    (media_inspector.py/video_engine.py/music_mixer.py/
    subtitle_render_engine.py/overlay_render_engine.py) use its own
    real default_runner -- each raises its own correctly-typed
    NotFoundError family on a missing ffmpeg/ffprobe binary, which this
    module's own except clauses already handle. Tests inject a single
    shared FakeRunner explicitly instead.

    subtitle_render_config=None / overlay_render_config=None /
    overlay_asset_config=None (the defaults) load the real
    config/video/subtitle_render.yaml / overlay_render.yaml /
    overlay_assets.yaml -- the same "None means use the real loader"
    convention already used for engine_config/inspector_config/
    music_config internally. Exposed as explicit parameters (unlike
    those) because font resolution / overlay-asset resolution
    genuinely depend on filesystem fixtures a caller may need to point
    at test-specific paths; none of the three is exposed via the CLI,
    which always uses the real configs.

    overlay_asset_inspector=None (the default) lets
    overlay_asset_resolver.resolve_overlay_assets() use its own real
    default_image_inspector (lazy Pillow import) when the config
    requires one -- tests inject a FakeInspector instead, mirroring
    the resolver's own existing convention, so this module's own test
    suite never needs real Pillow.
    """
    renderer_plan_path = Path(renderer_plan_path)
    timeline_path = Path(timeline_path)
    overlay_plan_path = Path(overlay_plan_path)
    output_path = Path(output_path)

    effective_dry_run = config.dry_run_default if dry_run is None else dry_run
    effective_allow_partial = (
        config.allow_partial_execution_default if allow_partial_execution is None else allow_partial_execution
    )

    if output_path.resolve() == renderer_plan_path.resolve():
        raise UnsafeRendererExecutionOutputError("--output must not be the same path as --renderer-plan")
    if output_path.resolve() == timeline_path.resolve():
        raise UnsafeRendererExecutionOutputError("--output must not be the same path as --timeline")
    if output_path.resolve() == overlay_plan_path.resolve():
        raise UnsafeRendererExecutionOutputError("--output must not be the same path as --overlay-plan")

    started_at = _now_iso()

    plan, timeline, overlay_plan = _load_and_cross_check_plans(renderer_plan_path, timeline_path, overlay_plan_path)

    if not effective_dry_run and output_path.exists() and not force:
        raise RendererExecutionOutputExistsError(f"{output_path} already exists; pass force=True to overwrite.")

    # Checked up front, before any pass executes: a pass discovered only
    # *after* an earlier pass had already written directly to the final
    # --output path would leave a misleading partial file there. Failing
    # before touching anything is the only way to guarantee that never
    # happens. A subtitle pass is supported only when its renderer hint
    # is "ass"/"drawtext" (Phase 11F.1); an "overlay" pass, or a
    # subtitle pass with any other hint, remains unsupported.
    unsupported_passes = [
        pass_
        for pass_ in plan.passes
        if pass_.pass_type in (renderer_plan_engine.RenderPassType.SUBTITLE, renderer_plan_engine.RenderPassType.OVERLAY)
        and not _pass_is_supported(pass_, plan)
    ]
    if unsupported_passes and not effective_allow_partial:
        first = unsupported_passes[0]
        raise RendererExecutionUnsupportedPassError(
            f"Pass {first.pass_id} ({first.pass_type}) is not executable "
            "(an overlay pass is only executable with an 'overlay_png' or "
            "'overlay_alpha' renderer hint; a subtitle pass is only "
            "executable with an 'ass' or 'drawtext' renderer hint). Pass "
            "allow_partial_execution=True to proceed without it."
        )

    producing_pass_types = (
        renderer_plan_engine.RenderPassType.VIDEO,
        renderer_plan_engine.RenderPassType.SUBTITLE,
        renderer_plan_engine.RenderPassType.OVERLAY,
        renderer_plan_engine.RenderPassType.MUSIC,
    )
    producing_passes = [p for p in plan.passes if p.pass_type in producing_pass_types and _pass_is_supported(p, plan)]
    last_producing_pass_id = producing_passes[-1].pass_id if producing_passes else None

    def _target_path_for(pass_: renderer_plan_engine.RendererPass) -> Path:
        if pass_.pass_id == last_producing_pass_id:
            return output_path
        if pass_.pass_type == renderer_plan_engine.RenderPassType.VIDEO:
            return output_path.with_name(f"{output_path.stem}{config.intermediate_filename_suffix}{output_path.suffix}")
        if pass_.pass_type == renderer_plan_engine.RenderPassType.SUBTITLE:
            return output_path.with_name(
                f"{output_path.stem}{config.subtitle_intermediate_filename_suffix}{output_path.suffix}"
            )
        if pass_.pass_type == renderer_plan_engine.RenderPassType.OVERLAY:
            return output_path.with_name(
                f"{output_path.stem}{config.overlay_intermediate_filename_suffix}{output_path.suffix}"
            )
        return output_path.with_name(f"{output_path.stem}_after_music{output_path.suffix}")

    pass_results: list[PassExecutionResult] = []
    warnings: list[str] = []
    errors: list[str] = []
    final_output_path: str | None = None
    current_video_path: Path | None = None

    try:
        engine_config = video_engine.load_engine_config()
        inspector_config = media_inspector.load_inspector_config()
        music_config = music_mixer.load_music_mixer_config()
        effective_subtitle_render_config = subtitle_render_config or subtitle_render_engine.load_subtitle_render_config()
        effective_overlay_render_config = overlay_render_config or overlay_render_engine.load_overlay_render_config()
        effective_overlay_asset_config = overlay_asset_config or overlay_asset_resolver.load_overlay_asset_config()

        probes = _probe_video_assets(plan, inspector_config, runner=runner)

        overlay_asset_manifest: overlay_asset_resolver.OverlayAssetManifest | None = None
        has_supported_overlay_pass = any(
            p.pass_type == renderer_plan_engine.RenderPassType.OVERLAY and _pass_is_supported(p, plan)
            for p in plan.passes
        )
        if has_supported_overlay_pass:
            overlay_asset_manifest = _resolve_overlay_assets(
                plan, overlay_plan, effective_overlay_asset_config, inspector=overlay_asset_inspector,
            )

        context = RendererExecutionContext(
            plan=plan, timeline=timeline, overlay_plan=overlay_plan, probes=probes,
            overlay_asset_manifest=overlay_asset_manifest, engine_config=engine_config,
            inspector_config=inspector_config, music_config=music_config,
            subtitle_render_config=effective_subtitle_render_config,
            overlay_render_config=effective_overlay_render_config,
        )

        for pass_ in plan.passes:
            if pass_.pass_type == renderer_plan_engine.RenderPassType.VIDEO:
                target = _target_path_for(pass_)
                result = execute_single_pass(
                    pass_, context, video_input=None, output_path=target,
                    dry_run=effective_dry_run, force=force, runner=runner,
                )
                pass_results.append(result)
                current_video_path = target
                if pass_.pass_id == last_producing_pass_id:
                    final_output_path = str(target)

            elif pass_.pass_type == renderer_plan_engine.RenderPassType.SUBTITLE and _pass_is_supported(pass_, plan):
                target = _target_path_for(pass_)
                result = execute_single_pass(
                    pass_, context, video_input=current_video_path, output_path=target,
                    dry_run=effective_dry_run, force=force, runner=runner,
                )
                pass_results.append(result)
                current_video_path = target
                if pass_.pass_id == last_producing_pass_id:
                    final_output_path = str(target)

            elif pass_.pass_type == renderer_plan_engine.RenderPassType.OVERLAY and _pass_is_supported(pass_, plan):
                target = _target_path_for(pass_)
                result = execute_single_pass(
                    pass_, context, video_input=current_video_path, output_path=target,
                    dry_run=effective_dry_run, force=force, runner=runner,
                )
                pass_results.append(result)
                current_video_path = target
                if pass_.pass_id == last_producing_pass_id:
                    final_output_path = str(target)

            elif pass_.pass_type == renderer_plan_engine.RenderPassType.MUSIC:
                target = _target_path_for(pass_)
                result = execute_single_pass(
                    pass_, context, video_input=current_video_path, output_path=target,
                    dry_run=effective_dry_run, force=force, runner=runner,
                )
                pass_results.append(result)
                if not effective_dry_run:
                    current_video_path = target
                if pass_.pass_id == last_producing_pass_id:
                    final_output_path = str(target)

            elif pass_.pass_type in (
                renderer_plan_engine.RenderPassType.SUBTITLE,
                renderer_plan_engine.RenderPassType.OVERLAY,
            ):
                # Reaching here means allow_partial_execution was true --
                # otherwise the upfront pre-flight check above already
                # raised before any pass (including video) ran.
                message = f"Pass {pass_.pass_id} ({pass_.pass_type}) skipped -- not executable"
                warnings.append(message)
                pass_results.append(
                    PassExecutionResult(
                        pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="skipped", warnings=[message]
                    )
                )

            elif pass_.pass_type == renderer_plan_engine.RenderPassType.FINAL_ENCODE:
                target = Path(final_output_path) if final_output_path else output_path
                result = execute_single_pass(
                    pass_, context, video_input=None, output_path=target,
                    dry_run=effective_dry_run, force=force, runner=runner,
                )
                pass_results.append(result)

            else:
                message = f"Pass {pass_.pass_id} has an unrecognized pass_type {pass_.pass_type!r}; skipped"
                warnings.append(message)
                pass_results.append(
                    PassExecutionResult(
                        pass_id=pass_.pass_id, pass_type=pass_.pass_type, status="skipped", warnings=[message]
                    )
                )

    except RendererExecutionEngineError as exc:
        errors.append(str(exc))
        failure_result = RendererExecutionResult(
            renderer_plan_id=plan.renderer_plan_id, timeline_id=timeline.timeline_id, dry_run=effective_dry_run,
            started_at=started_at, finished_at=_now_iso(), passes=pass_results,
            final_output_path=final_output_path, warnings=warnings, errors=errors, result="failed",
        )
        if not effective_dry_run:
            try:
                log_path = output_path.with_name(f"{output_path.stem}{config.log_filename_suffix}")
                _save_execution_log(failure_result, log_path)
            except RendererExecutionLogError:
                pass
        raise

    skipped_count = sum(1 for p in pass_results if p.status == "skipped")
    if effective_dry_run:
        overall_result = "dry_run"
    elif skipped_count > 0:
        overall_result = "partial"
    else:
        overall_result = "success"

    execution_result = RendererExecutionResult(
        renderer_plan_id=plan.renderer_plan_id, timeline_id=timeline.timeline_id, dry_run=effective_dry_run,
        started_at=started_at, finished_at=_now_iso(), passes=pass_results,
        final_output_path=final_output_path, warnings=warnings, errors=errors, result=overall_result,
    )

    if not effective_dry_run:
        log_path = output_path.with_name(f"{output_path.stem}{config.log_filename_suffix}")
        _save_execution_log(execution_result, log_path)

    return execution_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Renderer Execution Engine (Phase 11F). Executes a validated "
            "renderer_plan.json by orchestrating video_engine.py/music_mixer.py's "
            "own real ffmpeg execution -- never constructing ffmpeg commands "
            "itself. Defaults to --dry-run; pass --execute to actually render."
        )
    )

    parser.add_argument("--renderer-plan", dest="renderer_plan", required=True, help="Path to the source renderer_plan.json.")
    parser.add_argument("--timeline", required=True, help="Path to the source timeline.json.")
    parser.add_argument(
        "--overlay-plan", dest="overlay_plan", required=True,
        help="Path to the source overlay_plan.json (used to resolve real subtitle cue text/timing/style).",
    )
    parser.add_argument("--output", required=True, help="Path to write the final rendered video.")
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/renderer_execution.yaml.")
    parser.add_argument("--execute", action="store_true", help="Actually invoke ffmpeg. Without this, the engine only plans and prints.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-partial-execution",
        dest="allow_partial_execution",
        action="store_true",
        help="Proceed even when the plan contains a subtitle/overlay pass this phase cannot execute.",
    )
    parser.add_argument("--json", dest="as_json", action="store_true")

    return parser.parse_args(argv)


def _print_result(result: RendererExecutionResult) -> None:
    print()
    print("AIKO Renderer Execution Engine (Phase 11F)")
    print("-----------------------------------------------")
    print(f"renderer_plan_id:  {result.renderer_plan_id}")
    print(f"timeline_id:       {result.timeline_id}")
    print(f"dry_run:           {result.dry_run}")
    print(f"result:            {result.result}")
    for pass_result in result.passes:
        suffix = f" -> {pass_result.output_path}" if pass_result.output_path else ""
        print(f"  pass {pass_result.pass_id} ({pass_result.pass_type}): {pass_result.status}{suffix}")
        if pass_result.command:
            print(f"    command: {' '.join(pass_result.command)}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    for error in result.errors:
        print(f"error:   {error}")
    print(f"final_output_path: {result.final_output_path}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_renderer_execution_config(arguments.config)
        result = execute_renderer_plan(
            arguments.renderer_plan,
            arguments.timeline,
            arguments.overlay_plan,
            config,
            output_path=arguments.output,
            dry_run=not arguments.execute,
            allow_partial_execution=arguments.allow_partial_execution,
            force=arguments.force,
        )

        if arguments.as_json:
            print(json.dumps(renderer_execution_result_to_dict(result), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_result(result)

        if result.result == "failed":
            raise SystemExit(1)
    except RendererExecutionEngineError as exc:
        print(f"[RendererExecutionEngine] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
