from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from . import filter_graph_builder, filter_graph_serializer, overlay_asset_resolver, overlay_plan_engine, renderer_plan_engine

# Phase 11F.6 — Overlay Render Engine. The module permitted to burn
# real overlay compositing (logos, watermarks, stickers, CTAs,
# location badges) into video. Standalone and renderer-neutral,
# mirroring subtitle_render_engine.py's own shape: it accepts a
# validated overlay render request (a video, an already-resolved
# OverlayAssetManifest, and the RendererPlan/OverlayPlan that describe
# which overlays are active), resolves it into a typed FilterGraph via
# filter_graph_builder.py, converts that into real ffmpeg filter
# syntax via filter_graph_serializer.py, builds one safe multi-input
# ffmpeg command, executes it exactly once via an injectable runner,
# verifies the output, and returns structured diagnostics. This module
# never resolves overlay assets itself (that is Renderer Execution's
# job, mirroring how it already probes video assets before any pass
# runs), never derives filter parameters itself, and never constructs
# or edits a single character of filter syntax -- every character of
# the -filter_complex expression it uses comes verbatim from
# filter_graph_serializer.SerializedFilterGraph.
#
# It intentionally does NOT reuse video_engine.VideoEngine: that
# class's ConcatPlan/execute_plan() are shaped entirely around N-input
# *concatenation* (lossless_copy / normalized_render / rejected) and
# have no representation for an arbitrary multi-input filter_complex
# graph. Its own DISALLOWED_ACTIONS tuple explicitly bars
# "add_logo"/"add_watermark" -- a standing policy that class never
# grows overlay-compositing logic. Rather than bend that model, this
# module owns its own small ProcessResult/SubprocessRunner/
# default_runner trio and command builder, the same convention
# video_engine.py/music_mixer.py/media_inspector.py/
# subtitle_render_engine.py each already follow independently.

DEFAULT_OVERLAY_RENDER_CONFIG_RELATIVE_PATH = Path("config") / "video" / "overlay_render.yaml"


def _runtime_root() -> Path:
    """
    overlay_render_engine.py location:
    10_apps/claude_runtime/src/overlay_render_engine.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OverlayRenderEngineError(RuntimeError):
    """Base error for the Phase 11F.6 Overlay Render Engine."""


class OverlayRenderConfigError(OverlayRenderEngineError):
    """Raised when config/video/overlay_render.yaml is missing or invalid."""


class OverlayRenderRequestError(OverlayRenderEngineError):
    """Raised for a structurally invalid overlay render request."""


class OverlayVideoNotFoundError(OverlayRenderEngineError):
    """Raised when the input video does not exist or is not a regular file."""


class OverlayVideoEmptyError(OverlayRenderEngineError):
    """Raised when the input video is zero bytes."""


class UnsafeOverlayRenderOutputError(OverlayRenderEngineError):
    """Raised when the output path is a directory or resolves to the
    same path as an input asset."""


class OverlayRenderOutputExistsError(OverlayRenderEngineError):
    """Raised when the output already exists and force was not given."""


class OverlayRenderFilterGraphError(OverlayRenderEngineError):
    """Raised when filter_graph_builder.py planning/validation or
    filter_graph_serializer.py serialization fails while building an
    overlay render plan. The original error message is preserved."""


class FFmpegNotFoundError(OverlayRenderEngineError):
    """Raised when the configured ffmpeg binary cannot be found/executed."""


class FFmpegExecutionError(OverlayRenderEngineError):
    """Raised when ffmpeg exits non-zero."""


class FFmpegTimeoutError(OverlayRenderEngineError):
    """Raised when ffmpeg does not finish within the configured timeout."""


class OverlayRenderVerificationError(OverlayRenderEngineError):
    """Raised when the output file is missing or zero bytes after a
    reported-successful run."""


class OverlayRenderLogError(OverlayRenderEngineError):
    """Raised when the diagnostics log cannot be safely written."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OverlayRenderConfig:
    schema_version: str = "1.0"

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

    overwrite_requires_force: bool = True
    verify_non_zero_bytes: bool = True

    diagnostics_write_log: bool = True
    diagnostics_log_filename_suffix: str = "_overlay_render_log.json"


def default_overlay_render_config_path() -> Path:
    return _runtime_root() / DEFAULT_OVERLAY_RENDER_CONFIG_RELATIVE_PATH


def load_overlay_render_config(config_path: str | Path | None = None) -> OverlayRenderConfig:
    """Load config/video/overlay_render.yaml (or an alternate path)
    into an OverlayRenderConfig. Owns NO filter-graph or
    asset-resolution settings -- those stay owned by
    config/video/filter_graph.yaml, filter_graph_serializer.yaml, and
    overlay_assets.yaml. Raises OverlayRenderConfigError if the file is
    missing or invalid."""
    path = Path(config_path) if config_path else default_overlay_render_config_path()

    if not path.exists():
        raise OverlayRenderConfigError(f"Overlay render config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise OverlayRenderConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise OverlayRenderConfigError(f"Overlay render config is empty or invalid: {path}")

    overlay_render_section = raw.get("overlay_render") or {}
    ffmpeg_section = raw.get("ffmpeg") or {}
    video_section = raw.get("video") or {}
    output_section = raw.get("output") or {}
    diagnostics_section = raw.get("diagnostics") or {}

    return OverlayRenderConfig(
        schema_version=str(overlay_render_section.get("schema_version", "1.0")),
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
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        verify_non_zero_bytes=bool(output_section.get("verify_non_zero_bytes", True)),
        diagnostics_write_log=bool(diagnostics_section.get("write_log", True)),
        diagnostics_log_filename_suffix=str(
            diagnostics_section.get("log_filename_suffix", "_overlay_render_log.json")
        ),
    )


# ---------------------------------------------------------------------------
# Runner (injectable, mockable) -- module-owned, never imported from or
# into video_engine.py/music_mixer.py/media_inspector.py/
# subtitle_render_engine.py.
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
            f"Executable not found: {command[0]!r}. Install ffmpeg to enable real overlay rendering."
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
class OverlayRenderRequest:
    video_path: Path
    output_path: Path
    renderer_plan: renderer_plan_engine.RendererPlan
    overlay_plan: overlay_plan_engine.OverlayPlan
    overlay_asset_manifest: overlay_asset_resolver.OverlayAssetManifest
    force: bool = False
    canvas_width: int | None = None
    canvas_height: int | None = None


@dataclass(slots=True)
class OverlayRenderPlan:
    render_id: str = ""
    command: list[str] = field(default_factory=list)
    input_video: str = ""
    output_video: str = ""
    # Resolved overlay image paths, in the exact order they were added
    # as extra -i inputs (matches serialized.extra_inputs order).
    extra_inputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    filter_graph_id: str = ""
    filter_graph_validation_passed: bool = False
    serialization_id: str = ""
    output_mode: str = ""


@dataclass(slots=True)
class OverlayRenderResult:
    render_id: str = ""
    input_video: str = ""
    output_video: str = ""
    extra_inputs: list[str] = field(default_factory=list)
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
    error: str | None = None
    filter_graph_id: str = ""
    filter_graph_validation_passed: bool = False
    serialization_id: str = ""
    output_mode: str = ""


# ---------------------------------------------------------------------------
# Request construction — all pass/fail policy for inputs lives here.
# ---------------------------------------------------------------------------


def build_overlay_render_request(
    *,
    video_path: str | Path,
    output_path: str | Path,
    renderer_plan: renderer_plan_engine.RendererPlan,
    overlay_plan: overlay_plan_engine.OverlayPlan,
    overlay_asset_manifest: overlay_asset_resolver.OverlayAssetManifest,
    config: OverlayRenderConfig,
    force: bool = False,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
) -> OverlayRenderRequest:
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise OverlayVideoNotFoundError(f"Input video not found: {video_path}")
    if not video_path.is_file():
        raise OverlayVideoNotFoundError(f"Input video is not a regular file: {video_path}")
    if video_path.stat().st_size == 0:
        raise OverlayVideoEmptyError(f"Input video is zero bytes: {video_path}")

    resolved_video = video_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_video:
        raise UnsafeOverlayRenderOutputError("output_path must not be the same path as video_path")

    if output_path.exists():
        if output_path.is_dir():
            raise UnsafeOverlayRenderOutputError(f"output_path is a directory: {output_path}")
        if not force:
            raise OverlayRenderOutputExistsError(f"{output_path} already exists; pass force=True to overwrite.")

    parent = output_path.parent
    if parent.exists() and not parent.is_dir():
        raise UnsafeOverlayRenderOutputError(f"Output parent exists but is not a directory: {parent}")

    return OverlayRenderRequest(
        video_path=video_path,
        output_path=output_path,
        renderer_plan=renderer_plan,
        overlay_plan=overlay_plan,
        overlay_asset_manifest=overlay_asset_manifest,
        force=force,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


# ---------------------------------------------------------------------------
# Command construction — the filter portion (-filter_complex) comes
# entirely from an already-serialized FilterGraph (filter_graph_serializer.py);
# everything else (inputs, mapping, codec, audio, faststart, output
# path) is assembled here, never edited into the filter string itself.
# ---------------------------------------------------------------------------


def build_overlay_ffmpeg_command(
    request: OverlayRenderRequest,
    config: OverlayRenderConfig,
    *,
    serialized: filter_graph_serializer.SerializedFilterGraph,
) -> list[str]:
    command = [config.ffmpeg_binary]

    if config.ffmpeg_hide_banner:
        command.append("-hide_banner")

    command.extend(["-v", config.ffmpeg_loglevel])
    command.append("-y" if request.force else "-n")
    command.extend(["-i", str(request.video_path)])

    for extra_input in serialized.extra_inputs:
        command.extend(["-i", extra_input.resolved_path])

    command.extend([serialized.filter_argument_name, serialized.filter_expression])

    # Explicit -map, unlike subtitle_render_engine's single-input
    # implicit stream selection: overlay rendering is genuinely
    # multi-input (1 video + N images), so this follows
    # video_engine.VideoEngine.build_normalized_concat_command()'s own
    # existing precedent of explicit -map for its own multi-input
    # filter_complex case, rather than relying on ffmpeg's implicit
    # multi-input stream-selection heuristics.
    command.extend(["-map", f"[{serialized.final_output_label}]"])
    command.extend(["-map", "0:a?"])

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
# Deterministic render_id — computed from the request paths plus the
# already-deterministic filter_graph_id/serialization_id, since those
# two already fully capture the semantic overlay content (positions,
# z-order, resolved assets, alpha).
# ---------------------------------------------------------------------------


def _compute_render_id(request: OverlayRenderRequest, filter_graph_id: str, serialization_id: str) -> str:
    payload = {
        "video_path": str(request.video_path),
        "output_path": str(request.output_path),
        "canvas_width": request.canvas_width,
        "canvas_height": request.canvas_height,
        "filter_graph_id": filter_graph_id,
        "serialization_id": serialization_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Planning — never executes ffmpeg.
# ---------------------------------------------------------------------------


def build_overlay_render_plan(
    request: OverlayRenderRequest,
    config: OverlayRenderConfig,
    *,
    filter_graph_builder_callable=None,
    filter_graph_wrapper_callable=None,
    filter_graph_serializer_callable=None,
    fg_config: filter_graph_builder.FilterGraphConfig | None = None,
    serializer_config: filter_graph_serializer.FilterGraphSerializerConfig | None = None,
) -> OverlayRenderPlan:
    """
    Flow: OverlayRenderRequest -> filter_graph_builder.build_overlay_filter_spec()
    -> filter_graph_builder.build_filter_graph_from_filters(extra_inputs=...)
    -> filter_graph_serializer.serialize_filter_graph() -> ffmpeg command.
    The four *_callable/config-override parameters are additive
    dependency-injection hooks (default to the real implementations),
    mirroring subtitle_render_engine.build_subtitle_render_plan()'s own
    convention, for tests that need to observe/replace one stage.
    """
    build_filters = filter_graph_builder_callable or filter_graph_builder.build_overlay_filter_spec
    wrap_graph = filter_graph_wrapper_callable or filter_graph_builder.build_filter_graph_from_filters
    serialize = filter_graph_serializer_callable or filter_graph_serializer.serialize_filter_graph

    effective_fg_config = fg_config or filter_graph_builder.load_filter_graph_config()
    effective_serializer_config = serializer_config or filter_graph_serializer.load_filter_graph_serializer_config()

    try:
        filters, extra_inputs, skipped_overlay_ids = build_filters(
            request.overlay_plan, request.renderer_plan, request.overlay_asset_manifest, effective_fg_config,
        )
    except filter_graph_builder.FilterGraphBuilderError as exc:
        raise OverlayRenderFilterGraphError(f"Failed to build overlay filter spec: {exc}") from exc

    try:
        graph = wrap_graph(
            filters, pass_id="pass_overlay", pass_type="overlay", hint_type="overlay_png",
            renderer_plan_id=request.renderer_plan.renderer_plan_id,
            overlay_plan_id=request.overlay_plan.plan_id,
            extra_inputs=extra_inputs, metadata={"skipped_overlays": skipped_overlay_ids},
            config=effective_fg_config,
        )
    except filter_graph_builder.FilterGraphBuilderError as exc:
        raise OverlayRenderFilterGraphError(f"Failed to build filter graph: {exc}") from exc

    if not graph.validation.passed:
        raise OverlayRenderFilterGraphError(
            f"Filter graph {graph.graph_id} failed validation: {'; '.join(graph.validation.errors)}"
        )

    try:
        serialized = serialize(graph, effective_serializer_config)
    except filter_graph_serializer.FilterGraphSerializerError as exc:
        raise OverlayRenderFilterGraphError(f"Failed to serialize filter graph: {exc}") from exc

    command = build_overlay_ffmpeg_command(request, config, serialized=serialized)
    render_id = _compute_render_id(request, graph.graph_id, serialized.serialization_id)

    return OverlayRenderPlan(
        render_id=render_id,
        command=command,
        input_video=str(request.video_path),
        output_video=str(request.output_path),
        extra_inputs=[e.resolved_path for e in serialized.extra_inputs],
        warnings=[],
        filter_graph_id=graph.graph_id,
        filter_graph_validation_passed=graph.validation.passed,
        serialization_id=serialized.serialization_id,
        output_mode=serialized.output_mode,
    )


# ---------------------------------------------------------------------------
# Output verification — read-only.
# ---------------------------------------------------------------------------


def verify_overlay_render_output(
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
        raise UnsafeOverlayRenderOutputError("Output path resolved to the same file as the input video")

    if inspector is not None:
        inspector(output_path)

    return True, output_path.stat().st_size


# ---------------------------------------------------------------------------
# Execution — runs the built command exactly once via the injected
# runner. No automatic retry, no hidden fallback command.
# ---------------------------------------------------------------------------


def execute_overlay_render_plan(
    request: OverlayRenderRequest,
    config: OverlayRenderConfig,
    *,
    runner: SubprocessRunner = default_runner,
    inspector: Callable[[Path], Any] | None = None,
    filter_graph_builder_callable=None,
    filter_graph_wrapper_callable=None,
    filter_graph_serializer_callable=None,
    fg_config: filter_graph_builder.FilterGraphConfig | None = None,
    serializer_config: filter_graph_serializer.FilterGraphSerializerConfig | None = None,
) -> OverlayRenderResult:
    plan = build_overlay_render_plan(
        request, config,
        filter_graph_builder_callable=filter_graph_builder_callable,
        filter_graph_wrapper_callable=filter_graph_wrapper_callable,
        filter_graph_serializer_callable=filter_graph_serializer_callable,
        fg_config=fg_config, serializer_config=serializer_config,
    )

    if request.output_path.exists():
        if request.output_path.is_dir():
            raise UnsafeOverlayRenderOutputError(f"output_path is a directory: {request.output_path}")
        if not request.force:
            raise OverlayRenderOutputExistsError(
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

    output_exists, output_size_bytes = verify_overlay_render_output(
        request.output_path, request.video_path, inspector=inspector
    )

    if not output_exists:
        raise OverlayRenderVerificationError(
            f"ffmpeg reported success but no output file exists: {request.output_path}"
        )
    if config.verify_non_zero_bytes and output_size_bytes == 0:
        raise OverlayRenderVerificationError(f"Output file is zero bytes: {request.output_path}")

    audio_preserved: bool | None = None
    if inspector is not None:
        info = inspector(request.output_path)
        audio_preserved = bool(getattr(info, "has_audio", None))

    result = OverlayRenderResult(
        render_id=plan.render_id,
        input_video=plan.input_video,
        output_video=plan.output_video,
        extra_inputs=list(plan.extra_inputs),
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
        save_overlay_render_log(result, log_path)

    return result


# ---------------------------------------------------------------------------
# Diagnostics log — atomic write, never logs secrets/env vars.
# ---------------------------------------------------------------------------


def overlay_render_result_to_dict(result: OverlayRenderResult) -> dict[str, Any]:
    return asdict(result)


def save_overlay_render_log(result: OverlayRenderResult, path: Path) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(overlay_render_result_to_dict(result), indent=2, sort_keys=True, ensure_ascii=False)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise OverlayRenderLogError(f"Failed to write overlay render log {path}: {exc}") from exc
    return path


# ---------------------------------------------------------------------------
# CLI — a lower-level debugging entry point. Loads an already-built
# overlay_asset_manifest.json from disk rather than resolving it
# itself; resolution stays Renderer Execution's job in the real
# pipeline (same relationship subtitle_render_engine.py's CLI has to
# renderer_execution_engine.py's real orchestration).
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Overlay Render Engine (Phase 11F.6). Burns overlay compositing "
            "into a video via a safe, injectable, multi-input ffmpeg command. "
            "Defaults to real execution; pass --dry-run to only build and print "
            "the command."
        )
    )

    parser.add_argument("--video", required=True, help="Path to the source video.")
    parser.add_argument("--output", required=True, help="Path to write the overlay-rendered video.")
    parser.add_argument("--renderer-plan", dest="renderer_plan", required=True, help="Path to the source renderer_plan.json.")
    parser.add_argument("--overlay-plan", dest="overlay_plan", required=True, help="Path to the source overlay_plan.json.")
    parser.add_argument(
        "--overlay-asset-manifest", dest="overlay_asset_manifest", required=True,
        help="Path to an already-built overlay_asset_manifest.json (Phase 11F.4).",
    )
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/overlay_render.yaml.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")

    return parser.parse_args(argv)


def _print_plan(plan: OverlayRenderPlan) -> None:
    print()
    print("AIKO Overlay Render Engine — dry run (Phase 11F.6)")
    print("---------------------------------------------------------")
    print(f"render_id:     {plan.render_id}")
    print(f"input_video:   {plan.input_video}")
    print(f"output_video:  {plan.output_video}")
    print(f"extra_inputs:  {plan.extra_inputs}")
    print(f"command:       {' '.join(plan.command)}")
    print()


def _print_result(result: OverlayRenderResult) -> None:
    print()
    print("AIKO Overlay Render Engine (Phase 11F.6)")
    print("---------------------------------------------")
    print(f"render_id:      {result.render_id}")
    print(f"output_video:   {result.output_video}")
    print(f"output_exists:  {result.output_exists}")
    print(f"return_code:    {result.return_code}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_overlay_render_config(arguments.config)

        try:
            renderer_plan = renderer_plan_engine.load_renderer_plan(arguments.renderer_plan)
        except renderer_plan_engine.RendererPlanEngineError as exc:
            raise OverlayRenderRequestError(f"Failed to load renderer plan {arguments.renderer_plan}: {exc}") from exc

        try:
            overlay_plan = overlay_plan_engine.load_overlay_plan(arguments.overlay_plan)
        except overlay_plan_engine.OverlayPlanEngineError as exc:
            raise OverlayRenderRequestError(f"Failed to load overlay plan {arguments.overlay_plan}: {exc}") from exc

        try:
            overlay_asset_manifest = overlay_asset_resolver.load_overlay_asset_manifest(arguments.overlay_asset_manifest)
        except overlay_asset_resolver.OverlayAssetResolverError as exc:
            raise OverlayRenderRequestError(
                f"Failed to load overlay asset manifest {arguments.overlay_asset_manifest}: {exc}"
            ) from exc

        request = build_overlay_render_request(
            video_path=arguments.video,
            output_path=arguments.output,
            renderer_plan=renderer_plan,
            overlay_plan=overlay_plan,
            overlay_asset_manifest=overlay_asset_manifest,
            config=config,
            force=arguments.force,
        )

        if arguments.dry_run:
            plan = build_overlay_render_plan(request, config)
            if arguments.as_json:
                print(json.dumps(asdict(plan), indent=2, sort_keys=True, ensure_ascii=False))
            else:
                _print_plan(plan)
            return

        result = execute_overlay_render_plan(request, config)
        if arguments.as_json:
            print(json.dumps(overlay_render_result_to_dict(result), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_result(result)
    except OverlayRenderEngineError as exc:
        print(f"[OverlayRenderEngine] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
