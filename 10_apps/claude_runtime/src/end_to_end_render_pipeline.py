from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from . import (
    media_inspector,
    overlay_asset_resolver,
    renderer_execution_engine,
    renderer_plan_engine,
    video_validator,
)

# Phase 11F.7 — End-to-End Render Pipeline. A pure orchestration layer
# over renderer_execution_engine.py's per-pass seam
# (prepare_execution_context()/execute_single_pass(), Phase 11F.7's own
# additive change to that module -- see its module docstring). This
# module adds everything renderer_execution_engine.py does not own: a
# deterministic, ownership-marked workspace; per-pass resume/retry
# checkpointing (real sha256 checksums + command fingerprints, never
# filename-existence alone); a candidate/promotion split so the
# requested --output path is never touched until every pass has
# succeeded and the candidate has passed real final-media verification
# (media_inspector.py + video_validator.py, both already-implemented
# and reused as-is); atomic Path.replace() promotion; cleanup; and two
# atomic JSON reports. It never builds ffmpeg syntax, never serializes
# filters, never resolves overlay assets or fonts itself, never renders
# a subtitle/overlay/music pass directly, and never mutates a source
# planning file -- every one of those responsibilities stays owned by
# renderer_execution_engine.py and the engines it already delegates to.

DEFAULT_END_TO_END_RENDER_CONFIG_RELATIVE_PATH = Path("config") / "video" / "end_to_end_render.yaml"


def _runtime_root() -> Path:
    """
    end_to_end_render_pipeline.py location:
    10_apps/claude_runtime/src/end_to_end_render_pipeline.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EndToEndRenderError(RuntimeError):
    """Base error for the Phase 11F.7 End-to-End Render Pipeline."""


class EndToEndRenderConfigError(EndToEndRenderError):
    """Raised when config/video/end_to_end_render.yaml is missing or invalid."""


class EndToEndRenderRequestError(EndToEndRenderError):
    """Raised for a structurally invalid request (missing/malformed
    inputs, an overlay file required but not given, resume+restart
    both set)."""


class RenderWorkspaceError(EndToEndRenderError):
    """Raised for a general workspace I/O failure."""


class UnsafeRenderWorkspaceError(EndToEndRenderError):
    """Raised when the workspace path is unsafe (e.g. a directory
    collision, or exists as a non-directory)."""


class RenderWorkspaceOwnershipError(EndToEndRenderError):
    """Raised when a target workspace directory exists but lacks this
    module's own ownership marker -- never touched, cleaned, or reused."""


class RenderStateError(EndToEndRenderError):
    """Raised when workspace state.json is missing, malformed, or
    cannot be written."""


class RenderResumeError(EndToEndRenderError):
    """Raised when --resume cannot proceed (no prior state, identity
    mismatch, or a workspace exists but neither --resume nor --restart
    was given)."""


class RenderRestartError(EndToEndRenderError):
    """Raised when --restart cannot proceed (workspace ownership marker
    missing)."""


class RenderPassOrderError(EndToEndRenderError):
    """Raised when the renderer plan's pass order/types don't match
    config.supported_pass_order, or a duplicate pass_type is present
    and not allowed."""


class RenderPassExecutionError(EndToEndRenderError):
    """Raised when a pass fails execution, or an unsupported pass is
    discovered during the pre-flight check."""


class RenderPassVerificationError(EndToEndRenderError):
    """Raised when a pass reports success but its output is missing or
    empty."""


class FinalCandidateVerificationError(EndToEndRenderError):
    """Raised when the final candidate fails verification
    (media_inspector.py / video_validator.py checks)."""


class FinalPromotionError(EndToEndRenderError):
    """Raised when atomic promotion of the verified candidate fails."""


class CrossDevicePromotionError(FinalPromotionError):
    """Raised when promotion would cross a filesystem boundary and
    promotion.allow_cross_device_copy is false."""


class FinalOutputExistsError(EndToEndRenderError):
    """Raised when --output already exists and --force was not given."""


class UnsafeFinalOutputError(EndToEndRenderError):
    """Raised when --output collides with a source input path."""


class RenderCleanupError(EndToEndRenderError):
    """Raised for a hard cleanup failure (individual deletion failures
    are recorded in CleanupResult.failed instead, never raised)."""


class RenderReportError(EndToEndRenderError):
    """Raised when a report/log cannot be written."""


# ---------------------------------------------------------------------------
# String-constant "enum"
# ---------------------------------------------------------------------------


class PipelineStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    INVALIDATED = "invalidated"
    PROMOTED = "promoted"
    ALL = (PENDING, RUNNING, SUCCEEDED, FAILED, SKIPPED, INVALIDATED, PROMOTED)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EndToEndRenderConfig:
    schema_version: str = "1.0"
    supported_pass_order: tuple[str, ...] = ("video", "subtitle", "overlay", "music", "final_encode")
    allow_duplicate_pass_types: bool = False
    fail_on_unsupported_pass: bool = True

    workspace_default_root: str = "output/render_workspaces"
    workspace_ownership_marker: str = ".ai_influencer_render_workspace"
    workspace_state_filename: str = "state.json"
    workspace_manifest_filename: str = "execution_manifest.json"
    workspace_logs_directory: str = "logs"
    workspace_diagnostics_directory: str = "diagnostics"
    workspace_intermediates_directory: str = "intermediates"
    workspace_final_directory: str = "final"

    intermediate_video_filename: str = "01_video.mp4"
    intermediate_subtitle_filename: str = "02_subtitled.mp4"
    intermediate_overlay_filename: str = "03_overlayed.mp4"
    intermediate_music_filename: str = "04_music.mp4"
    intermediate_candidate_filename: str = "05_final_candidate.mp4"
    intermediates_keep_on_success: bool = False
    intermediates_keep_on_failure: bool = True

    resume_enabled: bool = True
    resume_require_checksum_match: bool = True
    resume_require_command_fingerprint_match: bool = True
    resume_invalidate_downstream: bool = True

    promotion_atomic_required: bool = True
    promotion_allow_cross_device_copy: bool = False
    promotion_preserve_existing_until_verified: bool = True

    verification_enabled: bool = True
    verification_duration_tolerance_seconds: float = 0.25
    verification_require_video_stream: bool = True
    verification_require_vertical_orientation: bool = True
    verification_require_audio_when_planned: bool = True
    verification_validator_profile: str = "instagram_reel"

    cleanup_verify_workspace_ownership: bool = True
    cleanup_preserve_reports: bool = True
    cleanup_preserve_logs: bool = True
    cleanup_preserve_state: bool = True

    reporting_write_detailed_log: bool = True
    reporting_write_execution_report: bool = True
    reporting_detailed_log_filename: str = "renderer_execution_log.json"
    reporting_report_filename: str = "end_to_end_render_report.json"
    reporting_atomic_write: bool = True


def default_end_to_end_render_config_path() -> Path:
    return _runtime_root() / DEFAULT_END_TO_END_RENDER_CONFIG_RELATIVE_PATH


def load_end_to_end_render_config(config_path: str | Path | None = None) -> EndToEndRenderConfig:
    """Load config/video/end_to_end_render.yaml (or an alternate path)
    into an EndToEndRenderConfig. Owns NO ffmpeg/ffprobe, filter-graph,
    or asset-resolution settings -- those stay owned by every engine's
    own config, reached only through renderer_execution_engine.py's
    per-pass seam. Raises EndToEndRenderConfigError if the file is
    missing or invalid."""
    path = Path(config_path) if config_path else default_end_to_end_render_config_path()

    if not path.exists():
        raise EndToEndRenderConfigError(f"End-to-end render config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise EndToEndRenderConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise EndToEndRenderConfigError(f"End-to-end render config is empty or invalid: {path}")

    pipeline_section = raw.get("pipeline") or {}
    workspace_section = raw.get("workspace") or {}
    intermediates_section = raw.get("intermediates") or {}
    resume_section = raw.get("resume") or {}
    promotion_section = raw.get("promotion") or {}
    verification_section = raw.get("verification") or {}
    cleanup_section = raw.get("cleanup") or {}
    reporting_section = raw.get("reporting") or {}

    return EndToEndRenderConfig(
        schema_version=str(pipeline_section.get("schema_version", "1.0")),
        supported_pass_order=tuple(
            str(p) for p in (pipeline_section.get("supported_pass_order") or ["video", "subtitle", "overlay", "music", "final_encode"])
        ),
        allow_duplicate_pass_types=bool(pipeline_section.get("allow_duplicate_pass_types", False)),
        fail_on_unsupported_pass=bool(pipeline_section.get("fail_on_unsupported_pass", True)),
        workspace_default_root=str(workspace_section.get("default_root", "output/render_workspaces")),
        workspace_ownership_marker=str(workspace_section.get("ownership_marker", ".ai_influencer_render_workspace")),
        workspace_state_filename=str(workspace_section.get("state_filename", "state.json")),
        workspace_manifest_filename=str(workspace_section.get("manifest_filename", "execution_manifest.json")),
        workspace_logs_directory=str(workspace_section.get("logs_directory", "logs")),
        workspace_diagnostics_directory=str(workspace_section.get("diagnostics_directory", "diagnostics")),
        workspace_intermediates_directory=str(workspace_section.get("intermediates_directory", "intermediates")),
        workspace_final_directory=str(workspace_section.get("final_directory", "final")),
        intermediate_video_filename=str(intermediates_section.get("video_filename", "01_video.mp4")),
        intermediate_subtitle_filename=str(intermediates_section.get("subtitle_filename", "02_subtitled.mp4")),
        intermediate_overlay_filename=str(intermediates_section.get("overlay_filename", "03_overlayed.mp4")),
        intermediate_music_filename=str(intermediates_section.get("music_filename", "04_music.mp4")),
        intermediate_candidate_filename=str(intermediates_section.get("candidate_filename", "05_final_candidate.mp4")),
        intermediates_keep_on_success=bool(intermediates_section.get("keep_on_success", False)),
        intermediates_keep_on_failure=bool(intermediates_section.get("keep_on_failure", True)),
        resume_enabled=bool(resume_section.get("enabled", True)),
        resume_require_checksum_match=bool(resume_section.get("require_checksum_match", True)),
        resume_require_command_fingerprint_match=bool(resume_section.get("require_command_fingerprint_match", True)),
        resume_invalidate_downstream=bool(resume_section.get("invalidate_downstream", True)),
        promotion_atomic_required=bool(promotion_section.get("atomic_required", True)),
        promotion_allow_cross_device_copy=bool(promotion_section.get("allow_cross_device_copy", False)),
        promotion_preserve_existing_until_verified=bool(promotion_section.get("preserve_existing_until_verified", True)),
        verification_enabled=bool(verification_section.get("enabled", True)),
        verification_duration_tolerance_seconds=float(verification_section.get("duration_tolerance_seconds", 0.25)),
        verification_require_video_stream=bool(verification_section.get("require_video_stream", True)),
        verification_require_vertical_orientation=bool(verification_section.get("require_vertical_orientation", True)),
        verification_require_audio_when_planned=bool(verification_section.get("require_audio_when_planned", True)),
        verification_validator_profile=str(verification_section.get("validator_profile", "instagram_reel")),
        cleanup_verify_workspace_ownership=bool(cleanup_section.get("verify_workspace_ownership", True)),
        cleanup_preserve_reports=bool(cleanup_section.get("preserve_reports", True)),
        cleanup_preserve_logs=bool(cleanup_section.get("preserve_logs", True)),
        cleanup_preserve_state=bool(cleanup_section.get("preserve_state", True)),
        reporting_write_detailed_log=bool(reporting_section.get("write_detailed_log", True)),
        reporting_write_execution_report=bool(reporting_section.get("write_execution_report", True)),
        reporting_detailed_log_filename=str(reporting_section.get("detailed_log_filename", "renderer_execution_log.json")),
        reporting_report_filename=str(reporting_section.get("report_filename", "end_to_end_render_report.json")),
        reporting_atomic_write=bool(reporting_section.get("atomic_write", True)),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EndToEndRenderRequest:
    timeline_path: Path
    renderer_plan_path: Path
    output_path: Path
    overlay_plan_path: Path | None = None
    overlay_asset_manifest_path: Path | None = None
    workspace_path: Path | None = None
    resume: bool = False
    restart: bool = False
    keep_intermediates: bool = False
    force: bool = False
    dry_run: bool = True
    production_date: str | None = None


@dataclass(slots=True)
class RenderPassCheckpoint:
    pass_id: str = ""
    pass_type: str = ""
    status: str = PipelineStatus.PENDING
    input_paths: list[str] = field(default_factory=list)
    output_path: str | None = None
    started_at: str = ""
    finished_at: str = ""
    command_hash: str = ""
    output_checksum: str | None = None
    verified: bool = False
    reusable: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostic_log: str | None = None
    duration_seconds: float | None = None


@dataclass(slots=True)
class RenderWorkspace:
    workspace_path: Path
    intermediates_dir: Path
    final_dir: Path
    logs_dir: Path
    diagnostics_dir: Path
    state_path: Path
    manifest_path: Path
    ownership_marker_path: Path
    owned: bool = False


@dataclass(slots=True)
class EndToEndRenderState:
    schema_version: str = "1.0"
    pipeline_id: str = ""
    renderer_plan_id: str = ""
    timeline_id: str = ""
    overlay_plan_id: str | None = None
    overlay_asset_manifest_id: str | None = None
    execution_id: str = ""
    workspace_path: str = ""
    requested_output_path: str = ""
    started_at: str = ""
    updated_at: str = ""
    status: str = PipelineStatus.PENDING
    current_pass: str | None = None
    passes: list[RenderPassCheckpoint] = field(default_factory=list)
    candidate_output: str | None = None
    final_output: str | None = None
    verification: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class ResumeDecision:
    reused_pass_ids: list[str] = field(default_factory=list)
    invalidated_pass_ids: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    first_pass_to_execute: str | None = None


@dataclass(slots=True)
class FinalPromotionResult:
    promoted: bool = False
    candidate_path: str = ""
    final_path: str = ""
    candidate_checksum: str = ""
    final_checksum: str = ""
    candidate_size_bytes: int = 0
    final_size_bytes: int = 0
    strategy: str = ""
    backup_path: str | None = None


@dataclass(slots=True)
class CleanupResult:
    deleted: list[str] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    ownership_verified: bool = False


@dataclass(slots=True)
class PipelineWarning:
    code: str = ""
    message: str = ""
    pass_id: str | None = None


@dataclass(slots=True)
class EndToEndRenderResult:
    schema_version: str = "1.0"
    pipeline_id: str = ""
    renderer_plan_id: str = ""
    timeline_id: str = ""
    overlay_plan_id: str | None = None
    overlay_asset_manifest_id: str | None = None
    started_at: str = ""
    finished_at: str = ""
    status: str = ""
    resumed: bool = False
    reused_pass_count: int = 0
    executed_pass_count: int = 0
    passes: list[RenderPassCheckpoint] = field(default_factory=list)
    candidate_output: str | None = None
    final_output: str | None = None
    final_verified: bool = False
    promotion: FinalPromotionResult | None = None
    cleanup: CleanupResult | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def end_to_end_render_result_to_dict(result: EndToEndRenderResult) -> dict[str, Any]:
    return asdict(result)


# ---------------------------------------------------------------------------
# Deterministic identity
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, *, chunk_size: int = 1048576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _compute_pipeline_id(
    renderer_plan_id: str,
    timeline_id: str,
    overlay_plan_id: str | None,
    overlay_asset_manifest_id: str | None,
    output_path_canonical: str,
    config_schema_version: str,
) -> str:
    payload = {
        "renderer_plan_id": renderer_plan_id,
        "timeline_id": timeline_id,
        "overlay_plan_id": overlay_plan_id,
        "overlay_asset_manifest_id": overlay_asset_manifest_id,
        "output_path": output_path_canonical,
        "config_schema_version": config_schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _video_pass_source_paths(pass_: renderer_plan_engine.RendererPass, context: "renderer_execution_engine.RendererExecutionContext") -> list[str]:
    """
    The real source files a video pass depends on. For a
    LOSSLESS_COPY plan, video_engine.py's own -i argument points at a
    generated concat *manifest* file (which only embeds source PATHS
    as text, never their content) -- so scanning the resolved command
    for existing-file tokens alone would miss a source video being
    replaced in place at the same path. Read directly from the plan
    instead, which always names the real files regardless of which
    concat strategy gets chosen.
    """
    if pass_.pass_type != "video":
        return []
    plan = context.plan
    if not plan.video_tracks:
        return []
    asset_by_id = {asset.asset_id: asset for asset in plan.assets}
    return [
        asset_by_id[asset_id].source_path
        for asset_id in plan.video_tracks[0].asset_ids
        if asset_id in asset_by_id
    ]


def _compute_command_fingerprint(
    pass_id: str, pass_type: str, command: list[str] | None, config_schema_version: str,
    *, extra_source_paths: list[str] | None = None,
) -> str:
    """
    A command fingerprint alone only catches PATH-level changes (a
    different source file, a different font). It would miss the SAME
    path being overwritten with different bytes (source media, a font,
    an overlay asset, or a music file replaced in place) -- exactly the
    "if source media/subtitle/overlay asset/music changes, invalidate"
    requirement in Phase 11F.7 §11. Every command token that resolves
    to a real, existing file is therefore also checksummed and folded
    into the fingerprint (this alone already covers subtitle/overlay/
    music passes, whose real source paths always appear directly as
    literal command tokens), plus any `extra_source_paths` the caller
    names explicitly (needed only for the video pass -- see
    _video_pass_source_paths()).
    """
    referenced_file_checksums: dict[str, str] = {}
    for token in list(command or []) + list(extra_source_paths or []):
        try:
            candidate = Path(token)
            if candidate.is_file():
                referenced_file_checksums[token] = _sha256_file(candidate)
        except OSError:
            continue

    payload = {
        "pass_id": pass_id, "pass_type": pass_type,
        "command": list(command) if command is not None else None,
        "referenced_file_checksums": referenced_file_checksums,
        "config_schema_version": config_schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Request normalization / path safety
# ---------------------------------------------------------------------------


def _normalize_request(request: EndToEndRenderRequest) -> EndToEndRenderRequest:
    return dataclasses.replace(
        request,
        timeline_path=Path(request.timeline_path),
        renderer_plan_path=Path(request.renderer_plan_path),
        output_path=Path(request.output_path),
        overlay_plan_path=Path(request.overlay_plan_path) if request.overlay_plan_path else None,
        overlay_asset_manifest_path=Path(request.overlay_asset_manifest_path) if request.overlay_asset_manifest_path else None,
        workspace_path=Path(request.workspace_path) if request.workspace_path else None,
    )


def _check_path_collisions(request: EndToEndRenderRequest) -> None:
    named_paths: dict[str, Path] = {"timeline": request.timeline_path, "renderer-plan": request.renderer_plan_path}
    if request.overlay_plan_path:
        named_paths["overlay-plan"] = request.overlay_plan_path
    if request.overlay_asset_manifest_path:
        named_paths["overlay-asset-manifest"] = request.overlay_asset_manifest_path

    output_resolved = request.output_path.resolve()
    for name, path in named_paths.items():
        if path.resolve() == output_resolved:
            raise UnsafeFinalOutputError(f"--output must not be the same path as --{name}")


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def _resolve_workspace(request: EndToEndRenderRequest, config: EndToEndRenderConfig, pipeline_id: str) -> RenderWorkspace:
    if request.workspace_path is not None:
        root = request.workspace_path
    else:
        root = _runtime_root() / config.workspace_default_root / pipeline_id
    return RenderWorkspace(
        workspace_path=root,
        intermediates_dir=root / config.workspace_intermediates_directory,
        final_dir=root / config.workspace_final_directory,
        logs_dir=root / config.workspace_logs_directory,
        diagnostics_dir=root / config.workspace_diagnostics_directory,
        state_path=root / config.workspace_state_filename,
        manifest_path=root / config.workspace_manifest_filename,
        ownership_marker_path=root / config.workspace_ownership_marker,
    )


def _open_or_create_workspace(
    workspace: RenderWorkspace, pipeline_id: str, *, resume: bool, restart: bool,
) -> None:
    """
    Ensures `workspace.workspace_path` is safe to use. Creates it (and
    writes the ownership marker) if it doesn't exist yet. If it exists
    without the ownership marker, refuses unconditionally -- never
    touches a directory this module did not create. If it exists WITH
    the marker, requires the caller to have explicitly said --resume or
    --restart ("no reuse based only on filename existence").
    """
    root = workspace.workspace_path

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        workspace.intermediates_dir.mkdir(parents=True, exist_ok=True)
        workspace.final_dir.mkdir(parents=True, exist_ok=True)
        workspace.logs_dir.mkdir(parents=True, exist_ok=True)
        workspace.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        marker_payload = {"pipeline_id": pipeline_id, "created_at": _now_iso()}
        workspace.ownership_marker_path.write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True), encoding="utf-8",
        )
        workspace.owned = True
        return

    if not root.is_dir():
        raise UnsafeRenderWorkspaceError(f"Workspace path exists and is not a directory: {root}")

    if not workspace.ownership_marker_path.is_file():
        raise RenderWorkspaceOwnershipError(
            f"{root} exists but has no ownership marker ({workspace.ownership_marker_path.name}); "
            "refusing to touch a directory this module did not create."
        )

    if not resume and not restart:
        raise RenderResumeError(
            f"Workspace {root} already exists from a prior run; pass resume=True to continue it or "
            "restart=True to clear it and start over."
        )

    for directory in (workspace.intermediates_dir, workspace.final_dir, workspace.logs_dir, workspace.diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    workspace.owned = True


def _perform_restart(workspace: RenderWorkspace, config: EndToEndRenderConfig) -> CleanupResult:
    if not workspace.ownership_marker_path.is_file():
        raise RenderRestartError(f"Cannot restart {workspace.workspace_path}: no ownership marker present.")

    deleted: list[str] = []
    retained: list[str] = []
    failed: list[str] = []

    def _clear_directory(directory: Path) -> None:
        if not directory.is_dir():
            return
        for item in directory.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    deleted.append(str(item))
            except OSError:
                failed.append(str(item))

    _clear_directory(workspace.intermediates_dir)

    if workspace.state_path.is_file():
        try:
            workspace.state_path.unlink()
            deleted.append(str(workspace.state_path))
        except OSError:
            failed.append(str(workspace.state_path))

    if config.cleanup_preserve_logs:
        if workspace.logs_dir.is_dir():
            retained.append(str(workspace.logs_dir))
    else:
        _clear_directory(workspace.logs_dir)

    if config.cleanup_preserve_reports:
        if workspace.diagnostics_dir.is_dir():
            retained.append(str(workspace.diagnostics_dir))
    else:
        _clear_directory(workspace.diagnostics_dir)

    return CleanupResult(deleted=deleted, retained=retained, failed=failed, skipped=[], ownership_verified=True)


# ---------------------------------------------------------------------------
# Pass order enforcement
# ---------------------------------------------------------------------------


def _enforce_pass_order(plan: renderer_plan_engine.RendererPlan, config: EndToEndRenderConfig) -> None:
    present_types = [p.pass_type for p in plan.passes]

    if not config.allow_duplicate_pass_types:
        seen: set[str] = set()
        for pass_type in present_types:
            if pass_type in seen:
                raise RenderPassOrderError(f"Duplicate pass_type {pass_type!r} in renderer plan; not allowed by config.")
            seen.add(pass_type)

    if config.fail_on_unsupported_pass:
        for pass_type in present_types:
            if pass_type not in config.supported_pass_order:
                raise RenderPassOrderError(
                    f"Pass_type {pass_type!r} is not in config.supported_pass_order {config.supported_pass_order}."
                )

    cursor = -1
    for pass_type in present_types:
        if pass_type not in config.supported_pass_order:
            continue
        index = config.supported_pass_order.index(pass_type)
        if index <= cursor:
            raise RenderPassOrderError(
                f"Renderer plan pass order {present_types} does not match config.supported_pass_order "
                f"{config.supported_pass_order} (pass_type {pass_type!r} is out of order)."
            )
        cursor = index


def _intermediate_filename_for(pass_type: str, *, is_last: bool, config: EndToEndRenderConfig) -> str:
    if is_last:
        return config.intermediate_candidate_filename
    return {
        "video": config.intermediate_video_filename,
        "subtitle": config.intermediate_subtitle_filename,
        "overlay": config.intermediate_overlay_filename,
        "music": config.intermediate_music_filename,
    }[pass_type]


# ---------------------------------------------------------------------------
# Per-pass execution / resume
# ---------------------------------------------------------------------------


def _try_reuse_checkpoint(
    pass_: renderer_plan_engine.RendererPass,
    checkpoint: RenderPassCheckpoint,
    context: "renderer_execution_engine.RendererExecutionContext",
    *,
    video_input: Path | None,
    output_path: Path,
    runner: Any,
    config: EndToEndRenderConfig,
) -> tuple[bool, str]:
    """
    Reuse is allowed only when every check in Phase 11F.7 §11 passes:
    the recorded output still exists/is non-zero/checksum-matches, and
    a fresh dry-run of this exact pass (safe now -- its real upstream
    input already exists) resolves to the same command fingerprint.
    Never reuses based on filename existence alone. force=True on the
    dry-run call below is always safe here -- it only affects whether
    a real (non-dry-run) pass could overwrite `output_path`, and this
    path is always inside the workspace this module owns, never the
    externally-requested --output (whose own overwrite policy is
    request.force, enforced once up front in run_end_to_end_render()).
    """
    if checkpoint.output_path is None:
        return False, "no recorded output path"
    recorded_output = Path(checkpoint.output_path)
    if not recorded_output.is_file():
        return False, "recorded output missing"
    if recorded_output.stat().st_size == 0:
        return False, "recorded output is zero bytes"
    if not checkpoint.verified:
        return False, "checkpoint was never verified"

    if config.resume_require_checksum_match:
        current_checksum = _sha256_file(recorded_output)
        if current_checksum != checkpoint.output_checksum:
            return False, "output checksum mismatch"

    if config.resume_require_command_fingerprint_match:
        try:
            dry_result = renderer_execution_engine.execute_single_pass(
                pass_, context, video_input=video_input, output_path=output_path,
                dry_run=True, force=True, runner=runner,
            )
        except renderer_execution_engine.RendererExecutionEngineError as exc:
            return False, f"dry-run failed: {exc}"

        current_fingerprint = _compute_command_fingerprint(
            pass_.pass_id, pass_.pass_type, dry_result.command, config.schema_version,
            extra_source_paths=_video_pass_source_paths(pass_, context),
        )
        if current_fingerprint != checkpoint.command_hash:
            return False, "command fingerprint changed"

    if recorded_output != output_path:
        try:
            shutil.copy2(recorded_output, output_path)
        except OSError as exc:
            return False, f"could not stage reused output at {output_path}: {exc}"

    return True, ""


def _execute_and_checkpoint(
    pass_: renderer_plan_engine.RendererPass,
    context: "renderer_execution_engine.RendererExecutionContext",
    *,
    video_input: Path | None,
    output_path: Path,
    dry_run: bool,
    runner: Any,
    config: EndToEndRenderConfig,
) -> RenderPassCheckpoint:
    """force=True below is always safe: `output_path` is always inside
    the workspace this module owns (a stale intermediate from an
    earlier, now-invalidated attempt at this same pass is exactly what
    a real re-execution is meant to overwrite) -- never the externally
    requested --output, whose overwrite policy is enforced once, up
    front, in run_end_to_end_render()."""
    pass_started = _now_iso()
    try:
        result = renderer_execution_engine.execute_single_pass(
            pass_, context, video_input=video_input, output_path=output_path,
            dry_run=dry_run, force=True, runner=runner,
        )
    except renderer_execution_engine.RendererExecutionEngineError as exc:
        raise RenderPassExecutionError(f"Pass {pass_.pass_id} ({pass_.pass_type}) execution failed: {exc}") from exc

    finished = _now_iso()
    fingerprint = _compute_command_fingerprint(
        pass_.pass_id, pass_.pass_type, result.command, config.schema_version,
        extra_source_paths=_video_pass_source_paths(pass_, context),
    )

    checksum = None
    verified = False
    if not dry_run:
        if not result.output_path:
            raise RenderPassVerificationError(f"Pass {pass_.pass_id} reported success but produced no output_path.")
        real_output = Path(result.output_path)
        if not real_output.is_file() or real_output.stat().st_size == 0:
            raise RenderPassVerificationError(
                f"Pass {pass_.pass_id} reported {result.status!r} but output {real_output} is missing or empty."
            )
        checksum = _sha256_file(real_output)
        verified = True

    return RenderPassCheckpoint(
        pass_id=pass_.pass_id, pass_type=pass_.pass_type,
        status=PipelineStatus.SUCCEEDED if not dry_run else PipelineStatus.PENDING,
        input_paths=[str(video_input)] if video_input else [],
        output_path=result.output_path, started_at=pass_started, finished_at=finished,
        command_hash=fingerprint, output_checksum=checksum, verified=verified, reusable=False,
        error=result.error, warnings=list(result.warnings), diagnostic_log=result.diagnostic_log,
        duration_seconds=result.duration_seconds,
    )


# ---------------------------------------------------------------------------
# Final verification — injectable, defaults to media_inspector.py /
# video_validator.py, both already-implemented and reused as-is.
# ---------------------------------------------------------------------------


def _verify_candidate(
    candidate_path: Path,
    plan: renderer_plan_engine.RendererPlan,
    timeline: Any,
    config: EndToEndRenderConfig,
    *,
    inspector: Callable[[Path], "media_inspector.MediaInfo"] | None,
    validator: Callable[["media_inspector.MediaInfo", Any], Any] | None,
) -> dict[str, Any]:
    if not candidate_path.is_file():
        raise FinalCandidateVerificationError(f"Candidate output missing: {candidate_path}")
    if candidate_path.stat().st_size == 0:
        raise FinalCandidateVerificationError(f"Candidate output is zero bytes: {candidate_path}")

    effective_inspector = inspector or (
        lambda path: media_inspector.inspect_file(path, media_inspector.load_inspector_config())
    )
    media_info = effective_inspector(candidate_path)

    checks: list[str] = []
    failures: list[str] = []

    checks.append("video_stream")
    if config.verification_require_video_stream and not getattr(media_info, "has_video", False):
        failures.append("no video stream present")

    checks.append("duration")
    duration = getattr(media_info, "duration_seconds", None)
    timeline_duration = getattr(timeline, "duration_seconds", None)
    if duration is not None and timeline_duration is not None:
        if abs(duration - timeline_duration) > config.verification_duration_tolerance_seconds:
            failures.append(f"duration {duration} outside tolerance of timeline duration {timeline_duration}")

    checks.append("orientation")
    if config.verification_require_vertical_orientation:
        is_vertical = getattr(media_info, "is_vertical", None)
        if is_vertical is False:
            failures.append("output is not vertically oriented")

    checks.append("audio_presence")
    has_planned_music = bool(getattr(plan, "music_tracks", None))
    if config.verification_require_audio_when_planned and has_planned_music:
        if not getattr(media_info, "has_audio", False):
            failures.append("planned audio track missing from output")

    validator_result_dict: dict[str, Any] | None = None
    effective_validator = validator or video_validator.validate_media
    try:
        validator_config = video_validator.load_validator_config()
        profile = validator_config.profiles[config.verification_validator_profile]
        validator_result = effective_validator(media_info, profile)
        validator_result_dict = {
            "passed": validator_result.passed,
            "score": validator_result.score,
            "summary": validator_result.summary,
            "failed_checks": [asdict(c) for c in validator_result.failed_checks],
            "warnings": [asdict(c) for c in validator_result.warnings],
        }
        if not validator_result.passed:
            if validator_result.failed_checks:
                failures.extend(f"validator: {c.message}" for c in validator_result.failed_checks)
            else:
                failures.append(f"validator reported failed (score={validator_result.score})")
    except KeyError:
        failures.append(f"unknown validator profile {config.verification_validator_profile!r}")
    except video_validator.VideoValidatorError as exc:
        failures.append(f"validator check failed: {exc}")
    checks.append("validator")

    passed = not failures
    return {
        "passed": passed,
        "checks": checks,
        "failures": failures,
        "summary": "PASSED" if passed else f"FAILED: {'; '.join(failures)}",
        "validator": validator_result_dict,
    }


# ---------------------------------------------------------------------------
# Final promotion — atomic Path.replace(), documented cross-device
# fallback only when explicitly configured.
# ---------------------------------------------------------------------------


def _promote_candidate(candidate_path: Path, output_path: Path, config: EndToEndRenderConfig) -> FinalPromotionResult:
    candidate_checksum = _sha256_file(candidate_path)
    candidate_size = candidate_path.stat().st_size

    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategy = "atomic_replace"
    try:
        candidate_path.replace(output_path)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            if not config.promotion_allow_cross_device_copy:
                raise CrossDevicePromotionError(
                    f"Promoting {candidate_path} to {output_path} would cross a filesystem boundary; "
                    "set promotion.allow_cross_device_copy to enable an explicit copy fallback."
                ) from exc
            strategy = "cross_device_copy"
            shutil.copy2(candidate_path, output_path)
            with open(output_path, "rb") as file_handle:
                os.fsync(file_handle.fileno())
            copied_checksum = _sha256_file(output_path)
            if copied_checksum != candidate_checksum:
                raise FinalPromotionError(
                    f"Cross-device copy checksum mismatch promoting {candidate_path} to {output_path}."
                )
            candidate_path.unlink(missing_ok=True)
        else:
            raise FinalPromotionError(f"Failed to promote {candidate_path} to {output_path}: {exc}") from exc

    if not output_path.is_file():
        raise FinalPromotionError(f"Promotion reported success but final output is missing: {output_path}")
    final_checksum = _sha256_file(output_path)
    final_size = output_path.stat().st_size
    if final_checksum != candidate_checksum:
        raise FinalPromotionError(f"Post-promotion checksum mismatch for {output_path}.")

    return FinalPromotionResult(
        promoted=True, candidate_path=str(candidate_path), final_path=str(output_path),
        candidate_checksum=candidate_checksum, final_checksum=final_checksum,
        candidate_size_bytes=candidate_size, final_size_bytes=final_size, strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def _cleanup_workspace(
    workspace: RenderWorkspace, config: EndToEndRenderConfig, checkpoints: list[RenderPassCheckpoint],
    *, keep_intermediates: bool, success: bool,
) -> CleanupResult:
    ownership_verified = workspace.ownership_marker_path.is_file()
    if config.cleanup_verify_workspace_ownership and not ownership_verified:
        return CleanupResult(deleted=[], retained=[], failed=[], skipped=["workspace"], ownership_verified=False)

    deleted: list[str] = []
    retained: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    if not success or keep_intermediates:
        for checkpoint in checkpoints:
            if checkpoint.output_path:
                retained.append(checkpoint.output_path)
        return CleanupResult(deleted=deleted, retained=retained, failed=failed, skipped=skipped, ownership_verified=ownership_verified)

    for checkpoint in checkpoints:
        if not checkpoint.output_path:
            continue
        path = Path(checkpoint.output_path)
        try:
            if path.is_file():
                path.unlink()
                deleted.append(str(path))
            else:
                skipped.append(str(path))
        except OSError:
            failed.append(str(path))

    return CleanupResult(deleted=deleted, retained=retained, failed=failed, skipped=skipped, ownership_verified=ownership_verified)


# ---------------------------------------------------------------------------
# State — atomic write, same convention as every save_*() in this repo.
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any], *, error_cls: type[EndToEndRenderError]) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise error_cls(f"Failed to write {path}: {exc}") from exc
    return path


def _build_state(
    pipeline_id: str, plan: renderer_plan_engine.RendererPlan, context: "renderer_execution_engine.RendererExecutionContext",
    workspace: RenderWorkspace, request: EndToEndRenderRequest, started_at: str, checkpoints: list[RenderPassCheckpoint],
    *, status: str, current_pass: str | None, error: str | None = None, warnings: list[str] | None = None,
    verification: dict[str, Any] | None = None, candidate_output: str | None = None, final_output: str | None = None,
) -> EndToEndRenderState:
    return EndToEndRenderState(
        pipeline_id=pipeline_id, renderer_plan_id=plan.renderer_plan_id, timeline_id=plan.timeline_id,
        overlay_plan_id=plan.overlay_plan_id,
        overlay_asset_manifest_id=context.overlay_asset_manifest.manifest_id if context.overlay_asset_manifest else None,
        execution_id=hashlib.sha256(f"{pipeline_id}:{started_at}".encode("utf-8")).hexdigest()[:16],
        workspace_path=str(workspace.workspace_path), requested_output_path=str(request.output_path),
        started_at=started_at, updated_at=_now_iso(), status=status, current_pass=current_pass,
        passes=list(checkpoints), candidate_output=candidate_output, final_output=final_output,
        verification=verification, warnings=list(warnings or []), error=error,
    )


def _save_state(workspace: RenderWorkspace, state: EndToEndRenderState) -> None:
    _atomic_write_json(workspace.state_path, asdict(state), error_cls=RenderStateError)


def _load_state(workspace: RenderWorkspace) -> EndToEndRenderState:
    if not workspace.state_path.is_file():
        raise RenderStateError(f"No state.json found in workspace {workspace.workspace_path}; cannot resume.")
    try:
        data = json.loads(workspace.state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RenderStateError(f"Malformed state.json in workspace {workspace.workspace_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RenderStateError(f"state.json root must be an object: {workspace.state_path}")

    checkpoint_fields = {f.name for f in dataclasses.fields(RenderPassCheckpoint)}
    try:
        checkpoints = [
            RenderPassCheckpoint(**{key: value for key, value in entry.items() if key in checkpoint_fields})
            for entry in data.get("passes", [])
        ]
        return EndToEndRenderState(
            schema_version=data.get("schema_version", "1.0"), pipeline_id=data.get("pipeline_id", ""),
            renderer_plan_id=data.get("renderer_plan_id", ""), timeline_id=data.get("timeline_id", ""),
            overlay_plan_id=data.get("overlay_plan_id"), overlay_asset_manifest_id=data.get("overlay_asset_manifest_id"),
            execution_id=data.get("execution_id", ""), workspace_path=data.get("workspace_path", ""),
            requested_output_path=data.get("requested_output_path", ""), started_at=data.get("started_at", ""),
            updated_at=data.get("updated_at", ""), status=data.get("status", PipelineStatus.PENDING),
            current_pass=data.get("current_pass"), passes=checkpoints,
            candidate_output=data.get("candidate_output"), final_output=data.get("final_output"),
            verification=data.get("verification"), warnings=list(data.get("warnings", [])), error=data.get("error"),
        )
    except (TypeError, KeyError) as exc:
        raise RenderStateError(f"Malformed state.json in workspace {workspace.workspace_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Reporting — two atomic JSON writes.
# ---------------------------------------------------------------------------


def _write_reports(
    workspace: RenderWorkspace, config: EndToEndRenderConfig, checkpoints: list[RenderPassCheckpoint],
    result: EndToEndRenderResult,
) -> None:
    if config.reporting_write_detailed_log:
        detailed = renderer_execution_engine.RendererExecutionResult(
            renderer_plan_id=result.renderer_plan_id, timeline_id=result.timeline_id,
            dry_run=(result.final_output is None and result.candidate_output is None and result.status != PipelineStatus.FAILED),
            started_at=result.started_at, finished_at=result.finished_at,
            passes=[
                renderer_execution_engine.PassExecutionResult(
                    pass_id=c.pass_id, pass_type=c.pass_type,
                    status="reused" if c.reusable else ("dry_run" if c.status == PipelineStatus.PENDING else "executed"),
                    output_path=c.output_path, duration_seconds=c.duration_seconds,
                    warnings=list(c.warnings), error=c.error, diagnostic_log=c.diagnostic_log,
                )
                for c in checkpoints
            ],
            final_output_path=result.final_output, warnings=list(result.warnings),
            errors=[result.error] if result.error else [], result=result.status,
        )
        log_path = workspace.logs_dir / config.reporting_detailed_log_filename
        _atomic_write_json(
            log_path, renderer_execution_engine.renderer_execution_result_to_dict(detailed), error_cls=RenderReportError,
        )

    if config.reporting_write_execution_report:
        report_path = workspace.diagnostics_dir / config.reporting_report_filename
        _atomic_write_json(report_path, end_to_end_render_result_to_dict(result), error_cls=RenderReportError)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_end_to_end_render(
    request: EndToEndRenderRequest,
    config: EndToEndRenderConfig,
    *,
    runner: Any = None,
    subtitle_render_config: Any = None,
    overlay_render_config: Any = None,
    overlay_asset_config: Any = None,
    overlay_asset_inspector: Any = None,
    final_inspector: Callable[[Path], "media_inspector.MediaInfo"] | None = None,
    final_validator: Callable[["media_inspector.MediaInfo", Any], Any] | None = None,
) -> EndToEndRenderResult:
    """
    Loads/validates/probes/resolves exactly once (via
    renderer_execution_engine.prepare_execution_context() -- never
    duplicated here), then drives each pass through
    renderer_execution_engine.execute_single_pass() with resume-aware
    reuse checks, writes every intermediate inside an isolated
    workspace, verifies the final candidate, atomically promotes it to
    `request.output_path`, cleans up, and writes both reports. Raises
    on any failure (matching execute_renderer_plan()'s own convention)
    -- a failed run never returns an EndToEndRenderResult with
    status="failed"; callers catch EndToEndRenderError instead.
    """
    request = _normalize_request(request)
    _check_path_collisions(request)

    if request.resume and request.restart:
        raise EndToEndRenderRequestError("resume and restart are mutually exclusive.")

    if not request.dry_run and request.output_path.exists() and not request.force:
        raise FinalOutputExistsError(f"{request.output_path} already exists; pass force=True to overwrite.")

    started_at = _now_iso()

    try:
        peeked_plan = renderer_plan_engine.load_renderer_plan(request.renderer_plan_path)
    except renderer_plan_engine.RendererPlanEngineError as exc:
        raise EndToEndRenderRequestError(f"Failed to load renderer plan {request.renderer_plan_path}: {exc}") from exc

    if peeked_plan.overlay_plan_id is not None and request.overlay_plan_path is None:
        raise EndToEndRenderRequestError(
            f"Renderer plan references overlay_plan_id {peeked_plan.overlay_plan_id!r} but no "
            "overlay_plan_path was given."
        )

    peeked_manifest_id: str | None = None
    if request.overlay_asset_manifest_path is not None:
        try:
            peeked_manifest_id = overlay_asset_resolver.load_overlay_asset_manifest(
                request.overlay_asset_manifest_path
            ).manifest_id
        except overlay_asset_resolver.OverlayAssetResolverError as exc:
            raise EndToEndRenderRequestError(
                f"Failed to load overlay asset manifest {request.overlay_asset_manifest_path}: {exc}"
            ) from exc

    pipeline_id = _compute_pipeline_id(
        peeked_plan.renderer_plan_id, peeked_plan.timeline_id, peeked_plan.overlay_plan_id,
        peeked_manifest_id, str(request.output_path.resolve()), config.schema_version,
    )

    workspace = _resolve_workspace(request, config, pipeline_id)
    _open_or_create_workspace(workspace, pipeline_id, resume=request.resume, restart=request.restart)

    if request.restart:
        _perform_restart(workspace, config)

    prior_state: EndToEndRenderState | None = None
    if request.resume and workspace.state_path.is_file():
        prior_state = _load_state(workspace)
        if prior_state.pipeline_id != pipeline_id:
            raise RenderResumeError(
                f"Workspace state pipeline_id {prior_state.pipeline_id!r} does not match this "
                f"request's pipeline_id {pipeline_id!r}; refusing to resume a mismatched workspace."
            )

    try:
        context = renderer_execution_engine.prepare_execution_context(
            request.renderer_plan_path, request.timeline_path, request.overlay_plan_path,
            runner=runner, subtitle_render_config=subtitle_render_config,
            overlay_render_config=overlay_render_config, overlay_asset_config=overlay_asset_config,
            overlay_asset_inspector=overlay_asset_inspector,
        )
    except renderer_execution_engine.RendererExecutionEngineError as exc:
        raise EndToEndRenderRequestError(f"Failed to prepare execution context: {exc}") from exc

    plan = context.plan
    _enforce_pass_order(plan, config)

    producing_types = ("video", "subtitle", "overlay", "music")
    producing_passes = [p for p in plan.passes if p.pass_type in producing_types]

    # Unsupported pass fails before any media write: checked for every
    # producing pass up front, mirroring execute_renderer_plan()'s own
    # pre-flight check, before touching anything.
    for pass_ in producing_passes:
        if pass_.pass_type in ("subtitle", "overlay") and not renderer_execution_engine.is_pass_supported(pass_, plan):
            raise RenderPassExecutionError(
                f"Pass {pass_.pass_id} ({pass_.pass_type}) is not executable by this renderer plan's "
                "renderer hint; refusing to start any pass execution."
            )

    checkpoints_by_id: dict[str, RenderPassCheckpoint] = (
        {c.pass_id: c for c in prior_state.passes} if prior_state is not None else {}
    )

    checkpoints: list[RenderPassCheckpoint] = []
    current_input: Path | None = None
    reused_count = 0
    executed_count = 0
    invalidated_from_here = False
    warnings: list[str] = []

    try:
        for index, pass_ in enumerate(producing_passes):
            is_last = index == len(producing_passes) - 1
            target = workspace.intermediates_dir / _intermediate_filename_for(pass_.pass_type, is_last=is_last, config=config)

            prior_checkpoint = checkpoints_by_id.get(pass_.pass_id)
            reuse = False

            if (
                not invalidated_from_here
                and config.resume_enabled
                and request.resume
                and prior_checkpoint is not None
                and prior_checkpoint.status == PipelineStatus.SUCCEEDED
            ):
                reuse, reason = _try_reuse_checkpoint(
                    pass_, prior_checkpoint, context, video_input=current_input, output_path=target,
                    runner=runner, config=config,
                )
                if not reuse:
                    invalidated_from_here = True
                    warnings.append(f"Pass {pass_.pass_id}: cannot reuse checkpoint ({reason}); re-executing from here.")
            elif prior_checkpoint is not None:
                invalidated_from_here = True

            if reuse:
                checkpoint = dataclasses.replace(prior_checkpoint, output_path=str(target), reusable=True)
                reused_count += 1
            else:
                checkpoint = _execute_and_checkpoint(
                    pass_, context, video_input=current_input, output_path=target,
                    dry_run=request.dry_run, runner=runner, config=config,
                )
                executed_count += 1

            checkpoints.append(checkpoint)
            if not request.dry_run:
                current_input = target

            _save_state(
                workspace,
                _build_state(
                    pipeline_id, plan, context, workspace, request, started_at, checkpoints,
                    status=PipelineStatus.RUNNING, current_pass=pass_.pass_id, warnings=warnings,
                ),
            )

        if request.dry_run:
            result = EndToEndRenderResult(
                pipeline_id=pipeline_id, renderer_plan_id=plan.renderer_plan_id, timeline_id=plan.timeline_id,
                overlay_plan_id=plan.overlay_plan_id, overlay_asset_manifest_id=peeked_manifest_id,
                started_at=started_at, finished_at=_now_iso(), status=PipelineStatus.SUCCEEDED,
                resumed=bool(request.resume), reused_pass_count=reused_count, executed_pass_count=executed_count,
                passes=checkpoints, candidate_output=None, final_output=None, final_verified=False,
                warnings=warnings,
            )
            _write_reports(workspace, config, checkpoints, result)
            return result

        candidate_path = current_input
        candidate_final_path = workspace.intermediates_dir / config.intermediate_candidate_filename
        if candidate_path is not None and candidate_path != candidate_final_path:
            candidate_path.replace(candidate_final_path)
            for checkpoint in checkpoints:
                if checkpoint.output_path == str(candidate_path):
                    checkpoint.output_path = str(candidate_final_path)
        candidate_path = candidate_final_path

        verification_dict: dict[str, Any] | None = None
        if config.verification_enabled:
            verification_dict = _verify_candidate(
                candidate_path, plan, context.timeline, config, inspector=final_inspector, validator=final_validator,
            )
            if not verification_dict["passed"]:
                raise FinalCandidateVerificationError(
                    f"Candidate {candidate_path} failed final verification: {verification_dict['summary']}"
                )

        promotion = _promote_candidate(candidate_path, request.output_path, config)

        cleanup = _cleanup_workspace(
            workspace, config, checkpoints, keep_intermediates=request.keep_intermediates, success=True,
        )

        result = EndToEndRenderResult(
            pipeline_id=pipeline_id, renderer_plan_id=plan.renderer_plan_id, timeline_id=plan.timeline_id,
            overlay_plan_id=plan.overlay_plan_id, overlay_asset_manifest_id=peeked_manifest_id,
            started_at=started_at, finished_at=_now_iso(), status=PipelineStatus.PROMOTED,
            resumed=bool(request.resume), reused_pass_count=reused_count, executed_pass_count=executed_count,
            passes=checkpoints, candidate_output=str(candidate_path), final_output=str(request.output_path),
            final_verified=bool(verification_dict is None or verification_dict["passed"]),
            promotion=promotion, cleanup=cleanup, warnings=warnings,
        )

        final_state = _build_state(
            pipeline_id, plan, context, workspace, request, started_at, checkpoints,
            status=PipelineStatus.PROMOTED, current_pass=None, warnings=warnings,
            verification=verification_dict, candidate_output=None, final_output=str(request.output_path),
        )
        _save_state(workspace, final_state)
        _write_reports(workspace, config, checkpoints, result)
        return result

    except EndToEndRenderError as exc:
        error_message = str(exc)
        failure_state = _build_state(
            pipeline_id, plan, context, workspace, request, started_at, checkpoints,
            status=PipelineStatus.FAILED,
            current_pass=(checkpoints[-1].pass_id if checkpoints else None),
            error=error_message, warnings=warnings,
        )
        try:
            _save_state(workspace, failure_state)
        except RenderStateError:
            pass

        failure_result = EndToEndRenderResult(
            pipeline_id=pipeline_id, renderer_plan_id=plan.renderer_plan_id, timeline_id=plan.timeline_id,
            overlay_plan_id=plan.overlay_plan_id, overlay_asset_manifest_id=peeked_manifest_id,
            started_at=started_at, finished_at=_now_iso(), status=PipelineStatus.FAILED,
            resumed=bool(request.resume), reused_pass_count=reused_count, executed_pass_count=executed_count,
            passes=checkpoints, candidate_output=None, final_output=None, final_verified=False,
            warnings=warnings, error=error_message,
        )
        try:
            _write_reports(workspace, config, checkpoints, failure_result)
        except RenderReportError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO End-to-End Render Pipeline (Phase 11F.7). Orchestrates the "
            "existing video/subtitle/overlay/music/final_encode pass chain "
            "(via renderer_execution_engine.py) into one retry-safe, "
            "deterministic render: workspace, resume/retry, atomic final "
            "promotion, verification, cleanup, and a full execution report."
        )
    )

    parser.add_argument("--timeline", required=True, help="Path to the source timeline.json.")
    parser.add_argument("--renderer-plan", dest="renderer_plan", required=True, help="Path to the source renderer_plan.json.")
    parser.add_argument("--overlay-plan", dest="overlay_plan", default=None, help="Path to the source overlay_plan.json.")
    parser.add_argument(
        "--overlay-asset-manifest", dest="overlay_asset_manifest", default=None,
        help="Path to an already-built overlay_asset_manifest.json (used for identity/report metadata).",
    )
    parser.add_argument("--output", required=True, help="Path to write the final rendered video (e.g. reel_final.mp4).")
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/end_to_end_render.yaml.")
    parser.add_argument("--workspace", default=None, help="Explicit workspace directory (default: deterministic path under output/render_workspaces/).")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", help="Continue a prior workspace, reusing verified passes.")
    resume_group.add_argument("--restart", action="store_true", help="Clear a prior owned workspace and start over.")

    parser.add_argument("--keep-intermediates", dest="keep_intermediates", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")

    return parser.parse_args(argv)


def _print_result(result: EndToEndRenderResult) -> None:
    print()
    print("AIKO End-to-End Render Pipeline (Phase 11F.7)")
    print("--------------------------------------------------")
    print(f"pipeline_id:      {result.pipeline_id}")
    print(f"status:           {result.status}")
    print(f"resumed:          {result.resumed}")
    print(f"reused/executed:  {result.reused_pass_count}/{result.executed_pass_count}")
    for checkpoint in result.passes:
        reused_marker = " [reused]" if checkpoint.reusable else ""
        print(f"  pass {checkpoint.pass_id} ({checkpoint.pass_type}): {checkpoint.status}{reused_marker}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    if result.error:
        print(f"error:   {result.error}")
    print(f"final_output:     {result.final_output}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_end_to_end_render_config(arguments.config)
        request = EndToEndRenderRequest(
            timeline_path=Path(arguments.timeline), renderer_plan_path=Path(arguments.renderer_plan),
            output_path=Path(arguments.output),
            overlay_plan_path=Path(arguments.overlay_plan) if arguments.overlay_plan else None,
            overlay_asset_manifest_path=Path(arguments.overlay_asset_manifest) if arguments.overlay_asset_manifest else None,
            workspace_path=Path(arguments.workspace) if arguments.workspace else None,
            resume=arguments.resume, restart=arguments.restart, keep_intermediates=arguments.keep_intermediates,
            force=arguments.force, dry_run=arguments.dry_run,
        )
        result = run_end_to_end_render(request, config)

        if arguments.as_json:
            print(json.dumps(end_to_end_render_result_to_dict(result), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_result(result)
    except EndToEndRenderError as exc:
        print(f"[EndToEndRenderPipeline] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
