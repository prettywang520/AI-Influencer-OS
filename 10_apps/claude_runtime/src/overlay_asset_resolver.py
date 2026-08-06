from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from . import overlay_plan_engine, renderer_plan_engine

# Phase 11F.4 — Overlay Asset Resolver. A renderer-neutral adapter that
# converts asset references already sitting in an OverlayPlan/
# RendererPlan into validated, typed OverlayRenderAsset records: safe
# local-path resolution, image metadata via an injectable inspector,
# alpha/dimension policy checks, checksums, and a deterministic
# manifest. This module never renders anything, never calls ffmpeg,
# never calls ffprobe (directly or indirectly), never builds a filter
# graph, never resizes/converts/downloads an asset, never mutates the
# OverlayPlan/RendererPlan it reads, and never picks a replacement
# asset on the caller's behalf.

DEFAULT_OVERLAY_ASSET_CONFIG_RELATIVE_PATH = Path("config") / "video" / "overlay_assets.yaml"


def _runtime_root() -> Path:
    """
    overlay_asset_resolver.py location:
    10_apps/claude_runtime/src/overlay_asset_resolver.py
    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OverlayAssetResolverError(RuntimeError):
    """Base error for the Phase 11F.4 Overlay Asset Resolver."""


class OverlayAssetConfigError(OverlayAssetResolverError):
    """Raised when config/video/overlay_assets.yaml is missing or invalid."""


class OverlayAssetReferenceError(OverlayAssetResolverError):
    """Raised for a structurally invalid asset reference, or a failure
    loading/validating a source OverlayPlan/RendererPlan."""


class OverlayAssetNotFoundError(OverlayAssetResolverError):
    """Raised when an overlay asset path does not exist."""


class OverlayAssetEmptyError(OverlayAssetResolverError):
    """Raised when an overlay asset file is zero bytes."""


class UnsupportedOverlayAssetError(OverlayAssetResolverError):
    """Raised when an overlay asset's extension is not supported (or is
    explicitly disabled, e.g. .gif/.svg by default)."""


class UnsafeOverlayAssetPathError(OverlayAssetResolverError):
    """Raised for a directory, a rejected symlink, a hidden file when
    disallowed, or any other path-safety violation."""


class OverlayAssetInspectorUnavailableError(OverlayAssetResolverError):
    """Raised when no image inspector is available and one is required."""


class OverlayAssetInspectionError(OverlayAssetResolverError):
    """Raised when the injected image inspector raises while inspecting
    an asset -- the original message is preserved."""


class OverlayAssetAlphaError(OverlayAssetResolverError):
    """Raised when an asset fails its role's alpha-channel policy."""


class OverlayAssetDimensionError(OverlayAssetResolverError):
    """Raised when an asset fails its role's dimension/aspect-ratio policy."""


class OverlayAssetAnimatedError(OverlayAssetResolverError):
    """Raised when an animated asset is rejected by config."""


class OverlayAssetChecksumError(OverlayAssetResolverError):
    """Raised when a checksum cannot be computed safely."""


class OverlayAssetConflictError(OverlayAssetResolverError):
    """Raised when two references to the same (path, role) disagree on
    a hard requirement (e.g. explicit alpha requirement) and config
    does not permit silently picking one."""


class OverlayAssetManifestJSONError(OverlayAssetResolverError):
    """Raised when manifest JSON is malformed or missing required fields."""


class OverlayAssetManifestExistsError(OverlayAssetResolverError):
    """Raised when the manifest output path already exists and --force
    was not given."""


class UnsafeOverlayAssetManifestPathError(OverlayAssetResolverError):
    """Raised when the manifest output path is a directory, or resolves
    to the same path as a source plan file."""


# ---------------------------------------------------------------------------
# String-constant "enums"
# ---------------------------------------------------------------------------


class OverlayAssetType:
    PNG = "png"
    WEBP = "webp"
    JPEG = "jpeg"
    GIF = "gif"
    SVG = "svg"
    UNKNOWN = "unknown"
    ALL = (PNG, WEBP, JPEG, GIF, SVG, UNKNOWN)


_EXTENSION_TO_TYPE: dict[str, str] = {
    ".png": OverlayAssetType.PNG,
    ".webp": OverlayAssetType.WEBP,
    ".jpg": OverlayAssetType.JPEG,
    ".jpeg": OverlayAssetType.JPEG,
    ".gif": OverlayAssetType.GIF,
    ".svg": OverlayAssetType.SVG,
}


def _classify_extension(raw_path: str) -> str:
    return _EXTENSION_TO_TYPE.get(Path(raw_path).suffix.lower(), OverlayAssetType.UNKNOWN)


class OverlayLogicalRole:
    LOGO = "logo"
    WATERMARK = "watermark"
    STICKER = "sticker"
    CTA = "cta"
    LOCATION = "location"
    DECORATIVE = "decorative"
    CUSTOM = "custom"
    ALL = (LOGO, WATERMARK, STICKER, CTA, LOCATION, DECORATIVE, CUSTOM)


class AlphaMode:
    NONE = "none"
    STRAIGHT = "straight"
    PREMULTIPLIED = "premultiplied"
    UNKNOWN = "unknown"
    ALL = (NONE, STRAIGHT, PREMULTIPLIED, UNKNOWN)


class AssetResolutionStatus:
    RESOLVED = "resolved"
    MISSING = "missing"
    INVALID = "invalid"
    ALL = (RESOLVED, MISSING, INVALID)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RoleDimensionPolicy:
    minimum_width: int | None = None
    minimum_height: int | None = None
    maximum_width: int | None = None
    maximum_height: int | None = None
    maximum_megapixels: float | None = None
    maximum_aspect_ratio: float | None = None
    minimum_aspect_ratio: float | None = None
    preferred_aspect_ratio_min: float | None = None
    preferred_aspect_ratio_max: float | None = None


@dataclass(slots=True)
class OverlayAssetResolverConfig:
    schema_version: str = "1.0"
    project_root: str = "."
    allow_absolute_paths: bool = True
    follow_symlinks: bool = True
    reject_hidden_files: bool = False
    fail_on_missing_required_asset: bool = True
    fail_on_invalid_optional_asset: bool = False

    supported_extensions_enabled: tuple[str, ...] = (".png", ".webp", ".jpg", ".jpeg")
    supported_extensions_disabled: tuple[str, ...] = (".gif", ".svg")

    inspection_backend: str = "pillow"
    require_inspector: bool = True
    allow_unknown_alpha: bool = False
    reject_animated_assets: bool = True

    alpha_roles_requiring_alpha: tuple[str, ...] = (
        OverlayLogicalRole.LOGO, OverlayLogicalRole.WATERMARK, OverlayLogicalRole.STICKER,
    )
    alpha_missing_alpha_policy: dict[str, str] = field(default_factory=dict)

    dimension_policies: dict[str, RoleDimensionPolicy] = field(default_factory=dict)

    duplicate_same_path_same_role: str = "merge_references"
    duplicate_same_path_different_role: str = "separate_assets"
    duplicate_conflicting_requirements: str = "error"
    duplicate_include_source_overlay_in_asset_id: bool = False

    checksum_enabled: bool = True
    checksum_algorithm: str = "sha256"
    checksum_chunk_size_bytes: int = 1048576

    overwrite_requires_force: bool = True
    atomic_write: bool = True
    manifest_filename: str = "overlay_asset_manifest.json"

    def dimension_policy(self, role: str) -> RoleDimensionPolicy:
        default = self.dimension_policies.get("default") or RoleDimensionPolicy()
        role_policy = self.dimension_policies.get(role)
        if role_policy is None:
            return default
        merged = dataclasses.replace(default)
        for policy_field in dataclasses.fields(RoleDimensionPolicy):
            value = getattr(role_policy, policy_field.name)
            if value is not None:
                setattr(merged, policy_field.name, value)
        return merged


def default_overlay_asset_config_path() -> Path:
    return _runtime_root() / DEFAULT_OVERLAY_ASSET_CONFIG_RELATIVE_PATH


def load_overlay_asset_config(config_path: str | Path | None = None) -> OverlayAssetResolverConfig:
    """Load config/video/overlay_assets.yaml (or an alternate path) into
    an OverlayAssetResolverConfig. Raises OverlayAssetConfigError if the
    file is missing or invalid."""
    path = Path(config_path) if config_path else default_overlay_asset_config_path()

    if not path.exists():
        raise OverlayAssetConfigError(f"Overlay asset config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise OverlayAssetConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise OverlayAssetConfigError(f"Overlay asset config is empty or invalid: {path}")

    resolver_section = raw.get("resolver") or {}
    ext_section = raw.get("supported_extensions") or {}
    inspection_section = raw.get("inspection") or {}
    alpha_section = raw.get("alpha") or {}
    dims_section = raw.get("dimensions") or {}
    dup_section = raw.get("duplicates") or {}
    checksum_section = raw.get("checksum") or {}
    output_section = raw.get("output") or {}

    dimension_policies: dict[str, RoleDimensionPolicy] = {}
    for role_name, role_data in dims_section.items():
        role_data = role_data or {}
        dimension_policies[str(role_name)] = RoleDimensionPolicy(
            minimum_width=role_data.get("minimum_width"),
            minimum_height=role_data.get("minimum_height"),
            maximum_width=role_data.get("maximum_width"),
            maximum_height=role_data.get("maximum_height"),
            maximum_megapixels=role_data.get("maximum_megapixels"),
            maximum_aspect_ratio=role_data.get("maximum_aspect_ratio"),
            minimum_aspect_ratio=role_data.get("minimum_aspect_ratio"),
            preferred_aspect_ratio_min=role_data.get("preferred_aspect_ratio_min"),
            preferred_aspect_ratio_max=role_data.get("preferred_aspect_ratio_max"),
        )

    return OverlayAssetResolverConfig(
        schema_version=str(resolver_section.get("schema_version", "1.0")),
        project_root=str(resolver_section.get("project_root", ".")),
        allow_absolute_paths=bool(resolver_section.get("allow_absolute_paths", True)),
        follow_symlinks=bool(resolver_section.get("follow_symlinks", True)),
        reject_hidden_files=bool(resolver_section.get("reject_hidden_files", False)),
        fail_on_missing_required_asset=bool(resolver_section.get("fail_on_missing_required_asset", True)),
        fail_on_invalid_optional_asset=bool(resolver_section.get("fail_on_invalid_optional_asset", False)),
        supported_extensions_enabled=tuple(
            str(e).lower() for e in (ext_section.get("enabled") or [".png", ".webp", ".jpg", ".jpeg"])
        ),
        supported_extensions_disabled=tuple(str(e).lower() for e in (ext_section.get("disabled") or [".gif", ".svg"])),
        inspection_backend=str(inspection_section.get("backend", "pillow")),
        require_inspector=bool(inspection_section.get("require_inspector", True)),
        allow_unknown_alpha=bool(inspection_section.get("allow_unknown_alpha", False)),
        reject_animated_assets=bool(inspection_section.get("reject_animated_assets", True)),
        alpha_roles_requiring_alpha=tuple(
            str(r) for r in (alpha_section.get("roles_requiring_alpha") or ["logo", "watermark", "sticker"])
        ),
        alpha_missing_alpha_policy=dict(alpha_section.get("missing_alpha_policy") or {}),
        dimension_policies=dimension_policies,
        duplicate_same_path_same_role=str(dup_section.get("same_path_same_role", "merge_references")),
        duplicate_same_path_different_role=str(dup_section.get("same_path_different_role", "separate_assets")),
        duplicate_conflicting_requirements=str(dup_section.get("conflicting_requirements", "error")),
        duplicate_include_source_overlay_in_asset_id=bool(
            dup_section.get("include_source_overlay_in_asset_id", False)
        ),
        checksum_enabled=bool(checksum_section.get("enabled", True)),
        checksum_algorithm=str(checksum_section.get("algorithm", "sha256")),
        checksum_chunk_size_bytes=int(checksum_section.get("chunk_size_bytes", 1048576)),
        overwrite_requires_force=bool(output_section.get("overwrite_requires_force", True)),
        atomic_write=bool(output_section.get("atomic_write", True)),
        manifest_filename=str(output_section.get("manifest_filename", "overlay_asset_manifest.json")),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OverlayAssetReference:
    reference_id: str = ""
    source_plan_id: str = ""
    source_overlay_id: str = ""
    source_asset_id: str = ""
    logical_role: str = ""
    asset_type: str = OverlayAssetType.UNKNOWN
    raw_path: str = ""
    required: bool = True
    requires_alpha: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OverlayAssetInspection:
    width: int | None = None
    height: int | None = None
    format: str | None = None
    color_mode: str | None = None
    has_alpha: bool | None = None
    alpha_mode: str = AlphaMode.UNKNOWN
    animated: bool | None = None
    frame_count: int | None = None


ImageInspector = Callable[[Path], OverlayAssetInspection]


@dataclass(slots=True)
class OverlayRenderAsset:
    asset_id: str = ""
    # The spec names this field "reference_id" (singular) but also
    # requires "one asset with multiple reference IDs" for merged
    # duplicates -- a list is the only shape that can hold that.
    reference_ids: list[str] = field(default_factory=list)
    logical_role: str = ""
    asset_type: str = OverlayAssetType.UNKNOWN
    source_path: str = ""
    resolved_path: str = ""
    filename: str = ""
    extension: str = ""
    file_size_bytes: int = 0
    width: int | None = None
    height: int | None = None
    aspect_ratio: float | None = None
    has_alpha: bool | None = None
    alpha_mode: str = AlphaMode.UNKNOWN
    color_mode: str | None = None
    animated: bool | None = None
    checksum: str | None = None
    # Same pluralization reasoning as reference_ids.
    source_plan_ids: list[str] = field(default_factory=list)
    source_overlay_ids: list[str] = field(default_factory=list)
    required: bool = True
    status: str = AssetResolutionStatus.RESOLVED
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OverlayAssetWarning:
    code: str = ""
    message: str = ""
    reference_id: str | None = None


@dataclass(slots=True)
class OverlayAssetResolutionResult:
    resolved: list[OverlayRenderAsset] = field(default_factory=list)
    assets: list[OverlayRenderAsset] = field(default_factory=list)
    missing: list[OverlayRenderAsset] = field(default_factory=list)
    invalid: list[OverlayRenderAsset] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest_id: str = ""
    summary: str = ""


@dataclass(slots=True)
class OverlayAssetSetValidationResult:
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    asset_count: int = 0
    resolved_count: int = 0
    missing_count: int = 0
    invalid_count: int = 0
    summary: str = ""


@dataclass(slots=True)
class OverlayAssetManifest:
    schema_version: str = "1.0"
    manifest_id: str = ""
    created_at: str = ""
    source_plan_ids: list[str] = field(default_factory=list)
    asset_count: int = 0
    resolved_count: int = 0
    missing_count: int = 0
    invalid_count: int = 0
    assets: list[OverlayRenderAsset] = field(default_factory=list)
    references: list[OverlayAssetReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reference collection — never loads a plan itself, never invents a path.
# ---------------------------------------------------------------------------


def _compute_reference_id(source_plan_id: str, source_overlay_id: str, source_asset_id: str, logical_role: str, raw_path: str) -> str:
    payload = f"{source_plan_id}|{source_overlay_id}|{source_asset_id}|{logical_role}|{raw_path}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _overlay_logical_role(overlay: overlay_plan_engine.PlannedOverlay) -> str | None:
    if overlay.overlay_type == overlay_plan_engine.OverlayType.SUBTITLE:
        return None
    if overlay.overlay_type == overlay_plan_engine.OverlayType.CUSTOM:
        return overlay.custom_overlay_type or OverlayLogicalRole.CUSTOM
    return overlay.overlay_type


def _raw_path_from_overlay(overlay: overlay_plan_engine.PlannedOverlay) -> str | None:
    style_snapshot = overlay.style_snapshot
    if style_snapshot is not None and style_snapshot.asset_reference:
        return style_snapshot.asset_reference
    metadata_reference = (overlay.metadata or {}).get("asset_reference")
    if metadata_reference:
        return str(metadata_reference)
    if overlay.content and _classify_extension(overlay.content) != OverlayAssetType.UNKNOWN:
        return overlay.content
    return None


def collect_overlay_asset_references(
    *,
    overlay_plan: overlay_plan_engine.OverlayPlan | None = None,
    renderer_plan: renderer_plan_engine.RendererPlan | None = None,
    config: OverlayAssetResolverConfig,
) -> list[OverlayAssetReference]:
    """
    Collects explicit overlay asset references from an already-loaded,
    already-validated OverlayPlan and/or RendererPlan. Never loads a
    plan itself, never mutates either plan, and never invents a path
    for an overlay/asset that has none -- such entries are skipped, not
    guessed.
    """
    references: list[OverlayAssetReference] = []

    if overlay_plan is not None:
        plan_id = overlay_plan.plan_id
        for overlay in overlay_plan.overlays:
            if not overlay.enabled:
                continue
            role = _overlay_logical_role(overlay)
            if role is None:
                continue
            raw_path = _raw_path_from_overlay(overlay)
            if not raw_path:
                continue
            reference_id = _compute_reference_id(plan_id, overlay.overlay_id, "", role, raw_path)
            references.append(
                OverlayAssetReference(
                    reference_id=reference_id, source_plan_id=plan_id, source_overlay_id=overlay.overlay_id,
                    source_asset_id="", logical_role=role, asset_type=_classify_extension(raw_path),
                    raw_path=raw_path, required=True,
                    requires_alpha=role in config.alpha_roles_requiring_alpha,
                    metadata=dict(overlay.metadata or {}),
                )
            )

    if renderer_plan is not None:
        plan_id = renderer_plan.renderer_plan_id
        for asset in renderer_plan.assets:
            if asset.asset_type != renderer_plan_engine.AssetType.IMAGE:
                continue
            if not asset.source_path:
                continue
            role = asset.logical_role or OverlayLogicalRole.CUSTOM
            reference_id = _compute_reference_id(plan_id, "", asset.asset_id, role, asset.source_path)
            references.append(
                OverlayAssetReference(
                    reference_id=reference_id, source_plan_id=plan_id, source_overlay_id="",
                    source_asset_id=asset.asset_id, logical_role=role, asset_type=_classify_extension(asset.source_path),
                    raw_path=asset.source_path, required=True,
                    requires_alpha=role in config.alpha_roles_requiring_alpha,
                    metadata=dict(asset.metadata or {}),
                )
            )

    return references


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _check_extension(raw_path: str, config: OverlayAssetResolverConfig) -> str:
    extension = Path(raw_path).suffix.lower()
    if extension in config.supported_extensions_disabled:
        raise UnsupportedOverlayAssetError(f"Extension {extension!r} is explicitly disabled: {raw_path}")
    if extension not in config.supported_extensions_enabled:
        raise UnsupportedOverlayAssetError(f"Extension {extension!r} is not supported: {raw_path}")
    return _classify_extension(raw_path)


def _resolve_candidate_path(raw_path: str, config: OverlayAssetResolverConfig) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        if not config.allow_absolute_paths:
            raise UnsafeOverlayAssetPathError(f"Absolute paths are not allowed by config: {raw_path}")
        return path
    return Path(config.project_root) / path


def _validate_asset_path(path: Path, config: OverlayAssetResolverConfig) -> Path:
    if path.is_symlink() and not config.follow_symlinks:
        raise UnsafeOverlayAssetPathError(f"Path is a symlink and resolver.follow_symlinks is false: {path}")
    if not path.exists():
        raise OverlayAssetNotFoundError(f"Overlay asset not found: {path}")
    if not path.is_file():
        raise UnsafeOverlayAssetPathError(f"Overlay asset path is not a regular file: {path}")
    if config.reject_hidden_files and path.name.startswith("."):
        raise UnsafeOverlayAssetPathError(f"Hidden files are rejected by config: {path}")
    resolved = path.resolve()
    if resolved.stat().st_size == 0:
        raise OverlayAssetEmptyError(f"Overlay asset is zero bytes: {path}")
    return resolved


# ---------------------------------------------------------------------------
# Checksum — chunked, never loads a whole large asset into memory.
# ---------------------------------------------------------------------------


def _compute_checksum(path: Path, config: OverlayAssetResolverConfig) -> str | None:
    if not config.checksum_enabled:
        return None
    try:
        digest = hashlib.new(config.checksum_algorithm)
        with path.open("rb") as file:
            while True:
                chunk = file.read(config.checksum_chunk_size_bytes)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError) as exc:
        raise OverlayAssetChecksumError(f"Failed to compute checksum for {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Alpha / dimension policy — never synthesizes alpha, never resizes.
# ---------------------------------------------------------------------------


def _validate_alpha(role: str, inspection: OverlayAssetInspection, config: OverlayAssetResolverConfig) -> tuple[list[str], str | None]:
    """
    Returns (warnings, error_message_or_none). A role is checked at all
    when it has an explicit entry in alpha.missing_alpha_policy (every
    role in the suggested config does) or is listed in
    alpha.roles_requiring_alpha; a role configured in neither is never
    checked. roles_requiring_alpha roles default to "error" when no
    explicit policy entry is present; every other configured role
    defaults to "warning".
    """
    policy = config.alpha_missing_alpha_policy.get(role)
    if policy is None:
        if role not in config.alpha_roles_requiring_alpha:
            return [], None
        policy = "error"

    if inspection.has_alpha is True:
        return [], None

    if inspection.has_alpha is None:
        message = f"Alpha channel presence is unknown for role {role!r}"
        if config.allow_unknown_alpha:
            return [message], None
        return ([], message) if policy == "error" else ([message], None)

    message = f"Role {role!r} requires an alpha channel but the asset has none"
    return ([], message) if policy == "error" else ([message], None)


def _validate_dimensions(role: str, inspection: OverlayAssetInspection, config: OverlayAssetResolverConfig) -> list[str]:
    policy = config.dimension_policy(role)
    width, height = inspection.width, inspection.height
    if width is None or height is None:
        return []

    errors: list[str] = []
    if width <= 0 or height <= 0:
        return [f"Asset has non-positive dimensions: {width}x{height}"]

    if policy.minimum_width is not None and width < policy.minimum_width:
        errors.append(f"width {width} is below minimum_width {policy.minimum_width} for role {role!r}")
    if policy.minimum_height is not None and height < policy.minimum_height:
        errors.append(f"height {height} is below minimum_height {policy.minimum_height} for role {role!r}")
    if policy.maximum_width is not None and width > policy.maximum_width:
        errors.append(f"width {width} exceeds maximum_width {policy.maximum_width} for role {role!r}")
    if policy.maximum_height is not None and height > policy.maximum_height:
        errors.append(f"height {height} exceeds maximum_height {policy.maximum_height} for role {role!r}")
    if policy.maximum_megapixels is not None:
        megapixels = (width * height) / 1_000_000
        if megapixels > policy.maximum_megapixels:
            errors.append(f"{megapixels:.2f} MP exceeds maximum_megapixels {policy.maximum_megapixels} for role {role!r}")

    aspect_ratio = width / height
    if policy.maximum_aspect_ratio is not None and aspect_ratio > policy.maximum_aspect_ratio:
        errors.append(f"aspect ratio {aspect_ratio:.3f} exceeds maximum_aspect_ratio {policy.maximum_aspect_ratio} for role {role!r}")
    if policy.minimum_aspect_ratio is not None and aspect_ratio < policy.minimum_aspect_ratio:
        errors.append(f"aspect ratio {aspect_ratio:.3f} is below minimum_aspect_ratio {policy.minimum_aspect_ratio} for role {role!r}")

    return errors


def _validate_preferred_aspect_ratio(role: str, inspection: OverlayAssetInspection, config: OverlayAssetResolverConfig) -> list[str]:
    policy = config.dimension_policy(role)
    warnings: list[str] = []
    if inspection.width and inspection.height:
        aspect_ratio = inspection.width / inspection.height
        if policy.preferred_aspect_ratio_min is not None and aspect_ratio < policy.preferred_aspect_ratio_min:
            warnings.append(
                f"aspect ratio {aspect_ratio:.3f} is below preferred_aspect_ratio_min "
                f"{policy.preferred_aspect_ratio_min} for role {role!r}"
            )
        if policy.preferred_aspect_ratio_max is not None and aspect_ratio > policy.preferred_aspect_ratio_max:
            warnings.append(
                f"aspect ratio {aspect_ratio:.3f} exceeds preferred_aspect_ratio_max "
                f"{policy.preferred_aspect_ratio_max} for role {role!r}"
            )
    return warnings


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _compute_asset_id(
    resolved_path: str, checksum: str | None, logical_role: str, asset_type: str,
    source_overlay_ids: list[str], config: OverlayAssetResolverConfig,
) -> str:
    payload: dict[str, Any] = {
        "resolved_path": resolved_path, "checksum": checksum, "logical_role": logical_role,
        "asset_type": asset_type, "schema_version": config.schema_version,
    }
    if config.duplicate_include_source_overlay_in_asset_id and source_overlay_ids:
        payload["source_overlay_id"] = sorted(source_overlay_ids)[0]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _compute_manifest_id(assets: list[OverlayRenderAsset], config: OverlayAssetResolverConfig) -> str:
    payload = {
        "assets": [
            {"asset_id": a.asset_id, "resolved_path": a.resolved_path, "checksum": a.checksum, "logical_role": a.logical_role}
            for a in assets
        ],
        "schema_version": config.schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_overlay_asset_inner(
    reference: OverlayAssetReference, config: OverlayAssetResolverConfig, *, inspector: ImageInspector | None,
) -> OverlayRenderAsset:
    asset_type = _check_extension(reference.raw_path, config)

    candidate_path = _resolve_candidate_path(reference.raw_path, config)
    resolved_path = _validate_asset_path(candidate_path, config)

    checksum = _compute_checksum(resolved_path, config)

    if inspector is None:
        if config.require_inspector:
            raise OverlayAssetInspectorUnavailableError(
                "No image inspector was provided and inspection.require_inspector is true"
            )
        inspection = OverlayAssetInspection()
    else:
        try:
            inspection = inspector(resolved_path)
        except OverlayAssetResolverError:
            raise
        except Exception as exc:  # noqa: BLE001 -- deliberately wraps any inspector failure
            raise OverlayAssetInspectionError(f"Image inspector failed for {resolved_path}: {exc}") from exc

    if config.reject_animated_assets and inspection.animated:
        raise OverlayAssetAnimatedError(f"Animated overlay assets are rejected by config: {resolved_path}")

    warnings: list[str] = []

    alpha_warnings, alpha_error = _validate_alpha(reference.logical_role, inspection, config)
    warnings.extend(alpha_warnings)
    if alpha_error:
        raise OverlayAssetAlphaError(alpha_error)

    dimension_errors = _validate_dimensions(reference.logical_role, inspection, config)
    if dimension_errors:
        raise OverlayAssetDimensionError("; ".join(dimension_errors))
    warnings.extend(_validate_preferred_aspect_ratio(reference.logical_role, inspection, config))

    aspect_ratio = (inspection.width / inspection.height) if inspection.width and inspection.height else None
    source_overlay_ids = [reference.source_overlay_id] if reference.source_overlay_id else []

    asset_id = _compute_asset_id(str(resolved_path), checksum, reference.logical_role, asset_type, source_overlay_ids, config)

    return OverlayRenderAsset(
        asset_id=asset_id,
        reference_ids=[reference.reference_id],
        logical_role=reference.logical_role,
        asset_type=asset_type,
        source_path=reference.raw_path,
        resolved_path=str(resolved_path),
        filename=resolved_path.name,
        extension=resolved_path.suffix.lower(),
        file_size_bytes=resolved_path.stat().st_size,
        width=inspection.width,
        height=inspection.height,
        aspect_ratio=aspect_ratio,
        has_alpha=inspection.has_alpha,
        alpha_mode=inspection.alpha_mode or AlphaMode.UNKNOWN,
        color_mode=inspection.color_mode,
        animated=inspection.animated,
        checksum=checksum,
        source_plan_ids=[reference.source_plan_id] if reference.source_plan_id else [],
        source_overlay_ids=source_overlay_ids,
        required=reference.required,
        status=AssetResolutionStatus.RESOLVED,
        warnings=warnings,
        metadata=dict(reference.metadata),
    )


def resolve_overlay_asset(
    reference: OverlayAssetReference, config: OverlayAssetResolverConfig, *, inspector: ImageInspector | None = None,
) -> OverlayRenderAsset:
    """
    Resolves exactly one OverlayAssetReference into an OverlayRenderAsset.
    Raises for a structural failure unless the failure policy for this
    reference's required-ness says otherwise (fail_on_missing_required_asset/
    fail_on_invalid_optional_asset), in which case a soft MISSING/INVALID
    record is returned instead. Never resizes, converts, or synthesizes
    anything; never picks a replacement asset.
    """
    try:
        return _resolve_overlay_asset_inner(reference, config, inspector=inspector)
    except OverlayAssetResolverError as exc:
        should_raise = (
            (reference.required and config.fail_on_missing_required_asset)
            or (not reference.required and config.fail_on_invalid_optional_asset)
        )
        if should_raise:
            raise
        status = AssetResolutionStatus.MISSING if isinstance(exc, OverlayAssetNotFoundError) else AssetResolutionStatus.INVALID
        return OverlayRenderAsset(
            asset_id="", reference_ids=[reference.reference_id], logical_role=reference.logical_role,
            asset_type=reference.asset_type, source_path=reference.raw_path, resolved_path="",
            filename=Path(reference.raw_path).name, extension=Path(reference.raw_path).suffix.lower(),
            file_size_bytes=0, width=None, height=None, aspect_ratio=None, has_alpha=None,
            alpha_mode=AlphaMode.UNKNOWN, color_mode=None, animated=None, checksum=None,
            source_plan_ids=[reference.source_plan_id] if reference.source_plan_id else [],
            source_overlay_ids=[reference.source_overlay_id] if reference.source_overlay_id else [],
            required=reference.required, status=status, warnings=[str(exc)], metadata=dict(reference.metadata),
        )


def resolve_overlay_assets(
    references: list[OverlayAssetReference], config: OverlayAssetResolverConfig, *, inspector: ImageInspector | None = None,
) -> OverlayAssetResolutionResult:
    """
    Resolves a batch of references, deduplicating by (resolved_path,
    logical_role): same path + same role merges into one asset with
    every contributing reference_id preserved (never silently dropped);
    same path + different role stays as separate asset records. A hard
    per-reference failure is recorded in `errors` rather than aborting
    the whole batch -- validate_overlay_asset_set() is what turns that
    into an overall pass/fail decision.
    """
    inspection_cache: dict[str, OverlayAssetInspection] = {}

    def cached_inspector(path: Path) -> OverlayAssetInspection:
        key = str(path)
        if key not in inspection_cache:
            inspection_cache[key] = inspector(path)  # type: ignore[misc]
        return inspection_cache[key]

    effective_inspector = cached_inspector if inspector is not None else None

    errors: list[str] = []
    resolved_individually: list[tuple[OverlayAssetReference, OverlayRenderAsset]] = []
    for reference in references:
        try:
            asset = resolve_overlay_asset(reference, config, inspector=effective_inspector)
        except OverlayAssetResolverError as exc:
            errors.append(str(exc))
            continue
        resolved_individually.append((reference, asset))

    groups: dict[tuple[str, str], list[tuple[OverlayAssetReference, OverlayRenderAsset]]] = {}
    order: list[tuple[str, str]] = []
    for reference, asset in resolved_individually:
        key = (asset.resolved_path or asset.source_path, asset.logical_role)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((reference, asset))

    final_assets: list[OverlayRenderAsset] = []
    for key in order:
        members = groups[key]
        if len(members) == 1 or config.duplicate_same_path_same_role != "merge_references":
            final_assets.extend(asset for _, asset in members)
            continue

        requires_alpha_values = {reference.requires_alpha for reference, _ in members}
        if len(requires_alpha_values) > 1 and config.duplicate_conflicting_requirements == "error":
            raise OverlayAssetConflictError(
                f"Conflicting alpha requirements for path {key[0]!r} role {key[1]!r}: {sorted(requires_alpha_values)}"
            )

        first_asset = members[0][1]
        merged_reference_ids = [reference.reference_id for reference, _ in members]
        merged_overlay_ids = sorted({oid for _, asset in members for oid in asset.source_overlay_ids})
        merged_plan_ids = sorted({pid for _, asset in members for pid in asset.source_plan_ids})
        merged_warnings = [w for _, asset in members for w in asset.warnings]
        merged_asset_id = _compute_asset_id(
            first_asset.resolved_path, first_asset.checksum, first_asset.logical_role, first_asset.asset_type,
            merged_overlay_ids, config,
        )
        final_assets.append(
            dataclasses.replace(
                first_asset, asset_id=merged_asset_id, reference_ids=merged_reference_ids,
                source_overlay_ids=merged_overlay_ids, source_plan_ids=merged_plan_ids, warnings=merged_warnings,
            )
        )

    resolved = [a for a in final_assets if a.status == AssetResolutionStatus.RESOLVED]
    missing = [a for a in final_assets if a.status == AssetResolutionStatus.MISSING]
    invalid = [a for a in final_assets if a.status == AssetResolutionStatus.INVALID]

    manifest_id = _compute_manifest_id(final_assets, config)
    summary = (
        f"Overlay assets: {len(resolved)} resolved, {len(missing)} missing, "
        f"{len(invalid)} invalid, {len(errors)} error(s)"
    )

    return OverlayAssetResolutionResult(
        resolved=resolved, assets=final_assets, missing=missing, invalid=invalid,
        warnings=[], errors=errors, manifest_id=manifest_id, summary=summary,
    )


# ---------------------------------------------------------------------------
# Validation — pure, never raises.
# ---------------------------------------------------------------------------


def validate_overlay_render_asset(asset: OverlayRenderAsset, config: OverlayAssetResolverConfig) -> list[str]:
    errors: list[str] = []
    if not asset.asset_id and asset.status == AssetResolutionStatus.RESOLVED:
        errors.append("asset.asset_id: empty field: expected non-empty for a resolved asset")
    if not asset.reference_ids:
        errors.append("asset.reference_ids: empty field: expected at least one reference")
    if asset.logical_role not in OverlayLogicalRole.ALL:
        errors.append(f"asset.logical_role={asset.logical_role!r} field: expected one of {OverlayLogicalRole.ALL}")
    if asset.asset_type not in OverlayAssetType.ALL:
        errors.append(f"asset.asset_type={asset.asset_type!r} field: expected one of {OverlayAssetType.ALL}")
    if asset.status not in AssetResolutionStatus.ALL:
        errors.append(f"asset.status={asset.status!r} field: expected one of {AssetResolutionStatus.ALL}")
    try:
        json.dumps(asset.metadata)
    except TypeError:
        errors.append("asset.metadata: not JSON-serializable")
    return errors


def validate_overlay_asset_set(
    result: OverlayAssetResolutionResult, config: OverlayAssetResolverConfig,
) -> OverlayAssetSetValidationResult:
    errors: list[str] = []
    warnings: list[str] = list(result.warnings)

    asset_ids = [a.asset_id for a in result.assets if a.asset_id]
    if len(asset_ids) != len(set(asset_ids)):
        errors.append("result.assets: duplicate asset_id values field: asset_id expected unique")

    all_reference_ids = [rid for a in result.assets for rid in a.reference_ids]
    if len(all_reference_ids) != len(set(all_reference_ids)):
        errors.append("result.assets: duplicate reference_id values across assets field: expected unique")

    for asset in result.assets:
        for message in validate_overlay_render_asset(asset, config):
            errors.append(f"asset {asset.asset_id or asset.source_path}: {message}")

    errors.extend(result.errors)

    required_unresolved = [a for a in result.assets if a.required and a.status != AssetResolutionStatus.RESOLVED]
    if required_unresolved and config.fail_on_missing_required_asset:
        errors.append(f"{len(required_unresolved)} required asset(s) failed to resolve")

    passed = len(errors) == 0
    summary = f"Overlay asset set: {'PASSED' if passed else 'FAILED'} ({len(errors)} error(s), {len(warnings)} warning(s))"

    return OverlayAssetSetValidationResult(
        passed=passed, errors=errors, warnings=warnings, asset_count=len(result.assets),
        resolved_count=len(result.resolved), missing_count=len(result.missing), invalid_count=len(result.invalid),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_overlay_asset_manifest(
    result: OverlayAssetResolutionResult, references: list[OverlayAssetReference], config: OverlayAssetResolverConfig,
    *, source_plan_ids: list[str],
) -> OverlayAssetManifest:
    manifest = OverlayAssetManifest(
        schema_version=config.schema_version, manifest_id=result.manifest_id, source_plan_ids=list(source_plan_ids),
        asset_count=len(result.assets), resolved_count=len(result.resolved), missing_count=len(result.missing),
        invalid_count=len(result.invalid), assets=list(result.assets), references=list(references),
        warnings=list(result.warnings),
    )
    manifest.created_at = _now_iso()
    return manifest


def _reference_to_dict(reference: OverlayAssetReference) -> dict[str, Any]:
    return asdict(reference)


def _reference_from_dict(data: dict[str, Any]) -> OverlayAssetReference:
    known = {f.name for f in dataclasses.fields(OverlayAssetReference)}
    try:
        return OverlayAssetReference(**{key: value for key, value in data.items() if key in known})
    except TypeError as exc:
        raise OverlayAssetManifestJSONError(f"Malformed overlay asset reference in manifest JSON: {exc}") from exc


def _asset_to_dict(asset: OverlayRenderAsset) -> dict[str, Any]:
    return asdict(asset)


def _asset_from_dict(data: dict[str, Any]) -> OverlayRenderAsset:
    known = {f.name for f in dataclasses.fields(OverlayRenderAsset)}
    try:
        return OverlayRenderAsset(**{key: value for key, value in data.items() if key in known})
    except TypeError as exc:
        raise OverlayAssetManifestJSONError(f"Malformed overlay render asset in manifest JSON: {exc}") from exc


def overlay_render_asset_to_dict(asset: OverlayRenderAsset) -> dict[str, Any]:
    return _asset_to_dict(asset)


def overlay_render_asset_from_dict(data: dict[str, Any]) -> OverlayRenderAsset:
    return _asset_from_dict(data)


def overlay_asset_manifest_to_dict(manifest: OverlayAssetManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "manifest_id": manifest.manifest_id,
        "created_at": manifest.created_at,
        "source_plan_ids": list(manifest.source_plan_ids),
        "asset_count": manifest.asset_count,
        "resolved_count": manifest.resolved_count,
        "missing_count": manifest.missing_count,
        "invalid_count": manifest.invalid_count,
        "assets": [_asset_to_dict(a) for a in manifest.assets],
        "references": [_reference_to_dict(r) for r in manifest.references],
        "warnings": list(manifest.warnings),
        "metadata": manifest.metadata,
    }


def overlay_asset_manifest_from_dict(data: dict[str, Any]) -> OverlayAssetManifest:
    try:
        return OverlayAssetManifest(
            schema_version=data.get("schema_version", "1.0"),
            manifest_id=data.get("manifest_id", ""),
            created_at=data.get("created_at", ""),
            source_plan_ids=list(data.get("source_plan_ids", [])),
            asset_count=data.get("asset_count", 0),
            resolved_count=data.get("resolved_count", 0),
            missing_count=data.get("missing_count", 0),
            invalid_count=data.get("invalid_count", 0),
            assets=[_asset_from_dict(a) for a in data.get("assets", [])],
            references=[_reference_from_dict(r) for r in data.get("references", [])],
            warnings=list(data.get("warnings", [])),
            metadata=data.get("metadata", {}),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise OverlayAssetManifestJSONError(f"Malformed overlay asset manifest JSON: {exc}") from exc


def load_overlay_asset_manifest(path: str | Path) -> OverlayAssetManifest:
    path = Path(path)
    if not path.is_file():
        raise OverlayAssetManifestJSONError(f"Overlay asset manifest not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OverlayAssetManifestJSONError(f"Invalid overlay asset manifest JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise OverlayAssetManifestJSONError(f"Overlay asset manifest JSON root must be an object: {path}")

    return overlay_asset_manifest_from_dict(data)


def save_overlay_asset_manifest(manifest: OverlayAssetManifest, path: str | Path, *, force: bool = False) -> Path:
    """Atomically writes manifest JSON via temp-file + Path.replace() so
    a reader never observes a partially-written file. Refuses to
    overwrite an existing file unless force=True."""
    path = Path(path)

    if path.exists():
        if path.is_dir():
            raise UnsafeOverlayAssetManifestPathError(f"Output path is a directory: {path}")
        if not force:
            raise OverlayAssetManifestExistsError(f"{path} already exists; pass force=True to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(overlay_asset_manifest_to_dict(manifest), indent=2, sort_keys=True, ensure_ascii=False)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
    return path


# ---------------------------------------------------------------------------
# Real image inspector — lazy Pillow import, never required by tests.
# ---------------------------------------------------------------------------


def default_image_inspector(path: Path) -> OverlayAssetInspection:
    """
    Real inspector backend. Opens read-only, closes immediately, never
    saves/converts/alters the image. Raises
    OverlayAssetInspectorUnavailableError if Pillow is not installed --
    this module never installs it automatically.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise OverlayAssetInspectorUnavailableError(
            "Pillow is not installed. Provide an inspector callable, or install Pillow to enable "
            "real image inspection (this module never installs it automatically)."
        ) from exc

    try:
        with Image.open(path) as image:
            width, height = image.size
            color_mode = image.mode
            has_alpha = color_mode in ("RGBA", "LA", "PA") or "transparency" in image.info
            animated = bool(getattr(image, "is_animated", False))
            frame_count = getattr(image, "n_frames", 1)
            image_format = image.format
    except OverlayAssetResolverError:
        raise
    except Exception as exc:  # noqa: BLE001 -- deliberately wraps any Pillow failure
        raise OverlayAssetInspectionError(f"Failed to inspect image {path}: {exc}") from exc

    return OverlayAssetInspection(
        width=width, height=height, format=image_format, color_mode=color_mode, has_alpha=has_alpha,
        alpha_mode=AlphaMode.STRAIGHT if has_alpha else AlphaMode.NONE, animated=animated, frame_count=frame_count,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Overlay Asset Resolver (Phase 11F.4). Resolves and validates local overlay "
            "image assets referenced by an OverlayPlan/RendererPlan. Never renders, never calls "
            "ffmpeg/ffprobe, never downloads or transforms an asset."
        )
    )

    parser.add_argument("--overlay-plan", dest="overlay_plan", default=None, help="Path to a source overlay_plan.json.")
    parser.add_argument("--renderer-plan", dest="renderer_plan", default=None, help="Path to a source renderer_plan.json.")
    parser.add_argument("--output", default=None, help="Path to write the overlay asset manifest JSON.")
    parser.add_argument("--config", default=None, help="Path to an alternate config/video/overlay_assets.yaml.")
    parser.add_argument("--validate-only", dest="validate_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")

    arguments = parser.parse_args(argv)

    if not arguments.overlay_plan and not arguments.renderer_plan:
        parser.error("At least one of --overlay-plan/--renderer-plan is required.")
    if arguments.validate_only and arguments.output:
        parser.error("--output cannot be combined with --validate-only.")
    if not arguments.validate_only and not arguments.output:
        parser.error("--output is required unless --validate-only is given.")

    return arguments


def _print_summary(manifest: OverlayAssetManifest, output_path: Path | None) -> None:
    print()
    print("AIKO Overlay Asset Resolver (Phase 11F.4)")
    print("----------------------------------------------")
    print(f"manifest_id:     {manifest.manifest_id}")
    print(f"source_plan_ids: {manifest.source_plan_ids}")
    print(f"assets:          {manifest.asset_count} (resolved={manifest.resolved_count}, "
          f"missing={manifest.missing_count}, invalid={manifest.invalid_count})")
    for warning in manifest.warnings:
        print(f"warning: {warning}")
    print(f"output_path:     {output_path if output_path else '(not written)'}")
    print()


def _print_validation_summary(result: OverlayAssetSetValidationResult) -> None:
    print()
    print("AIKO Overlay Asset Resolver — validation (Phase 11F.4)")
    print("----------------------------------------------------------------")
    print(result.summary)
    for error in result.errors:
        print(f"  error:   {error}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print()


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)

    try:
        config = load_overlay_asset_config(arguments.config)

        overlay_plan = None
        overlay_plan_path: Path | None = None
        if arguments.overlay_plan:
            overlay_plan_path = Path(arguments.overlay_plan)
            try:
                overlay_plan = overlay_plan_engine.load_overlay_plan(overlay_plan_path)
            except overlay_plan_engine.OverlayPlanEngineError as exc:
                raise OverlayAssetReferenceError(f"Failed to load overlay plan {overlay_plan_path}: {exc}") from exc
            overlay_plan_config = overlay_plan_engine.load_overlay_plan_config()
            overlay_validation = overlay_plan_engine.validate_overlay_plan(overlay_plan, overlay_plan_config)
            if not overlay_validation.passed:
                raise OverlayAssetReferenceError(
                    f"Overlay plan {overlay_plan_path} failed validation: {'; '.join(overlay_validation.errors)}"
                )

        renderer_plan = None
        renderer_plan_path: Path | None = None
        if arguments.renderer_plan:
            renderer_plan_path = Path(arguments.renderer_plan)
            try:
                renderer_plan = renderer_plan_engine.load_renderer_plan(renderer_plan_path)
            except renderer_plan_engine.RendererPlanEngineError as exc:
                raise OverlayAssetReferenceError(f"Failed to load renderer plan {renderer_plan_path}: {exc}") from exc
            renderer_plan_config = renderer_plan_engine.load_renderer_plan_config()
            renderer_validation = renderer_plan_engine.validate_renderer_plan(renderer_plan, renderer_plan_config)
            if not renderer_validation.passed:
                raise OverlayAssetReferenceError(
                    f"Renderer plan {renderer_plan_path} failed validation: {'; '.join(renderer_validation.errors)}"
                )

        if arguments.output:
            output_path = Path(arguments.output)
            for source_path in (overlay_plan_path, renderer_plan_path):
                if source_path is not None and output_path.resolve() == source_path.resolve():
                    raise UnsafeOverlayAssetManifestPathError(f"--output must not be the same path as {source_path}")

        references = collect_overlay_asset_references(overlay_plan=overlay_plan, renderer_plan=renderer_plan, config=config)
        inspector = default_image_inspector if config.require_inspector else None
        result = resolve_overlay_assets(references, config, inspector=inspector)

        source_plan_ids = [
            plan_id for plan_id in (
                overlay_plan.plan_id if overlay_plan else None,
                renderer_plan.renderer_plan_id if renderer_plan else None,
            )
            if plan_id
        ]
        manifest = build_overlay_asset_manifest(result, references, config, source_plan_ids=source_plan_ids)

        if arguments.validate_only:
            set_result = validate_overlay_asset_set(result, config)
            if arguments.as_json:
                print(json.dumps(asdict(set_result), indent=2, sort_keys=True))
            else:
                _print_validation_summary(set_result)
            if not set_result.passed:
                raise SystemExit(1)
            return

        output_path_written: Path | None = None
        if arguments.output:
            output_path_written = save_overlay_asset_manifest(manifest, arguments.output, force=arguments.force)

        if arguments.as_json:
            print(json.dumps(overlay_asset_manifest_to_dict(manifest), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print_summary(manifest, output_path_written)

        if result.errors:
            raise SystemExit(1)
    except OverlayAssetResolverError as exc:
        print(f"[OverlayAssetResolver] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
