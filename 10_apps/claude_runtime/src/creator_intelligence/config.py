"""Configuration loading for the Creator Intelligence Framework.
Reads config/creator_intelligence/framework.yaml (thresholds, excerpt
limits, analyzer weights) and config/creator_intelligence/evidence_types.yaml
(the allowed EvidenceType set), matching every other `load_*_config()`
in this codebase: raises a module-specific error if the file is
missing or malformed, never silently substitutes a different file or
fabricates missing sections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .confidence import ConfidenceThresholds
from .evidence import EvidenceType

DEFAULT_FRAMEWORK_CONFIG_RELATIVE_PATH = Path("config") / "creator_intelligence" / "framework.yaml"
DEFAULT_EVIDENCE_TYPES_CONFIG_RELATIVE_PATH = Path("config") / "creator_intelligence" / "evidence_types.yaml"


def _runtime_root() -> Path:
    """
    config.py location: 10_apps/claude_runtime/src/creator_intelligence/config.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_framework_config_path() -> Path:
    return _runtime_root() / DEFAULT_FRAMEWORK_CONFIG_RELATIVE_PATH


def default_evidence_types_config_path() -> Path:
    return _runtime_root() / DEFAULT_EVIDENCE_TYPES_CONFIG_RELATIVE_PATH


class CreatorIntelligenceConfigError(RuntimeError):
    """Raised when a Creator Intelligence config file is missing or invalid."""


@dataclass(slots=True)
class CreatorIntelligenceConfig:
    schema_version: str
    confidence_min_evidence_for_medium: int
    confidence_min_evidence_for_high: int
    confidence_min_corroboration_for_verified: int
    caption_excerpt_max_chars: int
    analyzer_weights: dict[str, float] = field(default_factory=dict)
    allowed_evidence_types: tuple[str, ...] = EvidenceType.ALL

    @property
    def confidence_thresholds(self) -> ConfidenceThresholds:
        return ConfidenceThresholds(
            min_evidence_for_medium=self.confidence_min_evidence_for_medium,
            min_evidence_for_high=self.confidence_min_evidence_for_high,
            min_corroboration_for_verified=self.confidence_min_corroboration_for_verified,
        )

    def weight_for(self, analyzer_name: str) -> float:
        return float(self.analyzer_weights.get(analyzer_name, 1.0))


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise CreatorIntelligenceConfigError(f"Creator Intelligence config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise CreatorIntelligenceConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise CreatorIntelligenceConfigError(f"Creator Intelligence config is empty or invalid: {path}")
    return raw


def load_framework_config(
    framework_config_path: str | Path | None = None,
    evidence_types_config_path: str | Path | None = None,
) -> CreatorIntelligenceConfig:
    """Loads framework.yaml + evidence_types.yaml into one
    CreatorIntelligenceConfig. Raises CreatorIntelligenceConfigError if
    either file is missing or malformed."""
    framework_path = Path(framework_config_path) if framework_config_path else default_framework_config_path()
    evidence_types_path = (
        Path(evidence_types_config_path) if evidence_types_config_path else default_evidence_types_config_path()
    )

    framework_raw = _load_yaml(framework_path)
    evidence_types_raw = _load_yaml(evidence_types_path)

    framework_section = framework_raw.get("framework") or {}
    confidence_section = framework_raw.get("confidence") or {}
    evidence_section = framework_raw.get("evidence") or {}
    weights_section = framework_raw.get("analyzer_weights") or {}

    allowed_section = evidence_types_raw.get("evidence_types") or {}
    allowed = allowed_section.get("allowed") or []
    if not allowed:
        raise CreatorIntelligenceConfigError(f"evidence_types.allowed must be non-empty in {evidence_types_path}")
    for entry in allowed:
        if entry not in EvidenceType.ALL:
            raise CreatorIntelligenceConfigError(
                f"evidence_types.allowed contains unknown type {entry!r} in {evidence_types_path}"
            )

    return CreatorIntelligenceConfig(
        schema_version=str(framework_section.get("schema_version", "1.0")),
        confidence_min_evidence_for_medium=int(confidence_section.get("min_evidence_for_medium", 2)),
        confidence_min_evidence_for_high=int(confidence_section.get("min_evidence_for_high", 4)),
        confidence_min_corroboration_for_verified=int(
            confidence_section.get("min_corroboration_for_verified", 2)
        ),
        caption_excerpt_max_chars=int(evidence_section.get("excerpt_max_chars", 280)),
        analyzer_weights={str(k): float(v) for k, v in weights_section.items()},
        allowed_evidence_types=tuple(allowed),
    )
