"""Configuration loading for the Creator Learning Engine. Matches the
exact `load_*_config()` convention used throughout this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .exceptions import LearningConfigError

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "creator_learning" / "engine.yaml"


def _runtime_root() -> Path:
    """
    config.py location: 10_apps/claude_runtime/src/creator_learning/config.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return _runtime_root() / DEFAULT_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class LearningConfig:
    schema_version: str
    knowledge_base_root: str
    report_formats: tuple[str, ...]
    generate_reports_on_every_session: bool

    def resolved_knowledge_base_root(self) -> Path:
        return _runtime_root() / self.knowledge_base_root

    def creator_directory(self, creator_id: str) -> Path:
        return self.resolved_knowledge_base_root() / creator_id


def load_learning_config(config_path: str | Path | None = None) -> LearningConfig:
    path = Path(config_path) if config_path else default_config_path()
    if not path.exists():
        raise LearningConfigError(f"Learning engine config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise LearningConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise LearningConfigError(f"Learning engine config is empty or invalid: {path}")

    learning_section = raw.get("learning") or {}
    knowledge_base_section = raw.get("knowledge_base") or {}
    reports_section = raw.get("reports") or {}

    return LearningConfig(
        schema_version=str(learning_section.get("schema_version", "1.0")),
        knowledge_base_root=str(knowledge_base_section.get("root_directory", "output/creator_learning")),
        report_formats=tuple(reports_section.get("formats", ["markdown"])),
        generate_reports_on_every_session=bool(reports_section.get("generate_on_every_session", True)),
    )
