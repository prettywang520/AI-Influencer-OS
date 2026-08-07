"""ResearchSession -- the runtime container orchestrator.run_job()
builds and mutates while driving one job through its planned states.
No field for a browser, cookie jar, or login token exists on this
class at all -- a structural guarantee, not just a convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .connector import BaseConnector
from .evidence_queue import EvidenceQueue
from .exceptions import ResearchConfigError
from .jobs import ResearchJob
from .progress import ResearchProgress
from .report import ResearchReport

DEFAULT_SESSION_CONFIG_RELATIVE_PATH = Path("config") / "creator_research" / "session.yaml"


def _runtime_root() -> Path:
    """
    session.py location: 10_apps/claude_runtime/src/creator_research/session.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_session_config_path() -> Path:
    return _runtime_root() / DEFAULT_SESSION_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class SessionConfig:
    """Reserved for future session-level policy -- only schema_version
    exists today (ResearchSession itself has no operational policy
    yet)."""

    schema_version: str


def load_session_config(config_path: str | Path | None = None) -> SessionConfig:
    path = Path(config_path) if config_path else default_session_config_path()
    if not path.exists():
        raise ResearchConfigError(f"Session config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ResearchConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ResearchConfigError(f"Session config is empty or invalid: {path}")
    section = raw.get("session") or {}
    return SessionConfig(schema_version=str(section.get("schema_version", "1.0")))


@dataclass(slots=True)
class ResearchSession:
    job: ResearchJob
    progress: ResearchProgress
    connector: BaseConnector
    evidence_queue: EvidenceQueue = field(default_factory=EvidenceQueue)
    statistics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    report: ResearchReport | None = None

    def record_attempt(self, section: str) -> None:
        self.statistics[section] = self.statistics.get(section, 0) + 1

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
        self.progress.add_warning(warning)
