"""run_job() -- the real execution entry point for the Creator
Research Agent. Drives a ResearchJob through every state in
state.JobStatus.WORKING_STATES (the fixed pipeline every job flows
through, regardless of which sections it actually requested -- a
state with no requested sections simply collects nothing and the job
still transitions through it, matching the task's own linear state
diagram literally). Never touches the network, a browser, or
03_personas/ -- the only thing it calls is whatever Connector method
the job's own requested_sections name.

config/creator_research/cli.py never calls this function -- it has no
way to construct a Connector safely. This is intentional (see
docs/creator_research/architecture.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .connector import BaseConnector
from .evidence_queue import EvidenceQueue
from .exceptions import ResearchConfigError
from .jobs import ResearchJob
from .planner import ResearchPlanner
from .progress import ResearchProgress
from .report import build_report
from .session import ResearchSession
from .state import JobStatus

DEFAULT_ORCHESTRATOR_CONFIG_RELATIVE_PATH = Path("config") / "creator_research" / "orchestrator.yaml"


def _runtime_root() -> Path:
    """
    orchestrator.py location: 10_apps/claude_runtime/src/creator_research/orchestrator.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_orchestrator_config_path() -> Path:
    return _runtime_root() / DEFAULT_ORCHESTRATOR_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class OrchestratorConfig:
    schema_version: str
    fail_fast: bool
    required_sections: tuple[str, ...]


def load_orchestrator_config(config_path: str | Path | None = None) -> OrchestratorConfig:
    path = Path(config_path) if config_path else default_orchestrator_config_path()
    if not path.exists():
        raise ResearchConfigError(f"Orchestrator config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ResearchConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ResearchConfigError(f"Orchestrator config is empty or invalid: {path}")

    section = raw.get("orchestrator") or {}
    return OrchestratorConfig(
        schema_version=str(section.get("schema_version", "1.0")),
        fail_fast=bool(section.get("fail_fast", False)),
        required_sections=tuple(section.get("required_sections", ["profile"])),
    )


def _connector_method(connector: BaseConnector, section: str):
    return getattr(connector, f"collect_{section}")


def run_job(
    job: ResearchJob,
    connector: BaseConnector,
    config: OrchestratorConfig,
    *,
    planner: ResearchPlanner | None = None,
    evidence_queue: EvidenceQueue | None = None,
) -> ResearchSession:
    planner = planner or ResearchPlanner()
    evidence_queue = evidence_queue if evidence_queue is not None else EvidenceQueue()
    progress = ResearchProgress(total_steps=len(JobStatus.WORKING_STATES))
    session = ResearchSession(job=job, progress=progress, connector=connector, evidence_queue=evidence_queue)

    if job.status == JobStatus.QUEUED:
        job.advance(JobStatus.STARTING)

    sections_failed: list[str] = []

    for state in JobStatus.WORKING_STATES:
        progress.mark_step_started(state)
        for section in planner.sections_for_state(job, state):
            session.record_attempt(section)
            method = _connector_method(connector, section)
            try:
                bundle = method(job)
            except Exception as exc:  # connector failures are data, not orchestrator bugs
                warning = f"section {section!r} failed: {exc}"
                session.add_warning(warning)
                sections_failed.append(section)
                if config.fail_fast:
                    job.advance(JobStatus.FAILED)
                    session.report = build_report(job, progress, evidence_queue, sections_failed=sections_failed)
                    return session
                continue
            evidence_queue.add(bundle)
        progress.mark_step_completed(state)
        job.advance(state)

    coverage = evidence_queue.counts_by_section()
    required_met = all(coverage.get(section, 0) > 0 for section in config.required_sections)
    job.advance(JobStatus.COMPLETE if required_met else JobStatus.FAILED)

    session.report = build_report(job, progress, evidence_queue, sections_failed=sections_failed)
    return session
