"""WorkflowRunner -- the stateful, checkpointed driver connecting
Creator Research -> Creator Learning -> Creator DNA -> Reports.
Composes creator_research.orchestrator.run_job(),
learning_session.run_learning_session(), and
reports.builder/reports.markdown -- all unmodified -- around a
persisted WorkflowCheckpoint. Never opens a browser itself: a
Connector is only ever run if a caller supplies one directly to
start()/resume()/retry()/restart(), through the exact same
orchestrator.run_job() path Phase 12B.1/12B.2 already gated.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from src.creator_intelligence.config import CreatorIntelligenceConfig, load_framework_config
from src.creator_intelligence.models import CreatorDNA
from src.creator_intelligence.serialization import creator_dna_to_dict
from src.creator_research.connector import BaseConnector
from src.creator_research.interfaces import ConnectorSection
from src.creator_research.jobs import ResearchJob
from src.creator_research.orchestrator import load_orchestrator_config, run_job
from src.creator_research.state import JobStatus

from .config import LearningConfig, load_learning_config
from .creator_url import parse_creator_url
from .engine import REPORT_NAMES
from .knowledge_base import CreatorKnowledgeBase
from .learning_history import LearningHistoryEntry
from .learning_session import run_learning_session
from .reports.builder import build_all_reports
from .reports.markdown import render_all_reports
from .style_evolution import StyleEvolutionRecord, diff_creator_dna
from .workflow import WorkflowConfig, load_workflow_config
from .workflow_checkpoint import find_checkpoint, load_checkpoint, save_checkpoint
from .workflow_exceptions import (
    WorkflowAlreadyTerminalError,
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowRetryLimitExceededError,
)
from .workflow_models import WorkflowCheckpoint, new_workflow_id
from .workflow_report import WorkflowReport, build_workflow_report, save_workflow_report
from .workflow_state import WorkflowState, validate_transition

_RESEARCHING_INDEX = WorkflowState.ORDER.index(WorkflowState.RESEARCHING)
_LEARNING_INDEX = WorkflowState.ORDER.index(WorkflowState.LEARNING)
_DNA_UPDATED_INDEX = WorkflowState.ORDER.index(WorkflowState.DNA_UPDATED)
_REPORTING_INDEX = WorkflowState.ORDER.index(WorkflowState.REPORTING)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        Path(tmp_name).replace(path)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


def _current_index(checkpoint: WorkflowCheckpoint) -> int:
    if checkpoint.last_completed_step is None:
        return 0
    return WorkflowState.ORDER.index(checkpoint.last_completed_step)


@dataclass(slots=True)
class WorkflowRunResult:
    workflow_id: str
    creator_id: str
    state: str
    checkpoint: WorkflowCheckpoint
    dna: CreatorDNA | None = None
    reports: dict[str, str] = field(default_factory=dict)
    report: WorkflowReport | None = None


class WorkflowRunner:
    def __init__(
        self,
        learning_config: LearningConfig | None = None,
        workflow_config: WorkflowConfig | None = None,
        *,
        intelligence_config: CreatorIntelligenceConfig | None = None,
    ) -> None:
        self.learning_config = learning_config or load_learning_config()
        self.workflow_config = workflow_config or load_workflow_config()
        self.intelligence_config = intelligence_config or load_framework_config()

    def knowledge_base_root(self) -> Path:
        return self.learning_config.resolved_knowledge_base_root()

    def _kb_root(self) -> Path:
        return self.knowledge_base_root()

    def _kb_for(self, creator_id: str) -> CreatorKnowledgeBase:
        return CreatorKnowledgeBase(self._kb_root(), creator_id)

    def _save(self, checkpoint: WorkflowCheckpoint) -> None:
        save_checkpoint(self._kb_root(), checkpoint)

    def _workflow_report_path(self, creator_id: str) -> Path:
        return self._kb_root() / creator_id / "workflow_report.json"

    def _require_checkpoint(self, workflow_id: str, creator_id: str | None) -> WorkflowCheckpoint:
        if creator_id is not None:
            checkpoint = load_checkpoint(self._kb_root(), creator_id, workflow_id)
        else:
            checkpoint = find_checkpoint(self._kb_root(), workflow_id)
        if checkpoint is None:
            raise WorkflowNotFoundError(f"No workflow found with workflow_id={workflow_id!r}")
        return checkpoint

    def _export_creator_dna(self, kb: CreatorKnowledgeBase, dna: CreatorDNA | None) -> None:
        if dna is None:
            return
        _atomic_write_text(kb.root / "creator_dna.json", json.dumps(creator_dna_to_dict(dna), indent=2, sort_keys=True))

    def _style_evolution_for(self, kb: CreatorKnowledgeBase, checkpoint: WorkflowCheckpoint) -> StyleEvolutionRecord:
        history = kb.load_learning_history()
        session_ids = [entry["session_id"] for entry in history]
        if checkpoint.learning_session_id in session_ids:
            index = session_ids.index(checkpoint.learning_session_id)
            previous_session_id = session_ids[index - 1] if index > 0 else None
        else:
            previous_session_id = session_ids[-2] if len(session_ids) >= 2 else None
        previous_dna = kb.load_dna_snapshot(previous_session_id) if previous_session_id else None
        current_dna = (
            kb.load_dna_snapshot(checkpoint.learning_session_id) if checkpoint.learning_session_id else kb.latest_dna()
        )
        return diff_creator_dna(previous_dna, current_dna)

    # -- public API -----------------------------------------------------

    def start(
        self,
        profile_url: str,
        *,
        connector: BaseConnector | None = None,
        requested_sections: tuple[str, ...] | None = None,
        connector_name: str = "unassigned",
        intake_evidence_bundle_path: str | Path | None = None,
    ) -> WorkflowRunResult:
        parsed = parse_creator_url(profile_url)
        checkpoint = WorkflowCheckpoint(
            workflow_id=new_workflow_id(),
            creator_id=parsed.creator_id,
            platform=parsed.platform,
            username=parsed.username,
            profile_url=parsed.profile_url,
            connector_name=connector.__class__.__name__ if connector is not None else connector_name,
            requested_sections=requested_sections or (),
        )
        self._save(checkpoint)
        return self._run_from(checkpoint, connector=connector, intake_evidence_bundle_path=intake_evidence_bundle_path)

    def resume(
        self,
        workflow_id: str,
        creator_id: str | None = None,
        *,
        connector: BaseConnector | None = None,
        intake_evidence_bundle_path: str | Path | None = None,
    ) -> WorkflowRunResult:
        """For a workflow whose checkpoint is non-terminal (the
        process died mid-run without ever reaching FAILED). Raises
        WorkflowAlreadyTerminalError for COMPLETED/CANCELLED/FAILED --
        a FAILED workflow must go through retry() instead."""
        checkpoint = self._require_checkpoint(workflow_id, creator_id)
        if checkpoint.state in WorkflowState.TERMINAL:
            raise WorkflowAlreadyTerminalError(
                f"workflow_id={workflow_id!r} is already {checkpoint.state!r}; "
                "cannot resume (use retry() if FAILED, restart() to redo from scratch)"
            )
        checkpoint.resume_count += 1
        self._save(checkpoint)
        return self._run_from(checkpoint, connector=connector, intake_evidence_bundle_path=intake_evidence_bundle_path)

    def retry(
        self,
        workflow_id: str,
        creator_id: str | None = None,
        *,
        connector: BaseConnector | None = None,
        intake_evidence_bundle_path: str | Path | None = None,
    ) -> WorkflowRunResult:
        """For a workflow whose checkpoint reached FAILED. Resets to
        the step after last_completed_step (same pattern as
        ResearchQueue.retry() resetting FAILED -> QUEUED directly),
        capped by workflow.yaml's max_retry_attempts."""
        checkpoint = self._require_checkpoint(workflow_id, creator_id)
        if checkpoint.state != WorkflowState.FAILED:
            raise WorkflowError(f"only FAILED workflows can be retried (workflow_id={workflow_id!r} is {checkpoint.state!r})")
        if checkpoint.retry_count >= self.workflow_config.max_retry_attempts:
            raise WorkflowRetryLimitExceededError(
                f"workflow_id={workflow_id!r} has exceeded max_retry_attempts={self.workflow_config.max_retry_attempts}"
            )
        checkpoint.retry_count += 1
        checkpoint.state = checkpoint.last_completed_step or WorkflowState.CREATED
        self._save(checkpoint)
        return self._run_from(checkpoint, connector=connector, intake_evidence_bundle_path=intake_evidence_bundle_path)

    def restart(
        self,
        workflow_id: str,
        creator_id: str | None = None,
        *,
        connector: BaseConnector | None = None,
        requested_sections: tuple[str, ...] | None = None,
        intake_evidence_bundle_path: str | Path | None = None,
    ) -> WorkflowRunResult:
        """Deliberate full re-run from CREATED, keeping the same
        workflow_id and creator identity. The one operation that
        intentionally repeats already-finished steps, because it was
        explicitly requested."""
        checkpoint = self._require_checkpoint(workflow_id, creator_id)
        checkpoint.restart_count += 1
        checkpoint.research_job_id = None
        checkpoint.learning_session_id = None
        checkpoint.last_completed_step = None
        checkpoint.errors = []
        checkpoint.cancelled = False
        checkpoint.state = WorkflowState.CREATED
        if requested_sections is not None:
            checkpoint.requested_sections = requested_sections
        self._save(checkpoint)
        return self._run_from(checkpoint, connector=connector, intake_evidence_bundle_path=intake_evidence_bundle_path)

    def cancel(self, workflow_id: str, creator_id: str | None = None) -> WorkflowRunResult:
        checkpoint = self._require_checkpoint(workflow_id, creator_id)
        if checkpoint.state in WorkflowState.TERMINAL:
            raise WorkflowAlreadyTerminalError(f"workflow_id={workflow_id!r} is already {checkpoint.state!r}; cannot cancel")
        validate_transition(checkpoint.state, WorkflowState.CANCELLED)
        checkpoint.cancelled = True
        checkpoint.record_transition(WorkflowState.CANCELLED)
        self._save(checkpoint)

        kb = self._kb_for(checkpoint.creator_id)
        dna = kb.latest_dna()
        report = build_workflow_report(checkpoint, dna=dna)
        save_workflow_report(self._workflow_report_path(checkpoint.creator_id), report)
        return WorkflowRunResult(
            workflow_id=checkpoint.workflow_id, creator_id=checkpoint.creator_id, state=checkpoint.state,
            checkpoint=checkpoint, dna=dna, reports={}, report=report,
        )

    def status(self, workflow_id: str, creator_id: str | None = None) -> WorkflowCheckpoint:
        return self._require_checkpoint(workflow_id, creator_id)

    # -- internal driver --------------------------------------------------

    def _run_from(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        connector: BaseConnector | None,
        intake_evidence_bundle_path: str | Path | None,
    ) -> WorkflowRunResult:
        kb = self._kb_for(checkpoint.creator_id)
        session_result = None

        try:
            current_index = _current_index(checkpoint)

            # -- RESEARCHING: only run the connector if this workflow
            # hasn't yet started its learning hand-off. Once LEARNING
            # has been reached, evidence from a fresh research run
            # would never be picked up anyway (run_learning_session()
            # has already consumed whatever was collected), so
            # re-running research here would just be a wasted/duplicate
            # live call -- "never repeat finished steps" applies most
            # concretely right here.
            if current_index < _LEARNING_INDEX:
                if checkpoint.state != WorkflowState.RESEARCHING:
                    validate_transition(checkpoint.state, WorkflowState.RESEARCHING)
                    checkpoint.record_transition(WorkflowState.RESEARCHING)
                session = None
                if connector is not None:
                    job = ResearchJob(
                        creator_id=checkpoint.creator_id,
                        platform=checkpoint.platform,
                        username=checkpoint.username,
                        profile_url=checkpoint.profile_url,
                        connector_name=checkpoint.connector_name,
                        requested_sections=checkpoint.requested_sections or ConnectorSection.ALL,
                    )
                    session = run_job(job, connector, load_orchestrator_config())
                    checkpoint.research_job_id = job.job_id
                    checkpoint.warnings.extend(session.warnings)
                    if job.status == JobStatus.FAILED:
                        # orchestrator.run_job() itself never raises on a
                        # connector/section failure (partial results are
                        # still useful for research, by that module's own
                        # design) -- but at the WORKFLOW level, a research
                        # job that failed to meet its required sections is
                        # a real failure the caller needs to see and be
                        # able to retry(), not a silently-degraded
                        # continuation into LEARNING with missing evidence.
                        raise WorkflowError(
                            f"research job {job.job_id!r} did not complete "
                            f"(status={job.status!r}); warnings={session.warnings}"
                        )
                self._save(checkpoint)
                current_index = _RESEARCHING_INDEX
            else:
                session = None

            # -- LEARNING / KNOWLEDGE_UPDATED / DNA_UPDATED: one atomic
            # call into run_learning_session() (unmodified). Persisted
            # as LEARNING *before* the call (so a crash inside it
            # leaves a resumable, re-attemptable state), then
            # KNOWLEDGE_UPDATED + DNA_UPDATED are recorded together
            # immediately after it returns -- see docs/creator_learning/
            # workflow.md for why re-running this call on resume is safe.
            if current_index < _DNA_UPDATED_INDEX:
                if checkpoint.state != WorkflowState.LEARNING:
                    validate_transition(checkpoint.state, WorkflowState.LEARNING)
                    checkpoint.record_transition(WorkflowState.LEARNING)
                    self._save(checkpoint)

                session_result = run_learning_session(
                    kb, checkpoint.username, self.intelligence_config,
                    session=session, intake_evidence_bundle_path=intake_evidence_bundle_path,
                )
                checkpoint.learning_session_id = session_result.session_id

                validate_transition(checkpoint.state, WorkflowState.KNOWLEDGE_UPDATED)
                checkpoint.record_transition(WorkflowState.KNOWLEDGE_UPDATED)
                validate_transition(checkpoint.state, WorkflowState.DNA_UPDATED)
                checkpoint.record_transition(WorkflowState.DNA_UPDATED)
                self._save(checkpoint)
                current_index = _DNA_UPDATED_INDEX

            # -- REPORTING
            rendered: dict[str, str] = {}
            if current_index < _REPORTING_INDEX:
                validate_transition(checkpoint.state, WorkflowState.REPORTING)
                checkpoint.record_transition(WorkflowState.REPORTING)

                dna = kb.latest_dna()
                evidence = kb.load_evidence()
                history = [LearningHistoryEntry.from_dict(entry) for entry in kb.load_learning_history()]
                evolution = self._style_evolution_for(kb, checkpoint)

                report_dataclasses = build_all_reports(
                    creator_id=checkpoint.creator_id, session_id=checkpoint.learning_session_id or "",
                    dna=dna, evidence=evidence, history=history, evolution=evolution,
                )
                rendered = render_all_reports(report_dataclasses)
                if self.workflow_config.generate_reports_on_completion:
                    session_id_for_reports = checkpoint.learning_session_id or checkpoint.workflow_id
                    for name, text in rendered.items():
                        kb.save_report(session_id_for_reports, name, text)
                    self._export_creator_dna(kb, dna)

                self._save(checkpoint)
                current_index = _REPORTING_INDEX
            else:
                dna = kb.latest_dna()
                if checkpoint.learning_session_id:
                    for name in REPORT_NAMES:
                        text = kb.load_report(checkpoint.learning_session_id, name)
                        if text is not None:
                            rendered[name] = text

            # -- COMPLETED
            validate_transition(checkpoint.state, WorkflowState.COMPLETED)
            checkpoint.record_transition(WorkflowState.COMPLETED)

            report = build_workflow_report(
                checkpoint,
                evidence_count_before=session_result.history_entry.evidence_count_before if session_result else None,
                evidence_count_after=session_result.history_entry.evidence_count_after if session_result else None,
                new_evidence_count=session_result.history_entry.new_evidence_count if session_result else None,
                dna=dna,
            )
            save_workflow_report(self._workflow_report_path(checkpoint.creator_id), report)
            self._save(checkpoint)

            return WorkflowRunResult(
                workflow_id=checkpoint.workflow_id, creator_id=checkpoint.creator_id, state=checkpoint.state,
                checkpoint=checkpoint, dna=dna, reports=rendered, report=report,
            )
        except Exception as exc:
            checkpoint.errors.append(str(exc))
            validate_transition(checkpoint.state, WorkflowState.FAILED)
            checkpoint.record_transition(WorkflowState.FAILED)
            self._save(checkpoint)
            failure_report = build_workflow_report(checkpoint, dna=kb.latest_dna())
            save_workflow_report(self._workflow_report_path(checkpoint.creator_id), failure_report)
            raise
