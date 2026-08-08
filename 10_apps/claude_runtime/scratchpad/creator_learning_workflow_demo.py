"""Safe demo for Phase 12C.0.1 -- Creator Learning Workflow.

Entirely synthetic: DummyConnector (10 collect_* methods, all
synthetic EvidenceBundles) run through WorkflowRunner.start() end to
end, then a simulated failure + retry(), then a cancel() on a
separate workflow. Uses a temp directory for the knowledge base --
writes nothing to the real output/creator_learning/. Not part of the
automated test suite.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_learning.config import LearningConfig
from src.creator_learning.workflow import WorkflowConfig
from src.creator_learning.workflow_checkpoint import save_checkpoint
from src.creator_learning.workflow_models import WorkflowCheckpoint, new_workflow_id
from src.creator_learning.workflow_runner import WorkflowRunner
from src.creator_learning.workflow_state import WorkflowState
from src.creator_research.connector import BaseConnector
from src.creator_research.interfaces import ConnectorSection, EvidenceBundle

PROFILE_URL = "https://www.instagram.com/demo_workflow_creator/"


class DummyConnector(BaseConnector):
    """Fake, in-memory Connector -- no browser, no network."""

    def __init__(self, fail_once: bool = False):
        self.calls: list[str] = []
        self._fail_once = fail_once
        self._failed_already = False

    def _bundle(self, section, n=1):
        self.calls.append(section)
        items = [
            Evidence(evidence_type=EvidenceType.OPERATOR_OBSERVATION, source_description=f"{section} item {i}")
            for i in range(n)
        ]
        return EvidenceBundle(section=section, items=items)

    def collect_profile(self, job):
        if self._fail_once and not self._failed_already:
            self._failed_already = True
            raise RuntimeError("simulated transient failure")
        return self._bundle(ConnectorSection.PROFILE)

    def collect_grid(self, job):
        return self._bundle(ConnectorSection.GRID, 2)

    def collect_posts(self, job):
        return self._bundle(ConnectorSection.POSTS, 2)

    def collect_captions(self, job):
        return self._bundle(ConnectorSection.CAPTIONS, 2)

    def collect_comments(self, job):
        return self._bundle(ConnectorSection.COMMENTS, 2)

    def collect_creator_replies(self, job):
        return self._bundle(ConnectorSection.CREATOR_REPLIES, 2)

    def collect_reels(self, job):
        return self._bundle(ConnectorSection.REELS, 1)

    def collect_highlights(self, job):
        return self._bundle(ConnectorSection.HIGHLIGHTS, 1)

    def collect_relationships(self, job):
        return self._bundle(ConnectorSection.RELATIONSHIPS, 1)

    def collect_visual_examples(self, job):
        return self._bundle(ConnectorSection.VISUAL_EXAMPLES, 1)


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    learning_config = LearningConfig(
        schema_version="1.0", knowledge_base_root=str(tmp_path / "kb"),
        report_formats=("markdown",), generate_reports_on_every_session=True,
    )
    workflow_config = WorkflowConfig(schema_version="1.0", max_retry_attempts=3, generate_reports_on_completion=True)
    runner = WorkflowRunner(learning_config, workflow_config)

    print("=== 1. start() -- DummyConnector run through the full state machine ===")
    connector = DummyConnector()
    result = runner.start(PROFILE_URL, connector=connector, requested_sections=(ConnectorSection.PROFILE,))
    print("  workflow_id:", result.workflow_id)
    print("  state:", result.state)
    print("  step_history:", [s["state"] for s in result.checkpoint.step_history])
    print("  reports:", sorted(result.reports.keys()))

    print("\n=== 2. workflow_report.json ===")
    report_path = runner.knowledge_base_root() / result.creator_id / "workflow_report.json"
    print("  path:", report_path)
    print("  ", json.dumps(json.loads(report_path.read_text()), indent=2)[:600])

    print("\n=== 3. Simulated transient failure + retry() ===")
    flaky = DummyConnector(fail_once=True)
    try:
        runner.start(
            "https://www.instagram.com/demo_flaky_creator/", connector=flaky, requested_sections=(ConnectorSection.PROFILE,)
        )
    except Exception as exc:
        print("  start() failed as expected:", exc)

    candidates = [p for p in runner.knowledge_base_root().glob("*/workflow/*.json") if p.name != "latest.json"]
    flaky_checkpoint_path = [p for p in candidates if json.loads(p.read_text())["username"] == "demo_flaky_creator"][0]
    flaky_workflow_id = json.loads(flaky_checkpoint_path.read_text())["workflow_id"]
    print("  state after failure:", runner.status(flaky_workflow_id).state)
    retry_result = runner.retry(flaky_workflow_id, connector=flaky)
    print("  state after retry():", retry_result.state, "retry_count:", retry_result.checkpoint.retry_count)

    print("\n=== 4. cancel() on a fresh, non-terminal workflow ===")
    pending = WorkflowCheckpoint(
        workflow_id=new_workflow_id(), creator_id="demo_cancel_creator", platform="instagram",
        username="demo_cancel_creator", profile_url="https://www.instagram.com/demo_cancel_creator/",
        connector_name="dummy",
    )
    pending.record_transition(WorkflowState.RESEARCHING)
    save_checkpoint(runner.knowledge_base_root(), pending)
    cancel_result = runner.cancel(pending.workflow_id)
    print("  state after cancel():", cancel_result.state)

    print("\n=== 5. resume() correctly rejects an already-completed workflow ===")
    try:
        runner.resume(result.workflow_id)
        print("  BUG: should have raised")
    except Exception as exc:
        print("  correctly raised:", type(exc).__name__)

print("\nDemo complete. Synthetic connector/creator only. No network access. No real creator analyzed.")
