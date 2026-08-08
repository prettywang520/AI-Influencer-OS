import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_learning.config import LearningConfig
from src.creator_learning.workflow import WorkflowConfig
from src.creator_learning.workflow_checkpoint import load_checkpoint, save_checkpoint
from src.creator_learning.workflow_exceptions import (
    WorkflowAlreadyTerminalError,
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowRetryLimitExceededError,
)
from src.creator_learning.workflow_models import WorkflowCheckpoint, new_workflow_id
from src.creator_learning.workflow_runner import WorkflowRunner
from src.creator_learning.workflow_state import WorkflowState
from src.creator_research.connector import BaseConnector
from src.creator_research.interfaces import ConnectorSection, EvidenceBundle

PROFILE_URL = "https://www.instagram.com/workflow_test_creator/"


class DummyConnector(BaseConnector):
    """Fake, in-memory Connector -- no browser, no network. Implements
    all 10 collect_* methods with small synthetic EvidenceBundles,
    mirroring creator_research/test_orchestrator.py's FakeConnector."""

    def __init__(self):
        self.calls: list[str] = []

    def _bundle(self, section, n=1):
        self.calls.append(section)
        items = [
            Evidence(evidence_type=EvidenceType.OPERATOR_OBSERVATION, source_description=f"{section} item {i}")
            for i in range(n)
        ]
        return EvidenceBundle(section=section, items=items)

    def collect_profile(self, job):
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


class FlakyConnector(BaseConnector):
    """Fails on the first collect_profile() call, succeeds after."""

    def __init__(self):
        self.attempts = 0

    def collect_profile(self, job):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("simulated failure")
        return EvidenceBundle(
            section=ConnectorSection.PROFILE,
            items=[Evidence(evidence_type=EvidenceType.OPERATOR_OBSERVATION, source_description="ok now")],
        )


class AlwaysFailsConnector(BaseConnector):
    def collect_profile(self, job):
        raise RuntimeError("always fails")


def _runner(tmp, *, max_retry_attempts=3):
    learning_config = LearningConfig(
        schema_version="1.0", knowledge_base_root=str(Path(tmp) / "kb"),
        report_formats=("markdown",), generate_reports_on_every_session=True,
    )
    workflow_config = WorkflowConfig(
        schema_version="1.0", max_retry_attempts=max_retry_attempts, generate_reports_on_completion=True,
    )
    return WorkflowRunner(learning_config, workflow_config)


class StartHappyPathTests(unittest.TestCase):
    def test_reaches_completed_through_all_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            connector = DummyConnector()
            result = runner.start(PROFILE_URL, connector=connector, requested_sections=(ConnectorSection.PROFILE,))
            self.assertEqual(result.state, WorkflowState.COMPLETED)
            self.assertEqual(
                [step["state"] for step in result.checkpoint.step_history],
                ["researching", "learning", "knowledge_updated", "dna_updated", "reporting", "completed"],
            )

    def test_only_requested_section_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            connector = DummyConnector()
            runner.start(PROFILE_URL, connector=connector, requested_sections=(ConnectorSection.PROFILE,))
            self.assertEqual(connector.calls, ["profile"])

    def test_reports_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            self.assertEqual(
                set(result.reports.keys()), {"learning", "visual", "relationship", "reply", "caption", "style"}
            )

    def test_workflow_report_saved_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            path = runner.knowledge_base_root() / result.creator_id / "workflow_report.json"
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"], "completed")

    def test_creator_dna_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            path = runner.knowledge_base_root() / result.creator_id / "creator_dna.json"
            self.assertTrue(path.is_file())

    def test_checkpoint_persisted_and_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            loaded = load_checkpoint(runner.knowledge_base_root(), result.creator_id, result.workflow_id)
            self.assertEqual(loaded, result.checkpoint)

    def test_workflow_id_fresh_and_unique_per_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            first = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            second = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            self.assertNotEqual(first.workflow_id, second.workflow_id)
            self.assertEqual(len(first.workflow_id), 16)

    def test_zero_connector_mode_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL)
            self.assertEqual(result.state, WorkflowState.COMPLETED)
            self.assertEqual(result.dna.overall_confidence, "unknown")

    def test_deterministic_dna_across_independent_runs(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            result_a = _runner(tmp_a).start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            result_b = _runner(tmp_b).start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            self.assertEqual(result_a.dna.dna_id, result_b.dna.dna_id)


class ResearchFailurePropagationTests(unittest.TestCase):
    def test_connector_failure_marks_workflow_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            with self.assertRaises(WorkflowError):
                runner.start(PROFILE_URL, connector=AlwaysFailsConnector(), requested_sections=(ConnectorSection.PROFILE,))

            root = runner.knowledge_base_root()
            files = [p for p in root.glob("*/workflow/*.json") if p.name != "latest.json"]
            self.assertEqual(len(files), 1)
            checkpoint = load_checkpoint(root, json.loads(files[0].read_text())["creator_id"], json.loads(files[0].read_text())["workflow_id"])
            self.assertEqual(checkpoint.state, WorkflowState.FAILED)
            self.assertTrue(checkpoint.errors)

    def test_failure_report_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            with self.assertRaises(WorkflowError):
                runner.start(PROFILE_URL, connector=AlwaysFailsConnector(), requested_sections=(ConnectorSection.PROFILE,))
            root = runner.knowledge_base_root()
            report_paths = list(root.glob("*/workflow_report.json"))
            self.assertEqual(len(report_paths), 1)
            payload = json.loads(report_paths[0].read_text())
            self.assertEqual(payload["state"], "failed")
            self.assertTrue(payload["errors"])


class RetryTests(unittest.TestCase):
    def _failed_workflow_id(self, runner, connector):
        with self.assertRaises(WorkflowError):
            runner.start(PROFILE_URL, connector=connector, requested_sections=(ConnectorSection.PROFILE,))
        root = runner.knowledge_base_root()
        files = [p for p in root.glob("*/workflow/*.json") if p.name != "latest.json"]
        return json.loads(files[0].read_text())["workflow_id"]

    def test_retry_succeeds_after_transient_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            connector = FlakyConnector()
            workflow_id = self._failed_workflow_id(runner, connector)
            result = runner.retry(workflow_id, connector=connector)
            self.assertEqual(result.state, WorkflowState.COMPLETED)
            self.assertEqual(result.checkpoint.retry_count, 1)
            self.assertEqual(connector.attempts, 2)

    def test_retry_on_non_failed_workflow_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            with self.assertRaises(WorkflowError):
                runner.retry(result.workflow_id)

    def test_retry_limit_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp, max_retry_attempts=2)
            connector = AlwaysFailsConnector()
            workflow_id = self._failed_workflow_id(runner, connector)
            for _ in range(2):
                with self.assertRaises(WorkflowError):
                    runner.retry(workflow_id, connector=connector)
            with self.assertRaises(WorkflowRetryLimitExceededError):
                runner.retry(workflow_id, connector=connector)

    def test_retry_unknown_workflow_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            with self.assertRaises(WorkflowNotFoundError):
                runner.retry("does_not_exist")


class ResumeTests(unittest.TestCase):
    def test_resume_never_repeats_a_finished_research_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            connector = DummyConnector()
            first = runner.start(PROFILE_URL, connector=connector, requested_sections=(ConnectorSection.PROFILE,))
            self.assertEqual(connector.calls, ["profile"])

            # Simulate a crash right after DNA_UPDATED was durably
            # saved, before REPORTING ever ran: a consistent
            # checkpoint (state == last_completed_step == dna_updated).
            root = runner.knowledge_base_root()
            loaded = load_checkpoint(root, first.creator_id, first.workflow_id)
            loaded.state = WorkflowState.DNA_UPDATED
            loaded.last_completed_step = WorkflowState.DNA_UPDATED
            loaded.step_history = [s for s in loaded.step_history if s["state"] != "reporting" and s["state"] != "completed"]
            save_checkpoint(root, loaded)

            second = runner.resume(first.workflow_id, connector=connector)
            self.assertEqual(connector.calls, ["profile"])  # not called again
            self.assertEqual(second.state, WorkflowState.COMPLETED)
            self.assertEqual(second.checkpoint.resume_count, 1)

    def test_resume_on_completed_workflow_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            with self.assertRaises(WorkflowAlreadyTerminalError):
                runner.resume(result.workflow_id)

    def test_resume_on_failed_workflow_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            with self.assertRaises(WorkflowError):
                runner.start(PROFILE_URL, connector=AlwaysFailsConnector(), requested_sections=(ConnectorSection.PROFILE,))
            root = runner.knowledge_base_root()
            files = [p for p in root.glob("*/workflow/*.json") if p.name != "latest.json"]
            workflow_id = json.loads(files[0].read_text())["workflow_id"]
            with self.assertRaises(WorkflowAlreadyTerminalError):
                runner.resume(workflow_id)

    def test_resume_unknown_workflow_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            with self.assertRaises(WorkflowNotFoundError):
                runner.resume("does_not_exist")


class RestartTests(unittest.TestCase):
    def test_restart_deliberately_reruns_research(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            connector = DummyConnector()
            first = runner.start(PROFILE_URL, connector=connector, requested_sections=(ConnectorSection.PROFILE,))
            self.assertEqual(connector.calls, ["profile"])

            second = runner.restart(first.workflow_id, connector=connector)
            self.assertEqual(connector.calls, ["profile", "profile"])
            self.assertEqual(second.state, WorkflowState.COMPLETED)
            self.assertEqual(second.checkpoint.restart_count, 1)


class CancelTests(unittest.TestCase):
    def test_cancel_before_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            pending = WorkflowCheckpoint(
                workflow_id=new_workflow_id(), creator_id="cancel_me", platform="instagram", username="cancel_me",
                profile_url=PROFILE_URL, connector_name="dummy",
            )
            pending.record_transition(WorkflowState.RESEARCHING)
            save_checkpoint(runner.knowledge_base_root(), pending)

            result = runner.cancel(pending.workflow_id)
            self.assertEqual(result.state, WorkflowState.CANCELLED)
            self.assertTrue(result.checkpoint.cancelled)

    def test_cancel_already_terminal_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            with self.assertRaises(WorkflowAlreadyTerminalError):
                runner.cancel(result.workflow_id)

    def test_cancel_unknown_workflow_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            with self.assertRaises(WorkflowNotFoundError):
                runner.cancel("does_not_exist")


class StatusTests(unittest.TestCase):
    def test_status_returns_current_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _runner(tmp)
            result = runner.start(PROFILE_URL, connector=DummyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            status = runner.status(result.workflow_id)
            self.assertEqual(status.state, WorkflowState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
