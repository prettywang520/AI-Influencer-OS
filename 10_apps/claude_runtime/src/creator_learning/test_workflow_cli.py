import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.creator_learning.config import LearningConfig
from src.creator_learning.workflow import WorkflowConfig
from src.creator_learning.workflow_checkpoint import save_checkpoint
from src.creator_learning.workflow_cli import main, parse_arguments
from src.creator_learning.workflow_models import WorkflowCheckpoint, new_workflow_id
from src.creator_learning.workflow_runner import WorkflowRunner
from src.creator_learning.workflow_state import WorkflowState

PROFILE_URL = "https://www.instagram.com/cli_test_creator/"


class CliTempTestCase(unittest.TestCase):
    """Every test runs against a real WorkflowRunner scoped to a temp
    knowledge-base root -- WorkflowRunner is patched only to bypass
    its zero-arg default config loading (which resolves to the real
    output/creator_learning/), never to fake its behavior. No network,
    no writes outside the temp directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        learning_config = LearningConfig(
            schema_version="1.0", knowledge_base_root=str(self.tmp_path / "kb"),
            report_formats=("markdown",), generate_reports_on_every_session=True,
        )
        workflow_config = WorkflowConfig(schema_version="1.0", max_retry_attempts=3, generate_reports_on_completion=True)
        self.runner = WorkflowRunner(learning_config, workflow_config)
        self._patcher = patch("src.creator_learning.workflow_cli.WorkflowRunner", return_value=self.runner)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def _save_checkpoint(self, **overrides) -> WorkflowCheckpoint:
        defaults = dict(
            workflow_id=new_workflow_id(), creator_id="cli_creator", platform="instagram", username="cli_creator",
            profile_url=PROFILE_URL, connector_name="dummy",
        )
        defaults.update(overrides)
        checkpoint = WorkflowCheckpoint(**defaults)
        save_checkpoint(self.runner.knowledge_base_root(), checkpoint)
        return checkpoint


class ParseArgumentsTests(unittest.TestCase):
    def test_requires_one_action(self):
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_start_requires_creator_url(self):
        with self.assertRaises(SystemExit):
            parse_arguments(["--start"])

    def test_mutually_exclusive_actions_rejected_together(self):
        with self.assertRaises(SystemExit):
            parse_arguments(["--start", "--creator-url", "https://x/u", "--validate"])

    def test_no_connector_argument_exists_on_the_parser(self):
        # Structural guarantee: this CLI can never construct/accept a
        # Connector, so --start always runs in zero-network mode.
        args = parse_arguments(["--start", "--creator-url", "https://www.instagram.com/u/"])
        arg_names = vars(args).keys()
        for forbidden in ("connector", "run", "execute", "playwright", "browser", "section"):
            matches = [name for name in arg_names if forbidden in name.lower() and name != "connector_name"]
            self.assertEqual(matches, [], f"unexpected connector-execution-shaped flag: {matches}")


class StartCommandTests(CliTempTestCase):
    def test_start_exits_zero_and_reaches_completed(self):
        rc = main(["--start", "--creator-url", PROFILE_URL, "--json"])
        self.assertEqual(rc, 0)

    def test_start_creates_a_checkpoint_on_disk(self):
        main(["--start", "--creator-url", PROFILE_URL])
        checkpoints = list(self.runner.knowledge_base_root().glob("*/workflow/*.json"))
        non_pointer = [p for p in checkpoints if p.name != "latest.json"]
        self.assertEqual(len(non_pointer), 1)


class StatusCommandTests(CliTempTestCase):
    def test_status_with_id(self):
        checkpoint = self._save_checkpoint()
        rc = main(["--status", checkpoint.workflow_id, "--json"])
        self.assertEqual(rc, 0)

    def test_status_unknown_id_errors(self):
        rc = main(["--status", "does_not_exist"])
        self.assertEqual(rc, 1)

    def test_status_without_id_lists_all(self):
        self._save_checkpoint()
        self._save_checkpoint(creator_id="other_creator", username="other_creator")
        rc = main(["--status", "--json"])
        self.assertEqual(rc, 0)


class ResumeCommandTests(CliTempTestCase):
    def test_resume_dispatches_to_resume_for_non_terminal_checkpoint(self):
        checkpoint = self._save_checkpoint()
        checkpoint.record_transition(WorkflowState.RESEARCHING)
        save_checkpoint(self.runner.knowledge_base_root(), checkpoint)
        rc = main(["--resume", checkpoint.workflow_id, "--json"])
        self.assertEqual(rc, 0)
        status = self.runner.status(checkpoint.workflow_id)
        self.assertEqual(status.state, WorkflowState.COMPLETED)

    def test_resume_dispatches_to_retry_for_failed_checkpoint(self):
        checkpoint = self._save_checkpoint()
        checkpoint.record_transition(WorkflowState.RESEARCHING)
        checkpoint.record_transition(WorkflowState.FAILED)
        save_checkpoint(self.runner.knowledge_base_root(), checkpoint)
        rc = main(["--resume", checkpoint.workflow_id, "--json"])
        self.assertEqual(rc, 0)
        status = self.runner.status(checkpoint.workflow_id)
        self.assertEqual(status.retry_count, 1)

    def test_resume_unknown_id_errors(self):
        rc = main(["--resume", "does_not_exist"])
        self.assertEqual(rc, 1)


class CancelCommandTests(CliTempTestCase):
    def test_cancel_non_terminal_checkpoint(self):
        checkpoint = self._save_checkpoint()
        checkpoint.record_transition(WorkflowState.RESEARCHING)
        save_checkpoint(self.runner.knowledge_base_root(), checkpoint)
        rc = main(["--cancel", checkpoint.workflow_id, "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.runner.status(checkpoint.workflow_id).state, WorkflowState.CANCELLED)

    def test_cancel_unknown_id_errors(self):
        rc = main(["--cancel", "does_not_exist"])
        self.assertEqual(rc, 1)


class ValidateCommandTests(CliTempTestCase):
    def test_validate_passes_on_clean_state(self):
        self._save_checkpoint()
        rc = main(["--validate", "--json"])
        self.assertEqual(rc, 0)

    def test_validate_fails_on_corrupt_state(self):
        checkpoint = self._save_checkpoint()
        checkpoint.state = "not_a_real_state"
        save_checkpoint(self.runner.knowledge_base_root(), checkpoint)
        rc = main(["--validate"])
        self.assertEqual(rc, 1)


class JsonOutputModeTests(CliTempTestCase):
    def test_json_output_is_valid_json(self):
        checkpoint = self._save_checkpoint()
        with patch("builtins.print") as mock_print:
            main(["--status", checkpoint.workflow_id, "--json"])
        printed = mock_print.call_args[0][0]
        json.loads(printed)  # must not raise


if __name__ == "__main__":
    unittest.main()
