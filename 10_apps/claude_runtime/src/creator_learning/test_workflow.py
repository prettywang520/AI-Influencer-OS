import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.creator_learning.workflow import default_workflow_config_path, load_workflow_config, main
from src.creator_learning.workflow_exceptions import WorkflowConfigError


class DefaultConfigPathTests(unittest.TestCase):
    def test_default_config_path_exists_and_loads(self):
        path = default_workflow_config_path()
        self.assertTrue(path.is_file(), f"expected {path} to exist")
        config = load_workflow_config()
        self.assertEqual(config.schema_version, "1.0")
        self.assertGreaterEqual(config.max_retry_attempts, 1)


class LoadWorkflowConfigTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(WorkflowConfigError):
                load_workflow_config(Path(tmp) / "does_not_exist.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.yaml"
            path.write_text("workflow: [unterminated\n")
            with self.assertRaises(WorkflowConfigError):
                load_workflow_config(path)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.yaml"
            path.write_text("")
            with self.assertRaises(WorkflowConfigError):
                load_workflow_config(path)

    def test_missing_individual_keys_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.yaml"
            path.write_text("workflow: {}\n")
            config = load_workflow_config(path)
            self.assertEqual(config.schema_version, "1.0")
            self.assertEqual(config.max_retry_attempts, 3)
            self.assertTrue(config.generate_reports_on_completion)

    def test_custom_values_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.yaml"
            path.write_text(
                "workflow:\n  schema_version: '2.0'\n  max_retry_attempts: 7\n  generate_reports_on_completion: false\n"
            )
            config = load_workflow_config(path)
            self.assertEqual(config.schema_version, "2.0")
            self.assertEqual(config.max_retry_attempts, 7)
            self.assertFalse(config.generate_reports_on_completion)


class _FakeResult:
    def __init__(self):
        self.workflow_id = "fake0000workflow"
        self.state = "completed"

        class _Checkpoint:
            def to_dict(self_inner):
                return {"workflow_id": "fake0000workflow", "state": "completed"}

        self.checkpoint = _Checkpoint()


class _FakeRunner:
    def __init__(self, *args, **kwargs):
        pass

    def start(self, creator_url, *, intake_evidence_bundle_path=None):
        self.creator_url = creator_url
        return _FakeResult()


class MainEntryPointTests(unittest.TestCase):
    """Exercises main()'s control flow via a patched WorkflowRunner --
    this module's __main__ entry point must never touch the real
    output/creator_learning/ directory during tests."""

    def test_requires_creator_url(self):
        with self.assertRaises(SystemExit):
            main([])

    @patch("src.creator_learning.workflow_runner.WorkflowRunner", _FakeRunner)
    def test_success_path_prints_and_returns_zero(self):
        rc = main(["--creator-url", "https://www.instagram.com/demo/", "--json"])
        self.assertEqual(rc, 0)

    @patch("src.creator_learning.workflow_runner.WorkflowRunner", _FakeRunner)
    def test_human_readable_output_mode(self):
        rc = main(["--creator-url", "https://www.instagram.com/demo/"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
