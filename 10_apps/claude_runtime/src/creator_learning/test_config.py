import tempfile
import unittest
from pathlib import Path

from src.creator_learning.config import LearningConfig, default_config_path, load_learning_config
from src.creator_learning.exceptions import LearningConfigError


class DefaultConfigPathTests(unittest.TestCase):
    def test_default_config_path_exists_and_loads(self):
        path = default_config_path()
        self.assertTrue(path.is_file(), f"expected {path} to exist")
        config = load_learning_config()
        self.assertEqual(config.schema_version, "1.0")
        self.assertTrue(config.knowledge_base_root)


class LoadLearningConfigTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LearningConfigError):
                load_learning_config(Path(tmp) / "does_not_exist.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.yaml"
            path.write_text("learning: [unterminated\n")
            with self.assertRaises(LearningConfigError):
                load_learning_config(path)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.yaml"
            path.write_text("")
            with self.assertRaises(LearningConfigError):
                load_learning_config(path)

    def test_missing_individual_keys_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.yaml"
            path.write_text("learning: {}\n")
            config = load_learning_config(path)
            self.assertEqual(config.schema_version, "1.0")
            self.assertEqual(config.knowledge_base_root, "output/creator_learning")
            self.assertEqual(config.report_formats, ("markdown",))
            self.assertTrue(config.generate_reports_on_every_session)

    def test_custom_values_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.yaml"
            path.write_text(
                "learning:\n  schema_version: '2.0'\n"
                "knowledge_base:\n  root_directory: custom/kb\n"
                "reports:\n  formats: [markdown, json]\n  generate_on_every_session: false\n"
            )
            config = load_learning_config(path)
            self.assertEqual(config.schema_version, "2.0")
            self.assertEqual(config.knowledge_base_root, "custom/kb")
            self.assertEqual(config.report_formats, ("markdown", "json"))
            self.assertFalse(config.generate_reports_on_every_session)


class LearningConfigMethodsTests(unittest.TestCase):
    def test_resolved_knowledge_base_root_is_absolute(self):
        config = LearningConfig(
            schema_version="1.0", knowledge_base_root="output/creator_learning",
            report_formats=("markdown",), generate_reports_on_every_session=True,
        )
        resolved = config.resolved_knowledge_base_root()
        self.assertTrue(resolved.is_absolute())
        self.assertTrue(str(resolved).endswith("output/creator_learning"))

    def test_creator_directory_is_scoped_under_root(self):
        config = LearningConfig(
            schema_version="1.0", knowledge_base_root="output/creator_learning",
            report_formats=("markdown",), generate_reports_on_every_session=True,
        )
        creator_dir = config.creator_directory("abc123")
        self.assertEqual(creator_dir, config.resolved_knowledge_base_root() / "abc123")


if __name__ == "__main__":
    unittest.main()
