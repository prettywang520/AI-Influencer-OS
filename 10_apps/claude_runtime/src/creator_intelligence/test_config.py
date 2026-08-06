import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence.config import (
    CreatorIntelligenceConfigError,
    default_evidence_types_config_path,
    default_framework_config_path,
    load_framework_config,
)
from src.creator_intelligence.evidence import EvidenceType


class DefaultPathTests(unittest.TestCase):
    def test_default_framework_config_path_exists_on_disk(self):
        self.assertTrue(default_framework_config_path().is_file())

    def test_default_evidence_types_config_path_exists_on_disk(self):
        self.assertTrue(default_evidence_types_config_path().is_file())

    def test_default_paths_under_config_creator_intelligence(self):
        self.assertIn("creator_intelligence", str(default_framework_config_path()))


class LoadFrameworkConfigTests(unittest.TestCase):
    def test_loads_real_default_config_successfully(self):
        config = load_framework_config()
        self.assertEqual(config.schema_version, "1.0")

    def test_confidence_thresholds_are_positive(self):
        config = load_framework_config()
        self.assertGreater(config.confidence_min_evidence_for_medium, 0)
        self.assertGreater(config.confidence_min_evidence_for_high, 0)
        self.assertGreater(config.confidence_min_corroboration_for_verified, 0)

    def test_allowed_evidence_types_is_subset_of_evidence_type_all(self):
        config = load_framework_config()
        self.assertTrue(set(config.allowed_evidence_types).issubset(set(EvidenceType.ALL)))

    def test_weight_for_known_analyzer_returns_configured_value(self):
        config = load_framework_config()
        self.assertEqual(config.weight_for("persona"), config.analyzer_weights.get("persona", 1.0))

    def test_weight_for_unknown_analyzer_defaults_to_one(self):
        config = load_framework_config()
        self.assertEqual(config.weight_for("not_a_real_analyzer"), 1.0)

    def test_missing_framework_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "does_not_exist.yaml"
            with self.assertRaises(CreatorIntelligenceConfigError):
                load_framework_config(missing, default_evidence_types_config_path())

    def test_missing_evidence_types_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "does_not_exist.yaml"
            with self.assertRaises(CreatorIntelligenceConfigError):
                load_framework_config(default_framework_config_path(), missing)

    def test_empty_framework_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty = Path(tmp_dir) / "empty.yaml"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(CreatorIntelligenceConfigError):
                load_framework_config(empty, default_evidence_types_config_path())

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad = Path(tmp_dir) / "bad.yaml"
            bad.write_text("framework: [unclosed", encoding="utf-8")
            with self.assertRaises(CreatorIntelligenceConfigError):
                load_framework_config(bad, default_evidence_types_config_path())

    def test_evidence_types_with_unknown_entry_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad = Path(tmp_dir) / "evidence_types.yaml"
            bad.write_text("evidence_types:\n  allowed:\n    - scraped\n", encoding="utf-8")
            with self.assertRaises(CreatorIntelligenceConfigError):
                load_framework_config(default_framework_config_path(), bad)

    def test_evidence_types_empty_allowed_list_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad = Path(tmp_dir) / "evidence_types.yaml"
            bad.write_text("evidence_types:\n  allowed: []\n", encoding="utf-8")
            with self.assertRaises(CreatorIntelligenceConfigError):
                load_framework_config(default_framework_config_path(), bad)

    def test_custom_framework_file_is_honored(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom = Path(tmp_dir) / "framework.yaml"
            custom.write_text(
                "framework:\n  schema_version: '2.0'\nconfidence:\n  min_evidence_for_medium: 9\n"
                "  min_evidence_for_high: 10\n  min_corroboration_for_verified: 3\nevidence:\n  excerpt_max_chars: 50\n",
                encoding="utf-8",
            )
            config = load_framework_config(custom, default_evidence_types_config_path())
            self.assertEqual(config.schema_version, "2.0")
            self.assertEqual(config.confidence_min_evidence_for_medium, 9)
            self.assertEqual(config.caption_excerpt_max_chars, 50)


if __name__ == "__main__":
    unittest.main()
