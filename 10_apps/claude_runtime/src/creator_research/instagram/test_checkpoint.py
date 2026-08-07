import json
import tempfile
import unittest
from pathlib import Path

from src.creator_research.instagram.checkpoint import (
    CheckpointMismatchError,
    InstagramCheckpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from src.creator_research.instagram.exceptions import CheckpointError


def _checkpoint(**overrides):
    defaults = dict(job_id="job1", creator_id="c1", connector_version="12B.2", section="grid")
    defaults.update(overrides)
    return InstagramCheckpoint(**defaults)


class CheckpointTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class SaveLoadTests(CheckpointTempTestCase):
    def test_save_creates_file(self):
        save_checkpoint(self.tmp_path, _checkpoint())
        self.assertTrue((self.tmp_path / "job1_grid.json").is_file())

    def test_load_round_trips_fields(self):
        checkpoint = _checkpoint(processed_source_ids=["a", "b"], evidence_ids=["e1"], scroll_round=3)
        save_checkpoint(self.tmp_path, checkpoint)
        loaded = load_checkpoint(self.tmp_path, "job1", "grid")
        self.assertEqual(loaded.processed_source_ids, ["a", "b"])
        self.assertEqual(loaded.evidence_ids, ["e1"])
        self.assertEqual(loaded.scroll_round, 3)

    def test_load_missing_returns_none(self):
        self.assertIsNone(load_checkpoint(self.tmp_path, "does_not_exist", "grid"))

    def test_load_malformed_json_raises(self):
        path = self.tmp_path / "job1_grid.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid", encoding="utf-8")
        with self.assertRaises(CheckpointError):
            load_checkpoint(self.tmp_path, "job1", "grid")

    def test_separate_sections_get_separate_files(self):
        save_checkpoint(self.tmp_path, _checkpoint(section="grid"))
        save_checkpoint(self.tmp_path, _checkpoint(section="captions"))
        self.assertTrue((self.tmp_path / "job1_grid.json").is_file())
        self.assertTrue((self.tmp_path / "job1_captions.json").is_file())

    def test_save_is_atomic_no_temp_file_left_behind(self):
        save_checkpoint(self.tmp_path, _checkpoint())
        remaining = list(self.tmp_path.iterdir())
        self.assertEqual(remaining, [self.tmp_path / "job1_grid.json"])

    def test_saved_file_is_valid_json(self):
        save_checkpoint(self.tmp_path, _checkpoint())
        json.loads((self.tmp_path / "job1_grid.json").read_text(encoding="utf-8"))


class ValidateCheckpointTests(unittest.TestCase):
    def test_matching_identity_passes_no_warnings(self):
        checkpoint = _checkpoint()
        warnings = validate_checkpoint(checkpoint, job_id="job1", creator_id="c1", connector_version="12B.2")
        self.assertEqual(warnings, [])

    def test_mismatched_job_id_raises(self):
        checkpoint = _checkpoint(job_id="job1")
        with self.assertRaises(CheckpointMismatchError):
            validate_checkpoint(checkpoint, job_id="job2", creator_id="c1", connector_version="12B.2")

    def test_mismatched_creator_id_raises(self):
        checkpoint = _checkpoint(creator_id="c1")
        with self.assertRaises(CheckpointMismatchError):
            validate_checkpoint(checkpoint, job_id="job1", creator_id="c2", connector_version="12B.2")

    def test_mismatched_connector_version_warns_but_does_not_raise(self):
        checkpoint = _checkpoint(connector_version="12B.1")
        warnings = validate_checkpoint(checkpoint, job_id="job1", creator_id="c1", connector_version="12B.2")
        self.assertEqual(len(warnings), 1)
        self.assertIn("connector_version", warnings[0])

    def test_never_silently_adopts_a_mismatched_checkpoint(self):
        checkpoint = _checkpoint(job_id="other_job")
        with self.assertRaises(CheckpointMismatchError):
            validate_checkpoint(checkpoint, job_id="job1", creator_id="c1", connector_version="12B.2")


class ResumeScenarioTests(CheckpointTempTestCase):
    def test_resume_preserves_already_collected_evidence(self):
        first = _checkpoint(processed_source_ids=["p1", "p2"], evidence_ids=["e1", "e2"])
        save_checkpoint(self.tmp_path, first)

        loaded = load_checkpoint(self.tmp_path, "job1", "grid")
        validate_checkpoint(loaded, job_id="job1", creator_id="c1", connector_version="12B.2")
        # simulate continuing collection: new items found, merged with existing
        loaded.processed_source_ids = sorted(set(loaded.processed_source_ids) | {"p3"})
        loaded.evidence_ids = sorted(set(loaded.evidence_ids) | {"e3"})
        save_checkpoint(self.tmp_path, loaded)

        final = load_checkpoint(self.tmp_path, "job1", "grid")
        self.assertEqual(set(final.processed_source_ids), {"p1", "p2", "p3"})
        self.assertEqual(set(final.evidence_ids), {"e1", "e2", "e3"})

    def test_mismatched_creator_resume_fails_before_reusing_any_data(self):
        checkpoint = _checkpoint(creator_id="real_creator", processed_source_ids=["p1"])
        save_checkpoint(self.tmp_path, checkpoint)
        loaded = load_checkpoint(self.tmp_path, "job1", "grid")
        with self.assertRaises(CheckpointMismatchError):
            validate_checkpoint(loaded, job_id="job1", creator_id="wrong_creator", connector_version="12B.2")

    def test_incomplete_section_resumes_from_scroll_round(self):
        checkpoint = _checkpoint(scroll_round=7)
        save_checkpoint(self.tmp_path, checkpoint)
        loaded = load_checkpoint(self.tmp_path, "job1", "grid")
        self.assertEqual(loaded.scroll_round, 7)


class ToFromDictTests(unittest.TestCase):
    def test_round_trip(self):
        checkpoint = _checkpoint(warnings=["w1"], last_processed_source="https://x/1")
        payload = checkpoint.to_dict()
        rebuilt = InstagramCheckpoint.from_dict(payload)
        self.assertEqual(rebuilt.warnings, ["w1"])
        self.assertEqual(rebuilt.last_processed_source, "https://x/1")

    def test_from_dict_defaults_missing_optional_fields(self):
        rebuilt = InstagramCheckpoint.from_dict({"job_id": "j", "creator_id": "c", "section": "grid"})
        self.assertEqual(rebuilt.processed_source_ids, [])
        self.assertEqual(rebuilt.scroll_round, 0)


if __name__ == "__main__":
    unittest.main()
