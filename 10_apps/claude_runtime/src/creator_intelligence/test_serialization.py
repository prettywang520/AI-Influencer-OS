import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence import serialization
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.models import CreatorDNA, RelationshipBasis, RelationshipClaim, TraitScore
from src.creator_intelligence.serialization import (
    CreatorIntelligenceSerializationError,
    creator_dna_from_dict,
    creator_dna_to_dict,
    load_creator_dna,
    load_creator_dna_yaml,
    save_creator_dna,
    save_creator_dna_yaml,
)


def _dna():
    return CreatorDNA(
        subject_label="Demo Creator X",
        evidence_index=["e1"],
        persona=TraitScore(trait_name="persona", score=0.5, confidence=ConfidenceLevel.MEDIUM, evidence_ids=["e1"]),
        relationship_claims=[RelationshipClaim(description="x", basis=RelationshipBasis.INFERRED, evidence_ids=["e1"])],
        overall_confidence=ConfidenceLevel.MEDIUM,
    )


class SerializationTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class DictRoundTripTests(unittest.TestCase):
    def test_to_dict_contains_dna_id(self):
        payload = creator_dna_to_dict(_dna())
        self.assertIn("dna_id", payload)

    def test_from_dict_reconstructs_equal_dna_id(self):
        dna = _dna()
        payload = creator_dna_to_dict(dna)
        rebuilt = creator_dna_from_dict(payload)
        self.assertEqual(rebuilt.dna_id, dna.dna_id)

    def test_from_dict_reconstructs_relationship_claims(self):
        dna = _dna()
        payload = creator_dna_to_dict(dna)
        rebuilt = creator_dna_from_dict(payload)
        self.assertEqual(len(rebuilt.relationship_claims), 1)
        self.assertEqual(rebuilt.relationship_claims[0].basis, RelationshipBasis.INFERRED)

    def test_from_dict_rejects_tampered_dna_id(self):
        payload = creator_dna_to_dict(_dna())
        payload["dna_id"] = "0" * 16
        with self.assertRaises(CreatorIntelligenceSerializationError):
            creator_dna_from_dict(payload)

    def test_from_dict_missing_dna_id_key_is_tolerated(self):
        payload = creator_dna_to_dict(_dna())
        del payload["dna_id"]
        rebuilt = creator_dna_from_dict(payload)
        self.assertTrue(rebuilt.dna_id)


class JsonRoundTripTests(SerializationTempTestCase):
    def test_save_creates_file(self):
        path = self.tmp_path / "dna.json"
        save_creator_dna(_dna(), path)
        self.assertTrue(path.is_file())

    def test_save_is_valid_json(self):
        path = self.tmp_path / "dna.json"
        save_creator_dna(_dna(), path)
        json.loads(path.read_text(encoding="utf-8"))

    def test_load_round_trips_dna_id(self):
        dna = _dna()
        path = self.tmp_path / "dna.json"
        save_creator_dna(dna, path)
        loaded = load_creator_dna(path)
        self.assertEqual(loaded.dna_id, dna.dna_id)

    def test_load_missing_file_raises(self):
        with self.assertRaises(CreatorIntelligenceSerializationError):
            load_creator_dna(self.tmp_path / "does_not_exist.json")

    def test_load_invalid_json_raises(self):
        path = self.tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(CreatorIntelligenceSerializationError):
            load_creator_dna(path)

    def test_save_creates_parent_directories(self):
        path = self.tmp_path / "nested" / "dir" / "dna.json"
        save_creator_dna(_dna(), path)
        self.assertTrue(path.is_file())

    def test_save_leaves_no_temp_file_behind(self):
        path = self.tmp_path / "dna.json"
        save_creator_dna(_dna(), path)
        remaining = list(self.tmp_path.iterdir())
        self.assertEqual(remaining, [path])


class YamlRoundTripTests(SerializationTempTestCase):
    def test_save_and_load_round_trips_when_pyyaml_available(self):
        if serialization._yaml is None:
            self.skipTest("PyYAML not installed in this environment")
        dna = _dna()
        path = self.tmp_path / "dna.yaml"
        save_creator_dna_yaml(dna, path)
        loaded = load_creator_dna_yaml(path)
        self.assertEqual(loaded.dna_id, dna.dna_id)

    def test_save_raises_clean_error_when_pyyaml_missing(self):
        original = serialization._yaml
        serialization._yaml = None
        try:
            with self.assertRaises(CreatorIntelligenceSerializationError):
                save_creator_dna_yaml(_dna(), self.tmp_path / "dna.yaml")
        finally:
            serialization._yaml = original

    def test_load_raises_clean_error_when_pyyaml_missing(self):
        original = serialization._yaml
        serialization._yaml = None
        try:
            with self.assertRaises(CreatorIntelligenceSerializationError):
                load_creator_dna_yaml(self.tmp_path / "dna.yaml")
        finally:
            serialization._yaml = original

    def test_missing_pyyaml_error_never_attempts_install(self):
        original = serialization._yaml
        serialization._yaml = None
        try:
            with self.assertRaises(CreatorIntelligenceSerializationError) as ctx:
                save_creator_dna_yaml(_dna(), self.tmp_path / "dna.yaml")
            self.assertNotIn("pip install", str(ctx.exception))
        finally:
            serialization._yaml = original


if __name__ == "__main__":
    unittest.main()
