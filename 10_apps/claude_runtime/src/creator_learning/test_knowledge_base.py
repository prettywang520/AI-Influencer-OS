import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.models import CreatorDNA, TraitScore
from src.creator_learning.exceptions import KnowledgeBaseError
from src.creator_learning.knowledge_base import CreatorKnowledgeBase, evidence_from_dict, evidence_to_dict


def _note(text="a note", tags=None):
    return Evidence(
        evidence_type=EvidenceType.MANUAL_NOTE, source_description=text, content_excerpt=text, tags=tags or []
    )


class EvidenceDictRoundTripTests(unittest.TestCase):
    def test_round_trip_preserves_identity(self):
        original = _note("round trip note", tags=["coffee"])
        restored = evidence_from_dict(evidence_to_dict(original))
        self.assertEqual(original.evidence_id, restored.evidence_id)
        self.assertEqual(original.tags, restored.tags)


class EvidenceStoreTests(unittest.TestCase):
    def test_empty_store_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            self.assertEqual(kb.load_evidence(), [])

    def test_merge_persists_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            e1, e2 = _note("first"), _note("second")
            merged = kb.merge_evidence([e1, e2])
            self.assertEqual(len(merged), 2)

            merged_again = kb.merge_evidence([e1])
            self.assertEqual(len(merged_again), 2)
            self.assertEqual({item.evidence_id for item in merged_again}, {e1.evidence_id, e2.evidence_id})

    def test_merge_never_drops_existing_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            kb.merge_evidence([_note("keep me")])
            kb.merge_evidence([_note("add me too")])
            self.assertEqual(len(kb.load_evidence()), 2)

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            kb.evidence_store_path.parent.mkdir(parents=True, exist_ok=True)
            kb.evidence_store_path.write_text("{not valid json")
            with self.assertRaises(KnowledgeBaseError):
                kb.load_evidence()


class DnaSnapshotTests(unittest.TestCase):
    def test_missing_snapshot_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            self.assertIsNone(kb.load_dna_snapshot("nonexistent"))

    def test_save_and_load_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            dna = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.4, confidence="low"))
            kb.save_dna_snapshot("session1", dna)
            loaded = kb.load_dna_snapshot("session1")
            self.assertEqual(loaded.dna_id, dna.dna_id)

    def test_latest_dna_uses_learning_state_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            self.assertIsNone(kb.latest_dna())
            dna = CreatorDNA(subject_label="c1")
            kb.save_dna_snapshot("session1", dna)
            kb.save_learning_state({"last_session_id": "session1"})
            self.assertEqual(kb.latest_dna().dna_id, dna.dna_id)


class LearningHistoryPersistenceTests(unittest.TestCase):
    def test_empty_history_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            self.assertEqual(kb.load_learning_history(), [])

    def test_append_is_additive_never_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            kb.append_learning_history({"session_id": "s1"})
            kb.append_learning_history({"session_id": "s2"})
            history = kb.load_learning_history()
            self.assertEqual([entry["session_id"] for entry in history], ["s1", "s2"])


class LearningStateTests(unittest.TestCase):
    def test_missing_state_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            self.assertEqual(kb.load_learning_state(), {})

    def test_save_and_load_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            kb.save_learning_state({"last_session_id": "s1", "evidence_count": 3})
            self.assertEqual(kb.load_learning_state(), {"last_session_id": "s1", "evidence_count": 3})


class ReportPersistenceTests(unittest.TestCase):
    def test_missing_report_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            self.assertIsNone(kb.load_report("session1", "learning"))

    def test_save_and_load_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            path = kb.save_report("session1", "learning", "# Learning Report\n")
            self.assertTrue(path.is_file())
            self.assertEqual(kb.load_report("session1", "learning"), "# Learning Report\n")

    def test_reports_are_scoped_per_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            kb.save_report("session1", "learning", "session one")
            kb.save_report("session2", "learning", "session two")
            self.assertEqual(kb.load_report("session1", "learning"), "session one")
            self.assertEqual(kb.load_report("session2", "learning"), "session two")


class PathScopingTests(unittest.TestCase):
    def test_root_scoped_under_creator_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator_xyz")
            self.assertEqual(kb.root, Path(tmp) / "creator_xyz")
            self.assertTrue(str(kb.evidence_store_path).startswith(str(kb.root)))
            self.assertTrue(str(kb.dna_snapshot_path("s1")).startswith(str(kb.root)))
            self.assertTrue(str(kb.reports_directory("s1")).startswith(str(kb.root)))


if __name__ == "__main__":
    unittest.main()
