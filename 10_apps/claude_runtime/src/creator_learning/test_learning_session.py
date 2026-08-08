import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_learning.knowledge_base import CreatorKnowledgeBase, evidence_to_dict
from src.creator_learning.learning_session import run_learning_session

_INTELLIGENCE_CONFIG = load_framework_config()


def _bundle_path(tmp, items):
    path = Path(tmp) / "evidence_bundle.json"
    path.write_text(json.dumps([evidence_to_dict(item) for item in items]))
    return path


def _note(text, tags=None):
    return Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description=text, content_excerpt=text, tags=tags or [])


class RunLearningSessionTests(unittest.TestCase):
    def test_first_session_is_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            bundle = _bundle_path(tmp, [_note("first note")])
            result = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG, intake_evidence_bundle_path=bundle)
            self.assertTrue(result.style_evolution.is_baseline)
            self.assertIsNone(result.previous_dna)
            self.assertEqual(len(result.evidence), 1)

    def test_evidence_counts_before_and_after_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            bundle = _bundle_path(tmp, [_note("a"), _note("b")])
            result = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG, intake_evidence_bundle_path=bundle)
            self.assertEqual(result.history_entry.evidence_count_before, 0)
            self.assertEqual(result.history_entry.evidence_count_after, 2)
            self.assertEqual(result.history_entry.new_evidence_count, 2)

    def test_second_session_with_no_new_evidence_keeps_dna_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            bundle = _bundle_path(tmp, [_note("stable note")])
            first = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG, intake_evidence_bundle_path=bundle)
            second = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG)
            self.assertEqual(first.dna.dna_id, second.dna.dna_id)
            self.assertFalse(second.style_evolution.is_baseline)
            self.assertNotEqual(first.session_id, second.session_id)

    def test_second_session_with_new_evidence_grows_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            bundle1 = _bundle_path(tmp, [_note("first")])
            run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG, intake_evidence_bundle_path=bundle1)

            bundle2_dir = Path(tmp) / "second"
            bundle2_dir.mkdir()
            bundle2 = bundle2_dir / "evidence_bundle.json"
            bundle2.write_text(json.dumps([evidence_to_dict(_note("second"))]))
            second = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG, intake_evidence_bundle_path=bundle2)
            self.assertEqual(len(second.evidence), 2)

    def test_dna_snapshot_and_history_persist_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            bundle = _bundle_path(tmp, [_note("persisted note")])
            result = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG, intake_evidence_bundle_path=bundle)

            reloaded_dna = kb.load_dna_snapshot(result.session_id)
            self.assertEqual(reloaded_dna.dna_id, result.dna.dna_id)

            history = kb.load_learning_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["session_id"], result.session_id)

    def test_learning_state_records_last_session_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            bundle = _bundle_path(tmp, [_note("note")])
            result = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG, intake_evidence_bundle_path=bundle)
            state = kb.load_learning_state()
            self.assertEqual(state["last_session_id"], result.session_id)
            self.assertEqual(kb.latest_dna().dna_id, result.dna.dna_id)

    def test_gaps_reflect_uncovered_learning_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            bundle = _bundle_path(tmp, [_note("coffee note", tags=["coffee"])])
            result = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG, intake_evidence_bundle_path=bundle)
            self.assertNotIn("coffee", result.gaps["storytelling"])
            self.assertIn("travel", result.gaps["storytelling"])

    def test_zero_evidence_session_still_produces_unknown_confidence_dna(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            result = run_learning_session(kb, "subject", _INTELLIGENCE_CONFIG)
            self.assertEqual(result.dna.overall_confidence, "unknown")
            self.assertEqual(result.evidence, [])


if __name__ == "__main__":
    unittest.main()
