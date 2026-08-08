import unittest

from src.creator_intelligence.models import CreatorDNA, TraitScore
from src.creator_learning.learning_history import (
    LearningHistoryEntry,
    build_learning_history_entry,
)


class BuildLearningHistoryEntryTests(unittest.TestCase):
    def test_captures_counts_and_dna_identity(self):
        dna = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.5, confidence="medium"))
        entry = build_learning_history_entry(
            session_id="s1", triggered_at="2026-01-01T00:00:00+00:00",
            evidence_count_before=2, evidence_count_after=5, dna=dna,
        )
        self.assertEqual(entry.session_id, "s1")
        self.assertEqual(entry.new_evidence_count, 3)
        self.assertEqual(entry.dna_id, dna.dna_id)
        self.assertEqual(entry.overall_confidence, dna.overall_confidence)

    def test_per_trait_maps_only_include_populated_traits(self):
        dna = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.5, confidence="medium"))
        entry = build_learning_history_entry(
            session_id="s1", triggered_at="t", evidence_count_before=0, evidence_count_after=1, dna=dna,
        )
        self.assertEqual(entry.per_trait_confidence, {"persona": "medium"})
        self.assertEqual(entry.per_trait_score, {"persona": 0.5})
        self.assertNotIn("visual_realism", entry.per_trait_confidence)

    def test_no_populated_traits_yields_empty_maps(self):
        dna = CreatorDNA(subject_label="c1")
        entry = build_learning_history_entry(
            session_id="s1", triggered_at="t", evidence_count_before=0, evidence_count_after=0, dna=dna,
        )
        self.assertEqual(entry.per_trait_confidence, {})
        self.assertEqual(entry.per_trait_score, {})


class LearningHistoryEntryRoundTripTests(unittest.TestCase):
    def test_to_dict_from_dict_round_trip(self):
        original = LearningHistoryEntry(
            session_id="s1", triggered_at="t", evidence_count_before=1, evidence_count_after=2,
            new_evidence_count=1, dna_id="abc123", overall_confidence="low",
            per_trait_confidence={"persona": "low"}, per_trait_score={"persona": 0.1},
        )
        restored = LearningHistoryEntry.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_from_dict_defaults_missing_trait_maps(self):
        payload = {
            "session_id": "s1", "triggered_at": "t", "evidence_count_before": 0,
            "evidence_count_after": 0, "new_evidence_count": 0, "dna_id": "x", "overall_confidence": "unknown",
        }
        restored = LearningHistoryEntry.from_dict(payload)
        self.assertEqual(restored.per_trait_confidence, {})
        self.assertEqual(restored.per_trait_score, {})


if __name__ == "__main__":
    unittest.main()
