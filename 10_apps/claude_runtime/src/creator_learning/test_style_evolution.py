import unittest

from src.creator_intelligence.models import CreatorDNA, RelationshipClaim, TraitScore
from src.creator_learning.style_evolution import diff_creator_dna


class BaselineDiffTests(unittest.TestCase):
    def test_previous_none_yields_baseline_record(self):
        dna = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.5, confidence="low"))
        record = diff_creator_dna(None, dna)
        self.assertTrue(record.is_baseline)
        self.assertIsNone(record.previous_overall_confidence)
        self.assertEqual(record.current_overall_confidence, dna.overall_confidence)

    def test_baseline_trait_delta_has_no_previous_values(self):
        dna = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.5, confidence="low"))
        record = diff_creator_dna(None, dna)
        delta = record.trait_deltas["persona"]
        self.assertIsNone(delta.previous_score)
        self.assertIsNone(delta.previous_confidence)
        self.assertEqual(delta.current_score, 0.5)
        self.assertFalse(delta.confidence_changed)

    def test_baseline_relationship_claims_are_all_new(self):
        claim = RelationshipClaim(description="has a sibling", basis="presented_narrative", evidence_ids=["e1"])
        dna = CreatorDNA(subject_label="c1", relationship_claims=[claim])
        record = diff_creator_dna(None, dna)
        self.assertEqual(record.new_relationship_claims, [claim])
        self.assertEqual(record.changed_relationship_claims, [])

    def test_baseline_warnings_are_all_new(self):
        dna = CreatorDNA(subject_label="c1", warnings=["low evidence overall"])
        record = diff_creator_dna(None, dna)
        self.assertEqual(record.new_warnings, ["low evidence overall"])


class RealDiffTests(unittest.TestCase):
    def test_score_delta_and_confidence_change_detected(self):
        previous = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.5, confidence="low"))
        current = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.8, confidence="high"))
        record = diff_creator_dna(previous, current)
        self.assertFalse(record.is_baseline)
        delta = record.trait_deltas["persona"]
        self.assertAlmostEqual(delta.score_delta, 0.3)
        self.assertTrue(delta.confidence_changed)
        self.assertEqual(delta.previous_confidence, "low")
        self.assertEqual(delta.current_confidence, "high")

    def test_unchanged_trait_has_zero_delta_and_no_confidence_change(self):
        trait = TraitScore(trait_name="persona", score=0.5, confidence="medium")
        previous = CreatorDNA(subject_label="c1", persona=trait)
        current = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.5, confidence="medium"))
        record = diff_creator_dna(previous, current)
        delta = record.trait_deltas["persona"]
        self.assertEqual(delta.score_delta, 0.0)
        self.assertFalse(delta.confidence_changed)

    def test_trait_appearing_for_the_first_time_has_no_previous_score(self):
        previous = CreatorDNA(subject_label="c1")
        current = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.5, confidence="low"))
        record = diff_creator_dna(previous, current)
        delta = record.trait_deltas["persona"]
        self.assertIsNone(delta.previous_score)
        self.assertIsNone(delta.score_delta)
        self.assertEqual(delta.current_score, 0.5)

    def test_new_relationship_claim_detected(self):
        existing = RelationshipClaim(description="has a sibling", basis="presented_narrative", evidence_ids=["e1"])
        new_claim = RelationshipClaim(description="has a pet dog", basis="visually_depicted", evidence_ids=["e2"])
        previous = CreatorDNA(subject_label="c1", relationship_claims=[existing])
        current = CreatorDNA(subject_label="c1", relationship_claims=[existing, new_claim])
        record = diff_creator_dna(previous, current)
        self.assertEqual(record.new_relationship_claims, [new_claim])
        self.assertEqual(record.changed_relationship_claims, [])

    def test_changed_relationship_claim_evidence_detected(self):
        original = RelationshipClaim(description="has a sibling", basis="presented_narrative", evidence_ids=["e1"])
        updated = RelationshipClaim(description="has a sibling", basis="presented_narrative", evidence_ids=["e1", "e2"])
        previous = CreatorDNA(subject_label="c1", relationship_claims=[original])
        current = CreatorDNA(subject_label="c1", relationship_claims=[updated])
        record = diff_creator_dna(previous, current)
        self.assertEqual(record.new_relationship_claims, [])
        self.assertEqual(record.changed_relationship_claims, [updated])

    def test_unchanged_relationship_claim_is_neither_new_nor_changed(self):
        claim = RelationshipClaim(description="has a sibling", basis="presented_narrative", evidence_ids=["e1"])
        same_claim = RelationshipClaim(description="has a sibling", basis="presented_narrative", evidence_ids=["e1"])
        previous = CreatorDNA(subject_label="c1", relationship_claims=[claim])
        current = CreatorDNA(subject_label="c1", relationship_claims=[same_claim])
        record = diff_creator_dna(previous, current)
        self.assertEqual(record.new_relationship_claims, [])
        self.assertEqual(record.changed_relationship_claims, [])

    def test_new_warnings_detected_and_stale_ones_excluded(self):
        previous = CreatorDNA(subject_label="c1", warnings=["existing warning"])
        current = CreatorDNA(subject_label="c1", warnings=["existing warning", "brand new warning"])
        record = diff_creator_dna(previous, current)
        self.assertEqual(record.new_warnings, ["brand new warning"])

    def test_overall_confidence_carried_through(self):
        previous = CreatorDNA(subject_label="c1", overall_confidence="low")
        current = CreatorDNA(subject_label="c1", overall_confidence="medium")
        record = diff_creator_dna(previous, current)
        self.assertEqual(record.previous_overall_confidence, "low")
        self.assertEqual(record.current_overall_confidence, "medium")


if __name__ == "__main__":
    unittest.main()
