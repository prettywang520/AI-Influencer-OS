import unittest

from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.models import CreatorDNA, RelationshipBasis, RelationshipClaim, TraitScore
from src.creator_intelligence.validation import validate_creator_dna, validate_evidence


def _evidence(**overrides):
    defaults = dict(evidence_type=EvidenceType.MANUAL_NOTE, source_description="note", content_excerpt="short")
    defaults.update(overrides)
    return Evidence(**defaults)


class ValidateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()

    def test_valid_evidence_passes(self):
        result = validate_evidence(_evidence(), self.config)
        self.assertTrue(result.passed)
        self.assertEqual(result.failed_checks, [])

    def test_over_length_excerpt_fails(self):
        too_long = "x" * (self.config.caption_excerpt_max_chars + 1)
        result = validate_evidence(_evidence(content_excerpt=too_long), self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("content_excerpt" in check for check in result.failed_checks))

    def test_excerpt_at_exact_cap_passes(self):
        exact = "x" * self.config.caption_excerpt_max_chars
        result = validate_evidence(_evidence(content_excerpt=exact), self.config)
        self.assertTrue(result.passed)

    def test_empty_source_description_fails(self):
        result = validate_evidence(_evidence(source_description="   "), self.config)
        self.assertFalse(result.passed)

    def test_over_length_source_description_fails(self):
        too_long = "x" * 501
        result = validate_evidence(_evidence(source_description=too_long), self.config)
        self.assertFalse(result.passed)

    def test_summary_reflects_pass_state(self):
        passed = validate_evidence(_evidence(), self.config)
        self.assertEqual(passed.summary, "PASSED")
        failed = validate_evidence(_evidence(source_description=""), self.config)
        self.assertTrue(failed.summary.startswith("FAILED"))


class ValidateCreatorDnaTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()

    def _dna(self, **overrides):
        defaults = dict(
            subject_label="Demo Creator X",
            evidence_index=["e1"],
            persona=TraitScore(trait_name="persona", score=0.5, confidence=ConfidenceLevel.MEDIUM, evidence_ids=["e1"]),
            overall_confidence=ConfidenceLevel.MEDIUM,
        )
        defaults.update(overrides)
        return CreatorDNA(**defaults)

    def test_valid_dna_passes(self):
        result = validate_creator_dna(self._dna(), self.config)
        self.assertTrue(result.passed)

    def test_score_out_of_bounds_fails(self):
        bad_trait = TraitScore(trait_name="persona", score=1.5, confidence=ConfidenceLevel.MEDIUM, evidence_ids=["e1"])
        result = validate_creator_dna(self._dna(persona=bad_trait), self.config)
        self.assertFalse(result.passed)

    def test_unrecognized_confidence_fails(self):
        bad_trait = TraitScore(trait_name="persona", score=0.5, confidence="extremely_sure", evidence_ids=["e1"])
        result = validate_creator_dna(self._dna(persona=bad_trait), self.config)
        self.assertFalse(result.passed)

    def test_non_unknown_confidence_without_evidence_fails(self):
        bad_trait = TraitScore(trait_name="persona", score=0.5, confidence=ConfidenceLevel.MEDIUM, evidence_ids=[])
        result = validate_creator_dna(self._dna(persona=bad_trait), self.config)
        self.assertFalse(result.passed)

    def test_evidence_id_not_in_index_fails(self):
        bad_trait = TraitScore(
            trait_name="persona", score=0.5, confidence=ConfidenceLevel.MEDIUM, evidence_ids=["not_indexed"]
        )
        result = validate_creator_dna(self._dna(persona=bad_trait, evidence_index=["e1"]), self.config)
        self.assertFalse(result.passed)

    def test_none_traits_are_skipped_not_flagged(self):
        result = validate_creator_dna(self._dna(visual_realism=None), self.config)
        self.assertTrue(result.passed)

    def test_unrecognized_relationship_basis_fails(self):
        claim = RelationshipClaim(description="x", basis="fabricated_basis", evidence_ids=["e1"])
        result = validate_creator_dna(self._dna(relationship_claims=[claim]), self.config)
        self.assertFalse(result.passed)

    def test_valid_relationship_basis_passes(self):
        claim = RelationshipClaim(description="x", basis=RelationshipBasis.INFERRED, evidence_ids=["e1"])
        result = validate_creator_dna(self._dna(relationship_claims=[claim]), self.config)
        self.assertTrue(result.passed)

    def test_relationship_claim_evidence_id_not_in_index_fails(self):
        claim = RelationshipClaim(description="x", basis=RelationshipBasis.INFERRED, evidence_ids=["missing"])
        result = validate_creator_dna(self._dna(relationship_claims=[claim], evidence_index=["e1"]), self.config)
        self.assertFalse(result.passed)

    def test_unrecognized_overall_confidence_fails(self):
        result = validate_creator_dna(self._dna(overall_confidence="super_sure"), self.config)
        self.assertFalse(result.passed)

    def test_summary_reflects_failure_count(self):
        bad_trait = TraitScore(trait_name="persona", score=1.5, confidence="bogus", evidence_ids=[])
        result = validate_creator_dna(self._dna(persona=bad_trait, evidence_index=[]), self.config)
        self.assertFalse(result.passed)
        self.assertTrue(result.summary.startswith("FAILED"))


if __name__ == "__main__":
    unittest.main()
