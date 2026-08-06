import unittest

from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.models import CreatorDNA, RelationshipBasis, RelationshipClaim, TraitScore


def _trait(name="persona", score=0.5, confidence=ConfidenceLevel.MEDIUM, evidence_ids=None, rationale="r"):
    return TraitScore(
        trait_name=name,
        score=score,
        confidence=confidence,
        evidence_ids=list(evidence_ids or ["e1"]),
        rationale=rationale,
    )


def _dna(**overrides):
    defaults = dict(subject_label="Demo Creator X", evidence_index=["e1"], persona=_trait())
    defaults.update(overrides)
    return CreatorDNA(**defaults)


class RelationshipBasisTests(unittest.TestCase):
    def test_all_contains_every_named_constant(self):
        self.assertEqual(
            set(RelationshipBasis.ALL),
            {
                RelationshipBasis.PRESENTED_NARRATIVE,
                RelationshipBasis.VISUALLY_DEPICTED,
                RelationshipBasis.VERIFIED,
                RelationshipBasis.INFERRED,
                RelationshipBasis.UNKNOWN,
            },
        )


class TraitScoreTests(unittest.TestCase):
    def test_construction_defaults(self):
        trait = TraitScore(trait_name="persona", score=0.5)
        self.assertEqual(trait.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(trait.evidence_ids, [])
        self.assertEqual(trait.rationale, "")

    def test_default_evidence_ids_not_shared_between_instances(self):
        a = TraitScore(trait_name="a", score=0.1)
        b = TraitScore(trait_name="b", score=0.2)
        a.evidence_ids.append("x")
        self.assertEqual(b.evidence_ids, [])


class RelationshipClaimTests(unittest.TestCase):
    def test_construction(self):
        claim = RelationshipClaim(description="childhood photo narrative", basis=RelationshipBasis.PRESENTED_NARRATIVE)
        self.assertEqual(claim.evidence_ids, [])


class CreatorDNAIdentityTests(unittest.TestCase):
    def test_dna_id_is_populated(self):
        dna = _dna()
        self.assertTrue(dna.dna_id)

    def test_identical_content_yields_identical_id(self):
        first = _dna()
        second = _dna()
        self.assertEqual(first.dna_id, second.dna_id)

    def test_different_generated_at_does_not_change_id(self):
        first = _dna(generated_at="2026-01-01T00:00:00+00:00")
        second = _dna(generated_at="2026-12-31T23:59:59+00:00")
        self.assertEqual(first.dna_id, second.dna_id)

    def test_different_subject_label_changes_id(self):
        first = _dna(subject_label="Creator A")
        second = _dna(subject_label="Creator B")
        self.assertNotEqual(first.dna_id, second.dna_id)

    def test_different_trait_score_changes_id(self):
        first = _dna(persona=_trait(score=0.5))
        second = _dna(persona=_trait(score=0.9))
        self.assertNotEqual(first.dna_id, second.dna_id)

    def test_different_relationship_claims_changes_id(self):
        first = _dna(relationship_claims=[])
        second = _dna(
            relationship_claims=[RelationshipClaim(description="x", basis=RelationshipBasis.INFERRED)]
        )
        self.assertNotEqual(first.dna_id, second.dna_id)

    def test_relationship_claim_order_does_not_change_id(self):
        claim_a = RelationshipClaim(description="a", basis=RelationshipBasis.INFERRED)
        claim_b = RelationshipClaim(description="b", basis=RelationshipBasis.INFERRED)
        first = _dna(relationship_claims=[claim_a, claim_b])
        second = _dna(relationship_claims=[claim_b, claim_a])
        self.assertEqual(first.dna_id, second.dna_id)

    def test_evidence_index_order_does_not_change_id(self):
        first = _dna(evidence_index=["e1", "e2"])
        second = _dna(evidence_index=["e2", "e1"])
        self.assertEqual(first.dna_id, second.dna_id)

    def test_none_trait_differs_from_populated_trait(self):
        first = _dna(visual_realism=None)
        second = _dna(visual_realism=_trait(name="visual_realism"))
        self.assertNotEqual(first.dna_id, second.dna_id)

    def test_default_lists_not_shared_between_instances(self):
        first = _dna()
        second = _dna()
        first.warnings.append("w")
        self.assertEqual(second.warnings, [])


class CreatorDNADefaultsTests(unittest.TestCase):
    def test_all_trait_fields_default_to_none(self):
        dna = CreatorDNA(subject_label="x")
        for name in (
            "persona",
            "visual_realism",
            "photography",
            "human_authenticity",
            "relationships",
            "captions",
            "replies",
            "storytelling",
            "reels",
            "branding",
            "posting",
            "engagement",
            "growth",
        ):
            self.assertIsNone(getattr(dna, name))

    def test_overall_confidence_defaults_to_unknown(self):
        dna = CreatorDNA(subject_label="x")
        self.assertEqual(dna.overall_confidence, ConfidenceLevel.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
