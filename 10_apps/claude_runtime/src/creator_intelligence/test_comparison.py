import unittest

from src.creator_intelligence.comparison import build_comparison_ready_profile
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.summary_builder import build_creator_dna


def _dna(config):
    item = Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="note", tags=["persona"])
    return build_creator_dna("Demo Creator X", [item], config)


class ComparisonReadyProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.dna = _dna(self.config)

    def test_subject_label_preserved(self):
        profile = build_comparison_ready_profile(self.dna)
        self.assertEqual(profile.subject_label, self.dna.subject_label)

    def test_dna_id_preserved(self):
        profile = build_comparison_ready_profile(self.dna)
        self.assertEqual(profile.dna_id, self.dna.dna_id)

    def test_overall_confidence_preserved(self):
        profile = build_comparison_ready_profile(self.dna)
        self.assertEqual(profile.overall_confidence, self.dna.overall_confidence)

    def test_traits_dict_contains_persona(self):
        profile = build_comparison_ready_profile(self.dna)
        self.assertEqual(profile.traits["persona"], self.dna.persona)

    def test_traits_dict_covers_all_thirteen_categories(self):
        profile = build_comparison_ready_profile(self.dna)
        self.assertEqual(len(profile.traits), 13)

    def test_relationship_claims_preserved(self):
        profile = build_comparison_ready_profile(self.dna)
        self.assertEqual(profile.relationship_claims, self.dna.relationship_claims)

    def test_reshape_performs_no_comparison_logic(self):
        # The module exposes no diff/similarity/rank function of any kind.
        from src.creator_intelligence import comparison as comparison_module

        for forbidden in ("compare", "diff", "similarity", "rank"):
            self.assertFalse(
                any(forbidden in name.lower() for name in dir(comparison_module) if not name.startswith("_"))
            )


if __name__ == "__main__":
    unittest.main()
