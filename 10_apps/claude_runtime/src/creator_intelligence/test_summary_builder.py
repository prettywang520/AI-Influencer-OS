import random
import unittest

from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.models import RelationshipBasis
from src.creator_intelligence.summary_builder import build_creator_dna


def _evidence(tags, collected_by="operator_a", source_description="note", excerpt=""):
    return Evidence(
        evidence_type=EvidenceType.MANUAL_NOTE,
        source_description=source_description,
        content_excerpt=excerpt,
        collected_by=collected_by,
        tags=list(tags),
    )


class BuildCreatorDNATests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()

    def test_empty_evidence_yields_unknown_overall_confidence(self):
        dna = build_creator_dna("Demo Creator X", [], self.config)
        self.assertEqual(dna.overall_confidence, ConfidenceLevel.UNKNOWN)

    def test_subject_label_is_preserved_exactly(self):
        dna = build_creator_dna("Demo Creator X", [], self.config)
        self.assertEqual(dna.subject_label, "Demo Creator X")

    def test_all_trait_fields_populated_even_with_no_evidence(self):
        dna = build_creator_dna("Demo Creator X", [], self.config)
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
            self.assertIsNotNone(getattr(dna, name))

    def test_persona_evidence_reflected_in_persona_trait(self):
        item = _evidence(["persona"])
        dna = build_creator_dna("Demo Creator X", [item], self.config)
        self.assertIn(item.evidence_id, dna.persona.evidence_ids)

    def test_evidence_index_includes_every_cited_evidence_id(self):
        items = [_evidence(["persona"]), _evidence(["caption"])]
        dna = build_creator_dna("Demo Creator X", items, self.config)
        for item in items:
            self.assertIn(item.evidence_id, dna.evidence_index)

    def test_overall_confidence_is_weakest_link(self):
        # Give persona lots of well-corroborated evidence but leave
        # every other trait bare -- overall_confidence must stay
        # UNKNOWN because the weakest trait has none.
        items = [_evidence(["persona"], collected_by=f"op_{i}") for i in range(6)]
        dna = build_creator_dna("Demo Creator X", items, self.config)
        self.assertNotEqual(dna.persona.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(dna.overall_confidence, ConfidenceLevel.UNKNOWN)

    def test_relationship_claims_extracted_via_human_authenticity_analyzer(self):
        item = _evidence(["relationship", "presented_narrative"], source_description="childhood photo")
        dna = build_creator_dna("Demo Creator X", [item], self.config)
        self.assertEqual(len(dna.relationship_claims), 1)
        self.assertEqual(dna.relationship_claims[0].basis, RelationshipBasis.PRESENTED_NARRATIVE)

    def test_relationship_claim_evidence_ids_included_in_evidence_index(self):
        item = _evidence(["relationship"], source_description="claim")
        dna = build_creator_dna("Demo Creator X", [item], self.config)
        self.assertIn(item.evidence_id, dna.evidence_index)

    def test_caption_excerpt_over_cap_produces_warning(self):
        too_long = "x" * (self.config.caption_excerpt_max_chars + 1)
        item = _evidence(["caption"], excerpt=too_long)
        dna = build_creator_dna("Demo Creator X", [item], self.config)
        self.assertTrue(any(item.evidence_id in warning for warning in dna.warnings))

    def test_dna_id_stable_across_repeated_runs_on_identical_evidence(self):
        items = [_evidence(["persona"]), _evidence(["caption"])]
        first = build_creator_dna("Demo Creator X", items, self.config)
        second = build_creator_dna("Demo Creator X", items, self.config)
        self.assertEqual(first.dna_id, second.dna_id)

    def test_dna_id_stable_regardless_of_evidence_order(self):
        items = [_evidence(["persona"], collected_by="a"), _evidence(["caption"], collected_by="b")]
        shuffled = list(items)
        random.Random(42).shuffle(shuffled)
        first = build_creator_dna("Demo Creator X", items, self.config)
        second = build_creator_dna("Demo Creator X", shuffled, self.config)
        self.assertEqual(first.dna_id, second.dna_id)

    def test_different_subject_label_changes_dna_id(self):
        items = [_evidence(["persona"])]
        first = build_creator_dna("Creator A", items, self.config)
        second = build_creator_dna("Creator B", items, self.config)
        self.assertNotEqual(first.dna_id, second.dna_id)

    def test_no_real_creator_handle_ever_required(self):
        # subject_label accepts any operator-chosen text; the builder
        # never requires or injects a real account handle.
        dna = build_creator_dna("Anonymous Subject 1", [], self.config)
        self.assertNotIn("instagram.com", dna.subject_label)


if __name__ == "__main__":
    unittest.main()
