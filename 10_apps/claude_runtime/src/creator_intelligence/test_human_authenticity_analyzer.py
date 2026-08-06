import unittest

from src.creator_intelligence.analyzer_base import AnalyzerContext
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.human_authenticity_analyzer import HumanAuthenticityAnalyzer
from src.creator_intelligence.models import RelationshipBasis


def _evidence(tags, source_description="claim source", collected_by="operator_a"):
    return Evidence(
        evidence_type=EvidenceType.MANUAL_NOTE,
        source_description=source_description,
        collected_by=collected_by,
        tags=list(tags),
    )


class HumanAuthenticityAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.analyzer = HumanAuthenticityAnalyzer()

    def test_name_is_human_authenticity(self):
        self.assertEqual(self.analyzer.name, "human_authenticity")

    def test_no_evidence_is_unknown(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)

    def test_human_authenticity_tagged_evidence_scores_the_trait(self):
        item = _evidence(["human_authenticity"])
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [item.evidence_id])


class RelationshipClaimExtractionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.analyzer = HumanAuthenticityAnalyzer()

    def test_no_relationship_evidence_yields_no_claims(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        self.assertEqual(self.analyzer.extract_relationship_claims(context), [])

    def test_childhood_photo_narrative_is_presented_narrative_only(self):
        item = _evidence(
            ["relationship", "presented_narrative"], source_description="childhood photo caption"
        )
        context = AnalyzerContext(evidence=[item], config=self.config)
        claims = self.analyzer.extract_relationship_claims(context)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].basis, RelationshipBasis.PRESENTED_NARRATIVE)

    def test_visually_depicted_tag_maps_to_visually_depicted_basis(self):
        item = _evidence(["relationship", "visually_depicted"], source_description="photo with sibling")
        context = AnalyzerContext(evidence=[item], config=self.config)
        claims = self.analyzer.extract_relationship_claims(context)
        self.assertEqual(claims[0].basis, RelationshipBasis.VISUALLY_DEPICTED)

    def test_no_basis_tag_is_unknown_basis(self):
        item = _evidence(["relationship"], source_description="vague mention")
        context = AnalyzerContext(evidence=[item], config=self.config)
        claims = self.analyzer.extract_relationship_claims(context)
        self.assertEqual(claims[0].basis, RelationshipBasis.UNKNOWN)

    def test_single_verified_tagged_source_is_not_enough_for_verified(self):
        item = _evidence(["relationship", "verified"], source_description="claim x", collected_by="alice")
        context = AnalyzerContext(evidence=[item], config=self.config)
        claims = self.analyzer.extract_relationship_claims(context)
        self.assertNotEqual(claims[0].basis, RelationshipBasis.VERIFIED)

    def test_two_independent_verified_sources_upgrade_to_verified(self):
        items = [
            _evidence(["relationship", "verified"], source_description="claim x", collected_by="alice"),
            _evidence(["relationship", "verified"], source_description="claim x", collected_by="bob"),
        ]
        context = AnalyzerContext(evidence=items, config=self.config)
        claims = self.analyzer.extract_relationship_claims(context)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].basis, RelationshipBasis.VERIFIED)

    def test_two_verified_tagged_items_from_same_source_do_not_upgrade(self):
        items = [
            _evidence(["relationship", "verified"], source_description="claim x", collected_by="alice"),
            _evidence(["relationship", "verified"], source_description="claim x", collected_by="alice"),
        ]
        context = AnalyzerContext(evidence=items, config=self.config)
        claims = self.analyzer.extract_relationship_claims(context)
        self.assertNotEqual(claims[0].basis, RelationshipBasis.VERIFIED)

    def test_different_descriptions_produce_separate_claims(self):
        items = [
            _evidence(["relationship"], source_description="claim x"),
            _evidence(["relationship"], source_description="claim y"),
        ]
        context = AnalyzerContext(evidence=items, config=self.config)
        claims = self.analyzer.extract_relationship_claims(context)
        self.assertEqual(len(claims), 2)

    def test_claim_evidence_ids_include_all_grouped_items(self):
        items = [
            _evidence(["relationship"], source_description="claim x", collected_by="alice"),
            _evidence(["relationship"], source_description="claim x", collected_by="bob"),
        ]
        context = AnalyzerContext(evidence=items, config=self.config)
        claims = self.analyzer.extract_relationship_claims(context)
        self.assertEqual(len(claims), 1)
        self.assertEqual(set(claims[0].evidence_ids), {item.evidence_id for item in items})

    def test_non_relationship_evidence_excluded_from_claims(self):
        item = _evidence(["persona"], source_description="unrelated")
        context = AnalyzerContext(evidence=[item], config=self.config)
        self.assertEqual(self.analyzer.extract_relationship_claims(context), [])


if __name__ == "__main__":
    unittest.main()
