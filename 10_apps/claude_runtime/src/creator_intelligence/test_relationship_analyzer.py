import unittest

from src.creator_intelligence.analyzer_base import AnalyzerContext
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.relationship_analyzer import RelationshipAnalyzer


def _evidence(tags):
    return Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="note", tags=list(tags))


class RelationshipAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.analyzer = RelationshipAnalyzer()

    def test_name_is_relationships(self):
        self.assertEqual(self.analyzer.name, "relationships")

    def test_no_evidence_is_unknown(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)

    def test_relationship_tagged_evidence_is_used(self):
        item = _evidence(["relationship"])
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [item.evidence_id])

    def test_unrelated_evidence_is_ignored(self):
        context = AnalyzerContext(evidence=[_evidence(["persona"])], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [])

    def test_does_not_produce_relationship_claims_itself(self):
        # Claim extraction is owned exclusively by HumanAuthenticityAnalyzer.
        self.assertFalse(hasattr(self.analyzer, "extract_relationship_claims"))


if __name__ == "__main__":
    unittest.main()
