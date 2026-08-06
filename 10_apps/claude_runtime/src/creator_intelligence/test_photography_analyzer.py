import unittest

from src.creator_intelligence.analyzer_base import AnalyzerContext
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.photography_analyzer import PhotographyAnalyzer


def _evidence(tags):
    return Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="note", tags=list(tags))


class PhotographyAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.analyzer = PhotographyAnalyzer()

    def test_name_is_photography(self):
        self.assertEqual(self.analyzer.name, "photography")

    def test_no_evidence_is_unknown(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)

    def test_photography_tagged_evidence_is_used(self):
        item = _evidence(["photography"])
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [item.evidence_id])

    def test_unrelated_evidence_is_ignored(self):
        context = AnalyzerContext(evidence=[_evidence(["reels"])], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [])

    def test_trait_name_matches_analyzer(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.trait_name, "photography")


if __name__ == "__main__":
    unittest.main()
