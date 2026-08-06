import unittest

from src.creator_intelligence.analyzer_base import AnalyzerContext
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.persona_analyzer import PersonaAnalyzer


def _evidence(tags):
    return Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="note", tags=list(tags))


class PersonaAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.analyzer = PersonaAnalyzer()

    def test_name_is_persona(self):
        self.assertEqual(self.analyzer.name, "persona")

    def test_no_evidence_is_unknown(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)

    def test_persona_tagged_evidence_is_used(self):
        item = _evidence(["persona"])
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [item.evidence_id])

    def test_unrelated_evidence_is_ignored(self):
        context = AnalyzerContext(evidence=[_evidence(["caption"])], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [])

    def test_trait_name_matches_analyzer(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.trait_name, "persona")

    def test_no_warnings_produced(self):
        context = AnalyzerContext(evidence=[_evidence(["persona"])], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.warnings, [])


if __name__ == "__main__":
    unittest.main()
