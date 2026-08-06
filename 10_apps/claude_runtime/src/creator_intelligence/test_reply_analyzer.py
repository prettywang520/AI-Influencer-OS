import dataclasses
import unittest

from src.creator_intelligence.analyzer_base import AnalyzerContext
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.reply_analyzer import ReplyAnalyzer


def _evidence(tags):
    return Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="note", tags=list(tags))


class ReplyAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.analyzer = ReplyAnalyzer()

    def test_name_is_replies(self):
        self.assertEqual(self.analyzer.name, "replies")

    def test_no_evidence_is_unknown(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)

    def test_reply_tagged_evidence_is_used(self):
        item = _evidence(["reply"])
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [item.evidence_id])

    def test_unrelated_evidence_is_ignored(self):
        context = AnalyzerContext(evidence=[_evidence(["caption"])], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [])

    def test_result_never_carries_a_generated_reply_field(self):
        context = AnalyzerContext(evidence=[_evidence(["reply"])], config=self.config)
        result = self.analyzer.analyze(context)
        field_names = {f.name for f in dataclasses.fields(result.trait_score)}
        self.assertNotIn("draft_reply", field_names)
        self.assertNotIn("generated_reply", field_names)

    def test_analyzer_has_no_generation_method(self):
        self.assertFalse(hasattr(self.analyzer, "generate"))
        self.assertFalse(hasattr(self.analyzer, "generate_reply"))


if __name__ == "__main__":
    unittest.main()
