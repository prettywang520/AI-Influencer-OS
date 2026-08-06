import unittest

from src.creator_intelligence.analyzer_base import AnalyzerContext
from src.creator_intelligence.caption_analyzer import CaptionAnalyzer
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType


def _evidence(tags, excerpt="short excerpt"):
    return Evidence(
        evidence_type=EvidenceType.TEXT_EXCERPT, source_description="note", content_excerpt=excerpt, tags=list(tags)
    )


class CaptionAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.analyzer = CaptionAnalyzer()

    def test_name_is_captions(self):
        self.assertEqual(self.analyzer.name, "captions")

    def test_no_evidence_is_unknown(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)

    def test_caption_tagged_evidence_is_used(self):
        item = _evidence(["caption"])
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [item.evidence_id])

    def test_unrelated_evidence_is_ignored(self):
        context = AnalyzerContext(evidence=[_evidence(["reply"])], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [])

    def test_short_excerpt_produces_no_warning(self):
        context = AnalyzerContext(evidence=[_evidence(["caption"], excerpt="short")], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.warnings, [])

    def test_over_length_excerpt_produces_warning(self):
        too_long = "x" * (self.config.caption_excerpt_max_chars + 1)
        item = _evidence(["caption"], excerpt=too_long)
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn(item.evidence_id, result.warnings[0])

    def test_excerpt_at_exact_cap_produces_no_warning(self):
        exact = "x" * self.config.caption_excerpt_max_chars
        context = AnalyzerContext(evidence=[_evidence(["caption"], excerpt=exact)], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.warnings, [])

    def test_multiple_over_length_excerpts_each_warn(self):
        too_long = "x" * (self.config.caption_excerpt_max_chars + 5)
        items = [_evidence(["caption"], excerpt=too_long) for _ in range(3)]
        context = AnalyzerContext(evidence=items, config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(len(result.warnings), 3)


if __name__ == "__main__":
    unittest.main()
