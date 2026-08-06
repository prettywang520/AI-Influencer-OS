import unittest

from src.creator_intelligence.analyzer_base import AnalyzerContext, analyze_by_tag, evidence_for_tags
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType


def _evidence(tags, collected_by="operator_a"):
    return Evidence(
        evidence_type=EvidenceType.MANUAL_NOTE,
        source_description="note",
        content_excerpt="excerpt",
        collected_by=collected_by,
        tags=list(tags),
    )


class EvidenceForTagsTests(unittest.TestCase):
    def test_filters_by_intersecting_tag(self):
        items = [_evidence(["persona"]), _evidence(["caption"])]
        result = evidence_for_tags(items, ("persona",))
        self.assertEqual(len(result), 1)
        self.assertIn("persona", result[0].tags)

    def test_no_match_returns_empty(self):
        items = [_evidence(["caption"])]
        self.assertEqual(evidence_for_tags(items, ("persona",)), [])

    def test_empty_evidence_list_returns_empty(self):
        self.assertEqual(evidence_for_tags([], ("persona",)), [])

    def test_multi_tag_match(self):
        items = [_evidence(["persona", "caption"])]
        self.assertEqual(len(evidence_for_tags(items, ("caption",))), 1)


class AnalyzeByTagTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()

    def test_no_evidence_yields_unknown_confidence_and_zero_score(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = analyze_by_tag(
            context, trait_name="persona", tags=("persona",), rationale_with_evidence="a", rationale_without_evidence="b"
        )
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(result.trait_score.score, 0.0)
        self.assertEqual(result.trait_score.rationale, "b")

    def test_with_evidence_uses_evidence_rationale(self):
        context = AnalyzerContext(evidence=[_evidence(["persona"])], config=self.config)
        result = analyze_by_tag(
            context, trait_name="persona", tags=("persona",), rationale_with_evidence="a", rationale_without_evidence="b"
        )
        self.assertEqual(result.trait_score.rationale, "a")

    def test_score_increases_with_more_evidence(self):
        context_one = AnalyzerContext(evidence=[_evidence(["persona"])], config=self.config)
        context_many = AnalyzerContext(
            evidence=[_evidence(["persona"], collected_by=f"op_{i}") for i in range(5)], config=self.config
        )
        one = analyze_by_tag(context_one, trait_name="persona", tags=("persona",), rationale_with_evidence="a", rationale_without_evidence="b")
        many = analyze_by_tag(context_many, trait_name="persona", tags=("persona",), rationale_with_evidence="a", rationale_without_evidence="b")
        self.assertLess(one.trait_score.score, many.trait_score.score)

    def test_evidence_ids_recorded_on_trait_score(self):
        item = _evidence(["persona"])
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = analyze_by_tag(
            context, trait_name="persona", tags=("persona",), rationale_with_evidence="a", rationale_without_evidence="b"
        )
        self.assertEqual(result.trait_score.evidence_ids, [item.evidence_id])

    def test_unrelated_evidence_is_excluded(self):
        context = AnalyzerContext(evidence=[_evidence(["caption"])], config=self.config)
        result = analyze_by_tag(
            context, trait_name="persona", tags=("persona",), rationale_with_evidence="a", rationale_without_evidence="b"
        )
        self.assertEqual(result.trait_score.evidence_ids, [])
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)

    def test_no_warnings_by_default(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = analyze_by_tag(
            context, trait_name="persona", tags=("persona",), rationale_with_evidence="a", rationale_without_evidence="b"
        )
        self.assertEqual(result.warnings, [])


if __name__ == "__main__":
    unittest.main()
