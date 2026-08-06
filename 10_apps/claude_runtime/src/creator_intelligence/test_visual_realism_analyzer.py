import unittest

from src.creator_intelligence.analyzer_base import AnalyzerContext
from src.creator_intelligence.config import load_framework_config
from src.creator_intelligence.confidence import ConfidenceLevel
from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.visual_realism_analyzer import CAMERA_MODEL_DISCLAIMER, VisualRealismAnalyzer

# A real-world camera/device model name that must never appear asserted
# as fact anywhere in this analyzer's rationale output.
_FORBIDDEN_CAMERA_MODEL_STRINGS = ("iphone 15 pro", "canon eos r5", "sony a7")


def _evidence(tags):
    return Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="note", tags=list(tags))


class VisualRealismAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_framework_config()
        self.analyzer = VisualRealismAnalyzer()

    def test_name_is_visual_realism(self):
        self.assertEqual(self.analyzer.name, "visual_realism")

    def test_no_evidence_is_unknown(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, ConfidenceLevel.UNKNOWN)

    def test_visual_realism_tagged_evidence_is_used(self):
        item = _evidence(["visual_realism"])
        context = AnalyzerContext(evidence=[item], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [item.evidence_id])

    def test_unrelated_evidence_is_ignored(self):
        context = AnalyzerContext(evidence=[_evidence(["photography"])], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertEqual(result.trait_score.evidence_ids, [])

    def test_rationale_always_includes_camera_model_disclaimer_with_evidence(self):
        context = AnalyzerContext(evidence=[_evidence(["visual_realism"])], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertIn(CAMERA_MODEL_DISCLAIMER, result.trait_score.rationale)

    def test_rationale_always_includes_camera_model_disclaimer_without_evidence(self):
        context = AnalyzerContext(evidence=[], config=self.config)
        result = self.analyzer.analyze(context)
        self.assertIn(CAMERA_MODEL_DISCLAIMER, result.trait_score.rationale)

    def test_rationale_never_names_a_real_camera_model(self):
        context = AnalyzerContext(evidence=[_evidence(["visual_realism"])], config=self.config)
        result = self.analyzer.analyze(context)
        lowered = result.trait_score.rationale.lower()
        for forbidden in _FORBIDDEN_CAMERA_MODEL_STRINGS:
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
