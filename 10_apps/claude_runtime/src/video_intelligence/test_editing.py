import unittest

from src.video_intelligence.analyzer import AnalyzerContext
from src.video_intelligence.editing import EditingAnalyzer
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType, VideoPlatform, VideoRecord
from src.video_intelligence.models import VideoIntelligenceConfig


def _config():
    return VideoIntelligenceConfig(
        schema_version="1.0", confidence_min_evidence_for_medium=2, confidence_min_evidence_for_high=4,
        confidence_min_corroboration_for_verified=2, excerpt_max_chars=280,
    )


def _record():
    return VideoRecord(platform=VideoPlatform.INSTAGRAM_REELS, creator_label="ref", reference="demo")


def _evidence(video_id, tags, collected_by="operator"):
    return VideoEvidence(
        video_id=video_id, evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="editing note",
        content_excerpt="excerpt", tags=tags, collected_by=collected_by,
    )


class EditingAnalyzerTests(unittest.TestCase):
    def test_zero_evidence_is_unknown(self):
        record = _record()
        context = AnalyzerContext(video_id=record.video_id, evidence=[], config=_config())
        result = EditingAnalyzer().analyze(context)
        self.assertEqual(result.trait_score.confidence, "unknown")
        self.assertEqual(result.trait_score.score, 0.0)

    def test_scoped_evidence_increases_score(self):
        record = _record()
        evidence = [_evidence(record.video_id, ["editing", "fast_cut"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = EditingAnalyzer().analyze(context)
        self.assertGreater(result.trait_score.score, 0.0)
        self.assertEqual(result.trait_score.evidence_ids, [evidence[0].evidence_id])

    def test_corroboration_from_distinct_sources_raises_confidence(self):
        record = _record()
        evidence = [
            _evidence(record.video_id, ["editing", "jump_cut"], collected_by="operator_a"),
            _evidence(record.video_id, ["editing", "transition"], collected_by="operator_b"),
        ]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = EditingAnalyzer().analyze(context)
        self.assertEqual(result.trait_score.confidence, "medium")

    def test_evidence_from_other_video_ignored(self):
        record = _record()
        other = VideoRecord(platform=VideoPlatform.TIKTOK, creator_label="other", reference="other_ref")
        evidence = [_evidence(other.video_id, ["editing", "zoom"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = EditingAnalyzer().analyze(context)
        self.assertEqual(result.trait_score.confidence, "unknown")

    def test_deterministic(self):
        record = _record()
        evidence = [_evidence(record.video_id, ["editing", "motion"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        analyzer = EditingAnalyzer()
        self.assertEqual(analyzer.analyze(context).trait_score, analyzer.analyze(context).trait_score)


if __name__ == "__main__":
    unittest.main()
