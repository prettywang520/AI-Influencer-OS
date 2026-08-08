import unittest

from src.video_intelligence.analyzer import AnalyzerContext
from src.video_intelligence.cta import CTAAnalyzer
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType, VideoPlatform, VideoRecord
from src.video_intelligence.models import VideoIntelligenceConfig


def _config():
    return VideoIntelligenceConfig(
        schema_version="1.0", confidence_min_evidence_for_medium=2, confidence_min_evidence_for_high=4,
        confidence_min_corroboration_for_verified=2, excerpt_max_chars=280,
    )


def _record():
    return VideoRecord(platform=VideoPlatform.INSTAGRAM_REELS, creator_label="ref", reference="demo")


def _evidence(video_id, tags, timestamp=None, source_description="cta note"):
    return VideoEvidence(
        video_id=video_id, evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description=source_description,
        content_excerpt="excerpt", timestamp_seconds=timestamp, tags=tags,
    )


class CTAAnalyzerScoringTests(unittest.TestCase):
    def test_zero_evidence_is_unknown(self):
        record = _record()
        context = AnalyzerContext(video_id=record.video_id, evidence=[], config=_config())
        self.assertEqual(CTAAnalyzer().analyze(context).trait_score.confidence, "unknown")

    def test_scoped_evidence_increases_score(self):
        record = _record()
        evidence = [_evidence(record.video_id, ["cta", "follow"], timestamp=10.0)]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = CTAAnalyzer().analyze(context)
        self.assertGreater(result.trait_score.score, 0.0)

    def test_domain_tag_required_no_leakage_from_unrelated_question_tag(self):
        """Regression test: CTAType.QUESTION == "question" must never
        be treated as CTA-relevant unless the evidence also carries an
        explicit "cta"/"cta_timing" domain tag -- otherwise a hook
        tagged "question" (a legitimate hook_type value) would falsely
        leak into CTA scoring/extraction."""
        record = _record()
        evidence = [_evidence(record.video_id, ["hook", "question"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = CTAAnalyzer().analyze(context)
        self.assertEqual(result.trait_score.confidence, "unknown")
        self.assertEqual(CTAAnalyzer().extract_cta_observations(context), [])


class ExtractCTAObservationsTests(unittest.TestCase):
    def test_one_observation_per_cta_typed_item(self):
        record = _record()
        evidence = [
            _evidence(record.video_id, ["cta", "follow"], timestamp=5.0),
            _evidence(record.video_id, ["cta", "comment"], timestamp=8.0),
        ]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        observations = CTAAnalyzer().extract_cta_observations(context)
        self.assertEqual(len(observations), 2)
        self.assertEqual({obs.cta_type for obs in observations}, {"follow", "comment"})

    def test_timing_seconds_carried_through(self):
        record = _record()
        evidence = [_evidence(record.video_id, ["cta", "save"], timestamp=14.5)]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        observations = CTAAnalyzer().extract_cta_observations(context)
        self.assertEqual(observations[0].timing_seconds, 14.5)

    def test_cta_domain_tag_without_type_tag_is_not_extracted(self):
        record = _record()
        evidence = [_evidence(record.video_id, ["cta"])]  # no specific cta_type tag
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        self.assertEqual(CTAAnalyzer().extract_cta_observations(context), [])

    def test_evidence_from_other_video_ignored(self):
        record = _record()
        other = VideoRecord(platform=VideoPlatform.TIKTOK, creator_label="other", reference="other_ref")
        evidence = [_evidence(other.video_id, ["cta", "share"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        self.assertEqual(CTAAnalyzer().extract_cta_observations(context), [])

    def test_deterministic(self):
        record = _record()
        evidence = [_evidence(record.video_id, ["cta", "follow"], timestamp=3.0)]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        analyzer = CTAAnalyzer()
        self.assertEqual(analyzer.extract_cta_observations(context), analyzer.extract_cta_observations(context))


if __name__ == "__main__":
    unittest.main()
