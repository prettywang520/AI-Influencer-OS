import unittest

from src.video_intelligence.models import (
    CTAObservation,
    CTAType,
    BeatType,
    ConfidenceLevel,
    ConfidenceThresholds,
    StoryBeat,
    TraitScore,
    VideoIntelligenceConfig,
    compute_confidence_level,
    confidence_rank,
    min_confidence,
)


class ConfidenceRankTests(unittest.TestCase):
    def test_order_is_weakest_to_strongest(self):
        self.assertLess(confidence_rank(ConfidenceLevel.UNKNOWN), confidence_rank(ConfidenceLevel.LOW))
        self.assertLess(confidence_rank(ConfidenceLevel.LOW), confidence_rank(ConfidenceLevel.MEDIUM))
        self.assertLess(confidence_rank(ConfidenceLevel.MEDIUM), confidence_rank(ConfidenceLevel.HIGH))
        self.assertLess(confidence_rank(ConfidenceLevel.HIGH), confidence_rank(ConfidenceLevel.VERIFIED))

    def test_unrecognized_level_raises(self):
        with self.assertRaises(ValueError):
            confidence_rank("not_a_real_level")


class MinConfidenceTests(unittest.TestCase):
    def test_empty_list_is_unknown(self):
        self.assertEqual(min_confidence([]), ConfidenceLevel.UNKNOWN)

    def test_weakest_link_wins(self):
        levels = [ConfidenceLevel.HIGH, ConfidenceLevel.LOW, ConfidenceLevel.VERIFIED]
        self.assertEqual(min_confidence(levels), ConfidenceLevel.LOW)

    def test_all_same_level(self):
        self.assertEqual(min_confidence([ConfidenceLevel.MEDIUM, ConfidenceLevel.MEDIUM]), ConfidenceLevel.MEDIUM)


class ComputeConfidenceLevelTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = ConfidenceThresholds(
            min_evidence_for_medium=2, min_evidence_for_high=4, min_corroboration_for_verified=2
        )

    def test_zero_evidence_is_unknown(self):
        self.assertEqual(compute_confidence_level(0, 0, self.thresholds), ConfidenceLevel.UNKNOWN)

    def test_below_medium_threshold_is_low(self):
        self.assertEqual(compute_confidence_level(1, 1, self.thresholds), ConfidenceLevel.LOW)

    def test_at_medium_threshold(self):
        self.assertEqual(compute_confidence_level(2, 1, self.thresholds), ConfidenceLevel.MEDIUM)

    def test_at_high_threshold_without_corroboration(self):
        self.assertEqual(compute_confidence_level(4, 1, self.thresholds), ConfidenceLevel.HIGH)

    def test_verified_requires_both_high_evidence_and_corroboration(self):
        self.assertEqual(compute_confidence_level(4, 2, self.thresholds), ConfidenceLevel.VERIFIED)

    def test_high_corroboration_alone_is_not_verified(self):
        # corroboration meets the bar but evidence_count does not
        self.assertEqual(compute_confidence_level(1, 2, self.thresholds), ConfidenceLevel.LOW)


class TraitScoreTests(unittest.TestCase):
    def test_defaults(self):
        trait = TraitScore(trait_name="hook", score=0.5)
        self.assertEqual(trait.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(trait.evidence_ids, [])
        self.assertEqual(trait.rationale, "")


class StoryBeatAndCTAObservationTests(unittest.TestCase):
    def test_story_beat_construction(self):
        beat = StoryBeat(beat_type=BeatType.PROBLEM, description="sets up a relatable problem", evidence_ids=["e1"])
        self.assertEqual(beat.beat_type, "problem")
        self.assertIn(beat.beat_type, BeatType.ALL)

    def test_cta_observation_construction(self):
        observation = CTAObservation(cta_type=CTAType.FOLLOW, timing_seconds=10.0, evidence_ids=["e1"])
        self.assertEqual(observation.cta_type, "follow")
        self.assertIn(observation.cta_type, CTAType.ALL)

    def test_cta_observation_timing_optional(self):
        observation = CTAObservation(cta_type=CTAType.SAVE)
        self.assertIsNone(observation.timing_seconds)


class VideoIntelligenceConfigTests(unittest.TestCase):
    def _config(self, **overrides):
        defaults = dict(
            schema_version="1.0", confidence_min_evidence_for_medium=2, confidence_min_evidence_for_high=4,
            confidence_min_corroboration_for_verified=2, excerpt_max_chars=280,
        )
        defaults.update(overrides)
        return VideoIntelligenceConfig(**defaults)

    def test_confidence_thresholds_property(self):
        config = self._config()
        thresholds = config.confidence_thresholds
        self.assertEqual(thresholds.min_evidence_for_medium, 2)
        self.assertEqual(thresholds.min_evidence_for_high, 4)
        self.assertEqual(thresholds.min_corroboration_for_verified, 2)

    def test_weight_for_defaults_to_one(self):
        config = self._config(analyzer_weights={"hook": 2.0})
        self.assertEqual(config.weight_for("hook"), 2.0)
        self.assertEqual(config.weight_for("unmentioned"), 1.0)

    def test_resolved_output_root_is_absolute(self):
        config = self._config(output_root="output/video_intelligence")
        resolved = config.resolved_output_root()
        self.assertTrue(resolved.is_absolute())
        self.assertTrue(str(resolved).endswith("output/video_intelligence"))


if __name__ == "__main__":
    unittest.main()
