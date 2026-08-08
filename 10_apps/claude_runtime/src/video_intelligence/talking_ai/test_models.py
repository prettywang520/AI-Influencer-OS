import unittest

from src.video_intelligence.talking_ai.models import (
    NATURALNESS_DIMENSION_ALIASES,
    TRAIT_FIELDS,
    ConfidenceLevel,
    RecommendedRange,
    TalkingAIConfig,
    TalkingAIMetricResult,
    TalkingNaturalnessResult,
)


def _config(**overrides):
    defaults = dict(
        version="1.0", schema_version="1.0", engine_version="1.0.0",
        maximum_reasonable_sync_latency_seconds=0.5, pause_min_seconds=0.3,
        minimum_talking_videos=1, recommended_talking_videos=5, strong_sample=15,
        naturalness_dimensions=tuple(NATURALNESS_DIMENSION_ALIASES.values()),
        minimum_pattern_corroboration=2,
    )
    defaults.update(overrides)
    return TalkingAIConfig(**defaults)


class TraitFieldsAndAliasesTests(unittest.TestCase):
    def test_trait_fields_has_ten_entries(self):
        self.assertEqual(len(TRAIT_FIELDS), 10)

    def test_alias_map_covers_every_trait_field(self):
        self.assertEqual(set(NATURALNESS_DIMENSION_ALIASES.keys()), set(TRAIT_FIELDS))

    def test_alias_values_are_unique(self):
        values = list(NATURALNESS_DIMENSION_ALIASES.values())
        self.assertEqual(len(values), len(set(values)))

    def test_six_aliases_differ_from_dna_field_spelling(self):
        differing = [name for name, alias in NATURALNESS_DIMENSION_ALIASES.items() if name != alias]
        self.assertEqual(
            set(differing),
            {"blink", "gaze", "head_motion", "facial_motion", "gesture_sync", "framing"},
        )


class TalkingAIConfigTests(unittest.TestCase):
    def test_confidence_thresholds_property(self):
        config = _config(
            confidence_min_evidence_for_medium=2, confidence_min_evidence_for_high=4,
            confidence_min_corroboration_for_verified=2,
        )
        thresholds = config.confidence_thresholds
        self.assertEqual(thresholds.min_evidence_for_medium, 2)
        self.assertEqual(thresholds.min_evidence_for_high, 4)
        self.assertEqual(thresholds.min_corroboration_for_verified, 2)

    def test_resolved_output_root_is_absolute_and_ends_with_output_root(self):
        config = _config(output_root="output/video_intelligence/talking_ai")
        resolved = config.resolved_output_root()
        self.assertTrue(resolved.is_absolute())
        self.assertTrue(str(resolved).endswith("output/video_intelligence/talking_ai"))

    def test_defaults_for_confidence_fields(self):
        config = _config()
        self.assertEqual(config.confidence_min_evidence_for_medium, 2)
        self.assertEqual(config.confidence_min_evidence_for_high, 4)
        self.assertEqual(config.confidence_min_corroboration_for_verified, 2)

    def test_default_deidentify_and_verbatim_flags(self):
        config = _config()
        self.assertTrue(config.deidentify_source)
        self.assertFalse(config.preserve_verbatim_transcripts)


class TalkingAIMetricResultTests(unittest.TestCase):
    def test_defaults(self):
        result = TalkingAIMetricResult(trait_name="blink")
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(result.metrics, {})
        self.assertEqual(result.evidence_ids, [])
        self.assertEqual(result.score, 0.0)


class TalkingNaturalnessResultTests(unittest.TestCase):
    def test_defaults(self):
        result = TalkingNaturalnessResult()
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(result.dimension_scores, {})

    def test_has_no_human_ai_classification_field(self):
        result = TalkingNaturalnessResult()
        field_names = {f for f in result.__slots__}
        banned = {"is_human", "is_ai", "is_ai_generated", "human_or_ai", "classification", "verdict"}
        self.assertEqual(field_names & banned, set())


class RecommendedRangeTests(unittest.TestCase):
    def test_construction(self):
        range_ = RecommendedRange(metric_name="blink.blink_frequency", low=0.1, high=0.3, sample_count=3)
        self.assertEqual(range_.confidence, ConfidenceLevel.UNKNOWN)
        self.assertLess(range_.low, range_.high)


if __name__ == "__main__":
    unittest.main()
