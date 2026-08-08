import unittest

from src.video_intelligence.talking_ai.analyzer import TalkingAIAnalyzerContext
from src.video_intelligence.talking_ai.models import (
    ConfidenceLevel,
    NATURALNESS_DIMENSION_ALIASES,
    TalkingAIConfig,
    TalkingAIMetricResult,
    TalkingNaturalnessResult,
)
from src.video_intelligence.talking_ai.naturalness import NaturalnessAnalyzer


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


def _context():
    return TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())


def _result(trait_name, score, confidence, evidence_ids=None):
    return TalkingAIMetricResult(
        trait_name=trait_name, metrics={}, score=score, confidence=confidence,
        evidence_ids=evidence_ids or [f"{trait_name}-e1"],
    )


class NoEvidenceTests(unittest.TestCase):
    def test_empty_domain_results_is_insufficient_evidence(self):
        result = NaturalnessAnalyzer().analyze(_context(), {})
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.dimension_scores, {})
        self.assertIn("Insufficient evidence", result.rationale)

    def test_all_unknown_confidence_domains_excluded_from_score(self):
        domain_results = {
            trait: _result(trait, 0.9, ConfidenceLevel.UNKNOWN) for trait in NATURALNESS_DIMENSION_ALIASES
        }
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)


class AggregationTests(unittest.TestCase):
    def test_dimension_scores_use_naturalness_dimension_spelling(self):
        domain_results = {"blink": _result("blink", 0.8, ConfidenceLevel.MEDIUM)}
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        self.assertIn("blink_variability", result.dimension_scores)
        self.assertNotIn("blink", result.dimension_scores)

    def test_dimension_confidence_reported_for_missing_domains_as_unknown(self):
        domain_results = {"blink": _result("blink", 0.8, ConfidenceLevel.MEDIUM)}
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        self.assertEqual(result.dimension_confidence["gaze_variability"], ConfidenceLevel.UNKNOWN)

    def test_overall_score_is_mean_of_evidenced_dimensions_only(self):
        domain_results = {
            "blink": _result("blink", 1.0, ConfidenceLevel.HIGH),
            "gaze": _result("gaze", 0.0, ConfidenceLevel.HIGH),
            "head_motion": _result("head_motion", 0.5, ConfidenceLevel.UNKNOWN),
        }
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        self.assertAlmostEqual(result.score, 0.5)  # mean of 1.0 and 0.0 -- head_motion excluded (unknown)

    def test_overall_confidence_is_weakest_link(self):
        domain_results = {
            "blink": _result("blink", 0.8, ConfidenceLevel.VERIFIED),
            "gaze": _result("gaze", 0.8, ConfidenceLevel.LOW),
        }
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        self.assertEqual(result.confidence, ConfidenceLevel.LOW)

    def test_sample_count_is_union_of_evidence_ids(self):
        domain_results = {
            "blink": _result("blink", 0.8, ConfidenceLevel.MEDIUM, evidence_ids=["e1", "e2"]),
            "gaze": _result("gaze", 0.8, ConfidenceLevel.MEDIUM, evidence_ids=["e2", "e3"]),
        }
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        self.assertEqual(result.sample_count, 3)

    def test_warnings_from_domain_results_are_propagated(self):
        flagged = TalkingAIMetricResult(
            trait_name="blink", score=0.5, confidence=ConfidenceLevel.MEDIUM,
            evidence_ids=["e1"], warnings=["sparse blink evidence"],
        )
        result = NaturalnessAnalyzer().analyze(_context(), {"blink": flagged})
        self.assertIn("sparse blink evidence", result.warnings)


class RationaleLanguageTests(unittest.TestCase):
    def test_rationale_uses_only_comparative_vocabulary(self):
        domain_results = {
            "blink": _result("blink", 1.0, ConfidenceLevel.HIGH),
            "gaze": _result("gaze", 0.0, ConfidenceLevel.HIGH),
            "head_motion": _result("head_motion", 0.5, ConfidenceLevel.HIGH),
        }
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        for banned in ("is human", "is ai", "ai-generated", "real person", "fake", "not real", "artificial intelligence generated"):
            self.assertNotIn(banned, result.rationale.lower())

    def test_single_dimension_rationale_notes_insufficient_comparison(self):
        domain_results = {"blink": _result("blink", 0.8, ConfidenceLevel.MEDIUM)}
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        self.assertIn("Insufficient evidence to compare", result.rationale)


class NoHumanAIClassifierStructuralTests(unittest.TestCase):
    def test_result_type_has_no_classification_field(self):
        result = TalkingNaturalnessResult()
        banned_fields = {"is_human", "is_ai", "is_ai_generated", "human_or_ai", "classification", "verdict"}
        self.assertEqual(set(result.__slots__) & banned_fields, set())

    def test_analyzer_never_sets_an_undeclared_attribute(self):
        domain_results = {"blink": _result("blink", 0.8, ConfidenceLevel.MEDIUM)}
        result = NaturalnessAnalyzer().analyze(_context(), domain_results)
        # A slotted dataclass raises AttributeError for any field not
        # declared on the class -- this is a structural (not just
        # convention-based) guarantee that no verdict field exists.
        with self.assertRaises(AttributeError):
            result.is_ai_generated = True


if __name__ == "__main__":
    unittest.main()
