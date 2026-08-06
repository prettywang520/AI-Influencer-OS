import unittest

from src.creator_intelligence.scoring import SCORE_MAX, SCORE_MIN, clamp_score, weighted_average


class ClampScoreTests(unittest.TestCase):
    def test_within_range_unchanged(self):
        self.assertEqual(clamp_score(0.5), 0.5)

    def test_below_min_clamps_to_min(self):
        self.assertEqual(clamp_score(-1.0), SCORE_MIN)

    def test_above_max_clamps_to_max(self):
        self.assertEqual(clamp_score(2.0), SCORE_MAX)

    def test_boundary_values_unchanged(self):
        self.assertEqual(clamp_score(SCORE_MIN), SCORE_MIN)
        self.assertEqual(clamp_score(SCORE_MAX), SCORE_MAX)


class WeightedAverageTests(unittest.TestCase):
    def test_empty_values_returns_zero(self):
        self.assertEqual(weighted_average([]), SCORE_MIN)

    def test_equal_weights_default(self):
        self.assertAlmostEqual(weighted_average([0.0, 1.0]), 0.5)

    def test_explicit_weights_applied(self):
        result = weighted_average([1.0, 0.0], weights=[3.0, 1.0])
        self.assertAlmostEqual(result, 0.75)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            weighted_average([1.0, 0.0], weights=[1.0])

    def test_zero_total_weight_returns_min(self):
        self.assertEqual(weighted_average([1.0, 1.0], weights=[0.0, 0.0]), SCORE_MIN)

    def test_result_is_clamped(self):
        # weighted_average of in-range values can't exceed 1.0, but
        # confirm clamping is applied defensively regardless.
        result = weighted_average([1.0], weights=[1.0])
        self.assertLessEqual(result, SCORE_MAX)
        self.assertGreaterEqual(result, SCORE_MIN)


if __name__ == "__main__":
    unittest.main()
