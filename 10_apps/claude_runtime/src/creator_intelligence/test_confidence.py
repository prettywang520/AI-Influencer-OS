import unittest

from src.creator_intelligence.confidence import (
    ConfidenceLevel,
    ConfidenceThresholds,
    compute_confidence,
    confidence_rank,
    min_confidence,
)


def _thresholds(medium=2, high=4, verified_corroboration=2):
    return ConfidenceThresholds(
        min_evidence_for_medium=medium,
        min_evidence_for_high=high,
        min_corroboration_for_verified=verified_corroboration,
    )


class ConfidenceLevelTests(unittest.TestCase):
    def test_order_is_weakest_to_strongest(self):
        self.assertEqual(
            ConfidenceLevel.ORDER,
            (
                ConfidenceLevel.UNKNOWN,
                ConfidenceLevel.LOW,
                ConfidenceLevel.MEDIUM,
                ConfidenceLevel.HIGH,
                ConfidenceLevel.VERIFIED,
            ),
        )

    def test_confidence_rank_orders_correctly(self):
        self.assertLess(confidence_rank(ConfidenceLevel.UNKNOWN), confidence_rank(ConfidenceLevel.LOW))
        self.assertLess(confidence_rank(ConfidenceLevel.LOW), confidence_rank(ConfidenceLevel.MEDIUM))
        self.assertLess(confidence_rank(ConfidenceLevel.MEDIUM), confidence_rank(ConfidenceLevel.HIGH))
        self.assertLess(confidence_rank(ConfidenceLevel.HIGH), confidence_rank(ConfidenceLevel.VERIFIED))

    def test_confidence_rank_rejects_unknown_string(self):
        with self.assertRaises(ValueError):
            confidence_rank("not_a_real_level")


class MinConfidenceTests(unittest.TestCase):
    def test_empty_list_is_unknown(self):
        self.assertEqual(min_confidence([]), ConfidenceLevel.UNKNOWN)

    def test_returns_weakest_of_several(self):
        levels = [ConfidenceLevel.HIGH, ConfidenceLevel.LOW, ConfidenceLevel.VERIFIED]
        self.assertEqual(min_confidence(levels), ConfidenceLevel.LOW)

    def test_single_level_returns_itself(self):
        self.assertEqual(min_confidence([ConfidenceLevel.HIGH]), ConfidenceLevel.HIGH)

    def test_all_unknown_stays_unknown(self):
        levels = [ConfidenceLevel.UNKNOWN, ConfidenceLevel.UNKNOWN]
        self.assertEqual(min_confidence(levels), ConfidenceLevel.UNKNOWN)


class ComputeConfidenceTests(unittest.TestCase):
    def test_zero_evidence_is_always_unknown(self):
        self.assertEqual(compute_confidence(0, 0, _thresholds()), ConfidenceLevel.UNKNOWN)

    def test_below_medium_threshold_is_low(self):
        self.assertEqual(compute_confidence(1, 0, _thresholds()), ConfidenceLevel.LOW)

    def test_at_medium_threshold_is_medium(self):
        self.assertEqual(compute_confidence(2, 0, _thresholds()), ConfidenceLevel.MEDIUM)

    def test_at_high_threshold_without_corroboration_is_high(self):
        self.assertEqual(compute_confidence(4, 0, _thresholds()), ConfidenceLevel.HIGH)

    def test_high_threshold_with_corroboration_is_verified(self):
        self.assertEqual(compute_confidence(4, 2, _thresholds()), ConfidenceLevel.VERIFIED)

    def test_corroboration_alone_without_evidence_threshold_is_not_verified(self):
        # 3 evidence items but corroboration_count meets the verified bar --
        # still capped at HIGH because evidence_count < min_evidence_for_high.
        self.assertEqual(compute_confidence(3, 2, _thresholds()), ConfidenceLevel.MEDIUM)

    def test_negative_evidence_count_is_unknown(self):
        self.assertEqual(compute_confidence(-1, 0, _thresholds()), ConfidenceLevel.UNKNOWN)

    def test_never_fabricates_confidence_without_evidence(self):
        # Even with a huge corroboration_count, zero real evidence stays UNKNOWN.
        self.assertEqual(compute_confidence(0, 99, _thresholds()), ConfidenceLevel.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
