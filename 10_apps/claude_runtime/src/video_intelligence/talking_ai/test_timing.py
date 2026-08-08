import unittest

from src.video_intelligence.talking_ai.analyzer import (
    TalkingAIAnalyzerContext,
    clamp_score,
    confidence_for,
    corroboration_count,
    evidence_for_video,
    intervals,
    ratio_true,
    score_from_ratio,
    segments_for_video,
    sorted_by_timestamp,
    variability,
)
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.models import ConfidenceLevel, NATURALNESS_DIMENSION_ALIASES, TalkingAIConfig
from src.video_intelligence.talking_ai.timing import (
    mouth_motion_events,
    nearest_event_after,
    nearest_event_near,
    speech_active_intervals,
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


class SpeechActiveIntervalsTests(unittest.TestCase):
    def test_empty_segments(self):
        self.assertEqual(speech_active_intervals([]), [])

    def test_single_segment(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        self.assertEqual(speech_active_intervals(segments), [(0.0, 2.0)])

    def test_overlapping_segments_merge(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0),
            SpeechSegment(video_id="v", start_seconds=1.5, end_seconds=3.0),
        ]
        self.assertEqual(speech_active_intervals(segments), [(0.0, 3.0)])

    def test_adjacent_segments_merge(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0),
            SpeechSegment(video_id="v", start_seconds=2.0, end_seconds=3.0),
        ]
        self.assertEqual(speech_active_intervals(segments), [(0.0, 3.0)])

    def test_disjoint_segments_stay_separate(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0),
            SpeechSegment(video_id="v", start_seconds=5.0, end_seconds=6.0),
        ]
        self.assertEqual(speech_active_intervals(segments), [(0.0, 2.0), (5.0, 6.0)])

    def test_unsorted_input_still_merges_correctly(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=5.0, end_seconds=6.0),
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0),
        ]
        self.assertEqual(speech_active_intervals(segments), [(0.0, 2.0), (5.0, 6.0)])


class MouthMotionEventsTests(unittest.TestCase):
    def test_empty_evidence(self):
        self.assertEqual(mouth_motion_events([]), [])

    def test_open_ratio_above_threshold_is_an_event(self):
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, mouth_open_ratio=0.3)]
        self.assertEqual(mouth_motion_events(evidence), [1.0])

    def test_open_ratio_below_threshold_is_not_an_event(self):
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, mouth_open_ratio=0.05)]
        self.assertEqual(mouth_motion_events(evidence), [])

    def test_lip_motion_intensity_used_when_open_ratio_missing(self):
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, lip_motion_intensity=0.3)]
        self.assertEqual(mouth_motion_events(evidence), [1.0])

    def test_evidence_with_neither_field_never_counted_as_closed(self):
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, blink=True)]
        self.assertEqual(mouth_motion_events(evidence), [])

    def test_results_are_timestamp_sorted(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.0, mouth_open_ratio=0.3),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, mouth_open_ratio=0.3),
        ]
        self.assertEqual(mouth_motion_events(evidence), [1.0, 2.0])


class NearestEventAfterTests(unittest.TestCase):
    def test_no_candidates(self):
        self.assertIsNone(nearest_event_after(1.0, [], max_window=1.0))

    def test_finds_candidate_within_window(self):
        self.assertEqual(nearest_event_after(1.0, [1.3, 5.0], max_window=0.5), 1.3)

    def test_candidate_before_target_is_ignored(self):
        self.assertIsNone(nearest_event_after(1.0, [0.5], max_window=1.0))

    def test_candidate_outside_window_is_ignored(self):
        self.assertIsNone(nearest_event_after(1.0, [3.0], max_window=0.5))

    def test_exact_match_counts(self):
        self.assertEqual(nearest_event_after(1.0, [1.0], max_window=0.1), 1.0)


class NearestEventNearTests(unittest.TestCase):
    def test_finds_closer_candidate_before_or_after(self):
        self.assertEqual(nearest_event_near(2.0, [1.7, 2.6], max_window=1.0), 1.7)

    def test_outside_window_ignored(self):
        self.assertIsNone(nearest_event_near(2.0, [5.0], max_window=1.0))


class IntervalsVariabilityRatioTests(unittest.TestCase):
    def test_intervals_of_sorted_values(self):
        self.assertEqual(intervals([1.0, 1.5, 3.0]), [0.5, 1.5])

    def test_intervals_sorts_unsorted_input(self):
        self.assertEqual(intervals([3.0, 1.0, 1.5]), [0.5, 1.5])

    def test_intervals_of_single_value_is_empty(self):
        self.assertEqual(intervals([1.0]), [])

    def test_variability_none_below_two_samples(self):
        self.assertIsNone(variability([1.0]))
        self.assertIsNone(variability([]))

    def test_variability_of_identical_values_is_zero(self):
        self.assertEqual(variability([2.0, 2.0, 2.0]), 0.0)

    def test_variability_nonzero_for_varying_values(self):
        self.assertGreater(variability([1.0, 1.5, 3.0]), 0.0)

    def test_ratio_true_empty_is_none(self):
        self.assertIsNone(ratio_true([]))

    def test_ratio_true_computes_fraction(self):
        self.assertAlmostEqual(ratio_true([True, True, False]), 2 / 3)

    def test_clamp_score_bounds(self):
        self.assertEqual(clamp_score(-1.0), 0.0)
        self.assertEqual(clamp_score(2.0), 1.0)
        self.assertEqual(clamp_score(0.5), 0.5)

    def test_score_from_ratio_zero_evidenced_is_zero(self):
        self.assertEqual(score_from_ratio(0, 0), 0.0)

    def test_score_from_ratio_computes_clamped_ratio(self):
        self.assertEqual(score_from_ratio(4, 3), 0.75)


class ContextHelpersTests(unittest.TestCase):
    def test_evidence_for_video_filters_by_id(self):
        evidence = [
            TalkingAIEvidence(video_id="a", timestamp_seconds=0.0),
            TalkingAIEvidence(video_id="b", timestamp_seconds=0.0),
        ]
        filtered = evidence_for_video(evidence, "a")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].video_id, "a")

    def test_segments_for_video_filters_by_id(self):
        segments = [
            SpeechSegment(video_id="a", start_seconds=0.0, end_seconds=1.0),
            SpeechSegment(video_id="b", start_seconds=0.0, end_seconds=1.0),
        ]
        filtered = segments_for_video(segments, "b")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].video_id, "b")

    def test_sorted_by_timestamp(self):
        evidence = [
            TalkingAIEvidence(video_id="a", timestamp_seconds=2.0),
            TalkingAIEvidence(video_id="a", timestamp_seconds=1.0),
        ]
        ordered = sorted_by_timestamp(evidence)
        self.assertEqual([item.timestamp_seconds for item in ordered], [1.0, 2.0])

    def test_corroboration_count_counts_distinct_sources(self):
        self.assertEqual(corroboration_count(["operator", "operator", "reviewer"]), 2)

    def test_confidence_for_zero_evidence_is_unknown(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        self.assertEqual(confidence_for(context, evidence_count=0, corroboration_count=0), ConfidenceLevel.UNKNOWN)

    def test_confidence_for_scales_with_evidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        self.assertEqual(
            confidence_for(context, evidence_count=4, corroboration_count=2), ConfidenceLevel.VERIFIED
        )


if __name__ == "__main__":
    unittest.main()
