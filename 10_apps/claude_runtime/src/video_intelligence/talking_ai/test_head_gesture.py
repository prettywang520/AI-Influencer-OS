import unittest

from src.video_intelligence.talking_ai.analyzer import TalkingAIAnalyzerContext
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.gesture_sync import GestureSyncAnalyzer
from src.video_intelligence.talking_ai.head_motion import HeadMotionAnalyzer
from src.video_intelligence.talking_ai.models import ConfidenceLevel, NATURALNESS_DIMENSION_ALIASES, TalkingAIConfig


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


class HeadMotionTests(unittest.TestCase):
    def test_no_evidence_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = HeadMotionAnalyzer().analyze(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)

    def test_yaw_pitch_roll_range_and_variability(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, head_yaw=-10.0, head_pitch=2.0, head_roll=0.0),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, head_yaw=10.0, head_pitch=-2.0, head_roll=1.0),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = HeadMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["yaw_range"], 20.0)
        self.assertEqual(result.metrics["pitch_range"], 4.0)
        self.assertEqual(result.metrics["roll_range"], 1.0)
        self.assertIsNotNone(result.metrics["yaw_variability"])

    def test_micro_motion_vs_large_gesture_ratio(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, head_motion_intensity=0.2),  # micro
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, head_motion_intensity=0.8),  # large
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = HeadMotionAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["micro_motion_ratio"], 0.5)
        self.assertAlmostEqual(result.metrics["large_gesture_ratio"], 0.5)

    def test_head_freeze_duration_longest_run(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, head_motion_intensity=0.01),
            TalkingAIEvidence(video_id="v", timestamp_seconds=5.0, head_motion_intensity=0.01),
            TalkingAIEvidence(video_id="v", timestamp_seconds=6.0, head_motion_intensity=0.9),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = HeadMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["head_freeze_duration"], 5.0)

    def test_motion_during_emphasis_and_pause_ratios(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=1.0, emphasis=True, pause_before=0.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, head_motion_intensity=0.5)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = HeadMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["motion_during_emphasis_ratio"], 1.0)

    def test_motion_repetition_cycle_count_counts_local_peaks(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, head_motion_intensity=0.1),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, head_motion_intensity=0.5),
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.0, head_motion_intensity=0.1),
            TalkingAIEvidence(video_id="v", timestamp_seconds=3.0, head_motion_intensity=0.5),
            TalkingAIEvidence(video_id="v", timestamp_seconds=4.0, head_motion_intensity=0.1),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = HeadMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["motion_repetition_cycle_count"], 2.0)


class GestureSyncTests(unittest.TestCase):
    def test_no_evidence_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)

    def test_gesture_peak_alignment_within_emphasis_window(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0, emphasis=True)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, left_hand_motion=0.5)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["gesture_peak_alignment"], 1.0)

    def test_gesture_onset_before_emphasis(self):
        segments = [SpeechSegment(video_id="v", start_seconds=2.0, end_seconds=4.0, emphasis=True)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=1.8, right_hand_motion=0.5)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["gesture_onset_before_emphasis"], 1.0)

    def test_gesture_end_alignment(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0, emphasis=True)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=2.3, left_hand_motion=0.5)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["gesture_end_alignment"], 1.0)

    def test_gesture_variety_counts_distinct_types(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, gesture_type="wave"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, gesture_type="point"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.0, gesture_type="wave"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["gesture_variety"], 2.0)

    def test_overactivity_and_underactivity_ratios(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, left_hand_motion=0.9),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, right_hand_motion=0.01),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["gesture_overactivity"], 0.5)
        self.assertAlmostEqual(result.metrics["gesture_underactivity"], 0.5)

    def test_gesture_pause_behavior(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.6)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.7, left_hand_motion=0.02)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["gesture_pause_behavior"], 1.0)

    def test_speech_gesture_coherence_averages_alignment_ratios(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0, emphasis=True)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, left_hand_motion=0.5)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        self.assertGreater(result.metrics["speech_gesture_coherence"], 0.0)

    def test_generalized_description_never_includes_specific_timestamps(self):
        # Sanity check that the module docstring's discipline holds:
        # nothing in this analyzer's output metrics dict is a free-text
        # per-video choreography description.
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, left_hand_motion=0.5)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GestureSyncAnalyzer().analyze(context)
        for value in result.metrics.values():
            self.assertNotIsInstance(value, str)


if __name__ == "__main__":
    unittest.main()
