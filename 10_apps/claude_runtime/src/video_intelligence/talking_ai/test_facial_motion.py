import unittest

from src.video_intelligence.talking_ai.analyzer import TalkingAIAnalyzerContext
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.facial_motion import FacialMotionAnalyzer
from src.video_intelligence.talking_ai.framing import FramingAnalyzer
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


class FacialMotionTests(unittest.TestCase):
    def test_no_evidence_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = FacialMotionAnalyzer().analyze(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)

    def test_eyebrow_and_cheek_movement_are_structurally_none(self):
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, facial_expression="neutral")]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = FacialMotionAnalyzer().analyze(context)
        self.assertIsNone(result.metrics["eyebrow_movement"])
        self.assertIsNone(result.metrics["cheek_movement"])
        self.assertIn("not modeled", result.rationale)

    def test_expression_change_count(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, facial_expression="neutral"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, facial_expression="smile"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.0, facial_expression="neutral"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = FacialMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["expression_change_count"], 2.0)

    def test_frozen_face_duration_is_longest_neutral_run(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, facial_expression="neutral"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=5.0, facial_expression="neutral"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=6.0, facial_expression="smile"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = FacialMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["frozen_face_duration"], 5.0)

    def test_smile_onset_and_offset_counts(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, smile_intensity=0.1),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, smile_intensity=0.5),
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.0, smile_intensity=0.1),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = FacialMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["smile_onset_count"], 1.0)
        self.assertEqual(result.metrics["smile_offset_count"], 1.0)

    def test_repeated_expression_cycle_count_counts_local_peaks(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, smile_intensity=0.1),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, smile_intensity=0.5),
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.0, smile_intensity=0.1),
            TalkingAIEvidence(video_id="v", timestamp_seconds=3.0, smile_intensity=0.5),
            TalkingAIEvidence(video_id="v", timestamp_seconds=4.0, smile_intensity=0.1),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = FacialMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["repeated_expression_cycle_count"], 2.0)

    def test_expression_speech_timing_ratio(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, facial_expression="neutral"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, facial_expression="smile"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = FacialMotionAnalyzer().analyze(context)
        self.assertEqual(result.metrics["expression_speech_timing_ratio"], 1.0)


class FramingTests(unittest.TestCase):
    def test_no_camera_motion_evidence_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = FramingAnalyzer().analyze(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)

    def test_tripod_and_handheld_ratios(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, camera_motion="static"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, camera_motion="handheld"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = FramingAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["tripod_ratio"], 0.5)
        self.assertAlmostEqual(result.metrics["handheld_ratio"], 0.5)

    def test_shot_distance_headroom_and_face_center_are_structurally_none(self):
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, camera_motion="static")]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = FramingAnalyzer().analyze(context)
        self.assertIsNone(result.metrics["shot_distance_type"])
        self.assertIsNone(result.metrics["headroom"])
        self.assertIsNone(result.metrics["face_center_stability"])

    def test_shot_change_during_speech_ratio(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, shot_boundary=True, camera_motion="static"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=5.0, shot_boundary=True, camera_motion="static"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = FramingAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["shot_change_during_speech_ratio"], 0.5)

    def test_framing_stability_penalized_by_shot_boundaries(self):
        stable_evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=float(i), camera_motion="static") for i in range(5)
        ]
        stable_context = TalkingAIAnalyzerContext(
            video_id="v", evidence=stable_evidence, speech_segments=[], config=_config()
        )
        stable_result = FramingAnalyzer().analyze(stable_context)

        unstable_evidence = list(stable_evidence) + [
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.5, shot_boundary=True, camera_motion="static"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.5, shot_boundary=True, camera_motion="static"),
        ]
        unstable_context = TalkingAIAnalyzerContext(
            video_id="v", evidence=unstable_evidence, speech_segments=[], config=_config()
        )
        unstable_result = FramingAnalyzer().analyze(unstable_context)

        self.assertGreater(stable_result.metrics["framing_stability"], unstable_result.metrics["framing_stability"])


if __name__ == "__main__":
    unittest.main()
