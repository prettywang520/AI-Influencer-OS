import unittest

from src.video_intelligence.talking_ai.analyzer import TalkingAIAnalyzerContext
from src.video_intelligence.talking_ai.blink import BlinkAnalyzer
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.gaze import GazeAnalyzer
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


class BlinkTests(unittest.TestCase):
    def test_no_blink_evidence_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = BlinkAnalyzer().analyze(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(result.metrics["blink_count"], 0.0)

    def test_blink_count_and_frequency(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, blink=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.0, blink=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, blink=False),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = BlinkAnalyzer().analyze(context)
        self.assertEqual(result.metrics["blink_count"], 2.0)
        self.assertAlmostEqual(result.metrics["blink_frequency"], 1.0)  # 2 blinks over a 2s span

    def test_double_blink_frequency(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, blink=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, blink=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=10.0, blink=True),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = BlinkAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["double_blink_frequency"], 0.5)

    def test_long_no_blink_interval_count(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, blink=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=20.0, blink=True),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = BlinkAnalyzer().analyze(context)
        self.assertEqual(result.metrics["long_no_blink_interval_count"], 1.0)

    def test_blink_timing_relative_to_pause(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.6)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.7, blink=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=5.0, blink=True),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = BlinkAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["blink_timing_relative_to_pause_ratio"], 0.5)

    def test_blink_timing_relative_to_sentence_boundary(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.1, blink=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=8.0, blink=True),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = BlinkAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["blink_timing_relative_to_sentence_boundary_ratio"], 0.5)

    def test_rationale_never_diagnoses_a_medical_condition(self):
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, blink=True)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = BlinkAnalyzer().analyze(context)
        for banned in ("disorder", "condition", "diagnosis", "medical"):
            self.assertNotIn(banned, result.rationale.lower())


class GazeTests(unittest.TestCase):
    def test_no_gaze_evidence_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = GazeAnalyzer().analyze(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)

    def test_direct_camera_gaze_ratio_from_gaze_direction(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, gaze_direction="direct_camera"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, gaze_direction="away_left"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GazeAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["direct_camera_gaze_ratio"], 0.5)

    def test_direct_camera_gaze_ratio_falls_back_to_eye_contact(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, eye_contact=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, eye_contact=False),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GazeAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["direct_camera_gaze_ratio"], 0.5)

    def test_return_to_camera_ratio(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, gaze_direction="direct_camera"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, gaze_direction="away_left"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=2.0, gaze_direction="direct_camera"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GazeAnalyzer().analyze(context)
        self.assertEqual(result.metrics["return_to_camera_ratio"], 1.0)

    def test_fixed_stare_duration_is_longest_direct_run(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, gaze_direction="direct_camera"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=5.0, gaze_direction="direct_camera"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GazeAnalyzer().analyze(context)
        self.assertEqual(result.metrics["fixed_stare_duration"], 5.0)

    def test_gaze_away_frequency(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, gaze_direction="direct_camera"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, gaze_direction="away_left"),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GazeAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["gaze_away_frequency"], 0.5)

    def test_rationale_never_infers_emotional_state(self):
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, gaze_direction="direct_camera")]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = GazeAnalyzer().analyze(context)
        for banned in ("nervous", "happy", "sad", "anxious", "confident feeling"):
            self.assertNotIn(banned, result.rationale.lower())


if __name__ == "__main__":
    unittest.main()
