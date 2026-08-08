import unittest

from src.video_intelligence.talking_ai.analyzer import TalkingAIAnalyzerContext
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.models import ConfidenceLevel, NATURALNESS_DIMENSION_ALIASES, TalkingAIConfig
from src.video_intelligence.talking_ai.speech_sync import SpeechSyncAnalyzer


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


class SpeechRhythmTests(unittest.TestCase):
    def test_no_segments_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = SpeechSyncAnalyzer().analyze_speech_rhythm(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(result.score, 0.0)
        self.assertIsNone(result.metrics["average_speech_rate"])

    def test_average_speech_rate_and_variability(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=1.0, speech_rate_optional=3.0),
            SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, speech_rate_optional=5.0),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_speech_rhythm(context)
        self.assertEqual(result.metrics["average_speech_rate"], 4.0)
        self.assertIsNotNone(result.metrics["speech_rate_variability"])

    def test_question_and_emphasis_ratios(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=1.0, question_like=True, emphasis=False),
            SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, question_like=False, emphasis=True),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_speech_rhythm(context)
        self.assertAlmostEqual(result.metrics["question_ratio"], 0.5)
        self.assertAlmostEqual(result.metrics["emphasis_ratio"], 0.5)

    def test_language_mix_count_and_code_switching(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=1.0, language="cantonese", code_switching=True),
            SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, language="english", code_switching=False),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_speech_rhythm(context)
        self.assertEqual(result.metrics["language_mix_count"], 2.0)
        self.assertAlmostEqual(result.metrics["code_switching_ratio"], 0.5)

    def test_other_videos_segments_are_ignored(self):
        segments = [SpeechSegment(video_id="other", start_seconds=0.0, end_seconds=1.0, speech_rate_optional=3.0)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_speech_rhythm(context)
        self.assertEqual(result.sample_count, 0)


class PauseBehaviorTests(unittest.TestCase):
    def test_no_pauses_is_unknown_confidence(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_pause_behavior(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)

    def test_short_gap_below_pause_min_seconds_is_not_a_pause(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.1)]
        context = TalkingAIAnalyzerContext(
            video_id="v", evidence=[], speech_segments=segments, config=_config(pause_min_seconds=0.3),
        )
        result = SpeechSyncAnalyzer().analyze_pause_behavior(context)
        self.assertEqual(result.metrics["breath_like_pause_count"], 0.0)

    def test_rationale_uses_breath_like_language_never_breathing_detected(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.6)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_pause_behavior(context)
        self.assertIn("breath-like pause behavior", result.rationale.lower())
        self.assertNotIn("breathing detected", result.rationale.lower())

    def test_mouth_closure_during_pause_ratio(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.6)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, mouth_open_ratio=0.05),
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.8, mouth_open_ratio=0.30),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_pause_behavior(context)
        self.assertAlmostEqual(result.metrics["mouth_closure_during_pause_ratio"], 0.5)

    def test_blink_and_head_motion_near_pause_ratios(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.6)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, blink=True, head_motion_intensity=0.05),
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.8, blink=False, head_motion_intensity=0.20),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_pause_behavior(context)
        self.assertAlmostEqual(result.metrics["blink_near_pause_ratio"], 0.5)
        self.assertAlmostEqual(result.metrics["head_motion_near_pause_ratio"], 0.5)

    def test_evidence_with_no_field_reported_does_not_count_toward_ratio_denominator(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.6)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, gaze_direction="direct_camera")]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_pause_behavior(context)
        self.assertIsNone(result.metrics["mouth_closure_during_pause_ratio"])
        self.assertIsNone(result.metrics["blink_near_pause_ratio"])

    def test_pause_duration_metrics(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.4),
            SpeechSegment(video_id="v", start_seconds=5.0, end_seconds=6.0, pause_before=0.8),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=segments, config=_config())
        result = SpeechSyncAnalyzer().analyze_pause_behavior(context)
        self.assertAlmostEqual(result.metrics["average_pause_duration"], 0.6)
        self.assertIsNotNone(result.metrics["pause_duration_variability"])
        self.assertEqual(result.metrics["breath_like_pause_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
