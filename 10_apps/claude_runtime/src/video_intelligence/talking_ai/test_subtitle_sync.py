import unittest

from src.video_intelligence.talking_ai.analyzer import TalkingAIAnalyzerContext
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.models import ConfidenceLevel, NATURALNESS_DIMENSION_ALIASES, TalkingAIConfig
from src.video_intelligence.talking_ai.subtitle_sync import SubtitleSyncAnalyzer


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


class SubtitleSyncTests(unittest.TestCase):
    def test_no_evidence_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = SubtitleSyncAnalyzer().analyze(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)

    def test_highlight_alignment_is_structurally_none(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, subtitle_change=True, subtitle_visible=True)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = SubtitleSyncAnalyzer().analyze(context)
        self.assertIsNone(result.metrics["highlight_alignment"])

    def test_aligned_subtitle_change_at_speech_onset(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, subtitle_change=True)]
        context = TalkingAIAnalyzerContext(
            video_id="v", evidence=evidence, speech_segments=segments,
            config=_config(maximum_reasonable_sync_latency_seconds=0.5),
        )
        result = SubtitleSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["subtitle_segment_alignment"], 1.0)
        self.assertAlmostEqual(result.metrics["speech_subtitle_latency"], 0.1)

    def test_delayed_subtitle_change_beyond_tolerance(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.9, subtitle_change=True)]
        context = TalkingAIAnalyzerContext(
            video_id="v", evidence=evidence, speech_segments=segments,
            config=_config(maximum_reasonable_sync_latency_seconds=0.2),
        )
        result = SubtitleSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["subtitle_segment_alignment"], 0.0)

    def test_no_subtitle_change_at_all_scores_alignment_zero_not_fabricated(self):
        # No subtitle_change event anywhere -> the speech interval is
        # "checked" but has nothing to align to, so alignment is 0.0
        # (a real miss), while speech_subtitle_latency stays None
        # (there is nothing to average a latency from).
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, subtitle_visible=True)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = SubtitleSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["subtitle_segment_alignment"], 0.0)
        self.assertIsNone(result.metrics["speech_subtitle_latency"])

    def test_subtitle_visible_ratio(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, subtitle_visible=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, subtitle_visible=False),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = SubtitleSyncAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["subtitle_visible_ratio"], 0.5)

    def test_subtitle_density(self):
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, subtitle_change=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=4.0, subtitle_change=True),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=[], config=_config())
        result = SubtitleSyncAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["subtitle_density"], 0.5)

    def test_no_verbatim_transcript_in_metrics_or_rationale(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0, text_optional="a secret transcript")]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, subtitle_change=True)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = SubtitleSyncAnalyzer().analyze(context)
        self.assertNotIn("a secret transcript", result.rationale)
        for value in result.metrics.values():
            if isinstance(value, str):
                self.assertNotIn("a secret transcript", value)


if __name__ == "__main__":
    unittest.main()
