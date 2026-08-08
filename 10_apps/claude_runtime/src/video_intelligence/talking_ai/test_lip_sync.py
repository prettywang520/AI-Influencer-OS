import unittest

from src.video_intelligence.talking_ai.analyzer import TalkingAIAnalyzerContext
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.lip_sync import LipSyncAnalyzer
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


class InsufficientEvidenceTests(unittest.TestCase):
    def test_no_segments_or_evidence_is_unknown_confidence(self):
        context = TalkingAIAnalyzerContext(video_id="v", evidence=[], speech_segments=[], config=_config())
        result = LipSyncAnalyzer().analyze(context)
        self.assertEqual(result.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(result.score, 0.0)
        self.assertNotIn("coarse-timing lip sync", result.rationale)


class RationaleTests(unittest.TestCase):
    def test_rationale_always_labeled_coarse_timing_when_evidence_exists(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, mouth_open_ratio=0.3)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = LipSyncAnalyzer().analyze(context)
        self.assertIn("coarse-timing lip sync", result.rationale)
        self.assertNotIn("phoneme-level", result.rationale)


class AlignmentTests(unittest.TestCase):
    def test_mouth_onset_aligned_within_tolerance(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.2, mouth_open_ratio=0.3)]
        context = TalkingAIAnalyzerContext(
            video_id="v", evidence=evidence, speech_segments=segments,
            config=_config(maximum_reasonable_sync_latency_seconds=0.5),
        )
        result = LipSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["mouth_onset_alignment"], 1.0)

    def test_mouth_onset_delayed_beyond_tolerance(self):
        # search_window = max(max_window * 4, 1.0) = 1.0, so a mouth event
        # at 0.7s is within the search window (latency recorded) but
        # beyond the 0.2s alignment tolerance (not counted as an onset hit).
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.7, mouth_open_ratio=0.3)]
        context = TalkingAIAnalyzerContext(
            video_id="v", evidence=evidence, speech_segments=segments,
            config=_config(maximum_reasonable_sync_latency_seconds=0.2),
        )
        result = LipSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["mouth_onset_alignment"], 0.0)
        self.assertIsNotNone(result.metrics["speech_to_mouth_latency"])

    def test_talking_without_any_mouth_motion_evidence_elsewhere(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, mouth_open_ratio=0.3)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = LipSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["talking_without_mouth_motion"], 0.0)

    def test_frozen_mouth_events_detected_when_evidence_present_but_no_motion(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, mouth_open_ratio=0.01),
            TalkingAIEvidence(video_id="v", timestamp_seconds=1.0, mouth_open_ratio=0.02),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = LipSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["frozen_mouth_events"], 1.0)

    def test_mouth_motion_without_speech_counted_when_outside_all_intervals(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=1.0)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, mouth_open_ratio=0.3),
            TalkingAIEvidence(video_id="v", timestamp_seconds=5.0, mouth_open_ratio=0.3),
        ]
        context = TalkingAIAnalyzerContext(
            video_id="v", evidence=evidence, speech_segments=segments,
            config=_config(maximum_reasonable_sync_latency_seconds=0.2),
        )
        result = LipSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["mouth_motion_without_speech"], 1.0)

    def test_pause_closure_alignment(self):
        segments = [SpeechSegment(video_id="v", start_seconds=1.0, end_seconds=2.0, pause_before=0.6)]
        evidence = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.7, mouth_open_ratio=0.02)]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = LipSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["pause_closure_alignment"], 1.0)

    def test_over_and_under_articulation_ratios(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, mouth_open_ratio=0.9),
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, mouth_open_ratio=0.01),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = LipSyncAnalyzer().analyze(context)
        self.assertAlmostEqual(result.metrics["over_articulation"], 0.5)
        self.assertAlmostEqual(result.metrics["under_articulation"], 0.5)

    def test_jaw_motion_alignment_only_counts_evidence_during_speech(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=1.0)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, jaw_motion=0.5),
            TalkingAIEvidence(video_id="v", timestamp_seconds=5.0, jaw_motion=0.0),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = LipSyncAnalyzer().analyze(context)
        self.assertEqual(result.metrics["jaw_motion_alignment"], 1.0)

    def test_syllabic_motion_consistency_uses_lip_motion_variability_once(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=2.0)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.1, lip_motion_intensity=0.3),
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.5, lip_motion_intensity=0.5),
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = LipSyncAnalyzer().analyze(context)
        expected = 1.0 - min(1.0, result.metrics["lip_motion_variation"])
        self.assertAlmostEqual(result.metrics["syllabic_motion_consistency"], expected)


class ConfidenceGrowthTests(unittest.TestCase):
    def test_confidence_improves_with_more_evidence(self):
        segments = [SpeechSegment(video_id="v", start_seconds=float(i), end_seconds=float(i) + 0.5) for i in range(6)]
        evidence = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=float(i) + 0.1, mouth_open_ratio=0.3) for i in range(6)
        ]
        context = TalkingAIAnalyzerContext(video_id="v", evidence=evidence, speech_segments=segments, config=_config())
        result = LipSyncAnalyzer().analyze(context)
        self.assertIn(result.confidence, (ConfidenceLevel.HIGH, ConfidenceLevel.VERIFIED))


if __name__ == "__main__":
    unittest.main()
