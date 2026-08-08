import unittest

from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.live.exceptions import LiveAdapterValidationError
from src.video_intelligence.talking_ai.live.models import (
    CAMERA_OBSERVATION_VOCABULARY,
    COMPLETENESS_DOMAINS,
    FramingDistanceTag,
    LiveAdapterConfig,
    LiveBatchLearningReport,
    LiveTalkingLearningRecord,
    LiveTalkingReelRecord,
    compute_live_completeness,
)


def _config(**overrides):
    defaults = dict(
        version="1.0", schema_version="1.0", adapter_version="12D.3", source_platform="instagram_reels",
        minimum_talking_reels=5, recommended_talking_reels=15, strong_sample=30,
    )
    defaults.update(overrides)
    return LiveAdapterConfig(**defaults)


class LiveTalkingReelRecordConstructionTests(unittest.TestCase):
    def test_empty_source_url_raises(self):
        with self.assertRaises(LiveAdapterValidationError):
            LiveTalkingReelRecord(source_url="")

    def test_whitespace_source_url_raises(self):
        with self.assertRaises(LiveAdapterValidationError):
            LiveTalkingReelRecord(source_url="   ")

    def test_defaults(self):
        record = LiveTalkingReelRecord(source_url="https://www.instagram.com/reel/abc/")
        self.assertEqual(record.timeline_observations, [])
        self.assertEqual(record.speech_segments, [])
        self.assertEqual(record.warnings, [])
        self.assertTrue(record.reel_id)
        self.assertTrue(record.record_id)

    def test_invalid_language_observation_raises(self):
        with self.assertRaises(LiveAdapterValidationError):
            LiveTalkingReelRecord(source_url="https://x/", language_observations=["klingon"])

    def test_invalid_camera_observation_raises(self):
        with self.assertRaises(LiveAdapterValidationError):
            LiveTalkingReelRecord(source_url="https://x/", camera_observations=["not_a_real_tag"])

    def test_invalid_artifact_observation_raises(self):
        with self.assertRaises(LiveAdapterValidationError):
            LiveTalkingReelRecord(source_url="https://x/", artifact_observations=["not_a_real_artifact"])

    def test_valid_camera_observation_from_camera_motion_type(self):
        record = LiveTalkingReelRecord(source_url="https://x/", camera_observations=["static", "handheld"])
        self.assertIn("static", record.camera_observations)

    def test_valid_camera_observation_from_framing_distance_tag(self):
        record = LiveTalkingReelRecord(source_url="https://x/", camera_observations=[FramingDistanceTag.CLOSE_UP])
        self.assertIn(FramingDistanceTag.CLOSE_UP, record.camera_observations)


class ReelIdTests(unittest.TestCase):
    def test_same_url_and_creator_label_produce_same_reel_id(self):
        a = LiveTalkingReelRecord(source_url="https://x/reel/1/", creator_label="ref")
        b = LiveTalkingReelRecord(source_url="https://x/reel/1/", creator_label="ref")
        self.assertEqual(a.reel_id, b.reel_id)

    def test_different_creator_label_produces_different_reel_id(self):
        a = LiveTalkingReelRecord(source_url="https://x/reel/1/", creator_label="ref_a")
        b = LiveTalkingReelRecord(source_url="https://x/reel/1/", creator_label="ref_b")
        self.assertNotEqual(a.reel_id, b.reel_id)

    def test_different_url_produces_different_reel_id(self):
        a = LiveTalkingReelRecord(source_url="https://x/reel/1/")
        b = LiveTalkingReelRecord(source_url="https://x/reel/2/")
        self.assertNotEqual(a.reel_id, b.reel_id)


class RecordIdStabilityTests(unittest.TestCase):
    def _build(self, **overrides):
        defaults = dict(
            source_url="https://x/reel/1/", creator_label="ref", duration_seconds=10.0,
            language_observations=["cantonese"],
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.1, blink=True)],
            speech_segments=[SpeechSegment(video_id="", start_seconds=0.0, end_seconds=1.0)],
        )
        defaults.update(overrides)
        return LiveTalkingReelRecord(**defaults)

    def test_stable_across_two_independent_builds(self):
        a = self._build()
        b = self._build()
        self.assertEqual(a.record_id, b.record_id)

    def test_excludes_published_at(self):
        a = self._build(published_at="2026-01-01T00:00:00+00:00")
        b = self._build(published_at="2026-06-15T00:00:00+00:00")
        self.assertEqual(a.record_id, b.record_id)

    def test_excludes_collected_at(self):
        a = self._build()
        b = self._build()
        self.assertNotEqual(a.collected_at, "")
        self.assertEqual(a.record_id, b.record_id)  # collected_at differs (real clock) but record_id still matches

    def test_different_evidence_changes_record_id(self):
        a = self._build()
        b = self._build(timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.5, blink=False)])
        self.assertNotEqual(a.record_id, b.record_id)

    def test_different_warnings_changes_record_id(self):
        a = self._build()
        b = self._build(warnings=["something happened"])
        self.assertNotEqual(a.record_id, b.record_id)


class CompletenessTests(unittest.TestCase):
    def test_all_zero_for_empty_record(self):
        record = LiveTalkingReelRecord(source_url="https://x/")
        completeness = compute_live_completeness(record)
        self.assertEqual(set(completeness.keys()), set(COMPLETENESS_DOMAINS))
        self.assertTrue(all(value == 0.0 for value in completeness.values()))

    def test_speech_domain_from_speech_segments(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/", speech_segments=[SpeechSegment(video_id="", start_seconds=0.0, end_seconds=1.0)],
        )
        self.assertEqual(record.completeness["speech"], 1.0)

    def test_mouth_domain_from_mouth_open_ratio(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/",
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.0, mouth_open_ratio=0.3)],
        )
        self.assertEqual(record.completeness["mouth"], 1.0)
        self.assertEqual(record.completeness["blink"], 0.0)

    def test_subtitle_domain_from_reel_level_observation_list(self):
        record = LiveTalkingReelRecord(source_url="https://x/", subtitle_observations=["position:bottom"])
        self.assertEqual(record.completeness["subtitle"], 1.0)

    def test_camera_domain_from_reel_level_observation_list(self):
        record = LiveTalkingReelRecord(source_url="https://x/", camera_observations=["static"])
        self.assertEqual(record.completeness["camera"], 1.0)

    def test_artifact_domain_from_evidence_artifact_tags(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/",
            timeline_observations=[
                TalkingAIEvidence(video_id="", timestamp_seconds=0.0, artifact_tags=["gesture_loop"])
            ],
        )
        self.assertEqual(record.completeness["artifact"], 1.0)

    def test_timing_domain_from_any_timestamped_evidence(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/",
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.0)],
        )
        self.assertEqual(record.completeness["timing"], 1.0)

    def test_completeness_never_exceeds_one_or_below_zero(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/",
            timeline_observations=[
                TalkingAIEvidence(video_id="", timestamp_seconds=float(i), blink=True) for i in range(10)
            ],
        )
        self.assertTrue(all(0.0 <= value <= 1.0 for value in record.completeness.values()))


class CameraObservationVocabularyTests(unittest.TestCase):
    def test_includes_camera_motion_type_values(self):
        self.assertIn("static", CAMERA_OBSERVATION_VOCABULARY)
        self.assertIn("handheld", CAMERA_OBSERVATION_VOCABULARY)

    def test_includes_framing_distance_tags(self):
        self.assertIn(FramingDistanceTag.CLOSE_UP, CAMERA_OBSERVATION_VOCABULARY)
        self.assertIn(FramingDistanceTag.WALKING, CAMERA_OBSERVATION_VOCABULARY)


class LiveAdapterConfigTests(unittest.TestCase):
    def test_resolved_output_root_is_absolute(self):
        config = _config(output_root="output/video_intelligence/talking_ai/live")
        resolved = config.resolved_output_root()
        self.assertTrue(resolved.is_absolute())
        self.assertTrue(str(resolved).endswith("output/video_intelligence/talking_ai/live"))

    def test_defaults(self):
        config = _config()
        self.assertTrue(config.deterministic)
        self.assertTrue(config.allow_sparse_timeline)
        self.assertFalse(config.live_enabled_by_default)
        self.assertTrue(config.live_require_explicit_authorization)


class LiveTalkingLearningRecordTests(unittest.TestCase):
    def test_overall_confidence_unknown_without_dna(self):
        outcome = LiveTalkingLearningRecord(reel_id="r1", record_id="rec1", source_url="https://x/")
        self.assertEqual(outcome.overall_confidence, "unknown")


class LiveBatchLearningReportTests(unittest.TestCase):
    def test_defaults(self):
        report = LiveBatchLearningReport(total_reels_discovered=0, reels_sampled=0, reels_analyzed=0)
        self.assertEqual(report.production_dna_ids, [])
        self.assertEqual(report.knowledge_patterns_touched, 0)


if __name__ == "__main__":
    unittest.main()
