import unittest

from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.live.exceptions import LiveAdapterValidationError
from src.video_intelligence.talking_ai.live.models import LiveAdapterConfig, LiveTalkingReelRecord
from src.video_intelligence.talking_ai.live.validator import (
    require_valid_record,
    validate_config,
    validate_record,
    validate_records,
)


def _config(**overrides):
    defaults = dict(
        version="1.0", schema_version="1.0", adapter_version="12D.3", source_platform="instagram_reels",
        minimum_talking_reels=5, recommended_talking_reels=15, strong_sample=30,
    )
    defaults.update(overrides)
    return LiveAdapterConfig(**defaults)


class ValidateRecordTests(unittest.TestCase):
    def test_sparse_record_is_valid(self):
        record = LiveTalkingReelRecord(source_url="https://www.instagram.com/reel/abc/")
        self.assertEqual(validate_record(record), [])

    def test_fully_populated_record_is_valid(self):
        record = LiveTalkingReelRecord(
            source_url="https://www.instagram.com/reel/abc/", creator_label="ref",
            published_at="2026-01-01T00:00:00+00:00", duration_seconds=12.0,
            language_observations=["cantonese"],
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.1, blink=True)],
            speech_segments=[SpeechSegment(video_id="", start_seconds=0.0, end_seconds=1.0)],
        )
        self.assertEqual(validate_record(record), [])

    def test_malformed_url_fails(self):
        record = LiveTalkingReelRecord(source_url="not-a-url")
        errors = validate_record(record)
        self.assertTrue(any("absolute URL" in error for error in errors))

    def test_negative_duration_fails(self):
        record = LiveTalkingReelRecord(source_url="https://x/", duration_seconds=-1.0)
        errors = validate_record(record)
        self.assertTrue(any("duration_seconds" in error for error in errors))

    def test_nan_duration_fails(self):
        record = LiveTalkingReelRecord(source_url="https://x/", duration_seconds=float("nan"))
        errors = validate_record(record)
        self.assertTrue(any("finite" in error for error in errors))

    def test_infinite_duration_fails(self):
        record = LiveTalkingReelRecord(source_url="https://x/", duration_seconds=float("inf"))
        errors = validate_record(record)
        self.assertTrue(any("finite" in error for error in errors))

    def test_malformed_published_at_fails(self):
        record = LiveTalkingReelRecord(source_url="https://x/", published_at="not-a-date")
        errors = validate_record(record)
        self.assertTrue(any("ISO timestamp" in error for error in errors))

    def test_negative_evidence_timestamp_fails(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/", timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=-1.0)],
        )
        errors = validate_record(record)
        self.assertTrue(any("timestamp_seconds must not be negative" in error for error in errors))

    def test_nan_evidence_timestamp_fails(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/",
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=float("nan"))],
        )
        errors = validate_record(record)
        self.assertTrue(any("timestamp_seconds must be finite" in error for error in errors))

    def test_mismatched_evidence_video_id_flagged(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/",
            timeline_observations=[TalkingAIEvidence(video_id="totally_different_video", timestamp_seconds=0.0)],
        )
        errors = validate_record(record)
        self.assertTrue(any("does not match record reel_id" in error for error in errors))

    def test_empty_video_id_never_flagged_as_mismatch(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/", timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.0)],
        )
        self.assertEqual(validate_record(record), [])

    def test_negative_segment_timing_fails(self):
        record = LiveTalkingReelRecord(
            source_url="https://x/", speech_segments=[SpeechSegment(video_id="", start_seconds=5.0, end_seconds=5.0)],
        )
        # start/end both 5.0 (non-negative, finite) -- valid; verify a truly
        # negative one fails instead (end must be >= start, enforced by
        # SpeechSegment's own constructor, so we can't construct a negative
        # end < start pair -- test negative start instead, which end mirrors).
        errors = validate_record(record)
        self.assertEqual(errors, [])

        record2 = LiveTalkingReelRecord(
            source_url="https://x/", speech_segments=[SpeechSegment(video_id="", start_seconds=-2.0, end_seconds=-1.0)],
        )
        errors2 = validate_record(record2)
        self.assertTrue(any("start_seconds must not be negative" in error for error in errors2))

    def test_non_json_serializable_metadata_fails(self):
        record = LiveTalkingReelRecord(source_url="https://x/")
        record.metadata["bad"] = object()
        errors = validate_record(record)
        self.assertTrue(any("JSON-serializable" in error for error in errors))

    def test_require_valid_record_raises_on_invalid(self):
        record = LiveTalkingReelRecord(source_url="not-a-url")
        with self.assertRaises(LiveAdapterValidationError):
            require_valid_record(record)

    def test_require_valid_record_passes_silently_on_valid(self):
        record = LiveTalkingReelRecord(source_url="https://x/")
        require_valid_record(record)  # no exception


class ValidateRecordsBatchTests(unittest.TestCase):
    def test_all_valid_records_pass_through(self):
        records = [LiveTalkingReelRecord(source_url=f"https://x/{i}/") for i in range(5)]
        valid, errors = validate_records(records)
        self.assertEqual(len(valid), 5)
        self.assertEqual(errors, {})

    def test_duplicate_reel_id_reported(self):
        records = [
            LiveTalkingReelRecord(source_url="https://x/1/", creator_label="ref"),
            LiveTalkingReelRecord(source_url="https://x/1/", creator_label="ref"),
        ]
        valid, errors = validate_records(records)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(errors), 1)

    def test_invalid_record_excluded_from_valid_list(self):
        records = [LiveTalkingReelRecord(source_url="https://x/1/"), LiveTalkingReelRecord(source_url="not-a-url")]
        valid, errors = validate_records(records)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(errors), 1)

    def test_creator_mismatch_flagged(self):
        record = LiveTalkingReelRecord(source_url="https://x/1/", creator_label="ref_a")
        valid, errors = validate_records([record], expected_creator_label="ref_b")
        self.assertEqual(valid, [])
        self.assertTrue(any("creator mismatch" in msg for msg in errors[record.reel_id]))

    def test_no_creator_mismatch_when_labels_match(self):
        record = LiveTalkingReelRecord(source_url="https://x/1/", creator_label="ref_a")
        valid, errors = validate_records([record], expected_creator_label="ref_a")
        self.assertEqual(len(valid), 1)
        self.assertEqual(errors, {})

    def test_empty_batch(self):
        valid, errors = validate_records([])
        self.assertEqual((valid, errors), ([], {}))


class ValidateConfigTests(unittest.TestCase):
    def test_valid_platform(self):
        self.assertEqual(validate_config(_config(source_platform="instagram_reels")), [])

    def test_invalid_platform_fails(self):
        errors = validate_config(_config(source_platform="not_a_real_platform"))
        self.assertTrue(any("source_platform" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
