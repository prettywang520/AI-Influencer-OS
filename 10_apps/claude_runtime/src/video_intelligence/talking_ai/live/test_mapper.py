"""Covers mapper.py (Evidence -> LiveTalkingReelRecord) AND adapter.py
(LiveTalkingReelRecord -> TalkingAIEvidence/SpeechSegment) -- both are
"the mapping layer" (task §26 groups "Mapping" as one test area), and
the task's own TESTS list has no separate test_adapter.py (see the
approved plan's Exploration Summary for the full rationale).
"""
import unittest

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.live.adapter import live_talking_evidence_from_record
from src.video_intelligence.talking_ai.live.mapper import live_talking_reels_from_creator_research
from src.video_intelligence.talking_ai.live.models import LiveTalkingReelRecord
from src.video_intelligence.talking_ai.live.timing_bridge import assign_ordinal_timestamps


def _reel_evidence(**overrides) -> Evidence:
    defaults = dict(
        evidence_type=EvidenceType.OPERATOR_OBSERVATION,
        source_description=(
            "instagram reel | url=https://www.instagram.com/reel/abc/ | published_at=2026-01-01T00:00:00+00:00 "
            "| views=100 | likes=10 | comments=2 | duration_seconds=12.5"
        ),
        content_excerpt="a talking reel caption",
        collected_by="instagram_research_connector",
        tags=["instagram", "reels", "reliability:medium"],
    )
    defaults.update(overrides)
    return Evidence(**defaults)


class MapperTests(unittest.TestCase):
    def test_recovers_one_record_per_distinct_reel_url(self):
        evidence = [_reel_evidence(), _reel_evidence(source_description=_reel_evidence().source_description)]
        records = live_talking_reels_from_creator_research(evidence, creator_label="ref")
        self.assertEqual(len(records), 2)  # mapper.py does not dedupe -- that's sampler.py's job

    def test_recovers_url_published_at_duration(self):
        records = live_talking_reels_from_creator_research([_reel_evidence()], creator_label="ref")
        record = records[0]
        self.assertEqual(record.source_url, "https://www.instagram.com/reel/abc/")
        self.assertEqual(record.published_at, "2026-01-01T00:00:00+00:00")
        self.assertEqual(record.duration_seconds, 12.5)

    def test_caption_kept_in_metadata(self):
        records = live_talking_reels_from_creator_research([_reel_evidence()], creator_label="ref")
        self.assertEqual(records[0].metadata.get("caption"), "a talking reel caption")

    def test_non_reel_evidence_ignored(self):
        caption_evidence = Evidence(
            evidence_type=EvidenceType.TEXT_EXCERPT,
            source_description="instagram caption | post=https://www.instagram.com/reel/abc/",
            content_excerpt="a caption",
            collected_by="instagram_research_connector",
            tags=["instagram", "caption"],
        )
        records = live_talking_reels_from_creator_research([caption_evidence], creator_label="ref")
        self.assertEqual(records, [])

    def test_missing_url_is_skipped_never_guessed(self):
        malformed = Evidence(
            evidence_type=EvidenceType.OPERATOR_OBSERVATION,
            source_description="instagram reel | published_at=2026-01-01T00:00:00+00:00",
            content_excerpt="", collected_by="instagram_research_connector", tags=["instagram", "reels"],
        )
        records = live_talking_reels_from_creator_research([malformed], creator_label="ref")
        self.assertEqual(records, [])

    def test_source_evidence_ids_captured(self):
        item = _reel_evidence()
        records = live_talking_reels_from_creator_research([item], creator_label="ref")
        self.assertEqual(records[0].source_evidence_ids, [item.evidence_id])

    def test_never_populates_timeline_observations(self):
        records = live_talking_reels_from_creator_research([_reel_evidence()], creator_label="ref")
        self.assertEqual(records[0].timeline_observations, [])
        self.assertEqual(records[0].speech_segments, [])

    def test_placeholder_published_at_normalized_to_none(self):
        item = _reel_evidence(
            source_description=(
                "instagram reel | url=https://www.instagram.com/reel/abc/ | published_at=None "
                "| views=None | likes=None | comments=None | duration_seconds=None"
            ),
        )
        records = live_talking_reels_from_creator_research([item], creator_label="ref")
        self.assertIsNone(records[0].published_at)
        self.assertIsNone(records[0].duration_seconds)


class AdapterVideoIdRewritingTests(unittest.TestCase):
    def test_rewrites_video_id_on_evidence_and_segments(self):
        record = LiveTalkingReelRecord(
            source_url="https://www.instagram.com/reel/abc/", creator_label="ref",
            timeline_observations=[TalkingAIEvidence(video_id="placeholder", timestamp_seconds=0.1, blink=True)],
            speech_segments=[SpeechSegment(video_id="placeholder", start_seconds=0.0, end_seconds=1.0)],
        )
        adapted = live_talking_evidence_from_record(record)
        self.assertTrue(all(item.video_id == record.reel_id for item in adapted.evidence))
        self.assertTrue(all(segment.video_id == record.reel_id for segment in adapted.speech_segments))

    def test_already_correct_video_id_left_alone(self):
        record = LiveTalkingReelRecord(source_url="https://www.instagram.com/reel/abc/", creator_label="ref")
        item = TalkingAIEvidence(video_id=record.reel_id, timestamp_seconds=0.1, blink=True)
        record.timeline_observations.append(item)
        adapted = live_talking_evidence_from_record(record)
        self.assertEqual(adapted.evidence[0].evidence_id, item.evidence_id)

    def test_never_invents_a_field(self):
        record = LiveTalkingReelRecord(
            source_url="https://www.instagram.com/reel/abc/",
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.1, blink=True)],
        )
        adapted = live_talking_evidence_from_record(record)
        self.assertIsNone(adapted.evidence[0].gaze_direction)
        self.assertIsNone(adapted.evidence[0].mouth_open_ratio)

    def test_camera_and_subtitle_observations_not_merged_into_evidence(self):
        record = LiveTalkingReelRecord(
            source_url="https://www.instagram.com/reel/abc/",
            camera_observations=["handheld"],
            subtitle_observations=["position:bottom"],
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.1, blink=True)],
        )
        adapted = live_talking_evidence_from_record(record)
        self.assertIsNone(adapted.evidence[0].camera_motion)
        self.assertIsNone(adapted.evidence[0].subtitle_visible)

    def test_ordinal_timing_warning_and_count_surfaced(self):
        items = assign_ordinal_timestamps(
            [TalkingAIEvidence(video_id="", timestamp_seconds=0.0, blink=True) for _ in range(3)]
        )
        record = LiveTalkingReelRecord(source_url="https://www.instagram.com/reel/abc/", timeline_observations=items)
        adapted = live_talking_evidence_from_record(record)
        self.assertEqual(adapted.ordinal_timing_count, 3)
        self.assertTrue(any("ordinal placeholder timing" in warning for warning in adapted.warnings))

    def test_empty_record_warns(self):
        record = LiveTalkingReelRecord(source_url="https://www.instagram.com/reel/abc/")
        adapted = live_talking_evidence_from_record(record)
        self.assertTrue(any("no timeline_observations" in warning for warning in adapted.warnings))

    def test_no_warning_when_all_timing_is_exact(self):
        record = LiveTalkingReelRecord(
            source_url="https://www.instagram.com/reel/abc/",
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.1, blink=True)],
        )
        adapted = live_talking_evidence_from_record(record)
        self.assertEqual(adapted.warnings, [])


if __name__ == "__main__":
    unittest.main()
