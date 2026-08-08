import unittest

from src.creator_research.instagram.models import ReelRecord
from src.video_intelligence.talking_ai.live.observer_bridge import live_talking_reel_from_reel_record


class ObserverBridgeTests(unittest.TestCase):
    def test_recovers_identity_fields(self):
        reel = ReelRecord(
            reel_url="https://www.instagram.com/reel/abc/",
            caption="a talking reel",
            published_at="2026-01-01T00:00:00+00:00",
            duration_seconds=15.0,
        )
        record = live_talking_reel_from_reel_record(reel, creator_label="ref")
        self.assertEqual(record.source_url, reel.reel_url)
        self.assertEqual(record.published_at, reel.published_at)
        self.assertEqual(record.duration_seconds, reel.duration_seconds)

    def test_recovers_text_overlays_into_subtitle_observations(self):
        reel = ReelRecord(
            reel_url="https://www.instagram.com/reel/abc/",
            text_overlays=["hello everyone", "lets talk about durian"],
        )
        record = live_talking_reel_from_reel_record(reel)
        self.assertEqual(record.subtitle_observations, ["hello everyone", "lets talk about durian"])
        self.assertEqual(record.completeness["subtitle"], 1.0)

    def test_no_text_overlays_leaves_subtitle_observations_empty(self):
        reel = ReelRecord(reel_url="https://www.instagram.com/reel/abc/")
        record = live_talking_reel_from_reel_record(reel)
        self.assertEqual(record.subtitle_observations, [])
        self.assertEqual(record.completeness["subtitle"], 0.0)

    def test_caption_kept_in_metadata_only(self):
        reel = ReelRecord(reel_url="https://www.instagram.com/reel/abc/", caption="a caption")
        record = live_talking_reel_from_reel_record(reel)
        self.assertEqual(record.metadata.get("caption"), "a caption")

    def test_no_caption_leaves_metadata_empty(self):
        reel = ReelRecord(reel_url="https://www.instagram.com/reel/abc/")
        record = live_talking_reel_from_reel_record(reel)
        self.assertEqual(record.metadata, {})

    def test_never_populates_timeline_observations_or_speech_segments(self):
        reel = ReelRecord(reel_url="https://www.instagram.com/reel/abc/", text_overlays=["x"])
        record = live_talking_reel_from_reel_record(reel)
        self.assertEqual(record.timeline_observations, [])
        self.assertEqual(record.speech_segments, [])

    def test_source_evidence_ids_passthrough(self):
        reel = ReelRecord(reel_url="https://www.instagram.com/reel/abc/")
        record = live_talking_reel_from_reel_record(reel, source_evidence_ids=["e1", "e2"])
        self.assertEqual(record.source_evidence_ids, ["e1", "e2"])

    def test_source_evidence_ids_default_empty(self):
        reel = ReelRecord(reel_url="https://www.instagram.com/reel/abc/")
        record = live_talking_reel_from_reel_record(reel)
        self.assertEqual(record.source_evidence_ids, [])

    def test_reel_id_stable_for_same_url_and_creator_label(self):
        reel = ReelRecord(reel_url="https://www.instagram.com/reel/abc/")
        first = live_talking_reel_from_reel_record(reel, creator_label="ref")
        second = live_talking_reel_from_reel_record(reel, creator_label="ref")
        self.assertEqual(first.reel_id, second.reel_id)


if __name__ == "__main__":
    unittest.main()
