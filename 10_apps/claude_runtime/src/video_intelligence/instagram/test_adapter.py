import json
import unittest

from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType
from src.video_intelligence.instagram.models import (
    BatchLearningReport,
    InstagramReelEvidencePacket,
    InstagramReelLearningRecord,
    InstagramReelsConfig,
    TRAIT_TAGS,
    compute_completeness,
)
from src.video_intelligence.instagram.report import build_batch_report, render_batch_report_markdown
from src.video_intelligence.production_dna import TRAIT_FIELDS


def _packet(**overrides):
    defaults = dict(reel_url="https://www.instagram.com/reel/abc/", creator_label="ref")
    defaults.update(overrides)
    return InstagramReelEvidencePacket(**defaults)


def _evidence(video_id="v1", tags=None, timestamp=None):
    return VideoEvidence(
        video_id=video_id, evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="note",
        content_excerpt="excerpt", timestamp_seconds=timestamp, tags=tags or [],
    )


class InstagramReelEvidencePacketTests(unittest.TestCase):
    def test_deterministic_reel_id(self):
        a = _packet()
        b = _packet()
        self.assertEqual(a.reel_id, b.reel_id)

    def test_different_url_changes_reel_id(self):
        a = _packet(reel_url="https://www.instagram.com/reel/one/")
        b = _packet(reel_url="https://www.instagram.com/reel/two/")
        self.assertNotEqual(a.reel_id, b.reel_id)

    def test_different_creator_label_changes_reel_id(self):
        a = _packet(creator_label="creator_one")
        b = _packet(creator_label="creator_two")
        self.assertNotEqual(a.reel_id, b.reel_id)

    def test_unknown_fields_default_to_none_or_empty(self):
        packet = InstagramReelEvidencePacket(reel_url="https://x/y")
        self.assertIsNone(packet.published_at)
        self.assertIsNone(packet.duration_seconds)
        self.assertIsNone(packet.views)
        self.assertEqual(packet.annotations, [])
        self.assertEqual(packet.source_evidence_ids, [])

    def test_metadata_round_trips_through_json(self):
        packet = _packet(metadata={"note": "synthetic demo reel", "count": 3})
        payload = json.dumps(packet.metadata)
        self.assertEqual(json.loads(payload), packet.metadata)

    def test_engagement_ratio_none_without_views(self):
        packet = _packet(views=None, likes=10, comments=2)
        self.assertIsNone(packet.visible_engagement_ratio())
        self.assertIsNone(packet.comments_per_1000_views())
        self.assertIsNone(packet.likes_per_1000_views())

    def test_engagement_ratio_none_with_zero_views(self):
        packet = _packet(views=0, likes=10, comments=2)
        self.assertIsNone(packet.visible_engagement_ratio())

    def test_engagement_ratio_computed_when_denominator_valid(self):
        packet = _packet(views=1000, likes=50, comments=10)
        self.assertAlmostEqual(packet.visible_engagement_ratio(), 0.06)
        self.assertAlmostEqual(packet.likes_per_1000_views(), 50.0)
        self.assertAlmostEqual(packet.comments_per_1000_views(), 10.0)

    def test_never_invents_missing_metric(self):
        # A missing likes/comments value must never be treated as 0 --
        # that would silently understate true engagement.
        packet = _packet(views=1000, likes=None, comments=10)
        self.assertIsNone(packet.visible_engagement_ratio())
        self.assertIsNone(packet.likes_per_1000_views())
        self.assertIsNotNone(packet.comments_per_1000_views())


class ComputeCompletenessTests(unittest.TestCase):
    def test_all_trait_fields_present_as_keys(self):
        result = compute_completeness([])
        self.assertEqual(set(result.keys()), set(TRAIT_FIELDS))

    def test_zero_evidence_is_all_zero(self):
        result = compute_completeness([])
        self.assertTrue(all(value == 0.0 for value in result.values()))

    def test_matching_tag_marks_trait_covered(self):
        result = compute_completeness([_evidence(tags=["hook", "question"])])
        self.assertEqual(result["hook"], 1.0)
        self.assertEqual(result["camera"], 0.0)

    def test_lipsync_key_matches_real_trait_name_not_task_spelling(self):
        result = compute_completeness([_evidence(tags=["lipsync", "naturalness"])])
        self.assertEqual(result["lipsync"], 1.0)
        self.assertNotIn("lip_sync", result)


class InstagramReelLearningRecordTests(unittest.TestCase):
    def test_valid_construction(self):
        record = InstagramReelLearningRecord(
            reel_id="r1", source_url="https://x/y", source_evidence_ids=["e1", "e2"],
            completeness={"hook": 1.0}, warnings=["text_overlays unavailable"],
        )
        self.assertEqual(record.reel_id, "r1")
        self.assertEqual(record.patterns_touched, [])
        self.assertIsNone(record.dna)

    def test_metadata_round_trips(self):
        record = InstagramReelLearningRecord(reel_id="r1", source_url="https://x/y", metadata={"bucket": "talking"})
        self.assertEqual(json.loads(json.dumps(record.metadata)), record.metadata)


class BatchLearningReportTests(unittest.TestCase):
    def test_build_report_from_empty_records(self):
        report = build_batch_report([])
        self.assertEqual(report.total_reels, 0)
        self.assertEqual(report.reels_analyzed, 0)
        self.assertEqual(set(report.evidence_completeness.keys()), set(TRAIT_FIELDS))

    def test_build_report_aggregates_completeness(self):
        records = [
            InstagramReelLearningRecord(reel_id="r1", source_url="https://x/1", completeness={"hook": 1.0, "camera": 0.0}),
            InstagramReelLearningRecord(reel_id="r2", source_url="https://x/2", completeness={"hook": 0.0, "camera": 0.0}),
        ]
        report = build_batch_report(records)
        self.assertAlmostEqual(report.evidence_completeness["hook"], 0.5)
        self.assertEqual(report.missing_evidence["hook"], 1)
        self.assertEqual(report.missing_evidence["camera"], 2)

    def test_skipped_reels_counted_and_reported(self):
        report = build_batch_report([], skipped={"bad_id": ["reel_url must not be empty"]})
        self.assertEqual(report.total_reels, 1)
        self.assertEqual(report.reels_skipped, 1)
        self.assertTrue(any("bad_id" in warning for warning in report.warnings))

    def test_video_dna_ids_deduplicated_and_sorted(self):
        from src.video_intelligence.production_dna import VideoDNA

        dna_a = VideoDNA(video_id="v1", subject_label="a")
        dna_b = VideoDNA(video_id="v2", subject_label="b")
        records = [
            InstagramReelLearningRecord(reel_id="r1", source_url="https://x/1", dna=dna_a),
            InstagramReelLearningRecord(reel_id="r2", source_url="https://x/2", dna=dna_b),
        ]
        report = build_batch_report(records)
        self.assertEqual(report.video_dna_ids, sorted({dna_a.dna_id, dna_b.dna_id}))

    def test_render_markdown_never_includes_verbatim_caption(self):
        records = [InstagramReelLearningRecord(reel_id="r1", source_url="https://x/1", completeness={"hook": 1.0})]
        report = build_batch_report(records)
        markdown = render_batch_report_markdown(report)
        self.assertIn("Instagram Reels Learning Report", markdown)
        self.assertNotIn("caption", markdown.lower())  # report is stats-only, never verbatim content


if __name__ == "__main__":
    unittest.main()
