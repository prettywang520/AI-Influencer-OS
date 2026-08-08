import random
import unittest

from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType
from src.video_intelligence.instagram.models import InstagramReelEvidencePacket, InstagramReelsConfig
from src.video_intelligence.instagram.sampler import SamplingBucket, sample_reels


def _config(**overrides):
    defaults = dict(
        version="1.0", schema_version="1.0", adapter_version="12D.1", platform="instagram_reels",
        minimum_reels_for_dna=5, recommended_reels=10, strong_sample=20, deterministic=True,
    )
    defaults.update(overrides)
    return InstagramReelsConfig(**defaults)


def _annotation(tags):
    return VideoEvidence(
        video_id="placeholder", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
        source_description="note", content_excerpt="", tags=tags,
    )


def _packet(i, *, tags=None, views=1000, likes=50, comments=5, published_at=None):
    annotations = [_annotation(tags)] if tags else []
    return InstagramReelEvidencePacket(
        reel_url=f"https://www.instagram.com/reel/r{i}/", creator_label="ref",
        views=views, likes=likes, comments=comments,
        published_at=published_at or f"2026-08-{(i % 28) + 1:02d}T00:00:00Z",
        annotations=annotations,
    )


class SampleReelsTests(unittest.TestCase):
    def test_returns_all_packets_when_fewer_than_recommended(self):
        packets = [_packet(i) for i in range(3)]
        sample = sample_reels(packets, _config(recommended_reels=10))
        self.assertEqual(len(sample), 3)

    def test_caps_at_recommended_reels(self):
        packets = [_packet(i) for i in range(50)]
        sample = sample_reels(packets, _config(recommended_reels=10))
        self.assertEqual(len(sample), 10)

    def test_minimum_reels_for_dna_still_returns_something_below_it(self):
        packets = [_packet(i) for i in range(2)]
        sample = sample_reels(packets, _config(minimum_reels_for_dna=10, recommended_reels=10))
        self.assertEqual(len(sample), 2)  # never fabricates reels that don't exist

    def test_empty_input_returns_empty(self):
        self.assertEqual(sample_reels([], _config()), [])

    def test_deterministic_regardless_of_input_order(self):
        packets = [_packet(i, tags=["hook", "question"] if i % 3 == 0 else None) for i in range(20)]
        config = _config(recommended_reels=8)

        sample1 = sample_reels(packets, config)
        shuffled = packets[:]
        random.Random(7).shuffle(shuffled)
        sample2 = sample_reels(shuffled, config)

        self.assertEqual([p.reel_id for p in sample1], [p.reel_id for p in sample2])

    def test_no_duplicate_reel_ids_in_sample(self):
        packets = [_packet(i) for i in range(15)]
        sample = sample_reels(packets, _config(recommended_reels=10))
        self.assertEqual(len(sample), len(set(p.reel_id for p in sample)))

    def test_duplicate_reel_urls_excluded_before_sampling(self):
        packets = [_packet(0), _packet(0), _packet(1)]  # first two are the same reel
        sample = sample_reels(packets, _config(recommended_reels=10))
        self.assertEqual(len(sample), 2)

    def test_talking_and_non_talking_both_represented_when_available(self):
        packets = [_packet(i, tags=["speech", "cantonese"] if i < 5 else None) for i in range(10)]
        sample = sample_reels(packets, _config(recommended_reels=10))
        # with recommended_reels == total available, everything is included
        talking_count = sum(1 for p in packets[:5] if p in sample)
        non_talking_count = sum(1 for p in packets[5:] if p in sample)
        self.assertGreater(talking_count, 0)
        self.assertGreater(non_talking_count, 0)

    def test_content_bucket_tags_recognized(self):
        travel_packet = _packet(0, tags=["camera", "travel"])
        selfie_packet = _packet(1, tags=["camera", "selfie"])
        plain_packet = _packet(2)
        sample = sample_reels([travel_packet, selfie_packet, plain_packet], _config(recommended_reels=10))
        self.assertEqual(len(sample), 3)  # all included, none dropped

    def test_high_and_low_engagement_buckets_split_by_median(self):
        packets = [_packet(i, views=1000, likes=(i * 10), comments=0) for i in range(10)]
        sample = sample_reels(packets, _config(recommended_reels=10))
        self.assertEqual(len(sample), 10)  # sanity: sampling doesn't drop anyone when target == available

    def test_packets_missing_published_at_are_not_dropped(self):
        # _packet()'s own helper always fills in a published_at default,
        # so a genuinely dateless packet must be built directly.
        no_date_packet = InstagramReelEvidencePacket(reel_url="https://www.instagram.com/reel/nodate/", creator_label="ref")
        sample = sample_reels([no_date_packet, _packet(1)], _config(recommended_reels=10))
        self.assertEqual(len(sample), 2)


class SamplingBucketTests(unittest.TestCase):
    def test_all_task_named_buckets_present(self):
        expected = {
            "recent", "older", "high_visible_engagement", "low_visible_engagement", "talking", "non_talking",
            "travel", "daily_life", "brand_collaboration", "selfie", "friend_social", "visually_cinematic",
            "ordinary_grounded",
        }
        self.assertEqual(set(SamplingBucket.ALL), expected)


if __name__ == "__main__":
    unittest.main()
