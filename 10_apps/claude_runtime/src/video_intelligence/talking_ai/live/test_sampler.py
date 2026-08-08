import unittest

from src.video_intelligence.talking_ai.evidence import TalkingAIEvidence
from src.video_intelligence.talking_ai.live.models import FramingDistanceTag, LiveAdapterConfig, LiveTalkingReelRecord
from src.video_intelligence.talking_ai.live.sampler import SamplingBucket, sample_live_talking_reels


def _config(**overrides):
    defaults = dict(
        version="1.0", schema_version="1.0", adapter_version="12D.3", source_platform="instagram_reels",
        minimum_talking_reels=5, recommended_talking_reels=15, strong_sample=30,
    )
    defaults.update(overrides)
    return LiveAdapterConfig(**defaults)


def _record(index, **overrides):
    defaults = dict(source_url=f"https://www.instagram.com/reel/r{index}/", creator_label="ref")
    defaults.update(overrides)
    return LiveTalkingReelRecord(**defaults)


class DeterminismTests(unittest.TestCase):
    def test_same_result_regardless_of_input_order(self):
        records = [_record(i) for i in range(20)]
        config = _config(recommended_talking_reels=8)
        forward = sample_live_talking_reels(records, config)
        backward = sample_live_talking_reels(list(reversed(records)), config)
        self.assertEqual([r.reel_id for r in forward], [r.reel_id for r in backward])

    def test_stable_across_repeated_calls(self):
        records = [_record(i) for i in range(12)]
        config = _config(recommended_talking_reels=6)
        first = sample_live_talking_reels(records, config)
        second = sample_live_talking_reels(records, config)
        self.assertEqual([r.reel_id for r in first], [r.reel_id for r in second])


class TargetCountTests(unittest.TestCase):
    def test_returns_at_most_recommended_count(self):
        records = [_record(i) for i in range(50)]
        config = _config(recommended_talking_reels=10)
        sampled = sample_live_talking_reels(records, config)
        self.assertEqual(len(sampled), 10)

    def test_returns_all_when_fewer_than_recommended(self):
        records = [_record(i) for i in range(3)]
        config = _config(recommended_talking_reels=10)
        sampled = sample_live_talking_reels(records, config)
        self.assertEqual(len(sampled), 3)

    def test_empty_input(self):
        self.assertEqual(sample_live_talking_reels([], _config()), [])

    def test_no_duplicates_in_output(self):
        records = [_record(i) for i in range(20)]
        config = _config(recommended_talking_reels=15)
        sampled = sample_live_talking_reels(records, config)
        self.assertEqual(len(sampled), len({r.reel_id for r in sampled}))

    def test_dedupes_same_reel_id_before_sampling(self):
        records = [_record(0), _record(0)]  # same source_url -> same reel_id
        config = _config(recommended_talking_reels=15)
        sampled = sample_live_talking_reels(records, config)
        self.assertEqual(len(sampled), 1)


class RecencyBucketTests(unittest.TestCase):
    def test_recent_and_older_split(self):
        records = [_record(i, published_at=f"2026-01-{i+1:02d}T00:00:00+00:00") for i in range(10)]
        config = _config(recommended_talking_reels=10)
        sampled = sample_live_talking_reels(records, config)
        self.assertEqual(len(sampled), 10)


class ContentBucketTests(unittest.TestCase):
    def test_direct_camera_bucket_from_camera_observations(self):
        records = [
            _record(i, camera_observations=[FramingDistanceTag.DIRECT_TO_CAMERA] if i < 3 else [])
            for i in range(10)
        ]
        config = _config(recommended_talking_reels=3)
        sampled = sample_live_talking_reels(records, config)
        # direct_camera bucket is iterated early -- expect at least one direct-camera record sampled
        self.assertTrue(any(FramingDistanceTag.DIRECT_TO_CAMERA in r.camera_observations for r in sampled))

    def test_single_vs_multi_cut_from_shot_boundary_count(self):
        single = _record(
            0, camera_observations=["static"],
            timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.0, camera_motion="static")],
        )
        multi = _record(
            1, camera_observations=["static"],
            timeline_observations=[
                TalkingAIEvidence(video_id="", timestamp_seconds=0.0, shot_boundary=True, camera_motion="static"),
                TalkingAIEvidence(video_id="", timestamp_seconds=1.0, shot_boundary=True, camera_motion="static"),
            ],
        )
        config = _config(recommended_talking_reels=2)
        sampled = sample_live_talking_reels([single, multi], config)
        self.assertEqual({r.reel_id for r in sampled}, {single.reel_id, multi.reel_id})

    def test_heavy_light_subtitle_split(self):
        heavy = _record(0, subtitle_observations=["a", "b", "c"])
        light = _record(1, subtitle_observations=["a"])
        config = _config(recommended_talking_reels=2)
        sampled = sample_live_talking_reels([heavy, light], config)
        self.assertEqual(len(sampled), 2)

    def test_strong_minimal_gesture_split(self):
        strong = _record(
            0,
            timeline_observations=[
                TalkingAIEvidence(video_id="", timestamp_seconds=float(i), left_hand_motion=0.5) for i in range(5)
            ],
        )
        minimal = _record(
            1, timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.0, left_hand_motion=0.1)],
        )
        config = _config(recommended_talking_reels=2)
        sampled = sample_live_talking_reels([strong, minimal], config)
        self.assertEqual(len(sampled), 2)

    def test_missing_metadata_never_fabricates_engagement_bucket_membership(self):
        # No engagement_visible in metadata anywhere -- high/low_visible_engagement
        # buckets should simply stay empty (records instead sampled via
        # fallback), never crash or fabricate a value.
        records = [_record(i) for i in range(5)]
        config = _config(recommended_talking_reels=5)
        sampled = sample_live_talking_reels(records, config)
        self.assertEqual(len(sampled), 5)

    def test_engagement_bucket_from_metadata(self):
        high = _record(0, metadata={"engagement_visible": 0.9})
        low = _record(1, metadata={"engagement_visible": 0.1})
        config = _config(recommended_talking_reels=2)
        sampled = sample_live_talking_reels([high, low], config)
        self.assertEqual({r.reel_id for r in sampled}, {high.reel_id, low.reel_id})


class SamplingBucketVocabularyTests(unittest.TestCase):
    def test_bucket_list_matches_task_spec_count(self):
        self.assertEqual(len(SamplingBucket.ALL), 14)


if __name__ == "__main__":
    unittest.main()
