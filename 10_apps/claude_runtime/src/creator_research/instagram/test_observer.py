import unittest

from src.creator_research.instagram.config import load_instagram_connector_config
from src.creator_research.instagram.dedupe import Deduplicator
from src.creator_research.instagram.exceptions import (
    AccessLimitedError,
    AuthenticationRequiredError,
    ChallengeDetectedError,
    RateLimitedError,
)
from src.creator_research.instagram.models import (
    CaptionObservation,
    DiscoveredItem,
    PostRecord,
    RelationshipEvidenceBasis,
    VisualSamplingBucket,
)
from src.creator_research.instagram.navigator import FakeBrowserAdapter, FakeElement
from src.creator_research.instagram.selectors import default_instagram_selectors
from src.creator_research.instagram import observer

PROFILE_URL = "https://example.invalid/demo_creator"


class FakeJob:
    def __init__(self, username="demo_creator", profile_url=PROFILE_URL):
        self.username = username
        self.profile_url = profile_url


def _setup(**config_overrides):
    adapter = FakeBrowserAdapter()
    selectors = default_instagram_selectors()
    config = load_instagram_connector_config()
    for key, value in config_overrides.items():
        setattr(config, key, value)
    page = adapter.add_page(PROFILE_URL)
    return adapter, selectors, config, page


class ObserveProfileTests(unittest.TestCase):
    def test_parses_visible_fields(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.profile.username[0], [FakeElement(text="demo_creator")])
        page.register(selectors.profile.bio[0], [FakeElement(text="travel diary")])
        page.register(selectors.profile.follower_count[0], [FakeElement(text="12.3K")])
        profile = observer.observe_profile(adapter, FakeJob(), selectors, config)
        self.assertEqual(profile.bio, "travel diary")
        self.assertEqual(profile.follower_count, 12300)

    def test_missing_bio_stays_none(self):
        adapter, selectors, config, page = _setup()
        profile = observer.observe_profile(adapter, FakeJob(), selectors, config)
        self.assertIsNone(profile.bio)

    def test_missing_counts_stay_none(self):
        adapter, selectors, config, page = _setup()
        profile = observer.observe_profile(adapter, FakeJob(), selectors, config)
        self.assertIsNone(profile.follower_count)
        self.assertIsNone(profile.post_count)

    def test_verified_badge_present(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.profile.verified_badge[0], [FakeElement()])
        profile = observer.observe_profile(adapter, FakeJob(), selectors, config)
        self.assertTrue(profile.verified)

    def test_verified_badge_absent_is_none(self):
        adapter, selectors, config, page = _setup()
        profile = observer.observe_profile(adapter, FakeJob(), selectors, config)
        self.assertIsNone(profile.verified)

    def test_unknown_fields_remain_unknown(self):
        adapter, selectors, config, page = _setup()
        profile = observer.observe_profile(adapter, FakeJob(), selectors, config)
        self.assertIsNone(profile.category)
        self.assertIsNone(profile.external_link)

    def test_login_wall_raises_authentication_required(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.access_state.login_wall[0], [FakeElement()])
        with self.assertRaises(AuthenticationRequiredError):
            observer.observe_profile(adapter, FakeJob(), selectors, config)

    def test_private_account_raises_access_limited(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.access_state.private_account_marker[0], [FakeElement()])
        with self.assertRaises(AccessLimitedError):
            observer.observe_profile(adapter, FakeJob(), selectors, config)

    def test_rate_limit_marker_raises_rate_limited(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.access_state.rate_limit_marker[0], [FakeElement()])
        with self.assertRaises(RateLimitedError):
            observer.observe_profile(adapter, FakeJob(), selectors, config)

    def test_challenge_marker_raises_challenge_detected(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.access_state.challenge_marker[0], [FakeElement()])
        with self.assertRaises(ChallengeDetectedError):
            observer.observe_profile(adapter, FakeJob(), selectors, config)


class ObserveGridTests(unittest.TestCase):
    def test_discovers_posts_and_reels(self):
        adapter, selectors, config, page = _setup()
        items = [
            FakeElement(attributes={"href": f"/p/{i}/", "content_type": "reel" if i % 5 == 0 else "post"})
            for i in range(20)
        ]
        page.register(selectors.grid.grid_item[0], items)
        discovered, warnings = observer.observe_grid(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertEqual(len(discovered), 20)
        self.assertEqual(warnings, [])

    def test_dedup_across_scroll_rounds(self):
        adapter, selectors, config, page = _setup()
        items = [FakeElement(attributes={"href": f"/p/{i}/"}) for i in range(15)]
        page.register(selectors.grid.grid_item[0], items, reveal_batch=5)
        dedupe = Deduplicator()
        discovered, _ = observer.observe_grid(adapter, FakeJob(), selectors, config, dedupe)
        self.assertEqual(len(discovered), 15)
        self.assertEqual(len({d.source_url for d in discovered}), 15)

    def test_stops_at_max_items(self):
        adapter, selectors, config, page = _setup(max_posts_per_job=5, max_reels_per_job=0)
        items = [FakeElement(attributes={"href": f"/p/{i}/"}) for i in range(50)]
        page.register(selectors.grid.grid_item[0], items)
        discovered, _ = observer.observe_grid(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertEqual(len(discovered), 5)

    def test_stops_after_no_new_items(self):
        adapter, selectors, config, page = _setup(max_scroll_rounds_per_section=50)
        items = [FakeElement(attributes={"href": f"/p/{i}/"}) for i in range(5)]
        page.register(selectors.grid.grid_item[0], items)  # all visible immediately, no reveal_batch
        discovered, _ = observer.observe_grid(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertEqual(len(discovered), 5)
        # scroll_count should be small (stagnant-round stop), not 50
        self.assertLess(adapter.scroll_count, 50)

    def test_empty_grid_returns_empty_list(self):
        adapter, selectors, config, page = _setup()
        discovered, _ = observer.observe_grid(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertEqual(discovered, [])

    def test_access_limit_raises_before_any_collection(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.access_state.private_account_marker[0], [FakeElement()])
        with self.assertRaises(AccessLimitedError):
            observer.observe_grid(adapter, FakeJob(), selectors, config, Deduplicator())

    def test_content_type_defaults_to_post(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.grid.grid_item[0], [FakeElement(attributes={"href": "/p/1/"})])
        discovered, _ = observer.observe_grid(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertEqual(discovered[0].content_type, "post")


class ObservePostTests(unittest.TestCase):
    def test_full_post_metadata(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(selectors.post.caption[0], [FakeElement(text="hi #travel")])
        post_page.register(selectors.post.like_count[0], [FakeElement(text="1,234")])
        post_page.register(selectors.post.carousel_indicator[0], [FakeElement()])
        record = observer.observe_post(adapter, post_url, selectors, config)
        self.assertEqual(record.like_count, 1234)
        self.assertTrue(record.is_carousel)
        self.assertEqual(record.caption_reference, "hi #travel")

    def test_missing_metrics_stay_none(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/2/"
        adapter.add_page(post_url)
        record = observer.observe_post(adapter, post_url, selectors, config)
        self.assertIsNone(record.like_count)
        self.assertIsNone(record.comment_count)

    def test_location_label_captured(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/3/"
        post_page = adapter.add_page(post_url)
        post_page.register(selectors.post.location_label[0], [FakeElement(text="Tokyo")])
        record = observer.observe_post(adapter, post_url, selectors, config)
        self.assertEqual(record.location_label, "Tokyo")

    def test_collaboration_metadata_captured(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/4/"
        post_page = adapter.add_page(post_url)
        post_page.register(selectors.post.collaboration_label[0], [FakeElement(text="collab_partner")])
        record = observer.observe_post(adapter, post_url, selectors, config)
        self.assertEqual(record.collaboration_labels, ["collab_partner"])

    def test_access_limited_raises(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/5/"
        post_page = adapter.add_page(post_url)
        post_page.register(selectors.access_state.login_wall[0], [FakeElement()])
        with self.assertRaises(AuthenticationRequiredError):
            observer.observe_post(adapter, post_url, selectors, config)


class ObserveCaptionsTests(unittest.TestCase):
    def test_collects_captions_from_items(self):
        adapter, selectors, config, page = _setup()
        items = []
        for i in range(3):
            url = f"https://example.invalid/p/{i}/"
            post_page = adapter.add_page(url)
            post_page.register(selectors.post.caption[0], [FakeElement(text=f"caption {i} 台灣 #travel")])
            items.append(DiscoveredItem(source_url=url, content_type="post"))
        captions, warnings = observer.observe_captions(adapter, items, selectors, config)
        self.assertEqual(len(captions), 3)
        self.assertEqual(warnings, [])
        self.assertIn("travel", captions[0].hashtags)

    def test_unicode_captions_preserved(self):
        adapter, selectors, config, page = _setup()
        url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(url)
        text = "台灣旅行日記 素敵 🥰"
        post_page.register(selectors.post.caption[0], [FakeElement(text=text)])
        captions, _ = observer.observe_captions(adapter, [DiscoveredItem(source_url=url)], selectors, config)
        self.assertEqual(captions[0].caption_text, text)


class ObserveCommentsTests(unittest.TestCase):
    def test_visible_comments_collected(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.comment_item[0],
            [FakeElement(text="nice pic!", attributes={"username": "fan1", "timestamp": "2h"})],
        )
        comments, warnings = observer.observe_comments(adapter, post_url, selectors, config)
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0].comment_text, "nice pic!")

    def test_nested_replies_via_parent_id(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.comment_item[0],
            [
                FakeElement(text="original", attributes={"username": "fan1"}),
                FakeElement(text="reply to fan1", attributes={"username": "fan2", "parent_id": "c1"}),
            ],
        )
        comments, _ = observer.observe_comments(adapter, post_url, selectors, config)
        self.assertEqual(comments[1].thread_parent_id, "c1")

    def test_missing_timestamp_stays_none(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(selectors.comments.comment_item[0], [FakeElement(text="hi")])
        comments, _ = observer.observe_comments(adapter, post_url, selectors, config)
        self.assertIsNone(comments[0].timestamp_text)

    def test_respects_max_comments_per_post(self):
        adapter, selectors, config, page = _setup(max_comments_per_post=2)
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(selectors.comments.comment_item[0], [FakeElement(text=f"c{i}") for i in range(10)])
        comments, _ = observer.observe_comments(adapter, post_url, selectors, config)
        self.assertEqual(len(comments), 2)

    def test_comments_never_self_classify_as_creator(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.comment_item[0], [FakeElement(text="hi", attributes={"username": "demo_creator"})]
        )
        comments, _ = observer.observe_comments(adapter, post_url, selectors, config)
        self.assertFalse(comments[0].is_creator)


class ObserveCreatorRepliesTests(unittest.TestCase):
    def test_target_username_reply_recognized(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.reply_item[0],
            [FakeElement(text="thank you!!", attributes={"username": "demo_creator", "parent_comment_text": "love it"})],
        )
        replies, _ = observer.observe_creator_replies(adapter, post_url, selectors, config, username="demo_creator")
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0].creator_reply_text, "thank you!!")

    def test_other_account_reply_rejected(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.reply_item[0], [FakeElement(text="me too", attributes={"username": "random_fan"})]
        )
        replies, _ = observer.observe_creator_replies(adapter, post_url, selectors, config, username="demo_creator")
        self.assertEqual(replies, [])

    def test_case_insensitive_username_match(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.reply_item[0], [FakeElement(text="thanks", attributes={"username": "Demo_Creator"})]
        )
        replies, _ = observer.observe_creator_replies(adapter, post_url, selectors, config, username="demo_creator")
        self.assertEqual(len(replies), 1)

    def test_never_classifies_by_absent_username(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(selectors.comments.reply_item[0], [FakeElement(text="thanks")])
        replies, _ = observer.observe_creator_replies(adapter, post_url, selectors, config, username="demo_creator")
        self.assertEqual(replies, [])

    def test_parent_comment_linkage_preserved(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.reply_item[0],
            [FakeElement(text="thanks", attributes={"username": "demo_creator", "parent_comment_text": "love this trip"})],
        )
        replies, _ = observer.observe_creator_replies(adapter, post_url, selectors, config, username="demo_creator")
        self.assertEqual(replies[0].parent_comment_text, "love this trip")

    def test_emoji_preserved(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.reply_item[0],
            [FakeElement(text="thank you!! 🥰", attributes={"username": "demo_creator"})],
        )
        replies, _ = observer.observe_creator_replies(adapter, post_url, selectors, config, username="demo_creator")
        self.assertEqual(replies[0].emoji, ["🥰"])

    def test_thread_depth_captured(self):
        adapter, selectors, config, page = _setup()
        post_url = "https://example.invalid/p/1/"
        post_page = adapter.add_page(post_url)
        post_page.register(
            selectors.comments.reply_item[0],
            [FakeElement(text="thanks", attributes={"username": "demo_creator", "thread_depth": "2"})],
        )
        replies, _ = observer.observe_creator_replies(adapter, post_url, selectors, config, username="demo_creator")
        self.assertEqual(replies[0].thread_depth, 2)


class ObserveReelsTests(unittest.TestCase):
    def test_discovers_reels_with_metrics(self):
        adapter, selectors, config, page = _setup()
        reel_elements = [
            FakeElement(
                text="reel caption",
                attributes={"href": "/reel/1/", "views": "12.3K", "likes": "500", "comments": "20"},
            )
        ]
        page.register(selectors.reels.reel_item[0], reel_elements)
        reels, warnings = observer.observe_reels(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertEqual(len(reels), 1)
        self.assertEqual(reels[0].views, 12300)

    def test_missing_metrics_stay_none(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.reels.reel_item[0], [FakeElement(attributes={"href": "/reel/1/"})])
        reels, _ = observer.observe_reels(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertIsNone(reels[0].views)

    def test_screenshot_reference_present(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.reels.reel_item[0], [FakeElement(attributes={"href": "/reel/1/"})])
        reels, _ = observer.observe_reels(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertIsNotNone(reels[0].thumbnail_reference)

    def test_no_media_download_attempted(self):
        import inspect

        source = inspect.getsource(observer.observe_reels)
        for forbidden in ("requests.get", "urllib.request", ".download(", "yt_dlp"):
            self.assertNotIn(forbidden, source)

    def test_respects_max_reels_per_job(self):
        adapter, selectors, config, page = _setup(max_reels_per_job=2)
        page.register(
            selectors.reels.reel_item[0], [FakeElement(attributes={"href": f"/reel/{i}/"}) for i in range(10)]
        )
        reels, _ = observer.observe_reels(adapter, FakeJob(), selectors, config, Deduplicator())
        self.assertEqual(len(reels), 2)


class ObserveHighlightsTests(unittest.TestCase):
    def test_title_and_item_sequence(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.highlights.highlight_avatar[0], [FakeElement(attributes={"title": "Travel"})])
        page.register(
            selectors.highlights.highlight_item[0],
            [FakeElement(text="item0", attributes={"content_type": "photo"}), FakeElement(text="item1")],
        )
        highlights, warnings = observer.observe_highlights(adapter, FakeJob(), selectors, config)
        self.assertEqual(highlights[0].highlight_title, "Travel")
        self.assertEqual([h.item_index for h in highlights], [0, 1])

    def test_inaccessible_highlights_returns_warning(self):
        adapter, selectors, config, page = _setup()
        highlights, warnings = observer.observe_highlights(adapter, FakeJob(), selectors, config)
        self.assertEqual(highlights, [])
        self.assertEqual(len(warnings), 1)

    def test_respects_max_highlight_items(self):
        adapter, selectors, config, page = _setup(max_highlight_items=2)
        page.register(selectors.highlights.highlight_avatar[0], [FakeElement(attributes={"title": "Travel"})])
        page.register(selectors.highlights.highlight_item[0], [FakeElement(text=f"i{i}") for i in range(10)])
        highlights, _ = observer.observe_highlights(adapter, FakeJob(), selectors, config)
        self.assertEqual(len(highlights), 2)

    def test_access_limited_returns_warning_not_raise(self):
        adapter, selectors, config, page = _setup()
        page.register(selectors.access_state.private_account_marker[0], [FakeElement()])
        highlights, warnings = observer.observe_highlights(adapter, FakeJob(), selectors, config)
        self.assertEqual(highlights, [])
        self.assertTrue(any("access_limited" in w for w in warnings))


class ObserveRelationshipsTests(unittest.TestCase):
    def test_tagged_accounts_produce_evidence(self):
        posts = [PostRecord(post_url="https://x/p/1", tagged_accounts=["friend1"])]
        records = observer.observe_relationships(posts, [])
        self.assertEqual(records[0].basis, RelationshipEvidenceBasis.TAGGED_ACCOUNT)
        self.assertEqual(records[0].related_account, "friend1")

    def test_collaboration_labels_produce_evidence(self):
        posts = [PostRecord(post_url="https://x/p/1", collaboration_labels=["collab_partner"])]
        records = observer.observe_relationships(posts, [])
        self.assertEqual(records[0].basis, RelationshipEvidenceBasis.COLLABORATION_LABEL)

    def test_caption_mentions_produce_evidence(self):
        captions = [CaptionObservation(caption_text="x", post_reference="https://x/p/1", mentions=["friend2"])]
        records = observer.observe_relationships([], captions)
        self.assertEqual(records[0].basis, RelationshipEvidenceBasis.CAPTION_DECLARED)

    def test_no_visual_relationship_inference(self):
        # Structural guarantee: observe_relationships only reads
        # tagged_accounts/collaboration_labels/mentions -- it has no
        # parameter for an image or visual-analysis input at all.
        import inspect

        params = inspect.signature(observer.observe_relationships).parameters
        self.assertEqual(set(params), {"posts", "captions"})

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(observer.observe_relationships([], []), [])


class ObserveVisualExamplesTests(unittest.TestCase):
    def test_deterministic_bucket_assignment(self):
        items = [DiscoveredItem(source_url=f"https://x/p/{i}") for i in range(5)]
        first = observer.observe_visual_examples(items)
        second = observer.observe_visual_examples(items)
        self.assertEqual([r.sampling_bucket for r in first], [r.sampling_bucket for r in second])

    def test_buckets_are_recognized_values(self):
        items = [DiscoveredItem(source_url=f"https://x/p/{i}") for i in range(3)]
        records = observer.observe_visual_examples(items)
        for record in records:
            self.assertIn(record.sampling_bucket, VisualSamplingBucket.ALL)

    def test_empty_items_returns_empty_list(self):
        self.assertEqual(observer.observe_visual_examples([]), [])

    def test_makes_no_visual_realism_judgement(self):
        import inspect

        source = inspect.getsource(observer.observe_visual_examples)
        for forbidden in ("face_recognition", "cv2", "PIL", "torch"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
