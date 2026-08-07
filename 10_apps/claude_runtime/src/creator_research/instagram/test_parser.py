import unittest

from src.creator_research.instagram import parser


class ParseCountTests(unittest.TestCase):
    def test_plain_number(self):
        self.assertEqual(parser.parse_count("500"), 500)

    def test_comma_separated(self):
        self.assertEqual(parser.parse_count("1,234"), 1234)

    def test_k_suffix(self):
        self.assertEqual(parser.parse_count("12.3K"), 12300)

    def test_m_suffix(self):
        self.assertEqual(parser.parse_count("4.5M"), 4500000)

    def test_b_suffix(self):
        self.assertEqual(parser.parse_count("1.2B"), 1200000000)

    def test_lowercase_suffix(self):
        self.assertEqual(parser.parse_count("2.5k"), 2500)

    def test_trailing_text_ignored(self):
        self.assertEqual(parser.parse_count("1,234 posts"), 1234)

    def test_none_returns_none(self):
        self.assertIsNone(parser.parse_count(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parser.parse_count(""))

    def test_whitespace_only_returns_none(self):
        self.assertIsNone(parser.parse_count("   "))

    def test_garbage_returns_none(self):
        self.assertIsNone(parser.parse_count("not a number"))

    def test_never_raises_on_malformed_input(self):
        for bad in ("K", "-", "...", "১২৩", None, ""):
            try:
                parser.parse_count(bad)
            except Exception as exc:  # pragma: no cover - failure path
                self.fail(f"parse_count raised on {bad!r}: {exc}")


class ParseHashtagsTests(unittest.TestCase):
    def test_extracts_ascii_hashtags(self):
        self.assertEqual(parser.parse_hashtags("love this #travel #vacation"), ["travel", "vacation"])

    def test_extracts_traditional_chinese_hashtags(self):
        self.assertEqual(parser.parse_hashtags("好美 #台灣 #旅行"), ["台灣", "旅行"])

    def test_extracts_japanese_hashtags(self):
        self.assertEqual(parser.parse_hashtags("素敵 #日本語 #旅"), ["日本語", "旅"])

    def test_no_hashtags_returns_empty_list(self):
        self.assertEqual(parser.parse_hashtags("no tags here"), [])

    def test_none_returns_empty_list(self):
        self.assertEqual(parser.parse_hashtags(None), [])

    def test_deduplicates_preserving_first_order(self):
        self.assertEqual(parser.parse_hashtags("#travel again #travel"), ["travel"])


class ParseMentionsTests(unittest.TestCase):
    def test_extracts_mentions(self):
        self.assertEqual(parser.parse_mentions("thanks @friend.one and @another_user"), ["friend.one", "another_user"])

    def test_no_mentions_returns_empty_list(self):
        self.assertEqual(parser.parse_mentions("no mentions"), [])

    def test_none_returns_empty_list(self):
        self.assertEqual(parser.parse_mentions(None), [])

    def test_deduplicates(self):
        self.assertEqual(parser.parse_mentions("@friend hi @friend"), ["friend"])


class ExtractEmojiTests(unittest.TestCase):
    def test_extracts_emoji(self):
        self.assertEqual(parser.extract_emoji("thank you!! 🥰🥰 love you 😍"), ["🥰", "😍"])

    def test_no_emoji_returns_empty_list(self):
        self.assertEqual(parser.extract_emoji("plain text"), [])

    def test_none_returns_empty_list(self):
        self.assertEqual(parser.extract_emoji(None), [])


class ParseCaptionTests(unittest.TestCase):
    def test_preserves_text_exactly(self):
        text = "Exact text!! 🌸 #travel @friend"
        result = parser.parse_caption(text)
        self.assertEqual(result.text, text)

    def test_extracts_hashtags_and_mentions(self):
        result = parser.parse_caption("so grateful #travel with @friend")
        self.assertEqual(result.hashtags, ["travel"])
        self.assertEqual(result.mentions, ["friend"])

    def test_none_becomes_empty_string(self):
        result = parser.parse_caption(None)
        self.assertEqual(result.text, "")

    def test_multiline_text_preserved(self):
        text = "line one\nline two\n\nline four"
        result = parser.parse_caption(text)
        self.assertEqual(result.text, text)

    def test_unicode_caption_preserved_untranslated(self):
        text = "台灣旅行日記 #台灣"
        result = parser.parse_caption(text)
        self.assertEqual(result.text, text)
        self.assertEqual(result.hashtags, ["台灣"])


class ParseProfileTextTests(unittest.TestCase):
    def test_full_profile(self):
        raw = {
            "username": " demo_creator ", "display_name": "Demo Creator", "bio": "travel blogger",
            "post_count": "1,234", "follower_count": "12.3K", "following_count": "500",
            "category": "Creator", "external_link": "https://example.invalid", "verified": "yes",
        }
        parsed = parser.parse_profile_text(raw)
        self.assertEqual(parsed["username"], "demo_creator")
        self.assertEqual(parsed["post_count"], 1234)
        self.assertEqual(parsed["follower_count"], 12300)
        self.assertTrue(parsed["verified"])

    def test_missing_bio_stays_none(self):
        parsed = parser.parse_profile_text({"username": "demo"})
        self.assertIsNone(parsed["bio"])

    def test_missing_counts_stay_none(self):
        parsed = parser.parse_profile_text({"username": "demo"})
        self.assertIsNone(parsed["post_count"])
        self.assertIsNone(parsed["follower_count"])
        self.assertIsNone(parsed["following_count"])

    def test_verified_absent_is_none_not_false(self):
        parsed = parser.parse_profile_text({"username": "demo"})
        self.assertIsNone(parsed["verified"])

    def test_unknown_fields_stay_unknown_never_fabricated(self):
        parsed = parser.parse_profile_text({})
        self.assertEqual(parsed["username"], "")
        self.assertIsNone(parsed["category"])
        self.assertIsNone(parsed["external_link"])


class ParseCommentThreadTests(unittest.TestCase):
    def test_cleans_each_entry(self):
        raw = [{"text": " hi ", "username": " fan1 ", "timestamp": "2h", "parent_id": None}]
        cleaned = parser.parse_comment_thread(raw)
        self.assertEqual(cleaned[0]["text"], "hi")
        self.assertEqual(cleaned[0]["username"], "fan1")

    def test_preserves_order(self):
        raw = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        cleaned = parser.parse_comment_thread(raw)
        self.assertEqual([c["text"] for c in cleaned], ["a", "b", "c"])

    def test_preserves_parent_id_linkage(self):
        raw = [{"text": "reply", "parent_id": "abc123"}]
        cleaned = parser.parse_comment_thread(raw)
        self.assertEqual(cleaned[0]["parent_id"], "abc123")

    def test_missing_timestamp_is_none(self):
        raw = [{"text": "hi"}]
        cleaned = parser.parse_comment_thread(raw)
        self.assertIsNone(cleaned[0]["timestamp"])

    def test_empty_list_returns_empty_list(self):
        self.assertEqual(parser.parse_comment_thread([]), [])


class ParsePostMetadataTests(unittest.TestCase):
    def test_full_post(self):
        raw = {
            "caption": "hi #travel", "like_count": "1,234", "comment_count": "56", "timestamp": "2026-01-01",
            "carousel": True, "location": "Tokyo", "tagged_accounts": ["a", "b"], "collaboration_labels": ["c"],
        }
        parsed = parser.parse_post_metadata(raw)
        self.assertEqual(parsed["like_count"], 1234)
        self.assertEqual(parsed["comment_count"], 56)
        self.assertTrue(parsed["carousel"])
        self.assertEqual(parsed["tagged_accounts"], ["a", "b"])

    def test_missing_fields_stay_none_or_empty(self):
        parsed = parser.parse_post_metadata({})
        self.assertIsNone(parsed["like_count"])
        self.assertFalse(parsed["carousel"])
        self.assertEqual(parsed["tagged_accounts"], [])

    def test_filters_falsy_tagged_accounts(self):
        parsed = parser.parse_post_metadata({"tagged_accounts": ["a", None, "", "b"]})
        self.assertEqual(parsed["tagged_accounts"], ["a", "b"])


class ParseReelMetadataTests(unittest.TestCase):
    def test_full_reel(self):
        raw = {"caption": "reel caption", "views": "12.3K", "likes": "500", "comments": "20", "duration": "0:34"}
        parsed = parser.parse_reel_metadata(raw)
        self.assertEqual(parsed["views"], 12300)
        self.assertEqual(parsed["duration"], 34.0)

    def test_missing_metrics_stay_none(self):
        parsed = parser.parse_reel_metadata({})
        self.assertIsNone(parsed["views"])
        self.assertIsNone(parsed["likes"])
        self.assertIsNone(parsed["duration"])

    def test_duration_with_minutes(self):
        parsed = parser.parse_reel_metadata({"duration": "1:05"})
        self.assertEqual(parsed["duration"], 65.0)

    def test_duration_with_seconds_suffix(self):
        parsed = parser.parse_reel_metadata({"duration": "45s"})
        self.assertEqual(parsed["duration"], 45.0)

    def test_unparseable_duration_is_none(self):
        parsed = parser.parse_reel_metadata({"duration": "not a duration"})
        self.assertIsNone(parsed["duration"])


class ParseHighlightMetadataTests(unittest.TestCase):
    def test_full_highlight(self):
        parsed = parser.parse_highlight_metadata({"title": "Travel", "item_index": 2, "content_type": "photo", "caption": "beach"})
        self.assertEqual(parsed["title"], "Travel")
        self.assertEqual(parsed["item_index"], 2)

    def test_missing_fields_use_safe_defaults(self):
        parsed = parser.parse_highlight_metadata({})
        self.assertEqual(parsed["title"], "")
        self.assertEqual(parsed["item_index"], 0)
        self.assertIsNone(parsed["content_type"])


if __name__ == "__main__":
    unittest.main()
