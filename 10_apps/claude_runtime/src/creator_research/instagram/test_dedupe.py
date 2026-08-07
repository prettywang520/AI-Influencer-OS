import unittest

from src.creator_research.instagram.dedupe import Deduplicator, identity_for, normalize_url


class NormalizeUrlTests(unittest.TestCase):
    def test_lowercases_scheme_and_host(self):
        self.assertEqual(normalize_url("HTTPS://Example.Invalid/p/1/"), "https://example.invalid/p/1")

    def test_strips_query_string(self):
        self.assertEqual(normalize_url("https://example.invalid/p/1/?utm=abc"), "https://example.invalid/p/1")

    def test_strips_trailing_slash(self):
        self.assertEqual(normalize_url("https://example.invalid/p/1/"), normalize_url("https://example.invalid/p/1"))

    def test_none_returns_none(self):
        self.assertIsNone(normalize_url(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(normalize_url(""))

    def test_strips_fragment(self):
        self.assertEqual(normalize_url("https://example.invalid/p/1#comments"), "https://example.invalid/p/1")


class IdentityForTests(unittest.TestCase):
    def test_prefers_url(self):
        identity = identity_for(url="https://example.invalid/p/1/", stable_id="ignored")
        self.assertTrue(identity.startswith("url:"))

    def test_falls_back_to_stable_id(self):
        identity = identity_for(url=None, stable_id="media123")
        self.assertEqual(identity, "id:media123")

    def test_falls_back_to_content_hash(self):
        identity = identity_for(fallback_payload={"a": 1})
        self.assertTrue(identity.startswith("hash:"))

    def test_deterministic_content_hash(self):
        a = identity_for(fallback_payload={"a": 1, "b": 2})
        b = identity_for(fallback_payload={"b": 2, "a": 1})
        self.assertEqual(a, b)

    def test_no_arguments_raises(self):
        with self.assertRaises(ValueError):
            identity_for()

    def test_captions_alone_never_used_as_identity(self):
        # Structural guarantee: identity_for has no `caption`/`text`
        # parameter at all -- captions can never drive deduplication.
        import inspect

        params = inspect.signature(identity_for).parameters
        self.assertNotIn("caption", params)
        self.assertNotIn("text", params)


class DeduplicatorTests(unittest.TestCase):
    def setUp(self):
        self.dedupe = Deduplicator()

    def test_new_identity_not_duplicate(self):
        self.assertFalse(self.dedupe.is_duplicate("url:https://x/1"))

    def test_marked_identity_is_duplicate(self):
        self.dedupe.mark_seen("url:https://x/1")
        self.assertTrue(self.dedupe.is_duplicate("url:https://x/1"))

    def test_seed_preloads_identities(self):
        self.dedupe.seed(["url:https://x/1", "url:https://x/2"])
        self.assertTrue(self.dedupe.is_duplicate("url:https://x/1"))
        self.assertTrue(self.dedupe.is_duplicate("url:https://x/2"))

    def test_filter_new_removes_duplicates(self):
        items = [("id:1", "a"), ("id:2", "b"), ("id:1", "c")]
        result = self.dedupe.filter_new(items)
        self.assertEqual(result, ["a", "b"])
        self.assertEqual(self.dedupe.duplicates_skipped, 1)

    def test_filter_new_marks_seen(self):
        self.dedupe.filter_new([("id:1", "a")])
        self.assertTrue(self.dedupe.is_duplicate("id:1"))

    def test_seen_identities_sorted(self):
        self.dedupe.mark_seen("id:2")
        self.dedupe.mark_seen("id:1")
        self.assertEqual(self.dedupe.seen_identities(), ["id:1", "id:2"])

    def test_repeated_grid_item_across_scroll_rounds_deduped(self):
        first_round = [("url:https://x/p/1", "item1"), ("url:https://x/p/2", "item2")]
        second_round = [("url:https://x/p/1", "item1"), ("url:https://x/p/2", "item2"), ("url:https://x/p/3", "item3")]
        first_result = self.dedupe.filter_new(first_round)
        second_result = self.dedupe.filter_new(second_round)
        self.assertEqual(len(first_result), 2)
        self.assertEqual(second_result, ["item3"])


if __name__ == "__main__":
    unittest.main()
