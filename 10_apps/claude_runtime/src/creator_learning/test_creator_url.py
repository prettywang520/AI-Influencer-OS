import unittest

from src.creator_intelligence.intake import creator_id as compute_creator_id
from src.creator_learning.creator_url import parse_creator_url
from src.creator_learning.exceptions import InvalidCreatorUrlError


class ParseCreatorUrlTests(unittest.TestCase):
    def test_parses_instagram_profile_url(self):
        parsed = parse_creator_url("https://www.instagram.com/example_creator/")
        self.assertEqual(parsed.platform, "instagram")
        self.assertEqual(parsed.username, "example_creator")
        self.assertEqual(parsed.profile_url, "https://www.instagram.com/example_creator/")

    def test_parses_bare_host_without_www(self):
        parsed = parse_creator_url("https://instagram.com/example_creator")
        self.assertEqual(parsed.platform, "instagram")
        self.assertEqual(parsed.username, "example_creator")

    def test_creator_id_matches_shared_helper(self):
        parsed = parse_creator_url("https://www.instagram.com/example_creator/")
        self.assertEqual(parsed.creator_id, compute_creator_id("instagram", "example_creator"))

    def test_creator_id_is_deterministic(self):
        first = parse_creator_url("https://www.instagram.com/example_creator/")
        second = parse_creator_url("https://www.instagram.com/example_creator/")
        self.assertEqual(first.creator_id, second.creator_id)

    def test_different_usernames_produce_different_creator_ids(self):
        a = parse_creator_url("https://www.instagram.com/creator_a/")
        b = parse_creator_url("https://www.instagram.com/creator_b/")
        self.assertNotEqual(a.creator_id, b.creator_id)

    def test_extra_path_segments_are_ignored_beyond_username(self):
        parsed = parse_creator_url("https://www.instagram.com/example_creator/reels/")
        self.assertEqual(parsed.username, "example_creator")

    def test_empty_string_raises(self):
        with self.assertRaises(InvalidCreatorUrlError):
            parse_creator_url("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(InvalidCreatorUrlError):
            parse_creator_url("   ")

    def test_missing_scheme_raises(self):
        with self.assertRaises(InvalidCreatorUrlError):
            parse_creator_url("www.instagram.com/example_creator")

    def test_unrecognized_host_raises(self):
        with self.assertRaises(InvalidCreatorUrlError):
            parse_creator_url("https://example.com/example_creator")

    def test_no_username_path_segment_raises(self):
        with self.assertRaises(InvalidCreatorUrlError):
            parse_creator_url("https://www.instagram.com/")

    def test_result_is_frozen(self):
        parsed = parse_creator_url("https://www.instagram.com/example_creator/")
        with self.assertRaises(Exception):
            parsed.username = "someone_else"


if __name__ == "__main__":
    unittest.main()
