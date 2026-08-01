from __future__ import annotations

import unittest

from .instagram_models import ReplyStyleConfig
from .reply_safety import (
    AUTO_ELIGIBLE,
    HUMAN_REVIEW_REQUIRED,
    NEVER_REPLY,
    check_reply_style,
    count_emojis,
    count_words,
    enforce_emoji_limit,
    find_banned_phrase,
    resolve_safety_route,
)

EMOJI_PATTERN = (
    "[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF"
    "\U00002700-\U000027BF]+"
)


class SafetyRouteTableTests(unittest.TestCase):
    def test_auto_eligible_matches_spec_exactly(self) -> None:
        self.assertEqual(
            AUTO_ELIGIBLE,
            {
                "compliment",
                "emoji_only",
                "travel_question",
                "location_question",
                "food_question",
                "outfit_question",
                "camera_question",
                "hotel_question",
                "casual_chat",
            },
        )

    def test_human_review_required_matches_spec_exactly(self) -> None:
        self.assertEqual(
            HUMAN_REVIEW_REQUIRED,
            {
                "flirting",
                "negative",
                "brand_collaboration",
                "sensitive",
                "unknown",
            },
        )

    def test_never_reply_matches_spec_exactly(self) -> None:
        self.assertEqual(NEVER_REPLY, {"spam"})

    def test_three_tiers_are_disjoint(self) -> None:
        self.assertEqual(AUTO_ELIGIBLE & HUMAN_REVIEW_REQUIRED, set())
        self.assertEqual(AUTO_ELIGIBLE & NEVER_REPLY, set())
        self.assertEqual(HUMAN_REVIEW_REQUIRED & NEVER_REPLY, set())

    def test_resolve_route_for_each_auto_eligible_category(self) -> None:
        for category in AUTO_ELIGIBLE:
            self.assertEqual(resolve_safety_route(category), "auto_eligible")

    def test_resolve_route_for_each_human_review_category(self) -> None:
        for category in HUMAN_REVIEW_REQUIRED:
            self.assertEqual(
                resolve_safety_route(category), "human_review_required"
            )

    def test_resolve_route_for_spam_is_never_reply(self) -> None:
        self.assertEqual(resolve_safety_route("spam"), "never_reply")

    def test_unrecognised_category_defaults_to_human_review(self) -> None:
        self.assertEqual(
            resolve_safety_route("totally_new_category"),
            "human_review_required",
        )


class WordAndEmojiCountingTests(unittest.TestCase):
    def test_count_words_basic(self) -> None:
        self.assertEqual(count_words("thank you so much"), 4)

    def test_count_words_ignores_emoji(self) -> None:
        self.assertEqual(count_words("thank you so much 🤍"), 4)

    def test_count_words_handles_accented_portuguese_text(self) -> None:
        self.assertEqual(
            count_words("O vestido está lindo"), 4
        )

    def test_count_emojis_single(self) -> None:
        self.assertEqual(
            count_emojis("thank you 🤍", emoji_pattern=EMOJI_PATTERN), 1
        )

    def test_count_emojis_none(self) -> None:
        self.assertEqual(
            count_emojis("thank you", emoji_pattern=EMOJI_PATTERN), 0
        )

    def test_count_emojis_multiple(self) -> None:
        self.assertEqual(
            count_emojis("sending hugs 🤍✨🌸", emoji_pattern=EMOJI_PATTERN), 3
        )


class BannedPhraseTests(unittest.TestCase):
    def test_detects_ai_mention(self) -> None:
        found = find_banned_phrase(
            "as an ai i cannot do that",
            banned_phrases=["as an ai"],
        )
        self.assertEqual(found, "as an ai")

    def test_case_insensitive(self) -> None:
        found = find_banned_phrase(
            "As An AI, thanks!",
            banned_phrases=["as an ai"],
        )
        self.assertEqual(found, "as an ai")

    def test_none_when_clean(self) -> None:
        found = find_banned_phrase(
            "thank you so much 🤍",
            banned_phrases=["as an ai", "language model"],
        )
        self.assertIsNone(found)


class CheckReplyStyleTests(unittest.TestCase):
    def _style(self, **overrides) -> ReplyStyleConfig:
        style = ReplyStyleConfig(
            minimum_words=5,
            maximum_words=25,
            maximum_emojis=2,
            banned_phrases=["as an ai"],
        )
        for key, value in overrides.items():
            setattr(style, key, value)
        return style

    def test_passes_valid_reply(self) -> None:
        check = check_reply_style(
            "thank you so much for this 🤍",
            style=self._style(),
            emoji_pattern=EMOJI_PATTERN,
        )
        self.assertTrue(check.passed)
        self.assertEqual(check.issues, [])

    def test_fails_below_minimum_words(self) -> None:
        check = check_reply_style(
            "thanks 🤍",
            style=self._style(),
            emoji_pattern=EMOJI_PATTERN,
        )
        self.assertFalse(check.passed)
        self.assertIn("below_minimum_words", check.issues)

    def test_fails_above_maximum_words(self) -> None:
        long_text = " ".join(["word"] * 30)
        check = check_reply_style(
            long_text,
            style=self._style(),
            emoji_pattern=EMOJI_PATTERN,
        )
        self.assertFalse(check.passed)
        self.assertIn("above_maximum_words", check.issues)

    def test_fails_above_maximum_emojis(self) -> None:
        check = check_reply_style(
            "thank you so much for this 🤍✨🌸",
            style=self._style(),
            emoji_pattern=EMOJI_PATTERN,
        )
        self.assertFalse(check.passed)
        self.assertIn("above_maximum_emojis", check.issues)

    def test_fails_on_banned_phrase(self) -> None:
        check = check_reply_style(
            "as an ai i am happy to help you today",
            style=self._style(),
            emoji_pattern=EMOJI_PATTERN,
        )
        self.assertFalse(check.passed)
        self.assertTrue(
            any(issue.startswith("banned_phrase:") for issue in check.issues)
        )

    def test_fails_on_empty_reply(self) -> None:
        check = check_reply_style(
            "   ",
            style=self._style(),
            emoji_pattern=EMOJI_PATTERN,
        )
        self.assertFalse(check.passed)
        self.assertIn("empty_reply", check.issues)


class EnforceEmojiLimitTests(unittest.TestCase):
    def test_no_change_when_within_limit(self) -> None:
        text = "thank you so much 🤍✨"
        result = enforce_emoji_limit(
            text, maximum_emojis=2, emoji_pattern=EMOJI_PATTERN
        )
        self.assertEqual(result, text)

    def test_clamps_excess_emojis(self) -> None:
        result = enforce_emoji_limit(
            "sending hugs 🤍✨🌸🥹",
            maximum_emojis=2,
            emoji_pattern=EMOJI_PATTERN,
        )
        self.assertEqual(
            count_emojis(result, emoji_pattern=EMOJI_PATTERN), 2
        )
        self.assertIn("sending hugs", result)


if __name__ == "__main__":
    unittest.main()
