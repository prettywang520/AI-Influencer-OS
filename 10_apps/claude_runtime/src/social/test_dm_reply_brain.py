from __future__ import annotations

import unittest

from .dm_reply_brain import (
    DM_AUTO_ELIGIBLE,
    DM_HUMAN_REVIEW_REQUIRED,
    DM_NEVER_REPLY,
    DMClassificationResult,
    build_dm_reply_brain,
    load_dm_reply_rules_config,
    resolve_dm_safety_route,
)
from .reply_safety import check_reply_style, count_emojis, count_words

# None of these tests access the real Instagram website. All
# generation tests use a fixed random_seed for determinism.


class LoadDMReplyRulesConfigTests(unittest.TestCase):
    def test_loads_real_config_sections(self) -> None:
        config = load_dm_reply_rules_config()

        self.assertEqual(config.style.maximum_words, 80)
        self.assertEqual(config.style.maximum_emojis, 2)
        self.assertIn("i promise", config.style.banned_phrases)
        # base comment banned_phrases are still layered in.
        self.assertIn("as an ai", config.style.banned_phrases)

        for category in ("sexual", "harassment", "booking_request", "fan_support"):
            self.assertIn(category, config.extra_categories)

        self.assertEqual(config.extra_categories["sexual"].fallback_replies, [])
        self.assertEqual(config.extra_categories["harassment"].fallback_replies, [])
        self.assertGreater(
            len(config.extra_categories["booking_request"].fallback_replies), 0
        )
        self.assertGreater(
            len(config.extra_categories["fan_support"].fallback_replies), 0
        )


class SafetyRouteTests(unittest.TestCase):
    def test_routing_table_matches_spec(self) -> None:
        for category in (
            "compliment",
            "casual_chat",
            "travel_question",
            "location_question",
            "food_question",
            "outfit_question",
            "camera_question",
            "hotel_question",
            "fan_support",
        ):
            self.assertEqual(resolve_dm_safety_route(category), "auto_eligible")

        for category in (
            "flirting",
            "negative",
            "brand_collaboration",
            "booking_request",
            "sensitive",
            "unknown",
        ):
            self.assertEqual(
                resolve_dm_safety_route(category), "human_review_required"
            )

        for category in ("sexual", "harassment", "spam"):
            self.assertEqual(resolve_dm_safety_route(category), "never_reply")

    def test_unrecognised_category_defaults_to_human_review(self) -> None:
        self.assertEqual(
            resolve_dm_safety_route("something_never_seen_before"),
            "human_review_required",
        )

    def test_routing_sets_are_disjoint_and_complete(self) -> None:
        self.assertEqual(
            DM_AUTO_ELIGIBLE & DM_HUMAN_REVIEW_REQUIRED, frozenset()
        )
        self.assertEqual(DM_AUTO_ELIGIBLE & DM_NEVER_REPLY, frozenset())
        self.assertEqual(DM_HUMAN_REVIEW_REQUIRED & DM_NEVER_REPLY, frozenset())
        self.assertEqual(
            len(DM_AUTO_ELIGIBLE) + len(DM_HUMAN_REVIEW_REQUIRED) + len(DM_NEVER_REPLY),
            18,
        )


class ClassificationScenarioTests(unittest.TestCase):
    """
    Covers the required Phase 8F scenarios: normal fan compliment,
    travel question, emoji-only DM, flirting, sexual content,
    harassment, spam, brand collaboration, booking request.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.brain = build_dm_reply_brain(random_seed=7)

    def test_normal_fan_compliment(self) -> None:
        result = self.brain.classify("you are so beautiful, love this!")
        self.assertIsInstance(result, DMClassificationResult)
        self.assertEqual(result.classification, "compliment")
        self.assertEqual(resolve_dm_safety_route(result.classification), "auto_eligible")

    def test_travel_question(self) -> None:
        result = self.brain.classify("do you have any tips for my trip to japan?")
        self.assertEqual(result.classification, "travel_question")
        self.assertEqual(resolve_dm_safety_route(result.classification), "auto_eligible")

    def test_emoji_only_dm_via_message_type(self) -> None:
        result = self.brain.classify("😍🙌😍", message_type="emoji")
        self.assertEqual(result.classification, "compliment")
        self.assertEqual(result.reason, "structural_emoji_message")

    def test_emoji_only_dm_detected_structurally_without_message_type(self) -> None:
        result = self.brain.classify("✨🌸")
        self.assertEqual(result.classification, "compliment")

    def test_flirting(self) -> None:
        result = self.brain.classify("marry me please")
        self.assertEqual(result.classification, "flirting")
        self.assertEqual(
            resolve_dm_safety_route(result.classification), "human_review_required"
        )

    def test_sexual_content(self) -> None:
        result = self.brain.classify("send nudes")
        self.assertEqual(result.classification, "sexual")
        self.assertEqual(resolve_dm_safety_route(result.classification), "never_reply")

    def test_harassment(self) -> None:
        result = self.brain.classify("kys, i hate you so much")
        self.assertEqual(result.classification, "harassment")
        self.assertEqual(resolve_dm_safety_route(result.classification), "never_reply")

    def test_spam(self) -> None:
        result = self.brain.classify("follow for follow, check my profile")
        self.assertEqual(result.classification, "spam")
        self.assertEqual(resolve_dm_safety_route(result.classification), "never_reply")

    def test_brand_collaboration(self) -> None:
        result = self.brain.classify(
            "hi! we would love a paid collaboration with your account"
        )
        self.assertEqual(result.classification, "brand_collaboration")
        self.assertEqual(
            resolve_dm_safety_route(result.classification), "human_review_required"
        )

    def test_booking_request(self) -> None:
        result = self.brain.classify("hi, i'd like to book you for an event")
        self.assertEqual(result.classification, "booking_request")
        self.assertEqual(
            resolve_dm_safety_route(result.classification), "human_review_required"
        )

    def test_fan_support(self) -> None:
        result = self.brain.classify("you always make my day, keep going!!")
        self.assertEqual(result.classification, "fan_support")
        self.assertEqual(resolve_dm_safety_route(result.classification), "auto_eligible")

    def test_casual_chat(self) -> None:
        result = self.brain.classify("same, i would too")
        self.assertEqual(result.classification, "casual_chat")

    def test_negative(self) -> None:
        result = self.brain.classify("this looks so fake and boring honestly")
        self.assertEqual(result.classification, "negative")
        self.assertEqual(
            resolve_dm_safety_route(result.classification), "human_review_required"
        )

    def test_sensitive(self) -> None:
        result = self.brain.classify("can we meet in person sometime?")
        self.assertEqual(result.classification, "sensitive")

    def test_unrecognisable_text_is_unknown(self) -> None:
        result = self.brain.classify("bonjour ca va")
        self.assertEqual(result.classification, "unknown")
        self.assertEqual(
            resolve_dm_safety_route(result.classification), "human_review_required"
        )

    def test_empty_text_is_unknown(self) -> None:
        result = self.brain.classify(None)
        self.assertEqual(result.classification, "unknown")


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.brain = build_dm_reply_brain(random_seed=11)

    def test_never_reply_categories_never_generate_text(self) -> None:
        for category in ("sexual", "harassment", "spam"):
            reply = self.brain.generate_reply(category, used_reply_texts=set())
            self.assertIsNone(reply)

    def test_auto_eligible_category_generates_reply(self) -> None:
        reply = self.brain.generate_reply("compliment", used_reply_texts=set())
        self.assertIsInstance(reply, str)
        self.assertTrue(reply.strip())

    def test_booking_request_uses_professional_draft(self) -> None:
        reply = self.brain.generate_reply("booking_request", used_reply_texts=set())
        self.assertIsInstance(reply, str)
        self.assertIn("team", reply)

    def test_fan_support_uses_local_pool(self) -> None:
        reply = self.brain.generate_reply("fan_support", used_reply_texts=set())
        self.assertIsInstance(reply, str)
        self.assertTrue(reply.strip())

    def test_generated_replies_respect_dm_style(self) -> None:
        config = self.brain.dm_config

        for category in (
            "compliment",
            "casual_chat",
            "travel_question",
            "location_question",
            "food_question",
            "outfit_question",
            "camera_question",
            "hotel_question",
            "fan_support",
            "flirting",
            "negative",
            "brand_collaboration",
            "booking_request",
            "sensitive",
            "unknown",
        ):
            for _ in range(5):
                reply = self.brain.generate_reply(category, used_reply_texts=set())
                self.assertIsNotNone(reply)

                word_count = count_words(reply)
                emoji_count = count_emojis(reply, emoji_pattern=config.emoji_pattern)

                self.assertLessEqual(
                    word_count, 80, f"{category} reply too long: {reply!r}"
                )
                self.assertLessEqual(
                    emoji_count, 2, f"{category} reply has too many emojis: {reply!r}"
                )

    def test_never_mentions_being_ai(self) -> None:
        config = self.brain.dm_config

        for category in ("compliment", "fan_support", "booking_request", "unknown"):
            for _ in range(10):
                reply = self.brain.generate_reply(category, used_reply_texts=set())
                check = check_reply_style(
                    reply, style=config.style, emoji_pattern=config.emoji_pattern
                )
                self.assertTrue(
                    check.passed, f"{category} reply failed style check: {reply!r} {check.issues}"
                )

    def test_anti_repeat_avoids_recently_used_text(self) -> None:
        used: set[str] = set()
        pool_size = len(self.brain.dm_config.extra_categories["fan_support"].fallback_replies)

        for _ in range(pool_size):
            reply = self.brain.generate_reply("fan_support", used_reply_texts=used)
            self.assertNotIn(reply, used)
            used.add(reply)


if __name__ == "__main__":
    unittest.main()
