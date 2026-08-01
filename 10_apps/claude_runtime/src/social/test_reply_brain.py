from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .instagram_models import ReplyRulesConfigError
from .reply_brain import (
    ReplyBrain,
    build_reply_brain,
    load_reply_rules_config,
)
from .reply_safety import check_reply_style, resolve_safety_route

# None of these tests access the real Instagram website. All generation
# tests use a fixed random_seed for determinism.


class LoadReplyRulesConfigTests(unittest.TestCase):
    def test_loads_real_config_file(self) -> None:
        config = load_reply_rules_config()

        self.assertTrue(
            config.reply_brain_root.as_posix().endswith(
                "03_personas/aiko/reply_brain"
            )
        )
        self.assertEqual(config.style.minimum_words, 5)
        self.assertEqual(config.style.maximum_words, 25)
        self.assertEqual(config.style.maximum_emojis, 2)
        self.assertIn("as an ai", config.style.banned_phrases)
        self.assertIn("compliment", config.categories)
        self.assertIn("spam", config.categories)
        self.assertIn("brand_collaboration", config.categories)
        self.assertGreater(len(config.classification_priority), 0)

    def test_missing_config_file_raises(self) -> None:
        missing_path = Path(__file__).resolve().parent / "does_not_exist.yaml"

        with self.assertRaises(ReplyRulesConfigError):
            load_reply_rules_config(config_path=missing_path)

    def test_empty_config_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_config = Path(temp_dir) / "reply_rules.yaml"
            empty_config.write_text("", encoding="utf-8")

            with self.assertRaises(ReplyRulesConfigError):
                load_reply_rules_config(config_path=empty_config)

    def test_missing_priority_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_config = Path(temp_dir) / "reply_rules.yaml"
            bad_config.write_text("persona: {}\n", encoding="utf-8")

            with self.assertRaises(ReplyRulesConfigError):
                load_reply_rules_config(config_path=bad_config)


class ClassificationScenarioTests(unittest.TestCase):
    """
    Covers the required scenarios: Portuguese comments, emoji-only
    comments, English questions, spam, flirting, and brand
    collaboration.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.brain = build_reply_brain(random_seed=7)

    def test_emoji_only_comment(self) -> None:
        result = self.brain.classify("😍🙌😍")
        self.assertEqual(result.classification, "emoji_only")
        self.assertEqual(resolve_safety_route(result.classification), "auto_eligible")

    def test_emoji_only_with_variation_selectors(self) -> None:
        result = self.brain.classify("❤️✨")
        self.assertEqual(result.classification, "emoji_only")

    def test_english_camera_question(self) -> None:
        result = self.brain.classify("what camera do you use for these shots?")
        self.assertEqual(result.classification, "camera_question")
        self.assertEqual(resolve_safety_route(result.classification), "auto_eligible")

    def test_english_location_question(self) -> None:
        result = self.brain.classify("where is this place?")
        self.assertEqual(result.classification, "location_question")

    def test_english_hotel_question(self) -> None:
        result = self.brain.classify("which hotel is this? it looks amazing")
        self.assertEqual(result.classification, "hotel_question")

    def test_spam_comment(self) -> None:
        result = self.brain.classify(
            "follow for follow, check my profile for guaranteed followers"
        )
        self.assertEqual(result.classification, "spam")
        self.assertEqual(resolve_safety_route(result.classification), "never_reply")

    def test_flirting_comment(self) -> None:
        result = self.brain.classify("marry me you're gorgeous")
        self.assertEqual(result.classification, "flirting")
        self.assertEqual(
            resolve_safety_route(result.classification), "human_review_required"
        )

    def test_brand_collaboration_comment(self) -> None:
        result = self.brain.classify(
            "hi! we would love to collab with you for a sponsored post"
        )
        self.assertEqual(result.classification, "brand_collaboration")
        self.assertEqual(
            resolve_safety_route(result.classification), "human_review_required"
        )

    def test_portuguese_compliment_is_unknown_not_misclassified(self) -> None:
        """
        Real data captured from Phase 8B: a Portuguese compliment
        ("what a beautiful special smile"). This is an English
        keyword classifier, so non-English text correctly falls back
        to unknown -> human_review_required rather than being guessed
        at or, worse, silently auto-replied to. This also regression
        -tests a real bug found during manual testing: "sorriso"
        (Portuguese for "smile") contains the substring "iso" (a
        camera_question trigger keyword) and was, before a word
        -boundary fix, misclassified as camera_question.
        """
        result = self.brain.classify("Um sorriso muito lindo especial")
        self.assertEqual(result.classification, "unknown")
        self.assertNotEqual(result.classification, "camera_question")
        self.assertEqual(
            resolve_safety_route(result.classification), "human_review_required"
        )

    def test_portuguese_outfit_comment_is_unknown(self) -> None:
        result = self.brain.classify(
            "O vestido está lindo você o menor perfeitamente"
        )
        self.assertEqual(result.classification, "unknown")

    def test_empty_comment_is_unknown(self) -> None:
        result = self.brain.classify("")
        self.assertEqual(result.classification, "unknown")

    def test_none_comment_is_unknown(self) -> None:
        result = self.brain.classify(None)
        self.assertEqual(result.classification, "unknown")


class GenerationStyleComplianceTests(unittest.TestCase):
    """
    Every generated reply must satisfy the Aiko style rules regardless
    of which category produced it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.brain = build_reply_brain(random_seed=123)
        cls.config = load_reply_rules_config()

    def test_generated_replies_satisfy_style_for_every_auto_eligible_category(
        self,
    ) -> None:
        categories = [
            "compliment",
            "emoji_only",
            "travel_question",
            "location_question",
            "food_question",
            "outfit_question",
            "camera_question",
            "hotel_question",
            "casual_chat",
        ]

        for category in categories:
            for _ in range(5):
                reply = self.brain.generate_reply(category, used_reply_texts=set())
                check = check_reply_style(
                    reply,
                    style=self.config.style,
                    emoji_pattern=self.config.emoji_pattern,
                )
                self.assertTrue(
                    check.passed,
                    f"{category} produced non-compliant reply "
                    f"{reply!r}: {check.issues}",
                )
                self.assertNotIn("as an ai", reply.lower())
                self.assertEqual(reply, reply.strip())

    def test_generated_replies_satisfy_style_for_human_review_categories(
        self,
    ) -> None:
        for category in ("flirting", "negative", "brand_collaboration", "sensitive", "unknown"):
            reply = self.brain.generate_reply(category, used_reply_texts=set())
            check = check_reply_style(
                reply,
                style=self.config.style,
                emoji_pattern=self.config.emoji_pattern,
            )
            self.assertTrue(
                check.passed,
                f"{category} produced non-compliant reply {reply!r}: {check.issues}",
            )

    def test_never_never_mentions_being_ai(self) -> None:
        for category in self.config.categories:
            if category == "spam":
                continue

            for _ in range(8):
                reply = self.brain.generate_reply(category, used_reply_texts=set())
                lowered = reply.lower()
                for banned in self.config.style.banned_phrases:
                    self.assertNotIn(banned, lowered)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_produces_same_sequence(self) -> None:
        brain_a = build_reply_brain(random_seed=99)
        brain_b = build_reply_brain(random_seed=99)

        sequence_a = [
            brain_a.generate_reply("compliment", used_reply_texts=set())
            for _ in range(5)
        ]
        sequence_b = [
            brain_b.generate_reply("compliment", used_reply_texts=set())
            for _ in range(5)
        ]

        self.assertEqual(sequence_a, sequence_b)

    def test_different_seeds_can_diverge(self) -> None:
        brain_a = build_reply_brain(random_seed=1)
        brain_b = build_reply_brain(random_seed=2)

        sequence_a = [
            brain_a.generate_reply("compliment", used_reply_texts=set())
            for _ in range(10)
        ]
        sequence_b = [
            brain_b.generate_reply("compliment", used_reply_texts=set())
            for _ in range(10)
        ]

        self.assertNotEqual(sequence_a, sequence_b)


class AntiRepeatTests(unittest.TestCase):
    def test_avoids_reusing_texts_already_in_used_set(self) -> None:
        brain = ReplyBrain(config=load_reply_rules_config(), random_seed=5)

        used: set[str] = set()

        for _ in range(15):
            reply = brain.generate_reply("compliment", used_reply_texts=used)
            self.assertNotIn(
                reply, used, "generate_reply must not repeat an identical reply"
            )
            used.add(reply)

    def test_falls_back_gracefully_when_pool_fully_exhausted(self) -> None:
        brain = ReplyBrain(config=load_reply_rules_config(), random_seed=5)

        # Exhaust every possible candidate for a small category by
        # pre-filling `used` with everything the brain could produce.
        candidates = set()
        for _ in range(200):
            candidates.add(
                brain.generate_reply("brand_collaboration", used_reply_texts=set())
            )

        # With the whole pool marked "used", generation must not raise.
        reply = brain.generate_reply(
            "brand_collaboration", used_reply_texts=candidates
        )
        self.assertIsInstance(reply, str)
        self.assertTrue(reply)


class WordCountExtensionTests(unittest.TestCase):
    def test_short_base_reply_is_extended_with_follow_up_when_needed(self) -> None:
        """
        location.yaml's follow_up: entries exist specifically so a
        short base reply can be naturally extended to satisfy the
        5-word minimum while staying on-persona.
        """
        brain = ReplyBrain(config=load_reply_rules_config(), random_seed=3)

        found_multi_sentence = False

        for _ in range(20):
            reply = brain.generate_reply("location_question", used_reply_texts=set())
            if "?" in reply and reply.count(" ") > 3:
                found_multi_sentence = True

        self.assertTrue(
            found_multi_sentence,
            "expected at least one extended (base + follow_up) reply "
            "across repeated generation",
        )


if __name__ == "__main__":
    unittest.main()
