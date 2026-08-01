from __future__ import annotations

import random
import re

from .content_models import ContentMoment, DailyContentPlan

# Aiko-voice caption style (Phase 9, requirement 13):
# - under 80 words
# - lowercase English preferred
# - may include short Japanese
# - emojis allowed but not excessive
# - natural, not robotic
# - Feed and Stories must not reuse identical sentences
# - include a question where natural
#
# Previously this engine returned the same fixed "rainy bookstore"
# text every single day regardless of the actual plan — a stub, not a
# generator. This version builds every caption from the real
# ContentMoment fields (behavior, emotion, venue, daily_details) so
# output actually reflects that day's content.

_WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

_JAPANESE_SNIPPETS = [
    "一期一会",
    "また来るね",
    "こっちかな",
    "今日もいい一日",
    "少し贅沢な時間",
    "また会いに来るね",
]

_EMOJI_POOL = ["🤍", "✨", "🌿", "📖", "☕", "🌸", "😊", "☔"]

_FEED_QUESTIONS = [
    "what's your favorite way to spend a day like this?",
    "have you ever found something unexpected somewhere like this?",
    "where should the next journey begin?",
    "would you come explore this with me?",
    "what would you have done here?",
]

_STORY_QUESTIONS = [
    "guess where next?",
    "can you tell?",
    "worth it, right?",
]

_REEL_QUESTIONS = [
    "where should the next journey begin?",
    "which moment was your favorite?",
    "would you have stayed longer too?",
]


def _count_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def _clip_to_words(text: str, maximum_words: int) -> str:
    words = text.split()

    if len(words) <= maximum_words:
        return text

    return " ".join(words[:maximum_words]).rstrip(",.;:")


def _short_subject(text: str) -> str:
    """
    Lowercase, trailing-period-stripped, first-clause-only version of
    a behavior/emotion/interaction sentence, suitable for weaving into
    a caption without duplicating a full generated-prompt sentence.
    """
    cleaned = text.strip().rstrip(".").strip()
    first_clause = cleaned.split(".")[0]

    return first_clause.strip().lower()


class CaptionEngine:
    def apply(self, plan: DailyContentPlan) -> None:
        # Seeded per-day (not globally random) so the same plan
        # produces the same captions if regenerated, without pinning
        # every day to identical text like the previous stub did.
        rng = random.Random(f"{plan.date}:{plan.venue}:{plan.theme}")

        plan.feed_caption = self._feed_caption(plan, rng)
        plan.story_captions = self._story_captions(plan, rng)
        plan.reel_caption = self._reel_caption(plan, rng)

        # A Feed/Story caption collision is only possible if two
        # completely different moments happened to render into the
        # exact same sentence — retry once with a different emoji/
        # question pick before accepting (requirement 13: Feed and
        # Stories must not reuse identical sentences).
        attempts = 0

        while (
            plan.feed_caption in plan.story_captions
            and attempts < 5
        ):
            plan.feed_caption = self._feed_caption(plan, rng)
            attempts += 1

        plan.feed.caption = plan.feed_caption

        for story, caption in zip(
            plan.stories,
            plan.story_captions,
            strict=True,
        ):
            story.caption = caption

    def _feed_caption(
        self,
        plan: DailyContentPlan,
        rng: random.Random,
    ) -> str:
        behavior = _short_subject(plan.feed.behavior)
        emotion = _short_subject(plan.feed.emotion)
        emoji = "".join(
            rng.sample(_EMOJI_POOL, k=rng.randint(1, 2))
        )
        question = rng.choice(_FEED_QUESTIONS)

        japanese = (
            f" {rng.choice(_JAPANESE_SNIPPETS)}。"
            if rng.random() < 0.5
            else ""
        )

        text = (
            f"{behavior} at {plan.venue} today {emoji}{japanese} "
            f"feeling {emotion}. {question}"
        )

        return _clip_to_words(text.strip(), 80)

    def _story_captions(
        self,
        plan: DailyContentPlan,
        rng: random.Random,
    ) -> list[str]:
        captions: list[str] = []
        used: set[str] = set()

        for moment in plan.stories:
            caption = self._one_story_caption(moment, rng)
            attempts = 0

            while caption in used and attempts < 5:
                caption = self._one_story_caption(moment, rng)
                attempts += 1

            used.add(caption)
            captions.append(caption)

        return captions

    def _one_story_caption(
        self,
        moment: ContentMoment,
        rng: random.Random,
    ) -> str:
        behavior = _short_subject(moment.behavior)
        emoji = rng.choice(_EMOJI_POOL)

        japanese = (
            f" {rng.choice(_JAPANESE_SNIPPETS)}"
            if rng.random() < 0.5
            else ""
        )

        question = (
            f" {rng.choice(_STORY_QUESTIONS)}"
            if rng.random() < 0.4
            else ""
        )

        text = f"{behavior} {emoji}{japanese}{question}"

        return _clip_to_words(text.strip(), 80)

    def _reel_caption(
        self,
        plan: DailyContentPlan,
        rng: random.Random,
    ) -> str:
        emoji = "".join(
            rng.sample(_EMOJI_POOL, k=rng.randint(1, 2))
        )
        question = rng.choice(_REEL_QUESTIONS)

        summary = _short_subject(
            plan.story_summary
            or f"a full day exploring {plan.venue}"
        )

        text = f"{summary} {emoji} {question}"

        return _clip_to_words(text.strip(), 80)
