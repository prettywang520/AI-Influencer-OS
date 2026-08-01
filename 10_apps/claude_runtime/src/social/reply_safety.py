from __future__ import annotations

import re

from .instagram_models import ReplyStyleCheck, ReplyStyleConfig, SafetyRoute

# ---------------------------------------------------------------------------
# Safety Gate — Phase 8C
# ---------------------------------------------------------------------------
#
# This is the single source of truth for routing a classified comment
# to auto-reply, human review, or no reply at all. It is deliberately
# hardcoded (not read from config/social/reply_rules.yaml) so the
# routing table can never silently drift from the Phase 8C spec.

AUTO_ELIGIBLE = frozenset(
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
    }
)

HUMAN_REVIEW_REQUIRED = frozenset(
    {
        "flirting",
        "negative",
        "brand_collaboration",
        "sensitive",
        "unknown",
    }
)

NEVER_REPLY = frozenset({"spam"})


def resolve_safety_route(classification: str) -> SafetyRoute:
    """
    Map a classification to its safety route.

    Any classification value this table has never seen before is
    treated as human_review_required rather than auto_eligible: an
    unrecognised category must never be auto-replied to by default.
    """
    if classification in NEVER_REPLY:
        return "never_reply"

    if classification in HUMAN_REVIEW_REQUIRED:
        return "human_review_required"

    if classification in AUTO_ELIGIBLE:
        return "auto_eligible"

    return "human_review_required"


_WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def count_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def _compile_emoji_unit_pattern(emoji_pattern: str) -> re.Pattern[str]:
    """
    Build a pattern that matches ONE emoji "unit" at a time: a single
    base character from the configured emoji ranges, plus an optional
    trailing U+FE0F variation selector.

    emoji_pattern (from config) is a character class with a trailing
    `+`, e.g. "[...]+", meant for "strip every emoji" use (where a
    greedy run length doesn't matter). For counting/clamping, that
    same `+` is wrong: it silently merges any run of adjacent-but-
    distinct emoji (e.g. "🤍✨🌸") into a single match, undercounting
    them, while a single variation-selector sequence like "☺️" (base
    char + U+FE0F) is two code points and gets overcounted as 2 by a
    naive character-length sum. Matching one base char (+ optional
    U+FE0F) at a time, without `+`, fixes both.
    """
    base = emoji_pattern[:-1] if emoji_pattern.endswith("+") else emoji_pattern
    return re.compile(base + "\U0000FE0F?", flags=re.UNICODE)


def count_emojis(text: str, *, emoji_pattern: str) -> int:
    return len(_compile_emoji_unit_pattern(emoji_pattern).findall(text))


def find_banned_phrase(text: str, *, banned_phrases: list[str]) -> str | None:
    lowered = text.lower()

    for phrase in banned_phrases:
        if phrase.lower() in lowered:
            return phrase

    return None


def check_reply_style(
    text: str,
    *,
    style: ReplyStyleConfig,
    emoji_pattern: str,
) -> ReplyStyleCheck:
    """
    Validate one candidate Aiko reply against the style rules:
    5-25 words, max 2 emojis, no banned/AI-mention phrases.
    """
    cleaned = text.strip()
    issues: list[str] = []

    word_count = count_words(cleaned)
    emoji_count = count_emojis(cleaned, emoji_pattern=emoji_pattern)

    if not cleaned:
        issues.append("empty_reply")

    if word_count < style.minimum_words:
        issues.append("below_minimum_words")

    if word_count > style.maximum_words:
        issues.append("above_maximum_words")

    if emoji_count > style.maximum_emojis:
        issues.append("above_maximum_emojis")

    banned = find_banned_phrase(cleaned, banned_phrases=style.banned_phrases)

    if banned:
        issues.append(f"banned_phrase:{banned}")

    return ReplyStyleCheck(
        passed=not issues,
        issues=issues,
        word_count=word_count,
        emoji_count=emoji_count,
    )


def enforce_emoji_limit(
    text: str,
    *,
    maximum_emojis: int,
    emoji_pattern: str,
) -> str:
    """
    Best-effort clamp: keep only the first `maximum_emojis` emoji
    units found in the text, dropping any beyond that.
    """
    compiled = _compile_emoji_unit_pattern(emoji_pattern)
    matches = list(compiled.finditer(text))

    if len(matches) <= maximum_emojis:
        return text

    pieces: list[str] = []
    cursor = 0

    for index, match in enumerate(matches):
        if index < maximum_emojis:
            pieces.append(text[cursor:match.end()])
        else:
            pieces.append(text[cursor:match.start()])

        cursor = match.end()

    pieces.append(text[cursor:])

    rebuilt = "".join(pieces)
    return re.sub(r"[ \t]+", " ", rebuilt).strip()
