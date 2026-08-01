from __future__ import annotations

import re
from collections.abc import Iterable

from reply_models import (
    ContentContext,
    ReplyCandidate,
    ValidationResult,
)


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


class ReplyValidator:
    def __init__(
        self,
        maximum_words: int = 25,
        maximum_emojis: int = 3,
        prohibited_phrases: Iterable[str] | None = None,
    ) -> None:
        self.maximum_words = maximum_words
        self.maximum_emojis = maximum_emojis
        self.prohibited_phrases = {
            phrase.lower()
            for phrase in (
                prohibited_phrases
                or {
                    "dear customer",
                    "as an ai",
                    "i can assist you",
                    "contact customer service",
                }
            )
        }

    @staticmethod
    def count_words(text: str) -> int:
        return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))

    @staticmethod
    def count_emojis(text: str) -> int:
        return sum(len(match.group()) for match in EMOJI_PATTERN.finditer(text))

    @staticmethod
    def _find_unresolved_variables(text: str) -> list[str]:
        return re.findall(r"\{([a-zA-Z0-9_]+)\}", text)

    @staticmethod
    def _ai_claim_errors(
        text: str,
        context: ContentContext,
    ) -> list[str]:
        if not context.ai_generated:
            return []

        lowered = text.lower()
        physical_claims = {
            "i shot this with",
            "my photographer captured",
            "this spontaneous moment happened",
            "i tasted",
            "i stayed at",
        }

        return [
            claim
            for claim in physical_claims
            if claim in lowered
        ]

    def validate(
        self,
        candidate: ReplyCandidate,
        context: ContentContext,
        repeat_score: float = 0.0,
    ) -> ValidationResult:
        text = candidate.text.strip()
        failures: list[str] = []
        unsupported: list[str] = []

        word_count = self.count_words(text)
        emoji_count = self.count_emojis(text)

        if not text:
            failures.append("empty_reply")

        if word_count > self.maximum_words:
            failures.append("word_limit_exceeded")

        if emoji_count > self.maximum_emojis:
            failures.append("emoji_limit_exceeded")

        if text != text.lower():
            # Proper names and Japanese characters are allowed.
            words_with_caps = re.findall(r"\b[A-Z]{2,}\b", text)

            if words_with_caps:
                failures.append("excessive_uppercase")

        for phrase in self.prohibited_phrases:
            if phrase in text.lower():
                failures.append(f"prohibited_phrase:{phrase}")

        unresolved = self._find_unresolved_variables(text)

        if unresolved:
            failures.append("unresolved_context_variables")
            unsupported.extend(unresolved)

        ai_errors = self._ai_claim_errors(text, context)

        if ai_errors:
            failures.append("unsupported_physical_claim")
            unsupported.extend(ai_errors)

        if repeat_score >= 0.88:
            failures.append("repeat_score_rejected")
        elif repeat_score >= 0.60:
            failures.append("repeat_score_requires_rewrite")

        return ValidationResult(
            passed=not failures,
            failed_rules=failures,
            unsupported_claims=unsupported,
            word_count=word_count,
            emoji_count=emoji_count,
            repeat_score=repeat_score,
        )