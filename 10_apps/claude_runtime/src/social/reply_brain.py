from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import yaml

from .instagram_models import (
    ClassificationResult,
    ReplyCategoryConfig,
    ReplyGenerationError,
    ReplyRulesConfig,
    ReplyRulesConfigError,
    ReplyStyleConfig,
)
from .reply_safety import check_reply_style, count_words, enforce_emoji_limit

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "social" / "reply_rules.yaml"

DEFAULT_EMOJI_PATTERN = (
    "[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF"
    "\U00002700-\U000027BF\U0000FE0F\U0000200D]+"
)

_FALLBACK_REPLY = "thank you so much 🤍"


def _runtime_root() -> Path:
    """
    reply_brain.py location:

    10_apps/claude_runtime/src/social/reply_brain.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    """
    parents[4] resolves to the AI-Influencer-OS repository root, where
    03_personas/aiko/reply_brain/ lives.
    """
    return Path(__file__).resolve().parents[4]


def load_reply_rules_config(
    *,
    config_path: str | Path | None = None,
) -> ReplyRulesConfig:
    runtime_root = _runtime_root()
    project_root = _project_root()

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (runtime_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve()
    )

    if not resolved_config_path.exists():
        raise ReplyRulesConfigError(
            f"Reply rules config not found: {resolved_config_path}"
        )

    try:
        with resolved_config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ReplyRulesConfigError(
            f"Invalid YAML in {resolved_config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict) or not raw:
        raise ReplyRulesConfigError(
            f"Reply rules config is empty or invalid: {resolved_config_path}"
        )

    persona_section = raw.get("persona") or {}
    style_section = raw.get("style") or {}
    emoji_section = raw.get("emoji_detection") or {}
    classification_section = raw.get("classification") or {}

    reply_brain_root_value = str(
        persona_section.get(
            "reply_brain_root",
            "03_personas/aiko/reply_brain",
        )
    )

    reply_brain_root = (project_root / reply_brain_root_value).resolve()

    style = ReplyStyleConfig(
        minimum_words=int(style_section.get("minimum_words", 5)),
        maximum_words=int(style_section.get("maximum_words", 25)),
        maximum_emojis=int(style_section.get("maximum_emojis", 2)),
        banned_phrases=list(style_section.get("banned_phrases", [])),
    )

    emoji_pattern = str(
        emoji_section.get("pattern", DEFAULT_EMOJI_PATTERN)
    )

    priority = list(classification_section.get("priority", []))

    if not priority:
        raise ReplyRulesConfigError(
            "classification.priority must not be empty"
        )

    categories_raw = classification_section.get("categories") or {}
    categories: dict[str, ReplyCategoryConfig] = {}

    for name, category_data in categories_raw.items():
        category_data = category_data or {}

        categories[name] = ReplyCategoryConfig(
            name=str(name),
            source_library=category_data.get("source_library"),
            keywords=list(category_data.get("keywords", [])),
            fallback_replies=list(
                category_data.get("fallback_replies", [])
            ),
        )

    return ReplyRulesConfig(
        reply_brain_root=reply_brain_root,
        style=style,
        emoji_pattern=emoji_pattern,
        classification_priority=priority,
        categories=categories,
    )


def _extract_candidates(node: Any) -> list[tuple[str, int]]:
    """
    Recursively pull (text, weight) reply candidates out of a
    replies: / reply_groups: / follow_up: YAML section, tolerant of
    the several shapes used across the existing persona library files:

    - flat list of {id, text, weight, ...} dicts (compliments.yaml)
    - dict of group_name -> list of {id, text} dicts (location.yaml)
    - flat list of plain strings (some follow_up: sections)
    """
    results: list[tuple[str, int]] = []

    if isinstance(node, str):
        cleaned = node.strip()

        if cleaned:
            results.append((cleaned, 1))

    elif isinstance(node, dict):
        if "text" in node and isinstance(node["text"], str):
            weight_value = node.get("weight", 1)

            try:
                weight = max(int(weight_value), 1)
            except (TypeError, ValueError):
                weight = 1

            cleaned = node["text"].strip()

            if cleaned:
                results.append((cleaned, weight))
        else:
            for value in node.values():
                results.extend(_extract_candidates(value))

    elif isinstance(node, list):
        for item in node:
            results.extend(_extract_candidates(item))

    return results


class ReplyBrain:
    """
    Classifies a comment into one of the Phase 8C categories and
    generates one Aiko-style reply, reusing the existing persona reply
    libraries under 03_personas/aiko/reply_brain/ where possible.

    Never sends anything to Instagram. Never mentions being AI.
    """

    def __init__(
        self,
        *,
        config: ReplyRulesConfig,
        random_seed: int | None = None,
    ) -> None:
        self.config = config
        self.random = random.Random(random_seed)

        self._library_cache: dict[str, dict[str, Any]] = {}
        self._keyword_cache: dict[str, list[str]] = {}

    # -- library loading ----------------------------------------------------

    def _load_library_yaml(self, name: str) -> dict[str, Any]:
        if name in self._library_cache:
            return self._library_cache[name]

        path = self.config.reply_brain_root / f"{name}.yaml"

        if not path.exists():
            raise ReplyRulesConfigError(
                f"Reply brain library not found: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ReplyRulesConfigError(
                f"Invalid YAML in {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ReplyRulesConfigError(
                f"Expected a YAML mapping in {path}"
            )

        self._library_cache[name] = data
        return data

    def _load_library_triggers(self, name: str) -> list[str]:
        data = self._load_library_yaml(name)
        triggers = data.get("triggers") or {}

        exact = triggers.get("exact") or []
        contains = triggers.get("contains") or []

        return [str(keyword) for keyword in list(exact) + list(contains)]

    def _load_spam_keywords(self) -> list[str]:
        data = self._load_library_yaml("spam")
        categories = data.get("categories") or {}
        keywords: list[str] = []

        for category_data in categories.values():
            if isinstance(category_data, dict):
                keywords.extend(
                    str(keyword)
                    for keyword in category_data.get("keywords", []) or []
                )

        return keywords

    def _keywords_for(self, category: str) -> list[str]:
        if category in self._keyword_cache:
            return self._keyword_cache[category]

        category_config = self.config.categories.get(category)
        keywords = list(category_config.keywords) if category_config else []
        source_library = category_config.source_library if category_config else None

        if source_library == "spam":
            keywords = keywords + self._load_spam_keywords()
        elif source_library:
            keywords = keywords + self._load_library_triggers(source_library)

        self._keyword_cache[category] = keywords
        return keywords

    def _reply_candidates(self, classification: str) -> list[tuple[str, int]]:
        category_config = self.config.categories.get(classification)
        source_library = category_config.source_library if category_config else None

        candidates: list[tuple[str, int]] = []

        if source_library:
            data = self._load_library_yaml(source_library)
            node = data.get("replies")

            if node is None:
                node = data.get("reply_groups")

            if node:
                candidates.extend(_extract_candidates(node))

        if category_config and category_config.fallback_replies:
            candidates.extend(
                (text, 5) for text in category_config.fallback_replies
            )

        return candidates

    def _follow_up_candidates(self, classification: str) -> list[str]:
        category_config = self.config.categories.get(classification)
        source_library = category_config.source_library if category_config else None

        if not source_library:
            return []

        data = self._load_library_yaml(source_library)
        node = data.get("follow_up")

        if not node:
            return []

        return [text for text, _ in _extract_candidates(node)]

    def _viable_candidates(
        self,
        classification: str,
    ) -> tuple[list[tuple[str, int]], list[str]]:
        """
        Candidates + follow-ups, with a length-viability filter applied.

        Some source libraries (e.g. engagement.yaml, used for
        casual_chat) have no follow_up: section at all. A base
        candidate under the minimum word count can only be brought
        into range by extending it with a follow-up, so when no
        follow-ups exist, candidates that are already too short (and
        therefore unfixable) are dropped from the selection pool,
        provided at least one longer candidate remains. This is
        applied before random selection rather than after, so a
        hopeless-too-short pick is never made in the first place.
        """
        candidates = self._reply_candidates(classification)

        if not candidates:
            candidates = [(_FALLBACK_REPLY, 1)]

        follow_ups = self._follow_up_candidates(classification)

        if not follow_ups:
            long_enough = [
                candidate
                for candidate in candidates
                if count_words(candidate[0]) >= self.config.style.minimum_words
            ]
            candidates = long_enough or candidates

        return candidates, follow_ups

    # -- structural detection -------------------------------------------------

    def _is_emoji_only(self, text: str) -> bool:
        if not text.strip():
            return False

        compiled = re.compile(self.config.emoji_pattern, flags=re.UNICODE)
        stripped = compiled.sub("", text)

        return not stripped.strip()

    # -- classification ----------------------------------------------------

    @staticmethod
    def _keyword_matches(keyword: str, lowered_text: str) -> bool:
        """
        Word-boundary-aware keyword match.

        A plain substring check would let short keywords like "iso"
        (a camera_question trigger) falsely match inside unrelated
        words in other languages — e.g. the Portuguese "sorriso"
        ("smile") contains "iso". \\b boundaries prevent that while
        still matching multi-word phrases like "marry me" correctly.
        """
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        return re.search(pattern, lowered_text, flags=re.UNICODE) is not None

    def classify(self, comment_text: str | None) -> ClassificationResult:
        text = (comment_text or "").strip()
        lowered = text.lower()

        if self._is_emoji_only(text):
            return ClassificationResult(
                classification="emoji_only",
                matched_keyword=None,
                reason="structural_emoji_only_match",
            )

        for category in self.config.classification_priority:
            if category in ("emoji_only", "unknown"):
                continue

            for keyword in self._keywords_for(category):
                if keyword and self._keyword_matches(keyword, lowered):
                    return ClassificationResult(
                        classification=category,
                        matched_keyword=keyword,
                        reason="keyword_match",
                    )

        return ClassificationResult(
            classification="unknown",
            matched_keyword=None,
            reason="no_keyword_match",
        )

    # -- generation ----------------------------------------------------

    _MAXIMUM_GENERATION_ATTEMPTS = 20

    def _generate_one_attempt(
        self,
        classification: str,
        *,
        candidates: list[tuple[str, int]],
        follow_ups: list[str],
        used: set[str],
    ) -> str:
        unused = [candidate for candidate in candidates if candidate[0] not in used]
        pool = unused or candidates

        texts = [text for text, _ in pool]
        weights = [weight for _, weight in pool]

        final_text = self.random.choices(texts, weights=weights, k=1)[0]

        check = check_reply_style(
            final_text,
            style=self.config.style,
            emoji_pattern=self.config.emoji_pattern,
        )

        if "below_minimum_words" in check.issues:
            if follow_ups:
                extension = self.random.choice(follow_ups)
                extended_text = f"{final_text} {extension}".strip()

                extended_check = check_reply_style(
                    extended_text,
                    style=self.config.style,
                    emoji_pattern=self.config.emoji_pattern,
                )

                if extended_check.word_count <= self.config.style.maximum_words:
                    final_text = extended_text
                    check = extended_check

        if check.emoji_count > self.config.style.maximum_emojis:
            final_text = enforce_emoji_limit(
                final_text,
                maximum_emojis=self.config.style.maximum_emojis,
                emoji_pattern=self.config.emoji_pattern,
            )

        return final_text

    def generate_reply(
        self,
        classification: str,
        *,
        used_reply_texts: set[str] | None = None,
    ) -> str:
        """
        Generate one Aiko-style reply for a classification, never
        repeating a text already present in used_reply_texts.

        The follow-up extension step (used to satisfy the minimum
        word count) happens AFTER a base candidate is chosen, so the
        anti-repeat check must be applied to the final, post-extension
        text, not just the base candidate — otherwise two different
        base picks could still extend into an identical final reply.
        A bounded retry loop re-rolls until an unused final text is
        found; if the whole pool is genuinely exhausted, the last
        attempt is returned rather than raising or looping forever.
        """
        used = used_reply_texts or set()

        candidates, follow_ups = self._viable_candidates(classification)

        last_attempt = _FALLBACK_REPLY

        for _ in range(self._MAXIMUM_GENERATION_ATTEMPTS):
            last_attempt = self._generate_one_attempt(
                classification,
                candidates=candidates,
                follow_ups=follow_ups,
                used=used,
            )

            if last_attempt not in used:
                return last_attempt

        return last_attempt


def build_reply_brain(
    *,
    config_path: str | Path | None = None,
    random_seed: int | None = None,
) -> ReplyBrain:
    config = load_reply_rules_config(config_path=config_path)
    return ReplyBrain(config=config, random_seed=random_seed)
