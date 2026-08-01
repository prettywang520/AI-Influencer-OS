from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .instagram_models import ReplyRulesConfigError, ReplyStyleConfig
from .reply_brain import ReplyBrain, build_reply_brain
from .reply_safety import check_reply_style, enforce_emoji_limit

# ---------------------------------------------------------------------------
# Phase 8F — Instagram DM Reply Brain
# ---------------------------------------------------------------------------
#
# Nothing in this module ever sends anything to Instagram. It only
# classifies fan DM text and proposes one Aiko-style reply (or None,
# for never_reply categories) for a human to review.
#
# 14 of the 18 DM classifications reuse the exact same persona reply
# libraries as the comment pipeline (via ReplyBrain, imported
# unmodified from reply_brain.py — compliment, casual_chat,
# travel/location/food/outfit/camera/hotel_question, flirting,
# negative, spam, brand_collaboration, sensitive, unknown). Only 4 are
# DM-specific with no existing persona library (fan_support, sexual,
# harassment, booking_request); their keywords/fallback replies live
# in config/social/reply_rules.yaml's dm_extra_categories section,
# which the comment ReplyBrain never reads.

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "social" / "reply_rules.yaml"

_SAFE_FALLBACK_REPLY = "thank you so much for this 🤍"

# Hardcoded, not read from YAML — mirrors reply_safety.py's own
# reasoning (see resolve_safety_route there): a routing table this
# safety-critical must never be able to silently drift out of sync
# with the Phase 8F spec.
DM_AUTO_ELIGIBLE = frozenset(
    {
        "compliment",
        "casual_chat",
        "travel_question",
        "location_question",
        "food_question",
        "outfit_question",
        "camera_question",
        "hotel_question",
        "fan_support",
    }
)

DM_HUMAN_REVIEW_REQUIRED = frozenset(
    {
        "flirting",
        "negative",
        "brand_collaboration",
        "booking_request",
        "sensitive",
        "unknown",
    }
)

DM_NEVER_REPLY = frozenset({"sexual", "harassment", "spam"})

# Checked in this order; first keyword match wins. Safety-critical
# categories (sexual, harassment) are checked before anything else,
# same reasoning as reply_rules.yaml's own comment priority list
# checking spam/negative/flirting early. emoji-only messages are
# detected structurally (see DMReplyBrain.classify) and short-circuit
# straight to "compliment", the same reuse reply_rules.yaml documents
# for comments' emoji_only category — so it is intentionally absent
# from this list.
DM_CLASSIFICATION_PRIORITY = [
    "sexual",
    "harassment",
    "spam",
    "negative",
    "flirting",
    "booking_request",
    "brand_collaboration",
    "sensitive",
    "camera_question",
    "outfit_question",
    "hotel_question",
    "food_question",
    "travel_question",
    "location_question",
    "fan_support",
    "casual_chat",
    "compliment",
]


def resolve_dm_safety_route(classification: str) -> str:
    """
    Map a DM classification to its safety route.

    Any classification value never seen before defaults to
    human_review_required — an unrecognised category must never be
    auto-replied to.
    """
    if classification in DM_NEVER_REPLY:
        return "never_reply"

    if classification in DM_HUMAN_REVIEW_REQUIRED:
        return "human_review_required"

    if classification in DM_AUTO_ELIGIBLE:
        return "auto_eligible"

    return "human_review_required"


@dataclass(slots=True)
class DMClassificationResult:
    """
    Result of classifying one DM's text. A local, DM-scoped
    equivalent of reply_brain's ClassificationResult — kept separate
    because the DM classification set (sexual, harassment,
    booking_request, fan_support) is wider than
    instagram_models.ReplyClassification.
    """

    classification: str
    matched_keyword: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DMExtraCategoryConfig:
    """One of the 4 DM-only categories with no existing persona library."""

    name: str
    keywords: list[str] = field(default_factory=list)
    fallback_replies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DMReplyRulesConfig:
    """
    Fully-resolved DM-specific configuration, built from the
    dm_extra_categories / dm_style sections of
    config/social/reply_rules.yaml.
    """

    style: ReplyStyleConfig
    emoji_pattern: str
    extra_categories: dict[str, DMExtraCategoryConfig]

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style.to_dict(),
            "emoji_pattern": self.emoji_pattern,
            "extra_categories": {
                name: category.to_dict()
                for name, category in self.extra_categories.items()
            },
        }


def load_dm_reply_rules_config(
    *,
    config_path: str | Path | None = None,
    base_emoji_pattern: str | None = None,
    base_banned_phrases: list[str] | None = None,
) -> DMReplyRulesConfig:
    """
    Load the dm_extra_categories / dm_style sections of
    config/social/reply_rules.yaml.

    Independent loader (does not modify or depend on internal state
    of reply_brain.load_reply_rules_config), matching the pattern
    already used across every Phase 8 config loader. base_emoji_pattern
    / base_banned_phrases let the caller reuse an already-loaded
    ReplyRulesConfig's values instead of re-deriving them; DM banned
    phrases are the base list PLUS dm_style.extra_banned_phrases.
    """
    runtime_root = Path(__file__).resolve().parents[2]

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

    dm_extra_section = raw.get("dm_extra_categories") or {}
    categories_raw = dm_extra_section.get("categories") or {}

    extra_categories: dict[str, DMExtraCategoryConfig] = {}

    for name, category_data in categories_raw.items():
        category_data = category_data or {}

        extra_categories[name] = DMExtraCategoryConfig(
            name=str(name),
            keywords=list(category_data.get("keywords", [])),
            fallback_replies=list(category_data.get("fallback_replies", [])),
        )

    dm_style_section = raw.get("dm_style") or {}

    emoji_section = raw.get("emoji_detection") or {}
    emoji_pattern = base_emoji_pattern or str(emoji_section.get("pattern", ""))

    style_section = raw.get("style") or {}
    base_banned = (
        base_banned_phrases
        if base_banned_phrases is not None
        else list(style_section.get("banned_phrases", []))
    )
    extra_banned = list(dm_style_section.get("extra_banned_phrases", []))

    style = ReplyStyleConfig(
        minimum_words=int(dm_style_section.get("minimum_words", 3)),
        maximum_words=int(dm_style_section.get("maximum_words", 80)),
        maximum_emojis=int(dm_style_section.get("maximum_emojis", 2)),
        banned_phrases=base_banned + extra_banned,
    )

    return DMReplyRulesConfig(
        style=style,
        emoji_pattern=emoji_pattern,
        extra_categories=extra_categories,
    )


class DMReplyBrain:
    """
    Classifies a fan DM into one of the Phase 8F categories and
    generates one Aiko-style reply (max 80 words), reusing the
    existing comment ReplyBrain (unmodified) wherever a category maps
    onto an existing persona library.

    Never sends a DM. Never mentions being AI. Never generates a
    reply for a never_reply classification (sexual, harassment, spam)
    — those always resolve to None, by construction, regardless of
    what the caller does with the result.
    """

    def __init__(
        self,
        *,
        base: ReplyBrain,
        dm_config: DMReplyRulesConfig,
        random_seed: int | None = None,
    ) -> None:
        self.base = base
        self.dm_config = dm_config
        self.random = random.Random(random_seed)

    # -- classification ----------------------------------------------------

    def _keywords_for(self, category: str) -> list[str]:
        base_keywords = list(self.base._keywords_for(category))
        extra_category = self.dm_config.extra_categories.get(category)
        extra_keywords = list(extra_category.keywords) if extra_category else []

        return base_keywords + extra_keywords

    def classify(
        self,
        message_text: str | None,
        *,
        message_type: str | None = None,
    ) -> DMClassificationResult:
        text = (message_text or "").strip()
        lowered = text.lower()

        if message_type == "emoji" or self.base._is_emoji_only(text):
            return DMClassificationResult(
                classification="compliment",
                matched_keyword=None,
                reason="structural_emoji_message",
            )

        for category in DM_CLASSIFICATION_PRIORITY:
            for keyword in self._keywords_for(category):
                if keyword and self.base._keyword_matches(keyword, lowered):
                    return DMClassificationResult(
                        classification=category,
                        matched_keyword=keyword,
                        reason="keyword_match",
                    )

        return DMClassificationResult(
            classification="unknown",
            matched_keyword=None,
            reason="no_keyword_match",
        )

    # -- generation ----------------------------------------------------

    _MAXIMUM_GENERATION_ATTEMPTS = 20

    def _generate_from_pool(self, pool: list[str], *, used: set[str]) -> str:
        unused = [text for text in pool if text not in used]
        candidates = unused or pool

        for _ in range(self._MAXIMUM_GENERATION_ATTEMPTS):
            candidate = self.random.choice(candidates)

            if candidate not in used:
                return candidate

        return candidates[0]

    def _finalize(self, candidate: str, *, used: set[str]) -> str:
        """
        Defense-in-depth style check on the final DM text: word count,
        emoji count, and the extended DM banned-phrase list
        (political/religious/promises/contact-sharing/financial-
        medical-legal-advice triggers). The underlying reply pools are
        hand-curated to already comply, so a real failure here should
        never happen — if one ever does, fall back to a known-safe
        generic reply rather than surfacing the flagged text.
        """
        check = check_reply_style(
            candidate,
            style=self.dm_config.style,
            emoji_pattern=self.dm_config.emoji_pattern,
        )

        if "above_maximum_emojis" in check.issues:
            candidate = enforce_emoji_limit(
                candidate,
                maximum_emojis=self.dm_config.style.maximum_emojis,
                emoji_pattern=self.dm_config.emoji_pattern,
            )
            check = check_reply_style(
                candidate,
                style=self.dm_config.style,
                emoji_pattern=self.dm_config.emoji_pattern,
            )

        if not check.passed:
            if _SAFE_FALLBACK_REPLY not in used:
                return _SAFE_FALLBACK_REPLY

        return candidate

    def generate_reply(
        self,
        classification: str,
        *,
        used_reply_texts: set[str] | None = None,
    ) -> str | None:
        """
        Generate one Aiko-style DM reply, or None for a never_reply
        classification (requirement 18 — proposed_reply must stay
        null for sexual/harassment/spam).
        """
        if classification in DM_NEVER_REPLY:
            return None

        used = used_reply_texts or set()
        extra_category = self.dm_config.extra_categories.get(classification)

        if extra_category and extra_category.fallback_replies:
            candidate = self._generate_from_pool(
                extra_category.fallback_replies, used=used
            )
        else:
            candidate = self.base.generate_reply(
                classification, used_reply_texts=used
            )

        return self._finalize(candidate, used=used)


def build_dm_reply_brain(
    *,
    config_path: str | Path | None = None,
    random_seed: int | None = None,
) -> DMReplyBrain:
    base = build_reply_brain(config_path=config_path, random_seed=random_seed)

    dm_config = load_dm_reply_rules_config(
        config_path=config_path,
        base_emoji_pattern=base.config.emoji_pattern,
        base_banned_phrases=list(base.config.style.banned_phrases),
    )

    return DMReplyBrain(base=base, dm_config=dm_config, random_seed=random_seed)
