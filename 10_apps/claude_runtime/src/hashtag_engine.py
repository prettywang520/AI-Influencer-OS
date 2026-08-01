from __future__ import annotations

import re

from .content_models import DailyContentPlan

# Previously this engine returned the same fixed bookstore/tokyo/rain
# hashtag set every single day regardless of the actual plan — a
# stub, not a generator. This version derives tags from the real
# plan (venue, theme, city, country) so output actually reflects that
# day's content.

_BASE_TAGS = [
    "#aikotravel",
    "#travelcreator",
    "#luxurytravel",
    "#travelstorytelling",
    "#editorialtravel",
]

# Extra tags per normalised theme, layered on top of the base set.
_THEME_TAGS: dict[str, list[str]] = {
    "bookstore": ["#bookstore", "#booklover", "#travelbooks", "#photographybook"],
    "coffee_shop": ["#coffeeshop", "#coffeetime", "#cafehopping", "#travelcoffee"],
    "night_market": ["#nightmarket", "#streetfood", "#localfood", "#foodtravel"],
    "luxury_hotel": ["#luxuryhotel", "#hotellife", "#travelinstyle", "#staycation"],
    "airport": ["#airportstyle", "#travelday", "#jetsetter", "#wanderlust"],
    "shopping": ["#shoppingday", "#travelstyle", "#ootd", "#travelfashion"],
    "sightseeing": ["#sightseeing", "#exploring", "#citywalk", "#travelphotography"],
}

_REEL_EXTRA_TAGS = ["#travelreels", "#reels", "#dayinmylife"]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    return _NON_ALNUM.sub("", value.lower())


def _dedupe(tags: list[str]) -> list[str]:
    return list(dict.fromkeys(tags))


class HashtagEngine:
    BASE_TAGS = _BASE_TAGS
    THEME_TAGS = _THEME_TAGS

    def apply(self, plan: DailyContentPlan) -> None:
        theme_key = plan.theme.strip().lower()
        theme_tags = self.THEME_TAGS.get(theme_key, [])

        location_tags = [
            f"#{_slugify(plan.city)}",
            f"#{_slugify(plan.city)}travel",
            f"#{_slugify(plan.country)}",
        ]

        location_tags = [tag for tag in location_tags if len(tag) > 1]

        tags = [
            *self.BASE_TAGS,
            *theme_tags,
            *location_tags,
        ]

        plan.hashtags = _dedupe(tags)

    def feed_hashtags(
        self,
        plan: DailyContentPlan,
        *,
        maximum: int = 12,
    ) -> list[str]:
        return plan.hashtags[:maximum]

    def reel_hashtags(
        self,
        plan: DailyContentPlan,
        *,
        maximum: int = 8,
    ) -> list[str]:
        tags = _dedupe([*plan.hashtags, *_REEL_EXTRA_TAGS])
        return tags[:maximum]

    def threads_hashtags(
        self,
        plan: DailyContentPlan,
        *,
        maximum: int = 3,
    ) -> list[str]:
        # Threads culture favours few or no hashtags — a small,
        # generic-but-relevant subset only.
        return self.BASE_TAGS[:maximum]
