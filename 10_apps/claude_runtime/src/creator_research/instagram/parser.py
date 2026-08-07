"""Pure text-parsing functions -- zero Playwright/browser import,
deterministic, and tolerant of missing or localized input. Every
function keeps an unparseable/absent value as `None` (or an empty
list) rather than raising or guessing. `observer.py` is the only
caller; these functions never touch a live page themselves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_COUNT_PATTERN = re.compile(r"([\d,]+(?:\.\d+)?)\s*([KMB]?)", re.IGNORECASE)
_HASHTAG_PATTERN = re.compile(r"#(\w+)", re.UNICODE)
_MENTION_PATTERN = re.compile(r"@([A-Za-z0-9_.]+)")
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002702-\U000027B0"
    "]+"
)

_SUFFIX_MULTIPLIER = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def parse_count(raw: str | None) -> int | None:
    """Parses a visible count string ("1,234", "12.3K", "4.5M posts")
    into an int. Returns None for missing/unparseable input -- never
    raises, never guesses a value."""
    if not raw or not raw.strip():
        return None
    match = _COUNT_PATTERN.search(raw.strip())
    if not match:
        return None
    number_text, suffix = match.group(1), match.group(2).upper()
    try:
        number = float(number_text.replace(",", ""))
    except ValueError:
        return None
    multiplier = _SUFFIX_MULTIPLIER.get(suffix, None)
    if multiplier is None:
        return None
    return int(number * multiplier)


def parse_hashtags(text: str | None) -> list[str]:
    """Extracts #hashtags, Unicode-aware (Traditional Chinese/Japanese
    hashtags included). Order of first appearance, deduplicated."""
    if not text:
        return []
    seen: list[str] = []
    for match in _HASHTAG_PATTERN.finditer(text):
        tag = match.group(1)
        if tag not in seen:
            seen.append(tag)
    return seen


def extract_emoji(text: str | None) -> list[str]:
    """Extracts emoji characters/sequences from text, preserving order
    of first appearance. Never strips or alters the original text --
    this is additive metadata only."""
    if not text:
        return []
    matches: list[str] = []
    for match in _EMOJI_PATTERN.finditer(text):
        for char in match.group(0):
            if char not in matches:
                matches.append(char)
    return matches


def parse_mentions(text: str | None) -> list[str]:
    """Extracts @mentions (Instagram username charset: letters,
    digits, period, underscore). Order of first appearance, deduplicated."""
    if not text:
        return []
    seen: list[str] = []
    for match in _MENTION_PATTERN.finditer(text):
        mention = match.group(1)
        if mention not in seen:
            seen.append(mention)
    return seen


@dataclass(slots=True)
class ParsedCaption:
    text: str
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)


def parse_caption(raw: str | None) -> ParsedCaption:
    """Preserves the caption text exactly as observed -- never
    rewritten, translated, or summarized here (that stays a later,
    explicitly-approved analysis step, not acquisition)."""
    text = raw if raw is not None else ""
    return ParsedCaption(text=text, hashtags=parse_hashtags(text), mentions=parse_mentions(text))


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def parse_profile_text(raw: dict) -> dict:
    """raw keys (all optional, str|None): username, display_name, bio,
    post_count, follower_count, following_count, category,
    external_link, verified (any non-empty string means the verified
    badge element was found)."""
    return {
        "username": _clean(raw.get("username")) or "",
        "display_name": _clean(raw.get("display_name")),
        "bio": _clean(raw.get("bio")),
        "post_count": parse_count(raw.get("post_count")),
        "follower_count": parse_count(raw.get("follower_count")),
        "following_count": parse_count(raw.get("following_count")),
        "category": _clean(raw.get("category")),
        "external_link": _clean(raw.get("external_link")),
        "verified": bool(raw.get("verified")) if raw.get("verified") is not None else None,
    }


def parse_comment_thread(raw_items: list[dict]) -> list[dict]:
    """raw_items: list of {text, username, timestamp, parent_id}.
    Cleans each entry; preserves order and parent/child linkage as
    given -- does not restructure into a tree and does not decide
    creator-vs-audience (that identity check belongs to observer.py,
    which compares against the research job's own username)."""
    cleaned = []
    for item in raw_items:
        cleaned.append(
            {
                "text": _clean(item.get("text")) or "",
                "username": _clean(item.get("username")),
                "timestamp": _clean(item.get("timestamp")),
                "parent_id": _clean(item.get("parent_id")),
            }
        )
    return cleaned


def parse_post_metadata(raw: dict) -> dict:
    """raw keys: caption, like_count, comment_count, timestamp,
    carousel (bool), location, tagged_accounts (list[str]),
    collaboration_labels (list[str])."""
    return {
        "caption": parse_caption(raw.get("caption")),
        "like_count": parse_count(raw.get("like_count")),
        "comment_count": parse_count(raw.get("comment_count")),
        "timestamp": _clean(raw.get("timestamp")),
        "carousel": bool(raw.get("carousel", False)),
        "location": _clean(raw.get("location")),
        "tagged_accounts": [a for a in (raw.get("tagged_accounts") or []) if a],
        "collaboration_labels": [c for c in (raw.get("collaboration_labels") or []) if c],
    }


def parse_reel_metadata(raw: dict) -> dict:
    """raw keys: caption, views, likes, comments, duration,
    text_overlays (list[str])."""
    return {
        "caption": parse_caption(raw.get("caption")),
        "views": parse_count(raw.get("views")),
        "likes": parse_count(raw.get("likes")),
        "comments": parse_count(raw.get("comments")),
        "duration": _parse_duration(raw.get("duration")),
        "text_overlays": [t for t in (raw.get("text_overlays") or []) if t],
    }


def _parse_duration(raw: str | None) -> float | None:
    """Parses "0:34" or "34s" into seconds. None if unparseable."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if ":" in text:
        parts = text.split(":")
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            return None
        seconds = 0.0
        for number in numbers:
            seconds = seconds * 60 + number
        return seconds
    match = re.match(r"([\d.]+)\s*s", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_highlight_metadata(raw: dict) -> dict:
    """raw keys: title, item_index, content_type, caption."""
    return {
        "title": _clean(raw.get("title")) or "",
        "item_index": int(raw.get("item_index", 0) or 0),
        "content_type": _clean(raw.get("content_type")),
        "caption": _clean(raw.get("caption")),
    }
