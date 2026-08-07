"""Deduplication by strongest available identity: canonical URL ->
observed stable media ID -> content-hash of structured fields as a
last resort. Never deduplicates unrelated items solely because
captions match -- caption text is deliberately excluded from the
content-hash fallback payload by every caller in this package.
"""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str | None) -> str | None:
    """Lowercases scheme/host, strips query string/fragment, and any
    trailing slash. Returns None for empty input."""
    if not url or not url.strip():
        return None
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    normalized = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))
    return normalized or None


def _content_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def identity_for(
    *,
    url: str | None = None,
    stable_id: str | None = None,
    fallback_payload: dict | None = None,
) -> str:
    """Returns a stable identity string, preferring (in order):
    canonical URL, an observed stable id, then a content-hash of
    `fallback_payload`. Raises ValueError if none of the three are
    usable -- callers must supply at least one."""
    normalized = normalize_url(url)
    if normalized:
        return f"url:{normalized}"
    if stable_id and stable_id.strip():
        return f"id:{stable_id.strip()}"
    if fallback_payload:
        return f"hash:{_content_hash(fallback_payload)}"
    raise ValueError("identity_for requires at least one of url, stable_id, or fallback_payload")


class Deduplicator:
    """Tracks identities already seen (within one run, or seeded from
    a checkpoint's processed_source_ids on resume) and filters out
    repeats -- e.g. the same grid post discovered again on a later
    scroll round."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.duplicates_skipped: int = 0

    def seed(self, identities: list[str]) -> None:
        self._seen.update(identities)

    def is_duplicate(self, identity: str) -> bool:
        return identity in self._seen

    def mark_seen(self, identity: str) -> None:
        self._seen.add(identity)

    def filter_new(self, items_with_identity: list[tuple[str, object]]) -> list[object]:
        """Returns only the items whose identity hasn't been seen yet,
        marking each as seen and counting skipped duplicates."""
        result = []
        for identity, item in items_with_identity:
            if self.is_duplicate(identity):
                self.duplicates_skipped += 1
                continue
            self.mark_seen(identity)
            result.append(item)
        return result

    def seen_identities(self) -> list[str]:
        return sorted(self._seen)
