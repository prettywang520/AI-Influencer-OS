"""Parses a creator profile URL into (platform, username, profile_url,
creator_id). No network access -- pure string parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from src.creator_intelligence.intake import creator_id as compute_creator_id

from .exceptions import InvalidCreatorUrlError

_KNOWN_PLATFORM_HOSTS = {
    "instagram.com": "instagram",
    "www.instagram.com": "instagram",
}


@dataclass(slots=True, frozen=True)
class ParsedCreatorUrl:
    platform: str
    username: str
    profile_url: str
    creator_id: str


def parse_creator_url(profile_url: str) -> ParsedCreatorUrl:
    if not profile_url or not profile_url.strip():
        raise InvalidCreatorUrlError("profile_url must be a non-empty string")

    parsed = urlparse(profile_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise InvalidCreatorUrlError(f"profile_url is not a valid absolute URL: {profile_url!r}")

    platform = _KNOWN_PLATFORM_HOSTS.get(parsed.netloc.lower())
    if platform is None:
        raise InvalidCreatorUrlError(f"profile_url host is not a recognized creator platform: {parsed.netloc!r}")

    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        raise InvalidCreatorUrlError(f"profile_url has no username path segment: {profile_url!r}")

    username = segments[0]
    return ParsedCreatorUrl(
        platform=platform,
        username=username,
        profile_url=profile_url.strip(),
        creator_id=compute_creator_id(platform, username),
    )
