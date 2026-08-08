"""Converts already-collected creator_research Instagram evidence into
LiveTalkingReelRecord. Only creator_intelligence.evidence.Evidence (a
stable, public type) is imported -- never creator_research.instagram
itself, so this module can never transitively import Playwright.

Parses the same fixed "<kind> | key=value | ..." convention every
creator_research/instagram/evidence_mapper.py function already uses
(the identical convention Phase 12D.1's own
video_intelligence/instagram/mapper.py parses) -- reimplemented locally
rather than importing that package's own leading-underscore, module-
private helpers across a package boundary. `text_overlays` is
unavailable via this path (same honestly-documented gap Phase 12D.1
already found); observer_bridge.py recovers it from a richer,
still-offline `ReelRecord` input instead.
"""
from __future__ import annotations

from src.creator_intelligence.evidence import Evidence

from .models import LiveTalkingReelRecord


def _parse_source_description(text: str) -> dict[str, str]:
    """Never raises on an unexpected shape -- a segment without "="
    is ignored, an empty/malformed string yields {}. A key absent from
    the source text is simply absent from the result (unknown remains
    unknown, never guessed)."""
    if not text:
        return {}
    segments = [segment.strip() for segment in text.split("|")]
    parsed: dict[str, str] = {}
    for segment in segments[1:]:  # segment 0 is the "<kind>" prefix, not a key=value pair
        if "=" not in segment:
            continue
        key, _, value = segment.partition("=")
        parsed[key.strip()] = value.strip()
    return parsed


def _kind_of(text: str) -> str:
    if not text:
        return ""
    return text.split("|", 1)[0].strip()


def _none_if_placeholder(value: str | None) -> str | None:
    if value is None or value in ("", "None"):
        return None
    return value


def _to_float(value: str | None) -> float | None:
    value = _none_if_placeholder(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def live_talking_reels_from_creator_research(
    evidence: list[Evidence], *, creator_label: str = ""
) -> list[LiveTalkingReelRecord]:
    """Groups a flat creator_research Evidence list into one
    LiveTalkingReelRecord per distinct Reel URL -- identity/metadata
    only (url, published_at, duration_seconds, caption). Never
    populates timeline_observations/speech_segments from this path --
    a bare Reel Evidence item carries no per-timestamp talking-behavior
    fields to recover; those come from an operator's own direct
    TalkingAIEvidence/SpeechSegment construction, or from
    observer_bridge.py's richer ReelRecord input."""
    records: list[LiveTalkingReelRecord] = []
    for item in evidence:
        if _kind_of(item.source_description) != "instagram reel":
            continue
        parsed = _parse_source_description(item.source_description)
        reel_url = parsed.get("url", "")
        if not reel_url:
            continue  # cannot identify which reel this is about -- skip, never guess a URL

        records.append(
            LiveTalkingReelRecord(
                source_url=reel_url,
                creator_label=creator_label,
                source_evidence_ids=[item.evidence_id],
                published_at=_none_if_placeholder(parsed.get("published_at")),
                duration_seconds=_to_float(parsed.get("duration_seconds")),
                metadata={"caption": item.content_excerpt} if item.content_excerpt else {},
            )
        )
    return records
