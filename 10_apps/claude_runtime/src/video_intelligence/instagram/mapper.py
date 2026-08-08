"""Converts Instagram Reel evidence into video_intelligence's
VideoEvidence schema. Two entry points:

- reel_evidence_from_creator_research(evidence): parses the fixed
  "<kind> | key=value | ..." convention every
  creator_research/instagram/evidence_mapper.py function already
  uses, and links same-URL caption/creator_reply/relationship
  evidence to the Reel it belongs to. Only
  creator_intelligence.evidence.Evidence (a stable, public type) is
  imported -- never creator_research.instagram itself, so this module
  can never transitively import a browser.

- packet_to_video_evidence(packet): the final normalization into the
  schema every video_intelligence analyzer already consumes.

The adapter never classifies or interprets caption/annotation text --
see docs/video_intelligence/instagram_reels_adapter.md.
"""
from __future__ import annotations

from src.creator_intelligence.evidence import Evidence
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType

from .models import TRAIT_TAGS, InstagramReelEvidencePacket

_ALL_DOMAIN_TAGS = {tag for tags in TRAIT_TAGS.values() for tag in tags}


def _parse_source_description(text: str) -> dict[str, str]:
    """Parses the fixed "<kind> | key=value | key=value | ..."
    convention every creator_research/instagram/evidence_mapper.py
    function uses. Never raises on an unexpected shape -- a segment
    without "=" is ignored, an empty/malformed string yields {}.
    A key absent from the source text is simply absent from the
    result (unknown remains unknown, never guessed)."""
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


def _to_int(value: str | None) -> int | None:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else None


def reel_evidence_from_creator_research(
    evidence: list[Evidence], *, creator_label: str = ""
) -> list[InstagramReelEvidencePacket]:
    """Groups a flat creator_research Evidence list into one
    InstagramReelEvidencePacket per distinct Reel URL. `creator_label`
    is an optional, operator-chosen free-text label carried onto every
    resulting packet -- never inferred from the evidence itself (the
    connector's own Evidence objects never carry a creator-identity
    field to infer from)."""
    reel_items: list[tuple[Evidence, dict[str, str]]] = []
    linkable_items: list[tuple[Evidence, dict[str, str]]] = []

    for item in evidence:
        parsed = _parse_source_description(item.source_description)
        kind = _kind_of(item.source_description)
        if kind == "instagram reel":
            reel_items.append((item, parsed))
        elif "url" in parsed or "post" in parsed:
            linkable_items.append((item, parsed))

    packets: list[InstagramReelEvidencePacket] = []
    for item, parsed in reel_items:
        reel_url = parsed.get("url", "")
        if not reel_url:
            continue  # cannot identify which reel this is about -- skip, never guess a URL

        packet = InstagramReelEvidencePacket(
            reel_url=reel_url,
            published_at=_none_if_placeholder(parsed.get("published_at")),
            duration_seconds=_to_float(parsed.get("duration_seconds")),
            views=_to_int(parsed.get("views")),
            likes=_to_int(parsed.get("likes")),
            comments=_to_int(parsed.get("comments")),
            caption=item.content_excerpt or None,
            creator_label=creator_label,
            source_evidence_ids=[item.evidence_id],
        )
        for linked_item, linked_parsed in linkable_items:
            reference = linked_parsed.get("post") or linked_parsed.get("url")
            if reference != reel_url:
                continue
            packet.source_evidence_ids.append(linked_item.evidence_id)
            # Only fold into `annotations` (i.e. hand to the 13
            # analyzers) if it already carries a domain tag one of
            # them scopes by -- today's connector never sets one; this
            # is forward-compatible, not dead code (see
            # docs/video_intelligence/instagram_reels_adapter.md).
            if _ALL_DOMAIN_TAGS.intersection(linked_item.tags):
                packet.annotations.append(
                    VideoEvidence(
                        video_id=packet.reel_id,
                        evidence_type=linked_item.evidence_type,
                        source_description=linked_item.source_description,
                        content_excerpt=linked_item.content_excerpt,
                        collected_by="instagram_reels_adapter",
                        tags=list(linked_item.tags),
                    )
                )
        packets.append(packet)
    return packets


def packet_to_video_evidence(packet: InstagramReelEvidencePacket) -> list[VideoEvidence]:
    """Final normalization: one structural-summary VideoEvidence item
    (whenever any structural fact is known) plus one VideoEvidence per
    already-domain-tagged annotation. The summary item is only tagged
    "pacing" when duration_seconds is actually known -- its presence
    is never gated on any single field, but its tag membership (which
    controls analyzer scoping) always reflects only what's genuinely
    known, never guessed."""
    items: list[VideoEvidence] = []

    has_any_structural_fact = any(
        [
            packet.duration_seconds is not None, packet.views is not None, packet.likes is not None,
            packet.comments is not None, packet.published_at is not None, packet.caption,
        ]
    )
    if has_any_structural_fact:
        tags = ["instagram_reels"]
        if packet.duration_seconds is not None:
            tags.append("pacing")
        items.append(
            VideoEvidence(
                video_id=packet.reel_id,
                evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
                source_description=(
                    f"instagram reel | url={packet.reel_url} | published_at={packet.published_at} "
                    f"| views={packet.views} | likes={packet.likes} | comments={packet.comments}"
                ),
                content_excerpt=packet.caption or "",
                collected_by="instagram_reels_adapter",
                tags=tags,
            )
        )

    for annotation in packet.annotations:
        items.append(
            VideoEvidence(
                video_id=packet.reel_id,
                evidence_type=annotation.evidence_type,
                source_description=annotation.source_description,
                content_excerpt=annotation.content_excerpt,
                timestamp_seconds=annotation.timestamp_seconds,
                collected_by=annotation.collected_by,
                tags=list(annotation.tags),
            )
        )
    return items
