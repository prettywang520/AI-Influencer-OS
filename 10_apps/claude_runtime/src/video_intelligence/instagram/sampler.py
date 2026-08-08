"""Deterministic Reel sampling -- avoids learning only the most
recent posts (task §16). `sample_reels()` is a pure function: the same
input list, in any order, always produces the same output list.
Determinism comes from content-based ordering (sort by reel_url,
published_at, engagement ratio) and a fixed bucket iteration order --
never from input/arrival order, never from a seeded PRNG.
"""
from __future__ import annotations

from .models import InstagramReelEvidencePacket, InstagramReelsConfig


class SamplingBucket:
    """Task's own §16 bucket list."""

    RECENT = "recent"
    OLDER = "older"
    HIGH_VISIBLE_ENGAGEMENT = "high_visible_engagement"
    LOW_VISIBLE_ENGAGEMENT = "low_visible_engagement"
    TALKING = "talking"
    NON_TALKING = "non_talking"
    TRAVEL = "travel"
    DAILY_LIFE = "daily_life"
    BRAND_COLLABORATION = "brand_collaboration"
    SELFIE = "selfie"
    FRIEND_SOCIAL = "friend_social"
    VISUALLY_CINEMATIC = "visually_cinematic"
    ORDINARY_GROUNDED = "ordinary_grounded"

    ALL = (
        RECENT, OLDER, HIGH_VISIBLE_ENGAGEMENT, LOW_VISIBLE_ENGAGEMENT, TALKING, NON_TALKING,
        TRAVEL, DAILY_LIFE, BRAND_COLLABORATION, SELFIE, FRIEND_SOCIAL, VISUALLY_CINEMATIC,
        ORDINARY_GROUNDED,
    )


# Recognized annotation tags (as they'd appear on a packet's own
# VideoEvidence-shaped annotations) that indicate membership in a
# content-based bucket. A recognition list only -- membership is never
# invented if none of these tags is present.
_CONTENT_BUCKET_TAGS: dict[str, tuple[str, ...]] = {
    SamplingBucket.TRAVEL: ("travel",),
    SamplingBucket.DAILY_LIFE: ("daily_life", "home", "coffee"),
    SamplingBucket.BRAND_COLLABORATION: ("brand_collaboration", "collaboration_label", "sponsored"),
    SamplingBucket.SELFIE: ("selfie",),
    SamplingBucket.FRIEND_SOCIAL: ("friend_shot", "friends"),
    SamplingBucket.VISUALLY_CINEMATIC: ("cinematic", "tracking", "gimbal"),
}
_TALKING_TAGS = ("speech", "lipsync")


def _packet_tags(packet: InstagramReelEvidencePacket) -> set[str]:
    return {tag for annotation in packet.annotations for tag in annotation.tags}


def _dedupe(packets: list[InstagramReelEvidencePacket]) -> list[InstagramReelEvidencePacket]:
    by_id: dict[str, InstagramReelEvidencePacket] = {}
    for packet in sorted(packets, key=lambda p: p.reel_url):
        by_id.setdefault(packet.reel_id, packet)
    return list(by_id.values())


def _assign_buckets(
    packets: list[InstagramReelEvidencePacket],
) -> dict[str, list[InstagramReelEvidencePacket]]:
    buckets: dict[str, list[InstagramReelEvidencePacket]] = {name: [] for name in SamplingBucket.ALL}

    dated = sorted((p for p in packets if p.published_at), key=lambda p: (p.published_at, p.reel_url))
    older_half_ids = {p.reel_id for p in dated[: len(dated) // 2]}

    engagement_ratios = sorted(
        ((p, p.visible_engagement_ratio()) for p in packets), key=lambda pair: (pair[1] is None, pair[1], pair[0].reel_url)
    )
    computable = [pair for pair in engagement_ratios if pair[1] is not None]
    median_ratio = computable[len(computable) // 2][1] if computable else None

    for packet in sorted(packets, key=lambda p: p.reel_url):
        tags = _packet_tags(packet)

        if packet.published_at:
            buckets[SamplingBucket.OLDER if packet.reel_id in older_half_ids else SamplingBucket.RECENT].append(packet)

        ratio = packet.visible_engagement_ratio()
        if ratio is not None and median_ratio is not None:
            bucket = SamplingBucket.HIGH_VISIBLE_ENGAGEMENT if ratio >= median_ratio else SamplingBucket.LOW_VISIBLE_ENGAGEMENT
            buckets[bucket].append(packet)

        is_talking = bool(tags.intersection(_TALKING_TAGS))
        buckets[SamplingBucket.TALKING if is_talking else SamplingBucket.NON_TALKING].append(packet)

        matched_content_bucket = False
        for bucket_name, marker_tags in _CONTENT_BUCKET_TAGS.items():
            if tags.intersection(marker_tags):
                buckets[bucket_name].append(packet)
                matched_content_bucket = True
        if not matched_content_bucket:
            buckets[SamplingBucket.ORDINARY_GROUNDED].append(packet)

    return buckets


def sample_reels(
    packets: list[InstagramReelEvidencePacket], config: InstagramReelsConfig
) -> list[InstagramReelEvidencePacket]:
    """Deterministic, duplicate-excluded, bucketed sample of up to
    `config.recommended_reels` packets (or all available packets if
    fewer exist). Round-robins across the fixed SamplingBucket.ALL
    order so no single bucket (e.g. "recent") dominates the sample."""
    deduped = _dedupe(packets)
    target_count = min(len(deduped), config.recommended_reels)
    if target_count <= 0:
        return []

    buckets = _assign_buckets(deduped)
    selected: list[InstagramReelEvidencePacket] = []
    selected_ids: set[str] = set()
    cursor = {name: 0 for name in SamplingBucket.ALL}

    progressed = True
    while len(selected) < target_count and progressed:
        progressed = False
        for bucket_name in SamplingBucket.ALL:
            if len(selected) >= target_count:
                break
            bucket_items = buckets[bucket_name]
            index = cursor[bucket_name]
            while index < len(bucket_items) and bucket_items[index].reel_id in selected_ids:
                index += 1
            if index < len(bucket_items):
                packet = bucket_items[index]
                selected.append(packet)
                selected_ids.add(packet.reel_id)
                cursor[bucket_name] = index + 1
                progressed = True
            else:
                cursor[bucket_name] = index

    return selected
