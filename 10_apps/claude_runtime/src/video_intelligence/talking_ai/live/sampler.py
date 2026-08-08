"""Deterministic Reel sampling for talking Reels (task §15) -- mirrors
video_intelligence/instagram/sampler.py's sample_reels() shape exactly.
`sample_live_talking_reels()` is a pure function: the same input list,
in any order, always produces the same output list. Determinism comes
from content-based ordering (sort by source_url/published_at/derived
bucket values) and a fixed bucket iteration order -- never from
input/arrival order, never from a seeded PRNG.

Buckets needing data LiveTalkingReelRecord's own required fields don't
carry (visible engagement, walking/indoor/outdoor context) read an
optional key from `record.metadata`/`record.camera_observations` --
a record missing that signal simply never joins that bucket, never
fabricated (see the approved plan's Exploration Summary).
"""
from __future__ import annotations

from .models import FramingDistanceTag, LiveAdapterConfig, LiveTalkingReelRecord


class SamplingBucket:
    """Task's own §15 bucket list."""

    RECENT_TALKING = "recent_talking"
    OLDER_TALKING = "older_talking"
    HIGH_VISIBLE_ENGAGEMENT = "high_visible_engagement"
    LOW_VISIBLE_ENGAGEMENT = "low_visible_engagement"
    SINGLE_SHOT_TALKING = "single_shot_talking"
    MULTI_CUT_TALKING = "multi_cut_talking"
    DIRECT_CAMERA = "direct_camera"
    WALKING_TALKING = "walking_talking"
    INDOOR_TALKING = "indoor_talking"
    OUTDOOR_TALKING = "outdoor_talking"
    HEAVY_SUBTITLE = "heavy_subtitle"
    LIGHT_SUBTITLE = "light_subtitle"
    STRONG_GESTURE = "strong_gesture"
    MINIMAL_GESTURE = "minimal_gesture"

    ALL = (
        RECENT_TALKING, OLDER_TALKING, HIGH_VISIBLE_ENGAGEMENT, LOW_VISIBLE_ENGAGEMENT,
        SINGLE_SHOT_TALKING, MULTI_CUT_TALKING, DIRECT_CAMERA, WALKING_TALKING,
        INDOOR_TALKING, OUTDOOR_TALKING, HEAVY_SUBTITLE, LIGHT_SUBTITLE,
        STRONG_GESTURE, MINIMAL_GESTURE,
    )


def _dedupe(records: list[LiveTalkingReelRecord]) -> list[LiveTalkingReelRecord]:
    by_id: dict[str, LiveTalkingReelRecord] = {}
    for record in sorted(records, key=lambda r: r.source_url):
        by_id.setdefault(record.reel_id, record)
    return list(by_id.values())


def _engagement_value(record: LiveTalkingReelRecord) -> float | None:
    value = record.metadata.get("engagement_visible")
    return float(value) if isinstance(value, (int, float)) else None


def _subtitle_density_value(record: LiveTalkingReelRecord) -> float | None:
    if record.completeness.get("subtitle") != 1.0:
        return None
    change_events = sum(1 for item in record.timeline_observations if item.subtitle_change)
    return float(len(record.subtitle_observations) + change_events)


def _gesture_density_value(record: LiveTalkingReelRecord) -> float | None:
    if record.completeness.get("gesture") != 1.0:
        return None
    return float(
        sum(
            1 for item in record.timeline_observations
            if item.left_hand_motion is not None or item.right_hand_motion is not None or item.gesture_type is not None
        )
    )


def _shot_boundary_count(record: LiveTalkingReelRecord) -> int | None:
    if record.completeness.get("camera") != 1.0:
        return None
    return sum(1 for item in record.timeline_observations if item.shot_boundary)


def _median_split(
    records: list[LiveTalkingReelRecord],
    *,
    value_fn,
    high_bucket: str,
    low_bucket: str,
    buckets: dict[str, list[LiveTalkingReelRecord]],
) -> None:
    pairs = sorted(
        ((record, value_fn(record)) for record in records),
        key=lambda pair: (pair[1] is None, pair[1], pair[0].source_url),
    )
    computable = [pair for pair in pairs if pair[1] is not None]
    if not computable:
        return
    median = computable[len(computable) // 2][1]
    for record, value in pairs:
        if value is None:
            continue
        buckets[high_bucket if value >= median else low_bucket].append(record)


def _assign_buckets(records: list[LiveTalkingReelRecord]) -> dict[str, list[LiveTalkingReelRecord]]:
    buckets: dict[str, list[LiveTalkingReelRecord]] = {name: [] for name in SamplingBucket.ALL}

    dated = sorted((r for r in records if r.published_at), key=lambda r: (r.published_at, r.source_url))
    older_half_ids = {r.reel_id for r in dated[: len(dated) // 2]}
    for record in records:
        if record.published_at:
            bucket = SamplingBucket.OLDER_TALKING if record.reel_id in older_half_ids else SamplingBucket.RECENT_TALKING
            buckets[bucket].append(record)

    _median_split(
        records, value_fn=_engagement_value,
        high_bucket=SamplingBucket.HIGH_VISIBLE_ENGAGEMENT, low_bucket=SamplingBucket.LOW_VISIBLE_ENGAGEMENT,
        buckets=buckets,
    )
    _median_split(
        records, value_fn=_subtitle_density_value,
        high_bucket=SamplingBucket.HEAVY_SUBTITLE, low_bucket=SamplingBucket.LIGHT_SUBTITLE,
        buckets=buckets,
    )
    _median_split(
        records, value_fn=_gesture_density_value,
        high_bucket=SamplingBucket.STRONG_GESTURE, low_bucket=SamplingBucket.MINIMAL_GESTURE,
        buckets=buckets,
    )

    for record in sorted(records, key=lambda r: r.source_url):
        shot_boundaries = _shot_boundary_count(record)
        if shot_boundaries is not None:
            bucket = SamplingBucket.SINGLE_SHOT_TALKING if shot_boundaries == 0 else SamplingBucket.MULTI_CUT_TALKING
            buckets[bucket].append(record)

        if FramingDistanceTag.DIRECT_TO_CAMERA in record.camera_observations:
            buckets[SamplingBucket.DIRECT_CAMERA].append(record)
        if FramingDistanceTag.WALKING in record.camera_observations:
            buckets[SamplingBucket.WALKING_TALKING].append(record)
        if FramingDistanceTag.INDOOR in record.camera_observations:
            buckets[SamplingBucket.INDOOR_TALKING].append(record)
        if FramingDistanceTag.OUTDOOR in record.camera_observations:
            buckets[SamplingBucket.OUTDOOR_TALKING].append(record)

    return buckets


def sample_live_talking_reels(
    records: list[LiveTalkingReelRecord], config: LiveAdapterConfig
) -> list[LiveTalkingReelRecord]:
    """Deterministic, duplicate-excluded, bucketed sample of up to
    `config.recommended_talking_reels` records (or all available
    records if fewer exist). Round-robins across the fixed
    SamplingBucket.ALL order so no single bucket dominates the sample."""
    deduped = _dedupe(records)
    target_count = min(len(deduped), config.recommended_talking_reels)
    if target_count <= 0:
        return []

    buckets = _assign_buckets(deduped)
    selected: list[LiveTalkingReelRecord] = []
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
                record = bucket_items[index]
                selected.append(record)
                selected_ids.add(record.reel_id)
                cursor[bucket_name] = index + 1
                progressed = True
            else:
                cursor[bucket_name] = index

    # Records that matched no content bucket at all (e.g. no
    # published_at, no engagement metadata, no camera/subtitle/gesture
    # completeness) are still eligible via a final fallback pass in
    # source_url order, so a genuinely sparse batch still gets sampled
    # up to target_count rather than under-filling.
    if len(selected) < target_count:
        for record in sorted(deduped, key=lambda r: r.source_url):
            if len(selected) >= target_count:
                break
            if record.reel_id not in selected_ids:
                selected.append(record)
                selected_ids.add(record.reel_id)

    return selected
