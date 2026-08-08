"""Structural validation for LiveTalkingReelRecord batches (task §17).
Unknown data is always allowed (a `None`/empty field is never an
error); invalid data fails clearly with a specific, actionable
message. Never raises itself from `validate_record()`/`validate_records()`
-- callers decide whether to treat errors as fatal (`require_valid_record()`
does).
"""
from __future__ import annotations

import json
import math
from datetime import datetime

from src.video_intelligence.evidence import VideoPlatform, video_id_for

from .exceptions import LiveAdapterValidationError
from .models import LiveAdapterConfig, LiveTalkingReelRecord


def _check_finite_nonnegative(errors: list[str], *, name: str, value: float, context: str) -> None:
    if math.isnan(value) or math.isinf(value):
        errors.append(f"{name} must be finite: {value!r} ({context})")
    elif value < 0:
        errors.append(f"{name} must not be negative: {value!r} ({context})")


def validate_record(record: LiveTalkingReelRecord) -> list[str]:
    """Returns a list of validation errors (empty = valid). A fully
    sparse record (only `source_url` known) is valid."""
    errors: list[str] = []

    if not record.source_url or not record.source_url.strip():
        errors.append("source_url must not be empty")
    elif not (record.source_url.startswith("http://") or record.source_url.startswith("https://")):
        errors.append(f"source_url does not look like an absolute URL: {record.source_url!r}")

    if not record.reel_id:
        errors.append("reel_id was not computed (unexpected -- record construction should always set it)")
    if not record.record_id:
        errors.append("record_id was not computed (unexpected -- record construction should always set it)")

    if record.duration_seconds is not None:
        _check_finite_nonnegative(errors, name="duration_seconds", value=record.duration_seconds, context="record")

    if record.published_at is not None:
        try:
            datetime.fromisoformat(record.published_at.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"published_at is not a recognizable ISO timestamp: {record.published_at!r}")

    seen_evidence_ids: set[str] = set()
    for item in record.timeline_observations:
        if item.evidence_id in seen_evidence_ids:
            errors.append(f"duplicate TalkingAIEvidence within record: {item.evidence_id!r}")
        seen_evidence_ids.add(item.evidence_id)
        _check_finite_nonnegative(
            errors, name="timestamp_seconds", value=item.timestamp_seconds,
            context=f"evidence_id={item.evidence_id!r}",
        )
        if item.video_id and item.video_id != record.reel_id:
            errors.append(
                f"evidence video_id {item.video_id!r} does not match record reel_id {record.reel_id!r} "
                "(adapter.py rewrites this automatically, but a mismatch here may indicate an authoring mistake)"
            )

    seen_segment_ids: set[str] = set()
    for segment in record.speech_segments:
        if segment.segment_id in seen_segment_ids:
            errors.append(f"duplicate SpeechSegment within record: {segment.segment_id!r}")
        seen_segment_ids.add(segment.segment_id)
        # end_seconds >= start_seconds is already enforced by SpeechSegment's
        # own __post_init__ (raises InvalidTalkingAIEvidenceError) -- by the
        # time we have a SpeechSegment object here, that ordering is
        # already guaranteed, so it is not re-checked.
        for name, value in (("start_seconds", segment.start_seconds), ("end_seconds", segment.end_seconds)):
            _check_finite_nonnegative(errors, name=name, value=value, context=f"segment_id={segment.segment_id!r}")
        if segment.video_id and segment.video_id != record.reel_id:
            errors.append(
                f"segment video_id {segment.video_id!r} does not match record reel_id {record.reel_id!r}"
            )

    try:
        json.dumps(record.metadata)
    except (TypeError, ValueError):
        errors.append("metadata must be JSON-serializable")

    return errors


def validate_records(
    records: list[LiveTalkingReelRecord], *, expected_creator_label: str | None = None
) -> tuple[list[LiveTalkingReelRecord], dict[str, list[str]]]:
    """Batch validation: returns (valid_records, errors_by_reel_id).
    Detects duplicate Reels (same reel_id) across the batch -- the
    first occurrence (by source_url, for determinism) is kept, later
    duplicates are reported as errors, never silently merged. When
    `expected_creator_label` is supplied, flags any record whose
    reel_id does not match what that creator_label + its own
    source_url would produce -- a likely batch-authoring mistake
    (mixing records meant for different creators into one batch)."""
    errors_by_reel_id: dict[str, list[str]] = {}
    seen_reel_ids: set[str] = set()
    valid: list[LiveTalkingReelRecord] = []

    for record in sorted(records, key=lambda r: r.source_url):
        record_errors = validate_record(record)
        if record.reel_id in seen_reel_ids:
            record_errors.append(f"duplicate reel_id in batch: {record.reel_id!r} (source_url={record.source_url!r})")
        if expected_creator_label is not None:
            expected_reel_id = video_id_for(VideoPlatform.INSTAGRAM_REELS, expected_creator_label, record.source_url)
            if expected_reel_id != record.reel_id:
                record_errors.append(
                    f"creator mismatch: reel_id does not match expected_creator_label={expected_creator_label!r}"
                )
        if record_errors:
            errors_by_reel_id[record.reel_id] = record_errors
        else:
            seen_reel_ids.add(record.reel_id)
            valid.append(record)

    return valid, errors_by_reel_id


def require_valid_record(record: LiveTalkingReelRecord) -> None:
    errors = validate_record(record)
    if errors:
        raise LiveAdapterValidationError(f"record for source_url={record.source_url!r} failed validation: {errors}")


def validate_config(config: LiveAdapterConfig) -> list[str]:
    errors: list[str] = []
    if config.source_platform not in VideoPlatform.ALL:
        errors.append(f"source_platform must be one of {VideoPlatform.ALL}, got {config.source_platform!r}")
    return errors
