"""Structural validation for InstagramReelEvidencePacket batches
(task §24). Unknown data is always allowed (a `None` field is never
an error); invalid data fails clearly with a specific, actionable
message. Source platform is validated implicitly by construction --
every packet's reel_id is always computed against
VideoPlatform.INSTAGRAM_REELS (see models.py), a closed vocabulary
that would already reject anything else at construction time.
"""
from __future__ import annotations

import json
from datetime import datetime

from .exceptions import AdapterValidationError
from .models import InstagramReelEvidencePacket


def validate_packet(packet: InstagramReelEvidencePacket) -> list[str]:
    """Returns a list of validation errors (empty = valid). Never
    raises itself -- callers decide whether to treat errors as fatal."""
    errors: list[str] = []

    if not packet.reel_url or not packet.reel_url.strip():
        errors.append("reel_url must not be empty")
    elif not (packet.reel_url.startswith("http://") or packet.reel_url.startswith("https://")):
        errors.append(f"reel_url does not look like an absolute URL: {packet.reel_url!r}")

    if not packet.reel_id:
        errors.append("reel_id was not computed (unexpected -- packet construction should always set it)")

    if packet.duration_seconds is not None and packet.duration_seconds < 0:
        errors.append(f"duration_seconds must not be negative: {packet.duration_seconds!r}")
    for name, value in (("views", packet.views), ("likes", packet.likes), ("comments", packet.comments)):
        if value is not None and value < 0:
            errors.append(f"{name} must not be negative: {value!r}")

    if packet.published_at is not None:
        try:
            datetime.fromisoformat(packet.published_at.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"published_at is not a recognizable ISO timestamp: {packet.published_at!r}")

    # Note: annotation.video_id is NOT checked against packet.reel_id
    # here -- mapper.packet_to_video_evidence() always rewrites every
    # annotation's video_id to packet.reel_id regardless of what a
    # hand-built annotation was constructed with, so callers are free
    # to build annotations with a placeholder/empty video_id (the
    # natural, low-friction way to build a packet by hand -- see
    # docs/video_intelligence/instagram_reels_adapter.md).
    evidence_ids_seen: set[str] = set()
    for annotation in packet.annotations:
        if annotation.evidence_id in evidence_ids_seen:
            errors.append(f"duplicate VideoEvidence within packet: {annotation.evidence_id!r}")
        evidence_ids_seen.add(annotation.evidence_id)
        if annotation.timestamp_seconds is not None and annotation.timestamp_seconds < 0:
            errors.append(f"annotation timestamp_seconds must not be negative: {annotation.timestamp_seconds!r}")

    try:
        json.dumps(packet.metadata)
    except (TypeError, ValueError):
        errors.append("metadata must be JSON-serializable")

    return errors


def validate_packets(
    packets: list[InstagramReelEvidencePacket],
) -> tuple[list[InstagramReelEvidencePacket], dict[str, list[str]]]:
    """Batch validation: returns (valid_packets, errors_by_reel_id).
    Detects duplicate Reels (same reel_id) across the batch -- the
    first occurrence (by reel_url, for determinism) is kept, later
    duplicates are reported as errors, never silently merged."""
    errors_by_reel_id: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    valid: list[InstagramReelEvidencePacket] = []

    for packet in sorted(packets, key=lambda p: p.reel_url):
        packet_errors = validate_packet(packet)
        if packet.reel_id in seen_ids:
            packet_errors.append(f"duplicate reel_id in batch: {packet.reel_id!r} (reel_url={packet.reel_url!r})")
        if packet_errors:
            errors_by_reel_id[packet.reel_id] = packet_errors
        else:
            seen_ids.add(packet.reel_id)
            valid.append(packet)

    return valid, errors_by_reel_id


def require_valid_packet(packet: InstagramReelEvidencePacket) -> None:
    errors = validate_packet(packet)
    if errors:
        raise AdapterValidationError(f"packet for reel_url={packet.reel_url!r} failed validation: {errors}")
