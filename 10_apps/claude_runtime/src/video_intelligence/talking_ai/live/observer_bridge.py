"""Bridges creator_research.instagram.models.ReelRecord -- a stable,
dependency-free public dataclass, imported directly, never
creator_research.instagram.connector/.observer/.navigator -- into a
richer LiveTalkingReelRecord than mapper.py's Evidence-only path can
produce.

The integration finding this file exists for: `connector.collect_reels()`
always flattens `ReelRecord` into `creator_intelligence.evidence.Evidence`
via `evidence_mapper.reel_to_evidence()` before returning it, and that
flattening silently drops `ReelRecord.text_overlays` (`Evidence` has no
field for it -- the same gap Phase 12D.1's own docs already flagged).
A caller who already has a `ReelRecord` -- built by hand today (an
operator who watched the Reel), or from a future direct
`observer.observe_reels()` call this file doesn't care about the
provenance of -- can recover `text_overlays` here, before that lossy
flattening ever happens. This module opens no browser and makes no
network call; it is a pure, offline data transformation.
"""
from __future__ import annotations

from src.creator_research.instagram.models import ReelRecord

from .models import LiveTalkingReelRecord


def live_talking_reel_from_reel_record(
    reel: ReelRecord,
    *,
    creator_label: str = "",
    source_evidence_ids: list[str] | None = None,
) -> LiveTalkingReelRecord:
    """Recovers reel_url/published_at/duration_seconds plus
    `text_overlays` -> `subtitle_observations` (each overlay string
    kept as its own local, traceable entry -- never propagated into
    the generalized knowledge base, see adapter.py). Does not invent
    timeline_observations/speech_segments -- `ReelRecord` itself has no
    per-timestamp fields to recover; those still come from an
    operator's own direct TalkingAIEvidence/SpeechSegment construction."""
    return LiveTalkingReelRecord(
        source_url=reel.reel_url,
        creator_label=creator_label,
        source_evidence_ids=list(source_evidence_ids or []),
        published_at=reel.published_at,
        duration_seconds=reel.duration_seconds,
        subtitle_observations=list(reel.text_overlays),
        metadata={"caption": reel.caption} if reel.caption else {},
    )
