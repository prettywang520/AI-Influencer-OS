"""LiveTalkingReelRecord -> (TalkingAIEvidence, SpeechSegment) (task §3).
Because TalkingAIEvidence/SpeechSegment already are Phase 12D.2's
target schema, this module does not reinterpret or reclassify content
-- every item in `record.timeline_observations`/`record.speech_segments`
is expected to already be a fully-formed, native object (built by an
operator, observer_bridge.py, or a future automated annotator; timing
quality -- exact/approximate/ordinal -- is decided upstream by
timing_bridge.py, not guessed here). This layer's job is narrow and
mechanical: rewrite `video_id` to `record.reel_id` on a fresh copy of
each item (mirrors Phase 12D.1's own
`annotation.video_id = ""  # rewritten automatically` precedent
exactly) and surface, as warnings, how many items carry timing-bridge
placeholder markers -- never invent a field, never fabricate a
timestamp.

`camera_observations`/`subtitle_observations`/`artifact_observations`
(the record's coarse, reel-level context lists) are deliberately **not**
merged into individual TalkingAIEvidence items here -- doing so would
attribute a whole-Reel characterization to a specific timestamp that
never actually reported it, exactly the fabrication "unknown remains
unknown" forbids. They flow into sampler.py/report.py only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..evidence import SpeechSegment, TalkingAIEvidence
from .models import LiveTalkingReelRecord
from .timing_bridge import APPROXIMATE_TIMING_NOTE, ORDINAL_PLACEHOLDER_NOTE


@dataclass(slots=True)
class LiveAdaptedEvidence:
    evidence: list[TalkingAIEvidence] = field(default_factory=list)
    speech_segments: list[SpeechSegment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ordinal_timing_count: int = 0
    approximate_timing_count: int = 0


def _rewrite_video_id(item: TalkingAIEvidence, reel_id: str) -> TalkingAIEvidence:
    if item.video_id == reel_id:
        return item
    return replace(item, video_id=reel_id)


def _rewrite_segment_video_id(segment: SpeechSegment, reel_id: str) -> SpeechSegment:
    if segment.video_id == reel_id:
        return segment
    return replace(segment, video_id=reel_id)


def live_talking_evidence_from_record(record: LiveTalkingReelRecord) -> LiveAdaptedEvidence:
    """The one bridging step every mapper.py/observer_bridge.py/hand-
    built path shares: normalizes identity onto `record.reel_id` and
    reports (never silently absorbs) how much of this record's timing
    is placeholder rather than observed."""
    evidence = [_rewrite_video_id(item, record.reel_id) for item in record.timeline_observations]
    speech_segments = [_rewrite_segment_video_id(segment, record.reel_id) for segment in record.speech_segments]

    ordinal_count = sum(1 for item in evidence if ORDINAL_PLACEHOLDER_NOTE in item.notes)
    approximate_count = sum(
        1 for item in evidence if APPROXIMATE_TIMING_NOTE in item.notes and ORDINAL_PLACEHOLDER_NOTE not in item.notes
    )

    warnings: list[str] = []
    if ordinal_count:
        warnings.append(f"{ordinal_count} evidence item(s) used ordinal placeholder timing (not observed seconds)")
    if approximate_count:
        warnings.append(f"{approximate_count} evidence item(s) used approximate (not exact) timing")
    if not evidence and not speech_segments:
        warnings.append("no timeline_observations or speech_segments supplied for this record")

    return LiveAdaptedEvidence(
        evidence=evidence, speech_segments=speech_segments, warnings=warnings,
        ordinal_timing_count=ordinal_count, approximate_timing_count=approximate_count,
    )
