"""TalkingAIKnowledgeBase -- cross-video, de-identified pattern store
for talking-AI production technique (task §19). Mirrors
video_intelligence.knowledge_base.VideoKnowledgeBase's architecture
exactly (same de-identification guarantees, same corroboration-based
confidence, same idempotent membership-file design), operating over
TalkingAIProductionDNA instead of VideoDNA -- a sibling store, not an
extension, since TalkingAIProductionDNA is not VideoDNA-shaped.
"Never copy creators" is enforced structurally: nothing in
TalkingAIProductionPattern carries a creator_label, username, or
verbatim transcript -- only aggregate counts and short,
controlled-vocabulary-derived descriptions built from counted
categories only, never from an analyzer's free-text rationale.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .analyzer import evidence_for_video, segments_for_video
from .evidence import SpeechSegment, TalkingAIEvidence
from .exceptions import KnowledgeBaseError
from .models import ConfidenceThresholds, TRAIT_FIELDS, compute_confidence_level
from .production_dna import TalkingAIProductionDNA

MAX_EXAMPLE_VIDEO_IDS = 5

# Corroboration-based confidence: each supporting video is itself an
# independent source, so evidence_count == corroboration_count here --
# reuses the exact same weakest-link math every analyzer already uses,
# never a bespoke scoring rule.
DEFAULT_PATTERN_THRESHOLDS = ConfidenceThresholds(
    min_evidence_for_medium=2, min_evidence_for_high=4, min_corroboration_for_verified=4
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(payload: dict[str, Any], *, length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def _pattern_id(category: str, description: str) -> str:
    return _content_hash({"category": category, "description": description})


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        Path(tmp_name).replace(path)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


@dataclass(slots=True)
class TalkingAIProductionPattern:
    """A generalized, de-identified talking-AI production technique
    observed across one or more videos. Never carries a creator_label,
    username, or verbatim transcript. `evidence_coverage` (task's own
    required §19 field) is the mean, across this pattern's supporting
    videos, of how many of the 10 naturalness dimensions each source
    video actually had evidence for -- a pattern backed by videos with
    thin evidence coverage is visibly weaker than one backed by
    fully-evidenced videos, even at the same supporting_video_count."""

    category: str
    description: str
    supporting_video_count: int = 0
    confidence: str = "unknown"
    evidence_coverage: float = 0.0
    example_video_ids: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=_now_iso)
    pattern_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.pattern_id = _pattern_id(self.category, self.description)

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "category": self.category,
            "description": self.description,
            "supporting_video_count": self.supporting_video_count,
            "confidence": self.confidence,
            "evidence_coverage": self.evidence_coverage,
            "example_video_ids": list(self.example_video_ids),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TalkingAIProductionPattern":
        pattern = cls(
            category=payload["category"],
            description=payload["description"],
            supporting_video_count=int(payload.get("supporting_video_count", 0)),
            confidence=payload.get("confidence", "unknown"),
            evidence_coverage=float(payload.get("evidence_coverage", 0.0)),
            example_video_ids=list(payload.get("example_video_ids", [])),
            last_updated=payload.get("last_updated", _now_iso()),
        )
        stored_id = payload.get("pattern_id")
        if stored_id is not None and stored_id != pattern.pattern_id:
            raise KnowledgeBaseError(
                f"pattern_id integrity check failed: stored={stored_id!r} recomputed={pattern.pattern_id!r}"
            )
        return pattern


def _observed_values(items: list, getter: Callable[[Any], str | None]) -> set[str]:
    return {value for value in (getter(item) for item in items) if value is not None}


def _dimension_coverage(dna: TalkingAIProductionDNA) -> float:
    """Fraction of the 10 naturalness dimensions this DNA's own domain
    results actually had evidence for (confidence != unknown)."""
    if dna.naturalness is None:
        return 0.0
    evidenced = sum(1 for confidence in dna.naturalness.dimension_confidence.values() if confidence != "unknown")
    total = len(dna.naturalness.dimension_confidence) or len(TRAIT_FIELDS)
    return evidenced / total if total else 0.0


def _extract_pattern_candidates(
    dna: TalkingAIProductionDNA,
    evidence: list[TalkingAIEvidence],
    speech_segments: list[SpeechSegment],
) -> list[tuple[str, str]]:
    """Pure function: (category, description) candidates for one
    video, derived only from closed-vocabulary evidence/segment fields
    and counted-threshold domain metrics -- never from free-text
    rationale, and never from a specific video's literal timing
    numbers or transcript."""
    candidates: list[tuple[str, str]] = []

    own_evidence = evidence_for_video(evidence, dna.video_id)
    own_segments = segments_for_video(speech_segments, dna.video_id)

    for value in sorted(_observed_values(own_evidence, lambda item: item.gaze_direction)):
        candidates.append(("gaze", f"gaze direction observed: {value}"))
    for value in sorted(_observed_values(own_evidence, lambda item: item.facial_expression)):
        candidates.append(("facial_motion", f"facial expression observed: {value}"))
    for value in sorted(_observed_values(own_evidence, lambda item: item.gesture_type)):
        candidates.append(("gesture_sync", f"gesture type observed: {value}"))
    for value in sorted(_observed_values(own_evidence, lambda item: item.camera_motion)):
        candidates.append(("framing", f"camera motion observed: {value}"))
    for value in sorted(_observed_values(own_evidence, lambda item: item.mouth_shape_category)):
        candidates.append(("lip_sync", f"mouth shape observed: {value}"))
    for value in sorted(_observed_values(own_segments, lambda segment: segment.language)):
        candidates.append(("speech_rhythm", f"language observed: {value}"))
    for value in sorted({tag for item in own_evidence for tag in item.artifact_tags}):
        candidates.append(("artifact", f"artifact tag observed: {value}"))

    if dna.pause_behavior is not None and (dna.pause_behavior.metrics.get("breath_like_pause_count") or 0) > 0:
        candidates.append(("pause_behavior", "breath-like pause behavior observed"))
    if dna.subtitle_sync is not None and (dna.subtitle_sync.metrics.get("subtitle_visible_ratio") or 0.0) >= 0.5:
        candidates.append(("subtitle_sync", "subtitles visible for most of the video"))

    return candidates


class TalkingAIKnowledgeBase:
    def __init__(self, root_directory: str | Path) -> None:
        self.root = Path(root_directory)

    @property
    def patterns_path(self) -> Path:
        return self.root / "patterns.json"

    @property
    def membership_path(self) -> Path:
        return self.root / "membership.json"

    def load_patterns(self) -> list[TalkingAIProductionPattern]:
        if not self.patterns_path.exists():
            return []
        try:
            raw = json.loads(self.patterns_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"Invalid JSON in {self.patterns_path}: {exc}") from exc
        return [TalkingAIProductionPattern.from_dict(item) for item in raw]

    def _load_membership(self) -> dict[str, dict[str, float]]:
        if not self.membership_path.exists():
            return {}
        try:
            return json.loads(self.membership_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"Invalid JSON in {self.membership_path}: {exc}") from exc

    def _save(self, patterns: list[TalkingAIProductionPattern], membership: dict[str, dict[str, float]]) -> None:
        _atomic_write_text(
            self.patterns_path,
            json.dumps([pattern.to_dict() for pattern in patterns], indent=2, sort_keys=True),
        )
        _atomic_write_text(self.membership_path, json.dumps(membership, indent=2, sort_keys=True))

    def list_patterns(self, category: str | None = None) -> list[TalkingAIProductionPattern]:
        patterns = self.load_patterns()
        if category is not None:
            patterns = [pattern for pattern in patterns if pattern.category == category]
        return sorted(patterns, key=lambda pattern: (-pattern.supporting_video_count, pattern.pattern_id))

    def ingest_talking_ai_dna(
        self,
        dna: TalkingAIProductionDNA,
        *,
        evidence: list[TalkingAIEvidence],
        speech_segments: list[SpeechSegment],
    ) -> list[TalkingAIProductionPattern]:
        """Merges one video's structured patterns into the running,
        de-identified pattern library. Idempotent: re-ingesting the
        same video_id for a pattern it already contributed to never
        double-counts it (tracked via membership.json, which is never
        capped, unlike the persisted example_video_ids list)."""
        candidates = _extract_pattern_candidates(dna, evidence, speech_segments)
        coverage = _dimension_coverage(dna)

        patterns_by_id = {pattern.pattern_id: pattern for pattern in self.load_patterns()}
        membership = self._load_membership()
        touched_ids: list[str] = []

        for category, description in candidates:
            pattern_id = _pattern_id(category, description)
            members = dict(membership.get(pattern_id, {}))
            members[dna.video_id] = coverage
            membership[pattern_id] = members

            supporting_count = len(members)
            confidence = compute_confidence_level(supporting_count, supporting_count, DEFAULT_PATTERN_THRESHOLDS)
            patterns_by_id[pattern_id] = TalkingAIProductionPattern(
                category=category,
                description=description,
                supporting_video_count=supporting_count,
                confidence=confidence,
                evidence_coverage=mean(members.values()),
                example_video_ids=sorted(members)[:MAX_EXAMPLE_VIDEO_IDS],
            )
            touched_ids.append(pattern_id)

        self._save(list(patterns_by_id.values()), membership)
        return [patterns_by_id[pattern_id] for pattern_id in touched_ids]
