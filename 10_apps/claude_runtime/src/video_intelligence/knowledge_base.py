"""VideoKnowledgeBase -- cross-video, de-identified pattern store.
Ingests many VideoDNA records (from many videos, possibly many
creators) and extracts ProductionPattern records: short, generalized
descriptions plus aggregate corroboration statistics. "Never copy
creators" is enforced structurally: nothing in ProductionPattern
carries a `creator_label` or unbounded excerpt -- only aggregate
counts and short, controlled-vocabulary-derived descriptions,
traceable back to source videos only via a capped `example_video_ids`
list. Patterns are extracted only from genuinely structured sources
(story_beats, cta_observations, and the 5 yaml-driven controlled
vocabularies -- hook/pacing/speech/camera/subtitles) -- never by
parsing an analyzer's free-text rationale, which could otherwise risk
surfacing verbatim per-video wording as a "pattern."
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyzer import evidence_for_tags, evidence_for_video, observed_vocabulary
from .camera import CameraConfig, TAGS as CAMERA_TAGS, load_camera_config
from .evidence import VideoEvidence, VideoIntelligenceError
from .hook import HookConfig, TAGS as HOOK_TAGS, load_hook_config
from .models import ConfidenceThresholds, compute_confidence_level
from .pacing import PacingConfig, TAGS as PACING_TAGS, load_pacing_config
from .production_dna import VideoDNA
from .speech import SpeechConfig, TAGS as SPEECH_TAGS, load_speech_config
from .subtitles import SubtitlesConfig, TAGS as SUBTITLES_TAGS, load_subtitles_config

MAX_EXAMPLE_VIDEO_IDS = 5

# Corroboration-based confidence: each supporting video is itself an
# independent source, so evidence_count == corroboration_count here --
# reuses the exact same weakest-link math every analyzer already uses
# (models.compute_confidence_level()), never a bespoke scoring rule.
DEFAULT_PATTERN_THRESHOLDS = ConfidenceThresholds(
    min_evidence_for_medium=2, min_evidence_for_high=4, min_corroboration_for_verified=4
)


class KnowledgeBaseError(VideoIntelligenceError):
    """Raised for knowledge-base read/write/consistency failures."""


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
class ProductionPattern:
    """A generalized, de-identified production technique observed
    across one or more videos. Never carries a creator_label or
    unbounded excerpt."""

    category: str
    description: str
    supporting_video_count: int = 0
    confidence: str = "unknown"
    example_video_ids: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)
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
            "example_video_ids": list(self.example_video_ids),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ProductionPattern:
        pattern = cls(
            category=payload["category"],
            description=payload["description"],
            supporting_video_count=int(payload.get("supporting_video_count", 0)),
            confidence=payload.get("confidence", "unknown"),
            example_video_ids=list(payload.get("example_video_ids", [])),
            updated_at=payload.get("updated_at", _now_iso()),
        )
        stored_id = payload.get("pattern_id")
        if stored_id is not None and stored_id != pattern.pattern_id:
            raise KnowledgeBaseError(
                f"pattern_id integrity check failed: stored={stored_id!r} recomputed={pattern.pattern_id!r}"
            )
        return pattern


def _extract_pattern_candidates(
    dna: VideoDNA,
    evidence: list[VideoEvidence],
    *,
    hook_config: HookConfig,
    pacing_config: PacingConfig,
    speech_config: SpeechConfig,
    camera_config: CameraConfig,
    subtitles_config: SubtitlesConfig,
) -> list[tuple[str, str]]:
    """Pure function: (category, description) candidates for one
    video, derived only from structured sources (never free-text
    rationale)."""
    candidates: list[tuple[str, str]] = []

    for beat in dna.story_beats:
        candidates.append(("storytelling", f"story arc includes a '{beat.beat_type}' beat"))
    for observation in dna.cta_observations:
        candidates.append(("cta", f"uses a '{observation.cta_type}' call-to-action"))

    own_video_evidence = evidence_for_video(evidence, dna.video_id)

    hook_evidence = evidence_for_tags(own_video_evidence, HOOK_TAGS)
    for hook_type in observed_vocabulary(hook_evidence, hook_config.hook_types):
        candidates.append(("hook", f"hook type: {hook_type}"))

    pacing_evidence = evidence_for_tags(own_video_evidence, PACING_TAGS)
    for tempo in observed_vocabulary(pacing_evidence, pacing_config.tempo_types):
        candidates.append(("pacing", f"tempo: {tempo}"))
    for rhythm in observed_vocabulary(pacing_evidence, pacing_config.rhythm_types):
        candidates.append(("pacing", f"rhythm: {rhythm}"))

    speech_evidence = evidence_for_tags(own_video_evidence, SPEECH_TAGS)
    for language in observed_vocabulary(speech_evidence, speech_config.languages):
        candidates.append(("speech", f"language: {language}"))
    for speed in observed_vocabulary(speech_evidence, speech_config.speed_types):
        candidates.append(("speech", f"speed: {speed}"))
    for energy in observed_vocabulary(speech_evidence, speech_config.energy_types):
        candidates.append(("speech", f"energy: {energy}"))

    camera_evidence = evidence_for_tags(own_video_evidence, CAMERA_TAGS)
    for distance in observed_vocabulary(camera_evidence, camera_config.distance_types):
        candidates.append(("camera", f"distance: {distance}"))
    for equipment in observed_vocabulary(camera_evidence, camera_config.equipment_types):
        candidates.append(("camera", f"equipment: {equipment}"))

    subtitles_evidence = evidence_for_tags(own_video_evidence, SUBTITLES_TAGS)
    for style in observed_vocabulary(subtitles_evidence, subtitles_config.style_types):
        candidates.append(("subtitles", f"style: {style}"))

    return candidates


class VideoKnowledgeBase:
    def __init__(self, root_directory: str | Path) -> None:
        self.root = Path(root_directory)

    @property
    def patterns_path(self) -> Path:
        return self.root / "patterns.json"

    @property
    def membership_path(self) -> Path:
        return self.root / "membership.json"

    def load_patterns(self) -> list[ProductionPattern]:
        if not self.patterns_path.exists():
            return []
        try:
            raw = json.loads(self.patterns_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"Invalid JSON in {self.patterns_path}: {exc}") from exc
        return [ProductionPattern.from_dict(item) for item in raw]

    def _load_membership(self) -> dict[str, list[str]]:
        if not self.membership_path.exists():
            return {}
        try:
            return json.loads(self.membership_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"Invalid JSON in {self.membership_path}: {exc}") from exc

    def _save(self, patterns: list[ProductionPattern], membership: dict[str, list[str]]) -> None:
        _atomic_write_text(
            self.patterns_path,
            json.dumps([pattern.to_dict() for pattern in patterns], indent=2, sort_keys=True),
        )
        _atomic_write_text(self.membership_path, json.dumps(membership, indent=2, sort_keys=True))

    def list_patterns(self, category: str | None = None) -> list[ProductionPattern]:
        patterns = self.load_patterns()
        if category is not None:
            patterns = [pattern for pattern in patterns if pattern.category == category]
        return sorted(patterns, key=lambda pattern: (-pattern.supporting_video_count, pattern.pattern_id))

    def ingest_video_dna(
        self,
        dna: VideoDNA,
        evidence: list[VideoEvidence],
        *,
        hook_config: HookConfig | None = None,
        pacing_config: PacingConfig | None = None,
        speech_config: SpeechConfig | None = None,
        camera_config: CameraConfig | None = None,
        subtitles_config: SubtitlesConfig | None = None,
    ) -> list[ProductionPattern]:
        """Merges one video's structured patterns into the running,
        de-identified pattern library. Idempotent: re-ingesting the
        same video_id for a pattern it already contributed to never
        double-counts it (tracked via membership.json, which is never
        capped, unlike the persisted example_video_ids list)."""
        candidates = _extract_pattern_candidates(
            dna, evidence,
            hook_config=hook_config or load_hook_config(),
            pacing_config=pacing_config or load_pacing_config(),
            speech_config=speech_config or load_speech_config(),
            camera_config=camera_config or load_camera_config(),
            subtitles_config=subtitles_config or load_subtitles_config(),
        )

        patterns_by_id = {pattern.pattern_id: pattern for pattern in self.load_patterns()}
        membership = self._load_membership()
        touched_ids: list[str] = []

        for category, description in candidates:
            pattern_id = _pattern_id(category, description)
            member_ids = set(membership.get(pattern_id, []))
            if dna.video_id not in member_ids:
                member_ids.add(dna.video_id)
                membership[pattern_id] = sorted(member_ids)

            supporting_count = len(member_ids)
            confidence = compute_confidence_level(supporting_count, supporting_count, DEFAULT_PATTERN_THRESHOLDS)
            patterns_by_id[pattern_id] = ProductionPattern(
                category=category,
                description=description,
                supporting_video_count=supporting_count,
                confidence=confidence,
                example_video_ids=sorted(member_ids)[:MAX_EXAMPLE_VIDEO_IDS],
            )
            touched_ids.append(pattern_id)

        self._save(list(patterns_by_id.values()), membership)
        return [patterns_by_id[pattern_id] for pattern_id in touched_ids]
