"""Save/load VideoDNA and ProductionPattern records. Atomic write via
temp-file + Path.replace(), the same convention used throughout this
codebase. On load, the stored dna_id/pattern_id is recomputed from the
loaded fields and compared against the stored value, so a tampered or
corrupted file is rejected rather than silently trusted (mirrors
creator_intelligence.serialization exactly).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .evidence import VideoIntelligenceError
from .knowledge_base import ProductionPattern
from .models import CTAObservation, StoryBeat, TraitScore
from .production_dna import TRAIT_FIELDS, VideoDNA


class VideoIntelligenceSerializationError(VideoIntelligenceError):
    """Raised for save/load failures, including dna_id/pattern_id
    integrity mismatches on load."""


def video_dna_to_dict(dna: VideoDNA) -> dict:
    return asdict(dna)


def video_dna_from_dict(payload: dict) -> VideoDNA:
    kwargs = {
        "video_id": payload["video_id"],
        "subject_label": payload["subject_label"],
        "schema_version": payload.get("schema_version", "1.0"),
        "generated_at": payload.get("generated_at", ""),
        "evidence_index": list(payload.get("evidence_index", [])),
        "story_beats": [
            StoryBeat(
                beat_type=beat["beat_type"], description=beat["description"],
                evidence_ids=list(beat.get("evidence_ids", [])),
            )
            for beat in payload.get("story_beats", [])
        ],
        "cta_observations": [
            CTAObservation(
                cta_type=obs["cta_type"], timing_seconds=obs.get("timing_seconds"),
                evidence_ids=list(obs.get("evidence_ids", [])),
            )
            for obs in payload.get("cta_observations", [])
        ],
        "overall_confidence": payload.get("overall_confidence", "unknown"),
        "warnings": list(payload.get("warnings", [])),
    }
    for name in TRAIT_FIELDS:
        raw = payload.get(name)
        kwargs[name] = (
            TraitScore(
                trait_name=raw["trait_name"], score=raw["score"], confidence=raw.get("confidence", "unknown"),
                evidence_ids=list(raw.get("evidence_ids", [])), rationale=raw.get("rationale", ""),
            )
            if raw is not None
            else None
        )

    dna = VideoDNA(**kwargs)

    stored_id = payload.get("dna_id")
    if stored_id is not None and stored_id != dna.dna_id:
        raise VideoIntelligenceSerializationError(
            f"dna_id integrity check failed: stored={stored_id!r} recomputed={dna.dna_id!r}"
        )
    return dna


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


def save_video_dna(dna: VideoDNA, path: str | Path) -> Path:
    path = Path(path)
    _atomic_write_text(path, json.dumps(video_dna_to_dict(dna), indent=2, sort_keys=True))
    return path


def load_video_dna(path: str | Path) -> VideoDNA:
    path = Path(path)
    if not path.exists():
        raise VideoIntelligenceSerializationError(f"VideoDNA file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VideoIntelligenceSerializationError(f"Invalid JSON in {path}: {exc}") from exc
    return video_dna_from_dict(payload)


def save_production_patterns(patterns: list[ProductionPattern], path: str | Path) -> Path:
    path = Path(path)
    payload = [pattern.to_dict() for pattern in patterns]
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_production_patterns(path: str | Path) -> list[ProductionPattern]:
    path = Path(path)
    if not path.exists():
        raise VideoIntelligenceSerializationError(f"Production pattern file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VideoIntelligenceSerializationError(f"Invalid JSON in {path}: {exc}") from exc
    return [ProductionPattern.from_dict(item) for item in payload]
