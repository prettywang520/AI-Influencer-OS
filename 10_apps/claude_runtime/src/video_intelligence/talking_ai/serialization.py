"""Save/load TalkingAIProductionDNA and TalkingAIProductionPattern
records. Atomic write via temp-file + Path.replace(), the same
convention used throughout this codebase. On load, the stored
dna_id/pattern_id is recomputed from the loaded fields and compared
against the stored value, so a tampered or corrupted file is rejected
rather than silently trusted (mirrors video_intelligence.serialization
exactly).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .exceptions import SerializationError
from .knowledge_base import TalkingAIProductionPattern
from .models import ConfidenceLevel, TRAIT_FIELDS, TalkingAIMetricResult, TalkingNaturalnessResult
from .production_dna import TalkingAIProductionDNA


def _metric_result_from_dict(raw: dict | None) -> TalkingAIMetricResult | None:
    if raw is None:
        return None
    return TalkingAIMetricResult(
        trait_name=raw["trait_name"],
        metrics=dict(raw.get("metrics", {})),
        score=raw.get("score", 0.0),
        confidence=raw.get("confidence", ConfidenceLevel.UNKNOWN),
        sample_count=raw.get("sample_count", 0),
        evidence_ids=list(raw.get("evidence_ids", [])),
        warnings=list(raw.get("warnings", [])),
        rationale=raw.get("rationale", ""),
    )


def _naturalness_result_from_dict(raw: dict | None) -> TalkingNaturalnessResult | None:
    if raw is None:
        return None
    return TalkingNaturalnessResult(
        dimension_scores=dict(raw.get("dimension_scores", {})),
        dimension_confidence=dict(raw.get("dimension_confidence", {})),
        score=raw.get("score", 0.0),
        confidence=raw.get("confidence", ConfidenceLevel.UNKNOWN),
        sample_count=raw.get("sample_count", 0),
        evidence_ids=list(raw.get("evidence_ids", [])),
        warnings=list(raw.get("warnings", [])),
        rationale=raw.get("rationale", ""),
    )


def talking_ai_dna_to_dict(dna: TalkingAIProductionDNA) -> dict:
    return asdict(dna)


def talking_ai_dna_from_dict(payload: dict) -> TalkingAIProductionDNA:
    kwargs = {
        "video_id": payload["video_id"],
        "subject_label": payload["subject_label"],
        "schema_version": payload.get("schema_version", "1.0"),
        "generated_at": payload.get("generated_at", ""),
        "evidence_index": list(payload.get("evidence_index", [])),
        "naturalness": _naturalness_result_from_dict(payload.get("naturalness")),
        "artifact_patterns": dict(payload.get("artifact_patterns", {})),
        "overall_confidence": payload.get("overall_confidence", ConfidenceLevel.UNKNOWN),
        "warnings": list(payload.get("warnings", [])),
    }
    for name in TRAIT_FIELDS:
        kwargs[name] = _metric_result_from_dict(payload.get(name))

    dna = TalkingAIProductionDNA(**kwargs)

    stored_id = payload.get("dna_id")
    if stored_id is not None and stored_id != dna.dna_id:
        raise SerializationError(
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


def save_talking_ai_dna(dna: TalkingAIProductionDNA, path: str | Path) -> Path:
    path = Path(path)
    _atomic_write_text(path, json.dumps(talking_ai_dna_to_dict(dna), indent=2, sort_keys=True))
    return path


def load_talking_ai_dna(path: str | Path) -> TalkingAIProductionDNA:
    path = Path(path)
    if not path.exists():
        raise SerializationError(f"TalkingAIProductionDNA file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SerializationError(f"Invalid JSON in {path}: {exc}") from exc
    return talking_ai_dna_from_dict(payload)


def save_talking_ai_patterns(patterns: list[TalkingAIProductionPattern], path: str | Path) -> Path:
    path = Path(path)
    payload = [pattern.to_dict() for pattern in patterns]
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_talking_ai_patterns(path: str | Path) -> list[TalkingAIProductionPattern]:
    path = Path(path)
    if not path.exists():
        raise SerializationError(f"Talking AI production pattern file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SerializationError(f"Invalid JSON in {path}: {exc}") from exc
    return [TalkingAIProductionPattern.from_dict(item) for item in payload]
