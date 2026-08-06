"""Save/load CreatorDNA records. JSON is always available (atomic
write via temp-file + Path.replace(), matching every other save_*()
in this codebase); YAML is optional -- PyYAML is detected cleanly at
call time and this module NEVER attempts to install it. On load, the
stored dna_id is recomputed from the loaded fields and compared
against the stored value, so a tampered or corrupted file is rejected
rather than silently trusted.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import CreatorDNA, RelationshipClaim, TraitScore

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - exercised via sys.modules injection in tests
    _yaml = None


class CreatorIntelligenceSerializationError(RuntimeError):
    """Raised for save/load failures, including missing PyYAML and
    dna_id integrity mismatches on load."""


def creator_dna_to_dict(dna: CreatorDNA) -> dict:
    return asdict(dna)


def creator_dna_from_dict(payload: dict) -> CreatorDNA:
    trait_fields = (
        "persona",
        "visual_realism",
        "photography",
        "human_authenticity",
        "relationships",
        "captions",
        "replies",
        "storytelling",
        "reels",
        "branding",
        "posting",
        "engagement",
        "growth",
    )
    kwargs = {
        "subject_label": payload["subject_label"],
        "schema_version": payload.get("schema_version", "1.0"),
        "generated_at": payload.get("generated_at", ""),
        "evidence_index": list(payload.get("evidence_index", [])),
        "relationship_claims": [
            RelationshipClaim(
                description=c["description"],
                basis=c["basis"],
                evidence_ids=list(c.get("evidence_ids", [])),
            )
            for c in payload.get("relationship_claims", [])
        ],
        "overall_confidence": payload.get("overall_confidence", "unknown"),
        "warnings": list(payload.get("warnings", [])),
    }
    for name in trait_fields:
        raw = payload.get(name)
        kwargs[name] = (
            TraitScore(
                trait_name=raw["trait_name"],
                score=raw["score"],
                confidence=raw.get("confidence", "unknown"),
                evidence_ids=list(raw.get("evidence_ids", [])),
                rationale=raw.get("rationale", ""),
            )
            if raw is not None
            else None
        )

    dna = CreatorDNA(**kwargs)

    stored_id = payload.get("dna_id")
    if stored_id is not None and stored_id != dna.dna_id:
        raise CreatorIntelligenceSerializationError(
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


def save_creator_dna(dna: CreatorDNA, path: str | Path) -> Path:
    path = Path(path)
    payload = creator_dna_to_dict(dna)
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_creator_dna(path: str | Path) -> CreatorDNA:
    path = Path(path)
    if not path.exists():
        raise CreatorIntelligenceSerializationError(f"CreatorDNA file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CreatorIntelligenceSerializationError(f"Invalid JSON in {path}: {exc}") from exc
    return creator_dna_from_dict(payload)


def _require_yaml() -> None:
    if _yaml is None:
        raise CreatorIntelligenceSerializationError(
            "PyYAML is not installed; YAML serialization is unavailable. "
            "Install it manually if needed -- this module never installs packages itself."
        )


def save_creator_dna_yaml(dna: CreatorDNA, path: str | Path) -> Path:
    _require_yaml()
    path = Path(path)
    payload = creator_dna_to_dict(dna)
    _atomic_write_text(path, _yaml.safe_dump(payload, sort_keys=True))
    return path


def load_creator_dna_yaml(path: str | Path) -> CreatorDNA:
    _require_yaml()
    path = Path(path)
    if not path.exists():
        raise CreatorIntelligenceSerializationError(f"CreatorDNA file not found: {path}")
    try:
        payload = _yaml.safe_load(path.read_text(encoding="utf-8"))
    except _yaml.YAMLError as exc:
        raise CreatorIntelligenceSerializationError(f"Invalid YAML in {path}: {exc}") from exc
    return creator_dna_from_dict(payload)
