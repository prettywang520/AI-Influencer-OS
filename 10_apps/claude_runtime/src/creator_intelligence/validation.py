"""Validation for Evidence and CreatorDNA records. Mirrors the shape
of video_validator.ValidationResult (passed/failed_checks/warnings)
for consistency with the rest of this codebase's validator modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .confidence import ConfidenceLevel
from .config import CreatorIntelligenceConfig
from .evidence import Evidence
from .models import CreatorDNA, RelationshipBasis
from .scoring import SCORE_MAX, SCORE_MIN

DEFAULT_MAX_SOURCE_DESCRIPTION_CHARS = 500


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""


def validate_evidence(
    evidence: Evidence,
    config: CreatorIntelligenceConfig,
    *,
    max_source_description_chars: int = DEFAULT_MAX_SOURCE_DESCRIPTION_CHARS,
) -> ValidationResult:
    failed: list[str] = []
    warnings: list[str] = []

    if evidence.evidence_type not in config.allowed_evidence_types:
        failed.append(f"evidence_type {evidence.evidence_type!r} is not in the configured allowed set")
    if len(evidence.content_excerpt) > config.caption_excerpt_max_chars:
        failed.append(
            f"content_excerpt length {len(evidence.content_excerpt)} exceeds "
            f"caption_excerpt_max_chars={config.caption_excerpt_max_chars}"
        )
    if len(evidence.source_description) > max_source_description_chars:
        failed.append(
            f"source_description length {len(evidence.source_description)} exceeds "
            f"max_source_description_chars={max_source_description_chars}"
        )
    if not evidence.source_description.strip():
        failed.append("source_description must not be empty")

    passed = not failed
    summary = "PASSED" if passed else f"FAILED ({len(failed)} check(s))"
    return ValidationResult(passed=passed, failed_checks=failed, warnings=warnings, summary=summary)


def validate_creator_dna(dna: CreatorDNA, config: CreatorIntelligenceConfig) -> ValidationResult:
    failed: list[str] = []
    warnings: list[str] = []

    for name in (
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
    ):
        trait = getattr(dna, name)
        if trait is None:
            continue
        if not (SCORE_MIN <= trait.score <= SCORE_MAX):
            failed.append(f"{name}.score {trait.score} out of bounds [{SCORE_MIN}, {SCORE_MAX}]")
        if trait.confidence not in ConfidenceLevel.ORDER:
            failed.append(f"{name}.confidence {trait.confidence!r} is not a recognized ConfidenceLevel")
        if trait.confidence != ConfidenceLevel.UNKNOWN and not trait.evidence_ids:
            failed.append(f"{name} has confidence {trait.confidence!r} but no evidence_ids")
        for evidence_id in trait.evidence_ids:
            if evidence_id not in dna.evidence_index:
                failed.append(f"{name} cites evidence_id {evidence_id!r} not present in evidence_index")

    for claim in dna.relationship_claims:
        if claim.basis not in RelationshipBasis.ALL:
            failed.append(f"relationship_claim basis {claim.basis!r} is not recognized")
        for evidence_id in claim.evidence_ids:
            if evidence_id not in dna.evidence_index:
                failed.append(f"relationship_claim cites evidence_id {evidence_id!r} not present in evidence_index")

    if dna.overall_confidence not in ConfidenceLevel.ORDER:
        failed.append(f"overall_confidence {dna.overall_confidence!r} is not a recognized ConfidenceLevel")

    passed = not failed
    summary = "PASSED" if passed else f"FAILED ({len(failed)} check(s))"
    return ValidationResult(passed=passed, failed_checks=failed, warnings=warnings, summary=summary)
