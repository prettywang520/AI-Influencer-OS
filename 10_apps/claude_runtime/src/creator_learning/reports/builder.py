"""build_*_report() -- pure functions over an already-built
CreatorDNA + evidence + LearningHistoryEntry list + StyleEvolutionRecord.
No new analysis, no new confidence math; hashtag/mention frequency
counting is display-layer aggregation only, not a new analyzer.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from src.creator_intelligence.evidence import Evidence
from src.creator_intelligence.models import CreatorDNA, TraitScore

from ..learning_history import LearningHistoryEntry
from ..learning_targets import compute_learning_target_coverage
from ..style_evolution import TRAIT_FIELDS, StyleEvolutionRecord
from .models import CaptionReport, LearningReport, RelationshipReport, ReplyReport, StyleReport, VisualReport

_HASHTAG_RE = re.compile(r"#(\w+)")
_MENTION_RE = re.compile(r"@(\w+)")

_EXCERPT_LIMIT = 5


def _evidence_by_id(evidence: list[Evidence]) -> dict[str, Evidence]:
    return {item.evidence_id: item for item in evidence}


def _excerpts_for(trait: TraitScore | None, evidence_by_id: dict[str, Evidence], limit: int = _EXCERPT_LIMIT) -> list[str]:
    if trait is None:
        return []
    excerpts: list[str] = []
    for evidence_id in trait.evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item and item.content_excerpt:
            excerpts.append(item.content_excerpt)
        if len(excerpts) >= limit:
            break
    return excerpts


def build_learning_report(
    *,
    creator_id: str,
    session_id: str,
    dna: CreatorDNA,
    evidence: list[Evidence],
    history: list[LearningHistoryEntry],
) -> LearningReport:
    coverage = compute_learning_target_coverage(evidence)
    gaps = {trait: cov.missing_tags for trait, cov in coverage.items()}
    trend = [(entry.session_id, entry.overall_confidence) for entry in history]
    return LearningReport(
        creator_id=creator_id,
        session_id=session_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        evidence_count=len(evidence),
        overall_confidence=dna.overall_confidence,
        session_count=len(history),
        confidence_trend=trend,
        gaps=gaps,
    )


def build_visual_report(*, creator_id: str, session_id: str, dna: CreatorDNA, evidence: list[Evidence]) -> VisualReport:
    evidence_by_id = _evidence_by_id(evidence)
    coverage = compute_learning_target_coverage(evidence)
    visual_coverage = coverage.get("visual_realism")
    excerpts = _excerpts_for(dna.visual_realism, evidence_by_id) + _excerpts_for(dna.photography, evidence_by_id)
    return VisualReport(
        creator_id=creator_id,
        session_id=session_id,
        visual_realism_score=dna.visual_realism.score if dna.visual_realism else None,
        visual_realism_confidence=dna.visual_realism.confidence if dna.visual_realism else None,
        visual_realism_rationale=dna.visual_realism.rationale if dna.visual_realism else "",
        photography_score=dna.photography.score if dna.photography else None,
        photography_confidence=dna.photography.confidence if dna.photography else None,
        photography_rationale=dna.photography.rationale if dna.photography else "",
        covered_tags=visual_coverage.covered_tags if visual_coverage else (),
        missing_tags=visual_coverage.missing_tags if visual_coverage else (),
        cited_excerpts=excerpts[:_EXCERPT_LIMIT],
    )


def build_relationship_report(*, creator_id: str, session_id: str, dna: CreatorDNA) -> RelationshipReport:
    claims = [
        {"description": claim.description, "basis": claim.basis, "evidence_ids": list(claim.evidence_ids)}
        for claim in dna.relationship_claims
    ]
    return RelationshipReport(
        creator_id=creator_id,
        session_id=session_id,
        relationships_score=dna.relationships.score if dna.relationships else None,
        relationships_confidence=dna.relationships.confidence if dna.relationships else None,
        claims=claims,
    )


def build_reply_report(*, creator_id: str, session_id: str, dna: CreatorDNA, evidence: list[Evidence]) -> ReplyReport:
    evidence_by_id = _evidence_by_id(evidence)
    coverage = compute_learning_target_coverage(evidence)
    reply_coverage = coverage.get("replies")
    return ReplyReport(
        creator_id=creator_id,
        session_id=session_id,
        replies_score=dna.replies.score if dna.replies else None,
        replies_confidence=dna.replies.confidence if dna.replies else None,
        rationale=dna.replies.rationale if dna.replies else "",
        language_mix_covered=reply_coverage.covered_tags if reply_coverage else (),
        language_mix_missing=reply_coverage.missing_tags if reply_coverage else (),
        cited_excerpts=_excerpts_for(dna.replies, evidence_by_id),
    )


def build_caption_report(*, creator_id: str, session_id: str, dna: CreatorDNA, evidence: list[Evidence]) -> CaptionReport:
    evidence_by_id = _evidence_by_id(evidence)
    hashtags: dict[str, int] = {}
    mentions: dict[str, int] = {}
    if dna.captions:
        for evidence_id in dna.captions.evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if not item:
                continue
            for tag in _HASHTAG_RE.findall(item.content_excerpt):
                hashtags[tag.lower()] = hashtags.get(tag.lower(), 0) + 1
            for mention in _MENTION_RE.findall(item.content_excerpt):
                mentions[mention.lower()] = mentions.get(mention.lower(), 0) + 1
    return CaptionReport(
        creator_id=creator_id,
        session_id=session_id,
        captions_score=dna.captions.score if dna.captions else None,
        captions_confidence=dna.captions.confidence if dna.captions else None,
        rationale=dna.captions.rationale if dna.captions else "",
        hashtag_frequency=hashtags,
        mention_frequency=mentions,
        cited_excerpts=_excerpts_for(dna.captions, evidence_by_id),
    )


def build_style_report(*, creator_id: str, session_id: str, evolution: StyleEvolutionRecord) -> StyleReport:
    lines: list[str] = []
    for trait_name in TRAIT_FIELDS:
        delta = evolution.trait_deltas.get(trait_name)
        if delta is None:
            continue
        if evolution.is_baseline:
            if delta.current_score is not None:
                lines.append(
                    f"{trait_name}: new baseline score {delta.current_score:.2f} ({delta.current_confidence})"
                )
            continue
        if delta.score_delta is not None and abs(delta.score_delta) > 1e-9:
            direction = "increased" if delta.score_delta > 0 else "decreased"
            lines.append(
                f"{trait_name}: score {direction} by {abs(delta.score_delta):.2f} "
                f"({delta.previous_score:.2f} -> {delta.current_score:.2f})"
            )
        if delta.confidence_changed:
            lines.append(f"{trait_name}: confidence changed {delta.previous_confidence} -> {delta.current_confidence}")

    return StyleReport(
        creator_id=creator_id,
        session_id=session_id,
        is_baseline=evolution.is_baseline,
        new_relationship_claim_count=len(evolution.new_relationship_claims),
        changed_relationship_claim_count=len(evolution.changed_relationship_claims),
        new_warning_count=len(evolution.new_warnings),
        trait_summary=lines,
    )


def build_all_reports(
    *,
    creator_id: str,
    session_id: str,
    dna: CreatorDNA,
    evidence: list[Evidence],
    history: list[LearningHistoryEntry],
    evolution: StyleEvolutionRecord,
) -> dict[str, object]:
    """Builds all six report dataclasses. Does not touch the raw
    CreatorDNA export -- callers that also want the full DNA payload
    use creator_intelligence.serialization directly (see engine.py)."""
    return {
        "learning": build_learning_report(creator_id=creator_id, session_id=session_id, dna=dna, evidence=evidence, history=history),
        "visual": build_visual_report(creator_id=creator_id, session_id=session_id, dna=dna, evidence=evidence),
        "relationship": build_relationship_report(creator_id=creator_id, session_id=session_id, dna=dna),
        "reply": build_reply_report(creator_id=creator_id, session_id=session_id, dna=dna, evidence=evidence),
        "caption": build_caption_report(creator_id=creator_id, session_id=session_id, dna=dna, evidence=evidence),
        "style": build_style_report(creator_id=creator_id, session_id=session_id, evolution=evolution),
    }
