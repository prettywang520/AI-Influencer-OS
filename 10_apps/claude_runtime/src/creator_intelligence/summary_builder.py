"""Summary Builder -- runs every analyzer over a shared evidence set
and aggregates their outputs into one CreatorDNA record. Also computes
four lighter-weight meta-scores (branding, posting, engagement,
growth) directly via the same evidence-coverage math the analyzers
use, since these four categories have no distinguished
evidence-collection or claim-extraction rules of their own -- unlike
the 9 categories with dedicated analyzer files.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .analyzer_base import AnalyzerContext, analyze_by_tag
from .caption_analyzer import CaptionAnalyzer
from .config import CreatorIntelligenceConfig
from .confidence import min_confidence
from .evidence import Evidence
from .human_authenticity_analyzer import HumanAuthenticityAnalyzer
from .models import CreatorDNA
from .persona_analyzer import PersonaAnalyzer
from .photography_analyzer import PhotographyAnalyzer
from .reels_analyzer import ReelsAnalyzer
from .relationship_analyzer import RelationshipAnalyzer
from .reply_analyzer import ReplyAnalyzer
from .storytelling_analyzer import StorytellingAnalyzer
from .visual_realism_analyzer import VisualRealismAnalyzer

_META_TRAIT_TAGS = {
    "branding": ("branding",),
    "posting": ("posting",),
    "engagement": ("engagement",),
    "growth": ("growth",),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_creator_dna(
    subject_label: str,
    evidence: list[Evidence],
    config: CreatorIntelligenceConfig,
) -> CreatorDNA:
    """Builds a CreatorDNA record from operator-/test-supplied
    evidence only. `subject_label` is a free-text operator label
    (never required to be, and never auto-populated with, a real
    account handle)."""
    context = AnalyzerContext(evidence=evidence, config=config)

    persona_result = PersonaAnalyzer().analyze(context)
    visual_realism_result = VisualRealismAnalyzer().analyze(context)
    photography_result = PhotographyAnalyzer().analyze(context)
    human_authenticity = HumanAuthenticityAnalyzer()
    human_authenticity_result = human_authenticity.analyze(context)
    relationship_claims = human_authenticity.extract_relationship_claims(context)
    relationships_result = RelationshipAnalyzer().analyze(context)
    caption_result = CaptionAnalyzer().analyze(context)
    reply_result = ReplyAnalyzer().analyze(context)
    storytelling_result = StorytellingAnalyzer().analyze(context)
    reels_result = ReelsAnalyzer().analyze(context)

    meta_results = {
        name: analyze_by_tag(
            context,
            trait_name=name,
            tags=tags,
            rationale_with_evidence=f"{name.capitalize()} coverage observed across cited evidence.",
            rationale_without_evidence=f"No {name}-tagged evidence supplied; confidence is unknown.",
        )
        for name, tags in _META_TRAIT_TAGS.items()
    }

    all_results = [
        persona_result,
        visual_realism_result,
        photography_result,
        human_authenticity_result,
        relationships_result,
        caption_result,
        reply_result,
        storytelling_result,
        reels_result,
        *meta_results.values(),
    ]

    warnings: list[str] = []
    evidence_ids: set[str] = set()
    for result in all_results:
        warnings.extend(result.warnings)
        evidence_ids.update(result.trait_score.evidence_ids)
    for claim in relationship_claims:
        evidence_ids.update(claim.evidence_ids)

    overall_confidence = min_confidence([result.trait_score.confidence for result in all_results])

    dna = CreatorDNA(
        subject_label=subject_label,
        schema_version=config.schema_version,
        generated_at=_now_iso(),
        evidence_index=sorted(evidence_ids),
        persona=persona_result.trait_score,
        visual_realism=visual_realism_result.trait_score,
        photography=photography_result.trait_score,
        human_authenticity=human_authenticity_result.trait_score,
        relationships=relationships_result.trait_score,
        relationship_claims=relationship_claims,
        captions=caption_result.trait_score,
        replies=reply_result.trait_score,
        storytelling=storytelling_result.trait_score,
        reels=reels_result.trait_score,
        branding=meta_results["branding"].trait_score,
        posting=meta_results["posting"].trait_score,
        engagement=meta_results["engagement"].trait_score,
        growth=meta_results["growth"].trait_score,
        overall_confidence=overall_confidence,
        warnings=warnings,
    )
    return dna
