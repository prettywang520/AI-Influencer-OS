"""talking_ai_live_learning_report.json/.md (task §21). Pure functions
-- no I/O, no analysis, just formatting/aggregation over an
already-run batch of LiveTalkingLearningRecord outcomes. Never
reproduces a verbatim transcript/caption: only reads from
TalkingAIProductionDNA's own structured, already-de-identified fields
(unchanged from Phase 12D.2) plus each outcome's own local
reel_id/source_url (kept for traceability -- task's own §19 exception,
matching video_intelligence.instagram's InstagramReelLearningRecord
precedent).
"""
from __future__ import annotations

from statistics import mean

from ..models import ConfidenceLevel
from .models import COMPLETENESS_DOMAINS, LiveBatchLearningReport, LiveTalkingLearningRecord

# A record whose overall_confidence never rose above "unknown" produced
# no analyzable signal at all -- flagged separately from every other
# outcome so a batch report can distinguish "genuinely insufficient
# evidence" from "analyzed, just with low confidence."
INSUFFICIENT_EVIDENCE_CONFIDENCE = ConfidenceLevel.UNKNOWN


def build_live_learning_report(
    outcomes: list[LiveTalkingLearningRecord], *, total_reels_discovered: int, reels_sampled: int
) -> LiveBatchLearningReport:
    evidence_completeness: dict[str, float] = {}
    if outcomes:
        evidence_completeness = {
            domain: mean(outcome.completeness.get(domain, 0.0) for outcome in outcomes)
            for domain in COMPLETENESS_DOMAINS
        }

    production_dna_ids = sorted({outcome.dna.dna_id for outcome in outcomes if outcome.dna is not None})
    knowledge_patterns_touched = len({pattern_id for outcome in outcomes for pattern_id in outcome.patterns_touched})
    warnings = [warning for outcome in outcomes for warning in outcome.warnings]
    insufficient_evidence_reel_ids = sorted(
        outcome.reel_id for outcome in outcomes if outcome.overall_confidence == INSUFFICIENT_EVIDENCE_CONFIDENCE
    )

    return LiveBatchLearningReport(
        total_reels_discovered=total_reels_discovered,
        reels_sampled=reels_sampled,
        reels_analyzed=len(outcomes),
        evidence_completeness=evidence_completeness,
        production_dna_ids=production_dna_ids,
        knowledge_patterns_touched=knowledge_patterns_touched,
        warnings=warnings,
        insufficient_evidence_reel_ids=insufficient_evidence_reel_ids,
    )


def render_live_learning_report_markdown(report: LiveBatchLearningReport) -> str:
    lines = [
        "# Talking AI Live Learning Report",
        "",
        f"- Reels discovered: {report.total_reels_discovered}",
        f"- Reels sampled: {report.reels_sampled}",
        f"- Reels analyzed: {report.reels_analyzed}",
        f"- Knowledge patterns touched: {report.knowledge_patterns_touched}",
        f"- Generated at: {report.generated_at}",
        "",
        "## Evidence completeness (average across analyzed Reels)",
    ]
    if report.evidence_completeness:
        for domain in COMPLETENESS_DOMAINS:
            value = report.evidence_completeness.get(domain)
            if value is not None:
                lines.append(f"- {domain}: {value:.2f}")
    else:
        lines.append("- (no Reels analyzed)")

    lines.append("")
    lines.append("## Production DNA IDs")
    if report.production_dna_ids:
        lines.extend(f"- `{dna_id}`" for dna_id in report.production_dna_ids)
    else:
        lines.append("- (none produced)")

    lines.append("")
    lines.append("## Insufficient evidence")
    if report.insufficient_evidence_reel_ids:
        lines.extend(f"- {reel_id}" for reel_id in report.insufficient_evidence_reel_ids)
    else:
        lines.append("- (none -- every analyzed Reel produced at least low-confidence signal)")

    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report.warnings)

    return "\n".join(lines) + "\n"
