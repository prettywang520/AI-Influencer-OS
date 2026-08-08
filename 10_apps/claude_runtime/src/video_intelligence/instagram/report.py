"""Batch learning report (task §25): build_batch_report()/
render_batch_report_markdown(). Pure functions -- no I/O, no
analysis, just aggregation/formatting over already-built
InstagramReelLearningRecord objects. Never includes verbatim
copyrighted scripts/captions -- only ids, counts, and coverage
statistics.
"""
from __future__ import annotations

from .models import TRAIT_FIELDS, BatchLearningReport, InstagramReelLearningRecord


def build_batch_report(
    records: list[InstagramReelLearningRecord],
    *,
    skipped: dict[str, list[str]] | None = None,
    knowledge_patterns_added: int = 0,
) -> BatchLearningReport:
    skipped = skipped or {}

    evidence_completeness: dict[str, float] = {}
    analyzer_coverage: dict[str, int] = {name: 0 for name in TRAIT_FIELDS}
    missing_evidence: dict[str, int] = {name: 0 for name in TRAIT_FIELDS}

    for trait_name in TRAIT_FIELDS:
        scores = [record.completeness.get(trait_name, 0.0) for record in records]
        evidence_completeness[trait_name] = sum(scores) / len(scores) if scores else 0.0

    for record in records:
        for trait_name in TRAIT_FIELDS:
            if record.completeness.get(trait_name, 0.0) <= 0.0:
                missing_evidence[trait_name] += 1
            trait = getattr(record.dna, trait_name, None) if record.dna else None
            if trait is not None and trait.confidence != "unknown":
                analyzer_coverage[trait_name] += 1

    video_dna_ids = sorted({record.dna.dna_id for record in records if record.dna is not None})
    warnings = [warning for record in records for warning in record.warnings]
    warnings.extend(f"reel skipped ({reel_id}): {'; '.join(errors)}" for reel_id, errors in skipped.items())

    return BatchLearningReport(
        total_reels=len(records) + len(skipped),
        reels_analyzed=len(records),
        reels_skipped=len(skipped),
        evidence_completeness=evidence_completeness,
        analyzer_coverage=analyzer_coverage,
        video_dna_ids=video_dna_ids,
        knowledge_patterns_added=knowledge_patterns_added,
        warnings=warnings,
        missing_evidence=missing_evidence,
    )


def render_batch_report_markdown(report: BatchLearningReport) -> str:
    lines = [
        "# Instagram Reels Learning Report",
        "",
        f"- Total reels: {report.total_reels}",
        f"- Reels analyzed: {report.reels_analyzed}",
        f"- Reels skipped: {report.reels_skipped}",
        f"- Knowledge patterns added: {report.knowledge_patterns_added}",
        f"- Generated at: {report.generated_at}",
        "",
        "## Evidence completeness (coverage, not quality)",
    ]
    for trait_name in TRAIT_FIELDS:
        completeness = report.evidence_completeness.get(trait_name, 0.0)
        coverage = report.analyzer_coverage.get(trait_name, 0)
        lines.append(f"- **{trait_name}**: {completeness:.0%} of reels have evidence ({coverage} evidenced)")

    lines.append("")
    lines.append("## Missing evidence (reel count per domain with zero evidence)")
    for trait_name in TRAIT_FIELDS:
        missing = report.missing_evidence.get(trait_name, 0)
        if missing:
            lines.append(f"- {trait_name}: {missing} reel(s)")
    if not any(report.missing_evidence.values()):
        lines.append("- (no domain is fully missing across the batch)")

    lines.append("")
    lines.append(f"## VideoDNA ids ({len(report.video_dna_ids)})")
    if report.video_dna_ids:
        lines.extend(f"- `{dna_id}`" for dna_id in report.video_dna_ids)
    else:
        lines.append("- (none)")

    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report.warnings)

    return "\n".join(lines) + "\n"
