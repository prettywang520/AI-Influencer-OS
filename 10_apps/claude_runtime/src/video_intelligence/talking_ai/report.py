"""Human-readable reports: one per analyzed talking-AI video
(build_talking_ai_report()/render_talking_ai_report_markdown()) and
one over the cross-video pattern library
(build_talking_ai_pattern_report()/render_talking_ai_pattern_report_markdown()).
Pure functions -- no I/O, no analysis, just formatting over an
already-built TalkingAIProductionDNA / already-ingested
TalkingAIProductionPattern list. Never reproduces a verbatim
transcript: reports only ever read from the structured trait/
naturalness fields on TalkingAIProductionDNA, which carry no
transcript text at all (see task §20).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge_base import TalkingAIProductionPattern
from .models import TRAIT_FIELDS
from .production_dna import TalkingAIProductionDNA


@dataclass(slots=True)
class TalkingAIReport:
    video_id: str
    subject_label: str
    dna_id: str
    overall_confidence: str
    trait_summary: dict[str, dict] = field(default_factory=dict)
    naturalness_summary: dict | None = None
    artifact_patterns: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def build_talking_ai_report(dna: TalkingAIProductionDNA) -> TalkingAIReport:
    trait_summary = {}
    for name in TRAIT_FIELDS:
        trait = getattr(dna, name)
        if trait is not None:
            trait_summary[name] = {
                "score": trait.score,
                "confidence": trait.confidence,
                "sample_count": trait.sample_count,
                "rationale": trait.rationale,
            }

    naturalness_summary = None
    if dna.naturalness is not None:
        naturalness_summary = {
            "dimension_scores": dict(dna.naturalness.dimension_scores),
            "dimension_confidence": dict(dna.naturalness.dimension_confidence),
            "score": dna.naturalness.score,
            "confidence": dna.naturalness.confidence,
            "rationale": dna.naturalness.rationale,
        }

    return TalkingAIReport(
        video_id=dna.video_id,
        subject_label=dna.subject_label,
        dna_id=dna.dna_id,
        overall_confidence=dna.overall_confidence,
        trait_summary=trait_summary,
        naturalness_summary=naturalness_summary,
        artifact_patterns=dict(dna.artifact_patterns),
        warnings=list(dna.warnings),
    )


def render_talking_ai_report_markdown(report: TalkingAIReport) -> str:
    lines = [
        f"# Talking AI Intelligence Report -- {report.subject_label}",
        "",
        f"- video_id: `{report.video_id}`",
        f"- dna_id: `{report.dna_id}`",
        f"- Overall confidence: {report.overall_confidence}",
        "",
        "## Domain trait scores",
    ]
    if report.trait_summary:
        for name in TRAIT_FIELDS:
            trait = report.trait_summary.get(name)
            if trait is None:
                continue
            lines.append(
                f"- **{name}**: score {trait['score']:.2f}, confidence {trait['confidence']}, "
                f"{trait['sample_count']} evidence item(s) -- {trait['rationale']}"
            )
    else:
        lines.append("- (no evidence supplied for any domain)")

    lines.append("")
    lines.append("## Naturalness (multidimensional -- never a human/AI classification)")
    if report.naturalness_summary:
        summary = report.naturalness_summary
        lines.append(f"- Overall: score {summary['score']:.2f}, confidence {summary['confidence']}")
        lines.append(f"- {summary['rationale']}")
        lines.append("- Per-dimension scores:")
        for dimension, score in sorted(summary["dimension_scores"].items()):
            confidence = summary["dimension_confidence"].get(dimension, "unknown")
            lines.append(f"  - {dimension}: {score:.2f} (confidence {confidence})")
    else:
        lines.append("- (no naturalness result computed)")

    lines.append("")
    lines.append("## Observed AI-artifact tags (raw counts only -- no single tag implies AI generation)")
    if report.artifact_patterns:
        for tag, count in sorted(report.artifact_patterns.items()):
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- (none observed)")

    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report.warnings)

    return "\n".join(lines) + "\n"


@dataclass(slots=True)
class TalkingAIPatternReport:
    total_patterns: int
    categories: dict[str, int] = field(default_factory=dict)
    top_patterns: list[dict] = field(default_factory=list)


def build_talking_ai_pattern_report(
    patterns: list[TalkingAIProductionPattern], *, top_n: int = 10
) -> TalkingAIPatternReport:
    categories: dict[str, int] = {}
    for pattern in patterns:
        categories[pattern.category] = categories.get(pattern.category, 0) + 1
    ranked = sorted(patterns, key=lambda pattern: (-pattern.supporting_video_count, pattern.pattern_id))
    top_patterns = [
        {
            "category": pattern.category,
            "description": pattern.description,
            "supporting_video_count": pattern.supporting_video_count,
            "confidence": pattern.confidence,
            "evidence_coverage": pattern.evidence_coverage,
        }
        for pattern in ranked[:top_n]
    ]
    return TalkingAIPatternReport(total_patterns=len(patterns), categories=categories, top_patterns=top_patterns)


def render_talking_ai_pattern_report_markdown(report: TalkingAIPatternReport) -> str:
    lines = [
        "# Talking AI Intelligence -- Production Pattern Library",
        "",
        f"- Total patterns: {report.total_patterns}",
        "",
        "## Patterns by category",
    ]
    if report.categories:
        for category, count in sorted(report.categories.items()):
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- (no patterns yet)")

    lines.append("")
    lines.append("## Top patterns (by supporting video count)")
    if report.top_patterns:
        for pattern in report.top_patterns:
            lines.append(
                f"- [{pattern['category']}] {pattern['description']} "
                f"(supported by {pattern['supporting_video_count']} video(s), "
                f"confidence {pattern['confidence']}, evidence coverage {pattern['evidence_coverage']:.2f})"
            )
    else:
        lines.append("- (no patterns yet)")

    return "\n".join(lines) + "\n"
