"""Human-readable reports: one per analyzed video (build_video_report()/
render_video_report_markdown()) and one over the cross-video pattern
library (build_pattern_report()/render_pattern_report_markdown()).
Pure functions -- no I/O, no analysis, just formatting over an
already-built VideoDNA / already-ingested ProductionPattern list.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge_base import ProductionPattern
from .production_dna import TRAIT_FIELDS, VideoDNA


@dataclass(slots=True)
class VideoReport:
    video_id: str
    subject_label: str
    dna_id: str
    overall_confidence: str
    trait_summary: dict[str, dict] = field(default_factory=dict)
    story_beats: list[dict] = field(default_factory=list)
    cta_observations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_video_report(dna: VideoDNA) -> VideoReport:
    trait_summary = {}
    for name in TRAIT_FIELDS:
        trait = getattr(dna, name)
        if trait is not None:
            trait_summary[name] = {"score": trait.score, "confidence": trait.confidence, "rationale": trait.rationale}
    return VideoReport(
        video_id=dna.video_id,
        subject_label=dna.subject_label,
        dna_id=dna.dna_id,
        overall_confidence=dna.overall_confidence,
        trait_summary=trait_summary,
        story_beats=[
            {"beat_type": beat.beat_type, "description": beat.description} for beat in dna.story_beats
        ],
        cta_observations=[
            {"cta_type": obs.cta_type, "timing_seconds": obs.timing_seconds} for obs in dna.cta_observations
        ],
        warnings=list(dna.warnings),
    )


def render_video_report_markdown(report: VideoReport) -> str:
    lines = [
        f"# Video Intelligence Report -- {report.subject_label}",
        "",
        f"- video_id: `{report.video_id}`",
        f"- dna_id: `{report.dna_id}`",
        f"- Overall confidence: {report.overall_confidence}",
        "",
        "## Trait scores",
    ]
    if report.trait_summary:
        for name in TRAIT_FIELDS:
            trait = report.trait_summary.get(name)
            if trait is None:
                continue
            lines.append(f"- **{name}**: score {trait['score']:.2f}, confidence {trait['confidence']} -- {trait['rationale']}")
    else:
        lines.append("- (no evidence supplied for any trait)")

    lines.append("")
    lines.append("## Story beats")
    if report.story_beats:
        for beat in report.story_beats:
            lines.append(f"- {beat['beat_type']}: {beat['description']}")
    else:
        lines.append("- (none observed)")

    lines.append("")
    lines.append("## CTA observations")
    if report.cta_observations:
        for observation in report.cta_observations:
            timing = observation["timing_seconds"]
            timing_str = f"{timing:.1f}s" if timing is not None else "unknown timing"
            lines.append(f"- {observation['cta_type']} at {timing_str}")
    else:
        lines.append("- (none observed)")

    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in report.warnings)

    return "\n".join(lines) + "\n"


@dataclass(slots=True)
class PatternReport:
    total_patterns: int
    categories: dict[str, int] = field(default_factory=dict)
    top_patterns: list[dict] = field(default_factory=list)


def build_pattern_report(patterns: list[ProductionPattern], *, top_n: int = 10) -> PatternReport:
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
        }
        for pattern in ranked[:top_n]
    ]
    return PatternReport(total_patterns=len(patterns), categories=categories, top_patterns=top_patterns)


def render_pattern_report_markdown(report: PatternReport) -> str:
    lines = [
        "# Video Intelligence -- Production Pattern Library",
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
                f"(supported by {pattern['supporting_video_count']} video(s), confidence {pattern['confidence']})"
            )
    else:
        lines.append("- (no patterns yet)")

    return "\n".join(lines) + "\n"
