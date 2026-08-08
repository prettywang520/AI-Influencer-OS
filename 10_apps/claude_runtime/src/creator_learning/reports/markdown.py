"""Renders report dataclasses (reports/models.py) to Markdown text.
Pure string formatting -- no analysis, no I/O.
"""
from __future__ import annotations

from .models import CaptionReport, LearningReport, RelationshipReport, ReplyReport, StyleReport, VisualReport


def _score_str(score: float | None) -> str:
    return f"{score:.2f}" if score is not None else "unknown"


def render_learning_report(report: LearningReport) -> str:
    lines = [
        f"# Learning Report -- {report.creator_id}",
        "",
        f"- Session: `{report.session_id}` (session #{report.session_count})",
        f"- Generated at: {report.generated_at}",
        f"- Evidence in knowledge base: {report.evidence_count}",
        f"- Overall confidence: {report.overall_confidence}",
        "",
        "## Confidence trend",
    ]
    if report.confidence_trend:
        lines.extend(f"- `{session_id}`: {confidence}" for session_id, confidence in report.confidence_trend)
    else:
        lines.append("- (no prior sessions)")
    lines.append("")
    lines.append("## Coverage gaps (unknown remains unknown)")
    any_gap = False
    for trait, missing in sorted(report.gaps.items()):
        if missing:
            any_gap = True
            lines.append(f"- **{trait}**: missing evidence for {', '.join(missing)}")
    if not any_gap:
        lines.append("- No tracked learning-target gaps.")
    return "\n".join(lines) + "\n"


def render_visual_report(report: VisualReport) -> str:
    lines = [
        f"# Visual Report -- {report.creator_id}",
        "",
        f"- Session: `{report.session_id}`",
        f"- Visual realism: score {_score_str(report.visual_realism_score)}, "
        f"confidence {report.visual_realism_confidence or 'unknown'}",
        f"  - {report.visual_realism_rationale or '(no rationale recorded)'}",
        f"- Photography: score {_score_str(report.photography_score)}, "
        f"confidence {report.photography_confidence or 'unknown'}",
        f"  - {report.photography_rationale or '(no rationale recorded)'}",
        "",
        f"## Covered topics: {', '.join(report.covered_tags) or '(none yet)'}",
        f"## Missing topics: {', '.join(report.missing_tags) or '(none)'}",
        "",
        "## Cited evidence excerpts",
    ]
    if report.cited_excerpts:
        lines.extend(f"- {excerpt}" for excerpt in report.cited_excerpts)
    else:
        lines.append("- (no excerpts cited yet)")
    return "\n".join(lines) + "\n"


def render_relationship_report(report: RelationshipReport) -> str:
    lines = [
        f"# Relationship Report -- {report.creator_id}",
        "",
        f"- Session: `{report.session_id}`",
        f"- Relationships trait: score {_score_str(report.relationships_score)}, "
        f"confidence {report.relationships_confidence or 'unknown'}",
        "",
        "## Relationship claims (basis-qualified -- never presented as more certain than their basis)",
    ]
    if report.claims:
        for claim in report.claims:
            evidence_ids = ", ".join(claim["evidence_ids"]) or "none"
            lines.append(f"- {claim['description']} (basis: {claim['basis']}, evidence: {evidence_ids})")
    else:
        lines.append("- (no relationship claims yet)")
    return "\n".join(lines) + "\n"


def render_reply_report(report: ReplyReport) -> str:
    lines = [
        f"# Reply Report -- {report.creator_id}",
        "",
        f"- Session: `{report.session_id}`",
        f"- Replies trait: score {_score_str(report.replies_score)}, confidence {report.replies_confidence or 'unknown'}",
        f"  - {report.rationale or '(no rationale recorded)'}",
        "- This report never generates a reply; it only summarizes observed tone/cadence.",
        "",
        f"## Language mix covered: {', '.join(report.language_mix_covered) or '(none yet)'}",
        f"## Language mix missing: {', '.join(report.language_mix_missing) or '(none)'}",
        "",
        "## Cited evidence excerpts",
    ]
    if report.cited_excerpts:
        lines.extend(f"- {excerpt}" for excerpt in report.cited_excerpts)
    else:
        lines.append("- (no excerpts cited yet)")
    return "\n".join(lines) + "\n"


def render_caption_report(report: CaptionReport) -> str:
    lines = [
        f"# Caption Report -- {report.creator_id}",
        "",
        f"- Session: `{report.session_id}`",
        f"- Captions trait: score {_score_str(report.captions_score)}, confidence {report.captions_confidence or 'unknown'}",
        f"  - {report.rationale or '(no rationale recorded)'}",
        "",
        "## Hashtag frequency",
    ]
    if report.hashtag_frequency:
        for tag, count in sorted(report.hashtag_frequency.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- #{tag}: {count}")
    else:
        lines.append("- (none observed yet)")
    lines.append("")
    lines.append("## Mention frequency")
    if report.mention_frequency:
        for mention, count in sorted(report.mention_frequency.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- @{mention}: {count}")
    else:
        lines.append("- (none observed yet)")
    lines.append("")
    lines.append("## Cited evidence excerpts")
    if report.cited_excerpts:
        lines.extend(f"- {excerpt}" for excerpt in report.cited_excerpts)
    else:
        lines.append("- (no excerpts cited yet)")
    return "\n".join(lines) + "\n"


def render_style_report(report: StyleReport) -> str:
    lines = [
        f"# Style Report -- {report.creator_id}",
        "",
        f"- Session: `{report.session_id}`",
        f"- Baseline session: {report.is_baseline}",
        f"- New relationship claims: {report.new_relationship_claim_count}",
        f"- Changed relationship claims: {report.changed_relationship_claim_count}",
        f"- New warnings: {report.new_warning_count}",
        "",
        "## Baseline trait scores" if report.is_baseline else "## Factual changes since last session",
    ]
    if report.trait_summary:
        lines.extend(f"- {line}" for line in report.trait_summary)
    else:
        lines.append("- No trait changes observed.")
    return "\n".join(lines) + "\n"


RENDERERS = {
    "learning": render_learning_report,
    "visual": render_visual_report,
    "relationship": render_relationship_report,
    "reply": render_reply_report,
    "caption": render_caption_report,
    "style": render_style_report,
}


def render_all_reports(reports: dict[str, object]) -> dict[str, str]:
    return {name: RENDERERS[name](report) for name, report in reports.items() if name in RENDERERS}
