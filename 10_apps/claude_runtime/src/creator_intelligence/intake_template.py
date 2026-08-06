"""Phase 12A.1 handoff -- generates an evidence intake checklist for a
future phase to use when it performs real creator analysis. This
module performs NO analysis itself: it only lists what evidence
categories each analyzer needs, tagged for how it should be collected
(operator-mediated only). Every generated document carries an
explicit banner stating no analysis has been performed.
"""
from __future__ import annotations

NO_ANALYSIS_BANNER = (
    "NOTE: This is an intake checklist only. No analysis of the listed "
    "subject(s) has been performed by Phase 12A.0. All evidence must be "
    "collected manually by a human operator -- no scraping, login "
    "automation, or access-control bypass is permitted at any future phase."
)

# tag -> (category label, description). Mirrors the tag vocabulary the
# analyzers in this package key off of (see analyzer_base.evidence_for_tags).
EVIDENCE_CATEGORIES: dict[str, tuple[str, str]] = {
    "persona": ("Persona", "Tone, values, recurring themes, self-presentation."),
    "visual_realism": ("Visual Realism", "Lighting, composition, and editing style impressions."),
    "photography": ("Photography", "Framing, setting variety, cross-post consistency."),
    "human_authenticity": ("Human Authenticity", "Narrative-consistency observations."),
    "relationship": ("Relationships", "Relationship mentions, each with its evidentiary basis."),
    "caption": ("Captions", "Short, copyright-respecting caption excerpts and tone notes."),
    "reply": ("Replies", "Observations about past reply tone/length/cadence -- not drafts."),
    "storytelling": ("Storytelling", "Narrative arc and recurring motif observations."),
    "reels": ("Reels", "Hook style, pacing, editing cadence, audio usage observations."),
    "branding": ("Branding", "Visual/verbal branding consistency observations."),
    "posting": ("Posting", "Posting cadence and scheduling pattern observations."),
    "engagement": ("Engagement", "Publicly visible engagement pattern observations."),
    "growth": ("Growth", "Publicly visible growth trend observations, if available."),
}


def build_intake_checklist(subject_labels: list[str]) -> str:
    """Returns a Markdown intake checklist for the given subject
    labels (operator-chosen free text -- callers decide what label to
    use for each real or synthetic subject)."""
    lines = ["# Phase 12A.1 Evidence Intake Checklist", "", NO_ANALYSIS_BANNER, ""]
    for label in subject_labels:
        lines.append(f"## Subject: {label}")
        lines.append("")
        for tag, (category, description) in EVIDENCE_CATEGORIES.items():
            lines.append(f"- [ ] **{category}** (`{tag}`): {description}")
        lines.append("")
    return "\n".join(lines)
