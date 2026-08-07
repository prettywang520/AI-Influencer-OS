"""Pure typing/contracts for the Creator Research Agent -- no
behavior, no I/O, no platform knowledge. `EvidenceBundle` wraps
`creator_intelligence.evidence.Evidence` directly (reused, never
redefined) so a future integration phase can feed its contents
straight into `creator_intelligence.intake`'s `add_*()` functions
without a translation layer. `Connector` is the structural (Protocol)
contract; `connector.py`'s `BaseConnector` is the inheritable ABC
version of the same 10 methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.creator_intelligence.evidence import Evidence


class ConnectorSection:
    """The 10 evidence-collection sections a Connector implements --
    matches its 10 collect_* methods one-to-one."""

    PROFILE = "profile"
    GRID = "grid"
    POSTS = "posts"
    CAPTIONS = "captions"
    COMMENTS = "comments"
    CREATOR_REPLIES = "creator_replies"
    REELS = "reels"
    HIGHLIGHTS = "highlights"
    RELATIONSHIPS = "relationships"
    VISUAL_EXAMPLES = "visual_examples"

    ALL = (
        PROFILE,
        GRID,
        POSTS,
        CAPTIONS,
        COMMENTS,
        CREATOR_REPLIES,
        REELS,
        HIGHLIGHTS,
        RELATIONSHIPS,
        VISUAL_EXAMPLES,
    )


@dataclass(slots=True)
class EvidenceBundle:
    """What one Connector method call returns: a transient, in-memory
    collection of Evidence for a single section, plus any warnings the
    connector wants to surface. Distinct from
    creator_intelligence.intake's on-disk evidence_bundle.json (the
    merged, persisted store) -- this bundle is never itself written to
    disk by this package."""

    section: str
    items: list[Evidence] = field(default_factory=list)
    collected_at: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Connector(Protocol):
    """Structural contract every research connector satisfies. Never
    implemented in this package -- see connector.py's BaseConnector
    for the inheritable ABC form, and
    docs/creator_research/connector_contract.md for what a future real
    connector must (and must not) do."""

    def collect_profile(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_grid(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_posts(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_captions(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_comments(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_creator_replies(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_reels(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_highlights(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_relationships(self, job: "ResearchJob") -> EvidenceBundle: ...

    def collect_visual_examples(self, job: "ResearchJob") -> EvidenceBundle: ...
