"""Evidence model and intake boundaries.

Evidence is the only input this framework ever analyzes, and it is
always operator-supplied or synthetic (tests/demo) -- never fetched,
scraped, or automated. `EvidenceType` is a closed allow-list; nothing
implying automated collection (an API poll, a scrape, a browser
session) is a member, and `evidence.py` never imports network,
browser, or `social`/`publishing`/`instagram_connector` code. See
docs/creator_intelligence/evidence_guide.md for the operator-facing
boundary list this enforces in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import _hashing


class EvidenceType:
    """Closed set of allowed evidence sources -- all human-mediated."""

    SCREENSHOT = "screenshot"
    MANUAL_NOTE = "manual_note"
    TEXT_EXCERPT = "text_excerpt"
    EXPORTED_DATA = "exported_data"
    OPERATOR_OBSERVATION = "operator_observation"

    ALL = (SCREENSHOT, MANUAL_NOTE, TEXT_EXCERPT, EXPORTED_DATA, OPERATOR_OBSERVATION)


class EvidenceError(RuntimeError):
    """Base error for evidence construction/validation."""


class InvalidEvidenceTypeError(EvidenceError):
    """Raised when an evidence_type outside the allowed set is used."""


def _compute_evidence_id(
    evidence_type: str,
    source_description: str,
    content_excerpt: str,
    collected_by: str,
    tags: tuple[str, ...],
) -> str:
    payload = {
        "evidence_type": evidence_type,
        "source_description": source_description,
        "content_excerpt": content_excerpt,
        "collected_by": collected_by,
        "tags": sorted(tags),
    }
    return _hashing.content_hash(payload)


@dataclass(slots=True)
class Evidence:
    """One piece of operator-supplied evidence about a creator's
    public content. `evidence_id` is a stable content hash (excludes
    `collected_at`, so re-recording identical evidence at a different
    time does not mint a new ID). `content_excerpt` must stay short --
    this is enforced by validation.py's excerpt-length check, not
    here, so construction itself never silently truncates data."""

    evidence_type: str
    source_description: str
    content_excerpt: str = ""
    collected_at: str = ""
    collected_by: str = "operator"
    tags: list[str] = field(default_factory=list)
    evidence_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.evidence_type not in EvidenceType.ALL:
            raise InvalidEvidenceTypeError(
                f"evidence_type must be one of {EvidenceType.ALL}, got {self.evidence_type!r}"
            )
        self.evidence_id = _compute_evidence_id(
            self.evidence_type,
            self.source_description,
            self.content_excerpt,
            self.collected_by,
            tuple(self.tags),
        )
