"""Bridges Evidence into a CreatorKnowledgeBase from one of the two
already-built, already-safe sources this phase adds no new collection
capability beyond:

1. A creator_research.session.ResearchSession produced by running an
   existing Connector through creator_research.orchestrator.run_job()
   (duck-typed via `.evidence_queue.all_items()` -- this module never
   imports creator_research at module scope, so creator_learning stays
   usable with zero network/browser dependencies present).
2. An existing creator_intelligence.intake workspace's on-disk
   evidence_bundle.json (manually-collected evidence).

No network access, no browser automation, and no new evidence types
are introduced here.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.creator_intelligence.evidence import Evidence

from .exceptions import KnowledgeBaseError
from .knowledge_base import CreatorKnowledgeBase, evidence_from_dict


def evidence_from_research_session(session) -> list[Evidence]:
    return list(session.evidence_queue.all_items())


def evidence_from_intake_workspace(evidence_bundle_path: str | Path) -> list[Evidence]:
    path = Path(evidence_bundle_path)
    if not path.exists():
        raise KnowledgeBaseError(f"Intake evidence bundle not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise KnowledgeBaseError(f"Intake evidence bundle must be a JSON list: {path}")
    return [evidence_from_dict(item) for item in raw]


def merge_into_knowledge_base(
    kb: CreatorKnowledgeBase,
    *,
    session=None,
    intake_evidence_bundle_path: str | Path | None = None,
) -> list[Evidence]:
    """Collects Evidence from whichever of the two optional sources is
    supplied (both may be supplied together) and merges the result
    into `kb`. Returns the full merged store. Supplying neither is
    valid -- it re-merges zero new items, useful for regenerating a
    CreatorDNA/reports from evidence already on disk."""
    new_items: list[Evidence] = []
    if session is not None:
        new_items.extend(evidence_from_research_session(session))
    if intake_evidence_bundle_path is not None:
        new_items.extend(evidence_from_intake_workspace(intake_evidence_bundle_path))
    return kb.merge_evidence(new_items)
