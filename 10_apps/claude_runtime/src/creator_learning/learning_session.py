"""run_learning_session() -- the one orchestration entry point that
merges evidence into a creator's persistent knowledge base, rebuilds
CreatorDNA over the full accumulated store, diffs it against the
previous snapshot, and updates learning history/state. Performs no
evidence collection itself: if a Connector is involved, it must
already have been run through the caller's own
creator_research.orchestrator.run_job() before its resulting session
is handed in here.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.creator_intelligence.config import CreatorIntelligenceConfig
from src.creator_intelligence.evidence import Evidence
from src.creator_intelligence.models import CreatorDNA
from src.creator_intelligence.summary_builder import build_creator_dna

from .evidence_bridge import merge_into_knowledge_base
from .knowledge_base import CreatorKnowledgeBase
from .learning_history import LearningHistoryEntry, build_learning_history_entry
from .learning_targets import missing_tags_by_trait
from .style_evolution import StyleEvolutionRecord, diff_creator_dna


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_session_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(slots=True)
class LearningSessionResult:
    session_id: str
    creator_id: str
    dna: CreatorDNA
    previous_dna: CreatorDNA | None
    style_evolution: StyleEvolutionRecord
    history_entry: LearningHistoryEntry
    evidence: list[Evidence]
    gaps: dict[str, tuple[str, ...]] = field(default_factory=dict)


def run_learning_session(
    kb: CreatorKnowledgeBase,
    subject_label: str,
    intelligence_config: CreatorIntelligenceConfig,
    *,
    session=None,
    intake_evidence_bundle_path: str | Path | None = None,
) -> LearningSessionResult:
    """One learning session against `kb`. `session` (a
    creator_research ResearchSession) and/or
    `intake_evidence_bundle_path` are both optional -- supplying
    neither simply re-derives CreatorDNA from evidence already on
    disk (zero network, useful for regenerating reports)."""
    count_before = len(kb.load_evidence())

    merged_evidence = merge_into_knowledge_base(
        kb, session=session, intake_evidence_bundle_path=intake_evidence_bundle_path
    )
    count_after = len(merged_evidence)

    previous_dna = kb.latest_dna()
    dna = build_creator_dna(subject_label, merged_evidence, intelligence_config)
    evolution = diff_creator_dna(previous_dna, dna)

    session_id = _new_session_id()
    triggered_at = _now_iso()

    kb.save_dna_snapshot(session_id, dna)

    history_entry = build_learning_history_entry(
        session_id=session_id,
        triggered_at=triggered_at,
        evidence_count_before=count_before,
        evidence_count_after=count_after,
        dna=dna,
    )
    kb.append_learning_history(history_entry.to_dict())

    gaps = missing_tags_by_trait(merged_evidence)

    kb.save_learning_state(
        {
            "last_session_id": session_id,
            "updated_at": triggered_at,
            "evidence_count": count_after,
            "overall_confidence": dna.overall_confidence,
            "gaps": {trait: list(tags) for trait, tags in gaps.items()},
        }
    )

    return LearningSessionResult(
        session_id=session_id,
        creator_id=kb.creator_id,
        dna=dna,
        previous_dna=previous_dna,
        style_evolution=evolution,
        history_entry=history_entry,
        evidence=merged_evidence,
        gaps=gaps,
    )
