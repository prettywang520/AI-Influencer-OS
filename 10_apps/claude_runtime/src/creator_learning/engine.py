"""CreatorLearningEngine -- the one new public entry point this phase
adds. It composes existing, unmodified building blocks
(creator_research's orchestrator, creator_intelligence's
summary_builder, and this package's own knowledge base / learning
session / reports) rather than adding a second orchestration system.
A browser or network connection is only ever opened if the caller
supplies an already-constructed Connector, and even then only via
creator_research.orchestrator.run_job() -- the same gated path Phase
12B.1/12B.2 already established. `connector=None` performs zero
network activity and simply re-learns from evidence already on disk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.creator_intelligence.config import CreatorIntelligenceConfig, load_framework_config
from src.creator_intelligence.models import CreatorDNA
from src.creator_intelligence.serialization import creator_dna_to_dict
from src.creator_research.connector import BaseConnector
from src.creator_research.interfaces import ConnectorSection
from src.creator_research.jobs import ResearchJob
from src.creator_research.orchestrator import load_orchestrator_config, run_job

from .config import LearningConfig
from .creator_url import parse_creator_url
from .knowledge_base import CreatorKnowledgeBase
from .learning_history import LearningHistoryEntry
from .learning_session import LearningSessionResult, run_learning_session
from .reports.builder import build_all_reports
from .reports.markdown import render_all_reports

REPORT_NAMES = ("learning", "visual", "relationship", "reply", "caption", "style")


@dataclass(slots=True)
class LearningResult:
    session_id: str
    creator_id: str
    profile_url: str
    dna: CreatorDNA
    session: LearningSessionResult
    reports: dict[str, str] = field(default_factory=dict)


class CreatorLearningEngine:
    def __init__(
        self,
        config: LearningConfig,
        *,
        creator_intelligence_config: CreatorIntelligenceConfig | None = None,
    ) -> None:
        self.config = config
        self.intelligence_config = creator_intelligence_config or load_framework_config()

    def _knowledge_base_for(self, profile_url: str) -> CreatorKnowledgeBase:
        parsed = parse_creator_url(profile_url)
        return CreatorKnowledgeBase(self.config.resolved_knowledge_base_root(), parsed.creator_id)

    def learn(
        self,
        profile_url: str,
        *,
        connector: BaseConnector | None = None,
        requested_sections: tuple[str, ...] | None = None,
        intake_evidence_bundle_path: str | Path | None = None,
    ) -> LearningResult:
        """One learning session: resolve creator_id from profile_url,
        optionally run `connector` through orchestrator.run_job()
        and/or ingest an intake evidence_bundle.json, merge into the
        knowledge base, rebuild CreatorDNA, diff, save, report."""
        parsed = parse_creator_url(profile_url)
        kb = CreatorKnowledgeBase(self.config.resolved_knowledge_base_root(), parsed.creator_id)

        session = None
        if connector is not None:
            job = ResearchJob(
                creator_id=parsed.creator_id,
                platform=parsed.platform,
                username=parsed.username,
                profile_url=parsed.profile_url,
                connector_name=connector.__class__.__name__,
                requested_sections=requested_sections or ConnectorSection.ALL,
            )
            session = run_job(job, connector, load_orchestrator_config())

        result = run_learning_session(
            kb,
            parsed.username,
            self.intelligence_config,
            session=session,
            intake_evidence_bundle_path=intake_evidence_bundle_path,
        )

        history = [LearningHistoryEntry.from_dict(entry) for entry in kb.load_learning_history()]
        report_dataclasses = build_all_reports(
            creator_id=parsed.creator_id,
            session_id=result.session_id,
            dna=result.dna,
            evidence=result.evidence,
            history=history,
            evolution=result.style_evolution,
        )
        rendered = render_all_reports(report_dataclasses)
        if self.config.generate_reports_on_every_session:
            for name, text in rendered.items():
                kb.save_report(result.session_id, name, text)

        return LearningResult(
            session_id=result.session_id,
            creator_id=parsed.creator_id,
            profile_url=parsed.profile_url,
            dna=result.dna,
            session=result,
            reports=rendered,
        )

    def latest_dna(self, profile_url: str) -> CreatorDNA | None:
        return self._knowledge_base_for(profile_url).latest_dna()

    def history(self, profile_url: str) -> list[LearningHistoryEntry]:
        kb = self._knowledge_base_for(profile_url)
        return [LearningHistoryEntry.from_dict(entry) for entry in kb.load_learning_history()]

    def reports(self, profile_url: str, session_id: str | None = None) -> dict[str, str]:
        """Returns the six rendered Markdown reports plus a
        `creator_dna` JSON export (the seventh, implicit report) for
        one saved session -- the latest one if session_id is omitted.
        Returns {} if no session has ever run for this creator."""
        kb = self._knowledge_base_for(profile_url)
        session_id = session_id or kb.latest_session_id()
        if session_id is None:
            return {}

        result: dict[str, str] = {}
        for name in REPORT_NAMES:
            text = kb.load_report(session_id, name)
            if text is not None:
                result[name] = text

        dna = kb.load_dna_snapshot(session_id)
        if dna is not None:
            result["creator_dna"] = json.dumps(creator_dna_to_dict(dna), indent=2, sort_keys=True)
        return result
