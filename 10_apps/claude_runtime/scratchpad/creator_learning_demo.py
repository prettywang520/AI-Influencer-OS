"""Safe demo for Phase 12C.0 -- Creator Learning Engine.

Entirely synthetic: a fictional "demo_creator" and a fictional
StubConnector (no browser, no network -- same pattern as
creator_research's own DummyConnector demo). Runs TWO learning
sessions to show accumulation, a stable dna_id when no new evidence
is added, and Style Evolution once new evidence does arrive. Uses a
temp directory for the knowledge base -- writes nothing to the real
output/creator_learning/. Not part of the automated test suite.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_learning.config import LearningConfig
from src.creator_learning.engine import CreatorLearningEngine
from src.creator_research.connector import BaseConnector
from src.creator_research.interfaces import ConnectorSection, EvidenceBundle

PROFILE_URL = "https://www.instagram.com/demo_creator/"


class StubProfileConnector(BaseConnector):
    """Fake, in-memory Connector standing in for a real research
    connector. Only collect_profile() is implemented; every other
    collect_* method inherits BaseConnector's
    ConnectorNotImplementedError default."""

    def __init__(self, excerpts):
        self._excerpts = excerpts

    def collect_profile(self, job):
        return EvidenceBundle(
            section=ConnectorSection.PROFILE,
            items=[
                Evidence(
                    evidence_type=EvidenceType.OPERATOR_OBSERVATION,
                    source_description=f"profile observation #{i}",
                    content_excerpt=excerpt,
                    tags=tags,
                )
                for i, (excerpt, tags) in enumerate(self._excerpts)
            ],
        )


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    config = LearningConfig(
        schema_version="1.0",
        knowledge_base_root=str(tmp_path / "kb"),
        report_formats=("markdown",),
        generate_reports_on_every_session=True,
    )
    engine = CreatorLearningEngine(config)

    print("=== 1. Session one -- profile connector run through orchestrator.run_job() ===")
    connector_one = StubProfileConnector(
        [
            ("morning coffee routine, warm natural light #coffee", ["coffee", "lighting"]),
            ("travel photos from a recent trip #travel", ["travel"]),
        ]
    )
    result1 = engine.learn(PROFILE_URL, connector=connector_one, requested_sections=(ConnectorSection.PROFILE,))
    print("  session_id:", result1.session_id)
    print("  evidence collected:", len(result1.session.evidence))
    print("  dna overall_confidence:", result1.dna.overall_confidence)
    print("  style evolution baseline:", result1.session.style_evolution.is_baseline)
    print("  gaps[storytelling]:", result1.session.gaps["storytelling"])

    print("\n=== 2. Session two -- zero-network re-learn (no connector) ===")
    result2 = engine.learn(PROFILE_URL)
    print("  session_id:", result2.session_id, "(new session, same dna_id expected)")
    print("  dna_id stable:", result1.dna.dna_id == result2.dna.dna_id)
    print("  style evolution baseline:", result2.session.style_evolution.is_baseline)

    print("\n=== 3. Session three -- new evidence via a second connector run ===")
    connector_three = StubProfileConnector(
        [("family photo shared with a caption about a sister", ["family", "relationship"])]
    )
    result3 = engine.learn(PROFILE_URL, connector=connector_three, requested_sections=(ConnectorSection.PROFILE,))
    print("  evidence count now:", len(result3.session.evidence), "(expected 3: 2 from session one + 1 new)")
    print("  dna_id changed:", result1.dna.dna_id != result3.dna.dna_id)
    print("  gaps[human_authenticity] narrowed:", result3.session.gaps["human_authenticity"])

    print("\n=== 4. Learning history across all three sessions ===")
    history = engine.history(PROFILE_URL)
    for entry in history:
        print(f"  {entry.session_id}: +{entry.new_evidence_count} evidence, confidence={entry.overall_confidence}")

    print("\n=== 5. Reports for the latest session ===")
    reports = engine.reports(PROFILE_URL)
    print("  report keys:", sorted(reports.keys()))
    print("\n  --- learning.md (first 400 chars) ---")
    print(reports["learning"][:400])
    print("\n  --- style.md (first 400 chars) ---")
    print(reports["style"][:400])

print("\nDemo complete. Synthetic connector/creator only. No network access. No real creator analyzed.")
