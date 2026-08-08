import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_research.connector import BaseConnector
from src.creator_research.interfaces import ConnectorSection, EvidenceBundle
from src.creator_learning.config import LearningConfig
from src.creator_learning.engine import CreatorLearningEngine
from src.creator_learning.exceptions import InvalidCreatorUrlError
from src.creator_learning.knowledge_base import evidence_to_dict

PROFILE_URL = "https://www.instagram.com/demo_creator_engine_test/"


class _StubProfileOnlyConnector(BaseConnector):
    """Fake, in-memory Connector -- no browser, no network. Only
    collect_profile() is implemented; every other collect_* method
    inherits BaseConnector's default ConnectorNotImplementedError."""

    def __init__(self, excerpt="synthetic bio"):
        self.excerpt = excerpt
        self.calls = []

    def collect_profile(self, job):
        self.calls.append("profile")
        return EvidenceBundle(
            section=ConnectorSection.PROFILE,
            items=[
                Evidence(
                    evidence_type=EvidenceType.OPERATOR_OBSERVATION,
                    source_description="profile bio observed",
                    content_excerpt=self.excerpt,
                    tags=["persona"],
                )
            ],
        )


def _config(tmp):
    return LearningConfig(
        schema_version="1.0", knowledge_base_root=str(Path(tmp) / "kb"),
        report_formats=("markdown",), generate_reports_on_every_session=True,
    )


class LearnWithConnectorTests(unittest.TestCase):
    def test_learn_runs_connector_through_orchestrator_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            connector = _StubProfileOnlyConnector()
            result = engine.learn(PROFILE_URL, connector=connector, requested_sections=(ConnectorSection.PROFILE,))
            self.assertEqual(connector.calls, ["profile"])
            self.assertEqual(len(result.session.evidence), 1)
            self.assertTrue(result.session_id)
            self.assertEqual(set(result.reports.keys()), {"learning", "visual", "relationship", "reply", "caption", "style"})

    def test_reports_saved_to_disk_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            engine.learn(PROFILE_URL, connector=_StubProfileOnlyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            reports = engine.reports(PROFILE_URL)
            self.assertIn("learning", reports)
            self.assertIn("creator_dna", reports)

    def test_reports_not_saved_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            config.generate_reports_on_every_session = False
            engine = CreatorLearningEngine(config)
            engine.learn(PROFILE_URL, connector=_StubProfileOnlyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            reports = engine.reports(PROFILE_URL)
            self.assertNotIn("learning", reports)


class LearnWithoutConnectorTests(unittest.TestCase):
    def test_no_connector_and_no_intake_path_performs_zero_network_relearn(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            first = engine.learn(PROFILE_URL, connector=_StubProfileOnlyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            second = engine.learn(PROFILE_URL)
            self.assertEqual(first.dna.dna_id, second.dna.dna_id)
            self.assertNotEqual(first.session_id, second.session_id)

    def test_intake_evidence_bundle_path_merges_without_a_connector(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            bundle_path = Path(tmp) / "evidence_bundle.json"
            note = Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="manual note", content_excerpt="observed a morning coffee routine")
            bundle_path.write_text(json.dumps([evidence_to_dict(note)]))
            result = engine.learn(PROFILE_URL, intake_evidence_bundle_path=bundle_path)
            self.assertEqual(len(result.session.evidence), 1)


class EngineQueryMethodsTests(unittest.TestCase):
    def test_latest_dna_none_before_any_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            self.assertIsNone(engine.latest_dna(PROFILE_URL))

    def test_history_empty_before_any_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            self.assertEqual(engine.history(PROFILE_URL), [])

    def test_reports_empty_before_any_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            self.assertEqual(engine.reports(PROFILE_URL), {})

    def test_history_grows_across_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            engine.learn(PROFILE_URL, connector=_StubProfileOnlyConnector(), requested_sections=(ConnectorSection.PROFILE,))
            engine.learn(PROFILE_URL)
            self.assertEqual(len(engine.history(PROFILE_URL)), 2)

    def test_different_creators_use_isolated_knowledge_bases(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            engine.learn(
                "https://www.instagram.com/creator_one/", connector=_StubProfileOnlyConnector("bio one"),
                requested_sections=(ConnectorSection.PROFILE,),
            )
            self.assertIsNone(engine.latest_dna("https://www.instagram.com/creator_two/"))
            self.assertIsNotNone(engine.latest_dna("https://www.instagram.com/creator_one/"))


class InvalidProfileUrlTests(unittest.TestCase):
    def test_invalid_profile_url_raises_before_any_evidence_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = CreatorLearningEngine(_config(tmp))
            connector = _StubProfileOnlyConnector()
            with self.assertRaises(InvalidCreatorUrlError):
                engine.learn("https://example.com/not_a_platform_we_recognize", connector=connector)
            self.assertEqual(connector.calls, [])


if __name__ == "__main__":
    unittest.main()
