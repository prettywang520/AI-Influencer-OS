import json
import tempfile
import unittest
from pathlib import Path

from src.creator_research.connector import ConnectorRegistry
from src.creator_research.instagram.config import load_instagram_connector_config
from src.creator_research.instagram.connector import InstagramResearchConnector, register_instagram_connector
from src.creator_research.instagram.navigator import FakeBrowserAdapter, FakeElement
from src.creator_research.instagram.selectors import default_instagram_selectors
from src.creator_research.jobs import ResearchJob
from src.creator_research.orchestrator import OrchestratorConfig, run_job
from src.creator_research.state import JobStatus

PROFILE_URL = "https://example.invalid/demo_creator"


def _build_environment(tmp_path: Path, *, access_limit_behavior: str = "skip"):
    adapter = FakeBrowserAdapter()
    selectors = default_instagram_selectors()
    config = load_instagram_connector_config()
    config.access_limit_behavior = access_limit_behavior
    # Tests must never actually sleep: rate limiting is exercised in
    # isolation by test_rate_limit.py, not here.
    config.rate_limit_enabled = False

    page = adapter.add_page(PROFILE_URL)
    page.register(selectors.profile.username[0], [FakeElement(text="demo_creator")])
    page.register(selectors.profile.bio[0], [FakeElement(text="travel diary")])

    grid_items = [
        FakeElement(attributes={"href": f"https://example.invalid/p/{i}/", "content_type": "post"})
        for i in range(6)
    ]
    page.register(selectors.grid.grid_item[0], grid_items)

    for i in range(6):
        post_url = f"https://example.invalid/p/{i}/"
        post_page = adapter.add_page(post_url)
        post_page.register(selectors.post.caption[0], [FakeElement(text=f"caption {i} #travel")])
        post_page.register(selectors.post.like_count[0], [FakeElement(text="100")])
        post_page.register(
            selectors.comments.comment_item[0], [FakeElement(text=f"nice {i}", attributes={"username": "fan1"})]
        )
        post_page.register(
            selectors.comments.reply_item[0],
            [FakeElement(text="thanks!!", attributes={"username": "demo_creator", "parent_comment_text": "nice"})],
        )

    connector = InstagramResearchConnector(
        adapter, config, checkpoint_dir=tmp_path / "checkpoints", diagnostics_dir=tmp_path / "diagnostics"
    )
    job = ResearchJob(
        creator_id="c1", platform="instagram", username="demo_creator", profile_url=PROFILE_URL, connector_name="instagram"
    )
    return adapter, selectors, config, connector, job


class ConnectorTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class ConnectorRegistryIntegrationTests(ConnectorTempTestCase):
    def test_registers_and_resolves_by_name(self):
        _, _, config, connector, _ = _build_environment(self.tmp_path)
        registry = ConnectorRegistry()
        register_instagram_connector(registry, connector.adapter, config)
        resolved = registry.get("instagram")
        self.assertEqual(resolved.name, "instagram")

    def test_import_never_auto_registers(self):
        registry = ConnectorRegistry()
        self.assertEqual(registry.registered_names(), ())


class FullRunTests(ConnectorTempTestCase):
    def test_reaches_complete_via_orchestrator(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        orch_config = OrchestratorConfig(schema_version="1.0", fail_fast=False, required_sections=("profile",))
        session = run_job(job, connector, orch_config)
        self.assertEqual(job.status, JobStatus.COMPLETE)

    def test_report_coverage_reflects_sections(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        orch_config = OrchestratorConfig(schema_version="1.0", fail_fast=False, required_sections=("profile",))
        session = run_job(job, connector, orch_config)
        self.assertEqual(session.report.coverage["profile"], 1)
        self.assertEqual(session.report.coverage["captions"], 6)
        self.assertEqual(session.report.coverage["creator_replies"], 6)

    def test_creator_replies_only_include_target_username(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        orch_config = OrchestratorConfig(schema_version="1.0", fail_fast=False, required_sections=("profile",))
        run_job(job, connector, orch_config)
        bundle = connector.collect_creator_replies(job)
        for evidence in bundle.items:
            self.assertIn("creator_reply", evidence.tags)

    def test_evidence_items_are_real_evidence_objects(self):
        from src.creator_intelligence.evidence import Evidence

        _, _, _, connector, job = _build_environment(self.tmp_path)
        bundle = connector.collect_profile(job)
        self.assertIsInstance(bundle.items[0], Evidence)


class DiagnosticsTests(ConnectorTempTestCase):
    def test_diagnostics_saved_after_run(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        orch_config = OrchestratorConfig(schema_version="1.0", fail_fast=False, required_sections=("profile",))
        run_job(job, connector, orch_config)
        path = connector.save_diagnostics_for(job)
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())

    def test_diagnostics_content_has_no_secrets(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        orch_config = OrchestratorConfig(schema_version="1.0", fail_fast=False, required_sections=("profile",))
        run_job(job, connector, orch_config)
        path = connector.save_diagnostics_for(job)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for forbidden in ("password", "cookie", "token", "auth_header"):
            self.assertNotIn(forbidden, payload)

    def test_sections_attempted_and_completed_recorded(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        connector.collect_profile(job)
        recorder = connector._diagnostics_for(job)
        self.assertIn("profile", recorder.sections_attempted)
        self.assertIn("profile", recorder.sections_completed)

    def test_items_collected_counted(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        connector.collect_captions(job)
        recorder = connector._diagnostics_for(job)
        self.assertEqual(recorder.items_collected["captions"], 6)


class AccessLimitBehaviorTests(ConnectorTempTestCase):
    def test_skip_behavior_returns_warning_bundle(self):
        adapter, selectors, config, connector, job = _build_environment(self.tmp_path, access_limit_behavior="skip")
        page = adapter.pages[PROFILE_URL]
        page.register(selectors.access_state.private_account_marker[0], [FakeElement()])
        bundle = connector.collect_profile(job)
        self.assertEqual(bundle.items, [])
        self.assertEqual(len(bundle.warnings), 1)

    def test_stop_behavior_raises(self):
        from src.creator_research.instagram.exceptions import AccessLimitedError

        adapter, selectors, config, connector, job = _build_environment(self.tmp_path, access_limit_behavior="stop")
        page = adapter.pages[PROFILE_URL]
        page.register(selectors.access_state.private_account_marker[0], [FakeElement()])
        with self.assertRaises(AccessLimitedError):
            connector.collect_profile(job)

    def test_orchestrator_continues_past_skipped_access_limit(self):
        adapter, selectors, config, connector, job = _build_environment(self.tmp_path, access_limit_behavior="skip")
        page = adapter.pages[PROFILE_URL]
        page.register(selectors.access_state.rate_limit_marker[0], [FakeElement()])
        orch_config = OrchestratorConfig(schema_version="1.0", fail_fast=False, required_sections=())
        session = run_job(job, connector, orch_config)
        self.assertEqual(job.status, JobStatus.COMPLETE)
        # A connector method that returns (rather than raises) a
        # warning-bearing bundle surfaces its warnings through the
        # evidence queue / report, not session.warnings (which is
        # reserved for exceptions the orchestrator itself catches).
        self.assertTrue(any("rate_limited" in w for w in session.report.warnings))

    def test_required_section_access_limited_fails_job(self):
        adapter, selectors, config, connector, job = _build_environment(self.tmp_path, access_limit_behavior="skip")
        page = adapter.pages[PROFILE_URL]
        page.register(selectors.access_state.private_account_marker[0], [FakeElement()])
        orch_config = OrchestratorConfig(schema_version="1.0", fail_fast=False, required_sections=("profile",))
        run_job(job, connector, orch_config)
        self.assertEqual(job.status, JobStatus.FAILED)

    def test_fail_fast_true_stops_orchestrator_immediately(self):
        adapter, selectors, config, connector, job = _build_environment(self.tmp_path, access_limit_behavior="stop")
        page = adapter.pages[PROFILE_URL]
        page.register(selectors.access_state.private_account_marker[0], [FakeElement()])
        orch_config = OrchestratorConfig(schema_version="1.0", fail_fast=True, required_sections=("profile",))
        session = run_job(job, connector, orch_config)
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertEqual(session.progress.completed_steps, [])


class CheckpointResumeTests(ConnectorTempTestCase):
    def test_checkpoint_written_after_grid_collection(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        connector.collect_grid(job)
        checkpoint_path = self.tmp_path / "checkpoints" / f"{job.job_id}_grid.json"
        self.assertTrue(checkpoint_path.is_file())

    def test_resume_seeds_dedupe_and_skips_already_processed(self):
        adapter, selectors, config, connector, job = _build_environment(self.tmp_path)
        connector.collect_grid(job)

        # a fresh connector instance, same checkpoint dir, same adapter/job:
        connector2 = InstagramResearchConnector(
            adapter, config, checkpoint_dir=self.tmp_path / "checkpoints", diagnostics_dir=self.tmp_path / "diagnostics"
        )
        bundle = connector2.collect_grid(job)
        # everything was already processed in the checkpoint -- nothing new
        self.assertEqual(bundle.items, [])

    def test_mismatched_creator_checkpoint_raises(self):
        from src.creator_research.instagram.exceptions import CheckpointMismatchError

        _, _, _, connector, job = _build_environment(self.tmp_path)
        connector.collect_grid(job)

        other_job = ResearchJob(
            creator_id="different_creator", platform="instagram", username="other", profile_url=PROFILE_URL,
            connector_name="instagram",
        )
        # force the same job_id to simulate a corrupted/mismatched checkpoint scenario
        other_job.job_id = job.job_id
        with self.assertRaises(CheckpointMismatchError):
            connector._load_or_new_checkpoint(other_job, "grid")


class RedactionTests(ConnectorTempTestCase):
    def test_audience_usernames_redacted_by_default(self):
        _, _, _, connector, job = _build_environment(self.tmp_path)
        bundle = connector.collect_comments(job)
        for evidence in bundle.items:
            self.assertNotIn("fan1", evidence.source_description)


if __name__ == "__main__":
    unittest.main()
