import unittest

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_research.connector import BaseConnector
from src.creator_research.interfaces import ConnectorSection, EvidenceBundle
from src.creator_research.jobs import ResearchJob
from src.creator_research.orchestrator import OrchestratorConfig, run_job
from src.creator_research.planner import SECTION_STATE_MAP, ResearchPlanner
from src.creator_research.progress import ResearchProgress
from src.creator_research.state import JobStatus


def _job(**overrides):
    defaults = dict(
        creator_id="c1", platform="instagram", username="demo", profile_url="https://x/demo", connector_name="fake"
    )
    defaults.update(overrides)
    return ResearchJob(**defaults)


def _config(**overrides):
    defaults = dict(schema_version="1.0", fail_fast=False, required_sections=("profile",))
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)


class FakeConnector(BaseConnector):
    """Local fake connector for orchestrator tests only -- returns
    small synthetic EvidenceBundles, never touches the network."""

    name = "fake"

    def __init__(self, *, fail_sections=()):
        self.fail_sections = set(fail_sections)
        self.calls: list[str] = []

    def _bundle(self, section, n=1):
        self.calls.append(section)
        if section in self.fail_sections:
            raise RuntimeError(f"simulated failure for {section}")
        items = [
            Evidence(evidence_type=EvidenceType.OPERATOR_OBSERVATION, source_description=f"{section} item {i}")
            for i in range(n)
        ]
        return EvidenceBundle(section=section, items=items)

    def collect_profile(self, job):
        return self._bundle(ConnectorSection.PROFILE)

    def collect_grid(self, job):
        return self._bundle(ConnectorSection.GRID, 2)

    def collect_posts(self, job):
        return self._bundle(ConnectorSection.POSTS, 2)

    def collect_captions(self, job):
        return self._bundle(ConnectorSection.CAPTIONS, 3)

    def collect_comments(self, job):
        return self._bundle(ConnectorSection.COMMENTS, 2)

    def collect_creator_replies(self, job):
        return self._bundle(ConnectorSection.CREATOR_REPLIES, 2)

    def collect_reels(self, job):
        return self._bundle(ConnectorSection.REELS, 2)

    def collect_highlights(self, job):
        return self._bundle(ConnectorSection.HIGHLIGHTS, 1)

    def collect_relationships(self, job):
        return self._bundle(ConnectorSection.RELATIONSHIPS, 1)

    def collect_visual_examples(self, job):
        return self._bundle(ConnectorSection.VISUAL_EXAMPLES, 1)


class RunJobHappyPathTests(unittest.TestCase):
    def test_reaches_complete(self):
        job = _job()
        session = run_job(job, FakeConnector(), _config())
        self.assertEqual(job.status, JobStatus.COMPLETE)

    def test_all_eight_working_states_completed(self):
        job = _job()
        session = run_job(job, FakeConnector(), _config())
        self.assertEqual(list(session.progress.completed_steps), list(JobStatus.WORKING_STATES))

    def test_progress_reaches_one_hundred_percent(self):
        job = _job()
        session = run_job(job, FakeConnector(), _config())
        self.assertEqual(session.progress.percentage, 1.0)

    def test_report_coverage_matches_fake_data(self):
        job = _job()
        session = run_job(job, FakeConnector(), _config())
        self.assertEqual(session.report.coverage[ConnectorSection.CAPTIONS], 3)
        self.assertEqual(session.report.coverage[ConnectorSection.PROFILE], 1)

    def test_report_has_no_missing_sections(self):
        job = _job()
        session = run_job(job, FakeConnector(), _config())
        self.assertEqual(session.report.missing_sections, [])

    def test_session_statistics_record_one_attempt_per_section(self):
        job = _job()
        session = run_job(job, FakeConnector(), _config())
        self.assertEqual(session.statistics[ConnectorSection.PROFILE], 1)

    def test_evidence_queue_holds_all_collected_items(self):
        job = _job()
        session = run_job(job, FakeConnector(), _config())
        self.assertEqual(len(session.evidence_queue.all_items()), 1 + 2 + 2 + 3 + 2 + 2 + 2 + 1 + 1 + 1)


class RunJobFailureHandlingTests(unittest.TestCase):
    def test_fail_fast_false_continues_past_a_failed_section(self):
        job = _job()
        connector = FakeConnector(fail_sections={ConnectorSection.GRID})
        session = run_job(job, connector, _config(fail_fast=False))
        self.assertEqual(job.status, JobStatus.COMPLETE)
        self.assertIn(ConnectorSection.GRID, session.report.sections_failed)
        # collection continued past the failure to later sections:
        self.assertIn(ConnectorSection.CAPTIONS, connector.calls)

    def test_fail_fast_true_aborts_immediately(self):
        job = _job()
        connector = FakeConnector(fail_sections={ConnectorSection.GRID})
        run_job(job, connector, _config(fail_fast=True))
        self.assertEqual(job.status, JobStatus.FAILED)
        # never reached captions -- aborted right after grid failed
        self.assertNotIn(ConnectorSection.CAPTIONS, connector.calls)

    def test_required_section_never_collected_fails_job(self):
        job = _job()
        connector = FakeConnector(fail_sections={ConnectorSection.PROFILE})
        run_job(job, connector, _config(fail_fast=False, required_sections=("profile",)))
        self.assertEqual(job.status, JobStatus.FAILED)

    def test_warning_recorded_for_failed_section(self):
        job = _job()
        connector = FakeConnector(fail_sections={ConnectorSection.GRID})
        session = run_job(job, connector, _config(fail_fast=False))
        self.assertTrue(any("grid" in w for w in session.warnings))

    def test_optional_section_failure_does_not_block_completion(self):
        job = _job()
        connector = FakeConnector(fail_sections={ConnectorSection.HIGHLIGHTS})
        run_job(job, connector, _config(fail_fast=False, required_sections=("profile",)))
        self.assertEqual(job.status, JobStatus.COMPLETE)


class PlannerSectionOrderTests(unittest.TestCase):
    def test_section_order_is_deterministic_regardless_of_requested_order(self):
        planner = ResearchPlanner()
        job_a = _job(requested_sections=(ConnectorSection.CAPTIONS, ConnectorSection.PROFILE))
        job_b = _job(requested_sections=(ConnectorSection.PROFILE, ConnectorSection.CAPTIONS))
        self.assertEqual(planner.section_order(job_a), planner.section_order(job_b))

    def test_section_order_groups_by_state(self):
        planner = ResearchPlanner()
        job = _job(requested_sections=(ConnectorSection.POSTS, ConnectorSection.PROFILE, ConnectorSection.GRID))
        order = planner.section_order(job)
        self.assertEqual(order[0], ConnectorSection.PROFILE)
        self.assertIn(ConnectorSection.GRID, order[1:])
        self.assertIn(ConnectorSection.POSTS, order[1:])

    def test_sections_for_state_empty_when_not_requested(self):
        planner = ResearchPlanner()
        job = _job(requested_sections=(ConnectorSection.PROFILE,))
        self.assertEqual(planner.sections_for_state(job, JobStatus.REELS), ())


class ResumePointTests(unittest.TestCase):
    def test_fresh_progress_resumes_at_profile(self):
        planner = ResearchPlanner()
        job = _job()
        progress = ResearchProgress(total_steps=len(JobStatus.WORKING_STATES))
        self.assertEqual(planner.resume_point(job, progress), JobStatus.PROFILE)

    def test_partial_progress_resumes_at_first_incomplete_state(self):
        planner = ResearchPlanner()
        job = _job()
        progress = ResearchProgress(total_steps=len(JobStatus.WORKING_STATES))
        progress.mark_step_completed(JobStatus.PROFILE)
        progress.mark_step_completed(JobStatus.GRID)
        self.assertEqual(planner.resume_point(job, progress), JobStatus.CAPTIONS)

    def test_fully_completed_progress_resumes_at_complete(self):
        planner = ResearchPlanner()
        job = _job()
        progress = ResearchProgress(total_steps=len(JobStatus.WORKING_STATES))
        for state in JobStatus.WORKING_STATES:
            progress.mark_step_completed(state)
        self.assertEqual(planner.resume_point(job, progress), JobStatus.COMPLETE)

    def test_resuming_a_run_job_session_continues_from_where_it_stopped(self):
        # Simulate "resume" by running once with a fail_fast abort,
        # then running again with a connector that succeeds -- the
        # second run's own progress starts fresh (this phase does not
        # persist ResearchProgress across process runs), but the
        # planner's resume_point utility itself is exercised directly
        # against the first run's leftover progress object.
        planner = ResearchPlanner()
        job = _job()
        connector = FakeConnector(fail_sections={ConnectorSection.CAPTIONS})
        session = run_job(job, connector, _config(fail_fast=True))
        self.assertEqual(job.status, JobStatus.FAILED)
        resume_state = planner.resume_point(job, session.progress)
        self.assertEqual(resume_state, JobStatus.CAPTIONS)


class SectionStateMapTests(unittest.TestCase):
    def test_covers_all_ten_connector_sections(self):
        self.assertEqual(set(SECTION_STATE_MAP.keys()), set(ConnectorSection.ALL))

    def test_every_mapped_state_is_a_working_state(self):
        for state in SECTION_STATE_MAP.values():
            self.assertIn(state, JobStatus.WORKING_STATES)

    def test_posts_maps_to_grid(self):
        self.assertEqual(SECTION_STATE_MAP[ConnectorSection.POSTS], JobStatus.GRID)

    def test_relationships_maps_to_visual(self):
        self.assertEqual(SECTION_STATE_MAP[ConnectorSection.RELATIONSHIPS], JobStatus.VISUAL)


if __name__ == "__main__":
    unittest.main()
