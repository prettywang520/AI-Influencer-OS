import unittest

from src.creator_research.exceptions import InvalidTransitionError
from src.creator_research.interfaces import ConnectorSection
from src.creator_research.jobs import ResearchJob, compute_job_id
from src.creator_research.state import ALLOWED_TRANSITIONS, JobStatus, validate_transition


def _job(**overrides):
    defaults = dict(
        creator_id="c1", platform="instagram", username="demo", profile_url="https://x/demo", connector_name="dummy"
    )
    defaults.update(overrides)
    return ResearchJob(**defaults)


class ComputeJobIdTests(unittest.TestCase):
    def test_deterministic_for_identical_inputs(self):
        a = compute_job_id("c1", "instagram", "demo", ("profile",), "dummy")
        b = compute_job_id("c1", "instagram", "demo", ("profile",), "dummy")
        self.assertEqual(a, b)

    def test_section_order_does_not_affect_id(self):
        a = compute_job_id("c1", "instagram", "demo", ("profile", "grid"), "dummy")
        b = compute_job_id("c1", "instagram", "demo", ("grid", "profile"), "dummy")
        self.assertEqual(a, b)

    def test_different_creator_id_changes_id(self):
        a = compute_job_id("c1", "instagram", "demo", ("profile",), "dummy")
        b = compute_job_id("c2", "instagram", "demo", ("profile",), "dummy")
        self.assertNotEqual(a, b)

    def test_different_sections_changes_id(self):
        a = compute_job_id("c1", "instagram", "demo", ("profile",), "dummy")
        b = compute_job_id("c1", "instagram", "demo", ("profile", "grid"), "dummy")
        self.assertNotEqual(a, b)

    def test_different_connector_name_changes_id(self):
        a = compute_job_id("c1", "instagram", "demo", ("profile",), "dummy")
        b = compute_job_id("c1", "instagram", "demo", ("profile",), "other")
        self.assertNotEqual(a, b)


class ResearchJobConstructionTests(unittest.TestCase):
    def test_job_id_matches_compute_job_id(self):
        job = _job(requested_sections=("profile", "grid"))
        expected = compute_job_id("c1", "instagram", "demo", ("profile", "grid"), "dummy")
        self.assertEqual(job.job_id, expected)

    def test_default_status_is_queued(self):
        job = _job()
        self.assertEqual(job.status, JobStatus.QUEUED)

    def test_default_requested_sections_is_all(self):
        job = _job()
        self.assertEqual(job.requested_sections, ConnectorSection.ALL)

    def test_job_id_excludes_created_at(self):
        a = _job(created_at="2026-01-01T00:00:00+00:00")
        b = _job(created_at="2026-12-31T00:00:00+00:00")
        self.assertEqual(a.job_id, b.job_id)

    def test_job_id_excludes_priority_and_metadata(self):
        a = _job(priority=1, metadata={"x": 1})
        b = _job(priority=9, metadata={"y": 2})
        self.assertEqual(a.job_id, b.job_id)

    def test_default_metadata_not_shared_between_instances(self):
        a = _job()
        b = _job(username="other")
        a.metadata["x"] = 1
        self.assertEqual(b.metadata, {})


class AdvanceTests(unittest.TestCase):
    def test_valid_forward_transition_updates_status(self):
        job = _job()
        job.advance(JobStatus.STARTING)
        self.assertEqual(job.status, JobStatus.STARTING)

    def test_advance_bumps_updated_at(self):
        job = _job(updated_at="2020-01-01T00:00:00+00:00")
        job.advance(JobStatus.STARTING)
        self.assertNotEqual(job.updated_at, "2020-01-01T00:00:00+00:00")

    def test_invalid_transition_raises_and_does_not_change_status(self):
        job = _job()
        with self.assertRaises(InvalidTransitionError):
            job.advance(JobStatus.COMPLETE)
        self.assertEqual(job.status, JobStatus.QUEUED)

    def test_failed_reachable_from_any_non_terminal_state(self):
        for status in JobStatus.ORDER:
            if status in JobStatus.TERMINAL:
                continue
            job = _job()
            job.status = status
            job.advance(JobStatus.FAILED)
            self.assertEqual(job.status, JobStatus.FAILED)


class DictRoundTripTests(unittest.TestCase):
    def test_to_dict_contains_job_id(self):
        job = _job()
        payload = job.to_dict()
        self.assertEqual(payload["job_id"], job.job_id)

    def test_from_dict_reconstructs_equal_job_id(self):
        job = _job(requested_sections=("profile", "grid"))
        payload = job.to_dict()
        rebuilt = ResearchJob.from_dict(payload)
        self.assertEqual(rebuilt.job_id, job.job_id)

    def test_from_dict_restores_status_without_validating_transition(self):
        job = _job()
        job.status = JobStatus.COMPLETE  # direct set, bypassing advance() on purpose for this test
        payload = job.to_dict()
        rebuilt = ResearchJob.from_dict(payload)
        self.assertEqual(rebuilt.status, JobStatus.COMPLETE)

    def test_from_dict_restores_metadata(self):
        job = _job(metadata={"paused": True})
        rebuilt = ResearchJob.from_dict(job.to_dict())
        self.assertEqual(rebuilt.metadata, {"paused": True})


class StateTransitionTableTests(unittest.TestCase):
    def test_every_working_state_can_advance_to_failed(self):
        for status in JobStatus.WORKING_STATES:
            self.assertIn(JobStatus.FAILED, ALLOWED_TRANSITIONS[status])

    def test_complete_has_no_outgoing_transitions(self):
        self.assertEqual(ALLOWED_TRANSITIONS[JobStatus.COMPLETE], frozenset())

    def test_failed_has_no_outgoing_transitions(self):
        self.assertEqual(ALLOWED_TRANSITIONS[JobStatus.FAILED], frozenset())

    def test_queued_can_only_advance_to_starting_or_failed(self):
        self.assertEqual(ALLOWED_TRANSITIONS[JobStatus.QUEUED], frozenset({JobStatus.STARTING, JobStatus.FAILED}))

    def test_every_consecutive_pair_in_order_is_a_valid_transition(self):
        for current, target in zip(JobStatus.ORDER, JobStatus.ORDER[1:]):
            validate_transition(current, target)  # must not raise

    def test_skipping_a_state_is_invalid(self):
        with self.assertRaises(InvalidTransitionError):
            validate_transition(JobStatus.PROFILE, JobStatus.CAPTIONS)

    def test_backward_transition_is_invalid(self):
        with self.assertRaises(InvalidTransitionError):
            validate_transition(JobStatus.GRID, JobStatus.PROFILE)

    def test_unrecognized_current_status_raises(self):
        with self.assertRaises(InvalidTransitionError):
            validate_transition("not_a_real_status", JobStatus.STARTING)

    def test_unrecognized_target_status_raises(self):
        with self.assertRaises(InvalidTransitionError):
            validate_transition(JobStatus.QUEUED, "not_a_real_status")

    def test_all_contains_every_working_state_and_terminals(self):
        self.assertEqual(set(JobStatus.ALL), set(JobStatus.ORDER) | {JobStatus.FAILED})


if __name__ == "__main__":
    unittest.main()
