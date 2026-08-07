import unittest

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_research.evidence_queue import EvidenceQueue
from src.creator_research.interfaces import ConnectorSection, EvidenceBundle
from src.creator_research.jobs import ResearchJob
from src.creator_research.progress import ResearchProgress
from src.creator_research.report import build_report


def _evidence(n=1):
    return [Evidence(evidence_type=EvidenceType.OPERATOR_OBSERVATION, source_description=f"item {i}") for i in range(n)]


def _job(**overrides):
    defaults = dict(
        creator_id="c1", platform="instagram", username="demo", profile_url="https://x/demo", connector_name="dummy"
    )
    defaults.update(overrides)
    return ResearchJob(**defaults)


class ResearchProgressTests(unittest.TestCase):
    def test_zero_total_steps_percentage_is_zero(self):
        progress = ResearchProgress(total_steps=0)
        self.assertEqual(progress.percentage, 0.0)

    def test_zero_completed_percentage_is_zero(self):
        progress = ResearchProgress(total_steps=8)
        self.assertEqual(progress.percentage, 0.0)

    def test_partial_completion_percentage(self):
        progress = ResearchProgress(total_steps=8)
        progress.mark_step_completed("profile")
        progress.mark_step_completed("grid")
        self.assertAlmostEqual(progress.percentage, 0.25)

    def test_full_completion_percentage_is_one(self):
        progress = ResearchProgress(total_steps=2)
        progress.mark_step_completed("profile")
        progress.mark_step_completed("grid")
        self.assertEqual(progress.percentage, 1.0)

    def test_percentage_never_exceeds_one(self):
        progress = ResearchProgress(total_steps=1)
        progress.mark_step_completed("profile")
        progress.mark_step_completed("profile")  # duplicate, should not double count
        self.assertEqual(progress.percentage, 1.0)
        self.assertEqual(len(progress.completed_steps), 1)

    def test_estimated_remaining_steps(self):
        progress = ResearchProgress(total_steps=5)
        progress.mark_step_completed("profile")
        progress.mark_step_completed("grid")
        self.assertEqual(progress.estimated_remaining_steps, 3)

    def test_estimated_remaining_steps_never_negative(self):
        progress = ResearchProgress(total_steps=1)
        progress.mark_step_completed("profile")
        progress.mark_step_completed("grid")
        self.assertEqual(progress.estimated_remaining_steps, 0)

    def test_mark_step_started_sets_current_step(self):
        progress = ResearchProgress(total_steps=1)
        progress.mark_step_started("profile")
        self.assertEqual(progress.current_step, "profile")

    def test_mark_step_completed_clears_current_step_if_matching(self):
        progress = ResearchProgress(total_steps=1)
        progress.mark_step_started("profile")
        progress.mark_step_completed("profile")
        self.assertEqual(progress.current_step, "")

    def test_add_warning_appends(self):
        progress = ResearchProgress(total_steps=1)
        progress.add_warning("something happened")
        self.assertEqual(progress.warnings, ["something happened"])


class BuildReportTests(unittest.TestCase):
    def test_coverage_reflects_evidence_queue(self):
        job = _job(requested_sections=(ConnectorSection.PROFILE, ConnectorSection.GRID))
        progress = ResearchProgress(total_steps=2)
        queue = EvidenceQueue()
        queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence(1)))
        queue.add(EvidenceBundle(section=ConnectorSection.GRID, items=_evidence(3)))
        report = build_report(job, progress, queue)
        self.assertEqual(report.coverage[ConnectorSection.PROFILE], 1)
        self.assertEqual(report.coverage[ConnectorSection.GRID], 3)

    def test_missing_sections_flags_zero_coverage_requested_sections(self):
        job = _job(requested_sections=(ConnectorSection.PROFILE, ConnectorSection.CAPTIONS))
        progress = ResearchProgress(total_steps=2)
        queue = EvidenceQueue()
        queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence(1)))
        report = build_report(job, progress, queue)
        self.assertIn(ConnectorSection.CAPTIONS, report.missing_sections)
        self.assertNotIn(ConnectorSection.PROFILE, report.missing_sections)

    def test_sections_completed_only_includes_requested_with_coverage(self):
        job = _job(requested_sections=(ConnectorSection.PROFILE, ConnectorSection.CAPTIONS))
        progress = ResearchProgress(total_steps=2)
        queue = EvidenceQueue()
        queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence(1)))
        report = build_report(job, progress, queue)
        self.assertEqual(report.sections_completed, [ConnectorSection.PROFILE])

    def test_sections_failed_excluded_from_missing_sections(self):
        job = _job(requested_sections=(ConnectorSection.PROFILE, ConnectorSection.GRID))
        progress = ResearchProgress(total_steps=2)
        queue = EvidenceQueue()
        queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, items=_evidence(1)))
        report = build_report(job, progress, queue, sections_failed=[ConnectorSection.GRID])
        self.assertNotIn(ConnectorSection.GRID, report.missing_sections)
        self.assertIn(ConnectorSection.GRID, report.sections_failed)

    def test_warnings_combine_progress_and_evidence_queue(self):
        job = _job()
        progress = ResearchProgress(total_steps=1)
        progress.add_warning("progress warning")
        queue = EvidenceQueue()
        queue.add(EvidenceBundle(section=ConnectorSection.PROFILE, warnings=["bundle warning"]))
        report = build_report(job, progress, queue)
        self.assertIn("progress warning", report.warnings)
        self.assertIn("bundle warning", report.warnings)

    def test_duration_seconds_computed_from_job_timestamps(self):
        job = _job(created_at="2026-01-01T00:00:00+00:00", updated_at="2026-01-01T00:01:00+00:00")
        progress = ResearchProgress(total_steps=1)
        queue = EvidenceQueue()
        report = build_report(job, progress, queue)
        self.assertEqual(report.duration_seconds, 60.0)

    def test_duration_seconds_never_negative(self):
        job = _job(created_at="2026-01-02T00:00:00+00:00", updated_at="2026-01-01T00:00:00+00:00")
        progress = ResearchProgress(total_steps=1)
        queue = EvidenceQueue()
        report = build_report(job, progress, queue)
        self.assertEqual(report.duration_seconds, 0.0)

    def test_malformed_timestamps_do_not_raise(self):
        job = _job(created_at="not-a-date", updated_at="also-not-a-date")
        progress = ResearchProgress(total_steps=1)
        queue = EvidenceQueue()
        report = build_report(job, progress, queue)
        self.assertEqual(report.duration_seconds, 0.0)

    def test_job_id_and_creator_id_preserved(self):
        job = _job()
        progress = ResearchProgress(total_steps=1)
        queue = EvidenceQueue()
        report = build_report(job, progress, queue)
        self.assertEqual(report.job_id, job.job_id)
        self.assertEqual(report.creator_id, job.creator_id)

    def test_summary_is_non_empty_string(self):
        job = _job()
        progress = ResearchProgress(total_steps=1)
        queue = EvidenceQueue()
        report = build_report(job, progress, queue)
        self.assertTrue(report.summary)

    def test_build_report_performs_no_io(self):
        # Pure-function guarantee: no file/network access anywhere in
        # report.py's build_report().
        import inspect

        from src.creator_research import report as report_module

        source = inspect.getsource(report_module)
        for forbidden in ("open(", "requests.", "urllib."):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
