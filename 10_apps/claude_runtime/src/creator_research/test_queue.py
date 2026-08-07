import json
import tempfile
import unittest
from pathlib import Path

from src.creator_research.exceptions import JobNotFoundError, QueueError, RetryLimitExceededError
from src.creator_research.jobs import ResearchJob
from src.creator_research.queue import (
    QueueConfig,
    ResearchQueue,
    default_queue_config_path,
    load_queue_config,
    validate_queue,
)
from src.creator_research.state import JobStatus


def _job(**overrides):
    defaults = dict(
        creator_id="c1", platform="instagram", username="demo", profile_url="https://x/demo", connector_name="dummy"
    )
    defaults.update(overrides)
    return ResearchJob(**defaults)


class QueueTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.queue_path = self.tmp_path / "queue.json"
        self.queue = ResearchQueue(self.queue_path)

    def tearDown(self):
        self._tmp.cleanup()


class LoadQueueConfigTests(unittest.TestCase):
    def test_loads_real_default_config(self):
        config = load_queue_config()
        self.assertEqual(config.schema_version, "1.0")

    def test_default_path_exists_on_disk(self):
        self.assertTrue(default_queue_config_path().is_file())

    def test_resolved_default_queue_path_is_absolute(self):
        config = load_queue_config()
        self.assertTrue(config.resolved_default_queue_path().is_absolute())

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.yaml"
            with self.assertRaises(Exception):
                load_queue_config(missing)


class EnqueueTests(QueueTempTestCase):
    def test_enqueue_creates_file(self):
        self.queue.enqueue(_job())
        self.assertTrue(self.queue_path.is_file())

    def test_enqueue_is_idempotent_for_same_job_id(self):
        job = _job()
        first = self.queue.enqueue(job)
        second = self.queue.enqueue(_job())  # logically identical -> same job_id
        self.assertEqual(first.job_id, second.job_id)
        self.assertEqual(len(self.queue.list_jobs()), 1)

    def test_enqueue_different_jobs_both_stored(self):
        self.queue.enqueue(_job(username="a"))
        self.queue.enqueue(_job(username="b"))
        self.assertEqual(len(self.queue.list_jobs()), 2)


class DequeueOrderingTests(QueueTempTestCase):
    def test_highest_priority_dequeued_first(self):
        self.queue.enqueue(_job(username="low", priority=1))
        self.queue.enqueue(_job(username="high", priority=9))
        dequeued = self.queue.dequeue()
        self.assertEqual(dequeued.username, "high")

    def test_ties_broken_by_created_at_then_job_id(self):
        job_a = _job(username="a", priority=1, created_at="2026-01-01T00:00:00+00:00")
        job_b = _job(username="b", priority=1, created_at="2026-01-02T00:00:00+00:00")
        self.queue.enqueue(job_b)
        self.queue.enqueue(job_a)
        dequeued = self.queue.dequeue()
        self.assertEqual(dequeued.username, "a")  # earlier created_at wins

    def test_dequeue_advances_status_to_starting(self):
        self.queue.enqueue(_job())
        dequeued = self.queue.dequeue()
        self.assertEqual(dequeued.status, JobStatus.STARTING)
        self.assertEqual(self.queue.get(dequeued.job_id).status, JobStatus.STARTING)

    def test_dequeue_skips_paused_jobs(self):
        job = self.queue.enqueue(_job())
        self.queue.pause(job.job_id)
        self.assertIsNone(self.queue.dequeue())

    def test_dequeue_empty_queue_returns_none(self):
        self.assertIsNone(self.queue.dequeue())

    def test_dequeue_only_considers_queued_jobs(self):
        job = self.queue.enqueue(_job())
        self.queue.dequeue()  # now STARTING
        self.assertIsNone(self.queue.dequeue())

    def test_list_jobs_deterministic_order(self):
        self.queue.enqueue(_job(username="a", priority=1))
        self.queue.enqueue(_job(username="b", priority=5))
        self.queue.enqueue(_job(username="c", priority=3))
        usernames = [j.username for j in self.queue.list_jobs()]
        self.assertEqual(usernames, ["b", "c", "a"])


class GetAndListTests(QueueTempTestCase):
    def test_get_unknown_job_raises(self):
        with self.assertRaises(JobNotFoundError):
            self.queue.get("does_not_exist")

    def test_list_jobs_filters_by_status(self):
        job = self.queue.enqueue(_job())
        self.queue.dequeue()
        self.assertEqual(len(self.queue.list_jobs(status=JobStatus.STARTING)), 1)
        self.assertEqual(len(self.queue.list_jobs(status=JobStatus.QUEUED)), 0)


class PauseResumeTests(QueueTempTestCase):
    def test_pause_sets_metadata_flag(self):
        job = self.queue.enqueue(_job())
        paused = self.queue.pause(job.job_id)
        self.assertTrue(paused.metadata["paused"])
        self.assertEqual(paused.status, JobStatus.QUEUED)  # status unchanged

    def test_resume_clears_metadata_flag(self):
        job = self.queue.enqueue(_job())
        self.queue.pause(job.job_id)
        resumed = self.queue.resume(job.job_id)
        self.assertFalse(resumed.metadata["paused"])

    def test_pause_unknown_job_raises(self):
        with self.assertRaises(JobNotFoundError):
            self.queue.pause("does_not_exist")


class CancelTests(QueueTempTestCase):
    def test_cancel_sets_status_failed(self):
        job = self.queue.enqueue(_job())
        cancelled = self.queue.cancel(job.job_id)
        self.assertEqual(cancelled.status, JobStatus.FAILED)
        self.assertTrue(cancelled.metadata["cancelled"])

    def test_cancel_already_terminal_job_does_not_raise(self):
        job = self.queue.enqueue(_job())
        self.queue.cancel(job.job_id)
        cancelled_again = self.queue.cancel(job.job_id)
        self.assertEqual(cancelled_again.status, JobStatus.FAILED)


class RetryTests(QueueTempTestCase):
    def test_retry_resets_failed_job_to_queued(self):
        job = self.queue.enqueue(_job())
        self.queue.cancel(job.job_id)
        retried = self.queue.retry(job.job_id, max_attempts=3)
        self.assertEqual(retried.status, JobStatus.QUEUED)
        self.assertEqual(retried.metadata["retry_count"], 1)

    def test_retry_non_failed_job_raises(self):
        job = self.queue.enqueue(_job())
        with self.assertRaises(QueueError):
            self.queue.retry(job.job_id)

    def test_retry_exceeding_max_attempts_raises(self):
        job = self.queue.enqueue(_job())
        self.queue.cancel(job.job_id)
        self.queue.retry(job.job_id, max_attempts=1)
        self.queue.cancel(job.job_id)
        with self.assertRaises(RetryLimitExceededError):
            self.queue.retry(job.job_id, max_attempts=1)


class AtomicityAndPersistenceTests(QueueTempTestCase):
    def test_no_temp_file_left_behind_after_enqueue(self):
        self.queue.enqueue(_job())
        remaining = list(self.tmp_path.iterdir())
        self.assertEqual(remaining, [self.queue_path])

    def test_fresh_instance_sees_persisted_state(self):
        job = self.queue.enqueue(_job())
        second_instance = ResearchQueue(self.queue_path)
        self.assertEqual(second_instance.get(job.job_id).job_id, job.job_id)

    def test_file_contains_valid_json_list(self):
        self.queue.enqueue(_job())
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        self.assertIsInstance(raw, list)

    def test_malformed_queue_file_raises_on_read(self):
        self.queue_path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(QueueError):
            self.queue.list_jobs()


class ValidateQueueTests(QueueTempTestCase):
    def test_valid_queue_passes(self):
        self.queue.enqueue(_job())
        passed, errors = validate_queue(self.queue)
        self.assertTrue(passed)
        self.assertEqual(errors, [])

    def test_empty_queue_passes(self):
        passed, errors = validate_queue(self.queue)
        self.assertTrue(passed)

    def test_tampered_job_id_fails(self):
        self.queue.enqueue(_job())
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        raw[0]["job_id"] = "0" * 16
        self.queue_path.write_text(json.dumps(raw), encoding="utf-8")
        passed, errors = validate_queue(self.queue)
        self.assertFalse(passed)
        self.assertTrue(any("job_id mismatch" in e for e in errors))

    def test_unrecognized_status_fails(self):
        self.queue.enqueue(_job())
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        raw[0]["status"] = "not_a_real_status"
        self.queue_path.write_text(json.dumps(raw), encoding="utf-8")
        passed, errors = validate_queue(self.queue)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
