from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import publish_recovery
from .history_service import HistoryService
from .models import PublisherConfig, PublishJob, RecoveryConfig, app_root
from .publish_recovery import (
    AuditService,
    AuditServiceError,
    ConflictingPublishedRecordError,
    InvalidInstagramUrlError,
    LockNotFoundError,
    LockNotStaleError,
    LockOwnerActiveError,
    RecoveryConfirmationRequiredError,
    RecoveryIneligibleError,
    clear_stale_lock,
    inspect_job,
    list_stuck_jobs,
    mark_failed,
    mark_published,
    parse_instagram_post_id,
)
from .queue_service import PublishQueueError, QueueService

MODULE_PATH = Path(__file__).resolve().parent / "publish_recovery.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")

JOB_ID = "2026-08-01-instagram_feed"


def _make_config(temp_dir: Path, **recovery_overrides) -> PublisherConfig:
    recovery_kwargs = dict(
        publishing_stuck_after_minutes=15,
        lock_stale_after_minutes=30,
        audit_dir=str(temp_dir / "audit"),
        require_confirm_for_mutations=True,
        allowed_instagram_url_patterns=(
            "https://www.instagram.com/p/",
            "https://www.instagram.com/reel/",
        ),
    )
    recovery_kwargs.update(recovery_overrides)

    return PublisherConfig(
        paths={
            "queue_dir": str(temp_dir / "queue"),
            "queue_filename": "publish_queue.json",
            "history_dir": str(temp_dir / "history"),
            "logs_dir": str(temp_dir / "logs"),
            "screenshots_dir": str(temp_dir / "screenshots"),
            "locks_dir": str(temp_dir / "locks"),
        },
        recovery=RecoveryConfig(**recovery_kwargs),
    )


def _job(**overrides) -> PublishJob:
    defaults = dict(
        job_id=JOB_ID,
        production_date="2026-08-01",
        persona_id="aiko",
        platform="instagram",
        content_type="instagram_feed",
        status="approved",
        approved_at="2026-08-03T00:00:00+00:00",
    )
    defaults.update(overrides)
    return PublishJob(**defaults)


def _dead_pid() -> int:
    """A pid guaranteed not to belong to any running process."""
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def _write_lock(
    locks_dir: Path,
    job_id: str,
    *,
    pid: int,
    age_minutes: float = 0.0,
    command: str = "--publish-approved",
) -> Path:
    locks_dir.mkdir(parents=True, exist_ok=True)
    started_at = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
    path = locks_dir / f"{job_id}.lock.json"
    path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "pid": pid,
                "started_at": started_at,
                "command": command,
                "hostname": "test-host",
            }
        ),
        encoding="utf-8",
    )
    return path


class RecoveryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)

        self.config = _make_config(self.temp_dir)
        self.queue_service = QueueService(self.config.queue_path())
        self.history_service = HistoryService(self.config.history_dir())
        self.audit_service = AuditService(self.config.audit_dir())
        self.locks_dir = self.config.locks_dir()


# ---------------------------------------------------------------------------
# parse_instagram_post_id / is_allowed_instagram_url
# ---------------------------------------------------------------------------


class UrlHelperTests(unittest.TestCase):
    def test_parses_post_id_from_p_url(self) -> None:
        self.assertEqual(
            parse_instagram_post_id("https://www.instagram.com/p/ABC123xyz/"), "ABC123xyz"
        )

    def test_parses_post_id_from_reel_url(self) -> None:
        self.assertEqual(
            parse_instagram_post_id("https://www.instagram.com/reel/XYZ789/"), "XYZ789"
        )

    def test_unparsable_url_returns_none(self) -> None:
        self.assertIsNone(parse_instagram_post_id("https://www.instagram.com/aikotraveldiary/"))


# ---------------------------------------------------------------------------
# Inspect (read-only)
# ---------------------------------------------------------------------------


class InspectTests(RecoveryFixture):
    def test_missing_job_reports_not_found(self) -> None:
        report = inspect_job("does-not-exist", self.config)
        self.assertFalse(report.found)

    def test_inspect_does_not_modify_queue_history_or_lock(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)
        _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=60)
        before_queue = self.config.queue_path().read_bytes()

        inspect_job(job.job_id, self.config)

        self.assertEqual(self.config.queue_path().read_bytes(), before_queue)
        self.assertEqual(self.history_service.list_events("2026-08-01"), [])
        self.assertTrue((self.locks_dir / f"{job.job_id}.lock.json").exists())

    def test_inspect_reports_lock_details(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)
        _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=45)

        report = inspect_job(job.job_id, self.config)

        self.assertTrue(report.lock.exists)
        self.assertFalse(report.lock.pid_active)
        self.assertGreaterEqual(report.lock.age_minutes, 44)

    def test_inspect_includes_recent_history_events(self) -> None:
        job = _job()
        self.queue_service.add_job(job)
        self.history_service.record_approved(production_date="2026-08-01", job_id=job.job_id)

        report = inspect_job(job.job_id, self.config)

        self.assertEqual(len(report.recent_history_events), 1)
        self.assertEqual(report.recent_history_events[0]["event_type"], "approved")

    def test_inspect_never_opens_instagram_or_uses_playwright(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip().lower()
            self.assertFalse(stripped.startswith("import playwright"))
            self.assertFalse(stripped.startswith("from playwright"))
        self.assertNotIn("InstagramSession(", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# --list-stuck (read-only)
# ---------------------------------------------------------------------------


class ListStuckTests(RecoveryFixture):
    def test_old_publishing_job_is_stuck(self) -> None:
        job = _job(status="publishing")
        job.updated_at = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        self.queue_service.save_jobs([job])

        reports = list_stuck_jobs(self.config)

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].job_id, job.job_id)
        self.assertTrue(any("publishing" in reason for reason in reports[0].reasons))

    def test_recent_publishing_job_is_not_stuck(self) -> None:
        job = _job(status="publishing")
        self.queue_service.save_jobs([job])

        reports = list_stuck_jobs(self.config)

        self.assertEqual(reports, [])

    def test_failed_manual_check_required_job_is_stuck(self) -> None:
        job = _job(
            status="failed",
            error="publish_verification_inconclusive_manual_check_required",
        )
        self.queue_service.save_jobs([job])

        reports = list_stuck_jobs(self.config)

        self.assertEqual(len(reports), 1)
        self.assertTrue(any("manual-check-required" in reason for reason in reports[0].reasons))

    def test_inconsistent_published_record_is_stuck(self) -> None:
        job = _job(status="published", published_at=None, platform_url=None)
        self.queue_service.save_jobs([job])

        reports = list_stuck_jobs(self.config)

        self.assertEqual(len(reports), 1)
        self.assertTrue(any("published" in reason for reason in reports[0].reasons))

    def test_lock_with_no_active_process_is_stuck(self) -> None:
        job = _job(status="approved")
        self.queue_service.save_jobs([job])
        _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=1)

        reports = list_stuck_jobs(self.config)

        self.assertEqual(len(reports), 1)
        self.assertTrue(any("no matching active process" in reason for reason in reports[0].reasons))

    def test_lock_with_active_process_is_not_flagged_as_stuck(self) -> None:
        job = _job(status="approved")
        self.queue_service.save_jobs([job])
        _write_lock(self.locks_dir, job.job_id, pid=os.getpid(), age_minutes=1)

        reports = list_stuck_jobs(self.config)

        self.assertEqual(reports, [])

    def test_queue_and_history_disagreement_is_stuck(self) -> None:
        job = _job(status="approved")
        self.queue_service.save_jobs([job])
        self.history_service.record_event(
            production_date="2026-08-01",
            job_id=job.job_id,
            event_type="scheduled",
            previous_status="approved",
            new_status="scheduled",
        )

        reports = list_stuck_jobs(self.config)

        self.assertEqual(len(reports), 1)
        self.assertTrue(any("disagrees" in reason for reason in reports[0].reasons))

    def test_list_stuck_does_not_modify_anything(self) -> None:
        job = _job(status="publishing")
        job.updated_at = (datetime.now(timezone.utc) - timedelta(minutes=99)).isoformat()
        self.queue_service.save_jobs([job])
        before = self.config.queue_path().read_bytes()

        list_stuck_jobs(self.config)

        self.assertEqual(self.config.queue_path().read_bytes(), before)


# ---------------------------------------------------------------------------
# --mark-published
# ---------------------------------------------------------------------------


class MarkPublishedTests(RecoveryFixture):
    def _publishing_job(self, **overrides) -> PublishJob:
        job = _job(status="publishing", **overrides)
        self.queue_service.add_job(job)
        return job

    def _mark(self, job_id: str, **kwargs):
        defaults = dict(
            confirm=True,
            config=self.config,
            queue_service=self.queue_service,
            history_service=self.history_service,
            audit_service=self.audit_service,
            operator="tester",
        )
        defaults.update(kwargs)
        return mark_published(job_id, **defaults)

    def test_requires_confirm(self) -> None:
        job = self._publishing_job()
        with self.assertRaises(RecoveryConfirmationRequiredError):
            self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC/", confirm=False)

    def test_only_accepts_publishing_or_failed(self) -> None:
        job = _job(status="approved")
        self.queue_service.add_job(job)
        with self.assertRaises(RecoveryIneligibleError):
            self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC/")

    def test_invalid_url_rejected(self) -> None:
        job = self._publishing_job()
        with self.assertRaises(InvalidInstagramUrlError):
            self._mark(job.job_id, platform_url="https://example.com/not-instagram")

    def test_empty_url_rejected(self) -> None:
        job = self._publishing_job()
        with self.assertRaises(InvalidInstagramUrlError):
            self._mark(job.job_id, platform_url="")

    def test_valid_post_url_accepted(self) -> None:
        job = self._publishing_job()
        updated = self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC123/")
        self.assertEqual(updated.status, "published")

    def test_valid_reel_url_accepted_when_policy_allows(self) -> None:
        job = self._publishing_job()
        updated = self._mark(job.job_id, platform_url="https://www.instagram.com/reel/XYZ789/")
        self.assertEqual(updated.platform_post_id, "XYZ789")

    def test_reel_url_rejected_when_policy_disallows(self) -> None:
        restrictive_config = _make_config(
            self.temp_dir, allowed_instagram_url_patterns=("https://www.instagram.com/p/",)
        )
        job = self._publishing_job()
        with self.assertRaises(InvalidInstagramUrlError):
            self._mark(
                job.job_id,
                platform_url="https://www.instagram.com/reel/XYZ789/",
                config=restrictive_config,
            )

    def test_platform_post_id_parsed_when_possible(self) -> None:
        job = self._publishing_job()
        updated = self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC123xyz/")
        self.assertEqual(updated.platform_post_id, "ABC123xyz")

    def test_updates_queue_correctly(self) -> None:
        job = self._publishing_job(error="previous error")
        updated = self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC123/")

        self.assertEqual(updated.status, "published")
        self.assertIsNotNone(updated.published_at)
        self.assertEqual(updated.platform_url, "https://www.instagram.com/p/ABC123/")
        self.assertIsNone(updated.error)
        self.assertEqual(self.queue_service.get(job.job_id).status, "published")

    def test_writes_manual_reconciled_published_history(self) -> None:
        job = self._publishing_job()
        self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC123/")

        events = self.history_service.list_events("2026-08-01")
        self.assertEqual([event["event_type"] for event in events], ["manual_reconciled_published"])
        self.assertEqual(events[0]["previous_status"], "publishing")
        self.assertEqual(events[0]["new_status"], "published")

    def test_writes_redacted_audit_snapshot(self) -> None:
        job = self._publishing_job(caption_text="secret caption text", hashtags=["#a", "#b"])
        self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC123/")

        records = self.audit_service.list_records("2026-08-01")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["result"], "success")

        snapshot = record["queue_snapshot_after"]
        self.assertEqual(
            set(snapshot.keys()),
            {
                "job_id",
                "status",
                "approved_at",
                "published_at",
                "platform_post_id",
                "platform_url",
                "error",
                "updated_at",
            },
        )

        serialized = json.dumps(record)
        self.assertNotIn("secret caption text", serialized)
        self.assertNotIn("#a", serialized)

    def test_conflicting_published_record_refused(self) -> None:
        conflicting = _job(job_id="other-job", status="published", platform_post_id="ABC123")
        self.queue_service.add_job(conflicting)
        job = self._publishing_job()

        with self.assertRaises(ConflictingPublishedRecordError):
            self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC123/")

    def test_content_type_must_be_instagram_feed(self) -> None:
        job = _job(status="publishing", content_type="instagram_story")
        self.queue_service.add_job(job)
        with self.assertRaises(RecoveryIneligibleError):
            self._mark(job.job_id, platform_url="https://www.instagram.com/p/ABC123/")


# ---------------------------------------------------------------------------
# --mark-failed
# ---------------------------------------------------------------------------


class MarkFailedTests(RecoveryFixture):
    def _publishing_job(self, **overrides) -> PublishJob:
        job = _job(status="publishing", **overrides)
        self.queue_service.add_job(job)
        return job

    def _mark(self, job_id: str, **kwargs):
        defaults = dict(
            confirm=True,
            config=self.config,
            queue_service=self.queue_service,
            history_service=self.history_service,
            audit_service=self.audit_service,
            operator="tester",
        )
        defaults.update(kwargs)
        return mark_failed(job_id, **defaults)

    def test_requires_confirm(self) -> None:
        job = self._publishing_job()
        with self.assertRaises(RecoveryConfirmationRequiredError):
            self._mark(job.job_id, reason="timed out", confirm=False)

    def test_requires_non_empty_reason(self) -> None:
        job = self._publishing_job()
        with self.assertRaises(RecoveryIneligibleError):
            self._mark(job.job_id, reason="   ")

    def test_only_accepts_publishing_or_failed(self) -> None:
        job = _job(status="approved")
        self.queue_service.add_job(job)
        with self.assertRaises(RecoveryIneligibleError):
            self._mark(job.job_id, reason="not eligible")

    def test_updates_queue_correctly(self) -> None:
        job = self._publishing_job()
        updated = self._mark(job.job_id, reason="Share click never confirmed on Instagram")

        self.assertEqual(updated.status, "failed")
        self.assertEqual(updated.error, "Share click never confirmed on Instagram")
        self.assertIsNone(updated.published_at)

    def test_preserves_existing_platform_url_but_flags_it(self) -> None:
        job = self._publishing_job(
            platform_url="https://www.instagram.com/p/OLD123/", platform_post_id="OLD123"
        )
        updated = self._mark(job.job_id, reason="verification inconclusive after manual review")

        self.assertEqual(updated.platform_url, "https://www.instagram.com/p/OLD123/")
        self.assertEqual(updated.platform_post_id, "OLD123")

        events = self.history_service.list_events("2026-08-01")
        self.assertEqual(
            events[-1]["details"]["preserved_platform_url"], "https://www.instagram.com/p/OLD123/"
        )
        self.assertEqual(events[-1]["details"]["preserved_platform_post_id"], "OLD123")

    def test_writes_manual_reconciled_failed_history(self) -> None:
        job = self._publishing_job()
        self._mark(job.job_id, reason="never verified")

        events = self.history_service.list_events("2026-08-01")
        self.assertEqual([event["event_type"] for event in events], ["manual_reconciled_failed"])
        self.assertEqual(events[0]["previous_status"], "publishing")
        self.assertEqual(events[0]["new_status"], "failed")


# ---------------------------------------------------------------------------
# --clear-stale-lock
# ---------------------------------------------------------------------------


class ClearStaleLockTests(RecoveryFixture):
    def _clear(self, job_id: str, **kwargs):
        defaults = dict(
            confirm=True,
            config=self.config,
            history_service=self.history_service,
            audit_service=self.audit_service,
            operator="tester",
        )
        defaults.update(kwargs)
        return clear_stale_lock(job_id, **defaults)

    def test_requires_confirm(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)
        _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=60)

        with self.assertRaises(RecoveryConfirmationRequiredError):
            self._clear(job.job_id, confirm=False)

    def test_missing_lock_raises(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)

        with self.assertRaises(LockNotFoundError):
            self._clear(job.job_id)

    def test_recent_lock_cannot_be_cleared(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)
        _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=1)

        with self.assertRaises(LockNotStaleError):
            self._clear(job.job_id)

    def test_active_pid_lock_cannot_be_cleared(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)
        _write_lock(self.locks_dir, job.job_id, pid=os.getpid(), age_minutes=60)

        with self.assertRaises(LockOwnerActiveError):
            self._clear(job.job_id)

    def test_stale_inactive_lock_can_be_cleared(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)
        lock_path = _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=60)

        self._clear(job.job_id)

        self.assertFalse(lock_path.exists())
        self.assertEqual(self.queue_service.get(job.job_id).status, "publishing")

    def test_cleared_lock_contents_preserved_in_audit(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)
        _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=60)

        self._clear(job.job_id)

        records = self.audit_service.list_records("2026-08-01")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["lock_state"]["contents"]["job_id"], job.job_id)

    def test_lock_cleared_history_event_written_without_status_change(self) -> None:
        job = _job(status="publishing")
        self.queue_service.add_job(job)
        _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=60)

        self._clear(job.job_id)

        events = self.history_service.list_events("2026-08-01")
        self.assertEqual(events[-1]["event_type"], "lock_cleared")
        self.assertEqual(events[-1]["previous_status"], "publishing")
        self.assertEqual(events[-1]["new_status"], "publishing")


# ---------------------------------------------------------------------------
# Malformed JSON handling
# ---------------------------------------------------------------------------


class MalformedJsonTests(RecoveryFixture):
    def test_malformed_queue_json_fails_safely_on_inspect(self) -> None:
        self.config.queue_path().parent.mkdir(parents=True, exist_ok=True)
        self.config.queue_path().write_text("{not valid json", encoding="utf-8")

        with self.assertRaises(PublishQueueError):
            inspect_job("any-job", self.config)

    def test_malformed_history_json_degrades_inspect_instead_of_crashing(self) -> None:
        job = _job()
        self.queue_service.add_job(job)
        self.config.history_dir().mkdir(parents=True, exist_ok=True)
        (self.config.history_dir() / "history_2026-08-01.json").write_text("{bad", encoding="utf-8")

        report = inspect_job(job.job_id, self.config)

        self.assertTrue(report.found)
        self.assertIsNotNone(report.history_read_error)

    def test_malformed_audit_json_raises_clear_error(self) -> None:
        self.config.audit_dir().mkdir(parents=True, exist_ok=True)
        (self.config.audit_dir() / "audit_2026-08-01.json").write_text("{bad", encoding="utf-8")

        with self.assertRaises(AuditServiceError):
            self.audit_service.load("2026-08-01")


# ---------------------------------------------------------------------------
# No automatic retry
# ---------------------------------------------------------------------------


class NoRetryTests(unittest.TestCase):
    def test_no_retry_cli_flag_exists(self) -> None:
        self.assertNotIn('"--retry"', MODULE_SOURCE)
        self.assertNotIn("'--retry'", MODULE_SOURCE)
        self.assertNotIn("--retry-publish", MODULE_SOURCE)

    def test_no_automatic_retry_function_exists(self) -> None:
        for name in ("retry_publish", "auto_retry", "retry_send", "retry"):
            self.assertFalse(hasattr(publish_recovery, name))


# ---------------------------------------------------------------------------
# Structural safety scan
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def test_no_playwright_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip().lower()
            self.assertFalse(stripped.startswith("import playwright"))
            self.assertFalse(stripped.startswith("from playwright"))

    def test_no_share_click(self) -> None:
        self.assertNotIn(".click(", MODULE_SOURCE)

    def test_no_browser_launch(self) -> None:
        self.assertNotIn("launch_persistent_context", MODULE_SOURCE)
        self.assertNotIn("async_playwright", MODULE_SOURCE)

    def test_no_page_goto(self) -> None:
        self.assertNotIn("page.goto", MODULE_SOURCE)
        self.assertNotIn(".goto(", MODULE_SOURCE)

    def test_no_set_input_files(self) -> None:
        self.assertNotIn("set_input_files", MODULE_SOURCE)

    def test_no_keyboard_press(self) -> None:
        self.assertNotIn(".press(", MODULE_SOURCE)
        self.assertNotIn("keyboard", MODULE_SOURCE.lower())

    def test_no_instagram_session_dependency(self) -> None:
        self.assertNotIn("InstagramSession(", MODULE_SOURCE)
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from .instagram_session import"))
            self.assertFalse(stripped.startswith("from src.social.instagram_session import"))
            self.assertFalse(stripped.startswith("import instagram_session"))

    def test_module_is_entirely_synchronous(self) -> None:
        self.assertNotIn("async def", MODULE_SOURCE)
        self.assertNotIn("import asyncio", MODULE_SOURCE)

    def test_lock_unlink_occurs_exactly_once(self) -> None:
        matches = re.findall(r"\.unlink\(\)", MODULE_SOURCE)
        self.assertEqual(len(matches), 1, matches)

    def test_unlink_call_is_inside_clear_stale_lock_only(self) -> None:
        start = MODULE_SOURCE.index("def clear_stale_lock(")
        rest = MODULE_SOURCE[start:]
        next_def_match = re.search(r"\n(def |class )", rest[1:])
        function_body = rest if next_def_match is None else rest[: next_def_match.start() + 1]
        self.assertIn(".unlink()", function_body)


# ---------------------------------------------------------------------------
# src/social untouched
# ---------------------------------------------------------------------------


class SocialPackageUntouchedTests(RecoveryFixture):
    def test_social_files_are_byte_for_byte_unchanged(self) -> None:
        watched_files = [
            app_root() / "src" / "social" / "instagram_session.py",
            app_root() / "src" / "social" / "instagram_comment_reader.py",
            app_root() / "src" / "social" / "instagram_reply_controller.py",
            app_root() / "src" / "social" / "instagram_reply_sender.py",
            app_root() / "src" / "social" / "instagram_dm_reader.py",
            app_root() / "src" / "social" / "reply_brain.py",
        ]

        def _snapshot() -> dict[str, str]:
            return {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in watched_files
                if path.is_file()
            }

        before = _snapshot()

        job = _job(status="publishing")
        self.queue_service.add_job(job)
        _write_lock(self.locks_dir, job.job_id, pid=_dead_pid(), age_minutes=60)
        inspect_job(job.job_id, self.config)
        list_stuck_jobs(self.config)
        clear_stale_lock(
            job.job_id,
            confirm=True,
            config=self.config,
            history_service=self.history_service,
            audit_service=self.audit_service,
            operator="tester",
        )
        mark_failed(
            job.job_id,
            reason="reconciled during test",
            confirm=True,
            config=self.config,
            queue_service=self.queue_service,
            history_service=self.history_service,
            audit_service=self.audit_service,
            operator="tester",
        )

        after = _snapshot()
        self.assertEqual(before, after)
        self.assertTrue(before)


if __name__ == "__main__":
    unittest.main()
