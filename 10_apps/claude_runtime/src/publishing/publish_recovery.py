from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .history_service import HistoryService, HistoryServiceError, build_history_service
from .models import PublisherConfig, PublishJob, load_publisher_config, now_iso
from .queue_service import QueueService, build_queue_service, load_job_by_id

# Phase 10C.5 is a diagnostic and manual-reconciliation layer only. It is
# entirely synchronous — there is no asyncio, no Playwright import, no
# InstagramSession dependency, and no code path capable of opening a
# browser or clicking Share anywhere in this file. --inspect and
# --list-stuck are strictly read-only. --mark-published, --mark-failed,
# and --clear-stale-lock are the only three mutating actions, each gated
# behind an explicit --confirm flag, and none of them verifies anything
# on Instagram — they are human-certified manual corrections only. There
# is no batch mode and no automatic retry of anything.
DISALLOWED_ACTIONS = (
    "click_share",
    "open_instagram",
    "launch_browser",
    "auto_retry",
    "auto_publish",
    "auto_delete_lock",
    "batch_mark_published",
    "batch_mark_failed",
)

AUDIT_VERSION = "1.0"


def _runtime_root() -> Path:
    """
    publish_recovery.py location:

    10_apps/claude_runtime/src/publishing/publish_recovery.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RecoveryError(RuntimeError):
    """Base error for the Phase 10C.5 publish recovery tool."""


class RecoveryJobNotFoundError(RecoveryError):
    """Raised when the given job_id does not exist in the publish queue."""


class RecoveryConfirmationRequiredError(RecoveryError):
    """Raised when a mutating action is attempted without --confirm."""


class RecoveryIneligibleError(RecoveryError):
    """Raised when a job does not meet a mutating action's precondition."""


class InvalidInstagramUrlError(RecoveryError):
    """Raised when --platform-url is missing, empty, or not an allowed pattern."""


class ConflictingPublishedRecordError(RecoveryError):
    """Raised when another job already claims the same platform_post_id."""


class LockNotFoundError(RecoveryError):
    """Raised when --clear-stale-lock targets a job_id with no lock file."""


class LockNotStaleError(RecoveryError):
    """Raised when a lock is younger than the configured stale threshold."""


class LockOwnerActiveError(RecoveryError):
    """Raised when a lock's pid appears to still be running locally."""


class AuditServiceError(RuntimeError):
    """Raised for malformed audit data."""


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _age_minutes(iso_timestamp: str) -> float:
    parsed = datetime.fromisoformat(iso_timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).total_seconds() / 60.0


def _pid_is_active(pid: int) -> bool:
    """
    Best-effort local liveness probe via a signal-0 kill(). Never
    actually sends a real signal. ProcessLookupError -> not active;
    PermissionError -> owned by someone else but definitely running,
    so treated as active (the safer, more conservative answer for a
    tool that refuses to clear a lock it isn't sure about).
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


_POST_ID_PATTERN = re.compile(r"/(?:p|reel)/([A-Za-z0-9_-]+)/?")


def parse_instagram_post_id(url: str) -> str | None:
    match = _POST_ID_PATTERN.search(url)
    return match.group(1) if match else None


def is_allowed_instagram_url(url: str, patterns: tuple[str, ...]) -> bool:
    return bool(patterns) and any(url.startswith(pattern) for pattern in patterns)


def _queue_snapshot(job: PublishJob | None) -> dict[str, Any] | None:
    """
    Only the 7 fields a recovery audit trail needs — never metadata,
    caption_text, hashtags, or any other job content.
    """
    if job is None:
        return None

    return {
        "job_id": job.job_id,
        "status": job.status,
        "approved_at": job.approved_at,
        "published_at": job.published_at,
        "platform_post_id": job.platform_post_id,
        "platform_url": job.platform_url,
        "error": job.error,
        "updated_at": job.updated_at,
    }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class AuditService:
    """
    Append-safe, one-file-per-production-date audit log under
    output/publishing/audit/audit_<date>.json. Same shape/atomicity as
    HistoryService. Never stores cookies, passwords, tokens, browser
    profile contents, caption text, hashtags, or private DM/comment
    data — only the fields in the requirement-7 schema.
    """

    def __init__(self, audit_dir: str | Path) -> None:
        self.audit_dir = Path(audit_dir)

    def _path(self, production_date: str) -> Path:
        return self.audit_dir / f"audit_{production_date}.json"

    def load(self, production_date: str) -> dict[str, Any]:
        path = self._path(production_date)

        if not path.exists():
            return {"version": AUDIT_VERSION, "production_date": production_date, "records": []}

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise AuditServiceError(f"Invalid JSON in audit log: {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise AuditServiceError(f"Audit log must be a JSON object: {path}")

        data.setdefault("version", AUDIT_VERSION)
        data.setdefault("production_date", production_date)
        records = data.get("records")

        if not isinstance(records, list):
            raise AuditServiceError(f"Audit log 'records' must be a list: {path}")

        return data

    def _atomic_write(self, production_date: str, data: dict[str, Any]) -> None:
        path = self._path(production_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        temporary_path.replace(path)

    def record(
        self,
        *,
        production_date: str,
        command: str,
        job_id: str,
        operator: str,
        previous_status: str | None,
        new_status: str | None,
        lock_state: dict[str, Any] | None,
        queue_snapshot_before: dict[str, Any] | None,
        queue_snapshot_after: dict[str, Any] | None,
        history_events_added: list[str],
        details: dict[str, Any] | None,
        result: str,
        error: str | None,
    ) -> dict[str, Any]:
        data = self.load(production_date)
        record = {
            "audit_id": f"audit-{len(data['records']) + 1}-{uuid4().hex[:8]}",
            "created_at": now_iso(),
            "command": command,
            "job_id": job_id,
            "operator": operator,
            "previous_status": previous_status,
            "new_status": new_status,
            "lock_state": lock_state,
            "queue_snapshot_before": queue_snapshot_before,
            "queue_snapshot_after": queue_snapshot_after,
            "history_events_added": list(history_events_added),
            "details": details or {},
            "result": result,
            "error": error,
        }
        data["records"].append(record)
        self._atomic_write(production_date, data)
        return record

    def list_records(self, production_date: str) -> list[dict[str, Any]]:
        return self.load(production_date)["records"]


def build_audit_service(config: PublisherConfig | None = None) -> AuditService:
    config = config or load_publisher_config()
    return AuditService(config.audit_dir())


# ---------------------------------------------------------------------------
# Lock inspection (read-only helpers, reused by inspect/list-stuck/clear)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LockInfo:
    exists: bool
    path: str | None = None
    contents: dict[str, Any] | None = None
    age_minutes: float | None = None
    pid_active: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _inspect_lock(job_id: str, config: PublisherConfig) -> LockInfo:
    lock_path = config.locks_dir() / f"{job_id}.lock.json"

    if not lock_path.is_file():
        return LockInfo(exists=False)

    try:
        contents = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return LockInfo(exists=True, path=str(lock_path))

    started_at = contents.get("started_at")
    age = _age_minutes(started_at) if isinstance(started_at, str) else None
    pid = contents.get("pid")
    active = _pid_is_active(pid) if isinstance(pid, int) else None

    return LockInfo(exists=True, path=str(lock_path), contents=contents, age_minutes=age, pid_active=active)


def _lock_is_clearable(lock_path: Path, config: PublisherConfig) -> bool:
    """
    True only when a lock is BOTH older than recovery.lock_stale_after_minutes
    AND its pid cannot be confirmed active. Shared by clear_stale_lock()
    (the mutating path) and status_service.py's read-only stale_locks
    count, so the two never disagree.
    """
    if not lock_path.is_file():
        return False

    try:
        contents = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    started_at = contents.get("started_at")
    if not isinstance(started_at, str):
        return False

    if _age_minutes(started_at) < config.recovery.lock_stale_after_minutes:
        return False

    pid = contents.get("pid")
    if isinstance(pid, int) and _pid_is_active(pid):
        return False

    return True


# ---------------------------------------------------------------------------
# Inspect (read-only)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InspectionReport:
    job_id: str
    found: bool
    production_date: str | None = None
    persona_id: str | None = None
    status: str | None = None
    approved_at: str | None = None
    published_at: str | None = None
    platform_post_id: str | None = None
    platform_url: str | None = None
    error: str | None = None
    lock: LockInfo = field(default_factory=lambda: LockInfo(exists=False))
    recent_history_events: list[dict[str, Any]] = field(default_factory=list)
    latest_sender_log: dict[str, Any] | None = None
    available_screenshots: list[str] = field(default_factory=list)
    recommended_next_action: str = ""
    history_read_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lock"] = self.lock.to_dict()
        return data


def _find_latest_sender_log(job_id: str, config: PublisherConfig) -> dict[str, Any] | None:
    logs_dir = config.logs_dir()

    if not logs_dir.is_dir():
        return None

    candidates = sorted(logs_dir.glob(f"instagram_feed_sender_{job_id}_*.json"))

    if not candidates:
        return None

    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_screenshots(job_id: str, config: PublisherConfig) -> list[str]:
    screenshots_dir = config.screenshots_dir()

    if not screenshots_dir.is_dir():
        return []

    return sorted(str(path) for path in screenshots_dir.glob(f"*{job_id}*.png"))


def _recommend_action(job: PublishJob, lock: LockInfo) -> str:
    if job.status == "published":
        if not job.published_at or not job.platform_url:
            return (
                f"Inconsistent published record for {job.job_id} (missing "
                "published_at and/or platform_url). Verify on Instagram, then: "
                f"python3 -u -m src.publishing.publish_recovery --mark-published "
                f"{job.job_id} --platform-url URL --confirm"
            )
        return "No action needed — job is published and consistent."

    if job.status == "publishing":
        return (
            f"Job is stuck in 'publishing'. Check Instagram directly, then run "
            f"either: python3 -u -m src.publishing.publish_recovery "
            f"--mark-published {job.job_id} --platform-url URL --confirm   OR   "
            f"--mark-failed {job.job_id} --reason TEXT --confirm"
        )

    if job.status == "failed" and job.error == "publish_verification_inconclusive_manual_check_required":
        return (
            f"Share was clicked but not verified. Check Instagram directly, "
            f"then run either --mark-published or --mark-failed for "
            f"{job.job_id} (both require --confirm)."
        )

    if lock.exists and lock.pid_active is False:
        return (
            f"A lock exists for {job.job_id} with no confirmed active process. "
            "If it is genuinely stale: python3 -u -m src.publishing.publish_recovery "
            f"--clear-stale-lock {job.job_id} --confirm"
        )

    if job.status == "approved":
        return f"python3 -u -m src.publishing.instagram_feed_sender --preflight {job.job_id}"

    return "No action needed."


def inspect_job(job_id: str, config: PublisherConfig) -> InspectionReport:
    """
    Strictly read-only: does not change the queue, does not change
    history, does not remove a lock, does not open Instagram, and
    imports nothing Playwright-related.
    """
    job = load_job_by_id(config.queue_path(), job_id)
    lock = _inspect_lock(job_id, config)

    if job is None:
        return InspectionReport(
            job_id=job_id,
            found=False,
            lock=lock,
            recommended_next_action=f"No job found for job_id={job_id}.",
        )

    history_events: list[dict[str, Any]] = []
    history_read_error: str | None = None

    try:
        all_events = HistoryService(config.history_dir()).list_events(job.production_date)
        history_events = [event for event in all_events if event.get("job_id") == job_id][-5:]
    except HistoryServiceError as exc:
        history_read_error = str(exc)

    return InspectionReport(
        job_id=job.job_id,
        found=True,
        production_date=job.production_date,
        persona_id=job.persona_id,
        status=job.status,
        approved_at=job.approved_at,
        published_at=job.published_at,
        platform_post_id=job.platform_post_id,
        platform_url=job.platform_url,
        error=job.error,
        lock=lock,
        recent_history_events=history_events,
        latest_sender_log=_find_latest_sender_log(job_id, config),
        available_screenshots=_find_screenshots(job_id, config),
        recommended_next_action=_recommend_action(job, lock),
        history_read_error=history_read_error,
    )


# ---------------------------------------------------------------------------
# --list-stuck (read-only)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StuckJobReport:
    job_id: str
    production_date: str
    status: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_stuck_jobs(config: PublisherConfig) -> list[StuckJobReport]:
    """
    Read-only scan of the whole queue. Never calls save_jobs(), never
    deletes a lock, never opens Instagram. A job may collect more than
    one reason.
    """
    queue_service = QueueService(config.queue_path())
    jobs = queue_service.load_jobs()
    reports: list[StuckJobReport] = []

    for job in jobs:
        reasons: list[str] = []

        if job.status == "publishing":
            age = _age_minutes(job.updated_at)
            threshold = config.recovery.publishing_stuck_after_minutes
            if age >= threshold:
                reasons.append(
                    f"status=publishing for {age:.1f} minutes (threshold {threshold})"
                )

        if job.status == "failed" and job.error == "publish_verification_inconclusive_manual_check_required":
            reasons.append("status=failed with manual-check-required error")

        lock_path = config.locks_dir() / f"{job.job_id}.lock.json"
        if lock_path.is_file():
            try:
                contents = json.loads(lock_path.read_text(encoding="utf-8"))
                pid = contents.get("pid")
                active = _pid_is_active(pid) if isinstance(pid, int) else False
            except (OSError, json.JSONDecodeError):
                active = False
            if not active:
                reasons.append("lock exists but no matching active process could be confirmed")

        if job.status == "published" and (not job.published_at or not job.platform_url):
            reasons.append("status=published but missing published_at and/or platform_url")

        try:
            all_events = HistoryService(config.history_dir()).list_events(job.production_date)
            job_events = [event for event in all_events if event.get("job_id") == job.job_id]
        except HistoryServiceError:
            job_events = []

        if job_events:
            last_status = job_events[-1].get("new_status")
            if last_status is not None and last_status != job.status:
                reasons.append(
                    f"queue status ({job.status}) disagrees with latest history "
                    f"event new_status ({last_status})"
                )

        if reasons:
            reports.append(
                StuckJobReport(
                    job_id=job.job_id,
                    production_date=job.production_date,
                    status=job.status,
                    reasons=reasons,
                )
            )

    return reports


# ---------------------------------------------------------------------------
# Mutating recovery actions (all require --confirm)
# ---------------------------------------------------------------------------


def mark_published(
    job_id: str,
    *,
    platform_url: str | None,
    confirm: bool,
    config: PublisherConfig,
    queue_service: QueueService,
    history_service: HistoryService,
    audit_service: AuditService,
    operator: str,
) -> PublishJob:
    if not confirm:
        raise RecoveryConfirmationRequiredError("--confirm is required for --mark-published")

    job = queue_service.get(job_id)
    if job is None:
        raise RecoveryJobNotFoundError(f"No job found for job_id: {job_id}")

    if job.content_type != "instagram_feed":
        raise RecoveryIneligibleError(
            f"Job {job_id} is not instagram_feed (content_type={job.content_type})"
        )

    if job.status not in ("publishing", "failed"):
        raise RecoveryIneligibleError(
            f"Job {job_id} status must be 'publishing' or 'failed' to "
            f"mark-published (is {job.status!r})"
        )

    if not platform_url or not platform_url.strip():
        raise InvalidInstagramUrlError("--platform-url is required and must be non-empty")

    platform_url = platform_url.strip()

    if not is_allowed_instagram_url(platform_url, config.recovery.allowed_instagram_url_patterns):
        raise InvalidInstagramUrlError(
            f"URL does not match an allowed Instagram post/reel pattern: {platform_url}"
        )

    parsed_post_id = parse_instagram_post_id(platform_url)

    if parsed_post_id:
        for other in queue_service.load_jobs():
            if (
                other.job_id != job_id
                and other.status == "published"
                and other.platform_post_id == parsed_post_id
            ):
                raise ConflictingPublishedRecordError(
                    f"platform_post_id {parsed_post_id} is already recorded as "
                    f"published on job {other.job_id}; refusing to create a "
                    "conflicting record"
                )

    queue_snapshot_before = _queue_snapshot(job)
    previous_status = job.status
    lock_state = _inspect_lock(job_id, config).to_dict()

    print()
    print("=" * 72)
    print("MANUAL RECONCILIATION — mark-published")
    print("=" * 72)
    print(f"JOB_ID:                  {job_id}")
    print(f"current status:          {job.status}")
    print(f"current error:           {job.error}")
    print(f"platform URL:            {platform_url}")
    print(f"parsed platform post ID: {parsed_post_id}")
    print()
    print("WARNING: this manually marks the job as published in the local")
    print("         queue WITHOUT verifying anything on Instagram. Only")
    print("         proceed after confirming the post is real yourself.")
    print("=" * 72)
    print()

    published_at = now_iso()

    def _mutate(target: PublishJob) -> None:
        target.status = "published"
        target.published_at = published_at
        target.platform_url = platform_url
        if parsed_post_id:
            target.platform_post_id = parsed_post_id
        target.error = None

    history_events_added: list[str] = []
    updated = queue_service.update_job(job_id, _mutate)
    queue_snapshot_after = _queue_snapshot(updated)

    try:
        history_service.record_manual_reconciled_published(
            production_date=job.production_date,
            job_id=job_id,
            previous_status=previous_status,
            details={
                "platform_url": platform_url,
                "platform_post_id": parsed_post_id,
                "operator": operator,
            },
        )
        history_events_added.append("manual_reconciled_published")
    except Exception as exc:
        audit_service.record(
            production_date=job.production_date,
            command="--mark-published",
            job_id=job_id,
            operator=operator,
            previous_status=previous_status,
            new_status="published",
            lock_state=lock_state,
            queue_snapshot_before=queue_snapshot_before,
            queue_snapshot_after=queue_snapshot_after,
            history_events_added=history_events_added,
            details={
                "platform_url": platform_url,
                "platform_post_id": parsed_post_id,
                "partial_write_risk": (
                    "The queue was already updated to 'published' before the "
                    "history write failed. Queue and history are two separate "
                    "atomic writes, not one transaction — verify history.json "
                    "manually before assuming this reconciliation is complete."
                ),
            },
            result="failed",
            error=str(exc),
        )
        raise

    audit_service.record(
        production_date=job.production_date,
        command="--mark-published",
        job_id=job_id,
        operator=operator,
        previous_status=previous_status,
        new_status="published",
        lock_state=lock_state,
        queue_snapshot_before=queue_snapshot_before,
        queue_snapshot_after=queue_snapshot_after,
        history_events_added=history_events_added,
        details={"platform_url": platform_url, "platform_post_id": parsed_post_id},
        result="success",
        error=None,
    )

    return updated


def mark_failed(
    job_id: str,
    *,
    reason: str | None,
    confirm: bool,
    config: PublisherConfig,
    queue_service: QueueService,
    history_service: HistoryService,
    audit_service: AuditService,
    operator: str,
) -> PublishJob:
    if not confirm:
        raise RecoveryConfirmationRequiredError("--confirm is required for --mark-failed")

    job = queue_service.get(job_id)
    if job is None:
        raise RecoveryJobNotFoundError(f"No job found for job_id: {job_id}")

    if job.status not in ("publishing", "failed"):
        raise RecoveryIneligibleError(
            f"Job {job_id} status must be 'publishing' or 'failed' to "
            f"mark-failed (is {job.status!r})"
        )

    if not reason or not reason.strip():
        raise RecoveryIneligibleError("--reason is required and must be non-empty")

    reason = reason.strip()

    queue_snapshot_before = _queue_snapshot(job)
    previous_status = job.status
    preserved_url = job.platform_url
    preserved_post_id = job.platform_post_id
    lock_state = _inspect_lock(job_id, config).to_dict()

    print()
    print("=" * 72)
    print("MANUAL RECONCILIATION — mark-failed")
    print("=" * 72)
    print(f"JOB_ID:            {job_id}")
    print(f"current status:    {job.status}")
    print(f"current error:     {job.error}")
    print(f"reason:            {reason}")
    if preserved_url or preserved_post_id:
        print("NOTE: existing platform_url/platform_post_id will be PRESERVED, not cleared:")
        print(f"  platform_url:      {preserved_url}")
        print(f"  platform_post_id:  {preserved_post_id}")
    print("=" * 72)
    print()

    def _mutate(target: PublishJob) -> None:
        target.status = "failed"
        target.error = reason
        target.published_at = None

    details: dict[str, Any] = {"reason": reason}
    if preserved_url:
        details["preserved_platform_url"] = preserved_url
    if preserved_post_id:
        details["preserved_platform_post_id"] = preserved_post_id

    history_events_added: list[str] = []
    updated = queue_service.update_job(job_id, _mutate)
    queue_snapshot_after = _queue_snapshot(updated)

    try:
        history_service.record_manual_reconciled_failed(
            production_date=job.production_date,
            job_id=job_id,
            previous_status=previous_status,
            details=details,
        )
        history_events_added.append("manual_reconciled_failed")
    except Exception as exc:
        audit_service.record(
            production_date=job.production_date,
            command="--mark-failed",
            job_id=job_id,
            operator=operator,
            previous_status=previous_status,
            new_status="failed",
            lock_state=lock_state,
            queue_snapshot_before=queue_snapshot_before,
            queue_snapshot_after=queue_snapshot_after,
            history_events_added=history_events_added,
            details={
                **details,
                "partial_write_risk": (
                    "The queue was already updated to 'failed' before the "
                    "history write failed. Verify history.json manually."
                ),
            },
            result="failed",
            error=str(exc),
        )
        raise

    audit_service.record(
        production_date=job.production_date,
        command="--mark-failed",
        job_id=job_id,
        operator=operator,
        previous_status=previous_status,
        new_status="failed",
        lock_state=lock_state,
        queue_snapshot_before=queue_snapshot_before,
        queue_snapshot_after=queue_snapshot_after,
        history_events_added=history_events_added,
        details=details,
        result="success",
        error=None,
    )

    return updated


def clear_stale_lock(
    job_id: str,
    *,
    confirm: bool,
    config: PublisherConfig,
    history_service: HistoryService,
    audit_service: AuditService,
    operator: str,
) -> None:
    """
    Never deletes a lock automatically — only reachable via this
    function, only with --confirm, only when the lock is both older
    than recovery.lock_stale_after_minutes AND its pid cannot be
    confirmed active. Does NOT change job status.
    """
    if not confirm:
        raise RecoveryConfirmationRequiredError("--confirm is required for --clear-stale-lock")

    lock_path = config.locks_dir() / f"{job_id}.lock.json"

    if not lock_path.is_file():
        raise LockNotFoundError(f"No lock file found for job_id={job_id}")

    try:
        contents = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"Lock file for {job_id} could not be read: {exc}") from exc

    started_at = contents.get("started_at")
    if not isinstance(started_at, str):
        raise RecoveryError(
            f"Lock file for {job_id} has no started_at; refusing to guess its age"
        )

    age = _age_minutes(started_at)
    threshold = config.recovery.lock_stale_after_minutes

    if age < threshold:
        raise LockNotStaleError(
            f"Lock for {job_id} is only {age:.1f} minutes old (threshold "
            f"{threshold}); refusing to clear a recent lock"
        )

    pid = contents.get("pid")
    if isinstance(pid, int) and _pid_is_active(pid):
        raise LockOwnerActiveError(
            f"Lock for {job_id} belongs to pid {pid}, which appears to still "
            "be running on this machine; refusing to clear"
        )

    queue_service = QueueService(config.queue_path())
    job = queue_service.get(job_id)

    if job is None:
        raise RecoveryJobNotFoundError(
            f"No queue job found for job_id={job_id}; cannot record history/audit "
            "for lock clearing. The lock file itself was NOT removed."
        )

    lock_state = {
        "exists": True,
        "path": str(lock_path),
        "contents": contents,
        "age_minutes": age,
        "pid_active": False,
    }

    print()
    print("=" * 72)
    print("MANUAL RECONCILIATION — clear-stale-lock")
    print("=" * 72)
    print(f"JOB_ID:      {job_id}")
    print(f"lock age:    {age:.1f} minutes (threshold {threshold})")
    print(f"lock pid:    {pid} (active: False)")
    print("Job status will NOT be changed by this action.")
    print("=" * 72)
    print()

    lock_path.unlink()

    history_service.record_lock_cleared(
        production_date=job.production_date,
        job_id=job_id,
        status=job.status,
        details={"lock_contents": contents, "age_minutes": age},
    )

    audit_service.record(
        production_date=job.production_date,
        command="--clear-stale-lock",
        job_id=job_id,
        operator=operator,
        previous_status=job.status,
        new_status=job.status,
        lock_state=lock_state,
        queue_snapshot_before=_queue_snapshot(job),
        queue_snapshot_after=_queue_snapshot(job),
        history_events_added=["lock_cleared"],
        details={"lock_contents": contents},
        result="success",
        error=None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_operator() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Instagram Feed Publish Recovery & Audit (Phase 10C.5). "
            "--inspect and --list-stuck are read-only. --mark-published, "
            "--mark-failed, and --clear-stale-lock are explicit, "
            "--confirm-gated manual corrections. No Share click, no "
            "Instagram access, no automatic retry."
        )
    )

    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument("--inspect", metavar="JOB_ID", default=None)
    action.add_argument("--list-stuck", action="store_true")
    action.add_argument(
        "--mark-published", dest="mark_published", metavar="JOB_ID", default=None
    )
    action.add_argument(
        "--mark-failed", dest="mark_failed", metavar="JOB_ID", default=None
    )
    action.add_argument(
        "--clear-stale-lock", dest="clear_stale_lock", metavar="JOB_ID", default=None
    )

    parser.add_argument(
        "--platform-url", default=None, help="Required with --mark-published."
    )
    parser.add_argument("--reason", default=None, help="Required with --mark-failed.")
    parser.add_argument(
        "--confirm", action="store_true", help="Required for any mutating action."
    )
    parser.add_argument(
        "--operator", default=None, help="Recorded in the audit log. Defaults to $USER."
    )
    parser.add_argument(
        "--config", default=None, help="Path to an alternate config/publishing/publisher.yaml."
    )

    arguments = parser.parse_args(argv)

    if arguments.mark_published and not arguments.platform_url:
        parser.error("--platform-url is required with --mark-published")

    if arguments.mark_failed and not arguments.reason:
        parser.error("--reason is required with --mark-failed")

    if arguments.mark_published and not arguments.confirm:
        parser.error("--confirm is required with --mark-published")

    if arguments.mark_failed and not arguments.confirm:
        parser.error("--confirm is required with --mark-failed")

    if arguments.clear_stale_lock and not arguments.confirm:
        parser.error("--confirm is required with --clear-stale-lock")

    return arguments


def _print_inspection_report(report: InspectionReport) -> None:
    print()
    title = f"Inspection — {report.job_id}"
    print(title)
    print("-" * len(title))

    if not report.found:
        print("Job not found in the queue.")
        print(f"Recommended next action:\n  {report.recommended_next_action}")
        print()
        return

    print(f"production_date:     {report.production_date}")
    print(f"persona_id:          {report.persona_id}")
    print(f"status:              {report.status}")
    print(f"approved_at:         {report.approved_at}")
    print(f"published_at:        {report.published_at}")
    print(f"platform_post_id:    {report.platform_post_id}")
    print(f"platform_url:        {report.platform_url}")
    print(f"error:               {report.error}")
    print(f"lock exists:         {report.lock.exists}")

    if report.lock.exists:
        print(f"lock age (minutes):  {report.lock.age_minutes}")
        print(f"lock pid active:     {report.lock.pid_active}")
        print(f"lock contents:       {report.lock.contents}")

    print(f"latest history events ({len(report.recent_history_events)}):")
    for event in report.recent_history_events:
        print(
            f"  {event.get('created_at')}  {event.get('event_type')}  "
            f"{event.get('previous_status')} -> {event.get('new_status')}"
        )
    if report.history_read_error:
        print(f"  (history unreadable: {report.history_read_error})")

    print(f"latest sender log:   {'present' if report.latest_sender_log else 'none'}")

    print(f"available screenshots ({len(report.available_screenshots)}):")
    for path in report.available_screenshots:
        print(f"  {path}")

    print(f"recommended next action:\n  {report.recommended_next_action}")
    print()


def _print_stuck_jobs(reports: list[StuckJobReport]) -> None:
    print()
    print(f"Stuck jobs: {len(reports)}")
    print("-" * 20)
    for report in reports:
        print(f"{report.job_id}  (status={report.status}, date={report.production_date})")
        for reason in report.reasons:
            print(f"  - {reason}")
    print()


def _run(arguments: argparse.Namespace) -> int:
    config = load_publisher_config(arguments.config)
    operator = arguments.operator or _default_operator()

    if arguments.inspect:
        report = inspect_job(arguments.inspect, config)
        _print_inspection_report(report)
        return 0 if report.found else 1

    if arguments.list_stuck:
        reports = list_stuck_jobs(config)
        _print_stuck_jobs(reports)
        return 0

    queue_service = build_queue_service(config)
    history_service = build_history_service(config)
    audit_service = build_audit_service(config)

    if arguments.mark_published:
        try:
            job = mark_published(
                arguments.mark_published,
                platform_url=arguments.platform_url,
                confirm=arguments.confirm,
                config=config,
                queue_service=queue_service,
                history_service=history_service,
                audit_service=audit_service,
                operator=operator,
            )
        except RecoveryError as exc:
            print(f"[PublishRecovery] {exc}")
            return 1

        print(f"Marked {job.job_id} as published. published_at={job.published_at}")
        return 0

    if arguments.mark_failed:
        try:
            job = mark_failed(
                arguments.mark_failed,
                reason=arguments.reason,
                confirm=arguments.confirm,
                config=config,
                queue_service=queue_service,
                history_service=history_service,
                audit_service=audit_service,
                operator=operator,
            )
        except RecoveryError as exc:
            print(f"[PublishRecovery] {exc}")
            return 1

        print(f"Marked {job.job_id} as failed. error={job.error}")
        return 0

    if arguments.clear_stale_lock:
        try:
            clear_stale_lock(
                arguments.clear_stale_lock,
                confirm=arguments.confirm,
                config=config,
                history_service=history_service,
                audit_service=audit_service,
                operator=operator,
            )
        except RecoveryError as exc:
            print(f"[PublishRecovery] {exc}")
            return 1

        print(f"Cleared stale lock for {arguments.clear_stale_lock}.")
        return 0

    return 1


def main() -> None:
    arguments = parse_arguments()
    exit_code = _run(arguments)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
