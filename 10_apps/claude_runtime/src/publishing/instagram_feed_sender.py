from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.social.instagram_session import InstagramSession, load_config as load_session_config
from src.social.instagram_models import InstagramSessionError

from .history_service import HistoryService, build_history_service
from .instagram_feed_preview import (
    CaptionLimitExceededError,
    FeedPreviewConfig,
    FeedPreviewError,
    FeedPreviewIneligibleError,
    FinalCaptionResult,
    InstagramFeedPreviewController,
    build_final_caption,
    check_job_eligible,
    load_feed_preview_config,
)
from .models import PublisherConfig, PublishJob, load_publisher_config, now_iso
from .queue_service import QueueService, build_queue_service, load_job_by_id

# Phase 10C is the ONLY module in this codebase that actually clicks
# Share and publishes an Instagram Feed post for real. It performs
# exactly one Share click per invocation, for exactly one job_id, and
# only after re-running every Phase 10B check from scratch via the
# unmodified, shared InstagramFeedPreviewController._reach_share_ready
# helper. This module must never grow, and must never accidentally
# gain, any of the capabilities below. There is no batch mode, no
# scheduled/automatic publishing, and no automatic retry of a failed or
# inconclusive send.
DISALLOWED_ACTIONS = (
    "batch_publish",
    "publish_all",
    "schedule_publish",
    "auto_publish",
    "retry_send",
    "retry_publish",
    "press",
    "press_enter",
    "force_click",
    "click_via_js",
    "like",
    "follow",
    "unfollow",
    "comment",
    "reply",
    "send_dm",
    "delete",
)

DEFAULT_INSTAGRAM_CONFIG_RELATIVE_PATH = Path("config") / "social" / "instagram.yaml"


def _runtime_root() -> Path:
    """
    instagram_feed_sender.py location:

    10_apps/claude_runtime/src/publishing/instagram_feed_sender.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FeedSenderError(RuntimeError):
    """Base error for the Phase 10C Instagram Feed sender."""


class FeedSenderConfigError(FeedSenderError):
    """Raised when config/social/instagram.yaml's feed_sender section is missing/invalid."""


class FeedSenderIneligibleError(FeedSenderError):
    """Raised when the job fails a Phase 10C-specific publishability rule."""


class PublishLockError(FeedSenderError):
    """Raised when a live publish lock already exists for this job_id."""


class ShareClickFailedError(FeedSenderError):
    """Raised when clicking the verified Share button itself raises an error."""


class PublishVerificationInconclusiveError(FeedSenderError):
    """
    Raised when Share was clicked but fewer than two independent
    success signals (or any error signal) were found afterward. The
    job is left as 'failed', never 'published', on this outcome.
    """


# ---------------------------------------------------------------------------
# Eligibility (pure, no browser) — Phase 10C additions on top of Phase 10B
# ---------------------------------------------------------------------------


def check_job_publishable(job: PublishJob) -> None:
    """
    Raise unless the job may be sent for real.

    Reuses check_job_eligible() (content_type, status=='approved',
    exactly one existing non-zero-byte media file, non-empty caption)
    rather than duplicating those rules, then adds the Phase
    10C-specific double-post protections: approved_at must be present,
    and published_at/platform_post_id must both still be null.
    """
    check_job_eligible(job)

    if job.approved_at is None:
        raise FeedSenderIneligibleError(
            f"Job {job.job_id} has no approved_at timestamp; refusing to publish"
        )

    if job.published_at is not None:
        raise FeedSenderIneligibleError(
            f"Job {job.job_id} already has published_at set "
            f"({job.published_at}); refusing to publish again"
        )

    if job.platform_post_id is not None:
        raise FeedSenderIneligibleError(
            f"Job {job.job_id} already has a platform_post_id "
            f"({job.platform_post_id}); refusing to publish again"
        )


# ---------------------------------------------------------------------------
# Operation lock
# ---------------------------------------------------------------------------


class PublishLock:
    """
    One atomic local lock file per job_id under
    output/publishing/locks/<job_id>.lock.json.

    acquire() uses os.O_CREAT | os.O_EXCL for a race-free atomic
    exclusive create — if the file already exists, acquire() always
    refuses (no PID-liveness check, no TTL/expiry). A lock is only
    ever deleted by release(), and release() is only ever called by
    this module on a PRE-click failure. Once Share has been clicked,
    the lock is retained permanently as an audit trail and a
    structural block against a second send attempt — this module
    provides no command to remove it.
    """

    def __init__(self, lock_dir: str | Path) -> None:
        self.lock_dir = Path(lock_dir)

    def _path(self, job_id: str) -> Path:
        return self.lock_dir / f"{job_id}.lock.json"

    def acquire(self, job_id: str, *, command: str = "--publish-approved") -> Path:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(job_id)

        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = None

        payload = {
            "job_id": job_id,
            "pid": os.getpid(),
            "started_at": now_iso(),
            "command": command,
            "hostname": hostname,
        }

        try:
            file_descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise PublishLockError(
                f"A publish lock already exists for job_id={job_id}: {path}. "
                "This is not evidence that publishing succeeded — it means a "
                "send was already attempted. Manually review the lock, the "
                "queue, and Instagram itself before deciding what to do next."
            ) from exc

        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return path

    def release(self, job_id: str) -> None:
        try:
            self._path(job_id).unlink()
        except FileNotFoundError:
            pass

    def exists(self, job_id: str) -> bool:
        return self._path(job_id).exists()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeedSenderSelectors:
    success_message: list[str] = field(default_factory=list)
    publish_error_message: list[str] = field(default_factory=list)
    post_permalink_url_marker: str = "/p/"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeedSenderConfig:
    selectors: FeedSenderSelectors
    share_click_timeout_ms: int
    verification_wait_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selectors": self.selectors.to_dict(),
            "share_click_timeout_ms": self.share_click_timeout_ms,
            "verification_wait_ms": self.verification_wait_ms,
        }


def load_feed_sender_config(
    *,
    config_path: str | Path | None = None,
) -> FeedSenderConfig:
    """
    Load the feed_sender section from config/social/instagram.yaml.

    Independent loader, matching the pattern already used by every
    other module in this package (instagram_feed_preview.py included).
    """
    runtime_root = _runtime_root()

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (runtime_root / DEFAULT_INSTAGRAM_CONFIG_RELATIVE_PATH).resolve()
    )

    if not resolved_config_path.exists():
        raise FeedSenderConfigError(
            f"Instagram session config not found: {resolved_config_path}"
        )

    try:
        with resolved_config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise FeedSenderConfigError(
            f"Invalid YAML in {resolved_config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict) or not raw:
        raise FeedSenderConfigError(
            f"Instagram session config is empty or invalid: {resolved_config_path}"
        )

    section = raw.get("feed_sender") or {}
    selectors_section = section.get("selectors") or {}
    timeouts_section = section.get("timeouts") or {}

    selectors = FeedSenderSelectors(
        success_message=list(selectors_section.get("success_message", [])),
        publish_error_message=list(selectors_section.get("publish_error_message", [])),
        post_permalink_url_marker=str(
            selectors_section.get("post_permalink_url_marker", "/p/")
        ),
    )

    return FeedSenderConfig(
        selectors=selectors,
        share_click_timeout_ms=int(timeouts_section.get("share_click_timeout_ms", 5000)),
        verification_wait_ms=int(timeouts_section.get("verification_wait_ms", 3000)),
    )


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeedPublishResult:
    job_id: str
    production_date: str
    persona_id: str
    login_status: str
    media_path: str
    media_uploaded: bool
    media_preview_verified: bool
    caption_verified: bool
    final_caption_length: int
    share_button_found: bool
    share_button_enabled: bool
    share_button_interactive: bool
    share_click_count: int
    share_clicked: bool
    verification_signals: list[str]
    publish_verified: bool
    published: bool
    platform_post_id: str | None
    platform_url: str | None
    previous_status: str
    final_status: str
    screenshots: dict[str, str | None]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _print_confirmation_block(
    job: PublishJob,
    caption_result: FinalCaptionResult,
    *,
    share_candidates_found: int,
) -> None:
    print()
    print("=" * 72)
    print("CONFIRMATION — the next action publishes this post to Instagram")
    print("=" * 72)
    print(f"JOB_ID:                 {job.job_id}")
    print(f"production date:        {job.production_date}")
    print(f"persona ID:             {job.persona_id}")
    print(f"media path:             {job.media_paths[0]}")
    print(f"media filename:         {Path(job.media_paths[0]).name}")
    print(f"final caption ({caption_result.final_caption_length} chars):")
    print(caption_result.final_caption)
    print(f"current status:         {job.status}")
    print(f"approved_at:            {job.approved_at}")
    print(f"Share candidates found: {share_candidates_found}")
    print()
    print("WARNING: clicking Share now publishes this post publicly and")
    print("         visibly on the real Instagram account. This cannot be")
    print("         undone by this tool.")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class InstagramFeedSenderController(InstagramFeedPreviewController):
    """
    Publishes exactly one approved instagram_feed PublishJob for real,
    or verifies the send is ready without clicking Share (--preflight).

    Subclasses InstagramFeedPreviewController (Phase 10B) to reuse —
    never duplicate — its login detection, Create->Post navigation,
    media upload, crop/filter Next transitions, and caption fill+verify
    logic via the shared _reach_share_ready() helper. preflight() and
    publish() both call that exact same method for everything through
    "Share button located, verified, ready to click" — preflight is
    only a trustworthy predictor of what publish() will do next because
    it is not a re-implementation, it is the same code (mirrors
    instagram_reply_sender.InstagramReplySender's own rationale).

    Never presses Enter, never evaluates JS to click, never force
    -clicks, never retries a Share click, and never sends more than one
    Share click per invocation. preflight() never receives write access
    used for anything and never acquires the operation lock.
    """

    def __init__(
        self,
        *,
        session: InstagramSession,
        feed_config: FeedPreviewConfig,
        publisher_config: PublisherConfig,
        sender_config: FeedSenderConfig,
        queue_service: QueueService,
        history_service: HistoryService,
        screenshots_dir: str | Path | None = None,
        logs_dir: str | Path | None = None,
        locks_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            session=session,
            feed_config=feed_config,
            publisher_config=publisher_config,
            screenshots_dir=screenshots_dir,
            logs_dir=logs_dir,
        )
        self.sender_config = sender_config
        self.queue_service = queue_service
        self.history_service = history_service
        self.locks_dir = Path(locks_dir) if locks_dir else publisher_config.locks_dir()

    # -- platform post id/url parsing --------------------------------------

    def _extract_post_permalink(self, url: str) -> tuple[str | None, str | None]:
        marker = self.sender_config.selectors.post_permalink_url_marker

        if not marker or marker not in url:
            return None, None

        match = re.search(re.escape(marker) + r"([A-Za-z0-9_-]+)/?", url)

        if not match:
            return None, url

        return match.group(1), url

    # -- post-click verification --------------------------------------------

    async def _verify_publish(self, page: Any) -> tuple[list[str], bool]:
        """
        Require at least two independent positive signals, and zero
        error signals, before ever treating a click as verified
        published. A single signal alone is always "inconclusive," not
        "published."
        """
        signals: list[str] = []

        dialog_still_open = await self._find_first(
            page, self.feed_config.selectors.dialog_container
        )
        if dialog_still_open is None:
            signals.append("dialog_closed")

        success_element = await self._find_first(
            page, self.sender_config.selectors.success_message
        )
        if success_element is not None:
            signals.append("success_message_visible")

        marker = self.sender_config.selectors.post_permalink_url_marker
        if marker and marker in page.url:
            signals.append("url_is_post_permalink")

        error_element = await self._find_first(
            page, self.sender_config.selectors.publish_error_message
        )
        has_error_signal = error_element is not None

        verified = len(signals) >= 2 and not has_error_signal
        return signals, verified

    # -- diagnostics/audit log ----------------------------------------------

    def _write_send_log(
        self,
        *,
        started_at: str,
        job: PublishJob,
        command: str,
        diagnostics: dict[str, Any],
        screenshots: dict[str, str | None],
        previous_status: str,
        final_status: str,
        result: str,
        error: str | None,
        queue_transition_to_publishing_at: str | None = None,
        share_clicked_at: str | None = None,
        share_click_count: int = 0,
        verification_signals: list[str] | None = None,
        publish_verified: bool = False,
        platform_post_id: str | None = None,
        platform_url: str | None = None,
        lock_created_at: str | None = None,
        verification_finished_at: str | None = None,
        manual_check_required: bool = False,
        lock_cleanup_result: str | None = None,
    ) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        finished_at = self._now()
        run_id = finished_at.replace(":", "").replace("-", "").replace("+", "").replace(".", "")
        log_path = self.logs_dir / f"instagram_feed_sender_{job.job_id}_{run_id}.json"

        preflight_passed = bool(
            diagnostics.get("share_button_found") and diagnostics.get("caption_verified")
        )

        payload = {
            "started_at": started_at,
            "finished_at": finished_at,
            "command": command,
            "job_id": job.job_id,
            "production_date": job.production_date,
            "persona_id": job.persona_id,
            "media_path": job.media_paths[0] if job.media_paths else None,
            "final_caption_length": diagnostics.get("caption_length"),
            "login_status": diagnostics.get("login_status"),
            "preflight_passed": preflight_passed,
            "share_candidates": 1 if diagnostics.get("share_button_found") else 0,
            "share_button_enabled": diagnostics.get("share_button_enabled", False),
            "queue_transition_to_publishing_at": queue_transition_to_publishing_at,
            "lock_created_at": lock_created_at,
            "share_clicked_at": share_clicked_at,
            "verification_finished_at": verification_finished_at,
            "share_click_count": share_click_count,
            "verification_signals": verification_signals or [],
            "publish_verified": publish_verified,
            "platform_post_id": platform_post_id,
            "platform_url": platform_url,
            "previous_status": previous_status,
            "final_status": final_status,
            "manual_check_required": manual_check_required,
            "lock_cleanup_result": lock_cleanup_result,
            "screenshots": screenshots,
            "result": result,
            "error": error,
            # Never included: cookies, passwords, session tokens, or any
            # other browser_profile/instagram content. Only paths,
            # booleans, counts, and short diagnostic strings ever land
            # in this payload.
        }

        temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary_path.replace(log_path)
        return log_path

    # -- orchestration: preflight (never sends) ------------------------------

    async def preflight(
        self,
        job: PublishJob,
        caption_result: FinalCaptionResult,
    ) -> FeedPublishResult:
        """
        Reach the exact same verified, ready-to-share state publish()
        would reach, using a fresh browser context, then stop. Never
        clicks Share, never acquires the operation lock, never touches
        publish_queue.json or the history log.
        """
        started_at = self._now()
        diagnostics = self._default_diagnostics()
        diagnostics["caption_length"] = caption_result.final_caption_length
        diagnostics["hashtags_appended"] = caption_result.hashtags_appended
        send_screenshots: dict[str, str | None] = {
            "before_share": None,
            "after_share": None,
            "verified": None,
            "error": None,
        }

        playwright, context = await self.session._open_context()

        try:
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                state = await self._reach_share_ready(
                    page,
                    job,
                    caption_result,
                    diagnostics,
                    login_failure_screenshot_name=f"feed_send_error_{job.job_id}.png",
                    before_upload_screenshot_name=f"feed_send_before_upload_{job.job_id}.png",
                    after_upload_screenshot_name=f"feed_send_after_upload_{job.job_id}.png",
                    final_screenshot_name=f"feed_send_before_share_{job.job_id}.png",
                )
            except FeedPreviewError as exc:
                send_screenshots["error"] = diagnostics["screenshots"].get("ready")
                self._write_send_log(
                    started_at=started_at,
                    job=job,
                    command="--preflight",
                    diagnostics=diagnostics,
                    screenshots=send_screenshots,
                    previous_status=job.status,
                    final_status=job.status,
                    result="failed",
                    error=str(exc),
                )
                raise

            share_lookup = state.share_lookup
            send_screenshots["before_share"] = diagnostics["screenshots"]["ready"]

            result = FeedPublishResult(
                job_id=job.job_id,
                production_date=job.production_date,
                persona_id=job.persona_id,
                login_status=state.login_status,
                media_path=state.media_path,
                media_uploaded=diagnostics["media_uploaded"],
                media_preview_verified=state.media_preview_verified,
                caption_verified=state.caption_verified,
                final_caption_length=caption_result.final_caption_length,
                share_button_found=True,
                share_button_enabled=share_lookup.enabled,
                share_button_interactive=share_lookup.interactive,
                share_click_count=0,
                share_clicked=False,
                verification_signals=[],
                publish_verified=False,
                published=False,
                platform_post_id=None,
                platform_url=None,
                previous_status=job.status,
                final_status=job.status,
                screenshots=dict(send_screenshots),
                error=None,
            )

            self._write_send_log(
                started_at=started_at,
                job=job,
                command="--preflight",
                diagnostics=diagnostics,
                screenshots=send_screenshots,
                previous_status=job.status,
                final_status=job.status,
                result="preflight_passed",
                error=None,
            )

            return result

        finally:
            await context.close()
            await playwright.stop()

    # -- orchestration: real publish ------------------------------------

    async def publish(
        self,
        job: PublishJob,
        caption_result: FinalCaptionResult,
    ) -> FeedPublishResult:
        started_at = self._now()

        lock = PublishLock(self.locks_dir)
        lock.acquire(job.job_id, command="--publish-approved")
        lock_created_at = self._now()

        diagnostics = self._default_diagnostics()
        diagnostics["caption_length"] = caption_result.final_caption_length
        diagnostics["hashtags_appended"] = caption_result.hashtags_appended
        send_screenshots: dict[str, str | None] = {
            "before_share": None,
            "after_share": None,
            "verified": None,
            "error": None,
        }
        previous_status = job.status

        playwright, context = await self.session._open_context()

        try:
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                state = await self._reach_share_ready(
                    page,
                    job,
                    caption_result,
                    diagnostics,
                    login_failure_screenshot_name=f"feed_send_error_{job.job_id}.png",
                    before_upload_screenshot_name=f"feed_send_before_upload_{job.job_id}.png",
                    after_upload_screenshot_name=f"feed_send_after_upload_{job.job_id}.png",
                    final_screenshot_name=f"feed_send_before_share_{job.job_id}.png",
                )
            except FeedPreviewError as exc:
                # Nothing irreversible has happened: release the lock so
                # a legitimate retry (after fixing whatever failed) can
                # proceed, and leave the queue untouched (still approved).
                send_screenshots["error"] = diagnostics["screenshots"].get("ready")
                lock.release(job.job_id)
                self._write_send_log(
                    started_at=started_at,
                    job=job,
                    command="--publish-approved",
                    diagnostics=diagnostics,
                    screenshots=send_screenshots,
                    previous_status=previous_status,
                    final_status=job.status,
                    lock_created_at=lock_created_at,
                    manual_check_required=False,
                    lock_cleanup_result="released",
                    result="failed",
                    error=str(exc),
                )
                raise

            share_lookup = state.share_lookup
            send_screenshots["before_share"] = diagnostics["screenshots"]["ready"]

            # -- mandatory terminal confirmation, right before the click --
            _print_confirmation_block(
                job, caption_result, share_candidates_found=share_lookup.candidate_count
            )

            # -- atomic approved -> publishing, immediately before click --
            def _mark_publishing(target: PublishJob) -> None:
                target.status = "publishing"

            self.queue_service.update_job(job.job_id, _mark_publishing)
            queue_transition_at = self._now()
            self.history_service.record_publishing_started(
                production_date=job.production_date,
                job_id=job.job_id,
                previous_status=previous_status,
            )

            # -- the one and only permitted Share click -------------------
            try:
                await share_lookup.element.click(
                    timeout=self.sender_config.share_click_timeout_ms
                )
            except Exception as exc:
                error = f"share_click_failed: {exc}"
                send_screenshots["error"] = await self._save_screenshot(
                    page, filename=f"feed_send_error_{job.job_id}.png"
                )

                def _mark_click_failed(target: PublishJob) -> None:
                    target.status = "failed"
                    target.error = error

                self.queue_service.update_job(job.job_id, _mark_click_failed)
                self.history_service.record_failed(
                    production_date=job.production_date,
                    job_id=job.job_id,
                    previous_status="publishing",
                    details={"error": error},
                )
                self._write_send_log(
                    started_at=started_at,
                    job=job,
                    command="--publish-approved",
                    diagnostics=diagnostics,
                    screenshots=send_screenshots,
                    previous_status=previous_status,
                    final_status="failed",
                    queue_transition_to_publishing_at=queue_transition_at,
                    lock_created_at=lock_created_at,
                    share_clicked_at=None,
                    share_click_count=0,
                    manual_check_required=True,
                    lock_cleanup_result="retained",
                    result="failed",
                    error=error,
                )
                raise ShareClickFailedError(error) from exc

            share_clicked_at = self._now()
            send_screenshots["after_share"] = await self._save_screenshot(
                page, filename=f"feed_send_after_share_{job.job_id}.png"
            )
            await page.wait_for_timeout(self.sender_config.verification_wait_ms)

            signals, verified = await self._verify_publish(page)
            verification_finished_at = self._now()
            platform_post_id, platform_url = self._extract_post_permalink(page.url)

            if verified:
                send_screenshots["verified"] = await self._save_screenshot(
                    page, filename=f"feed_send_verified_{job.job_id}.png"
                )
                published_at = self._now()

                def _mark_published(target: PublishJob) -> None:
                    target.status = "published"
                    target.published_at = published_at
                    target.error = None
                    if platform_post_id:
                        target.platform_post_id = platform_post_id
                    if platform_url:
                        target.platform_url = platform_url

                self.queue_service.update_job(job.job_id, _mark_published)
                self.history_service.record_published(
                    production_date=job.production_date,
                    job_id=job.job_id,
                    details={
                        "platform_post_id": platform_post_id,
                        "platform_url": platform_url,
                        "verification_signals": signals,
                    },
                )

                result = FeedPublishResult(
                    job_id=job.job_id,
                    production_date=job.production_date,
                    persona_id=job.persona_id,
                    login_status=state.login_status,
                    media_path=state.media_path,
                    media_uploaded=diagnostics["media_uploaded"],
                    media_preview_verified=state.media_preview_verified,
                    caption_verified=state.caption_verified,
                    final_caption_length=caption_result.final_caption_length,
                    share_button_found=True,
                    share_button_enabled=share_lookup.enabled,
                    share_button_interactive=share_lookup.interactive,
                    share_click_count=1,
                    share_clicked=True,
                    verification_signals=signals,
                    publish_verified=True,
                    published=True,
                    platform_post_id=platform_post_id,
                    platform_url=platform_url,
                    previous_status=previous_status,
                    final_status="published",
                    screenshots=dict(send_screenshots),
                    error=None,
                )

                self._write_send_log(
                    started_at=started_at,
                    job=job,
                    command="--publish-approved",
                    diagnostics=diagnostics,
                    screenshots=send_screenshots,
                    previous_status=previous_status,
                    final_status="published",
                    queue_transition_to_publishing_at=queue_transition_at,
                    lock_created_at=lock_created_at,
                    share_clicked_at=share_clicked_at,
                    verification_finished_at=verification_finished_at,
                    share_click_count=1,
                    verification_signals=signals,
                    publish_verified=True,
                    platform_post_id=platform_post_id,
                    platform_url=platform_url,
                    manual_check_required=False,
                    lock_cleanup_result="retained",
                    result="published",
                    error=None,
                )

                return result

            # -- inconclusive: NEVER treated as published, NEVER retried --
            error = "publish_verification_inconclusive_manual_check_required"
            send_screenshots["error"] = await self._save_screenshot(
                page, filename=f"feed_send_error_{job.job_id}.png"
            )

            def _mark_inconclusive(target: PublishJob) -> None:
                target.status = "failed"
                target.error = error

            self.queue_service.update_job(job.job_id, _mark_inconclusive)
            self.history_service.record_failed(
                production_date=job.production_date,
                job_id=job.job_id,
                previous_status="publishing",
                details={"error": error, "verification_signals": signals},
            )
            self._write_send_log(
                started_at=started_at,
                job=job,
                command="--publish-approved",
                diagnostics=diagnostics,
                screenshots=send_screenshots,
                previous_status=previous_status,
                final_status="failed",
                queue_transition_to_publishing_at=queue_transition_at,
                lock_created_at=lock_created_at,
                share_clicked_at=share_clicked_at,
                verification_finished_at=verification_finished_at,
                share_click_count=1,
                verification_signals=signals,
                publish_verified=False,
                manual_check_required=True,
                lock_cleanup_result="retained",
                result="failed",
                error=error,
            )

            print()
            print("!" * 72)
            print("WARNING: Share was clicked but publication could NOT be")
            print("verified. Please inspect the real Instagram account")
            print("manually before doing anything else with this job_id.")
            print("No automatic retry will occur.")
            print("!" * 72)
            print()

            raise PublishVerificationInconclusiveError(error)

        finally:
            await context.close()
            await playwright.stop()


def build_feed_sender_controller(
    *,
    publisher_config_path: str | Path | None = None,
    instagram_config_path: str | Path | None = None,
    headless_override: bool | None = None,
) -> tuple[InstagramFeedSenderController, PublisherConfig]:
    publisher_config = load_publisher_config(publisher_config_path)
    session_config = load_session_config(
        config_path=instagram_config_path,
        headless_override=headless_override,
    )
    feed_config = load_feed_preview_config(config_path=instagram_config_path)
    sender_config = load_feed_sender_config(config_path=instagram_config_path)
    session = InstagramSession(session_config)
    queue_service = build_queue_service(publisher_config)
    history_service = build_history_service(publisher_config)

    controller = InstagramFeedSenderController(
        session=session,
        feed_config=feed_config,
        publisher_config=publisher_config,
        sender_config=sender_config,
        queue_service=queue_service,
        history_service=history_service,
    )
    return controller, publisher_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Instagram Feed Manual Publish (Phase 10C). "
            "--publish-approved sends exactly one approved Feed job for "
            "real. --preflight verifies the send is ready without "
            "sending anything. No batch mode, no scheduling, no retry."
        )
    )

    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument(
        "--publish-approved",
        dest="publish_approved",
        metavar="JOB_ID",
        default=None,
        help="Publish the approved instagram_feed job_id for real. Requires --confirm.",
    )

    action.add_argument(
        "--preflight",
        dest="preflight",
        metavar="JOB_ID",
        default=None,
        help=(
            "Verify the send is ready for this job_id without sending "
            "anything. Does not require --confirm."
        ),
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "Mandatory explicit confirmation when using --publish-approved. "
            "Not required (and has no effect) with --preflight."
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate config/publishing/publisher.yaml.",
    )

    parser.add_argument(
        "--instagram-config",
        default=None,
        help="Path to an alternate config/social/instagram.yaml.",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode (visible browser is the default).",
    )

    arguments = parser.parse_args(argv)

    if arguments.publish_approved and not arguments.confirm:
        parser.error("--confirm is required when using --publish-approved")

    return arguments


def _print_result(result: FeedPublishResult, *, title: str) -> None:
    print()
    print(title)
    print("-" * len(title))
    print(f"job id:                   {result.job_id}")
    print(f"login status:             {result.login_status}")
    print(f"media path:               {result.media_path}")
    print(f"media uploaded:           {result.media_uploaded}")
    print(f"media preview verified:   {result.media_preview_verified}")
    print(f"caption verified:         {result.caption_verified}")
    print(f"final caption length:     {result.final_caption_length}")
    print(f"share button found:       {result.share_button_found}")
    print(f"share button enabled:     {result.share_button_enabled}")
    print(f"share button interactive: {result.share_button_interactive}")
    print(f"share click count:        {result.share_click_count}")
    print(f"share clicked:            {result.share_clicked}")
    print(f"verification signals:     {result.verification_signals}")
    print(f"publish verified:         {result.publish_verified}")
    print(f"published:                {result.published}")
    print(f"platform post id:         {result.platform_post_id}")
    print(f"platform url:             {result.platform_url}")
    print(f"previous status:          {result.previous_status}")
    print(f"final status:             {result.final_status}")
    print(f"screenshots:              {result.screenshots}")
    print()


async def _run(arguments: argparse.Namespace) -> int:
    publisher_config = load_publisher_config(arguments.config)
    target_job_id = arguments.publish_approved or arguments.preflight

    job = load_job_by_id(publisher_config.queue_path(), target_job_id)

    if job is None:
        print(f"[InstagramFeedSender] No job found for job_id: {target_job_id}")
        return 1

    try:
        check_job_publishable(job)
    except (FeedPreviewIneligibleError, FeedSenderIneligibleError) as exc:
        print(f"[InstagramFeedSender] {exc}")
        return 1

    limit = (
        publisher_config.platform_limits.get("instagram_feed", {})
        .get("caption_max_characters")
    )

    try:
        caption_result = build_final_caption(
            job.caption_text or "",
            list(job.hashtags),
            max_characters=limit,
        )
    except CaptionLimitExceededError as exc:
        print(f"[InstagramFeedSender] {exc}")
        return 1

    controller, _ = build_feed_sender_controller(
        publisher_config_path=arguments.config,
        instagram_config_path=arguments.instagram_config,
        headless_override=True if arguments.headless else None,
    )

    if arguments.preflight:
        try:
            result = await controller.preflight(job, caption_result)
        except (InstagramSessionError, FeedPreviewError) as exc:
            print(f"[InstagramFeedSender] preflight failed: {exc}")
            return 1

        _print_result(result, title="AIKO Instagram Feed Sender — Preflight (no send)")
        return 0

    try:
        result = await controller.publish(job, caption_result)
    except PublishLockError as exc:
        print(f"[InstagramFeedSender] {exc}")
        return 1
    except PublishVerificationInconclusiveError as exc:
        print(f"[InstagramFeedSender] {exc}")
        return 1
    except (InstagramSessionError, FeedPreviewError, FeedSenderError) as exc:
        print(f"[InstagramFeedSender] publish failed: {exc}")
        return 1

    _print_result(result, title="AIKO Instagram Feed Sender — PUBLISHED")
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
