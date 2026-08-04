from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.social.instagram_session import (
    InstagramSession,
    PlaywrightTimeoutError,
    load_config as load_session_config,
)
from src.social.instagram_models import InstagramSessionError

from .models import PublisherConfig, PublishJob, load_publisher_config
from .queue_service import QueueService, load_job_by_id

# Phase 10D.1 is a PREVIEW ONLY module. It opens the real, logged-in
# Instagram session, prepares a Story completely — Story composer entry,
# media upload, canvas/CSS/video/image preview verification — locates
# every possible Story submission control (Your story, Share, Share to
# story, Close Friends) for verification only, and STOPS. It must never
# grow, and never accidentally gain, any of the capabilities below.
#
# This module is fully self-contained: it does not import from, subclass,
# or otherwise depend on instagram_feed_preview.py / instagram_feed_sender.py
# / publish_recovery.py / history_service.py, so nothing here can ever
# change Feed publish behavior, and it never writes to publish_queue.json
# or any history file.
DISALLOWED_ACTIONS = (
    "share",
    "publish",
    "submit",
    "press",
    "press_enter",
    "click_share",
    "click_your_story",
    "click_share_to_story",
    "click_close_friends",
    "force_click",
    "click_via_js",
    "post_feed",
    "post_reel",
    "like",
    "follow",
    "unfollow",
    "comment",
    "reply",
    "send_dm",
    "delete",
)

DEFAULT_PUBLISHER_CONFIG_RELATIVE_PATH = Path("config") / "publishing" / "publisher.yaml"
DEFAULT_INSTAGRAM_CONFIG_RELATIVE_PATH = Path("config") / "social" / "instagram.yaml"

STORY_ORDER_MIN = 1
STORY_ORDER_MAX = 4

OPTIONAL_FEATURE_NAMES: tuple[str, ...] = (
    "text_overlay",
    "mention",
    "location",
    "music",
    "link",
    "poll",
    "questions",
    "gif",
    "stickers",
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov"})


def _runtime_root() -> Path:
    """
    instagram_story_preview.py location:

    10_apps/claude_runtime/src/publishing/instagram_story_preview.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StoryPreviewError(RuntimeError):
    """Base error for the Phase 10D.1 Instagram Story preview controller."""


class StoryPreviewConfigError(StoryPreviewError):
    """Raised when config/social/instagram.yaml's story_preview section is missing/invalid."""


class JobNotFoundError(StoryPreviewError):
    """Raised when the given job_id does not exist in the publish queue."""


class StoryPreviewIneligibleError(StoryPreviewError):
    """Raised when the job does not meet the Phase 10D.1 eligibility rules."""


class StorySetValidationError(StoryPreviewError):
    """Raised when the target job's Story sibling set fails a cross-job safety check."""


class StoryPreviewNavigationError(StoryPreviewError):
    """Raised when Instagram cannot be opened at all."""


class StoryPreviewLoginError(StoryPreviewError):
    """Raised when the Instagram session is not logged_in."""


class CreateControlNotFoundError(StoryPreviewError):
    """Raised when the Story create_control cannot be located or clicked."""


class StoryOptionNotFoundError(StoryPreviewError):
    """Raised when story_option selectors are configured but none can be located."""


class MediaUploadError(StoryPreviewError):
    """Raised when the media file cannot be uploaded, or Instagram reports it unsupported."""


class MediaPreviewNotVerifiedError(StoryPreviewError):
    """Raised when no media canvas/preview element can be found after upload."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StoryPreviewSelectors:
    """
    Centralised CSS/Playwright selectors for the Story preview controller.

    UNTESTED placeholder defaults — see the story_preview: section comment
    block in config/social/instagram.yaml for how to tune these against a
    real session. There is deliberately no post_option/reel_option
    selector anywhere here — Post/Reel cannot be chosen even accidentally.
    """

    create_control: list[str] = field(default_factory=list)
    story_option: list[str] = field(default_factory=list)
    file_input: list[str] = field(default_factory=list)
    media_canvas: list[str] = field(default_factory=list)
    unsupported_media_error: list[str] = field(default_factory=list)
    share_candidates: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StoryPreviewConfig:
    """Fully-resolved runtime configuration for the Story preview controller."""

    selectors: StoryPreviewSelectors
    action_click_timeout_ms: int
    upload_settle_wait_ms: int
    navigation_settle_wait_ms: int
    create_menu_wait_ms: int
    dialog_open_wait_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selectors": self.selectors.to_dict(),
            "action_click_timeout_ms": self.action_click_timeout_ms,
            "upload_settle_wait_ms": self.upload_settle_wait_ms,
            "navigation_settle_wait_ms": self.navigation_settle_wait_ms,
            "create_menu_wait_ms": self.create_menu_wait_ms,
            "dialog_open_wait_ms": self.dialog_open_wait_ms,
        }


def load_story_preview_config(
    *,
    config_path: str | Path | None = None,
) -> StoryPreviewConfig:
    """
    Load the story_preview section from config/social/instagram.yaml.

    Independent loader, matching the pattern already used by every other
    module in this package (instagram_feed_preview.py included) — this
    function does not call that module's loader.
    """
    runtime_root = _runtime_root()

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (runtime_root / DEFAULT_INSTAGRAM_CONFIG_RELATIVE_PATH).resolve()
    )

    if not resolved_config_path.exists():
        raise StoryPreviewConfigError(
            f"Instagram session config not found: {resolved_config_path}"
        )

    try:
        with resolved_config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise StoryPreviewConfigError(
            f"Invalid YAML in {resolved_config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict) or not raw:
        raise StoryPreviewConfigError(
            f"Instagram session config is empty or invalid: {resolved_config_path}"
        )

    section = raw.get("story_preview") or {}
    selectors_section = section.get("selectors") or {}
    timeouts_section = section.get("timeouts") or {}

    share_candidates_section = selectors_section.get("share_candidates") or {}
    share_candidates = {
        str(name): list(candidate_selectors or [])
        for name, candidate_selectors in share_candidates_section.items()
    }

    selectors = StoryPreviewSelectors(
        create_control=list(selectors_section.get("create_control", [])),
        story_option=list(selectors_section.get("story_option", [])),
        file_input=list(selectors_section.get("file_input", [])),
        media_canvas=list(selectors_section.get("media_canvas", [])),
        unsupported_media_error=list(
            selectors_section.get("unsupported_media_error", [])
        ),
        share_candidates=share_candidates,
    )

    return StoryPreviewConfig(
        selectors=selectors,
        action_click_timeout_ms=int(
            timeouts_section.get("action_click_timeout_ms", 5000)
        ),
        upload_settle_wait_ms=int(
            timeouts_section.get("upload_settle_wait_ms", 3000)
        ),
        navigation_settle_wait_ms=int(
            timeouts_section.get("navigation_settle_wait_ms", 9000)
        ),
        create_menu_wait_ms=int(timeouts_section.get("create_menu_wait_ms", 1500)),
        dialog_open_wait_ms=int(timeouts_section.get("dialog_open_wait_ms", 2500)),
    )


# ---------------------------------------------------------------------------
# Eligibility (pure, no browser)
# ---------------------------------------------------------------------------


def _story_order(job: PublishJob) -> int:
    """
    Return job.metadata['story_order'] as a validated int, or raise
    StoryPreviewIneligibleError. bool is rejected explicitly since Python's
    bool is a subclass of int (isinstance(True, int) is True).
    """
    story_order = job.metadata.get("story_order")

    if story_order is None:
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} has no metadata.story_order"
        )

    if not isinstance(story_order, int) or isinstance(story_order, bool):
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} metadata.story_order must be an integer, "
            f"got {story_order!r}"
        )

    if not (STORY_ORDER_MIN <= story_order <= STORY_ORDER_MAX):
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} metadata.story_order must be between "
            f"{STORY_ORDER_MIN} and {STORY_ORDER_MAX}, got {story_order}"
        )

    return story_order


def check_story_job_eligible(job: PublishJob) -> int:
    """
    Raise StoryPreviewIneligibleError unless the job may be previewed.
    Returns the validated story_order on success.

    Runs entirely against local data (the job dict already loaded from the
    queue) — no browser is opened and the queue is never touched by this
    check.
    """
    if job.content_type != "instagram_story":
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} is not an instagram_story job "
            f"(content_type={job.content_type!r})"
        )

    if job.status != "approved":
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} is not eligible for preview "
            f"(status={job.status!r}; only 'approved' jobs are eligible)"
        )

    if len(job.media_paths) != 1:
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} must have exactly one media path, "
            f"found {len(job.media_paths)}"
        )

    media_path = Path(job.media_paths[0])

    if not media_path.is_file():
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} media file does not exist: {media_path}"
        )

    if media_path.stat().st_size == 0:
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} media file is zero bytes: {media_path}"
        )

    story_order = _story_order(job)

    if job.published_at is not None:
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} already has published_at set "
            f"({job.published_at}); refusing to preview"
        )

    if job.platform_post_id is not None:
        raise StoryPreviewIneligibleError(
            f"Job {job.job_id} already has a platform_post_id "
            f"({job.platform_post_id}); refusing to preview"
        )

    return story_order


# ---------------------------------------------------------------------------
# Story-set validation (pure, no browser)
# ---------------------------------------------------------------------------


def load_story_siblings(queue_path: str | Path, production_date: str) -> list[PublishJob]:
    """
    Read-only: load every instagram_story job for one production date.

    The only queue access this module performs. QueueService.load_jobs()
    is a read method; this function never calls save_jobs()/update_job()/
    add_job()/upsert_jobs() on the QueueService it constructs.
    """
    jobs = QueueService(queue_path).load_jobs()
    return [
        job
        for job in jobs
        if job.content_type == "instagram_story" and job.production_date == production_date
    ]


def validate_story_set(job: PublishJob, sibling_jobs: list[PublishJob]) -> None:
    """
    Raise StorySetValidationError unless the target job's Story sibling
    set (all instagram_story jobs sharing its production_date) is
    internally consistent. Never mutates or saves sibling_jobs — this
    function performs no queue write of any kind.
    """
    by_id = {sibling.job_id: sibling for sibling in sibling_jobs}
    queue_record = by_id.get(job.job_id)

    if queue_record is None:
        raise StorySetValidationError(
            f"Job {job.job_id} was not found among its own Story sibling set"
        )

    if queue_record.media_paths != job.media_paths:
        raise StorySetValidationError(
            f"Job {job.job_id} media path does not match the queue record "
            f"({queue_record.media_paths!r} != {job.media_paths!r})"
        )

    if queue_record.metadata.get("story_order") != job.metadata.get("story_order"):
        raise StorySetValidationError(
            f"Job {job.job_id} story_order does not match the queue record"
        )

    orders_seen: dict[int, list[str]] = {}
    media_seen: dict[str, list[str]] = {}

    for sibling in sibling_jobs:
        order = sibling.metadata.get("story_order")
        if isinstance(order, int) and not isinstance(order, bool):
            orders_seen.setdefault(order, []).append(sibling.job_id)

        for media_path in sibling.media_paths:
            media_seen.setdefault(media_path, []).append(sibling.job_id)

    for order, job_ids in orders_seen.items():
        if len(job_ids) > 1:
            raise StorySetValidationError(
                f"Duplicate story_order {order} shared by: {', '.join(sorted(job_ids))}"
            )

    for media_path, job_ids in media_seen.items():
        if len(job_ids) > 1:
            raise StorySetValidationError(
                f"Duplicate Story media path {media_path!r} shared by: "
                f"{', '.join(sorted(job_ids))}"
            )


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StoryPreviewResult:
    """
    Summary of one instagram_story_preview --preview run.

    Only ever constructed on full success — every failure mode raises a
    typed StoryPreviewError subclass instead. share_clicked and published
    are always False: there is no assignment site in this module that
    could ever set either to True.
    """

    job_id: str
    production_date: str
    persona_id: str
    story_order: int
    login_status: str
    media_path: str
    media_type: str
    media_uploaded: bool
    media_preview_verified: bool
    share_candidates: dict[str, int]
    share_button_found: bool
    share_button_enabled: bool
    share_clicked: bool
    published: bool
    optional_features: dict[str, str]
    screenshots: dict[str, str | None]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _media_type(media_path: str) -> str:
    extension = Path(media_path).suffix.lower()

    if extension in VIDEO_EXTENSIONS:
        return "video"

    if extension in IMAGE_EXTENSIONS:
        return "image"

    return "unknown"


def _default_optional_features() -> dict[str, str]:
    return {name: "not_applied" for name in OPTIONAL_FEATURE_NAMES}


@dataclass(slots=True)
class _ElementLookup:
    """
    Internal result of _locate_unique_visible_enabled(). A candidate is
    only returned when it is the SOLE visible, enabled match for some
    selector — zero or multiple candidates both fail safely (element is
    None) rather than guessing.
    """

    element: Any | None
    selector_string: str | None
    candidate_count: int
    ambiguous: bool = False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class InstagramStoryPreviewController:
    """
    Prepares one approved instagram_story PublishJob completely, verifies
    every possible Story submission control is present, then stops. Never
    clicks any submission control, never presses Enter, never evaluates JS
    to click, never force-clicks, never writes to publish_queue.json (it
    never even holds a writable QueueService — see load_story_siblings()),
    and never records a publish-history transition (history_service is
    never imported).

    Fully self-contained: does not subclass or import
    InstagramFeedPreviewController.
    """

    def __init__(
        self,
        *,
        session: InstagramSession,
        story_config: StoryPreviewConfig,
        publisher_config: PublisherConfig,
        screenshots_dir: str | Path | None = None,
        logs_dir: str | Path | None = None,
    ) -> None:
        self.session = session
        self.story_config = story_config
        self.publisher_config = publisher_config
        self.screenshots_dir = (
            Path(screenshots_dir) if screenshots_dir else publisher_config.screenshots_dir()
        )
        self.logs_dir = Path(logs_dir) if logs_dir else publisher_config.logs_dir()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- generic selector helpers --------------------------------------

    @staticmethod
    async def _find_first(root: Any, selectors: list[str]) -> Any:
        for selector in selectors:
            try:
                element = await root.query_selector(selector)
            except Exception:
                element = None

            if element is not None:
                return element

        return None

    async def _locate_unique_visible_enabled(
        self,
        root: Any,
        selectors: list[str],
    ) -> _ElementLookup:
        total_candidates = 0

        for selector in selectors:
            try:
                candidates = await root.query_selector_all(selector)
            except Exception:
                candidates = []

            if not candidates:
                continue

            total_candidates += len(candidates)
            valid = []

            for candidate in candidates:
                try:
                    visible = await candidate.is_visible()
                except Exception:
                    visible = False

                try:
                    enabled = await candidate.is_enabled()
                except Exception:
                    enabled = False

                if visible and enabled:
                    valid.append(candidate)

            if len(valid) == 1:
                return _ElementLookup(
                    element=valid[0],
                    selector_string=selector,
                    candidate_count=len(candidates),
                )

            if len(valid) > 1:
                return _ElementLookup(
                    element=None,
                    selector_string=selector,
                    candidate_count=len(candidates),
                    ambiguous=True,
                )

        return _ElementLookup(
            element=None,
            selector_string=None,
            candidate_count=total_candidates,
        )

    async def _count_visible_enabled(self, root: Any, selectors: list[str]) -> int:
        """
        Count visible+enabled matches across a selector fallback chain,
        for the share-candidate safety gate. Unlike
        _locate_unique_visible_enabled(), this never returns an element
        and never needs to disambiguate — Story composers may legitimately
        show more than one distinct named control at once, so the caller
        only ever needs a count, never a handle to click.
        """
        count = 0

        for selector in selectors:
            try:
                candidates = await root.query_selector_all(selector)
            except Exception:
                candidates = []

            for candidate in candidates:
                try:
                    visible = await candidate.is_visible()
                except Exception:
                    visible = False

                try:
                    enabled = await candidate.is_enabled()
                except Exception:
                    enabled = False

                if visible and enabled:
                    count += 1

        return count

    async def _detect_login(self, page: Any) -> tuple[str, str]:
        detection = self.session.config.login_detection

        found_logged_in = [
            selector
            for selector in detection.logged_in_selectors
            if await self._find_first(page, [selector]) is not None
        ]

        found_logged_out = [
            selector
            for selector in detection.logged_out_selectors
            if await self._find_first(page, [selector]) is not None
        ]

        return InstagramSession.classify(
            current_url=page.url,
            found_logged_in_selectors=found_logged_in,
            found_logged_out_selectors=found_logged_out,
            logged_out_url_markers=detection.logged_out_url_markers,
            pending_url_markers=detection.pending_url_markers,
        )

    # -- diagnostics ----------------------------------------------------

    async def _save_screenshot(self, page: Any, *, filename: str) -> str | None:
        try:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
            path = self.screenshots_dir / filename
            await page.screenshot(path=str(path), full_page=True)
            return str(path)
        except Exception:
            # Screenshot capture is best-effort diagnostics only.
            return None

    def _write_log(
        self,
        *,
        started_at: str,
        command: str,
        job: PublishJob,
        story_order: int,
        diagnostics: dict[str, Any],
        result: str,
        error: str | None,
    ) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        finished_at = self._now()
        run_id = finished_at.replace(":", "").replace("-", "").replace("+", "").replace(".", "")
        log_path = self.logs_dir / f"instagram_story_preview_{job.job_id}_{run_id}.json"

        payload = {
            "started_at": started_at,
            "finished_at": finished_at,
            "command": command,
            "job_id": job.job_id,
            "production_date": job.production_date,
            "persona_id": job.persona_id,
            "story_order": story_order,
            "media_path": job.media_paths[0] if job.media_paths else None,
            "media_type": diagnostics.get("media_type"),
            "login_status": diagnostics.get("login_status"),
            "create_control_found": diagnostics.get("create_control_found", False),
            "story_option_selected": diagnostics.get("story_option_selected", False),
            "media_uploaded": diagnostics.get("media_uploaded", False),
            "media_preview_verified": diagnostics.get("media_preview_verified", False),
            "share_candidates": diagnostics.get("share_candidates", {}),
            "share_clicked": False,
            "published": False,
            "optional_features": diagnostics.get("optional_features", {}),
            "screenshots": diagnostics.get("screenshots", {}),
            "result": result,
            "error": error,
            # Never included: cookies, passwords, session tokens, or any
            # other browser_profile/instagram content, nor caption/DM/
            # comment private data.
        }

        temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary_path.replace(log_path)
        return log_path

    @staticmethod
    def _default_diagnostics() -> dict[str, Any]:
        return {
            "login_status": "unknown",
            "create_control_found": False,
            "story_option_selected": False,
            "media_uploaded": False,
            "media_preview_verified": False,
            "media_type": None,
            "share_candidates": {},
            "optional_features": _default_optional_features(),
            "screenshots": {
                "before_upload": None,
                "after_upload": None,
                "ready": None,
                "error": None,
            },
        }

    # -- orchestration ----------------------------------------------------

    async def preview(self, job: PublishJob, story_order: int) -> StoryPreviewResult:
        """
        Prepare the Story completely, verify the media rendered, locate
        (never click) every possible submission control, then stop.
        """
        started_at = self._now()
        diagnostics = self._default_diagnostics()
        media_path = job.media_paths[0]
        media_type = _media_type(media_path)
        diagnostics["media_type"] = media_type

        playwright, context = await self.session._open_context()

        try:
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                result = await self._run_preview(page, job, story_order, media_path, media_type, diagnostics)
            except StoryPreviewError as exc:
                diagnostics["screenshots"]["error"] = await self._save_screenshot(
                    page, filename=f"story_preview_error_{job.job_id}.png"
                )
                self._write_log(
                    started_at=started_at,
                    command="--preview",
                    job=job,
                    story_order=story_order,
                    diagnostics=diagnostics,
                    result="failed",
                    error=str(exc),
                )
                raise

            self._write_log(
                started_at=started_at,
                command="--preview",
                job=job,
                story_order=story_order,
                diagnostics=diagnostics,
                result="ready_for_share",
                error=None,
            )

            return result

        finally:
            await context.close()
            await playwright.stop()

    async def _run_preview(
        self,
        page: Any,
        job: PublishJob,
        story_order: int,
        media_path: str,
        media_type: str,
        diagnostics: dict[str, Any],
    ) -> StoryPreviewResult:
        # -- navigation + login ------------------------------------------
        try:
            await page.goto(
                self.session.config.base_url,
                timeout=self.session.config.navigation_timeout_ms,
                wait_until="domcontentloaded",
            )
        except PlaywrightTimeoutError as exc:
            raise StoryPreviewNavigationError(
                f"Timed out opening Instagram: {exc}"
            ) from exc
        except Exception as exc:
            raise StoryPreviewNavigationError(
                f"Unable to open Instagram: {exc}"
            ) from exc

        await page.wait_for_timeout(self.story_config.navigation_settle_wait_ms)

        login_status, _login_reason = await self._detect_login(page)
        diagnostics["login_status"] = login_status

        if login_status != "logged_in":
            raise StoryPreviewLoginError(
                f"Instagram session is not logged_in (status={login_status})"
            )

        # -- open the Story composer --------------------------------------
        create_lookup = await self._locate_unique_visible_enabled(
            page, self.story_config.selectors.create_control
        )

        if create_lookup.element is None:
            raise CreateControlNotFoundError(
                "Could not locate a unique, visible, enabled Story "
                f"create_control (candidates={create_lookup.candidate_count}, "
                f"ambiguous={create_lookup.ambiguous})."
            )

        await create_lookup.element.click(timeout=self.story_config.action_click_timeout_ms)
        diagnostics["create_control_found"] = True
        await page.wait_for_timeout(self.story_config.create_menu_wait_ms)

        story_option_selectors = self.story_config.selectors.story_option

        if story_option_selectors:
            option_lookup = await self._locate_unique_visible_enabled(
                page, story_option_selectors
            )

            if option_lookup.element is None:
                raise StoryOptionNotFoundError(
                    "story_option selectors are configured but no unique, "
                    "visible, enabled candidate could be located "
                    f"(candidates={option_lookup.candidate_count}, "
                    f"ambiguous={option_lookup.ambiguous})."
                )

            await option_lookup.element.click(timeout=self.story_config.action_click_timeout_ms)
            diagnostics["story_option_selected"] = True
            await page.wait_for_timeout(self.story_config.dialog_open_wait_ms)
        else:
            # No intermediate menu configured: create_control is expected
            # to have opened the Story composer directly.
            diagnostics["story_option_selected"] = True
            await page.wait_for_timeout(self.story_config.dialog_open_wait_ms)

        diagnostics["screenshots"]["before_upload"] = await self._save_screenshot(
            page, filename=f"story_preview_before_upload_{job.job_id}.png"
        )

        # -- upload media ---------------------------------------------------
        file_input = await self._find_first(page, self.story_config.selectors.file_input)

        if file_input is None:
            raise MediaUploadError("Could not locate the Story media file input.")

        try:
            await file_input.set_input_files(media_path)
        except Exception as exc:
            raise MediaUploadError(
                f"Failed to upload media file {media_path}: {exc}"
            ) from exc

        await page.wait_for_timeout(self.story_config.upload_settle_wait_ms)

        unsupported = await self._find_first(
            page, self.story_config.selectors.unsupported_media_error
        )

        if unsupported is not None:
            raise MediaUploadError(
                f"Instagram reported unsupported media for {media_path!r}."
            )

        diagnostics["media_uploaded"] = True

        diagnostics["screenshots"]["after_upload"] = await self._save_screenshot(
            page, filename=f"story_preview_after_upload_{job.job_id}.png"
        )

        # -- Story canvas verification ---------------------------------
        media_canvas = await self._find_first(page, self.story_config.selectors.media_canvas)
        media_preview_verified = media_canvas is not None
        diagnostics["media_preview_verified"] = media_preview_verified

        if not media_preview_verified:
            raise MediaPreviewNotVerifiedError(
                "Story media canvas/preview could not be verified after upload."
            )

        # -- optional Story elements: inspect only, never applied ------
        optional_features = _default_optional_features()
        diagnostics["optional_features"] = optional_features

        # -- share-control safety gate: locate + count ONLY -------------
        share_candidates: dict[str, int] = {}

        for name, selectors in self.story_config.selectors.share_candidates.items():
            share_candidates[name] = await self._count_visible_enabled(page, selectors)

        diagnostics["share_candidates"] = share_candidates
        share_button_found = any(count > 0 for count in share_candidates.values())
        # Only visible+enabled elements are ever counted, so "found" and
        # "enabled" are the same fact reported under two names, matching
        # the StoryPreviewResult field the spec requires.
        share_button_enabled = share_button_found

        diagnostics["screenshots"]["ready"] = await self._save_screenshot(
            page, filename=f"story_preview_ready_{job.job_id}.png"
        )

        return StoryPreviewResult(
            job_id=job.job_id,
            production_date=job.production_date,
            persona_id=job.persona_id,
            story_order=story_order,
            login_status=login_status,
            media_path=media_path,
            media_type=media_type,
            media_uploaded=diagnostics["media_uploaded"],
            media_preview_verified=media_preview_verified,
            share_candidates=share_candidates,
            share_button_found=share_button_found,
            share_button_enabled=share_button_enabled,
            share_clicked=False,
            published=False,
            optional_features=optional_features,
            screenshots=dict(diagnostics["screenshots"]),
            error=None,
        )


def build_story_preview_controller(
    *,
    publisher_config_path: str | Path | None = None,
    instagram_config_path: str | Path | None = None,
    headless_override: bool | None = None,
) -> tuple[InstagramStoryPreviewController, PublisherConfig]:
    publisher_config = load_publisher_config(publisher_config_path)
    session_config = load_session_config(
        config_path=instagram_config_path,
        headless_override=headless_override,
    )
    story_config = load_story_preview_config(config_path=instagram_config_path)
    session = InstagramSession(session_config)

    controller = InstagramStoryPreviewController(
        session=session,
        story_config=story_config,
        publisher_config=publisher_config,
    )
    return controller, publisher_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Instagram Story Publisher — Preview Only (Phase 10D.1). "
            "Prepares an approved instagram_story job through to a "
            "ready-to-share state and stops. Never clicks any submission "
            "control."
        )
    )

    parser.add_argument(
        "--preview",
        metavar="JOB_ID",
        required=True,
        help="Preview this approved instagram_story job_id.",
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

    return parser.parse_args(argv)


def _print_pre_browser_block(job: PublishJob, story_order: int) -> None:
    print()
    print("AIKO Instagram Story Publisher — Preview Only (Phase 10D.1)")
    print("--------------------------------------------------------------")
    print(f"job id:            {job.job_id}")
    print(f"production date:   {job.production_date}")
    print(f"persona id:        {job.persona_id}")
    print(f"story order:       {story_order}")
    print(f"media path:        {job.media_paths[0]}")
    print(f"media type:        {_media_type(job.media_paths[0])}")
    print(f"current status:    {job.status}")
    print(f"approved_at:       {job.approved_at}")
    print()


def _print_result(result: StoryPreviewResult) -> None:
    print()
    print(f"job id:                  {result.job_id}")
    print(f"login status:            {result.login_status}")
    print(f"media path:              {result.media_path}")
    print(f"media type:              {result.media_type}")
    print(f"media uploaded:          {result.media_uploaded}")
    print(f"media preview verified:  {result.media_preview_verified}")
    print(f"optional features:       {result.optional_features}")
    print(f"share candidates:        {result.share_candidates}")
    print(f"share button found:      {result.share_button_found}")
    print(f"share button enabled:    {result.share_button_enabled}")
    print(f"share clicked:           {result.share_clicked}")
    print(f"published:               {result.published}")
    print(f"screenshots:             {result.screenshots}")
    print()


async def _run(arguments: argparse.Namespace) -> int:
    publisher_config = load_publisher_config(arguments.config)

    job = load_job_by_id(publisher_config.queue_path(), arguments.preview)

    if job is None:
        print(f"[InstagramStoryPreview] No job found for job_id: {arguments.preview}")
        return 1

    try:
        story_order = check_story_job_eligible(job)
    except StoryPreviewIneligibleError as exc:
        print(f"[InstagramStoryPreview] {exc}")
        return 1

    sibling_jobs = load_story_siblings(publisher_config.queue_path(), job.production_date)

    try:
        validate_story_set(job, sibling_jobs)
    except StorySetValidationError as exc:
        print(f"[InstagramStoryPreview] {exc}")
        return 1

    _print_pre_browser_block(job, story_order)

    controller, _ = build_story_preview_controller(
        publisher_config_path=arguments.config,
        instagram_config_path=arguments.instagram_config,
        headless_override=True if arguments.headless else None,
    )

    try:
        result = await controller.preview(job, story_order)
    except InstagramSessionError as exc:
        print(f"[InstagramStoryPreview] preview failed: {exc}")
        return 1
    except StoryPreviewError as exc:
        print(f"[InstagramStoryPreview] preview failed: {exc}")
        return 1

    _print_result(result)
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
