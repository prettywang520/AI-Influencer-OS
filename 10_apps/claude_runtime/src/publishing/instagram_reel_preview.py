from __future__ import annotations

import argparse
import asyncio
import json
import math
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

from .instagram_feed_preview import (
    CaptionLimitExceededError,
    FinalCaptionResult,
    build_final_caption,
)
from .models import PublisherConfig, PublishJob, app_root, load_publisher_config
from .queue_service import load_job_by_id

# Phase 10E.1 is a PREVIEW ONLY module. It opens the real, logged-in
# Instagram session, prepares a Reel completely — Create -> Reel (never
# Post/Story/Live), video upload, processing wait, Next transitions, cover
# and caption verification — locates the Share button for verification
# only, and STOPS. It must never grow, and never accidentally gain, any of
# the capabilities below.
#
# This module is fully self-contained: it does not subclass or depend on
# InstagramFeedPreviewController / InstagramStoryPreviewController /
# instagram_feed_sender.py / publish_recovery.py / history_service.py. It
# only imports pure, browser-free caption-building functions from
# instagram_feed_preview.py (build_final_caption, FinalCaptionResult,
# CaptionLimitExceededError) — the same "centralised caption + hashtags
# rules used by Feed" instagram_feed_sender.py already reuses — so nothing
# here can ever change Feed or Story publish behavior, and it never writes
# to publish_queue.json or any history file.
DISALLOWED_ACTIONS = (
    "share",
    "publish",
    "submit",
    "press",
    "press_enter",
    "click_share",
    "click_post",
    "force_click",
    "click_via_js",
    "post_feed",
    "post_story",
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

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov"})

OPTIONAL_FEATURE_NAMES: tuple[str, ...] = (
    "location",
    "people_tagging",
    "collaborators",
    "topics",
    "music",
    "audio_controls",
    "paid_partnership",
    "boost",
    "accessibility_alt_text",
)


def _runtime_root() -> Path:
    """
    instagram_reel_preview.py location:

    10_apps/claude_runtime/src/publishing/instagram_reel_preview.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReelPreviewError(RuntimeError):
    """Base error for the Phase 10E.1 Instagram Reel preview controller."""


class ReelPreviewConfigError(ReelPreviewError):
    """Raised when config/social/instagram.yaml's reel_preview section is missing/invalid."""


class ReelPreviewIneligibleError(ReelPreviewError):
    """Raised when the job does not meet the Phase 10E.1 eligibility rules."""


class ReelPreviewNavigationError(ReelPreviewError):
    """Raised when Instagram cannot be opened at all."""


class ReelPreviewLoginError(ReelPreviewError):
    """Raised when the Instagram session is not logged_in."""


class CreateControlNotFoundError(ReelPreviewError):
    """Raised when the Create control cannot be located or clicked."""


class ReelOptionNotFoundError(ReelPreviewError):
    """Raised when the Reel option cannot be located or clicked."""


class VideoUploadError(ReelPreviewError):
    """Raised when the video file cannot be uploaded, or Instagram reports it unsupported."""


class VideoPreviewNotVerifiedError(ReelPreviewError):
    """Raised when no video preview element can be found after upload."""


class ReelProcessingTimeoutError(ReelPreviewError):
    """Raised when Instagram's processing indicator never clears within the configured timeout."""


class AmbiguousNextButtonError(ReelPreviewError):
    """Raised when more than one visible, enabled Next button candidate is found."""


class CaptionBoxNotFoundError(ReelPreviewError):
    """Raised when the caption textbox cannot be located."""


class CaptionVerificationMismatchError(ReelPreviewError):
    """Raised when the caption textbox's value does not exactly match final_caption."""


class ShareButtonNotFoundError(ReelPreviewError):
    """Raised when no Share button candidate can be located at all."""


class ShareButtonAmbiguousError(ReelPreviewError):
    """Raised when more than one visible, enabled Share button candidate is found."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReelPreviewSelectors:
    """
    Centralised CSS/Playwright selectors for the Reel preview controller.

    UNTESTED placeholder defaults — see the reel_preview: section comment
    block in config/social/instagram.yaml for how to tune these against a
    real session. There is deliberately no post_option/story_option
    selector anywhere here — Post/Story cannot be chosen even accidentally.
    """

    create_control: list[str] = field(default_factory=list)
    reel_option: list[str] = field(default_factory=list)
    dialog_container: list[str] = field(default_factory=list)
    file_input: list[str] = field(default_factory=list)
    video_preview: list[str] = field(default_factory=list)
    unsupported_media_error: list[str] = field(default_factory=list)
    processing_indicator: list[str] = field(default_factory=list)
    next_button: list[str] = field(default_factory=list)
    cover_preview: list[str] = field(default_factory=list)
    caption_textbox: list[str] = field(default_factory=list)
    share_button: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReelPreviewConfig:
    """Fully-resolved runtime configuration for the Reel preview controller."""

    selectors: ReelPreviewSelectors
    action_click_timeout_ms: int
    upload_settle_wait_ms: int
    navigation_settle_wait_ms: int
    create_menu_wait_ms: int
    dialog_open_wait_ms: int
    max_next_transitions: int
    next_click_wait_ms: int
    caption_settle_wait_ms: int
    processing_poll_interval_ms: int
    processing_timeout_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selectors": self.selectors.to_dict(),
            "action_click_timeout_ms": self.action_click_timeout_ms,
            "upload_settle_wait_ms": self.upload_settle_wait_ms,
            "navigation_settle_wait_ms": self.navigation_settle_wait_ms,
            "create_menu_wait_ms": self.create_menu_wait_ms,
            "dialog_open_wait_ms": self.dialog_open_wait_ms,
            "max_next_transitions": self.max_next_transitions,
            "next_click_wait_ms": self.next_click_wait_ms,
            "caption_settle_wait_ms": self.caption_settle_wait_ms,
            "processing_poll_interval_ms": self.processing_poll_interval_ms,
            "processing_timeout_ms": self.processing_timeout_ms,
        }


def load_reel_preview_config(
    *,
    config_path: str | Path | None = None,
) -> ReelPreviewConfig:
    """
    Load the reel_preview section from config/social/instagram.yaml.

    Independent loader, matching the pattern already used by every other
    module in this package — this function does not call
    instagram_feed_preview.py's or instagram_story_preview.py's loaders.
    """
    runtime_root = _runtime_root()

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (runtime_root / DEFAULT_INSTAGRAM_CONFIG_RELATIVE_PATH).resolve()
    )

    if not resolved_config_path.exists():
        raise ReelPreviewConfigError(
            f"Instagram session config not found: {resolved_config_path}"
        )

    try:
        with resolved_config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ReelPreviewConfigError(
            f"Invalid YAML in {resolved_config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict) or not raw:
        raise ReelPreviewConfigError(
            f"Instagram session config is empty or invalid: {resolved_config_path}"
        )

    section = raw.get("reel_preview") or {}
    selectors_section = section.get("selectors") or {}
    timeouts_section = section.get("timeouts") or {}

    selectors = ReelPreviewSelectors(
        create_control=list(selectors_section.get("create_control", [])),
        reel_option=list(selectors_section.get("reel_option", [])),
        dialog_container=list(selectors_section.get("dialog_container", [])),
        file_input=list(selectors_section.get("file_input", [])),
        video_preview=list(selectors_section.get("video_preview", [])),
        unsupported_media_error=list(
            selectors_section.get("unsupported_media_error", [])
        ),
        processing_indicator=list(selectors_section.get("processing_indicator", [])),
        next_button=list(selectors_section.get("next_button", [])),
        cover_preview=list(selectors_section.get("cover_preview", [])),
        caption_textbox=list(selectors_section.get("caption_textbox", [])),
        share_button=list(selectors_section.get("share_button", [])),
    )

    return ReelPreviewConfig(
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
        max_next_transitions=int(timeouts_section.get("max_next_transitions", 2)),
        next_click_wait_ms=int(timeouts_section.get("next_click_wait_ms", 1200)),
        caption_settle_wait_ms=int(
            timeouts_section.get("caption_settle_wait_ms", 1500)
        ),
        processing_poll_interval_ms=int(
            timeouts_section.get("processing_poll_interval_ms", 2000)
        ),
        processing_timeout_ms=int(
            timeouts_section.get("processing_timeout_ms", 60000)
        ),
    )


# ---------------------------------------------------------------------------
# Eligibility (pure, no browser)
# ---------------------------------------------------------------------------


def check_reel_job_eligible(job: PublishJob, config: PublisherConfig) -> None:
    """
    Raise ReelPreviewIneligibleError unless the job may be previewed.

    Runs entirely against local data (the job dict already loaded from the
    queue) — no browser is opened and the queue is never touched by this
    check.
    """
    if job.content_type != "instagram_reel":
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} is not an instagram_reel job "
            f"(content_type={job.content_type!r})"
        )

    if job.status != "approved":
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} is not eligible for preview "
            f"(status={job.status!r}; only 'approved' jobs are eligible)"
        )

    if len(job.media_paths) != 1:
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} must have exactly one media path, "
            f"found {len(job.media_paths)}"
        )

    media_path = Path(job.media_paths[0])

    if not media_path.is_file():
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} media file does not exist: {media_path}"
        )

    if media_path.stat().st_size == 0:
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} media file is zero bytes: {media_path}"
        )

    extension = media_path.suffix.lower()

    if extension not in VIDEO_EXTENSIONS:
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} media file has an unsupported video "
            f"extension: {extension or '(none)'}"
        )

    if config.allowed_media_extensions and extension not in config.allowed_media_extensions:
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} media file extension {extension!r} is not in "
            "the configured allowed_media_extensions"
        )

    if not job.caption_text or not job.caption_text.strip():
        raise ReelPreviewIneligibleError(f"Job {job.job_id} caption_text is empty")

    if job.published_at is not None:
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} already has published_at set "
            f"({job.published_at}); refusing to preview"
        )

    if job.platform_post_id is not None:
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} already has a platform_post_id "
            f"({job.platform_post_id}); refusing to preview"
        )


def expected_final_reel_path(
    job: PublishJob,
    config: PublisherConfig,
    *,
    output_root: str | Path | None = None,
) -> Path:
    relative = config.paths.get("reel_final_video_file", "videos/reel_final.mp4")
    root = Path(output_root) if output_root is not None else (app_root() / "output")
    return root / job.production_date / relative


def check_final_reel_media(
    job: PublishJob,
    config: PublisherConfig,
    *,
    output_root: str | Path | None = None,
) -> None:
    """
    Raise ReelPreviewIneligibleError unless job.media_paths[0] is exactly
    the configured final-video path for its production_date — never an
    individual scene clip or an unrelated video. This is intentionally
    re-checked independently of check_reel_job_eligible(), even though
    publisher_service.py only ever builds instagram_reel jobs from this
    exact path today, as defense against a hand-edited or future-changed
    queue record.

    output_root defaults to the real app_root()/output (what the CLI
    uses); tests pass an injected temp directory instead, the same
    testability pattern PublisherService.output_root already uses.
    """
    expected_path = expected_final_reel_path(job, config, output_root=output_root)
    actual_path = Path(job.media_paths[0]).resolve()

    if actual_path != expected_path.resolve():
        raise ReelPreviewIneligibleError(
            f"Job {job.job_id} media path does not reference the configured "
            f"final Reel video for {job.production_date}. "
            f"Found: {actual_path}. Expected: {expected_path}"
        )

    if not expected_path.is_file():
        raise ReelPreviewIneligibleError(
            f"No final Reel video exists for job {job.job_id} at the "
            f"expected path: {expected_path}"
        )

    if expected_path.stat().st_size == 0:
        raise ReelPreviewIneligibleError(
            f"Final Reel video for job {job.job_id} is zero bytes: {expected_path}"
        )


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReelPreviewResult:
    """
    Summary of one instagram_reel_preview --preview run.

    Only ever constructed on full success — every failure mode raises a
    typed ReelPreviewError subclass instead. share_clicked and published
    are always False: there is no assignment site in this module that
    could ever set either to True.
    """

    job_id: str
    production_date: str
    persona_id: str
    login_status: str
    media_path: str
    video_uploaded: bool
    video_preview_verified: bool
    processing_started: str | None
    processing_finished: str | None
    processing_duration_seconds: float | None
    cover_verified: bool
    caption_filled: bool
    caption_verified: bool
    final_caption_length: int
    next_transitions: list[str]
    share_button_found: bool
    share_button_enabled: bool
    share_button_interactive: bool
    share_clicked: bool
    published: bool
    optional_features: dict[str, str]
    screenshots: dict[str, str | None]
    warnings: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    visible: bool = False
    enabled: bool = False
    interactive: bool = False
    ambiguous: bool = False
    disabled_candidate: bool = False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class InstagramReelPreviewController:
    """
    Prepares one approved instagram_reel PublishJob completely, verifies
    the video/processing/cover/caption state, locates the Share button for
    verification only, then stops. Never clicks Share or Post, never
    presses Enter, never writes to publish_queue.json (it never even holds
    a writable QueueService), and never records a publish-history
    transition (history_service is never imported).

    Fully self-contained: does not subclass InstagramFeedPreviewController
    or InstagramStoryPreviewController.
    """

    def __init__(
        self,
        *,
        session: InstagramSession,
        reel_config: ReelPreviewConfig,
        publisher_config: PublisherConfig,
        screenshots_dir: str | Path | None = None,
        logs_dir: str | Path | None = None,
    ) -> None:
        self.session = session
        self.reel_config = reel_config
        self.publisher_config = publisher_config
        self.screenshots_dir = (
            Path(screenshots_dir) if screenshots_dir else publisher_config.screenshots_dir()
        )
        self.logs_dir = Path(logs_dir) if logs_dir else publisher_config.logs_dir()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _elapsed_seconds(started_at: str, finished_at: str) -> float:
        return (
            datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
        ).total_seconds()

    # -- generic selector helpers ----------------------------------------

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

    @staticmethod
    async def _composer_current_text(element: Any) -> str:
        try:
            value = await element.input_value()
            if value:
                return value
        except Exception:
            pass

        try:
            return await element.inner_text()
        except Exception:
            return ""

    async def _locate_unique_visible_enabled(
        self,
        root: Any,
        selectors: list[str],
    ) -> _ElementLookup:
        total_candidates = 0
        saw_single_unusable_candidate = False

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
                    visible=True,
                    enabled=True,
                    interactive=True,
                )

            if len(valid) > 1:
                return _ElementLookup(
                    element=None,
                    selector_string=selector,
                    candidate_count=len(candidates),
                    visible=True,
                    enabled=True,
                    interactive=False,
                    ambiguous=True,
                )

            if len(candidates) == 1:
                saw_single_unusable_candidate = True

        return _ElementLookup(
            element=None,
            selector_string=None,
            candidate_count=total_candidates,
            disabled_candidate=saw_single_unusable_candidate,
        )

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

    # -- diagnostics -------------------------------------------------------

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
        diagnostics: dict[str, Any],
        result: str,
        error: str | None,
    ) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        finished_at = self._now()
        run_id = finished_at.replace(":", "").replace("-", "").replace("+", "").replace(".", "")
        log_path = self.logs_dir / f"instagram_reel_preview_{job.job_id}_{run_id}.json"

        media_path = job.media_paths[0] if job.media_paths else None
        media_size = None
        if media_path and Path(media_path).is_file():
            media_size = Path(media_path).stat().st_size

        payload = {
            "started_at": started_at,
            "finished_at": finished_at,
            "command": command,
            "job_id": job.job_id,
            "production_date": job.production_date,
            "persona_id": job.persona_id,
            "media_path": media_path,
            "media_size": media_size,
            "login_status": diagnostics.get("login_status"),
            "create_control_found": diagnostics.get("create_control_found", False),
            "reel_option_selected": diagnostics.get("reel_option_selected", False),
            "video_uploaded": diagnostics.get("video_uploaded", False),
            "video_preview_verified": diagnostics.get("video_preview_verified", False),
            "processing_started": diagnostics.get("processing_started"),
            "processing_finished": diagnostics.get("processing_finished"),
            "processing_duration_seconds": diagnostics.get("processing_duration_seconds"),
            "next_transitions": diagnostics.get("next_transitions", []),
            "cover_verified": diagnostics.get("cover_verified", False),
            "caption_verified": diagnostics.get("caption_verified", False),
            "share_button_found": diagnostics.get("share_button_found", False),
            "share_button_enabled": diagnostics.get("share_button_enabled", False),
            "share_clicked": False,
            "published": False,
            "optional_features": diagnostics.get("optional_features", {}),
            "screenshots": diagnostics.get("screenshots", {}),
            "warnings": diagnostics.get("warnings", []),
            "result": result,
            "error": error,
            # Never included: cookies, passwords, session tokens, browser
            # profile contents, or any DM/comment data.
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
            "reel_option_selected": False,
            "video_uploaded": False,
            "video_preview_verified": False,
            "processing_started": None,
            "processing_finished": None,
            "processing_duration_seconds": None,
            "next_transitions": [],
            "cover_verified": False,
            "caption_box_found": False,
            "caption_filled": False,
            "caption_verified": False,
            "share_button_found": False,
            "share_button_enabled": False,
            "optional_features": _default_optional_features(),
            "warnings": [],
            "screenshots": {
                "before_upload": None,
                "after_upload": None,
                "processing": None,
                "ready": None,
                "error": None,
            },
        }

    # -- orchestration ------------------------------------------------------

    async def preview(
        self,
        job: PublishJob,
        caption_result: FinalCaptionResult,
    ) -> ReelPreviewResult:
        """
        Prepare the Reel completely, verify the video/processing/cover/
        caption state, locate (never click) the Share button, then stop.
        """
        started_at = self._now()
        diagnostics = self._default_diagnostics()

        playwright, context = await self.session._open_context()

        try:
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                result = await self._run_preview(page, job, caption_result, diagnostics)
            except ReelPreviewError as exc:
                diagnostics["screenshots"]["error"] = await self._save_screenshot(
                    page, filename=f"reel_preview_error_{job.job_id}.png"
                )
                self._write_log(
                    started_at=started_at,
                    command="--preview",
                    job=job,
                    diagnostics=diagnostics,
                    result="failed",
                    error=str(exc),
                )
                raise

            self._write_log(
                started_at=started_at,
                command="--preview",
                job=job,
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
        caption_result: FinalCaptionResult,
        diagnostics: dict[str, Any],
    ) -> ReelPreviewResult:
        # -- navigation + login --------------------------------------------
        try:
            await page.goto(
                self.session.config.base_url,
                timeout=self.session.config.navigation_timeout_ms,
                wait_until="domcontentloaded",
            )
        except PlaywrightTimeoutError as exc:
            raise ReelPreviewNavigationError(
                f"Timed out opening Instagram: {exc}"
            ) from exc
        except Exception as exc:
            raise ReelPreviewNavigationError(
                f"Unable to open Instagram: {exc}"
            ) from exc

        await page.wait_for_timeout(self.reel_config.navigation_settle_wait_ms)

        login_status, _login_reason = await self._detect_login(page)
        diagnostics["login_status"] = login_status

        if login_status != "logged_in":
            raise ReelPreviewLoginError(
                f"Instagram session is not logged_in (status={login_status})"
            )

        # -- open the Reel composer -----------------------------------------
        create_lookup = await self._locate_unique_visible_enabled(
            page, self.reel_config.selectors.create_control
        )

        if create_lookup.element is None:
            raise CreateControlNotFoundError(
                "Could not locate a unique, visible, enabled Create control "
                f"(candidates={create_lookup.candidate_count}, "
                f"ambiguous={create_lookup.ambiguous})."
            )

        await create_lookup.element.click(timeout=self.reel_config.action_click_timeout_ms)
        diagnostics["create_control_found"] = True
        await page.wait_for_timeout(self.reel_config.create_menu_wait_ms)

        # Only ever searches for the Reel option — no post_option/
        # story_option selector exists anywhere in this module or its
        # config, so Post/Story cannot be chosen even accidentally.
        reel_lookup = await self._locate_unique_visible_enabled(
            page, self.reel_config.selectors.reel_option
        )

        if reel_lookup.element is None:
            raise ReelOptionNotFoundError(
                "Could not locate a unique, visible, enabled Reel option "
                f"(candidates={reel_lookup.candidate_count}, "
                f"ambiguous={reel_lookup.ambiguous})."
            )

        await reel_lookup.element.click(timeout=self.reel_config.action_click_timeout_ms)
        diagnostics["reel_option_selected"] = True
        await page.wait_for_timeout(self.reel_config.dialog_open_wait_ms)

        dialog_root = (
            await self._find_first(page, self.reel_config.selectors.dialog_container)
            or page
        )

        diagnostics["screenshots"]["before_upload"] = await self._save_screenshot(
            page, filename=f"reel_preview_before_upload_{job.job_id}.png"
        )

        # -- upload video -----------------------------------------------------
        media_path = job.media_paths[0]
        file_input = await self._find_first(dialog_root, self.reel_config.selectors.file_input)

        if file_input is None:
            raise VideoUploadError("Could not locate the Reel video file input.")

        try:
            await file_input.set_input_files(media_path)
        except Exception as exc:
            raise VideoUploadError(
                f"Failed to upload video file {media_path}: {exc}"
            ) from exc

        await page.wait_for_timeout(self.reel_config.upload_settle_wait_ms)

        unsupported = await self._find_first(
            dialog_root, self.reel_config.selectors.unsupported_media_error
        )

        if unsupported is not None:
            unsupported_text = await self._composer_current_text(unsupported)
            raise VideoUploadError(
                f"Instagram reported unsupported media: {unsupported_text!r}"
            )

        diagnostics["video_uploaded"] = True

        diagnostics["screenshots"]["after_upload"] = await self._save_screenshot(
            page, filename=f"reel_preview_after_upload_{job.job_id}.png"
        )

        video_preview = await self._find_first(dialog_root, self.reel_config.selectors.video_preview)
        video_preview_verified = video_preview is not None
        diagnostics["video_preview_verified"] = video_preview_verified

        if not video_preview_verified:
            raise VideoPreviewNotVerifiedError(
                "Reel video preview could not be verified after upload."
            )

        # -- processing wait --------------------------------------------------
        processing_started = self._now()
        diagnostics["processing_started"] = processing_started

        max_polls = max(
            1,
            math.ceil(
                self.reel_config.processing_timeout_ms
                / max(self.reel_config.processing_poll_interval_ms, 1)
            ),
        )

        processing_cleared = False
        for _poll_index in range(max_polls):
            indicator = await self._find_first(
                dialog_root, self.reel_config.selectors.processing_indicator
            )
            if indicator is None:
                processing_cleared = True
                break
            await page.wait_for_timeout(self.reel_config.processing_poll_interval_ms)

        processing_finished = self._now()
        diagnostics["processing_finished"] = processing_finished
        diagnostics["processing_duration_seconds"] = self._elapsed_seconds(
            processing_started, processing_finished
        )

        diagnostics["screenshots"]["processing"] = await self._save_screenshot(
            page, filename=f"reel_preview_processing_{job.job_id}.png"
        )

        if not processing_cleared:
            raise ReelProcessingTimeoutError(
                "Instagram's processing indicator did not clear within "
                f"{self.reel_config.processing_timeout_ms}ms."
            )

        # -- Next transitions ---------------------------------------------
        next_transitions: list[str] = []

        for step_index in range(self.reel_config.max_next_transitions):
            lookup = await self._locate_unique_visible_enabled(
                dialog_root, self.reel_config.selectors.next_button
            )

            if lookup.ambiguous:
                raise AmbiguousNextButtonError(
                    f"Found {lookup.candidate_count} ambiguous visible, "
                    f"enabled Next button candidates at transition "
                    f"{step_index + 1}; refusing to guess."
                )

            if lookup.element is None:
                # No more Next buttons: assume the details screen has
                # been reached. Not an error.
                break

            await lookup.element.click(timeout=self.reel_config.action_click_timeout_ms)
            next_transitions.append(f"next_transition_{step_index + 1}")
            await page.wait_for_timeout(self.reel_config.next_click_wait_ms)

        diagnostics["next_transitions"] = next_transitions

        # -- cover verification: soft signal only --------------------------
        cover_element = await self._find_first(dialog_root, self.reel_config.selectors.cover_preview)
        cover_verified = cover_element is not None
        diagnostics["cover_verified"] = cover_verified

        if not cover_verified:
            diagnostics["warnings"].append("cover_not_verified")

        # -- caption entry --------------------------------------------------
        caption_box = await self._find_first(dialog_root, self.reel_config.selectors.caption_textbox)

        if caption_box is None:
            raise CaptionBoxNotFoundError("Could not locate the caption textbox.")

        diagnostics["caption_box_found"] = True

        try:
            await caption_box.fill(caption_result.final_caption)
        except Exception:
            await caption_box.click(timeout=2000)
            await caption_box.type(caption_result.final_caption)

        diagnostics["caption_filled"] = True

        await page.wait_for_timeout(self.reel_config.caption_settle_wait_ms)

        final_caption_text = await self._composer_current_text(caption_box)
        caption_verified = final_caption_text == caption_result.final_caption
        diagnostics["caption_verified"] = caption_verified

        if not caption_verified:
            raise CaptionVerificationMismatchError(
                "Caption textbox value does not exactly match the "
                "intended final caption after filling."
            )

        # -- optional features: inspect only, never applied -----------------
        optional_features = _default_optional_features()
        diagnostics["optional_features"] = optional_features

        # -- Share button safety gate: locate + verify ONLY ------------------
        share_lookup = await self._locate_unique_visible_enabled(
            dialog_root, self.reel_config.selectors.share_button
        )

        diagnostics["share_button_found"] = share_lookup.element is not None
        diagnostics["share_button_enabled"] = share_lookup.enabled

        diagnostics["screenshots"]["ready"] = await self._save_screenshot(
            page, filename=f"reel_preview_ready_{job.job_id}.png"
        )

        if share_lookup.ambiguous:
            raise ShareButtonAmbiguousError(
                f"Found {share_lookup.candidate_count} ambiguous "
                "visible, enabled Share button candidates; refusing "
                "to guess. Nothing was clicked."
            )

        if share_lookup.element is None:
            raise ShareButtonNotFoundError(
                "Could not find a unique, visible, enabled Share "
                "button. Nothing was clicked."
            )

        return ReelPreviewResult(
            job_id=job.job_id,
            production_date=job.production_date,
            persona_id=job.persona_id,
            login_status=login_status,
            media_path=media_path,
            video_uploaded=diagnostics["video_uploaded"],
            video_preview_verified=video_preview_verified,
            processing_started=processing_started,
            processing_finished=processing_finished,
            processing_duration_seconds=diagnostics["processing_duration_seconds"],
            cover_verified=cover_verified,
            caption_filled=diagnostics["caption_filled"],
            caption_verified=caption_verified,
            final_caption_length=caption_result.final_caption_length,
            next_transitions=next_transitions,
            share_button_found=True,
            share_button_enabled=share_lookup.enabled,
            share_button_interactive=share_lookup.interactive,
            share_clicked=False,
            published=False,
            optional_features=optional_features,
            screenshots=dict(diagnostics["screenshots"]),
            warnings=list(diagnostics["warnings"]),
            error=None,
        )


def build_reel_preview_controller(
    *,
    publisher_config_path: str | Path | None = None,
    instagram_config_path: str | Path | None = None,
    headless_override: bool | None = None,
) -> tuple[InstagramReelPreviewController, PublisherConfig]:
    publisher_config = load_publisher_config(publisher_config_path)
    session_config = load_session_config(
        config_path=instagram_config_path,
        headless_override=headless_override,
    )
    reel_config = load_reel_preview_config(config_path=instagram_config_path)
    session = InstagramSession(session_config)

    controller = InstagramReelPreviewController(
        session=session,
        reel_config=reel_config,
        publisher_config=publisher_config,
    )
    return controller, publisher_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Instagram Reels Publisher — Preview Only (Phase 10E.1). "
            "Prepares an approved instagram_reel job through to a "
            "ready-to-share state and stops. Never clicks Share."
        )
    )

    parser.add_argument(
        "--preview",
        metavar="JOB_ID",
        required=True,
        help="Preview this approved instagram_reel job_id.",
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


def _print_pre_browser_block(job: PublishJob, caption_result: FinalCaptionResult) -> None:
    media_path = Path(job.media_paths[0])
    print()
    print("AIKO Instagram Reels Publisher — Preview Only (Phase 10E.1)")
    print("--------------------------------------------------------------")
    print(f"job id:            {job.job_id}")
    print(f"production date:   {job.production_date}")
    print(f"persona id:        {job.persona_id}")
    print(f"media path:        {media_path}")
    print(f"media filename:    {media_path.name}")
    print(f"file size:         {media_path.stat().st_size} bytes")
    print(f"video extension:   {media_path.suffix.lower()}")
    print(f"caption length:    {caption_result.final_caption_length}")
    print(f"current status:    {job.status}")
    print(f"approved_at:       {job.approved_at}")
    print()


def _print_result(result: ReelPreviewResult) -> None:
    print()
    print(f"job id:                    {result.job_id}")
    print(f"login status:              {result.login_status}")
    print(f"media path:                {result.media_path}")
    print(f"video uploaded:            {result.video_uploaded}")
    print(f"video preview verified:    {result.video_preview_verified}")
    print(f"processing started:        {result.processing_started}")
    print(f"processing finished:       {result.processing_finished}")
    print(f"processing duration (s):   {result.processing_duration_seconds}")
    print(f"next transitions:          {result.next_transitions}")
    print(f"cover verified:            {result.cover_verified}")
    print(f"caption filled:            {result.caption_filled}")
    print(f"caption verified:          {result.caption_verified}")
    print(f"final caption length:      {result.final_caption_length}")
    print(f"share button found:        {result.share_button_found}")
    print(f"share button enabled:      {result.share_button_enabled}")
    print(f"share clicked:             {result.share_clicked}")
    print(f"published:                 {result.published}")
    print(f"optional features:         {result.optional_features}")
    print(f"warnings:                  {result.warnings}")
    print(f"screenshots:               {result.screenshots}")
    print()


async def _run(arguments: argparse.Namespace) -> int:
    publisher_config = load_publisher_config(arguments.config)

    job = load_job_by_id(publisher_config.queue_path(), arguments.preview)

    if job is None:
        print(f"[InstagramReelPreview] No job found for job_id: {arguments.preview}")
        return 1

    try:
        check_reel_job_eligible(job, publisher_config)
        check_final_reel_media(job, publisher_config)
    except ReelPreviewIneligibleError as exc:
        print(f"[InstagramReelPreview] {exc}")
        return 1

    limit = (
        publisher_config.platform_limits.get("instagram_reel", {})
        .get("caption_max_characters")
    )

    try:
        caption_result = build_final_caption(
            job.caption_text or "",
            list(job.hashtags),
            max_characters=limit,
        )
    except CaptionLimitExceededError as exc:
        print(f"[InstagramReelPreview] {exc}")
        return 1

    _print_pre_browser_block(job, caption_result)

    controller, _ = build_reel_preview_controller(
        publisher_config_path=arguments.config,
        instagram_config_path=arguments.instagram_config,
        headless_override=True if arguments.headless else None,
    )

    try:
        result = await controller.preview(job, caption_result)
    except InstagramSessionError as exc:
        print(f"[InstagramReelPreview] preview failed: {exc}")
        return 1
    except ReelPreviewError as exc:
        print(f"[InstagramReelPreview] preview failed: {exc}")
        return 1

    _print_result(result)
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
