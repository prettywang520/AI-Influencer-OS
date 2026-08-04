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
from .queue_service import load_job_by_id

# Phase 10B is a PREVIEW ONLY module. It opens the real, logged-in
# Instagram session, prepares a Feed post completely — Create, Post
# (never Story/Reel), media upload, crop/filter/details transitions,
# caption entry and verification — locates the Share button for
# verification only, and STOPS. It must never grow, and never
# accidentally gain, any of the capabilities below.
DISALLOWED_ACTIONS = (
    "share",
    "publish",
    "submit",
    "press",
    "press_enter",
    "click_share",
    "post_story",
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


def _runtime_root() -> Path:
    """
    instagram_feed_preview.py location:

    10_apps/claude_runtime/src/publishing/instagram_feed_preview.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FeedPreviewError(RuntimeError):
    """Base error for the Phase 10B Instagram Feed preview controller."""


class FeedPreviewConfigError(FeedPreviewError):
    """Raised when config/social/instagram.yaml's feed_preview section is missing/invalid."""


class JobNotFoundError(FeedPreviewError):
    """Raised when the given job_id does not exist in the publish queue."""


class FeedPreviewIneligibleError(FeedPreviewError):
    """Raised when the job does not meet the Phase 10B eligibility rules."""


class CaptionLimitExceededError(FeedPreviewError):
    """Raised when the assembled final caption exceeds the configured character limit."""


class FeedPreviewNavigationError(FeedPreviewError):
    """Raised when Instagram cannot be opened at all."""


class FeedPreviewLoginError(FeedPreviewError):
    """Raised when the Instagram session is not logged_in."""


class CreateControlNotFoundError(FeedPreviewError):
    """Raised when the Create control cannot be located or clicked."""


class MediaUploadError(FeedPreviewError):
    """Raised when the media file cannot be uploaded, or Instagram reports it unsupported."""


class MediaPreviewNotVerifiedError(FeedPreviewError):
    """Raised when no media preview element can be found after upload."""


class AmbiguousNextButtonError(FeedPreviewError):
    """Raised when more than one visible, enabled Next button candidate is found."""


class CaptionBoxNotFoundError(FeedPreviewError):
    """Raised when the caption textbox cannot be located."""


class CaptionVerificationMismatchError(FeedPreviewError):
    """Raised when the caption textbox's value does not exactly match final_caption."""


class ShareButtonNotFoundError(FeedPreviewError):
    """Raised when no Share button candidate can be located at all."""


class ShareButtonDisabledError(FeedPreviewError):
    """Raised when a single Share button candidate was found but is not usable."""


class ShareButtonAmbiguousError(FeedPreviewError):
    """Raised when more than one visible, enabled Share button candidate is found."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeedPreviewSelectors:
    """
    Centralised CSS/Playwright selectors for the Feed preview controller.

    UNTESTED placeholder defaults — see the feed_preview: section
    comment block in config/social/instagram.yaml for how to tune
    these against a real session. Only Post-related selectors are
    ever defined here (create_button, post_option) — there is no
    story_option/reel_option selector anywhere, so Story/Reel cannot
    be chosen even accidentally.
    """

    create_button: list[str] = field(default_factory=list)
    post_option: list[str] = field(default_factory=list)
    dialog_container: list[str] = field(default_factory=list)
    file_input: list[str] = field(default_factory=list)
    media_preview: list[str] = field(default_factory=list)
    unsupported_media_error: list[str] = field(default_factory=list)
    next_button: list[str] = field(default_factory=list)
    caption_textbox: list[str] = field(default_factory=list)
    share_button: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeedPreviewConfig:
    """Fully-resolved runtime configuration for the Feed preview controller."""

    selectors: FeedPreviewSelectors
    max_next_transitions: int
    action_click_timeout_ms: int
    upload_settle_wait_ms: int
    next_click_wait_ms: int
    caption_settle_wait_ms: int
    navigation_settle_wait_ms: int
    create_menu_wait_ms: int
    dialog_open_wait_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selectors": self.selectors.to_dict(),
            "max_next_transitions": self.max_next_transitions,
            "action_click_timeout_ms": self.action_click_timeout_ms,
            "upload_settle_wait_ms": self.upload_settle_wait_ms,
            "next_click_wait_ms": self.next_click_wait_ms,
            "caption_settle_wait_ms": self.caption_settle_wait_ms,
            "navigation_settle_wait_ms": self.navigation_settle_wait_ms,
            "create_menu_wait_ms": self.create_menu_wait_ms,
            "dialog_open_wait_ms": self.dialog_open_wait_ms,
        }


def load_feed_preview_config(
    *,
    config_path: str | Path | None = None,
) -> FeedPreviewConfig:
    """
    Load the feed_preview section from config/social/instagram.yaml.

    Independent loader, matching the pattern already used by
    instagram_comment_reader.py / instagram_reply_controller.py /
    instagram_reply_sender.py / instagram_dm_reader.py.
    """
    runtime_root = _runtime_root()

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (runtime_root / DEFAULT_INSTAGRAM_CONFIG_RELATIVE_PATH).resolve()
    )

    if not resolved_config_path.exists():
        raise FeedPreviewConfigError(
            f"Instagram session config not found: {resolved_config_path}"
        )

    try:
        with resolved_config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise FeedPreviewConfigError(
            f"Invalid YAML in {resolved_config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict) or not raw:
        raise FeedPreviewConfigError(
            f"Instagram session config is empty or invalid: {resolved_config_path}"
        )

    section = raw.get("feed_preview") or {}
    selectors_section = section.get("selectors") or {}
    timeouts_section = section.get("timeouts") or {}

    selectors = FeedPreviewSelectors(
        create_button=list(selectors_section.get("create_button", [])),
        post_option=list(selectors_section.get("post_option", [])),
        dialog_container=list(selectors_section.get("dialog_container", [])),
        file_input=list(selectors_section.get("file_input", [])),
        media_preview=list(selectors_section.get("media_preview", [])),
        unsupported_media_error=list(
            selectors_section.get("unsupported_media_error", [])
        ),
        next_button=list(selectors_section.get("next_button", [])),
        caption_textbox=list(selectors_section.get("caption_textbox", [])),
        share_button=list(selectors_section.get("share_button", [])),
    )

    return FeedPreviewConfig(
        selectors=selectors,
        max_next_transitions=int(timeouts_section.get("max_next_transitions", 2)),
        action_click_timeout_ms=int(
            timeouts_section.get("action_click_timeout_ms", 5000)
        ),
        upload_settle_wait_ms=int(
            timeouts_section.get("upload_settle_wait_ms", 3000)
        ),
        next_click_wait_ms=int(timeouts_section.get("next_click_wait_ms", 1200)),
        caption_settle_wait_ms=int(
            timeouts_section.get("caption_settle_wait_ms", 1500)
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


def check_job_eligible(job: PublishJob) -> None:
    """
    Raise FeedPreviewIneligibleError unless the job may be previewed.

    Runs entirely against local data (the job dict already loaded from
    the queue) — no browser is opened and the queue is never touched
    by this check.
    """
    if job.content_type != "instagram_feed":
        raise FeedPreviewIneligibleError(
            f"Job {job.job_id} is not an instagram_feed job "
            f"(content_type={job.content_type!r})"
        )

    if job.status != "approved":
        raise FeedPreviewIneligibleError(
            f"Job {job.job_id} is not eligible for preview "
            f"(status={job.status!r}; only 'approved' jobs are eligible)"
        )

    if len(job.media_paths) != 1:
        raise FeedPreviewIneligibleError(
            f"Job {job.job_id} must have exactly one media path, "
            f"found {len(job.media_paths)}"
        )

    media_path = Path(job.media_paths[0])

    if not media_path.is_file():
        raise FeedPreviewIneligibleError(
            f"Job {job.job_id} media file does not exist: {media_path}"
        )

    if media_path.stat().st_size == 0:
        raise FeedPreviewIneligibleError(
            f"Job {job.job_id} media file is zero bytes: {media_path}"
        )

    if not job.caption_text or not job.caption_text.strip():
        raise FeedPreviewIneligibleError(
            f"Job {job.job_id} caption_text is empty"
        )


# ---------------------------------------------------------------------------
# Caption building (pure, no browser)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FinalCaptionResult:
    caption_text: str
    hashtags_text: str
    final_caption: str
    final_caption_length: int
    hashtags_appended: bool
    duplicate_hashtags_removed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_final_caption(
    caption_text: str,
    hashtags: list[str],
    *,
    max_characters: int | None,
) -> FinalCaptionResult:
    """
    Assemble the final Feed caption from caption_text + hashtags.

    - trailing whitespace is stripped; internal line breaks in
      caption_text are preserved verbatim.
    - hashtags already present as a literal substring of the caption
      (case-insensitive) are dropped rather than duplicated.
    - remaining hashtags are joined with a single space and appended
      after exactly one blank line.
    - if the assembled caption exceeds max_characters, raises
      CaptionLimitExceededError rather than truncating.
    """
    caption_stripped = caption_text.rstrip()
    caption_lower = caption_stripped.lower()

    seen: set[str] = set()
    filtered_hashtags: list[str] = []
    duplicate_hashtags_removed: list[str] = []

    for raw_tag in hashtags:
        tag = raw_tag.strip()

        if not tag:
            continue

        key = tag.lower()

        if key in seen:
            continue

        seen.add(key)

        if key in caption_lower:
            duplicate_hashtags_removed.append(tag)
            continue

        filtered_hashtags.append(tag)

    hashtags_text = " ".join(filtered_hashtags)
    hashtags_appended = bool(hashtags_text)

    final_caption = (
        f"{caption_stripped}\n\n{hashtags_text}" if hashtags_text else caption_stripped
    )
    final_caption_length = len(final_caption)

    if max_characters is not None and final_caption_length > max_characters:
        raise CaptionLimitExceededError(
            f"Final caption is {final_caption_length} characters, which "
            f"exceeds the configured limit of {max_characters}. Refusing "
            "to truncate — fix the source caption/hashtags instead."
        )

    return FinalCaptionResult(
        caption_text=caption_text,
        hashtags_text=hashtags_text,
        final_caption=final_caption,
        final_caption_length=final_caption_length,
        hashtags_appended=hashtags_appended,
        duplicate_hashtags_removed=duplicate_hashtags_removed,
    )


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeedPreviewResult:
    """
    Summary of one instagram_feed_preview --preview run.

    Only ever constructed on full success — every failure mode raises a
    typed FeedPreviewError subclass instead (mirrors ReplySendResult /
    ReplyPreflightResult's pattern in instagram_reply_sender.py).

    share_clicked and published are always False: there is no
    assignment site in this module that could ever set either to True.
    """

    job_id: str
    production_date: str
    persona_id: str
    login_status: str
    media_path: str
    caption_text: str
    hashtags_text: str
    final_caption: str
    final_caption_length: int
    hashtags_appended: bool
    duplicate_hashtags_removed: list[str]
    create_control_found: bool
    post_option_selected: bool
    media_uploaded: bool
    media_filename: str | None
    media_preview_verified: bool
    next_transitions: list[str]
    caption_box_found: bool
    caption_filled: bool
    caption_verified: bool
    existing_caption_text: str | None
    final_caption_text: str | None
    location_applied: str
    alt_text_applied: str
    share_button_found: bool
    share_button_enabled: bool
    share_button_interactive: bool
    share_clicked: bool
    published: bool
    screenshots: dict[str, str | None]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _ElementLookup:
    """
    Internal result of _locate_unique_visible_enabled(). A candidate is
    only returned when it is the SOLE visible, enabled match for some
    selector — zero or multiple candidates both fail safely (element is
    None) rather than guessing. Direct generalisation of
    instagram_reply_sender.InstagramReplySender._locate_post_button,
    minus that method's composer-container-scoping (callers here
    already pass the exact root to search).
    """

    element: Any | None
    selector_string: str | None
    candidate_count: int
    visible: bool
    enabled: bool
    interactive: bool
    ambiguous: bool = False
    disabled_candidate: bool = False


@dataclass(slots=True)
class _ShareReadyState:
    """
    Internal return value of _reach_share_ready(): everything a caller
    needs either to build a result (preview()) or to proceed straight
    to the single permitted Share click (instagram_feed_sender.py).
    share_lookup.element is guaranteed non-None, unique, visible, and
    enabled whenever this is returned — _reach_share_ready() raises
    instead of returning otherwise.
    """

    login_status: str
    media_path: str
    dialog_root: Any
    share_lookup: "_ElementLookup"
    caption_verified: bool
    existing_caption_text: str
    final_caption_text: str
    next_transitions: list[str]
    media_preview_verified: bool
    diagnostics: dict[str, Any]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class InstagramFeedPreviewController:
    """
    Prepares one approved instagram_feed PublishJob for Share, then
    stops. Never clicks Share, never presses Enter, never writes to
    publish_queue.json (it never even holds a writable QueueService),
    and never records a publish-history transition (the job's status
    is never touched).
    """

    def __init__(
        self,
        *,
        session: InstagramSession,
        feed_config: FeedPreviewConfig,
        publisher_config: PublisherConfig,
        screenshots_dir: str | Path | None = None,
        logs_dir: str | Path | None = None,
    ) -> None:
        self.session = session
        self.feed_config = feed_config
        self.publisher_config = publisher_config
        self.screenshots_dir = (
            Path(screenshots_dir) if screenshots_dir else publisher_config.screenshots_dir()
        )
        self.logs_dir = Path(logs_dir) if logs_dir else publisher_config.logs_dir()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- generic selector helpers (mirrors instagram_reply_controller.py) ----

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

    async def _detect_login(self, page: Any) -> tuple[str, str]:
        """
        Reuse Phase 8A's classification logic against this page,
        without modifying InstagramSession — same pattern as
        instagram_reply_controller.py's own _detect_login.
        """
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
            visible=False,
            enabled=False,
            interactive=False,
            disabled_candidate=saw_single_unusable_candidate,
        )

    # -- diagnostics ----------------------------------------------------------

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
        job: PublishJob,
        diagnostics: dict[str, Any],
        result: str,
        error: str | None,
    ) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        finished_at = self._now()
        run_id = finished_at.replace(":", "").replace("-", "").replace("+", "").replace(".", "")
        log_path = self.logs_dir / f"instagram_feed_preview_{job.job_id}_{run_id}.json"

        payload = {
            "started_at": started_at,
            "finished_at": finished_at,
            "job_id": job.job_id,
            "production_date": job.production_date,
            "persona_id": job.persona_id,
            "media_path": job.media_paths[0] if job.media_paths else None,
            **diagnostics,
            "result": result,
            "error": error,
        }

        temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary_path.replace(log_path)
        return log_path

    @staticmethod
    def _default_diagnostics() -> dict[str, Any]:
        return {
            "caption_length": None,
            "hashtags_appended": False,
            "login_status": "unknown",
            "create_control_found": False,
            "post_option_selected": False,
            "media_uploaded": False,
            "media_preview_verified": False,
            "next_transitions": [],
            "caption_box_found": False,
            "caption_verified": False,
            "share_button_found": False,
            "share_button_enabled": False,
            "share_clicked": False,
            "published": False,
            "screenshots": {"before_upload": None, "after_upload": None, "ready": None},
        }

    # -- shared: reach a verified, ready-to-share state --------------------

    async def _reach_share_ready(
        self,
        page: Any,
        job: PublishJob,
        caption_result: FinalCaptionResult,
        diagnostics: dict[str, Any],
        *,
        login_failure_screenshot_name: str,
        before_upload_screenshot_name: str,
        after_upload_screenshot_name: str,
        final_screenshot_name: str,
    ) -> _ShareReadyState:
        """
        Navigate to Instagram, verify login, open Create -> Post, upload
        the exact approved media, advance through the crop/filter Next
        transitions, fill and verify the caption, then locate and
        verify (never click) the Share button.

        Shared by InstagramFeedPreviewController.preview() (Phase 10B)
        and InstagramFeedSenderController.preflight()/publish() (Phase
        10C) so the caption-joining/upload/transition/Share-lookup
        logic exists in exactly one place. This method only ever
        *locates and verifies* Share — it has no code path that clicks
        it; only instagram_feed_sender.py's publish() does that, using
        the element this method returns.

        Mutates diagnostics in place and returns a _ShareReadyState.
        Raises a FeedPreviewError subclass on any failure; the caller
        is responsible for its own log-writing and cleanup.
        """
        try:
            await page.goto(
                self.session.config.base_url,
                timeout=self.session.config.navigation_timeout_ms,
                wait_until="domcontentloaded",
            )
        except PlaywrightTimeoutError as exc:
            raise FeedPreviewNavigationError(
                f"Timed out opening Instagram: {exc}"
            ) from exc
        except Exception as exc:
            raise FeedPreviewNavigationError(
                f"Unable to open Instagram: {exc}"
            ) from exc

        await page.wait_for_timeout(self.feed_config.navigation_settle_wait_ms)

        login_status, login_reason = await self._detect_login(page)
        diagnostics["login_status"] = login_status

        if login_status != "logged_in":
            screenshot_path = await self._save_screenshot(
                page, filename=login_failure_screenshot_name
            )
            diagnostics["screenshots"]["ready"] = screenshot_path
            raise FeedPreviewLoginError(
                "Instagram session is not logged_in "
                f"(status={login_status}). Screenshot: {screenshot_path}"
            )

        create_control = await self._find_first(
            page, self.feed_config.selectors.create_button
        )

        if create_control is None:
            raise CreateControlNotFoundError(
                "Could not locate the Create control."
            )

        await create_control.click(timeout=self.feed_config.action_click_timeout_ms)
        diagnostics["create_control_found"] = True
        await page.wait_for_timeout(self.feed_config.create_menu_wait_ms)

        # Only ever searches for the Post option — no story/reel
        # selector exists anywhere in this module or its config.
        post_option = await self._find_first(
            page, self.feed_config.selectors.post_option
        )

        if post_option is not None:
            await post_option.click(timeout=self.feed_config.action_click_timeout_ms)
            diagnostics["post_option_selected"] = True
            # The "Create new post" dialog takes noticeably longer to
            # fully mount (including its file input) than the Create
            # flyout menu did — confirmed live 2026-08-03.
            await page.wait_for_timeout(self.feed_config.dialog_open_wait_ms)

        dialog_root = (
            await self._find_first(page, self.feed_config.selectors.dialog_container)
            or page
        )

        diagnostics["screenshots"]["before_upload"] = await self._save_screenshot(
            page, filename=before_upload_screenshot_name
        )

        media_path = job.media_paths[0]

        file_input = await self._find_first(
            dialog_root, self.feed_config.selectors.file_input
        )

        if file_input is None:
            raise MediaUploadError("Could not locate the media file input.")

        try:
            await file_input.set_input_files(media_path)
        except Exception as exc:
            raise MediaUploadError(
                f"Failed to upload media file {media_path}: {exc}"
            ) from exc

        await page.wait_for_timeout(self.feed_config.upload_settle_wait_ms)

        unsupported = await self._find_first(
            dialog_root, self.feed_config.selectors.unsupported_media_error
        )

        if unsupported is not None:
            unsupported_text = await self._composer_current_text(unsupported)
            raise MediaUploadError(
                f"Instagram reported unsupported media: {unsupported_text!r}"
            )

        media_preview = await self._find_first(
            dialog_root, self.feed_config.selectors.media_preview
        )
        media_preview_verified = media_preview is not None

        diagnostics["media_uploaded"] = True
        diagnostics["media_preview_verified"] = media_preview_verified

        diagnostics["screenshots"]["after_upload"] = await self._save_screenshot(
            page, filename=after_upload_screenshot_name
        )

        if not media_preview_verified:
            raise MediaPreviewNotVerifiedError(
                "Media preview could not be verified after upload."
            )

        # -- crop/filter/details Next transitions ---------------------
        next_transitions: list[str] = []

        for step_index in range(self.feed_config.max_next_transitions):
            lookup = await self._locate_unique_visible_enabled(
                dialog_root, self.feed_config.selectors.next_button
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

            await lookup.element.click(timeout=self.feed_config.action_click_timeout_ms)
            next_transitions.append(f"next_transition_{step_index + 1}")
            await page.wait_for_timeout(self.feed_config.next_click_wait_ms)

        diagnostics["next_transitions"] = next_transitions

        # -- caption entry ---------------------------------------------
        caption_box = await self._find_first(
            dialog_root, self.feed_config.selectors.caption_textbox
        )

        if caption_box is None:
            raise CaptionBoxNotFoundError(
                "Could not locate the caption textbox."
            )

        diagnostics["caption_box_found"] = True

        existing_caption_text = await self._composer_current_text(caption_box)

        try:
            await caption_box.fill(caption_result.final_caption)
        except Exception:
            await caption_box.click(timeout=2000)
            await caption_box.type(caption_result.final_caption)

        diagnostics["caption_filled"] = True

        await page.wait_for_timeout(self.feed_config.caption_settle_wait_ms)

        final_caption_text = await self._composer_current_text(caption_box)
        caption_verified = final_caption_text == caption_result.final_caption
        diagnostics["caption_verified"] = caption_verified

        if not caption_verified:
            raise CaptionVerificationMismatchError(
                "Caption textbox value does not exactly match the "
                "intended final caption after filling."
            )

        # -- Share button safety gate: locate + verify ONLY ------------
        share_lookup = await self._locate_unique_visible_enabled(
            dialog_root, self.feed_config.selectors.share_button
        )

        diagnostics["share_button_found"] = share_lookup.element is not None
        diagnostics["share_button_enabled"] = share_lookup.enabled
        # share_clicked / published are never mutated anywhere in this
        # method: they stay False in whatever the caller reports.

        diagnostics["screenshots"]["ready"] = await self._save_screenshot(
            page, filename=final_screenshot_name
        )

        if share_lookup.ambiguous:
            raise ShareButtonAmbiguousError(
                f"Found {share_lookup.candidate_count} ambiguous "
                "visible, enabled Share button candidates; refusing "
                "to guess. Nothing was clicked."
            )

        if share_lookup.disabled_candidate:
            raise ShareButtonDisabledError(
                "The Share button was found but is not usable (not "
                "visible and/or not enabled)."
            )

        if share_lookup.element is None:
            raise ShareButtonNotFoundError(
                "Could not find a unique, visible, enabled Share "
                "button. Nothing was clicked."
            )

        return _ShareReadyState(
            login_status=login_status,
            media_path=media_path,
            dialog_root=dialog_root,
            share_lookup=share_lookup,
            caption_verified=caption_verified,
            existing_caption_text=existing_caption_text,
            final_caption_text=final_caption_text,
            next_transitions=next_transitions,
            media_preview_verified=media_preview_verified,
            diagnostics=diagnostics,
        )

    # -- orchestration ----------------------------------------------------

    async def preview(
        self,
        job: PublishJob,
        caption_result: FinalCaptionResult,
    ) -> FeedPreviewResult:
        """
        Prepare the Feed post completely, verify the Share button is
        present and enabled, then stop. Never clicks Share, never
        presses Enter, never changes job.status, never writes to
        publish_queue.json.
        """
        started_at = self._now()
        diagnostics = self._default_diagnostics()
        diagnostics["caption_length"] = caption_result.final_caption_length
        diagnostics["hashtags_appended"] = caption_result.hashtags_appended

        playwright, context = await self.session._open_context()

        try:
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                state = await self._reach_share_ready(
                    page,
                    job,
                    caption_result,
                    diagnostics,
                    login_failure_screenshot_name=f"feed_preview_login_failed_{job.job_id}.png",
                    before_upload_screenshot_name=f"feed_preview_before_upload_{job.job_id}.png",
                    after_upload_screenshot_name=f"feed_preview_after_upload_{job.job_id}.png",
                    final_screenshot_name=f"feed_preview_ready_{job.job_id}.png",
                )
            except FeedPreviewError as exc:
                self._write_log(
                    started_at=started_at,
                    job=job,
                    diagnostics=diagnostics,
                    result="failed",
                    error=str(exc),
                )
                raise

            share_lookup = state.share_lookup

            result = FeedPreviewResult(
                job_id=job.job_id,
                production_date=job.production_date,
                persona_id=job.persona_id,
                login_status=state.login_status,
                media_path=state.media_path,
                caption_text=caption_result.caption_text,
                hashtags_text=caption_result.hashtags_text,
                final_caption=caption_result.final_caption,
                final_caption_length=caption_result.final_caption_length,
                hashtags_appended=caption_result.hashtags_appended,
                duplicate_hashtags_removed=list(caption_result.duplicate_hashtags_removed),
                create_control_found=diagnostics["create_control_found"],
                post_option_selected=diagnostics["post_option_selected"],
                media_uploaded=diagnostics["media_uploaded"],
                media_filename=Path(state.media_path).name,
                media_preview_verified=state.media_preview_verified,
                next_transitions=state.next_transitions,
                caption_box_found=diagnostics["caption_box_found"],
                caption_filled=diagnostics["caption_filled"],
                caption_verified=state.caption_verified,
                existing_caption_text=state.existing_caption_text,
                final_caption_text=state.final_caption_text,
                location_applied="not_applied",
                alt_text_applied="not_applied",
                share_button_found=True,
                share_button_enabled=share_lookup.enabled,
                share_button_interactive=share_lookup.interactive,
                share_clicked=False,
                published=False,
                screenshots=dict(diagnostics["screenshots"]),
                error=None,
            )

            self._write_log(
                started_at=started_at,
                job=job,
                diagnostics=diagnostics,
                result="ready_for_share",
                error=None,
            )

            return result

        finally:
            await context.close()
            await playwright.stop()


def build_feed_preview_controller(
    *,
    publisher_config_path: str | Path | None = None,
    instagram_config_path: str | Path | None = None,
    headless_override: bool | None = None,
) -> tuple[InstagramFeedPreviewController, PublisherConfig]:
    publisher_config = load_publisher_config(publisher_config_path)
    session_config = load_session_config(
        config_path=instagram_config_path,
        headless_override=headless_override,
    )
    feed_config = load_feed_preview_config(config_path=instagram_config_path)
    session = InstagramSession(session_config)

    controller = InstagramFeedPreviewController(
        session=session,
        feed_config=feed_config,
        publisher_config=publisher_config,
    )
    return controller, publisher_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Instagram Feed Publisher — Preview Only (Phase 10B). "
            "Prepares an approved instagram_feed job through to a "
            "ready-to-Share state and stops. Never clicks Share."
        )
    )

    parser.add_argument(
        "--preview",
        metavar="JOB_ID",
        required=True,
        help="Preview this approved instagram_feed job_id.",
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


def _print_result(result: FeedPreviewResult) -> None:
    print()
    print("AIKO Instagram Feed Publisher — Preview Only (Phase 10B)")
    print("----------------------------------------------------------")
    print(f"job id:                  {result.job_id}")
    print(f"login status:            {result.login_status}")
    print(f"media path:              {result.media_path}")
    print(f"final caption length:    {result.final_caption_length}")
    print(f"hashtags appended:       {result.hashtags_appended}")
    print(f"duplicate hashtags:      {result.duplicate_hashtags_removed}")
    print(f"create control found:    {result.create_control_found}")
    print(f"post option selected:    {result.post_option_selected}")
    print(f"media uploaded:          {result.media_uploaded}")
    print(f"media preview verified:  {result.media_preview_verified}")
    print(f"next transitions:        {result.next_transitions}")
    print(f"caption box found:       {result.caption_box_found}")
    print(f"caption filled:          {result.caption_filled}")
    print(f"caption verified:        {result.caption_verified}")
    print(f"location applied:        {result.location_applied}")
    print(f"alt text applied:        {result.alt_text_applied}")
    print(f"share button found:      {result.share_button_found}")
    print(f"share button enabled:    {result.share_button_enabled}")
    print(f"share button interactive:{result.share_button_interactive}")
    print(f"share clicked:           {result.share_clicked}")
    print(f"published:               {result.published}")
    print(f"screenshots:             {result.screenshots}")
    print()


async def _run(arguments: argparse.Namespace) -> int:
    publisher_config = load_publisher_config(arguments.config)

    job = load_job_by_id(publisher_config.queue_path(), arguments.preview)

    if job is None:
        print(f"[InstagramFeedPreview] No job found for job_id: {arguments.preview}")
        return 1

    try:
        check_job_eligible(job)
    except FeedPreviewIneligibleError as exc:
        print(f"[InstagramFeedPreview] {exc}")
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
        print(f"[InstagramFeedPreview] {exc}")
        return 1

    print(f"job_id:        {job.job_id}")
    print(f"final caption:\n{caption_result.final_caption}")
    print()

    controller, _ = build_feed_preview_controller(
        publisher_config_path=arguments.config,
        instagram_config_path=arguments.instagram_config,
        headless_override=True if arguments.headless else None,
    )

    try:
        result = await controller.preview(job, caption_result)
    except InstagramSessionError as exc:
        print(f"[InstagramFeedPreview] preview failed: {exc}")
        return 1
    except FeedPreviewError as exc:
        print(f"[InstagramFeedPreview] preview failed: {exc}")
        return 1

    _print_result(result)
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
