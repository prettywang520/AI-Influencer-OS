from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from src.social.instagram_models import LoginDetectionConfig

from .instagram_feed_preview import (
    DISALLOWED_ACTIONS,
    AmbiguousNextButtonError,
    CaptionBoxNotFoundError,
    CaptionLimitExceededError,
    CaptionVerificationMismatchError,
    FeedPreviewIneligibleError,
    FeedPreviewLoginError,
    FeedPreviewResult,
    InstagramFeedPreviewController,
    MediaPreviewNotVerifiedError,
    MediaUploadError,
    ShareButtonAmbiguousError,
    ShareButtonDisabledError,
    ShareButtonNotFoundError,
    build_final_caption,
    check_job_eligible,
    load_feed_preview_config,
)
from .models import PublisherConfig, PublishJob, app_root
from .queue_service import QueueService, load_job_by_id

# None of these tests open a real browser or contact Instagram.
# Playwright is simulated with small fake objects only.

MODULE_PATH = Path(__file__).resolve().parent / "instagram_feed_preview.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")

JOB_ID = "2026-08-01-instagram_feed"
BASE_URL = "https://www.instagram.com/"

CREATE_SELECTOR = 'role=link[name="New post"]'
POST_OPTION_SELECTOR = 'role=button[name="Post"]'
DIALOG_SELECTOR = "div[role='dialog']"
LOGIN_SELECTOR = "svg[aria-label='Home']"
LOGOUT_SELECTOR = "input[type='password']"


# ---------------------------------------------------------------------------
# Fake Playwright-shaped objects
# ---------------------------------------------------------------------------


class FakeElement:
    def __init__(
        self,
        *,
        text: str = "",
        value: str | None = None,
        visible: bool = True,
        disabled: bool = False,
        fail_click: bool = False,
        fail_is_visible: bool = False,
        fail_is_enabled: bool = False,
        fail_set_input_files: bool = False,
        fill_writes_value: str | None = None,
    ) -> None:
        self._text = text
        self._value = value
        self._visible = visible
        self._disabled = disabled
        self._fail_click = fail_click
        self._fail_is_visible = fail_is_visible
        self._fail_is_enabled = fail_is_enabled
        self._fail_set_input_files = fail_set_input_files
        self._fill_writes_value = fill_writes_value
        self.clicked = False
        self.click_count = 0
        self.filled_text: str | None = None
        self.typed_text: str | None = None
        self.set_input_files_calls: list[str] = []

    async def query_selector(self, selector: str):
        return None

    async def query_selector_all(self, selector: str):
        return []

    async def get_attribute(self, name: str):
        return None

    async def inner_text(self) -> str:
        return self._text

    async def input_value(self) -> str:
        if self._value is None:
            raise RuntimeError("not an <input>/<textarea>")
        return self._value

    async def is_visible(self) -> bool:
        if self._fail_is_visible:
            raise RuntimeError("simulated is_visible() failure")
        return self._visible

    async def is_enabled(self) -> bool:
        if self._fail_is_enabled:
            raise RuntimeError("simulated is_enabled() failure")
        return not self._disabled

    async def click(self, timeout=None):
        if self._fail_click:
            raise RuntimeError("simulated click failure")
        self.clicked = True
        self.click_count += 1

    async def fill(self, text: str):
        self.filled_text = text
        self._value = text if self._fill_writes_value is None else self._fill_writes_value

    async def type(self, text: str):
        self.typed_text = (self.typed_text or "") + text
        self._value = (self._value or "") + text

    async def set_input_files(self, path: str):
        if self._fail_set_input_files:
            raise RuntimeError("simulated upload failure")
        self.set_input_files_calls.append(path)

    # Explicitly absent on purpose: press(), keyboard, submit(). If
    # production code ever tried to call these, the test would fail
    # with AttributeError rather than silently succeeding.


class StagedDialog:
    """
    Hand-rolled fake for the Create Post dialog whose query results
    change as the simulated flow advances (upload -> Next transitions
    -> caption/details), mirroring how Instagram's real dialog only
    reveals the media preview / Next button / caption box / Share
    button at the right moment. A single generic FakeElement with a
    static children dict cannot express this sequencing.
    """

    def __init__(
        self,
        *,
        next_clicks_needed: int = 2,
        unsupported_media: bool = False,
        no_media_preview: bool = False,
        share_ambiguous: bool = False,
        share_disabled: bool = False,
        share_missing: bool = False,
        caption_mismatch: bool = False,
        fail_set_input_files: bool = False,
        fail_next_click: bool = False,
        ambiguous_next: bool = False,
    ) -> None:
        self.uploaded = False
        self.next_clicks_done = 0
        self.next_clicks_needed = next_clicks_needed
        self.unsupported_media = unsupported_media
        self.no_media_preview = no_media_preview
        self.share_ambiguous = share_ambiguous
        self.share_disabled = share_disabled
        self.share_missing = share_missing
        self.ambiguous_next = ambiguous_next

        self.file_input = FakeElement(fail_set_input_files=fail_set_input_files)
        self.media_preview_el = FakeElement()
        self.unsupported_el = FakeElement(text="couldn't be uploaded")
        self.next_button_el = FakeElement(fail_click=fail_next_click)
        self.next_button_el_2 = FakeElement()
        self.caption_box = FakeElement(
            fill_writes_value="WRONG TEXT" if caption_mismatch else None
        )
        self.share_button = FakeElement()
        self.share_button_2 = FakeElement()
        self.share_button_disabled = FakeElement(disabled=True)

        real_upload = self.file_input.set_input_files

        async def _upload(path: str):
            await real_upload(path)
            self.uploaded = True

        self.file_input.set_input_files = _upload

        real_next_click = self.next_button_el.click

        async def _next_click(timeout=None):
            await real_next_click(timeout=timeout)
            self.next_clicks_done += 1

        self.next_button_el.click = _next_click

    async def query_selector(self, selector: str):
        results = await self.query_selector_all(selector)
        return results[0] if results else None

    async def query_selector_all(self, selector: str):
        if selector == "input[type='file']":
            return [self.file_input]

        if selector in (
            "img[alt*='preview' i]",
            "div[aria-label*='Photo' i] img",
            "canvas",
        ):
            if self.uploaded and not self.no_media_preview:
                return [self.media_preview_el]
            return []

        if selector in (
            "text=file format",
            "div:has-text('not supported')",
            "div:has-text(\"couldn't be uploaded\")",
        ):
            if self.uploaded and self.unsupported_media:
                return [self.unsupported_el]
            return []

        if selector in ('role=button[name="Next"]', 'role=button[name="Next" i]'):
            if not self.uploaded or self.next_clicks_done >= self.next_clicks_needed:
                return []
            if self.ambiguous_next:
                return [self.next_button_el, self.next_button_el_2]
            return [self.next_button_el]

        if selector in (
            "textarea[aria-label='Write a caption...']",
            "div[contenteditable='true'][aria-label*='caption' i]",
        ):
            if self.uploaded and self.next_clicks_done >= self.next_clicks_needed:
                return [self.caption_box]
            return []

        if selector in ('role=button[name="Share"]', 'role=button[name="Share" i]'):
            if not (self.uploaded and self.next_clicks_done >= self.next_clicks_needed):
                return []
            if self.share_missing:
                return []
            if self.share_ambiguous:
                return [self.share_button, self.share_button_2]
            if self.share_disabled:
                return [self.share_button_disabled]
            return [self.share_button]

        return []


class FakePage:
    def __init__(
        self,
        *,
        children: dict[str, object] | None = None,
        url: str = BASE_URL,
        goto_error: Exception | None = None,
    ) -> None:
        self._children = children or {}
        self.url = url
        self._goto_error = goto_error
        self.screenshot_calls: list[str] = []

    async def goto(self, url, timeout=None, wait_until=None):
        if self._goto_error is not None:
            raise self._goto_error

    async def wait_for_timeout(self, ms):
        pass

    async def query_selector(self, selector: str):
        return self._children.get(selector)

    async def query_selector_all(self, selector: str):
        element = self._children.get(selector)
        return [element] if element is not None else []

    async def screenshot(self, path, full_page=True):
        self.screenshot_calls.append(path)


class FakeContext:
    def __init__(self, page: FakePage) -> None:
        self.pages = [page]
        self.closed = False

    async def new_page(self):
        return self.pages[0]

    async def close(self):
        self.closed = True


class FakePlaywright:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self):
        self.stopped = True


class FakeSessionConfig:
    def __init__(self) -> None:
        self.base_url = BASE_URL
        self.navigation_timeout_ms = 30000
        self.login_detection = LoginDetectionConfig(
            logged_in_selectors=[LOGIN_SELECTOR],
            logged_out_selectors=[LOGOUT_SELECTOR],
            logged_out_url_markers=["accounts/login"],
            pending_url_markers=["checkpoint"],
        )


class FakeSession:
    """Duck-typed stand-in for InstagramSession — no real Playwright."""

    def __init__(self, page: FakePage) -> None:
        self.config = FakeSessionConfig()
        self._playwright = FakePlaywright()
        self._context = FakeContext(page)

    async def _open_context(self):
        return self._playwright, self._context


def _build_page(*, logged_in: bool = True, **dialog_kwargs) -> tuple[FakePage, StagedDialog | None]:
    dialog = StagedDialog(**dialog_kwargs)
    children: dict[str, object] = {
        CREATE_SELECTOR: FakeElement(),
        POST_OPTION_SELECTOR: FakeElement(),
        DIALOG_SELECTOR: dialog,
    }

    if logged_in:
        children[LOGIN_SELECTOR] = FakeElement()
    else:
        children[LOGOUT_SELECTOR] = FakeElement()

    return FakePage(children=children), dialog


def _default_feed_config():
    return load_feed_preview_config()


def _controller(page: FakePage, *, temp_dir: Path) -> InstagramFeedPreviewController:
    return InstagramFeedPreviewController(
        session=FakeSession(page),
        feed_config=_default_feed_config(),
        publisher_config=PublisherConfig(),
        screenshots_dir=temp_dir / "screenshots",
        logs_dir=temp_dir / "logs",
    )


def _job(**overrides) -> PublishJob:
    defaults = dict(
        job_id=JOB_ID,
        production_date="2026-08-01",
        persona_id="aiko",
        platform="instagram",
        content_type="instagram_feed",
        status="approved",
        media_paths=[],
        caption_text="hello world",
        hashtags=["#aikotravel", "#travelcreator"],
    )
    defaults.update(overrides)
    return PublishJob(**defaults)


class MediaTempMixin:
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.media_path = self.temp_dir / "feed_01.png"
        self.media_path.write_bytes(b"fake-image-bytes")


# ---------------------------------------------------------------------------
# Eligibility (pure, no browser)
# ---------------------------------------------------------------------------


class EligibilityTests(MediaTempMixin, unittest.TestCase):
    def test_approved_instagram_feed_job_is_eligible(self) -> None:
        job = _job(status="approved", media_paths=[str(self.media_path)])
        check_job_eligible(job)  # does not raise

    def test_pending_approval_is_rejected(self) -> None:
        job = _job(status="pending_approval", media_paths=[str(self.media_path)])
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)

    def test_rejected_is_rejected(self) -> None:
        job = _job(status="rejected", media_paths=[str(self.media_path)])
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)

    def test_published_is_rejected(self) -> None:
        job = _job(status="published", media_paths=[str(self.media_path)])
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)

    def test_scheduled_is_rejected(self) -> None:
        job = _job(status="scheduled", media_paths=[str(self.media_path)])
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)

    def test_non_feed_content_type_is_rejected(self) -> None:
        job = _job(
            status="approved",
            content_type="instagram_story",
            media_paths=[str(self.media_path)],
        )
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)

    def test_missing_media_file_fails_before_browser(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.temp_dir / "does_not_exist.png")],
        )
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)

    def test_zero_byte_media_fails_before_browser(self) -> None:
        zero_byte_path = self.temp_dir / "empty.png"
        zero_byte_path.write_bytes(b"")
        job = _job(status="approved", media_paths=[str(zero_byte_path)])
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)

    def test_empty_caption_fails_before_browser(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.media_path)],
            caption_text="   ",
        )
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)

    def test_wrong_media_count_fails(self) -> None:
        job = _job(status="approved", media_paths=[])
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_eligible(job)


# ---------------------------------------------------------------------------
# Caption building (pure, no browser)
# ---------------------------------------------------------------------------


class CaptionBuildingTests(unittest.TestCase):
    def test_caption_and_hashtags_joined_with_one_blank_line(self) -> None:
        result = build_final_caption(
            "hello world", ["#aikotravel", "#travelcreator"], max_characters=None
        )
        self.assertEqual(
            result.final_caption, "hello world\n\n#aikotravel #travelcreator"
        )
        self.assertTrue(result.hashtags_appended)

    def test_duplicate_hashtags_are_not_appended_twice(self) -> None:
        result = build_final_caption(
            "loving #aikotravel today",
            ["#aikotravel", "#travelcreator"],
            max_characters=None,
        )
        self.assertNotIn("#aikotravel", result.hashtags_text)
        self.assertIn("#aikotravel", result.duplicate_hashtags_removed)
        self.assertEqual(
            result.final_caption, "loving #aikotravel today\n\n#travelcreator"
        )

    def test_all_hashtags_duplicate_means_no_append(self) -> None:
        result = build_final_caption(
            "loving #aikotravel today", ["#aikotravel"], max_characters=None
        )
        self.assertFalse(result.hashtags_appended)
        self.assertEqual(result.final_caption, "loving #aikotravel today")

    def test_trailing_whitespace_stripped_line_breaks_preserved(self) -> None:
        result = build_final_caption(
            "line one\nline two   \n\n  ", [], max_characters=None
        )
        self.assertEqual(result.final_caption, "line one\nline two")

    def test_caption_limit_exceeded_raises(self) -> None:
        with self.assertRaises(CaptionLimitExceededError):
            build_final_caption("x" * 50, ["#a"], max_characters=10)

    def test_caption_limit_not_exceeded_passes(self) -> None:
        result = build_final_caption("short", ["#a"], max_characters=2200)
        self.assertEqual(result.final_caption_length, len(result.final_caption))


# ---------------------------------------------------------------------------
# Login safety
# ---------------------------------------------------------------------------


class LoginFailureTests(MediaTempMixin, unittest.TestCase):
    def test_logged_out_stops_before_create_control(self) -> None:
        page, dialog = _build_page(logged_in=False)

        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.media_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(FeedPreviewLoginError):
                asyncio.run(controller.preview(job, caption_result))

        create_element = page._children[CREATE_SELECTOR]
        self.assertFalse(create_element.clicked)
        self.assertTrue(page.screenshot_calls)

    def test_login_uncertain_stops_before_create_control(self) -> None:
        page, dialog = _build_page(logged_in=False)
        page._children.pop(LOGOUT_SELECTOR)  # neither in nor out -> unknown

        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.media_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(FeedPreviewLoginError):
                asyncio.run(controller.preview(job, caption_result))

        create_element = page._children[CREATE_SELECTOR]
        self.assertFalse(create_element.clicked)


# ---------------------------------------------------------------------------
# Full success path
# ---------------------------------------------------------------------------


class FullPreviewSuccessTests(MediaTempMixin, unittest.TestCase):
    def _run_success(self, **dialog_kwargs) -> tuple[FeedPreviewResult, FakePage, StagedDialog, Path]:
        page, dialog = _build_page(**dialog_kwargs)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        temp_dir = Path(temp_dir_ctx.name)
        controller = _controller(page, temp_dir=temp_dir)

        job = _job(
            media_paths=[str(self.media_path)],
            caption_text="line one\nline two",
        )
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=2200)

        result = asyncio.run(controller.preview(job, caption_result))
        return result, page, dialog, temp_dir

    def test_full_preview_reaches_ready_state(self) -> None:
        result, _page, _dialog, _temp_dir = self._run_success()
        self.assertEqual(result.login_status, "logged_in")
        self.assertTrue(result.media_uploaded)
        self.assertTrue(result.media_preview_verified)
        self.assertTrue(result.caption_verified)
        self.assertTrue(result.share_button_found)
        self.assertTrue(result.share_button_enabled)
        self.assertFalse(result.share_clicked)
        self.assertFalse(result.published)

    def test_create_control_is_clicked_exactly_once(self) -> None:
        _result, page, _dialog, _temp_dir = self._run_success()
        create_element = page._children[CREATE_SELECTOR]
        self.assertEqual(create_element.click_count, 1)

    def test_post_option_selected_not_story_or_reel(self) -> None:
        _result, page, _dialog, _temp_dir = self._run_success()
        post_element = page._children[POST_OPTION_SELECTOR]
        self.assertEqual(post_element.click_count, 1)

        # There is no story_option/reel_option field on the selectors
        # dataclass at all, so Story/Reel cannot be selected even
        # accidentally.
        config = _default_feed_config()
        self.assertFalse(hasattr(config.selectors, "story_option"))
        self.assertFalse(hasattr(config.selectors, "reel_option"))

    def test_exact_media_path_uploaded_once(self) -> None:
        result, _page, dialog, _temp_dir = self._run_success()
        self.assertEqual(dialog.file_input.set_input_files_calls, [str(self.media_path)])
        self.assertEqual(result.media_filename, self.media_path.name)

    def test_media_preview_must_be_verified(self) -> None:
        page, _dialog = _build_page(no_media_preview=True)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        controller = _controller(page, temp_dir=Path(temp_dir_ctx.name))
        job = _job(media_paths=[str(self.media_path)])
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

        with self.assertRaises(MediaPreviewNotVerifiedError):
            asyncio.run(controller.preview(job, caption_result))

    def test_unsupported_media_raises(self) -> None:
        page, _dialog = _build_page(unsupported_media=True)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        controller = _controller(page, temp_dir=Path(temp_dir_ctx.name))
        job = _job(media_paths=[str(self.media_path)])
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

        with self.assertRaises(MediaUploadError):
            asyncio.run(controller.preview(job, caption_result))

    def test_two_next_transitions_are_recorded(self) -> None:
        result, _page, dialog, _temp_dir = self._run_success(next_clicks_needed=2)
        self.assertEqual(dialog.next_clicks_done, 2)
        self.assertEqual(result.next_transitions, ["next_transition_1", "next_transition_2"])

    def test_single_next_transition_is_sufficient(self) -> None:
        result, _page, dialog, _temp_dir = self._run_success(next_clicks_needed=1)
        self.assertEqual(dialog.next_clicks_done, 1)
        self.assertEqual(result.next_transitions, ["next_transition_1"])

    def test_ambiguous_next_button_fails_safe(self) -> None:
        page, dialog = _build_page(ambiguous_next=True)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        controller = _controller(page, temp_dir=Path(temp_dir_ctx.name))
        job = _job(media_paths=[str(self.media_path)])
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

        with self.assertRaises(AmbiguousNextButtonError):
            asyncio.run(controller.preview(job, caption_result))

        self.assertFalse(dialog.next_button_el.clicked)
        self.assertFalse(dialog.next_button_el_2.clicked)

    def test_caption_filled_once_and_line_breaks_preserved(self) -> None:
        result, _page, dialog, _temp_dir = self._run_success()
        # filled exactly once, with the caption's line breaks intact
        # inside the final_caption (hashtags appended after them).
        self.assertEqual(dialog.caption_box.filled_text, result.final_caption)
        self.assertTrue(result.final_caption.startswith("line one\nline two\n\n"))
        self.assertIn("\n", result.final_caption_text)
        self.assertEqual(result.final_caption_text, result.final_caption)

    def test_caption_verification_mismatch_fails(self) -> None:
        page, _dialog = _build_page(caption_mismatch=True)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        controller = _controller(page, temp_dir=Path(temp_dir_ctx.name))
        job = _job(media_paths=[str(self.media_path)])
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

        with self.assertRaises(CaptionVerificationMismatchError):
            asyncio.run(controller.preview(job, caption_result))

    def test_share_button_found_but_never_clicked(self) -> None:
        result, _page, dialog, _temp_dir = self._run_success()
        self.assertTrue(result.share_button_found)
        self.assertFalse(dialog.share_button.clicked)
        self.assertEqual(dialog.share_button.click_count, 0)
        self.assertFalse(result.share_clicked)
        self.assertFalse(result.published)

    def test_share_button_ambiguous_fails_safe(self) -> None:
        page, dialog = _build_page(share_ambiguous=True)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        controller = _controller(page, temp_dir=Path(temp_dir_ctx.name))
        job = _job(media_paths=[str(self.media_path)])
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

        with self.assertRaises(ShareButtonAmbiguousError):
            asyncio.run(controller.preview(job, caption_result))

        self.assertFalse(dialog.share_button.clicked)
        self.assertFalse(dialog.share_button_2.clicked)

    def test_share_button_disabled_fails_safe(self) -> None:
        page, dialog = _build_page(share_disabled=True)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        controller = _controller(page, temp_dir=Path(temp_dir_ctx.name))
        job = _job(media_paths=[str(self.media_path)])
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

        with self.assertRaises(ShareButtonDisabledError):
            asyncio.run(controller.preview(job, caption_result))

        self.assertFalse(dialog.share_button_disabled.clicked)

    def test_share_button_missing_fails(self) -> None:
        page, _dialog = _build_page(share_missing=True)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        controller = _controller(page, temp_dir=Path(temp_dir_ctx.name))
        job = _job(media_paths=[str(self.media_path)])
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

        with self.assertRaises(ShareButtonNotFoundError):
            asyncio.run(controller.preview(job, caption_result))

    def test_exactly_three_required_screenshots_are_saved(self) -> None:
        result, page, _dialog, _temp_dir = self._run_success()
        self.assertIsNotNone(result.screenshots["before_upload"])
        self.assertIsNotNone(result.screenshots["after_upload"])
        self.assertIsNotNone(result.screenshots["ready"])
        self.assertTrue(result.screenshots["before_upload"].endswith(f"before_upload_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["after_upload"].endswith(f"after_upload_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["ready"].endswith(f"ready_{JOB_ID}.png"))

    def test_location_and_alt_text_reported_not_applied(self) -> None:
        result, _page, _dialog, _temp_dir = self._run_success()
        self.assertEqual(result.location_applied, "not_applied")
        self.assertEqual(result.alt_text_applied, "not_applied")


# ---------------------------------------------------------------------------
# Queue / status / history safety
# ---------------------------------------------------------------------------


class QueueSafetyTests(MediaTempMixin, unittest.TestCase):
    def _write_real_queue(self, temp_dir: Path) -> Path:
        queue_path = temp_dir / "publish_queue.json"
        service = QueueService(queue_path)
        service.add_job(
            _job(status="approved", media_paths=[str(self.media_path)])
        )
        return queue_path

    def test_queue_json_byte_for_byte_unchanged_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            self.assertIsNotNone(job)

            page, _dialog = _build_page()
            controller = _controller(page, temp_dir=temp_dir)
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            asyncio.run(controller.preview(job, caption_result))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)
            self.assertEqual(hashlib.md5(before).hexdigest(), hashlib.md5(after).hexdigest())

    def test_queue_json_byte_for_byte_unchanged_after_login_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            page, _dialog = _build_page(logged_in=False)
            controller = _controller(page, temp_dir=temp_dir)
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(FeedPreviewLoginError):
                asyncio.run(controller.preview(job, caption_result))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)

    def test_queue_json_byte_for_byte_unchanged_after_selector_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            page, _dialog = _build_page(share_missing=True)
            controller = _controller(page, temp_dir=temp_dir)
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(ShareButtonNotFoundError):
                asyncio.run(controller.preview(job, caption_result))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)

    def test_job_status_never_mutated_by_preview(self) -> None:
        page, _dialog = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(status="approved", media_paths=[str(self.media_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            asyncio.run(controller.preview(job, caption_result))
            self.assertEqual(job.status, "approved")
            self.assertIsNone(job.published_at)
            self.assertIsNone(job.platform_post_id)

    def test_no_history_file_is_written(self) -> None:
        page, _dialog = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            controller = _controller(page, temp_dir=temp_dir)
            job = _job(status="approved", media_paths=[str(self.media_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            asyncio.run(controller.preview(job, caption_result))

            history_dir = temp_dir / "history"
            self.assertFalse(history_dir.exists())

    def test_diagnostic_log_is_written_with_required_fields(self) -> None:
        page, _dialog = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            controller = _controller(page, temp_dir=temp_dir)
            job = _job(status="approved", media_paths=[str(self.media_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            asyncio.run(controller.preview(job, caption_result))

            logs_dir = temp_dir / "logs"
            log_files = list(logs_dir.glob(f"instagram_feed_preview_{JOB_ID}_*.json"))
            self.assertEqual(len(log_files), 1)

            payload = json.loads(log_files[0].read_text(encoding="utf-8"))
            for field_name in (
                "started_at",
                "finished_at",
                "job_id",
                "production_date",
                "persona_id",
                "media_path",
                "caption_length",
                "hashtags_appended",
                "login_status",
                "create_control_found",
                "post_option_selected",
                "media_uploaded",
                "media_preview_verified",
                "next_transitions",
                "caption_box_found",
                "caption_verified",
                "share_button_found",
                "share_button_enabled",
                "share_clicked",
                "published",
                "screenshots",
                "result",
                "error",
            ):
                self.assertIn(field_name, payload)

            self.assertFalse(payload["share_clicked"])
            self.assertFalse(payload["published"])


# ---------------------------------------------------------------------------
# Structural safety scan
# ---------------------------------------------------------------------------


class SafetyTests(unittest.TestCase):
    def test_no_press_or_keyboard_press_calls_exist(self) -> None:
        self.assertNotIn(".press(", MODULE_SOURCE)
        self.assertNotIn("keyboard.press", MODULE_SOURCE)
        self.assertNotIn("keyboard.type", MODULE_SOURCE)

    def test_no_share_click_call_pattern_exists(self) -> None:
        # Any call shaped like "<name containing 'share'>.click(" would
        # be an actual click on the Share button. None should exist —
        # the word "Share" may only appear in selectors, field names,
        # comments, and error/log messages.
        matches = re.findall(r"\b\w*[Ss]hare\w*\.click\(", MODULE_SOURCE)
        self.assertEqual(matches, [])

    def test_no_status_assignment_to_publishing_or_published(self) -> None:
        pattern = re.compile(r"""\.status\s*=\s*["'](publishing|published)["']""")
        self.assertIsNone(pattern.search(MODULE_SOURCE))

    def test_history_service_is_never_imported(self) -> None:
        self.assertNotIn("history_service", MODULE_SOURCE)

    def test_writable_queue_service_class_is_never_imported(self) -> None:
        self.assertNotIn("import QueueService", MODULE_SOURCE)
        self.assertNotIn("QueueService,", MODULE_SOURCE)
        self.assertIn("load_job_by_id", MODULE_SOURCE)

    def test_no_publish_flag_or_share_flag_cli_arguments(self) -> None:
        self.assertNotIn('"--publish"', MODULE_SOURCE)
        self.assertNotIn('"--share"', MODULE_SOURCE)

    def test_disallowed_action_methods_not_implemented_on_controller(self) -> None:
        for action in DISALLOWED_ACTIONS:
            self.assertFalse(
                hasattr(InstagramFeedPreviewController, action),
                f"InstagramFeedPreviewController must not implement '{action}'",
            )

    def test_fake_element_and_controller_have_no_press_capability(self) -> None:
        for forbidden_name in ("press", "keyboard", "submit"):
            self.assertFalse(hasattr(FakeElement, forbidden_name), "test fixture sanity check")
            self.assertFalse(
                hasattr(InstagramFeedPreviewController, forbidden_name),
                f"InstagramFeedPreviewController must not implement '{forbidden_name}'",
            )

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("share", DISALLOWED_ACTIONS)
        self.assertIn("publish", DISALLOWED_ACTIONS)
        self.assertIn("press_enter", DISALLOWED_ACTIONS)
        self.assertIn("post_story", DISALLOWED_ACTIONS)
        self.assertIn("post_reel", DISALLOWED_ACTIONS)


# ---------------------------------------------------------------------------
# src/social untouched
# ---------------------------------------------------------------------------


class SocialPackageUntouchedTests(unittest.TestCase):
    def test_social_files_are_byte_for_byte_unchanged_after_a_full_preview_run(self) -> None:
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

        page, _dialog = _build_page()
        media_dir_ctx = tempfile.TemporaryDirectory()
        media_dir = Path(media_dir_ctx.name)
        media_path = media_dir / "feed_01.png"
        media_path.write_bytes(b"fake-image-bytes")

        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(status="approved", media_paths=[str(media_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            asyncio.run(controller.preview(job, caption_result))

        media_dir_ctx.cleanup()

        after = _snapshot()
        self.assertEqual(before, after)
        self.assertTrue(before, "sanity check: at least one watched file must exist")


if __name__ == "__main__":
    unittest.main()
