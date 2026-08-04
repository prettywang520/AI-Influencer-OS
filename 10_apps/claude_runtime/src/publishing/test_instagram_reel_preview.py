from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from src.social.instagram_models import LoginDetectionConfig

from .instagram_reel_preview import (
    DISALLOWED_ACTIONS,
    AmbiguousNextButtonError,
    CaptionBoxNotFoundError,
    CaptionVerificationMismatchError,
    CreateControlNotFoundError,
    InstagramReelPreviewController,
    ReelOptionNotFoundError,
    ReelPreviewIneligibleError,
    ReelPreviewLoginError,
    ReelProcessingTimeoutError,
    ShareButtonAmbiguousError,
    ShareButtonNotFoundError,
    VideoPreviewNotVerifiedError,
    VideoUploadError,
    check_final_reel_media,
    check_reel_job_eligible,
    load_reel_preview_config,
)
from .instagram_feed_preview import CaptionLimitExceededError, build_final_caption
from .models import PublisherConfig, PublishJob, app_root
from .queue_service import QueueService, load_job_by_id

# None of these tests open a real browser or contact Instagram.
# Playwright is simulated with small fake objects only.

MODULE_PATH = Path(__file__).resolve().parent / "instagram_reel_preview.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")

JOB_ID = "2026-08-01-instagram_reel"
PRODUCTION_DATE = "2026-08-01"
BASE_URL = "https://www.instagram.com/"

CREATE_SELECTOR = "svg[aria-label='New post']"
REEL_SELECTOR = "svg[aria-label='Reel']"
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
        visible: bool = True,
        disabled: bool = False,
        fail_click: bool = False,
        fail_set_input_files: bool = False,
        fill_writes_value: str | None = None,
    ) -> None:
        self._text = text
        self._visible = visible
        self._disabled = disabled
        self._fail_click = fail_click
        self._fail_set_input_files = fail_set_input_files
        self._fill_writes_value = fill_writes_value
        self._value: str | None = None
        self.clicked = False
        self.click_count = 0
        self.filled_text: str | None = None
        self.set_input_files_calls: list[str] = []

    async def query_selector(self, selector: str):
        return None

    async def query_selector_all(self, selector: str):
        return []

    async def inner_text(self) -> str:
        return self._text

    async def input_value(self) -> str:
        if self._value is None:
            raise RuntimeError("not an <input>/<textarea>")
        return self._value

    async def is_visible(self) -> bool:
        return self._visible

    async def is_enabled(self) -> bool:
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
        self._value = (self._value or "") + text

    async def set_input_files(self, path: str):
        if self._fail_set_input_files:
            raise RuntimeError("simulated upload failure")
        self.set_input_files_calls.append(path)

    # Explicitly absent on purpose: press(), keyboard, submit(). If
    # production code ever tried to call these, the test would fail
    # with AttributeError rather than silently succeeding.


class StagedReelDialog:
    """
    Hand-rolled fake for the Reel composer dialog whose query results
    change as the simulated flow advances (upload -> processing ->
    Next transitions -> cover/caption/details), mirroring
    test_instagram_feed_preview.StagedDialog's approach.
    """

    def __init__(
        self,
        *,
        next_clicks_needed: int = 1,
        unsupported_media: bool = False,
        no_video_preview: bool = False,
        processing_polls_before_clear: int = 0,
        processing_never_clears: bool = False,
        no_cover: bool = False,
        caption_mismatch: bool = False,
        fail_set_input_files: bool = False,
        ambiguous_next: bool = False,
        share_ambiguous: bool = False,
        share_missing: bool = False,
    ) -> None:
        self.uploaded = False
        self.next_clicks_done = 0
        self.next_clicks_needed = next_clicks_needed
        self.unsupported_media = unsupported_media
        self.no_video_preview = no_video_preview
        self.processing_polls_remaining = processing_polls_before_clear
        self.processing_never_clears = processing_never_clears
        self.no_cover = no_cover
        self.ambiguous_next = ambiguous_next
        self.share_ambiguous = share_ambiguous
        self.share_missing = share_missing

        self.file_input = FakeElement(fail_set_input_files=fail_set_input_files)
        self.video_preview_el = FakeElement()
        self.unsupported_el = FakeElement(text="couldn't be uploaded")
        self.processing_el = FakeElement()
        self.next_button_el = FakeElement()
        self.next_button_el_2 = FakeElement()
        self.cover_el = FakeElement()
        self.caption_box = FakeElement(
            fill_writes_value="WRONG TEXT" if caption_mismatch else None
        )
        self.share_button = FakeElement()
        self.share_button_2 = FakeElement()

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

        if selector == "video":
            if self.uploaded and not self.no_video_preview:
                return [self.video_preview_el]
            return []

        if selector in (
            "text=file format",
            "div:has-text('not supported')",
            "div:has-text(\"couldn't be uploaded\")",
        ):
            if self.uploaded and self.unsupported_media:
                return [self.unsupported_el]
            return []

        if selector == "text=Processing":
            if not self.uploaded:
                return []
            if self.processing_never_clears:
                return [self.processing_el]
            if self.processing_polls_remaining > 0:
                self.processing_polls_remaining -= 1
                return [self.processing_el]
            return []

        if selector in ('role=button[name="Next"]', 'role=button[name="Next" i]'):
            if not self.uploaded or self.next_clicks_done >= self.next_clicks_needed:
                return []
            if self.ambiguous_next:
                return [self.next_button_el, self.next_button_el_2]
            return [self.next_button_el]

        if selector == "div[aria-label*='cover' i]":
            if self.uploaded and self.next_clicks_done >= self.next_clicks_needed and not self.no_cover:
                return [self.cover_el]
            return []

        if selector == "div[contenteditable='true']":
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
        results = await self.query_selector_all(selector)
        return results[0] if results else None

    async def query_selector_all(self, selector: str):
        element = self._children.get(selector)
        if element is None:
            return []
        if isinstance(element, list):
            return element
        return [element]

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


def _build_page(*, logged_in: bool = True, login_unknown: bool = False, **dialog_kwargs):
    dialog = StagedReelDialog(**dialog_kwargs)
    children: dict[str, object] = {
        CREATE_SELECTOR: FakeElement(),
        REEL_SELECTOR: FakeElement(),
        DIALOG_SELECTOR: dialog,
    }

    if logged_in:
        children[LOGIN_SELECTOR] = FakeElement()
    elif not login_unknown:
        children[LOGOUT_SELECTOR] = FakeElement()

    return FakePage(children=children), dialog


def _default_reel_config():
    return load_reel_preview_config()


def _controller(page: FakePage, *, temp_dir: Path) -> InstagramReelPreviewController:
    return InstagramReelPreviewController(
        session=FakeSession(page),
        reel_config=_default_reel_config(),
        publisher_config=PublisherConfig(),
        screenshots_dir=temp_dir / "screenshots",
        logs_dir=temp_dir / "logs",
    )


def _job(**overrides) -> PublishJob:
    defaults = dict(
        job_id=JOB_ID,
        production_date=PRODUCTION_DATE,
        persona_id="aiko",
        platform="instagram",
        content_type="instagram_reel",
        status="approved",
        media_paths=[],
        caption_text="hello reel",
        hashtags=["#aikotravel", "#travelcreator"],
    )
    defaults.update(overrides)
    return PublishJob(**defaults)


class MediaTempMixin:
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.output_root = self.temp_dir / "output"
        self.video_dir = self.output_root / PRODUCTION_DATE / "videos"
        self.video_dir.mkdir(parents=True)
        self.video_path = self.video_dir / "reel_final.mp4"
        self.video_path.write_bytes(b"fake-video-bytes")
        self.config = PublisherConfig()


# ---------------------------------------------------------------------------
# Eligibility (pure, no browser)
# ---------------------------------------------------------------------------


class EligibilityTests(MediaTempMixin, unittest.TestCase):
    def test_approved_instagram_reel_job_is_eligible(self) -> None:
        job = _job(status="approved", media_paths=[str(self.video_path)])
        check_reel_job_eligible(job, self.config)  # does not raise

    def test_instagram_feed_content_type_is_rejected(self) -> None:
        job = _job(
            status="approved",
            content_type="instagram_feed",
            media_paths=[str(self.video_path)],
        )
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_instagram_story_content_type_is_rejected(self) -> None:
        job = _job(
            status="approved",
            content_type="instagram_story",
            media_paths=[str(self.video_path)],
        )
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_threads_post_content_type_is_rejected(self) -> None:
        job = _job(
            status="approved",
            content_type="threads_post",
            media_paths=[str(self.video_path)],
        )
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_pending_approval_is_rejected(self) -> None:
        job = _job(status="pending_approval", media_paths=[str(self.video_path)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_rejected_is_rejected(self) -> None:
        job = _job(status="rejected", media_paths=[str(self.video_path)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_scheduled_is_rejected(self) -> None:
        job = _job(status="scheduled", media_paths=[str(self.video_path)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_publishing_is_rejected(self) -> None:
        job = _job(status="publishing", media_paths=[str(self.video_path)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_published_is_rejected(self) -> None:
        job = _job(status="published", media_paths=[str(self.video_path)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_missing_media_file_fails_before_browser(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.video_dir / "does_not_exist.mp4")],
        )
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_zero_byte_media_fails_before_browser(self) -> None:
        zero_byte_path = self.video_dir / "empty.mp4"
        zero_byte_path.write_bytes(b"")
        job = _job(status="approved", media_paths=[str(zero_byte_path)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_invalid_extension_fails_before_browser(self) -> None:
        png_path = self.video_dir / "reel_final.png"
        png_path.write_bytes(b"not-a-video")
        job = _job(status="approved", media_paths=[str(png_path)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_empty_caption_fails_before_browser(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.video_path)],
            caption_text="   ",
        )
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_wrong_media_count_fails(self) -> None:
        job = _job(status="approved", media_paths=[])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_already_published_at_is_rejected(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.video_path)],
            published_at="2026-08-01T00:00:00+00:00",
        )
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)

    def test_already_has_platform_post_id_is_rejected(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.video_path)],
            platform_post_id="abc123",
        )
        with self.assertRaises(ReelPreviewIneligibleError):
            check_reel_job_eligible(job, self.config)


# ---------------------------------------------------------------------------
# Final Reel media validation (pure, no browser)
# ---------------------------------------------------------------------------


class FinalReelMediaTests(MediaTempMixin, unittest.TestCase):
    def test_exact_final_video_path_passes(self) -> None:
        job = _job(media_paths=[str(self.video_path)])
        check_final_reel_media(job, self.config, output_root=self.output_root)  # does not raise

    def test_individual_scene_path_is_rejected(self) -> None:
        scene_dir = self.video_dir / "reel_scenes"
        scene_dir.mkdir()
        scene_path = scene_dir / "shot_01.mp4"
        scene_path.write_bytes(b"scene-bytes")
        job = _job(media_paths=[str(scene_path)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_final_reel_media(job, self.config, output_root=self.output_root)

    def test_missing_final_video_reports_exact_expected_path(self) -> None:
        self.video_path.unlink()
        job = _job(media_paths=[str(self.video_path)])
        with self.assertRaisesRegex(ReelPreviewIneligibleError, re.escape(str(self.video_path))):
            check_final_reel_media(job, self.config, output_root=self.output_root)

    def test_unrelated_video_path_is_rejected(self) -> None:
        unrelated = self.temp_dir / "unrelated.mp4"
        unrelated.write_bytes(b"unrelated-bytes")
        job = _job(media_paths=[str(unrelated)])
        with self.assertRaises(ReelPreviewIneligibleError):
            check_final_reel_media(job, self.config, output_root=self.output_root)


# ---------------------------------------------------------------------------
# Login safety
# ---------------------------------------------------------------------------


class LoginFailureTests(MediaTempMixin, unittest.TestCase):
    def test_logged_out_stops_before_create_control(self) -> None:
        page, _dialog = _build_page(logged_in=False)

        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(ReelPreviewLoginError):
                asyncio.run(controller.preview(job, caption_result))

        create_elements = page._children[CREATE_SELECTOR]
        self.assertFalse(create_elements.clicked)
        self.assertTrue(page.screenshot_calls)

    def test_login_uncertain_stops_before_create_control(self) -> None:
        page, _dialog = _build_page(logged_in=False, login_unknown=True)

        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(ReelPreviewLoginError):
                asyncio.run(controller.preview(job, caption_result))

        create_elements = page._children[CREATE_SELECTOR]
        self.assertFalse(create_elements.clicked)


# ---------------------------------------------------------------------------
# Full success path
# ---------------------------------------------------------------------------


class FullPreviewSuccessTests(MediaTempMixin, unittest.TestCase):
    def _run_success(self, **dialog_kwargs):
        page, dialog = _build_page(**dialog_kwargs)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        temp_dir = Path(temp_dir_ctx.name)
        controller = _controller(page, temp_dir=temp_dir)

        job = _job(media_paths=[str(self.video_path)], caption_text="line one\nline two")
        caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=2200)

        result = asyncio.run(controller.preview(job, caption_result))
        return result, page, dialog, temp_dir

    def test_full_preview_reaches_ready_state(self) -> None:
        result, _page, _dialog, _temp_dir = self._run_success()
        self.assertEqual(result.login_status, "logged_in")
        self.assertTrue(result.video_uploaded)
        self.assertTrue(result.video_preview_verified)
        self.assertTrue(result.caption_verified)
        self.assertTrue(result.cover_verified)
        self.assertTrue(result.share_button_found)
        self.assertTrue(result.share_button_enabled)
        self.assertFalse(result.share_clicked)
        self.assertFalse(result.published)
        self.assertIsNone(result.error)

    def test_create_control_is_clicked_exactly_once(self) -> None:
        _result, page, _dialog, _temp_dir = self._run_success()
        create_element = page._children[CREATE_SELECTOR]
        self.assertEqual(create_element.click_count, 1)

    def test_reel_option_selected_not_post_or_story(self) -> None:
        _result, page, _dialog, _temp_dir = self._run_success()
        reel_element = page._children[REEL_SELECTOR]
        self.assertEqual(reel_element.click_count, 1)

        config = _default_reel_config()
        self.assertFalse(hasattr(config.selectors, "post_option"))
        self.assertFalse(hasattr(config.selectors, "story_option"))

    def test_create_control_missing_fails_safely(self) -> None:
        page, _dialog = _build_page()
        page._children[CREATE_SELECTOR] = []
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(CreateControlNotFoundError):
                asyncio.run(controller.preview(job, caption_result))

    def test_reel_option_missing_fails_safely(self) -> None:
        page, _dialog = _build_page()
        page._children[REEL_SELECTOR] = []
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(ReelOptionNotFoundError):
                asyncio.run(controller.preview(job, caption_result))

    def test_exact_video_path_uploaded_once(self) -> None:
        _result, _page, dialog, _temp_dir = self._run_success()
        self.assertEqual(dialog.file_input.set_input_files_calls, [str(self.video_path)])

    def test_video_preview_must_be_verified(self) -> None:
        page, _dialog = _build_page(no_video_preview=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(VideoPreviewNotVerifiedError):
                asyncio.run(controller.preview(job, caption_result))

    def test_unsupported_media_raises(self) -> None:
        page, _dialog = _build_page(unsupported_media=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(VideoUploadError):
                asyncio.run(controller.preview(job, caption_result))

    def test_upload_failure_raises(self) -> None:
        page, _dialog = _build_page(fail_set_input_files=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(VideoUploadError):
                asyncio.run(controller.preview(job, caption_result))

    def test_processing_clears_after_several_polls(self) -> None:
        result, _page, _dialog, _temp_dir = self._run_success(processing_polls_before_clear=2)
        self.assertIsNotNone(result.processing_started)
        self.assertIsNotNone(result.processing_finished)
        self.assertIsNotNone(result.processing_duration_seconds)
        self.assertGreaterEqual(result.processing_duration_seconds, 0)

    def test_processing_timeout_fails_safely(self) -> None:
        page, dialog = _build_page(processing_never_clears=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(ReelProcessingTimeoutError):
                asyncio.run(controller.preview(job, caption_result))

        # Never advances past processing when it never clears.
        self.assertFalse(dialog.next_button_el.clicked)

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
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(AmbiguousNextButtonError):
                asyncio.run(controller.preview(job, caption_result))

        self.assertFalse(dialog.next_button_el.clicked)
        self.assertFalse(dialog.next_button_el_2.clicked)

    def test_next_click_maximum_is_enforced(self) -> None:
        # next_clicks_needed exceeds max_next_transitions (default 2):
        # the loop must stop after 2 clicks, never reaching "done".
        page, dialog = _build_page(next_clicks_needed=5)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            # The caption box only appears once next_clicks_done reaches
            # next_clicks_needed, which never happens when capped at 2 of
            # a required 5 — so this fails safely at CaptionBoxNotFoundError
            # rather than looping indefinitely or clicking a 3rd time.
            with self.assertRaises(CaptionBoxNotFoundError):
                asyncio.run(controller.preview(job, caption_result))

        self.assertEqual(dialog.next_clicks_done, 2)

    def test_cover_verified_when_present(self) -> None:
        result, _page, _dialog, _temp_dir = self._run_success()
        self.assertTrue(result.cover_verified)
        self.assertNotIn("cover_not_verified", result.warnings)

    def test_cover_missing_produces_warning_not_error(self) -> None:
        result, _page, _dialog, _temp_dir = self._run_success(no_cover=True)
        self.assertFalse(result.cover_verified)
        self.assertIn("cover_not_verified", result.warnings)
        self.assertIsNone(result.error)

    def test_caption_filled_once_and_line_breaks_preserved(self) -> None:
        result, _page, dialog, _temp_dir = self._run_success()
        self.assertEqual(len(dialog.caption_box.filled_text), result.final_caption_length)
        self.assertTrue(dialog.caption_box.filled_text.startswith("line one\nline two\n\n"))

    def test_caption_verification_mismatch_fails(self) -> None:
        page, _dialog = _build_page(caption_mismatch=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(CaptionVerificationMismatchError):
                asyncio.run(controller.preview(job, caption_result))

    def test_caption_and_hashtags_join_correctly(self) -> None:
        result = build_final_caption(
            "hello world", ["#aikotravel", "#travelcreator"], max_characters=None
        )
        self.assertEqual(result.final_caption, "hello world\n\n#aikotravel #travelcreator")
        self.assertTrue(result.hashtags_appended)

    def test_duplicate_hashtags_are_not_appended_twice(self) -> None:
        result = build_final_caption(
            "loving #aikotravel today", ["#aikotravel", "#travelcreator"], max_characters=None
        )
        self.assertNotIn("#aikotravel", result.hashtags_text)
        self.assertIn("#aikotravel", result.duplicate_hashtags_removed)

    def test_caption_limit_exceeded_stops_before_browser(self) -> None:
        with self.assertRaises(CaptionLimitExceededError):
            build_final_caption("x" * 50, ["#a"], max_characters=10)

    def test_share_button_found_but_never_clicked(self) -> None:
        result, _page, dialog, _temp_dir = self._run_success()
        self.assertTrue(result.share_button_found)
        self.assertFalse(dialog.share_button.clicked)
        self.assertEqual(dialog.share_button.click_count, 0)
        self.assertFalse(result.share_clicked)
        self.assertFalse(result.published)

    def test_share_button_ambiguous_fails_safe(self) -> None:
        page, dialog = _build_page(share_ambiguous=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(ShareButtonAmbiguousError):
                asyncio.run(controller.preview(job, caption_result))

        self.assertFalse(dialog.share_button.clicked)
        self.assertFalse(dialog.share_button_2.clicked)

    def test_share_button_missing_fails(self) -> None:
        page, _dialog = _build_page(share_missing=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(ShareButtonNotFoundError):
                asyncio.run(controller.preview(job, caption_result))

    def test_optional_features_all_reported_not_applied(self) -> None:
        result, _page, _dialog, _temp_dir = self._run_success()
        self.assertTrue(result.optional_features)
        for name, status in result.optional_features.items():
            self.assertEqual(status, "not_applied", f"{name} must be not_applied")
        for expected_name in (
            "location",
            "people_tagging",
            "collaborators",
            "topics",
            "music",
            "audio_controls",
            "paid_partnership",
            "boost",
            "accessibility_alt_text",
        ):
            self.assertIn(expected_name, result.optional_features)

    def test_four_required_screenshots_are_saved(self) -> None:
        result, _page, _dialog, _temp_dir = self._run_success()
        self.assertIsNotNone(result.screenshots["before_upload"])
        self.assertIsNotNone(result.screenshots["after_upload"])
        self.assertIsNotNone(result.screenshots["processing"])
        self.assertIsNotNone(result.screenshots["ready"])
        self.assertTrue(result.screenshots["before_upload"].endswith(f"before_upload_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["after_upload"].endswith(f"after_upload_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["processing"].endswith(f"processing_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["ready"].endswith(f"ready_{JOB_ID}.png"))

    def test_error_screenshot_saved_on_failure(self) -> None:
        page, _dialog = _build_page(share_missing=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            with self.assertRaises(ShareButtonNotFoundError):
                asyncio.run(controller.preview(job, caption_result))

        error_screenshots = [c for c in page.screenshot_calls if f"error_{JOB_ID}.png" in c]
        self.assertEqual(len(error_screenshots), 1)


# ---------------------------------------------------------------------------
# Queue / status / history safety
# ---------------------------------------------------------------------------


class QueueSafetyTests(MediaTempMixin, unittest.TestCase):
    def _write_real_queue(self, temp_dir: Path) -> Path:
        queue_path = temp_dir / "publish_queue.json"
        service = QueueService(queue_path)
        service.add_job(_job(status="approved", media_paths=[str(self.video_path)]))
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

            with self.assertRaises(ReelPreviewLoginError):
                asyncio.run(controller.preview(job, caption_result))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)

    def test_queue_json_byte_for_byte_unchanged_after_upload_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            page, _dialog = _build_page(fail_set_input_files=True)
            controller = _controller(page, temp_dir=temp_dir)
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(VideoUploadError):
                asyncio.run(controller.preview(job, caption_result))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)

    def test_queue_json_byte_for_byte_unchanged_after_processing_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            page, _dialog = _build_page(processing_never_clears=True)
            controller = _controller(page, temp_dir=temp_dir)
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(ReelProcessingTimeoutError):
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

    def test_queue_json_byte_for_byte_unchanged_after_caption_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            page, _dialog = _build_page(caption_mismatch=True)
            controller = _controller(page, temp_dir=temp_dir)
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)

            with self.assertRaises(CaptionVerificationMismatchError):
                asyncio.run(controller.preview(job, caption_result))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)

    def test_job_status_never_mutated_by_preview(self) -> None:
        page, _dialog = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(status="approved", media_paths=[str(self.video_path)])
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
            job = _job(status="approved", media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            asyncio.run(controller.preview(job, caption_result))

            history_dir = temp_dir / "history"
            self.assertFalse(history_dir.exists())

    def test_diagnostic_log_is_written_with_required_fields(self) -> None:
        page, _dialog = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            controller = _controller(page, temp_dir=temp_dir)
            job = _job(status="approved", media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            asyncio.run(controller.preview(job, caption_result))

            logs_dir = temp_dir / "logs"
            log_files = list(logs_dir.glob(f"instagram_reel_preview_{JOB_ID}_*.json"))
            self.assertEqual(len(log_files), 1)

            payload = json.loads(log_files[0].read_text(encoding="utf-8"))
            for field_name in (
                "started_at",
                "finished_at",
                "command",
                "job_id",
                "production_date",
                "persona_id",
                "media_path",
                "media_size",
                "login_status",
                "create_control_found",
                "reel_option_selected",
                "video_uploaded",
                "video_preview_verified",
                "processing_started",
                "processing_finished",
                "processing_duration_seconds",
                "next_transitions",
                "cover_verified",
                "caption_verified",
                "share_button_found",
                "share_button_enabled",
                "share_clicked",
                "published",
                "optional_features",
                "screenshots",
                "warnings",
                "result",
                "error",
            ):
                self.assertIn(field_name, payload)

            self.assertFalse(payload["share_clicked"])
            self.assertFalse(payload["published"])


# ---------------------------------------------------------------------------
# Structural safety scan
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def test_no_press_or_keyboard_calls_exist(self) -> None:
        self.assertNotIn(".press(", MODULE_SOURCE)
        self.assertNotIn("keyboard.press", MODULE_SOURCE)
        self.assertNotIn("keyboard.type", MODULE_SOURCE)

    def test_no_share_or_post_click_call_pattern_exists(self) -> None:
        # Any call shaped like "<name containing share/post>.click(" would
        # be an actual submission click. None should exist — those words
        # may only appear in selectors, field names, comments, and
        # error/log messages.
        for fragment in ("share", "post"):
            pattern = re.compile(rf"\b\w*{fragment}\w*\.click\(", re.IGNORECASE)
            self.assertEqual(
                pattern.findall(MODULE_SOURCE),
                [],
                f"found a click() call site on a '{fragment}'-named reference",
            )

    def test_no_evaluate_or_force_click_calls_exist(self) -> None:
        self.assertNotIn(".evaluate(", MODULE_SOURCE)
        self.assertNotIn("force=True", MODULE_SOURCE)

    def test_no_status_assignment_to_publishing_or_published(self) -> None:
        pattern = re.compile(r"""\.status\s*=\s*["'](publishing|published)["']""")
        self.assertIsNone(pattern.search(MODULE_SOURCE))

    def test_history_service_is_never_imported(self) -> None:
        # Checked line-by-line against actual import statements rather
        # than a blanket substring search: this module's own explanatory
        # comments legitimately mention "history_service.py" by name when
        # describing what it does NOT depend on.
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import history_service"))
            self.assertFalse(stripped.startswith("from .history_service"))
            self.assertFalse(stripped.startswith("from src.publishing.history_service"))
        self.assertNotIn("HistoryService(", MODULE_SOURCE)
        self.assertNotIn("build_history_service(", MODULE_SOURCE)

    def test_writable_queue_methods_are_never_called(self) -> None:
        for forbidden in (".save_jobs(", ".update_job(", ".add_job(", ".upsert_jobs("):
            self.assertNotIn(forbidden, MODULE_SOURCE)
        self.assertIn("load_job_by_id", MODULE_SOURCE)

    def test_no_publish_send_or_share_cli_arguments(self) -> None:
        self.assertNotIn('"--publish"', MODULE_SOURCE)
        self.assertNotIn('"--send"', MODULE_SOURCE)
        self.assertNotIn('"--share"', MODULE_SOURCE)

    def test_no_publish_or_send_method_on_controller(self) -> None:
        self.assertFalse(hasattr(InstagramReelPreviewController, "publish"))
        self.assertFalse(hasattr(InstagramReelPreviewController, "send"))

    def test_disallowed_action_methods_not_implemented_on_controller(self) -> None:
        for action in DISALLOWED_ACTIONS:
            self.assertFalse(
                hasattr(InstagramReelPreviewController, action),
                f"InstagramReelPreviewController must not implement '{action}'",
            )

    def test_fake_element_and_controller_have_no_press_capability(self) -> None:
        for forbidden_name in ("press", "keyboard", "submit"):
            self.assertFalse(hasattr(FakeElement, forbidden_name), "test fixture sanity check")
            self.assertFalse(
                hasattr(InstagramReelPreviewController, forbidden_name),
                f"InstagramReelPreviewController must not implement '{forbidden_name}'",
            )

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("share", DISALLOWED_ACTIONS)
        self.assertIn("publish", DISALLOWED_ACTIONS)
        self.assertIn("press_enter", DISALLOWED_ACTIONS)
        self.assertIn("click_share", DISALLOWED_ACTIONS)
        self.assertIn("click_post", DISALLOWED_ACTIONS)


# ---------------------------------------------------------------------------
# Feed / Story / Recovery files untouched
# ---------------------------------------------------------------------------


class FeedStoryRecoveryFilesUntouchedTests(MediaTempMixin, unittest.TestCase):
    def test_files_are_byte_for_byte_unchanged_after_a_full_preview_run(self) -> None:
        watched_files = [
            app_root() / "src" / "publishing" / "instagram_feed_preview.py",
            app_root() / "src" / "publishing" / "instagram_feed_sender.py",
            app_root() / "src" / "publishing" / "instagram_story_preview.py",
            app_root() / "src" / "publishing" / "publish_recovery.py",
            app_root() / "src" / "publishing" / "history_service.py",
        ]

        def _snapshot() -> dict[str, str]:
            return {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in watched_files
                if path.is_file()
            }

        before = _snapshot()

        page, _dialog = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(status="approved", media_paths=[str(self.video_path)])
            caption_result = build_final_caption(job.caption_text, job.hashtags, max_characters=None)
            asyncio.run(controller.preview(job, caption_result))

        after = _snapshot()
        self.assertEqual(before, after)
        self.assertTrue(before, "sanity check: at least one watched file must exist")


if __name__ == "__main__":
    unittest.main()
