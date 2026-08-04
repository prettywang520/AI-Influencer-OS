from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from src.social.instagram_models import LoginDetectionConfig

from .instagram_story_preview import (
    DISALLOWED_ACTIONS,
    CreateControlNotFoundError,
    InstagramStoryPreviewController,
    MediaPreviewNotVerifiedError,
    MediaUploadError,
    StoryPreviewIneligibleError,
    StoryPreviewLoginError,
    StorySetValidationError,
    check_story_job_eligible,
    load_story_preview_config,
    load_story_siblings,
    validate_story_set,
)
from .models import PublisherConfig, PublishJob, app_root
from .queue_service import QueueService, load_job_by_id

# None of these tests open a real browser or contact Instagram.
# Playwright is simulated with small fake objects only.

MODULE_PATH = Path(__file__).resolve().parent / "instagram_story_preview.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")

JOB_ID = "2026-08-01-instagram_story-1"
PRODUCTION_DATE = "2026-08-01"
BASE_URL = "https://www.instagram.com/"

CREATE_SELECTOR = "div[aria-label='Add to your story' i]"
FILE_INPUT_SELECTOR = "input[type='file']"
CANVAS_SELECTOR = "canvas"
YOUR_STORY_SELECTOR = 'role=button[name="Your story"]'
SHARE_SELECTOR = 'role=button[name="Share"]'
SHARE_TO_STORY_SELECTOR = 'role=button[name="Share to story"]'
CLOSE_FRIENDS_SELECTOR = 'role=button[name="Close Friends"]'
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
        fail_is_visible: bool = False,
        fail_is_enabled: bool = False,
        fail_set_input_files: bool = False,
    ) -> None:
        self._text = text
        self._visible = visible
        self._disabled = disabled
        self._fail_click = fail_click
        self._fail_is_visible = fail_is_visible
        self._fail_is_enabled = fail_is_enabled
        self._fail_set_input_files = fail_set_input_files
        self.clicked = False
        self.click_count = 0
        self.set_input_files_calls: list[str] = []

    async def query_selector(self, selector: str):
        return None

    async def query_selector_all(self, selector: str):
        return []

    async def inner_text(self) -> str:
        return self._text

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

    async def set_input_files(self, path: str):
        if self._fail_set_input_files:
            raise RuntimeError("simulated upload failure")
        self.set_input_files_calls.append(path)

    # Explicitly absent on purpose: press(), keyboard, submit(), fill(),
    # type(). If production code ever tried to call these, the test would
    # fail with AttributeError rather than silently succeeding.


class FakeStoryPage:
    """
    Hand-rolled fake for the entire Story flow. Unlike the Feed preview's
    FakePage+StagedDialog split, instagram_story_preview.py queries `page`
    directly for every element (there is no separate composer-root
    variable), so one stateful object plays every role: navigation,
    login detection, Create control, media upload, canvas verification,
    and all four named share candidates.
    """

    def __init__(
        self,
        *,
        url: str = BASE_URL,
        logged_in: bool = True,
        login_unknown: bool = False,
        goto_error: Exception | None = None,
        create_control_missing: bool = False,
        create_control_ambiguous: bool = False,
        file_input_missing: bool = False,
        fail_set_input_files: bool = False,
        unsupported_media: bool = False,
        no_media_preview: bool = False,
        your_story_count: int = 1,
        share_count: int = 1,
        share_to_story_count: int = 1,
        close_friends_count: int = 1,
    ) -> None:
        self.url = url
        self._goto_error = goto_error
        self.uploaded = False
        self.unsupported_media = unsupported_media
        self.no_media_preview = no_media_preview
        self.file_input_missing = file_input_missing
        self.screenshot_calls: list[str] = []

        self.login_el = FakeElement() if logged_in else None
        self.logout_el = FakeElement() if (not logged_in and not login_unknown) else None

        self.create_control_missing = create_control_missing
        self.create_control_ambiguous = create_control_ambiguous
        self.create_control_el = FakeElement()
        self.create_control_el_2 = FakeElement()

        self.file_input = FakeElement(fail_set_input_files=fail_set_input_files)
        self.media_canvas_el = FakeElement()
        self.unsupported_el = FakeElement(text="couldn't be uploaded")

        self.your_story_els = [FakeElement() for _ in range(your_story_count)]
        self.share_els = [FakeElement() for _ in range(share_count)]
        self.share_to_story_els = [FakeElement() for _ in range(share_to_story_count)]
        self.close_friends_els = [FakeElement() for _ in range(close_friends_count)]

        real_upload = self.file_input.set_input_files

        async def _upload(path: str):
            await real_upload(path)
            self.uploaded = True

        self.file_input.set_input_files = _upload

    async def goto(self, url, timeout=None, wait_until=None):
        if self._goto_error is not None:
            raise self._goto_error

    async def wait_for_timeout(self, ms):
        pass

    async def screenshot(self, path, full_page=True):
        self.screenshot_calls.append(path)

    async def query_selector(self, selector: str):
        results = await self.query_selector_all(selector)
        return results[0] if results else None

    async def query_selector_all(self, selector: str):
        if selector == LOGIN_SELECTOR:
            return [self.login_el] if self.login_el else []

        if selector == LOGOUT_SELECTOR:
            return [self.logout_el] if self.logout_el else []

        if selector == CREATE_SELECTOR:
            if self.create_control_missing:
                return []
            if self.create_control_ambiguous:
                return [self.create_control_el, self.create_control_el_2]
            return [self.create_control_el]

        if selector == FILE_INPUT_SELECTOR:
            return [] if self.file_input_missing else [self.file_input]

        if selector == CANVAS_SELECTOR:
            if self.uploaded and not self.no_media_preview:
                return [self.media_canvas_el]
            return []

        if selector in (
            "text=file format",
            "div:has-text('not supported')",
            "div:has-text(\"couldn't be uploaded\")",
        ):
            if self.uploaded and self.unsupported_media:
                return [self.unsupported_el]
            return []

        if selector == YOUR_STORY_SELECTOR:
            return list(self.your_story_els)

        if selector == SHARE_SELECTOR:
            return list(self.share_els)

        if selector == SHARE_TO_STORY_SELECTOR:
            return list(self.share_to_story_els)

        if selector == CLOSE_FRIENDS_SELECTOR:
            return list(self.close_friends_els)

        return []


class FakeContext:
    def __init__(self, page: FakeStoryPage) -> None:
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

    def __init__(self, page: FakeStoryPage) -> None:
        self.config = FakeSessionConfig()
        self._playwright = FakePlaywright()
        self._context = FakeContext(page)

    async def _open_context(self):
        return self._playwright, self._context


def _build_page(**kwargs) -> FakeStoryPage:
    return FakeStoryPage(**kwargs)


def _default_story_config():
    return load_story_preview_config()


def _controller(page: FakeStoryPage, *, temp_dir: Path) -> InstagramStoryPreviewController:
    return InstagramStoryPreviewController(
        session=FakeSession(page),
        story_config=_default_story_config(),
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
        content_type="instagram_story",
        status="approved",
        media_paths=[],
        metadata={"story_order": 1},
    )
    defaults.update(overrides)
    return PublishJob(**defaults)


class MediaTempMixin:
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.image_path = self.temp_dir / "story_01.png"
        self.image_path.write_bytes(b"fake-image-bytes")
        self.video_path = self.temp_dir / "story_01.mp4"
        self.video_path.write_bytes(b"fake-video-bytes")


# ---------------------------------------------------------------------------
# Eligibility (pure, no browser)
# ---------------------------------------------------------------------------


class EligibilityTests(MediaTempMixin, unittest.TestCase):
    def test_approved_instagram_story_job_is_eligible(self) -> None:
        job = _job(status="approved", media_paths=[str(self.image_path)])
        story_order = check_story_job_eligible(job)
        self.assertEqual(story_order, 1)

    def test_instagram_feed_content_type_is_rejected(self) -> None:
        job = _job(
            status="approved",
            content_type="instagram_feed",
            media_paths=[str(self.image_path)],
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_instagram_reel_content_type_is_rejected(self) -> None:
        job = _job(
            status="approved",
            content_type="instagram_reel",
            media_paths=[str(self.image_path)],
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_threads_post_content_type_is_rejected(self) -> None:
        job = _job(
            status="approved",
            content_type="threads_post",
            media_paths=[str(self.image_path)],
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_pending_approval_is_rejected(self) -> None:
        job = _job(status="pending_approval", media_paths=[str(self.image_path)])
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_rejected_is_rejected(self) -> None:
        job = _job(status="rejected", media_paths=[str(self.image_path)])
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_scheduled_is_rejected(self) -> None:
        job = _job(status="scheduled", media_paths=[str(self.image_path)])
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_publishing_is_rejected(self) -> None:
        job = _job(status="publishing", media_paths=[str(self.image_path)])
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_published_is_rejected(self) -> None:
        job = _job(status="published", media_paths=[str(self.image_path)])
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_missing_media_file_fails_before_browser(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.temp_dir / "does_not_exist.png")],
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_zero_byte_media_fails_before_browser(self) -> None:
        zero_byte_path = self.temp_dir / "empty.png"
        zero_byte_path.write_bytes(b"")
        job = _job(status="approved", media_paths=[str(zero_byte_path)])
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_wrong_media_count_fails(self) -> None:
        job = _job(status="approved", media_paths=[])
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_story_order_required(self) -> None:
        job = _job(status="approved", media_paths=[str(self.image_path)], metadata={})
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_story_order_must_be_int(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.image_path)],
            metadata={"story_order": "1"},
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_story_order_bool_is_rejected(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.image_path)],
            metadata={"story_order": True},
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_story_order_zero_is_rejected(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.image_path)],
            metadata={"story_order": 0},
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_story_order_five_is_rejected(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.image_path)],
            metadata={"story_order": 5},
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_story_order_four_is_eligible(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.image_path)],
            metadata={"story_order": 4},
        )
        self.assertEqual(check_story_job_eligible(job), 4)

    def test_already_published_at_is_rejected(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.image_path)],
            published_at="2026-08-01T00:00:00+00:00",
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)

    def test_already_has_platform_post_id_is_rejected(self) -> None:
        job = _job(
            status="approved",
            media_paths=[str(self.image_path)],
            platform_post_id="abc123",
        )
        with self.assertRaises(StoryPreviewIneligibleError):
            check_story_job_eligible(job)


# ---------------------------------------------------------------------------
# Story-set validation (pure, no browser)
# ---------------------------------------------------------------------------


class StorySetValidationTests(MediaTempMixin, unittest.TestCase):
    def test_unique_orders_and_media_paths_pass(self) -> None:
        target = _job(job_id="2026-08-01-instagram_story-1", media_paths=[str(self.image_path)])
        sibling = _job(
            job_id="2026-08-01-instagram_story-2",
            media_paths=[str(self.video_path)],
            metadata={"story_order": 2},
        )
        validate_story_set(target, [target, sibling])  # does not raise

    def test_duplicate_story_order_fails(self) -> None:
        target = _job(job_id="2026-08-01-instagram_story-1", media_paths=[str(self.image_path)])
        duplicate = _job(
            job_id="2026-08-01-instagram_story-2",
            media_paths=[str(self.video_path)],
            metadata={"story_order": 1},
        )
        with self.assertRaises(StorySetValidationError):
            validate_story_set(target, [target, duplicate])

    def test_duplicate_media_path_fails(self) -> None:
        target = _job(job_id="2026-08-01-instagram_story-1", media_paths=[str(self.image_path)])
        duplicate = _job(
            job_id="2026-08-01-instagram_story-2",
            media_paths=[str(self.image_path)],
            metadata={"story_order": 2},
        )
        with self.assertRaises(StorySetValidationError):
            validate_story_set(target, [target, duplicate])

    def test_job_not_present_in_its_own_sibling_set_fails(self) -> None:
        target = _job(job_id="2026-08-01-instagram_story-1", media_paths=[str(self.image_path)])
        other = _job(
            job_id="2026-08-01-instagram_story-2",
            media_paths=[str(self.video_path)],
            metadata={"story_order": 2},
        )
        with self.assertRaises(StorySetValidationError):
            validate_story_set(target, [other])

    def test_media_path_mismatch_against_queue_record_fails(self) -> None:
        target = _job(job_id="2026-08-01-instagram_story-1", media_paths=[str(self.image_path)])
        stale_record = _job(job_id="2026-08-01-instagram_story-1", media_paths=[str(self.video_path)])
        with self.assertRaises(StorySetValidationError):
            validate_story_set(target, [stale_record])

    def test_load_story_siblings_filters_by_content_type_and_date(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            queue_path = Path(temp_name) / "publish_queue.json"
            service = QueueService(queue_path)
            service.add_job(_job(job_id="2026-08-01-instagram_story-1", media_paths=[str(self.image_path)]))
            service.add_job(
                _job(
                    job_id="2026-08-01-instagram_story-2",
                    media_paths=[str(self.video_path)],
                    metadata={"story_order": 2},
                )
            )
            service.add_job(
                _job(
                    job_id="2026-08-01-instagram_feed",
                    content_type="instagram_feed",
                    media_paths=[str(self.image_path)],
                    metadata={},
                )
            )
            service.add_job(
                _job(
                    job_id="2026-08-02-instagram_story-1",
                    production_date="2026-08-02",
                    media_paths=[str(self.image_path)],
                )
            )

            siblings = load_story_siblings(queue_path, "2026-08-01")
            sibling_ids = {job.job_id for job in siblings}
            self.assertEqual(
                sibling_ids, {"2026-08-01-instagram_story-1", "2026-08-01-instagram_story-2"}
            )


# ---------------------------------------------------------------------------
# Login safety
# ---------------------------------------------------------------------------


class LoginFailureTests(MediaTempMixin, unittest.TestCase):
    def test_logged_out_stops_before_create_control(self) -> None:
        page = _build_page(logged_in=False)

        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])

            with self.assertRaises(StoryPreviewLoginError):
                asyncio.run(controller.preview(job, 1))

        self.assertFalse(page.create_control_el.clicked)
        self.assertTrue(page.screenshot_calls)

    def test_login_uncertain_stops_before_create_control(self) -> None:
        page = _build_page(logged_in=False, login_unknown=True)

        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])

            with self.assertRaises(StoryPreviewLoginError):
                asyncio.run(controller.preview(job, 1))

        self.assertFalse(page.create_control_el.clicked)


# ---------------------------------------------------------------------------
# Full success path
# ---------------------------------------------------------------------------


class FullPreviewSuccessTests(MediaTempMixin, unittest.TestCase):
    def _run_success(self, *, media_path: Path | None = None, **page_kwargs):
        page = _build_page(**page_kwargs)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_ctx.cleanup)
        temp_dir = Path(temp_dir_ctx.name)
        controller = _controller(page, temp_dir=temp_dir)

        job = _job(media_paths=[str(media_path or self.image_path)])
        result = asyncio.run(controller.preview(job, 1))
        return result, page, temp_dir

    def test_full_preview_reaches_ready_state(self) -> None:
        result, _page, _temp_dir = self._run_success()
        self.assertEqual(result.login_status, "logged_in")
        self.assertTrue(result.media_uploaded)
        self.assertTrue(result.media_preview_verified)
        self.assertTrue(result.share_button_found)
        self.assertTrue(result.share_button_enabled)
        self.assertFalse(result.share_clicked)
        self.assertFalse(result.published)
        self.assertIsNone(result.error)

    def test_create_control_is_clicked_exactly_once(self) -> None:
        _result, page, _temp_dir = self._run_success()
        self.assertEqual(page.create_control_el.click_count, 1)

    def test_story_option_selected_with_no_post_or_reel_selector(self) -> None:
        result, _page, _temp_dir = self._run_success()
        self.assertTrue(result.error is None)

        config = _default_story_config()
        self.assertFalse(hasattr(config.selectors, "post_option"))
        self.assertFalse(hasattr(config.selectors, "reel_option"))

    def test_create_control_missing_fails_safely(self) -> None:
        page = _build_page(create_control_missing=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])
            with self.assertRaises(CreateControlNotFoundError):
                asyncio.run(controller.preview(job, 1))

    def test_create_control_ambiguous_fails_safely(self) -> None:
        page = _build_page(create_control_ambiguous=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])
            with self.assertRaises(CreateControlNotFoundError):
                asyncio.run(controller.preview(job, 1))

        self.assertFalse(page.create_control_el.clicked)
        self.assertFalse(page.create_control_el_2.clicked)

    def test_exact_media_path_uploaded_once(self) -> None:
        _result, page, _temp_dir = self._run_success()
        self.assertEqual(page.file_input.set_input_files_calls, [str(self.image_path)])

    def test_image_preview_verification_succeeds(self) -> None:
        result, _page, _temp_dir = self._run_success(media_path=self.image_path)
        self.assertEqual(result.media_type, "image")
        self.assertTrue(result.media_preview_verified)

    def test_video_preview_verification_succeeds(self) -> None:
        result, _page, _temp_dir = self._run_success(media_path=self.video_path)
        self.assertEqual(result.media_type, "video")
        self.assertTrue(result.media_preview_verified)

    def test_missing_media_preview_fails_safely(self) -> None:
        page = _build_page(no_media_preview=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])
            with self.assertRaises(MediaPreviewNotVerifiedError):
                asyncio.run(controller.preview(job, 1))

    def test_unsupported_media_raises(self) -> None:
        page = _build_page(unsupported_media=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])
            with self.assertRaises(MediaUploadError):
                asyncio.run(controller.preview(job, 1))

    def test_upload_failure_raises(self) -> None:
        page = _build_page(fail_set_input_files=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])
            with self.assertRaises(MediaUploadError):
                asyncio.run(controller.preview(job, 1))

    def test_file_input_missing_raises(self) -> None:
        page = _build_page(file_input_missing=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])
            with self.assertRaises(MediaUploadError):
                asyncio.run(controller.preview(job, 1))

    def test_your_story_found_but_never_clicked(self) -> None:
        _result, page, _temp_dir = self._run_success()
        for element in page.your_story_els:
            self.assertFalse(element.clicked)
            self.assertEqual(element.click_count, 0)

    def test_share_found_but_never_clicked(self) -> None:
        _result, page, _temp_dir = self._run_success()
        for element in page.share_els:
            self.assertFalse(element.clicked)
            self.assertEqual(element.click_count, 0)

    def test_share_to_story_found_but_never_clicked(self) -> None:
        _result, page, _temp_dir = self._run_success()
        for element in page.share_to_story_els:
            self.assertFalse(element.clicked)
            self.assertEqual(element.click_count, 0)

    def test_close_friends_found_but_never_clicked(self) -> None:
        _result, page, _temp_dir = self._run_success()
        for element in page.close_friends_els:
            self.assertFalse(element.clicked)
            self.assertEqual(element.click_count, 0)

    def test_share_candidates_counts_are_reported(self) -> None:
        result, _page, _temp_dir = self._run_success(
            your_story_count=1, share_count=2, share_to_story_count=0, close_friends_count=1
        )
        self.assertEqual(result.share_candidates["your_story"], 1)
        self.assertEqual(result.share_candidates["share"], 2)
        self.assertEqual(result.share_candidates["share_to_story"], 0)
        self.assertEqual(result.share_candidates["close_friends"], 1)
        self.assertTrue(result.share_button_found)

    def test_all_share_candidates_absent_does_not_fail(self) -> None:
        result, _page, _temp_dir = self._run_success(
            your_story_count=0, share_count=0, share_to_story_count=0, close_friends_count=0
        )
        self.assertIsNone(result.error)
        self.assertFalse(result.share_button_found)
        self.assertFalse(result.share_button_enabled)
        self.assertFalse(result.share_clicked)
        self.assertFalse(result.published)

    def test_exactly_three_required_screenshots_are_saved(self) -> None:
        result, _page, _temp_dir = self._run_success()
        self.assertIsNotNone(result.screenshots["before_upload"])
        self.assertIsNotNone(result.screenshots["after_upload"])
        self.assertIsNotNone(result.screenshots["ready"])
        self.assertTrue(result.screenshots["before_upload"].endswith(f"before_upload_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["after_upload"].endswith(f"after_upload_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["ready"].endswith(f"ready_{JOB_ID}.png"))

    def test_error_screenshot_saved_on_failure(self) -> None:
        page = _build_page(no_media_preview=True)
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(media_paths=[str(self.image_path)])
            with self.assertRaises(MediaPreviewNotVerifiedError):
                asyncio.run(controller.preview(job, 1))

        error_screenshots = [c for c in page.screenshot_calls if f"error_{JOB_ID}.png" in c]
        self.assertEqual(len(error_screenshots), 1)

    def test_optional_features_all_reported_not_applied(self) -> None:
        result, _page, _temp_dir = self._run_success()
        self.assertTrue(result.optional_features)
        for name, status in result.optional_features.items():
            self.assertEqual(status, "not_applied", f"{name} must be not_applied")
        for expected_name in (
            "text_overlay",
            "mention",
            "location",
            "music",
            "link",
            "poll",
            "questions",
            "gif",
            "stickers",
        ):
            self.assertIn(expected_name, result.optional_features)


# ---------------------------------------------------------------------------
# Queue / status / history safety
# ---------------------------------------------------------------------------


class QueueSafetyTests(MediaTempMixin, unittest.TestCase):
    def _write_real_queue(self, temp_dir: Path) -> Path:
        queue_path = temp_dir / "publish_queue.json"
        service = QueueService(queue_path)
        service.add_job(_job(status="approved", media_paths=[str(self.image_path)]))
        return queue_path

    def test_queue_json_byte_for_byte_unchanged_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            self.assertIsNotNone(job)

            page = _build_page()
            controller = _controller(page, temp_dir=temp_dir)
            asyncio.run(controller.preview(job, 1))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)
            self.assertEqual(hashlib.md5(before).hexdigest(), hashlib.md5(after).hexdigest())

    def test_queue_json_byte_for_byte_unchanged_after_login_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            page = _build_page(logged_in=False)
            controller = _controller(page, temp_dir=temp_dir)

            with self.assertRaises(StoryPreviewLoginError):
                asyncio.run(controller.preview(job, 1))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)

    def test_queue_json_byte_for_byte_unchanged_after_upload_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            page = _build_page(fail_set_input_files=True)
            controller = _controller(page, temp_dir=temp_dir)

            with self.assertRaises(MediaUploadError):
                asyncio.run(controller.preview(job, 1))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)

    def test_queue_json_byte_for_byte_unchanged_after_media_verification_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            queue_path = self._write_real_queue(temp_dir)
            before = queue_path.read_bytes()

            job = load_job_by_id(queue_path, JOB_ID)
            page = _build_page(no_media_preview=True)
            controller = _controller(page, temp_dir=temp_dir)

            with self.assertRaises(MediaPreviewNotVerifiedError):
                asyncio.run(controller.preview(job, 1))

            after = queue_path.read_bytes()
            self.assertEqual(before, after)

    def test_job_status_never_mutated_by_preview(self) -> None:
        page = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(status="approved", media_paths=[str(self.image_path)])
            asyncio.run(controller.preview(job, 1))
            self.assertEqual(job.status, "approved")
            self.assertIsNone(job.published_at)
            self.assertIsNone(job.platform_post_id)

    def test_no_history_file_is_written(self) -> None:
        page = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            controller = _controller(page, temp_dir=temp_dir)
            job = _job(status="approved", media_paths=[str(self.image_path)])
            asyncio.run(controller.preview(job, 1))

            history_dir = temp_dir / "history"
            self.assertFalse(history_dir.exists())

    def test_diagnostic_log_is_written_with_required_fields(self) -> None:
        page = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            controller = _controller(page, temp_dir=temp_dir)
            job = _job(status="approved", media_paths=[str(self.image_path)])
            asyncio.run(controller.preview(job, 1))

            logs_dir = temp_dir / "logs"
            log_files = list(logs_dir.glob(f"instagram_story_preview_{JOB_ID}_*.json"))
            self.assertEqual(len(log_files), 1)

            payload = json.loads(log_files[0].read_text(encoding="utf-8"))
            for field_name in (
                "started_at",
                "finished_at",
                "command",
                "job_id",
                "production_date",
                "persona_id",
                "story_order",
                "media_path",
                "media_type",
                "login_status",
                "create_control_found",
                "story_option_selected",
                "media_uploaded",
                "media_preview_verified",
                "share_candidates",
                "share_clicked",
                "published",
                "optional_features",
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


class StructuralSafetyTests(unittest.TestCase):
    def test_no_press_or_keyboard_calls_exist(self) -> None:
        self.assertNotIn(".press(", MODULE_SOURCE)
        self.assertNotIn("keyboard.press", MODULE_SOURCE)
        self.assertNotIn("keyboard.type", MODULE_SOURCE)
        self.assertNotIn('"Enter"', MODULE_SOURCE)
        self.assertNotIn("'Enter'", MODULE_SOURCE)

    def test_no_submission_click_call_pattern_exists(self) -> None:
        # Any call shaped like "<name containing share/your_story/
        # close_friends>.click(" would be an actual submission click.
        # None should exist — those words may only appear in selectors,
        # dict keys, comments, and error/log messages. share_candidates
        # is deliberately never stored as clickable element handles at
        # all — only visible+enabled counts are kept.
        for fragment in ("share", "your_story", "close_friends", "share_to_story"):
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
        # describing what it does NOT depend on, which would otherwise
        # false-fail a naive assertNotIn("history_service", ...) check.
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
        self.assertIn(".load_jobs()", MODULE_SOURCE)

    def test_no_publish_send_or_share_cli_arguments(self) -> None:
        self.assertNotIn('"--publish"', MODULE_SOURCE)
        self.assertNotIn('"--send"', MODULE_SOURCE)
        self.assertNotIn('"--share"', MODULE_SOURCE)

    def test_no_publish_or_send_method_on_controller(self) -> None:
        self.assertFalse(hasattr(InstagramStoryPreviewController, "publish"))
        self.assertFalse(hasattr(InstagramStoryPreviewController, "send"))

    def test_disallowed_action_methods_not_implemented_on_controller(self) -> None:
        for action in DISALLOWED_ACTIONS:
            self.assertFalse(
                hasattr(InstagramStoryPreviewController, action),
                f"InstagramStoryPreviewController must not implement '{action}'",
            )

    def test_fake_element_and_controller_have_no_press_capability(self) -> None:
        for forbidden_name in ("press", "keyboard", "submit"):
            self.assertFalse(hasattr(FakeElement, forbidden_name), "test fixture sanity check")
            self.assertFalse(
                hasattr(InstagramStoryPreviewController, forbidden_name),
                f"InstagramStoryPreviewController must not implement '{forbidden_name}'",
            )

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("share", DISALLOWED_ACTIONS)
        self.assertIn("publish", DISALLOWED_ACTIONS)
        self.assertIn("press_enter", DISALLOWED_ACTIONS)
        self.assertIn("click_your_story", DISALLOWED_ACTIONS)
        self.assertIn("click_close_friends", DISALLOWED_ACTIONS)
        self.assertIn("post_feed", DISALLOWED_ACTIONS)
        self.assertIn("post_reel", DISALLOWED_ACTIONS)


# ---------------------------------------------------------------------------
# Feed/recovery files untouched
# ---------------------------------------------------------------------------


class FeedFilesUntouchedTests(MediaTempMixin, unittest.TestCase):
    def test_feed_and_recovery_files_are_byte_for_byte_unchanged_after_a_full_preview_run(self) -> None:
        watched_files = [
            app_root() / "src" / "publishing" / "instagram_feed_preview.py",
            app_root() / "src" / "publishing" / "instagram_feed_sender.py",
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

        page = _build_page()
        with tempfile.TemporaryDirectory() as temp_name:
            controller = _controller(page, temp_dir=Path(temp_name))
            job = _job(status="approved", media_paths=[str(self.image_path)])
            asyncio.run(controller.preview(job, 1))

        after = _snapshot()
        self.assertEqual(before, after)
        self.assertTrue(before, "sanity check: at least one watched file must exist")


if __name__ == "__main__":
    unittest.main()
