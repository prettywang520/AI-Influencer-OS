from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from src.social.instagram_models import LoginDetectionConfig

from .history_service import HistoryService
from .instagram_feed_preview import (
    FeedPreviewIneligibleError,
    build_final_caption,
    load_feed_preview_config,
)
from .instagram_feed_sender import (
    DISALLOWED_ACTIONS,
    FeedPublishResult,
    FeedSenderIneligibleError,
    InstagramFeedSenderController,
    PublishLock,
    PublishLockError,
    PublishVerificationInconclusiveError,
    ShareClickFailedError,
    check_job_publishable,
    load_feed_sender_config,
    parse_arguments,
)
from .models import PublisherConfig, PublishJob, app_root
from .queue_service import QueueService

MODULE_PATH = Path(__file__).resolve().parent / "instagram_feed_sender.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")
PREVIEW_MODULE_SOURCE = (
    Path(__file__).resolve().parent / "instagram_feed_preview.py"
).read_text(encoding="utf-8")

JOB_ID = "2026-08-01-instagram_feed"
BASE_URL = "https://www.instagram.com/"

CREATE_SELECTOR = 'role=link[name="New post"]'
POST_OPTION_SELECTOR = 'role=button[name="Post"]'
DIALOG_SELECTOR = "div[role='dialog']"
LOGIN_SELECTOR = "svg[aria-label='Home']"
LOGOUT_SELECTOR = "input[type='password']"

SUCCESS_SELECTORS = ("text=Your post has been shared", "text=Post shared")
ERROR_SELECTORS = (
    "div:has-text('error')",
    "text=Something went wrong",
    "text=Please try again",
)


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
        fill_writes_value: str | None = None,
    ) -> None:
        self._text = text
        self._value = value
        self._visible = visible
        self._disabled = disabled
        self._fail_click = fail_click
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
        self.typed_text = (self.typed_text or "") + text
        self._value = (self._value or "") + text

    async def set_input_files(self, path: str):
        self.set_input_files_calls.append(path)

    # Explicitly absent on purpose: press(), keyboard, submit(),
    # evaluate(). If production code ever tried to call these, the
    # test would fail with AttributeError rather than silently passing.


class StagedDialog:
    """
    Hand-rolled fake for the Create Post dialog, extended (vs
    test_instagram_feed_preview.py's version) with post-Share-click
    behavior: the dialog can "close", a success message can appear,
    and page.url can change to a post permalink — exactly the signals
    InstagramFeedSenderController._verify_publish() looks for.
    """

    def __init__(
        self,
        *,
        next_clicks_needed: int = 2,
        share_ambiguous: bool = False,
        share_disabled: bool = False,
        share_missing: bool = False,
        fail_share_click: bool = False,
        caption_mismatch: bool = False,
        close_dialog_on_share: bool = True,
        show_success_message: bool = True,
        change_url_on_share: bool = True,
        show_error_message: bool = False,
    ) -> None:
        self.uploaded = False
        self.next_clicks_done = 0
        self.next_clicks_needed = next_clicks_needed
        self.share_ambiguous = share_ambiguous
        self.share_disabled = share_disabled
        self.share_missing = share_missing
        self.close_dialog_on_share = close_dialog_on_share
        self.show_success_message = show_success_message
        self.change_url_on_share = change_url_on_share
        self.show_error_message = show_error_message
        self.closed = False

        self.file_input = FakeElement()
        self.media_preview_el = FakeElement()
        self.next_button_el = FakeElement()
        self.caption_box = FakeElement(
            fill_writes_value="WRONG TEXT" if caption_mismatch else None
        )
        self.share_button = FakeElement(fail_click=fail_share_click)
        self.share_button_2 = FakeElement()
        self.share_button_disabled = FakeElement(disabled=True)
        self.success_message_el = FakeElement(text="Your post has been shared")
        self.error_message_el = FakeElement(text="Something went wrong")

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

        real_share_click = self.share_button.click

        async def _share_click(timeout=None):
            await real_share_click(timeout=timeout)  # raises first if fail_share_click
            if self.close_dialog_on_share:
                self.closed = True
            if self.change_url_on_share:
                self._page_url_override = "https://www.instagram.com/p/ABC123xyz/"

        self.share_button.click = _share_click
        self._page_url_override: str | None = None

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
            "div[style*='background-image']",
        ):
            return [self.media_preview_el] if self.uploaded else []

        if selector in ('role=button[name="Next"]', 'role=button[name="Next" i]'):
            if not self.uploaded or self.next_clicks_done >= self.next_clicks_needed:
                return []
            return [self.next_button_el]

        if selector in (
            "div[contenteditable='true']",
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

    def page_wide(self, selector: str):
        if selector in SUCCESS_SELECTORS and self.share_button.clicked and self.show_success_message:
            return self.success_message_el
        if selector in ERROR_SELECTORS and self.share_button.clicked and self.show_error_message:
            return self.error_message_el
        return None


class FakePage:
    def __init__(
        self,
        *,
        children: dict[str, object] | None = None,
        url: str = BASE_URL,
        goto_error: Exception | None = None,
        dialog: StagedDialog | None = None,
    ) -> None:
        self._children = children or {}
        self._url = url
        self._goto_error = goto_error
        self.screenshot_calls: list[str] = []
        self.dialog = dialog

    @property
    def url(self) -> str:
        if self.dialog is not None and self.dialog._page_url_override:
            return self.dialog._page_url_override
        return self._url

    @url.setter
    def url(self, value: str) -> None:
        self._url = value

    async def goto(self, url, timeout=None, wait_until=None):
        if self._goto_error is not None:
            raise self._goto_error

    async def wait_for_timeout(self, ms):
        pass

    async def query_selector(self, selector: str):
        if selector == DIALOG_SELECTOR and self.dialog is not None and self.dialog.closed:
            return None
        if self.dialog is not None:
            dynamic = self.dialog.page_wide(selector)
            if dynamic is not None:
                return dynamic
        return self._children.get(selector)

    async def query_selector_all(self, selector: str):
        element = await self.query_selector(selector)
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
    def __init__(self, page: FakePage) -> None:
        self.config = FakeSessionConfig()
        self._playwright = FakePlaywright()
        self._context = FakeContext(page)

    async def _open_context(self):
        return self._playwright, self._context


def _build_page(*, logged_in: bool = True, **dialog_kwargs) -> tuple[FakePage, StagedDialog]:
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

    page = FakePage(children=children, dialog=dialog)
    return page, dialog


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
        hashtags=["#aikotravel"],
        approved_at="2026-08-03T00:00:00+00:00",
    )
    defaults.update(overrides)
    return PublishJob(**defaults)


class SenderFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.media_path = self.temp_dir / "feed_01.png"
        self.media_path.write_bytes(b"fake-image-bytes")
        self.queue_service = QueueService(self.temp_dir / "queue.json")
        self.history_service = HistoryService(self.temp_dir / "history")

    def _controller(self, page: FakePage) -> InstagramFeedSenderController:
        return InstagramFeedSenderController(
            session=FakeSession(page),
            feed_config=load_feed_preview_config(),
            publisher_config=PublisherConfig(),
            sender_config=load_feed_sender_config(),
            queue_service=self.queue_service,
            history_service=self.history_service,
            screenshots_dir=self.temp_dir / "screenshots",
            logs_dir=self.temp_dir / "logs",
            locks_dir=self.temp_dir / "locks",
        )

    def _caption(self, job: PublishJob):
        return build_final_caption(job.caption_text, job.hashtags, max_characters=2200)


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------


class CliArgumentTests(unittest.TestCase):
    def test_confirm_is_required_with_publish_approved(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--publish-approved", JOB_ID])

    def test_publish_approved_with_confirm_parses(self) -> None:
        args = parse_arguments(["--publish-approved", JOB_ID, "--confirm"])
        self.assertEqual(args.publish_approved, JOB_ID)
        self.assertTrue(args.confirm)

    def test_preflight_does_not_require_confirm(self) -> None:
        args = parse_arguments(["--preflight", JOB_ID])
        self.assertEqual(args.preflight, JOB_ID)
        self.assertFalse(args.confirm)

    def test_preflight_and_publish_approved_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--preflight", JOB_ID, "--publish-approved", JOB_ID, "--confirm"])

    def test_neither_flag_raises(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_only_a_single_job_id_accepted_per_flag(self) -> None:
        # argparse metavar=JOB_ID takes exactly one value; passing a
        # second positional is rejected as an unrecognized argument.
        with self.assertRaises(SystemExit):
            parse_arguments(["--preflight", JOB_ID, "extra-job-id"])


# ---------------------------------------------------------------------------
# Eligibility (pure, no browser)
# ---------------------------------------------------------------------------


class EligibilityTests(SenderFixture):
    def test_approved_job_is_publishable(self) -> None:
        job = _job(media_paths=[str(self.media_path)])
        check_job_publishable(job)  # does not raise

    def test_every_non_approved_status_is_rejected(self) -> None:
        for status in (
            "draft",
            "validation_failed",
            "pending_approval",
            "rejected",
            "scheduled",
            "publishing",
            "published",
            "failed",
            "cancelled",
        ):
            with self.subTest(status=status):
                job = _job(status=status, media_paths=[str(self.media_path)])
                with self.assertRaises(FeedPreviewIneligibleError):
                    check_job_publishable(job)

    def test_missing_approved_at_fails(self) -> None:
        job = _job(media_paths=[str(self.media_path)], approved_at=None)
        with self.assertRaises(FeedSenderIneligibleError):
            check_job_publishable(job)

    def test_already_published_at_fails(self) -> None:
        job = _job(
            media_paths=[str(self.media_path)],
            published_at="2026-08-02T00:00:00+00:00",
        )
        with self.assertRaises(FeedSenderIneligibleError):
            check_job_publishable(job)

    def test_already_has_platform_post_id_fails(self) -> None:
        job = _job(media_paths=[str(self.media_path)], platform_post_id="abc123")
        with self.assertRaises(FeedSenderIneligibleError):
            check_job_publishable(job)

    def test_missing_media_file_fails_before_browser(self) -> None:
        job = _job(media_paths=[str(self.temp_dir / "missing.png")])
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_publishable(job)

    def test_zero_byte_media_fails_before_browser(self) -> None:
        zero_byte = self.temp_dir / "empty.png"
        zero_byte.write_bytes(b"")
        job = _job(media_paths=[str(zero_byte)])
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_publishable(job)

    def test_empty_caption_fails_before_browser(self) -> None:
        job = _job(media_paths=[str(self.media_path)], caption_text="   ")
        with self.assertRaises(FeedPreviewIneligibleError):
            check_job_publishable(job)


# ---------------------------------------------------------------------------
# Preflight (never sends)
# ---------------------------------------------------------------------------


class PreflightTests(SenderFixture):
    def test_approved_feed_reaches_send_ready_state(self) -> None:
        page, dialog = _build_page()
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        result = asyncio.run(controller.preflight(job, self._caption(job)))

        self.assertTrue(result.share_button_found)
        self.assertTrue(result.share_button_enabled)
        self.assertFalse(result.share_clicked)
        self.assertFalse(result.published)
        self.assertEqual(result.share_click_count, 0)
        self.assertEqual(result.previous_status, "approved")
        self.assertEqual(result.final_status, "approved")

    def test_preflight_never_clicks_share(self) -> None:
        page, dialog = _build_page()
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        asyncio.run(controller.preflight(job, self._caption(job)))

        self.assertFalse(dialog.share_button.clicked)
        self.assertEqual(dialog.share_button.click_count, 0)

    def test_preflight_does_not_acquire_lock(self) -> None:
        page, _dialog = _build_page()
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        asyncio.run(controller.preflight(job, self._caption(job)))

        lock = PublishLock(self.temp_dir / "locks")
        self.assertFalse(lock.exists(JOB_ID))

    def test_preflight_does_not_touch_queue_or_history(self) -> None:
        page, _dialog = _build_page()
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)
        before = (self.temp_dir / "queue.json").read_bytes()

        asyncio.run(controller.preflight(job, self._caption(job)))

        after = (self.temp_dir / "queue.json").read_bytes()
        self.assertEqual(before, after)
        self.assertFalse((self.temp_dir / "history").exists())

    def test_preflight_writes_before_share_screenshot(self) -> None:
        page, _dialog = _build_page()
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        result = asyncio.run(controller.preflight(job, self._caption(job)))

        self.assertIsNotNone(result.screenshots["before_share"])
        self.assertTrue(result.screenshots["before_share"].endswith(f"before_share_{JOB_ID}.png"))


# ---------------------------------------------------------------------------
# Share safety gate (pre-click failures)
# ---------------------------------------------------------------------------


class ShareSafetyGateTests(SenderFixture):
    def test_zero_share_candidates_fail_safely(self) -> None:
        page, dialog = _build_page(share_missing=True)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(Exception):
            asyncio.run(controller.publish(job, self._caption(job)))

        self.assertEqual(self.queue_service.get(JOB_ID).status, "approved")

    def test_multiple_share_candidates_fail_safely(self) -> None:
        page, dialog = _build_page(share_ambiguous=True)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(Exception):
            asyncio.run(controller.publish(job, self._caption(job)))

        self.assertFalse(dialog.share_button.clicked)
        self.assertFalse(dialog.share_button_2.clicked)
        self.assertEqual(self.queue_service.get(JOB_ID).status, "approved")

    def test_disabled_share_fails_safely(self) -> None:
        page, dialog = _build_page(share_disabled=True)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(Exception):
            asyncio.run(controller.publish(job, self._caption(job)))

        self.assertFalse(dialog.share_button_disabled.clicked)
        self.assertEqual(self.queue_service.get(JOB_ID).status, "approved")

    def test_caption_mismatch_fails_before_share(self) -> None:
        page, dialog = _build_page(caption_mismatch=True)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(Exception):
            asyncio.run(controller.publish(job, self._caption(job)))

        self.assertFalse(dialog.share_button.clicked)
        self.assertEqual(self.queue_service.get(JOB_ID).status, "approved")

    def test_pre_click_failure_releases_lock(self) -> None:
        page, _dialog = _build_page(share_missing=True)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(Exception):
            asyncio.run(controller.publish(job, self._caption(job)))

        lock = PublishLock(self.temp_dir / "locks")
        self.assertFalse(lock.exists(JOB_ID))

    def test_pre_click_failure_writes_no_history_event(self) -> None:
        page, _dialog = _build_page(share_missing=True)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(Exception):
            asyncio.run(controller.publish(job, self._caption(job)))

        self.assertEqual(self.history_service.list_events("2026-08-01"), [])


# ---------------------------------------------------------------------------
# Successful publish
# ---------------------------------------------------------------------------


class PublishSuccessTests(SenderFixture):
    def _publish(self, **dialog_kwargs):
        page, dialog = _build_page(**dialog_kwargs)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)
        result = asyncio.run(controller.publish(job, self._caption(job)))
        return result, page, dialog, job

    def test_exactly_one_share_click_occurs(self) -> None:
        result, _page, dialog, _job_obj = self._publish()
        self.assertEqual(dialog.share_button.click_count, 1)
        self.assertEqual(result.share_click_count, 1)
        self.assertTrue(result.share_clicked)

    def test_verified_success_changes_publishing_to_published(self) -> None:
        result, _page, _dialog, _job_obj = self._publish()
        self.assertTrue(result.publish_verified)
        self.assertTrue(result.published)
        self.assertEqual(result.previous_status, "approved")
        self.assertEqual(result.final_status, "published")

        stored = self.queue_service.get(JOB_ID)
        self.assertEqual(stored.status, "published")
        self.assertIsNotNone(stored.published_at)
        self.assertIsNone(stored.error)

    def test_platform_post_id_and_url_captured_when_confirmed(self) -> None:
        result, _page, _dialog, _job_obj = self._publish()
        self.assertEqual(result.platform_post_id, "ABC123xyz")
        self.assertEqual(result.platform_url, "https://www.instagram.com/p/ABC123xyz/")
        stored = self.queue_service.get(JOB_ID)
        self.assertEqual(stored.platform_post_id, "ABC123xyz")
        self.assertEqual(stored.platform_url, "https://www.instagram.com/p/ABC123xyz/")

    def test_verification_signals_include_all_available_positive_signals(self) -> None:
        result, _page, _dialog, _job_obj = self._publish()
        self.assertIn("dialog_closed", result.verification_signals)
        self.assertIn("success_message_visible", result.verification_signals)
        self.assertIn("url_is_post_permalink", result.verification_signals)

    def test_history_event_order_on_success(self) -> None:
        self._publish()
        events = self.history_service.list_events("2026-08-01")
        self.assertEqual(
            [event["event_type"] for event in events],
            ["publishing_started", "published"],
        )
        self.assertEqual(events[0]["previous_status"], "approved")
        self.assertEqual(events[0]["new_status"], "publishing")
        self.assertEqual(events[1]["previous_status"], "publishing")
        self.assertEqual(events[1]["new_status"], "published")

    def test_queue_transition_to_publishing_happens_immediately_before_click(self) -> None:
        # Reconstruct by checking the queue file was written with
        # status="publishing" at least once during the run: the final
        # state is "published", but the history event proves the
        # intermediate transition occurred exactly once.
        self._publish()
        events = self.history_service.list_events("2026-08-01")
        publishing_events = [e for e in events if e["event_type"] == "publishing_started"]
        self.assertEqual(len(publishing_events), 1)

    def test_lock_is_retained_after_verified_success(self) -> None:
        self._publish()
        lock = PublishLock(self.temp_dir / "locks")
        self.assertTrue(lock.exists(JOB_ID))

    def test_three_required_screenshots_saved_on_success(self) -> None:
        result, _page, _dialog, _job_obj = self._publish()
        self.assertIsNotNone(result.screenshots["before_share"])
        self.assertIsNotNone(result.screenshots["after_share"])
        self.assertIsNotNone(result.screenshots["verified"])
        self.assertTrue(result.screenshots["before_share"].endswith(f"before_share_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["after_share"].endswith(f"after_share_{JOB_ID}.png"))
        self.assertTrue(result.screenshots["verified"].endswith(f"verified_{JOB_ID}.png"))

    def test_confirmation_block_prints_before_click(self) -> None:
        import io
        import contextlib

        page, dialog = _build_page()
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            asyncio.run(controller.publish(job, self._caption(job)))

        printed = buffer.getvalue()
        self.assertIn(JOB_ID, printed)
        self.assertIn("WARNING", printed)


# ---------------------------------------------------------------------------
# Failed / inconclusive publish
# ---------------------------------------------------------------------------


class PublishFailureTests(SenderFixture):
    def test_share_click_failure_changes_publishing_to_failed(self) -> None:
        page, dialog = _build_page(fail_share_click=True)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(ShareClickFailedError):
            asyncio.run(controller.publish(job, self._caption(job)))

        stored = self.queue_service.get(JOB_ID)
        self.assertEqual(stored.status, "failed")
        self.assertIsNotNone(stored.error)
        self.assertIsNone(stored.published_at)

        events = self.history_service.list_events("2026-08-01")
        self.assertEqual([e["event_type"] for e in events], ["publishing_started", "failed"])

    def test_share_click_failure_retains_lock(self) -> None:
        page, _dialog = _build_page(fail_share_click=True)
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(ShareClickFailedError):
            asyncio.run(controller.publish(job, self._caption(job)))

        lock = PublishLock(self.temp_dir / "locks")
        self.assertTrue(lock.exists(JOB_ID))

    def test_inconclusive_verification_changes_publishing_to_failed(self) -> None:
        page, dialog = _build_page(
            close_dialog_on_share=False,
            show_success_message=False,
            change_url_on_share=False,
        )
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(PublishVerificationInconclusiveError):
            asyncio.run(controller.publish(job, self._caption(job)))

        stored = self.queue_service.get(JOB_ID)
        self.assertEqual(stored.status, "failed")
        self.assertIsNone(stored.published_at)

    def test_inconclusive_error_requires_manual_check(self) -> None:
        page, dialog = _build_page(
            close_dialog_on_share=False,
            show_success_message=False,
            change_url_on_share=False,
        )
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(PublishVerificationInconclusiveError) as ctx:
            asyncio.run(controller.publish(job, self._caption(job)))

        self.assertEqual(
            str(ctx.exception),
            "publish_verification_inconclusive_manual_check_required",
        )
        stored = self.queue_service.get(JOB_ID)
        self.assertEqual(
            stored.error, "publish_verification_inconclusive_manual_check_required"
        )

    def test_single_signal_alone_is_inconclusive_not_published(self) -> None:
        # Only the URL changes; dialog stays open and no success
        # message appears. One positive signal must never be enough.
        page, dialog = _build_page(
            close_dialog_on_share=False,
            show_success_message=False,
            change_url_on_share=True,
        )
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(PublishVerificationInconclusiveError):
            asyncio.run(controller.publish(job, self._caption(job)))

        self.assertEqual(self.queue_service.get(JOB_ID).status, "failed")

    def test_error_message_present_vetoes_verification_even_with_other_signals(self) -> None:
        page, dialog = _build_page(
            close_dialog_on_share=True,
            show_success_message=True,
            change_url_on_share=True,
            show_error_message=True,
        )
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(PublishVerificationInconclusiveError):
            asyncio.run(controller.publish(job, self._caption(job)))

    def test_inconclusive_retains_lock_no_automatic_retry(self) -> None:
        page, dialog = _build_page(
            close_dialog_on_share=False,
            show_success_message=False,
            change_url_on_share=False,
        )
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        with self.assertRaises(PublishVerificationInconclusiveError):
            asyncio.run(controller.publish(job, self._caption(job)))

        lock = PublishLock(self.temp_dir / "locks")
        self.assertTrue(lock.exists(JOB_ID))
        # No retry CLI/method exists at all — see StructuralSafetyTests.
        self.assertFalse(hasattr(InstagramFeedSenderController, "retry"))
        self.assertFalse(hasattr(InstagramFeedSenderController, "retry_publish"))


# ---------------------------------------------------------------------------
# Operation lock
# ---------------------------------------------------------------------------


class LockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.lock = PublishLock(self.temp_dir / "locks")

    def test_acquire_creates_lock_with_required_fields(self) -> None:
        path = self.lock.acquire(JOB_ID)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("pid", payload)
        self.assertEqual(payload["job_id"], JOB_ID)
        self.assertIn("started_at", payload)

    def test_second_acquire_for_same_job_id_refuses(self) -> None:
        self.lock.acquire(JOB_ID)
        with self.assertRaises(PublishLockError):
            self.lock.acquire(JOB_ID)

    def test_different_job_ids_do_not_conflict(self) -> None:
        self.lock.acquire(JOB_ID)
        self.lock.acquire("2026-08-02-instagram_feed")  # does not raise

    def test_release_allows_reacquiring(self) -> None:
        self.lock.acquire(JOB_ID)
        self.lock.release(JOB_ID)
        self.lock.acquire(JOB_ID)  # does not raise

    def test_release_missing_lock_is_a_no_op(self) -> None:
        self.lock.release("never-locked")  # does not raise

    def test_lock_lives_under_the_given_temp_directory_only(self) -> None:
        path = self.lock.acquire(JOB_ID)
        self.assertTrue(str(path).startswith(str(self.temp_dir)))


# ---------------------------------------------------------------------------
# Structural safety scan
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def test_exactly_one_share_click_call_in_source(self) -> None:
        matches = re.findall(r"share_lookup\.element\.click\(", MODULE_SOURCE)
        self.assertEqual(len(matches), 1, matches)

    def test_no_press_or_keyboard_calls(self) -> None:
        self.assertNotIn(".press(", MODULE_SOURCE)
        self.assertNotIn("keyboard.press", MODULE_SOURCE)
        self.assertNotIn("keyboard.type", MODULE_SOURCE)

    def test_no_javascript_evaluate_click(self) -> None:
        self.assertNotIn(".evaluate(", MODULE_SOURCE)
        self.assertNotIn("evaluate_handle(", MODULE_SOURCE)

    def test_no_force_click(self) -> None:
        self.assertNotIn("force=True", MODULE_SOURCE)

    def test_no_form_submit(self) -> None:
        self.assertNotIn(".submit(", MODULE_SOURCE)
        self.assertNotIn("form.submit", MODULE_SOURCE)

    def test_no_retry_loop_around_share_click(self) -> None:
        # The click call must not appear inside a "for"/"while" block
        # anywhere in the file (a crude but effective structural
        # check: no loop keyword within 5 lines above the click line,
        # scoped to the click's own indentation level or shallower).
        lines = MODULE_SOURCE.splitlines()
        click_line_indices = [
            index for index, line in enumerate(lines) if "share_lookup.element.click(" in line
        ]
        self.assertTrue(click_line_indices)

        for index in click_line_indices:
            click_indent = len(lines[index]) - len(lines[index].lstrip())
            for back in range(max(0, index - 8), index):
                candidate = lines[back]
                candidate_indent = len(candidate) - len(candidate.lstrip())
                stripped = candidate.strip()
                if candidate_indent <= click_indent and (
                    stripped.startswith("for ") or stripped.startswith("while ")
                ):
                    self.fail(f"Found a loop near the Share click: {candidate!r}")

    def test_disallowed_action_methods_not_implemented(self) -> None:
        for action in DISALLOWED_ACTIONS:
            self.assertFalse(
                hasattr(InstagramFeedSenderController, action),
                f"InstagramFeedSenderController must not implement '{action}'",
            )

    def test_preview_module_still_has_no_share_click_capability(self) -> None:
        matches = re.findall(r"\b\w*[Ss]hare\w*\.click\(", PREVIEW_MODULE_SOURCE)
        self.assertEqual(matches, [])
        self.assertNotIn(".press(", PREVIEW_MODULE_SOURCE)

        pattern = re.compile(r"""\.status\s*=\s*["'](publishing|published)["']""")
        self.assertIsNone(pattern.search(PREVIEW_MODULE_SOURCE))

        for action in ("share", "publish", "submit", "press", "press_enter"):
            from .instagram_feed_preview import InstagramFeedPreviewController

            self.assertFalse(hasattr(InstagramFeedPreviewController, action))


# ---------------------------------------------------------------------------
# No secrets in logs/screenshots content
# ---------------------------------------------------------------------------


class NoSecretsInDiagnosticsTests(SenderFixture):
    FORBIDDEN_SUBSTRINGS = ("cookie", "password", "session_token", "sessionid", "csrftoken")

    def test_log_payload_contains_no_secret_looking_fields(self) -> None:
        page, _dialog = _build_page()
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)

        asyncio.run(controller.publish(job, self._caption(job)))

        log_files = list((self.temp_dir / "logs").glob(f"instagram_feed_sender_{JOB_ID}_*.json"))
        self.assertTrue(log_files)

        for log_file in log_files:
            text = log_file.read_text(encoding="utf-8").lower()
            for forbidden in self.FORBIDDEN_SUBSTRINGS:
                self.assertNotIn(forbidden, text)

    def test_module_source_never_reads_browser_profile_contents(self) -> None:
        # The module's own comment explicitly documents that cookies
        # etc. are never logged (search for that string is fine); what
        # must never exist is actual code reading profile/cookie data.
        self.assertNotIn("cookies(", MODULE_SOURCE.lower())
        self.assertNotIn("context.cookies", MODULE_SOURCE.lower())
        self.assertNotIn("open(str(self.session.config.profile_dir", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# src/social untouched
# ---------------------------------------------------------------------------


class SocialPackageUntouchedTests(SenderFixture):
    def test_social_sender_files_are_byte_for_byte_unchanged(self) -> None:
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
        controller = self._controller(page)
        job = _job(media_paths=[str(self.media_path)])
        self.queue_service.add_job(job)
        asyncio.run(controller.publish(job, self._caption(job)))

        after = _snapshot()
        self.assertEqual(before, after)
        self.assertTrue(before)


if __name__ == "__main__":
    unittest.main()
