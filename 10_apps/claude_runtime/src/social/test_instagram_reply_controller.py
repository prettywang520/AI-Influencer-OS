from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from .approval_service import ApprovalService
from .instagram_models import (
    ApprovalRecordInvalidError,
    ApprovalRecordNotApprovedError,
    ApprovalRecordNotFoundError,
    CommentNotFoundError,
    LoginDetectionConfig,
    ReplyControlNotFoundError,
    ReplyControllerLoginLostError,
    ReplyPostAccessError,
    ReplyTextboxNotFoundError,
)
from .instagram_reply_controller import (
    DISALLOWED_ACTIONS,
    InstagramReplyController,
    ReplyContextNotVerifiedError,
    ReplyPreviewResult,
    load_approved_record,
    load_reply_controller_config,
)

# None of these tests open a real browser or contact Instagram.
# Playwright is simulated with small fake objects; the approval queue
# is a real temp JSON file.

REAL_APPROVAL_ID = "appr-comment_id-999"
REAL_COMMENT_ID = "999"
REAL_POST_URL = "https://www.instagram.com/p/ABC123/"
REAL_USERNAME = "alice"


def _approved_record(**overrides) -> dict:
    record = {
        "approval_id": REAL_APPROVAL_ID,
        "stable_id": "comment_id:999",
        "comment_id": REAL_COMMENT_ID,
        "post_url": REAL_POST_URL,
        "username": REAL_USERNAME,
        "comment_text": "beautiful photo!",
        "classification": "compliment",
        "proposed_reply": "thank you so much 🤍",
        "safety_route": "auto_eligible",
        "status": "approved",
        "created_at": "2026-07-31T00:00:00Z",
        "approved_at": "2026-07-31T00:05:00Z",
        "rejected_at": None,
        "sent_at": None,
        "error": None,
    }
    record.update(overrides)
    return record


def _write_queue(temp_dir: str, records: list[dict]) -> ApprovalService:
    service = ApprovalService(queue_dir=temp_dir)
    queue = service.load_queue()
    queue["approvals"] = records
    service.save_queue(queue)
    return service


# ---------------------------------------------------------------------------
# Fake Playwright-shaped objects
# ---------------------------------------------------------------------------


class FakeElement:
    def __init__(
        self,
        *,
        children: dict[str, "FakeElement"] | None = None,
        child_lists: dict[str, list["FakeElement"]] | None = None,
        attrs: dict[str, str] | None = None,
        text: str = "",
        value: str | None = None,
        fail_click: bool = False,
        fail_input_value: bool = False,
    ) -> None:
        self._children = children or {}
        self._child_lists = child_lists or {}
        self._attrs = attrs or {}
        self._text = text
        self._value = value
        self._fail_click = fail_click
        self._fail_input_value = fail_input_value
        self.clicked = False
        self.filled_text: str | None = None
        self.typed_text: str | None = None

    async def query_selector(self, selector: str):
        return self._children.get(selector)

    async def query_selector_all(self, selector: str):
        return self._child_lists.get(selector, [])

    async def get_attribute(self, name: str):
        return self._attrs.get(name)

    async def inner_text(self) -> str:
        return self._text

    async def input_value(self) -> str:
        if self._fail_input_value:
            raise RuntimeError("not an <input>/<textarea>")
        return self._value or ""

    async def click(self, timeout=None):
        if self._fail_click:
            raise RuntimeError("simulated click failure")
        self.clicked = True

    async def fill(self, text: str):
        self.filled_text = text

    async def type(self, text: str):
        self.typed_text = (self.typed_text or "") + text

    # Explicitly absent on purpose: press(), submit(), send() — see
    # DisallowedActionsTests below.


class FakePage:
    def __init__(
        self,
        *,
        children: dict[str, FakeElement] | None = None,
        url: str = REAL_POST_URL,
        goto_error: Exception | None = None,
    ) -> None:
        self._children = children or {}
        self.url = url
        self._goto_error = goto_error
        self.screenshot_calls: list[str] = []
        self.wheel_calls = 0

        outer = self

        class _Mouse:
            async def wheel(self, x, y):
                outer.wheel_calls += 1

        self.mouse = _Mouse()

    async def goto(self, url, timeout=None, wait_until=None):
        if self._goto_error is not None:
            raise self._goto_error

    async def wait_for_timeout(self, ms):
        pass

    async def query_selector(self, selector: str):
        return self._children.get(selector)

    async def screenshot(self, path, full_page=True):
        self.screenshot_calls.append(path)
        # No real file is written; tests only assert on the path.


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
        self.login_detection = LoginDetectionConfig(
            logged_in_selectors=["svg[aria-label='Home']"],
            logged_out_selectors=["input[type='password']"],
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


def _real_selectors():
    return load_reply_controller_config().selectors


def _base_row_and_container(*, comment_id: str = REAL_COMMENT_ID, extra_row_children=None):
    """
    Build a comment row that matches comment_id (permalink present)
    and has a working Reply control, plus the container wrapping it.
    Returns (container, row, reply_button).
    """
    selectors = _real_selectors()

    permalink_selector = selectors.comment_permalink_template.format(
        comment_id=comment_id
    )
    reply_selector = selectors.reply_button[0]
    item_selector = selectors.comment_item[0]

    reply_button = FakeElement()
    children = {
        permalink_selector: FakeElement(),
        reply_selector: reply_button,
    }
    children.update(extra_row_children or {})

    row = FakeElement(children=children)
    container = FakeElement(child_lists={item_selector: [row]})

    return container, row, reply_button


def _build_page(*, container: FakeElement, extra_page_children=None) -> FakePage:
    selectors = _real_selectors()
    container_selector = selectors.comment_container[0]

    children = {
        "svg[aria-label='Home']": FakeElement(),
        container_selector: container,
    }
    children.update(extra_page_children or {})

    return FakePage(children=children)


def _build_success_page(
    *, comment_id: str = REAL_COMMENT_ID
) -> tuple[FakePage, FakeElement, FakeElement]:
    """
    A page that verifies via the row_scoped_composer signal: the
    comment row itself contains a composer element.
    """
    selectors = _real_selectors()
    row_composer_selector = selectors.row_reply_composer[0]

    composer = FakeElement()
    container, row, reply_button = _base_row_and_container(
        comment_id=comment_id,
        extra_row_children={row_composer_selector: composer},
    )
    page = _build_page(container=container)

    return page, reply_button, composer


# ---------------------------------------------------------------------------
# load_approved_record — pure validation, no Playwright
# ---------------------------------------------------------------------------


class LoadApprovedRecordTests(unittest.TestCase):
    def test_approved_record_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(temp_dir, [_approved_record()])
            record = load_approved_record(REAL_APPROVAL_ID, approval_service=service)
            self.assertEqual(record["approval_id"], REAL_APPROVAL_ID)

    def test_missing_approval_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(temp_dir, [])

            with self.assertRaises(ApprovalRecordNotFoundError):
                load_approved_record("appr-does-not-exist", approval_service=service)

    def test_pending_approval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(
                temp_dir, [_approved_record(status="pending_approval")]
            )

            with self.assertRaises(ApprovalRecordNotApprovedError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_rejected_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(temp_dir, [_approved_record(status="rejected")])

            with self.assertRaises(ApprovalRecordNotApprovedError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_skipped_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(temp_dir, [_approved_record(status="skipped")])

            with self.assertRaises(ApprovalRecordNotApprovedError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_missing_proposed_reply_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(
                temp_dir, [_approved_record(proposed_reply=None)]
            )

            with self.assertRaises(ApprovalRecordInvalidError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_missing_comment_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(temp_dir, [_approved_record(comment_id=None)])

            with self.assertRaises(ApprovalRecordInvalidError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_missing_post_url_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(temp_dir, [_approved_record(post_url="")])

            with self.assertRaises(ApprovalRecordInvalidError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)


# ---------------------------------------------------------------------------
# preview_reply — mocked Playwright end-to-end (success path)
# ---------------------------------------------------------------------------


class PreviewReplySuccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_reply_controller_config()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config.screenshots_dir = Path(self.temp_dir.name) / "screenshots"
        self.config.logs_dir = Path(self.temp_dir.name) / "logs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_approved_record_reaches_preview_state(self) -> None:
        page, reply_button, composer = _build_success_page()
        session = FakeSession(page)
        controller = InstagramReplyController(session=session, config=self.config)

        record = _approved_record()
        result = asyncio.run(controller.preview_reply(record))

        self.assertIsInstance(result, ReplyPreviewResult)
        self.assertEqual(result.login_status, "logged_in")
        self.assertEqual(result.approval_id, REAL_APPROVAL_ID)
        self.assertTrue(result.comment_found)
        self.assertTrue(result.reply_control_clicked)
        self.assertTrue(result.reply_context_verified)
        self.assertEqual(result.reply_context_method, "row_scoped_composer")
        self.assertEqual(result.reply_target_username, REAL_USERNAME)
        self.assertTrue(result.textbox_filled)
        self.assertFalse(result.sent)

        self.assertTrue(reply_button.clicked)
        self.assertEqual(composer.filled_text, record["proposed_reply"])

    def test_screenshot_path_includes_approval_id(self) -> None:
        page, _, _ = _build_success_page()
        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        result = asyncio.run(controller.preview_reply(_approved_record()))

        expected_name = f"reply_preview_{REAL_APPROVAL_ID}.png"
        self.assertEqual(Path(result.screenshot_path).name, expected_name)
        self.assertIn(page.screenshot_calls[-1], result.screenshot_path)

    def test_before_and_after_click_screenshots_are_saved(self) -> None:
        page, _, _ = _build_success_page()
        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        asyncio.run(controller.preview_reply(_approved_record()))

        names = {Path(p).name for p in page.screenshot_calls}
        self.assertIn(f"before_reply_click_{REAL_APPROVAL_ID}.png", names)
        self.assertIn(f"after_reply_click_{REAL_APPROVAL_ID}.png", names)
        self.assertIn(f"reply_preview_{REAL_APPROVAL_ID}.png", names)

        before_index = page.screenshot_calls.index(
            str(self.config.screenshots_dir / f"before_reply_click_{REAL_APPROVAL_ID}.png")
        )
        after_index = page.screenshot_calls.index(
            str(self.config.screenshots_dir / f"after_reply_click_{REAL_APPROVAL_ID}.png")
        )
        self.assertLess(before_index, after_index)

    def test_context_and_playwright_are_cleaned_up(self) -> None:
        page, _, _ = _build_success_page()
        session = FakeSession(page)
        controller = InstagramReplyController(session=session, config=self.config)

        asyncio.run(controller.preview_reply(_approved_record()))

        self.assertTrue(session._context.closed)
        self.assertTrue(session._playwright.stopped)

    def test_diagnostic_log_written_with_required_fields(self) -> None:
        page, _, _ = _build_success_page()
        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        asyncio.run(controller.preview_reply(_approved_record()))

        log_path = self.config.logs_dir / "reply_controller_diagnostics.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])

        required_fields = {
            "approval_id",
            "post_url",
            "comment_id",
            "username",
            "proposed_reply",
            "comment_permalink",
            "row_selector",
            "reply_selector",
            "composer_selector",
            "reply_context_signal",
            "target_username",
            "selector_used",
            "screenshot_path",
            "result",
            "error",
            "created_at",
        }
        self.assertTrue(required_fields.issubset(entry.keys()))
        self.assertEqual(entry["result"], "preview_success")
        self.assertIsNone(entry["error"])
        self.assertEqual(entry["reply_context_signal"], "row_scoped_composer")
        self.assertEqual(entry["target_username"], REAL_USERNAME)
        self.assertIn(REAL_COMMENT_ID, entry["comment_permalink"])


# ---------------------------------------------------------------------------
# preview_reply — threaded-reply context verification (Phase 8D.1)
# ---------------------------------------------------------------------------


class ReplyContextVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_reply_controller_config()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config.screenshots_dir = Path(self.temp_dir.name) / "screenshots"
        self.config.logs_dir = Path(self.temp_dir.name) / "logs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_generic_composer_without_reply_context_fails(self) -> None:
        """
        Only a page-level generic "Add a comment" box exists: no
        row-scoped composer, no @mention prefill, no "Replying to"
        text, no DOM attribute. This must NOT verify.
        """
        selectors = _real_selectors()
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        generic_composer = FakeElement(value="")
        page = _build_page(
            container=container,
            extra_page_children={generic_selector: generic_composer},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyContextNotVerifiedError):
            asyncio.run(controller.preview_reply(_approved_record()))

        # The generic composer must never have been filled.
        self.assertIsNone(generic_composer.filled_text)

    def test_at_mention_prefilled_composer_passes(self) -> None:
        selectors = _real_selectors()
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        prefilled_composer = FakeElement(value=f"@{REAL_USERNAME} ")
        page = _build_page(
            container=container,
            extra_page_children={generic_selector: prefilled_composer},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        result = asyncio.run(controller.preview_reply(_approved_record()))

        self.assertTrue(result.reply_context_verified)
        self.assertEqual(result.reply_context_method, "at_mention_prefill")
        # Phase 8D.2: the pre-existing @mention must be preserved, not
        # overwritten by a bare fill(proposed_reply).
        expected = f"@{REAL_USERNAME} {_approved_record()['proposed_reply']}"
        self.assertEqual(prefilled_composer.filled_text, expected)
        self.assertTrue(result.mention_preserved)

    def test_visible_replying_to_username_passes(self) -> None:
        selectors = _real_selectors()
        replying_selector = selectors.replying_to_text[0]
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        replying_label = FakeElement(text=f"Replying to @{REAL_USERNAME}")
        composer = FakeElement(value="")

        page = _build_page(
            container=container,
            extra_page_children={
                replying_selector: replying_label,
                generic_selector: composer,
            },
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        result = asyncio.run(controller.preview_reply(_approved_record()))

        self.assertTrue(result.reply_context_verified)
        self.assertEqual(result.reply_context_method, "replying_to_text")
        self.assertEqual(composer.filled_text, _approved_record()["proposed_reply"])

    def test_row_scoped_composer_passes(self) -> None:
        page, _reply_button, composer = _build_success_page()

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        result = asyncio.run(controller.preview_reply(_approved_record()))

        self.assertTrue(result.reply_context_verified)
        self.assertEqual(result.reply_context_method, "row_scoped_composer")
        self.assertEqual(composer.filled_text, _approved_record()["proposed_reply"])

    def test_dom_attribute_signal_passes(self) -> None:
        selectors = _real_selectors()
        attribute_selector = selectors.comment_context_attribute_template[0].format(
            comment_id=REAL_COMMENT_ID
        )
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        composer = FakeElement()

        page = _build_page(
            container=container,
            extra_page_children={
                attribute_selector: FakeElement(),
                generic_selector: composer,
            },
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        result = asyncio.run(controller.preview_reply(_approved_record()))

        self.assertTrue(result.reply_context_verified)
        self.assertEqual(result.reply_context_method, "dom_attribute")
        # dom_attribute does not itself locate a composer, so the
        # controller falls back to searching for one to fill.
        self.assertEqual(composer.filled_text, _approved_record()["proposed_reply"])

    def test_no_text_is_filled_when_verification_fails(self) -> None:
        selectors = _real_selectors()
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        generic_composer = FakeElement(value="")
        page = _build_page(
            container=container,
            extra_page_children={generic_selector: generic_composer},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyContextNotVerifiedError):
            asyncio.run(controller.preview_reply(_approved_record()))

        self.assertIsNone(generic_composer.filled_text)
        self.assertIsNone(generic_composer.typed_text)

    def test_verification_failure_still_takes_before_and_after_screenshots(
        self,
    ) -> None:
        selectors = _real_selectors()
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        page = _build_page(
            container=container,
            extra_page_children={generic_selector: FakeElement(value="")},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyContextNotVerifiedError):
            asyncio.run(controller.preview_reply(_approved_record()))

        names = {Path(p).name for p in page.screenshot_calls}
        self.assertIn(f"before_reply_click_{REAL_APPROVAL_ID}.png", names)
        self.assertIn(f"after_reply_click_{REAL_APPROVAL_ID}.png", names)
        # The reply is never sent to preview state, so the final
        # filled-textbox screenshot must NOT have been taken.
        self.assertNotIn(f"reply_preview_{REAL_APPROVAL_ID}.png", names)

    def test_verification_failure_logs_diagnostic_entry(self) -> None:
        selectors = _real_selectors()
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        page = _build_page(
            container=container,
            extra_page_children={generic_selector: FakeElement(value="")},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyContextNotVerifiedError):
            asyncio.run(controller.preview_reply(_approved_record()))

        log_path = self.config.logs_dir / "reply_controller_diagnostics.jsonl"
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())

        self.assertEqual(entry["result"], "failed")
        self.assertEqual(entry["error"], "reply_context_not_verified")
        self.assertIsNone(entry["reply_context_signal"])
        self.assertEqual(entry["target_username"], REAL_USERNAME)

    def test_verified_but_no_composer_findable_raises_textbox_not_found(self) -> None:
        """
        DOM-attribute verification succeeds (context confirmed) but no
        composer element exists anywhere to actually fill.
        """
        selectors = _real_selectors()
        attribute_selector = selectors.comment_context_attribute_template[0].format(
            comment_id=REAL_COMMENT_ID
        )

        container, row, _reply_button = _base_row_and_container()
        page = _build_page(
            container=container,
            extra_page_children={attribute_selector: FakeElement()},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyTextboxNotFoundError):
            asyncio.run(controller.preview_reply(_approved_record()))


# ---------------------------------------------------------------------------
# preview_reply — failure paths unrelated to context verification
# ---------------------------------------------------------------------------


class PreviewReplyFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_reply_controller_config()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config.screenshots_dir = Path(self.temp_dir.name) / "screenshots"
        self.config.logs_dir = Path(self.temp_dir.name) / "logs"
        # Keep failure-path tests fast: no real delay anyway (FakePage
        # wait_for_timeout is a no-op), but keep attempts small and
        # explicit for clarity.
        self.config.max_scroll_attempts = 2

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_post_cannot_open_raises(self) -> None:
        page = FakePage(goto_error=RuntimeError("navigation failed"))
        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyPostAccessError):
            asyncio.run(controller.preview_reply(_approved_record()))

    def test_login_lost_raises(self) -> None:
        # No logged_in selector present, and a logged_out one is.
        page = FakePage(children={"input[type='password']": FakeElement()})
        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyControllerLoginLostError):
            asyncio.run(controller.preview_reply(_approved_record()))

    def test_exact_comment_not_found_raises(self) -> None:
        selectors = _real_selectors()
        item_selector = selectors.comment_item[0]

        # A row exists, but none match the target comment_id.
        other_row = FakeElement(
            children={
                selectors.comment_permalink_template.format(
                    comment_id="not-the-one"
                ): FakeElement()
            }
        )
        container = FakeElement(child_lists={item_selector: [other_row]})
        page = _build_page(container=container)

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(CommentNotFoundError):
            asyncio.run(controller.preview_reply(_approved_record()))

        self.assertGreater(page.wheel_calls, 0)

    def test_reply_control_not_found_raises(self) -> None:
        selectors = _real_selectors()
        item_selector = selectors.comment_item[0]
        permalink_selector = selectors.comment_permalink_template.format(
            comment_id=REAL_COMMENT_ID
        )

        # Row matches the comment, but has no Reply control at all.
        row = FakeElement(children={permalink_selector: FakeElement()})
        container = FakeElement(child_lists={item_selector: [row]})
        page = _build_page(container=container)

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyControlNotFoundError):
            asyncio.run(controller.preview_reply(_approved_record()))

    def test_failure_paths_still_write_a_diagnostic_log_entry(self) -> None:
        page = FakePage(goto_error=RuntimeError("boom"))
        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        with self.assertRaises(ReplyPostAccessError):
            asyncio.run(controller.preview_reply(_approved_record()))

        log_path = self.config.logs_dir / "reply_controller_diagnostics.jsonl"
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["result"], "failed")
        self.assertIsNotNone(entry["error"])


# ---------------------------------------------------------------------------
# Strict dry-run safety
# ---------------------------------------------------------------------------


class DryRunSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_reply_controller_config()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config.screenshots_dir = Path(self.temp_dir.name) / "screenshots"
        self.config.logs_dir = Path(self.temp_dir.name) / "logs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_no_send_action_is_called(self) -> None:
        """
        The textbox is filled, never submitted: no press()/submit()/
        send() exists on FakeElement, so calling any of them would
        raise AttributeError. Asserting filled_text (not sent) proves
        only fill/type was used.
        """
        page, reply_button, composer = _build_success_page()
        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        record = _approved_record()
        result = asyncio.run(controller.preview_reply(record))

        self.assertEqual(composer.filled_text, record["proposed_reply"])
        self.assertIsNone(composer.typed_text)
        self.assertFalse(hasattr(composer, "sent"))
        self.assertFalse(result.sent)

    def test_queue_json_remains_byte_for_byte_unchanged_after_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            before_bytes = service.queue_path.read_bytes()

            record = load_approved_record(REAL_APPROVAL_ID, approval_service=service)

            page, _, _ = _build_success_page()
            controller = InstagramReplyController(
                session=FakeSession(page), config=self.config
            )
            asyncio.run(controller.preview_reply(record))

            after_bytes = service.queue_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)

    def test_queue_unchanged_even_on_a_failure_path(self) -> None:
        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            before_bytes = service.queue_path.read_bytes()

            record = load_approved_record(REAL_APPROVAL_ID, approval_service=service)

            page = FakePage(goto_error=RuntimeError("boom"))
            controller = InstagramReplyController(
                session=FakeSession(page), config=self.config
            )

            with self.assertRaises(ReplyPostAccessError):
                asyncio.run(controller.preview_reply(record))

            after_bytes = service.queue_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)

    def test_queue_unchanged_when_reply_context_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            before_bytes = service.queue_path.read_bytes()

            record = load_approved_record(REAL_APPROVAL_ID, approval_service=service)

            selectors = _real_selectors()
            generic_selector = selectors.reply_textbox[0]
            container, row, _reply_button = _base_row_and_container()
            page = _build_page(
                container=container,
                extra_page_children={generic_selector: FakeElement(value="")},
            )

            controller = InstagramReplyController(
                session=FakeSession(page), config=self.config
            )

            with self.assertRaises(ReplyContextNotVerifiedError):
                asyncio.run(controller.preview_reply(record))

            after_bytes = service.queue_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)

    def test_no_mutating_action_methods_exist_on_controller(self) -> None:
        for action in DISALLOWED_ACTIONS:
            self.assertFalse(
                hasattr(InstagramReplyController, action),
                f"InstagramReplyController must not implement '{action}'",
            )

        for forbidden_name in (
            "send",
            "post",
            "press_enter",
            "submit",
            "like",
            "follow",
            "unfollow",
            "hide",
            "delete",
            "report",
            "send_dm",
            "approve",
            "reject",
        ):
            self.assertFalse(
                hasattr(InstagramReplyController, forbidden_name),
                f"InstagramReplyController must not implement '{forbidden_name}'",
            )

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("send", DISALLOWED_ACTIONS)
        self.assertIn("post", DISALLOWED_ACTIONS)
        self.assertIn("press_enter", DISALLOWED_ACTIONS)


# ---------------------------------------------------------------------------
# _compose_final_text — pure unit tests (Phase 8D.2)
# ---------------------------------------------------------------------------


class ComposeFinalTextTests(unittest.TestCase):
    def test_preserves_existing_at_username_prefix(self) -> None:
        final_text, preserved = InstagramReplyController._compose_final_text(
            existing_text="@joaquimfrancisco.o",
            proposed_reply="aww thank you 🤍",
            target_username="joaquimfrancisco.o",
        )
        self.assertEqual(final_text, "@joaquimfrancisco.o aww thank you 🤍")
        self.assertTrue(preserved)

    def test_exactly_one_space_separates_mention_and_reply(self) -> None:
        # Existing text has a trailing space already — must not become
        # a double space in the final text.
        final_text, preserved = InstagramReplyController._compose_final_text(
            existing_text="@joaquimfrancisco.o ",
            proposed_reply="aww thank you 🤍",
            target_username="joaquimfrancisco.o",
        )
        self.assertEqual(final_text, "@joaquimfrancisco.o aww thank you 🤍")
        self.assertNotIn("  ", final_text)
        self.assertTrue(preserved)

    def test_mention_is_not_duplicated(self) -> None:
        final_text, _ = InstagramReplyController._compose_final_text(
            existing_text="@joaquimfrancisco.o",
            proposed_reply="aww thank you 🤍",
            target_username="joaquimfrancisco.o",
        )
        self.assertEqual(final_text.count("@joaquimfrancisco.o"), 1)

    def test_no_prefix_uses_proposed_reply_unchanged(self) -> None:
        final_text, preserved = InstagramReplyController._compose_final_text(
            existing_text="",
            proposed_reply="aww thank you 🤍",
            target_username="joaquimfrancisco.o",
        )
        self.assertEqual(final_text, "aww thank you 🤍")
        self.assertFalse(preserved)

    def test_unrelated_existing_text_does_not_match(self) -> None:
        final_text, preserved = InstagramReplyController._compose_final_text(
            existing_text="@someone_else",
            proposed_reply="aww thank you 🤍",
            target_username="joaquimfrancisco.o",
        )
        self.assertEqual(final_text, "aww thank you 🤍")
        self.assertFalse(preserved)

    def test_no_target_username_uses_proposed_reply_unchanged(self) -> None:
        final_text, preserved = InstagramReplyController._compose_final_text(
            existing_text="@someone",
            proposed_reply="aww thank you 🤍",
            target_username=None,
        )
        self.assertEqual(final_text, "aww thank you 🤍")
        self.assertFalse(preserved)

    def test_case_insensitive_prefix_match_preserves_original_casing(self) -> None:
        final_text, preserved = InstagramReplyController._compose_final_text(
            existing_text="@JoaquimFrancisco.o",
            proposed_reply="aww thank you 🤍",
            target_username="joaquimfrancisco.o",
        )
        self.assertTrue(preserved)
        # The ACTUAL existing text is preserved verbatim, not
        # reconstructed from target_username's casing.
        self.assertEqual(final_text, "@JoaquimFrancisco.o aww thank you 🤍")


# ---------------------------------------------------------------------------
# @mention preservation — mocked Playwright end-to-end (Phase 8D.2)
# ---------------------------------------------------------------------------


class MentionPreservationEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_reply_controller_config()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config.screenshots_dir = Path(self.temp_dir.name) / "screenshots"
        self.config.logs_dir = Path(self.temp_dir.name) / "logs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_at_mention_prefix_preserved_end_to_end(self) -> None:
        selectors = _real_selectors()
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        composer = FakeElement(value=f"@{REAL_USERNAME} ")
        page = _build_page(
            container=container,
            extra_page_children={generic_selector: composer},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        record = _approved_record()
        result = asyncio.run(controller.preview_reply(record))

        expected = f"@{REAL_USERNAME} {record['proposed_reply']}"
        self.assertEqual(composer.filled_text, expected)
        self.assertEqual(result.final_composer_text, expected)
        self.assertEqual(result.existing_composer_text, f"@{REAL_USERNAME} ")
        self.assertTrue(result.mention_preserved)
        self.assertEqual(expected.count(f"@{REAL_USERNAME}"), 1)

    def test_row_scoped_composer_with_existing_mention_is_preserved(self) -> None:
        selectors = _real_selectors()
        row_composer_selector = selectors.row_reply_composer[0]

        composer = FakeElement(value=f"@{REAL_USERNAME}")
        container, row, _reply_button = _base_row_and_container(
            extra_row_children={row_composer_selector: composer}
        )
        page = _build_page(container=container)

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        record = _approved_record()
        result = asyncio.run(controller.preview_reply(record))

        expected = f"@{REAL_USERNAME} {record['proposed_reply']}"
        self.assertEqual(composer.filled_text, expected)
        self.assertTrue(result.mention_preserved)

    def test_no_prefix_uses_proposed_reply_unchanged_end_to_end(self) -> None:
        # _build_success_page's row-scoped composer has no existing value.
        page, _reply_button, composer = _build_success_page()

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        record = _approved_record()
        result = asyncio.run(controller.preview_reply(record))

        self.assertEqual(composer.filled_text, record["proposed_reply"])
        self.assertEqual(result.final_composer_text, record["proposed_reply"])
        self.assertEqual(result.existing_composer_text, "")
        self.assertFalse(result.mention_preserved)

    def test_diagnostic_log_includes_mention_preservation_fields(self) -> None:
        selectors = _real_selectors()
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        composer = FakeElement(value=f"@{REAL_USERNAME} ")
        page = _build_page(
            container=container,
            extra_page_children={generic_selector: composer},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        asyncio.run(controller.preview_reply(_approved_record()))

        log_path = self.config.logs_dir / "reply_controller_diagnostics.jsonl"
        entry = json.loads(
            log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        )

        self.assertEqual(entry["existing_composer_text"], f"@{REAL_USERNAME} ")
        self.assertEqual(
            entry["final_composer_text"],
            f"@{REAL_USERNAME} {_approved_record()['proposed_reply']}",
        )
        self.assertTrue(entry["mention_preserved"])
        self.assertEqual(entry["target_username"], REAL_USERNAME)

    def test_queue_remains_byte_for_byte_unchanged_with_mention_preserved(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            before_bytes = service.queue_path.read_bytes()

            record = load_approved_record(REAL_APPROVAL_ID, approval_service=service)

            selectors = _real_selectors()
            generic_selector = selectors.reply_textbox[0]
            container, row, _reply_button = _base_row_and_container()
            composer = FakeElement(value=f"@{REAL_USERNAME} ")
            page = _build_page(
                container=container,
                extra_page_children={generic_selector: composer},
            )

            controller = InstagramReplyController(
                session=FakeSession(page), config=self.config
            )
            asyncio.run(controller.preview_reply(record))

            after_bytes = service.queue_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)

    def test_no_send_action_exists_with_mention_preserved(self) -> None:
        selectors = _real_selectors()
        generic_selector = selectors.reply_textbox[0]

        container, row, _reply_button = _base_row_and_container()
        composer = FakeElement(value=f"@{REAL_USERNAME} ")
        page = _build_page(
            container=container,
            extra_page_children={generic_selector: composer},
        )

        controller = InstagramReplyController(
            session=FakeSession(page), config=self.config
        )

        record = _approved_record()
        result = asyncio.run(controller.preview_reply(record))

        self.assertFalse(result.sent)
        self.assertIsNone(composer.typed_text)

        for forbidden_name in ("send", "post", "press_enter", "submit"):
            self.assertFalse(hasattr(InstagramReplyController, forbidden_name))


if __name__ == "__main__":
    unittest.main()
