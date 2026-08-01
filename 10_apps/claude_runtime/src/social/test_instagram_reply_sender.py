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
    CommentNotFoundError,
    LoginDetectionConfig,
    ReplyControlNotFoundError,
    ReplyControllerLoginLostError,
    ReplyPostAccessError,
)
from .instagram_reply_controller import ReplyContextNotVerifiedError, load_reply_controller_config
from .instagram_reply_sender import (
    DISALLOWED_ACTIONS,
    InstagramReplySender,
    PostButtonAmbiguousError,
    PostButtonDisabledError,
    PostButtonNotFoundError,
    ReplyPreflightResult,
    ReplySendResult,
    SendClickFailedError,
    SendVerificationFailedError,
    load_reply_sender_config,
    parse_arguments,
)

# None of these tests open a real browser or contact Instagram.
# Playwright is simulated with small fake objects; the approval queue
# is a real temp JSON file. No live send is performed anywhere here.

REAL_APPROVAL_ID = "appr-comment_id-999"
REAL_COMMENT_ID = "999"
REAL_POST_URL = "https://www.instagram.com/p/ABC123/"
REAL_USERNAME = "joaquimfrancisco.o"
REAL_PROPOSED_REPLY = "aww thank you 🤍"


def _approved_record(**overrides) -> dict:
    record = {
        "approval_id": REAL_APPROVAL_ID,
        "stable_id": "comment_id:999",
        "comment_id": REAL_COMMENT_ID,
        "post_url": REAL_POST_URL,
        "username": REAL_USERNAME,
        "comment_text": "beautiful photo!",
        "classification": "compliment",
        "proposed_reply": REAL_PROPOSED_REPLY,
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
# Fake Playwright-shaped objects (same pattern as test_instagram_reply_controller.py)
# ---------------------------------------------------------------------------


class _FakeJSHandle:
    """
    Stand-in for Playwright's JSHandle, as returned by
    ElementHandle.evaluate_handle(). Only as_element() is used by
    production code (InstagramReplySender._locate_composer_container).
    """

    def __init__(self, element: "FakeElement | None") -> None:
        self._element = element

    def as_element(self):
        return self._element


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
        disabled: bool = False,
        fail_is_disabled: bool = False,
        visible: bool = True,
        fail_is_visible: bool = False,
        fail_is_enabled: bool = False,
        container: "FakeElement | None" = None,
        fail_evaluate_handle: bool = False,
    ) -> None:
        self._children = children or {}
        self._child_lists = child_lists or {}
        self._attrs = attrs or {}
        self._text = text
        self._value = value
        self._fail_click = fail_click
        self._disabled = disabled
        self._fail_is_disabled = fail_is_disabled
        self._visible = visible
        self._fail_is_visible = fail_is_visible
        self._fail_is_enabled = fail_is_enabled
        self._container = container
        self._fail_evaluate_handle = fail_evaluate_handle
        self.clicked = False
        self.click_count = 0
        self.filled_text: str | None = None
        self.typed_text: str | None = None

    async def query_selector(self, selector: str):
        return self._children.get(selector)

    async def query_selector_all(self, selector: str):
        if selector in self._child_lists:
            return self._child_lists[selector]
        if selector in self._children:
            return [self._children[selector]]
        return []

    async def get_attribute(self, name: str):
        return self._attrs.get(name)

    async def inner_text(self) -> str:
        return self._text

    async def input_value(self) -> str:
        return self._value or ""

    async def is_disabled(self) -> bool:
        if self._fail_is_disabled:
            raise RuntimeError("simulated is_disabled() failure")
        return self._disabled

    async def is_enabled(self) -> bool:
        if self._fail_is_enabled:
            raise RuntimeError("simulated is_enabled() failure")
        return not self._disabled

    async def is_visible(self) -> bool:
        if self._fail_is_visible:
            raise RuntimeError("simulated is_visible() failure")
        return self._visible

    async def evaluate_handle(self, script: str):
        if self._fail_evaluate_handle:
            raise RuntimeError("simulated evaluate_handle() failure")
        return _FakeJSHandle(self._container)

    async def click(self, timeout=None):
        if self._fail_click:
            raise RuntimeError("simulated click failure")
        self.clicked = True
        self.click_count += 1

    async def fill(self, text: str):
        self.filled_text = text

    async def type(self, text: str):
        self.typed_text = (self.typed_text or "") + text

    # Explicitly absent on purpose: press(), submit(), send() — see
    # DisallowedActionsTests below. If code ever tried to call these,
    # the test would fail with AttributeError, not silently pass.


class FakePage:
    def __init__(
        self,
        *,
        children: dict[str, FakeElement] | None = None,
        child_lists: dict[str, list[FakeElement]] | None = None,
        url: str = REAL_POST_URL,
        goto_error: Exception | None = None,
    ) -> None:
        self._children = children or {}
        self._child_lists = child_lists or {}
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

    async def query_selector_all(self, selector: str):
        if selector in self._child_lists:
            return self._child_lists[selector]
        if selector in self._children:
            return [self._children[selector]]
        return []

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


def _controller_selectors():
    return load_reply_controller_config().selectors


def _sender_selectors():
    return load_reply_sender_config().selectors


def _base_row_and_container(
    *, comment_id: str = REAL_COMMENT_ID, extra_row_children=None
):
    selectors = _controller_selectors()

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

    row = FakeElement(children=children, text=f"{REAL_USERNAME} beautiful photo!")
    container = FakeElement(child_lists={item_selector: [row]})

    return container, row, reply_button


def _build_page(
    *, container: FakeElement, extra_page_children=None, extra_page_child_lists=None
) -> FakePage:
    selectors = _controller_selectors()
    container_selector = selectors.comment_container[0]

    children = {
        "svg[aria-label='Home']": FakeElement(),
        container_selector: container,
    }
    children.update(extra_page_children or {})

    return FakePage(children=children, child_lists=extra_page_child_lists or {})


def _build_success_scenario(
    *,
    comment_id: str = REAL_COMMENT_ID,
    sent_reply_text: str | None = None,
    post_button_missing: bool = False,
    post_button_fails: bool = False,
    post_button_disabled: bool = False,
    post_button_disabled_check_fails: bool = False,
):
    """
    A fully wired success scenario: row matches comment_id, reply
    button works, row-scoped composer verifies context, a Post button
    is present, and (unless sent_reply_text is overridden) the row's
    own text already includes what the final reply will say — so
    post-send verification passes once the expected text is known.
    """
    controller_selectors = _controller_selectors()
    sender_selectors = _sender_selectors()

    row_composer_selector = controller_selectors.row_reply_composer[0]
    post_button_selector = sender_selectors.post_button[0]

    composer = FakeElement()

    row_text = sent_reply_text or (
        f"{REAL_USERNAME} beautiful photo! @{REAL_USERNAME} {REAL_PROPOSED_REPLY}"
    )

    container, row, reply_button = _base_row_and_container(
        comment_id=comment_id,
        extra_row_children={row_composer_selector: composer},
    )
    row._text = row_text

    page_children = {}

    if not post_button_missing:
        page_children[post_button_selector] = FakeElement(
            fail_click=post_button_fails,
            disabled=post_button_disabled,
            fail_is_disabled=post_button_disabled_check_fails,
        )

    page = _build_page(container=container, extra_page_children=page_children)

    return page, reply_button, composer


def _build_container_scoped_scenario(
    *,
    comment_id: str = REAL_COMMENT_ID,
    container_candidates: list[FakeElement] | None = None,
    page_decoy: FakeElement | None = None,
):
    """
    Scenario for Phase 8E.2 container-scoping tests: the verified
    composer's evaluate_handle('...closest(form)...') resolves to its
    own wrapping container element, distinct from the page. The real
    Post button candidate(s) live inside that container; an unrelated
    page-level decoy (standing in for a hidden <title> or a "Boost
    post" button — anything sharing the same selector string but NOT
    belonging to the verified composer) is registered directly on the
    page and must be ignored whenever the container is successfully
    located.
    """
    controller_selectors = _controller_selectors()
    sender_selectors = _sender_selectors()
    post_button_selector = sender_selectors.post_button[0]

    row_composer_selector = controller_selectors.row_reply_composer[0]

    container_element = FakeElement(
        child_lists={post_button_selector: container_candidates or []}
    )
    composer = FakeElement(container=container_element)

    row_text = f"{REAL_USERNAME} beautiful photo! @{REAL_USERNAME} {REAL_PROPOSED_REPLY}"

    container, row, reply_button = _base_row_and_container(
        comment_id=comment_id,
        extra_row_children={row_composer_selector: composer},
    )
    row._text = row_text

    page_children = {}

    if page_decoy is not None:
        page_children[post_button_selector] = page_decoy

    page = _build_page(container=container, extra_page_children=page_children)

    return page, reply_button, composer, container_element


class SenderTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.controller_config = load_reply_controller_config()
        self.sender_config = load_reply_sender_config()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.controller_config.screenshots_dir = Path(self.temp_dir.name) / "screenshots"
        self.controller_config.logs_dir = Path(self.temp_dir.name) / "logs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _sender(self, page: FakePage) -> InstagramReplySender:
        return InstagramReplySender(
            session=FakeSession(page),
            controller_config=self.controller_config,
            sender_config=self.sender_config,
        )


# ---------------------------------------------------------------------------
# CLI argument requirements
# ---------------------------------------------------------------------------


class CliArgumentTests(unittest.TestCase):
    def test_confirm_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--send-approved", REAL_APPROVAL_ID])

    def test_send_approved_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--confirm"])

    def test_both_flags_together_parse(self) -> None:
        arguments = parse_arguments(
            ["--send-approved", REAL_APPROVAL_ID, "--confirm"]
        )
        self.assertEqual(arguments.send_approved, REAL_APPROVAL_ID)
        self.assertTrue(arguments.confirm)

    def test_preflight_does_not_require_confirm(self) -> None:
        arguments = parse_arguments(["--preflight", REAL_APPROVAL_ID])
        self.assertEqual(arguments.preflight, REAL_APPROVAL_ID)
        self.assertIsNone(arguments.send_approved)
        self.assertFalse(arguments.confirm)

    def test_preflight_and_send_approved_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(
                ["--send-approved", REAL_APPROVAL_ID, "--preflight", REAL_APPROVAL_ID]
            )

    def test_neither_flag_raises(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments([])


# ---------------------------------------------------------------------------
# send_reply — success path
# ---------------------------------------------------------------------------


class SendReplySuccessTests(SenderTestBase):
    def test_approved_record_can_send(self) -> None:
        page, reply_button, composer = _build_success_scenario()
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            record = _approved_record()

            result = asyncio.run(
                sender.send_reply(record, approval_service=service)
            )

            self.assertIsInstance(result, ReplySendResult)
            self.assertTrue(result.reply_context_verified)
            self.assertTrue(result.send_clicked)
            self.assertTrue(result.send_verified)
            self.assertTrue(result.sent)
            self.assertEqual(result.status, "sent")
            self.assertTrue(reply_button.clicked)

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "sent")
            self.assertIsNotNone(stored["sent_at"])
            self.assertIsNone(stored["error"])

    def test_mention_preserved_exactly_once(self) -> None:
        controller_selectors = _controller_selectors()
        sender_selectors = _sender_selectors()

        composer = FakeElement(value=f"@{REAL_USERNAME} ")
        container, row, _reply_button = _base_row_and_container()
        row._text = f"{REAL_USERNAME} @{REAL_USERNAME} {REAL_PROPOSED_REPLY}"

        page = _build_page(
            container=container,
            extra_page_children={
                controller_selectors.reply_textbox[0]: composer,
                sender_selectors.post_button[0]: FakeElement(),
            },
        )

        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            result = asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            expected = f"@{REAL_USERNAME} {REAL_PROPOSED_REPLY}"
            self.assertEqual(result.final_reply_text, expected)
            self.assertTrue(result.mention_preserved)
            self.assertEqual(composer.filled_text, expected)
            self.assertEqual(expected.count(f"@{REAL_USERNAME}"), 1)

    def test_before_and_after_send_screenshots_are_saved(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            result = asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            names = {Path(p).name for p in page.screenshot_calls}
            self.assertIn(f"before_send_{REAL_APPROVAL_ID}.png", names)
            self.assertIn(f"after_send_{REAL_APPROVAL_ID}.png", names)
            self.assertEqual(
                Path(result.before_send_screenshot_path).name,
                f"before_send_{REAL_APPROVAL_ID}.png",
            )
            self.assertEqual(
                Path(result.after_send_screenshot_path).name,
                f"after_send_{REAL_APPROVAL_ID}.png",
            )

    def test_only_one_post_click_occurs(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender_selectors = _sender_selectors()
        post_button = page._children[sender_selectors.post_button[0]]

        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            self.assertEqual(post_button.click_count, 1)

    def test_audit_log_written_with_required_fields(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

        log_path = self.controller_config.logs_dir / "reply_sender_audit.jsonl"
        entry = json.loads(
            log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        )

        required_fields = {
            "approval_id",
            "stable_id",
            "post_url",
            "comment_id",
            "username",
            "final_reply_text",
            "reply_context_method",
            "before_screenshot",
            "after_screenshot",
            "send_clicked",
            "send_verified",
            "result",
            "error",
            "created_at",
            "sent_at",
        }
        self.assertTrue(required_fields.issubset(entry.keys()))
        self.assertEqual(entry["result"], "sent")
        self.assertTrue(entry["send_clicked"])
        self.assertTrue(entry["send_verified"])
        self.assertIsNone(entry["error"])
        self.assertIsNotNone(entry["sent_at"])


# ---------------------------------------------------------------------------
# Refused statuses (non-approved / already-sent)
# ---------------------------------------------------------------------------


class RefusedStatusTests(unittest.TestCase):
    def _service(self, temp_dir: str, **overrides) -> tuple[ApprovalService, dict]:
        record = _approved_record(**overrides)
        service = _write_queue(temp_dir, [record])
        return service, record

    def test_pending_approval_is_refused(self) -> None:
        from .instagram_reply_controller import load_approved_record

        with tempfile.TemporaryDirectory() as temp_dir:
            service, _ = self._service(temp_dir, status="pending_approval")

            with self.assertRaises(ApprovalRecordNotApprovedError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_rejected_is_refused(self) -> None:
        from .instagram_reply_controller import load_approved_record

        with tempfile.TemporaryDirectory() as temp_dir:
            service, _ = self._service(temp_dir, status="rejected")

            with self.assertRaises(ApprovalRecordNotApprovedError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_skipped_is_refused(self) -> None:
        from .instagram_reply_controller import load_approved_record

        with tempfile.TemporaryDirectory() as temp_dir:
            service, _ = self._service(
                temp_dir, status="skipped", proposed_reply=None
            )

            with self.assertRaises(ApprovalRecordNotApprovedError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_already_sent_cannot_send_twice(self) -> None:
        from .instagram_reply_controller import load_approved_record

        with tempfile.TemporaryDirectory() as temp_dir:
            service, _ = self._service(
                temp_dir, status="sent", sent_at="2026-07-31T00:10:00Z"
            )

            with self.assertRaises(ApprovalRecordNotApprovedError):
                load_approved_record(REAL_APPROVAL_ID, approval_service=service)

    def test_missing_username_is_refused_by_sender(self) -> None:
        controller_config = load_reply_controller_config()
        sender_config = load_reply_sender_config()

        page, _reply_button, _composer = _build_success_scenario()
        sender = InstagramReplySender(
            session=FakeSession(page),
            controller_config=controller_config,
            sender_config=sender_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            service = _write_queue(temp_dir, [_approved_record(username=None)])
            record = _approved_record(username=None)

            with self.assertRaises(ApprovalRecordInvalidError):
                asyncio.run(sender.send_reply(record, approval_service=service))


# ---------------------------------------------------------------------------
# Reply context verification gates the send action
# ---------------------------------------------------------------------------


class ContextVerificationGatesSendTests(SenderTestBase):
    def test_unverified_context_refuses_to_send(self) -> None:
        controller_selectors = _controller_selectors()
        sender_selectors = _sender_selectors()

        # Only a generic page-level composer with no context signal at
        # all, plus a Post button that must never be reached.
        container, row, _reply_button = _base_row_and_container()
        post_button = FakeElement()

        page = _build_page(
            container=container,
            extra_page_children={
                controller_selectors.reply_textbox[0]: FakeElement(value=""),
                sender_selectors.post_button[0]: post_button,
            },
        )

        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(ReplyContextNotVerifiedError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            self.assertFalse(post_button.clicked)

            # The queue record itself is untouched apart from the
            # failure being persisted (Phase 8E.2, requirement C):
            # status/sent_at are unaffected, only error changes.
            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")
            self.assertIsNone(stored["sent_at"])
            self.assertIsNotNone(stored["error"])

    def test_verification_failure_leaves_status_approved(self) -> None:
        controller_selectors = _controller_selectors()
        sender_selectors = _sender_selectors()

        container, row, _reply_button = _base_row_and_container()

        page = _build_page(
            container=container,
            extra_page_children={
                controller_selectors.reply_textbox[0]: FakeElement(value=""),
                sender_selectors.post_button[0]: FakeElement(),
            },
        )

        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(ReplyContextNotVerifiedError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")
            self.assertIsNone(stored["sent_at"])


# ---------------------------------------------------------------------------
# Send-specific failure paths
# ---------------------------------------------------------------------------


class SendFailureTests(SenderTestBase):
    def test_post_button_not_found_raises(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_missing=True
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(PostButtonNotFoundError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")

    def test_post_click_failure_raises_and_leaves_status_approved(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_fails=True
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(SendClickFailedError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            # status/sent_at are unaffected; only error is persisted
            # (Phase 8E.2, requirement C) — see FailurePersistenceTests
            # for the dedicated coverage of that behavior.
            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")
            self.assertIsNone(stored["sent_at"])

    def test_send_verification_failure_raises_and_leaves_status_approved(
        self,
    ) -> None:
        # Post click succeeds, but the row never shows any evidence of
        # the new reply having landed.
        page, _reply_button, _composer = _build_success_scenario(
            sent_reply_text="nothing relevant here"
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(SendVerificationFailedError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")
            self.assertIsNone(stored["sent_at"])

    def test_login_lost_raises(self) -> None:
        controller_selectors = _controller_selectors()
        page = FakePage(children={"input[type='password']": FakeElement()})
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(ReplyControllerLoginLostError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

    def test_post_cannot_open_raises(self) -> None:
        page = FakePage(goto_error=RuntimeError("navigation failed"))
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(ReplyPostAccessError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

    def test_exact_comment_not_found_raises(self) -> None:
        controller_selectors = _controller_selectors()
        item_selector = controller_selectors.comment_item[0]

        other_row = FakeElement(
            children={
                controller_selectors.comment_permalink_template.format(
                    comment_id="not-the-one"
                ): FakeElement()
            }
        )
        container = FakeElement(child_lists={item_selector: [other_row]})
        page = _build_page(container=container)

        self.controller_config.max_scroll_attempts = 1
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(CommentNotFoundError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

    def test_reply_control_not_found_raises(self) -> None:
        controller_selectors = _controller_selectors()
        item_selector = controller_selectors.comment_item[0]
        permalink_selector = controller_selectors.comment_permalink_template.format(
            comment_id=REAL_COMMENT_ID
        )

        row = FakeElement(children={permalink_selector: FakeElement()})
        container = FakeElement(child_lists={item_selector: [row]})
        page = _build_page(container=container)

        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(ReplyControlNotFoundError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

    def test_failure_paths_still_write_an_audit_entry(self) -> None:
        page = FakePage(goto_error=RuntimeError("boom"))
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(ReplyPostAccessError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

        log_path = self.controller_config.logs_dir / "reply_sender_audit.jsonl"
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        self.assertEqual(entry["result"], "failed")
        self.assertIsNotNone(entry["error"])


# ---------------------------------------------------------------------------
# Atomic queue write
# ---------------------------------------------------------------------------


class AtomicQueueWriteTests(SenderTestBase):
    def test_queue_updates_only_after_confirmed_success(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            before_bytes = service.queue_path.read_bytes()

            asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            after_bytes = service.queue_path.read_bytes()
            self.assertNotEqual(before_bytes, after_bytes)

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "sent")

    def test_no_tmp_file_left_behind_after_send(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            leftover_tmp_files = list(Path(queue_dir).glob("*.tmp"))
            self.assertEqual(leftover_tmp_files, [])

    def test_queue_content_is_valid_json_after_send(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            data = json.loads(service.queue_path.read_text(encoding="utf-8"))
            self.assertIn("approvals", data)
            self.assertEqual(len(data["approvals"]), 1)
            self.assertEqual(data["approvals"][0]["status"], "sent")


# ---------------------------------------------------------------------------
# Strict send safety
# ---------------------------------------------------------------------------


class SendSafetyTests(SenderTestBase):
    def test_no_send_via_enter_key_path_exists(self) -> None:
        """
        FakeElement has no press()/keyboard-press capability at all —
        if send_reply() ever tried to press Enter, this would raise
        AttributeError rather than silently succeeding.
        """
        for forbidden_name in ("press", "press_enter", "keyboard"):
            self.assertFalse(
                hasattr(FakeElement, forbidden_name),
                "test fixture sanity check",
            )
            self.assertFalse(
                hasattr(InstagramReplySender, forbidden_name),
                f"InstagramReplySender must not implement '{forbidden_name}'",
            )

    def test_no_mutating_action_methods_exist_on_sender(self) -> None:
        for action in DISALLOWED_ACTIONS:
            self.assertFalse(
                hasattr(InstagramReplySender, action),
                f"InstagramReplySender must not implement '{action}'",
            )

        for forbidden_name in (
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
                hasattr(InstagramReplySender, forbidden_name),
                f"InstagramReplySender must not implement '{forbidden_name}'",
            )

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("press_enter", DISALLOWED_ACTIONS)
        self.assertIn("batch_send", DISALLOWED_ACTIONS)
        self.assertIn("watch", DISALLOWED_ACTIONS)
        self.assertIn("retry_send", DISALLOWED_ACTIONS)

    def test_context_and_playwright_cleaned_up_on_success(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        session = FakeSession(page)
        sender = InstagramReplySender(
            session=session,
            controller_config=self.controller_config,
            sender_config=self.sender_config,
        )

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

        self.assertTrue(session._context.closed)
        self.assertTrue(session._playwright.stopped)

    def test_context_and_playwright_cleaned_up_on_failure(self) -> None:
        page = FakePage(goto_error=RuntimeError("boom"))
        session = FakeSession(page)
        sender = InstagramReplySender(
            session=session,
            controller_config=self.controller_config,
            sender_config=self.sender_config,
        )

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(ReplyPostAccessError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

        self.assertTrue(session._context.closed)
        self.assertTrue(session._playwright.stopped)


# ---------------------------------------------------------------------------
# preflight — Phase 8E.1, never sends
# ---------------------------------------------------------------------------


class PreflightTests(SenderTestBase):
    def _sender_no_approval_service(self, page: FakePage) -> InstagramReplySender:
        # preflight() never accepts an ApprovalService reference at
        # all — structurally incapable of writing to the queue.
        return self._sender(page)

    def test_post_button_discovered_without_clicking(self) -> None:
        page, reply_button, _composer = _build_success_scenario()
        sender_selectors = _sender_selectors()
        post_button = page._children[sender_selectors.post_button[0]]

        sender = self._sender_no_approval_service(page)

        result = asyncio.run(sender.preflight(_approved_record()))

        self.assertIsInstance(result, ReplyPreflightResult)
        self.assertTrue(result.post_button_found)
        self.assertTrue(result.post_button_enabled)
        self.assertFalse(result.sent)

        # The defining property of preflight: the button was located,
        # but never clicked.
        self.assertFalse(post_button.clicked)
        self.assertEqual(post_button.click_count, 0)
        self.assertTrue(reply_button.clicked)  # Reply IS clicked, Post is not

    def test_disabled_post_button_fails_preflight(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_disabled=True
        )
        sender_selectors = _sender_selectors()
        post_button = page._children[sender_selectors.post_button[0]]

        sender = self._sender_no_approval_service(page)

        with self.assertRaises(PostButtonDisabledError):
            asyncio.run(sender.preflight(_approved_record()))

        self.assertFalse(post_button.clicked)

    def test_missing_post_button_fails_preflight(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_missing=True
        )
        sender = self._sender_no_approval_service(page)

        with self.assertRaises(PostButtonNotFoundError):
            asyncio.run(sender.preflight(_approved_record()))

    def test_sent_reply_verification_container_discovered(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender_no_approval_service(page)

        result = asyncio.run(sender.preflight(_approved_record()))

        self.assertTrue(result.sent_reply_container_found)

    def test_reply_context_verified_and_mention_preserved_reported(self) -> None:
        controller_selectors = _controller_selectors()
        sender_selectors = _sender_selectors()

        composer = FakeElement(value=f"@{REAL_USERNAME} ")
        container, row, _reply_button = _base_row_and_container()
        row._text = f"{REAL_USERNAME} @{REAL_USERNAME} {REAL_PROPOSED_REPLY}"

        page = _build_page(
            container=container,
            extra_page_children={
                controller_selectors.reply_textbox[0]: composer,
                sender_selectors.post_button[0]: FakeElement(),
            },
        )

        sender = self._sender_no_approval_service(page)
        result = asyncio.run(sender.preflight(_approved_record()))

        self.assertTrue(result.reply_context_verified)
        self.assertTrue(result.mention_preserved)
        self.assertEqual(
            result.final_reply_text, f"@{REAL_USERNAME} {REAL_PROPOSED_REPLY}"
        )
        self.assertEqual(composer.filled_text, result.final_reply_text)
        self.assertEqual(result.reply_target_username, REAL_USERNAME)

    def test_unverified_context_fails_preflight_before_locating_post_button(
        self,
    ) -> None:
        controller_selectors = _controller_selectors()
        sender_selectors = _sender_selectors()

        container, row, _reply_button = _base_row_and_container()
        post_button = FakeElement()

        page = _build_page(
            container=container,
            extra_page_children={
                controller_selectors.reply_textbox[0]: FakeElement(value=""),
                sender_selectors.post_button[0]: post_button,
            },
        )

        sender = self._sender_no_approval_service(page)

        with self.assertRaises(ReplyContextNotVerifiedError):
            asyncio.run(sender.preflight(_approved_record()))

        self.assertFalse(post_button.clicked)

    def test_screenshot_saved_with_approval_id_in_filename(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender_no_approval_service(page)

        result = asyncio.run(sender.preflight(_approved_record()))

        self.assertEqual(
            Path(result.screenshot_path).name,
            f"preflight_{REAL_APPROVAL_ID}.png",
        )

    def test_diagnostic_log_includes_selectors_and_results(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender_no_approval_service(page)

        asyncio.run(sender.preflight(_approved_record()))

        log_path = self.controller_config.logs_dir / "reply_sender_audit.jsonl"
        entry = json.loads(
            log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        )

        self.assertEqual(entry["result"], "preflight_success")
        self.assertTrue(entry["post_button_found"])
        self.assertTrue(entry["post_button_enabled"])
        self.assertIsNotNone(entry["post_button_selector"])
        self.assertTrue(entry["sent_reply_container_found"])
        self.assertIn("sent_reply_item_candidate_count", entry)
        self.assertFalse(entry["send_clicked"])
        self.assertIsNotNone(entry["screenshot"])
        self.assertIsNone(entry["error"])

    def test_queue_remains_unchanged_after_preflight(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender_no_approval_service(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            before_bytes = service.queue_path.read_bytes()

            asyncio.run(sender.preflight(_approved_record()))

            after_bytes = service.queue_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)

    def test_queue_remains_unchanged_after_failed_preflight(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_disabled=True
        )
        sender = self._sender_no_approval_service(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            before_bytes = service.queue_path.read_bytes()

            with self.assertRaises(PostButtonDisabledError):
                asyncio.run(sender.preflight(_approved_record()))

            after_bytes = service.queue_path.read_bytes()
            self.assertEqual(before_bytes, after_bytes)

    def test_send_clicked_remains_false_on_success(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender_no_approval_service(page)

        result = asyncio.run(sender.preflight(_approved_record()))

        self.assertFalse(result.sent)

        log_path = self.controller_config.logs_dir / "reply_sender_audit.jsonl"
        entry = json.loads(
            log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        )
        self.assertFalse(entry["send_clicked"])

    def test_send_clicked_remains_false_on_failure(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_missing=True
        )
        sender = self._sender_no_approval_service(page)

        with self.assertRaises(PostButtonNotFoundError):
            asyncio.run(sender.preflight(_approved_record()))

        log_path = self.controller_config.logs_dir / "reply_sender_audit.jsonl"
        entry = json.loads(
            log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        )
        self.assertFalse(entry["send_clicked"])

    def test_preflight_never_holds_an_approval_service_reference(self) -> None:
        """
        Structural guarantee, not just behavioral: preflight()'s
        signature accepts only a record, never an ApprovalService, so
        it is impossible for it to call save_queue().
        """
        import inspect

        signature = inspect.signature(InstagramReplySender.preflight)
        self.assertNotIn("approval_service", signature.parameters)

    def test_context_and_playwright_cleaned_up_after_preflight(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        session = FakeSession(page)
        sender = InstagramReplySender(
            session=session,
            controller_config=self.controller_config,
            sender_config=self.sender_config,
        )

        asyncio.run(sender.preflight(_approved_record()))

        self.assertTrue(session._context.closed)
        self.assertTrue(session._playwright.stopped)

    def test_missing_username_is_refused_by_preflight(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender_no_approval_service(page)

        with self.assertRaises(ApprovalRecordInvalidError):
            asyncio.run(
                sender.preflight(_approved_record(username=None))
            )


# ---------------------------------------------------------------------------
# Phase 8E.2 — Post-button selection: container scoping, exactly-one
# -candidate enforcement, and rejection of the previously-ambiguous
# text=Post matches (hidden <title>, "Boost post", wrong composer).
# ---------------------------------------------------------------------------


class PostButtonSelectionTests(SenderTestBase):
    def test_hidden_post_candidate_is_rejected(self) -> None:
        """
        Models the real send failure this phase fixes: a single
        selector match that is not actually visible (e.g. a hidden
        <title> element) must never be treated as the Post button.
        """
        sender_selectors = _sender_selectors()
        post_button_selector = sender_selectors.post_button[0]

        composer = FakeElement()
        row_composer_selector = _controller_selectors().row_reply_composer[0]
        container, row, _reply_button = _base_row_and_container(
            extra_row_children={row_composer_selector: composer}
        )
        row._text = f"{REAL_USERNAME} beautiful photo! @{REAL_USERNAME} {REAL_PROPOSED_REPLY}"

        hidden_candidate = FakeElement(visible=False)
        page = _build_page(
            container=container,
            extra_page_children={post_button_selector: hidden_candidate},
        )
        sender = self._sender(page)

        with self.assertRaises(PostButtonDisabledError):
            asyncio.run(sender.preflight(_approved_record()))

        self.assertFalse(hidden_candidate.clicked)

    def test_boost_post_decoy_is_ignored(self) -> None:
        """
        A "Boost post" control has a different accessible name from
        the real Post control, so it is registered under a selector
        string that is NOT in config's post_button list at all — it
        must never be found or clicked, and must not interfere with
        finding the real button.
        """
        sender_selectors = _sender_selectors()
        post_button_selector = sender_selectors.post_button[0]
        boost_post_decoy = FakeElement()

        page, reply_button, _composer = _build_success_scenario()
        page._children['role=button[name="Boost post"]'] = boost_post_decoy

        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            result = asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            self.assertTrue(result.sent)
            self.assertFalse(boost_post_decoy.clicked)

    def test_wrong_composer_button_is_rejected_in_favour_of_container_scoped_one(
        self,
    ) -> None:
        real_button = FakeElement()
        page_level_decoy = FakeElement()

        page, reply_button, composer, container_element = (
            _build_container_scoped_scenario(
                container_candidates=[real_button],
                page_decoy=page_level_decoy,
            )
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            result = asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            self.assertTrue(result.sent)
            self.assertTrue(real_button.clicked)
            self.assertFalse(page_level_decoy.clicked)

    def test_exact_composer_scoped_button_passes_and_reports_container_method(
        self,
    ) -> None:
        real_button = FakeElement()

        page, _reply_button, _composer, _container_element = (
            _build_container_scoped_scenario(container_candidates=[real_button])
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            result = asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            self.assertTrue(result.sent)
            self.assertEqual(result.post_button_selector_method, "composer_container")
            self.assertEqual(result.post_button_candidate_count, 1)
            self.assertTrue(result.post_button_visible)
            self.assertTrue(result.post_button_enabled)
            self.assertTrue(result.post_button_interactive)

    def test_multiple_valid_candidates_fail_safely(self) -> None:
        first_candidate = FakeElement()
        second_candidate = FakeElement()

        page, _reply_button, _composer, _container_element = (
            _build_container_scoped_scenario(
                container_candidates=[first_candidate, second_candidate]
            )
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(PostButtonAmbiguousError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            self.assertFalse(first_candidate.clicked)
            self.assertFalse(second_candidate.clicked)

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")
            self.assertIsNone(stored["sent_at"])
            self.assertIsNotNone(stored["error"])

    def test_multiple_valid_candidates_fail_safely_in_preflight(self) -> None:
        first_candidate = FakeElement()
        second_candidate = FakeElement()

        page, _reply_button, _composer, _container_element = (
            _build_container_scoped_scenario(
                container_candidates=[first_candidate, second_candidate]
            )
        )
        sender = self._sender(page)

        with self.assertRaises(PostButtonAmbiguousError):
            asyncio.run(sender.preflight(_approved_record()))

        self.assertFalse(first_candidate.clicked)
        self.assertFalse(second_candidate.clicked)

    def test_result_and_audit_include_new_post_button_fields(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])
            result = asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

        self.assertIsNotNone(result.post_button_selector_method)
        self.assertEqual(result.post_button_candidate_count, 1)
        self.assertTrue(result.post_button_visible)
        self.assertTrue(result.post_button_enabled)
        self.assertTrue(result.post_button_interactive)

        log_path = self.controller_config.logs_dir / "reply_sender_audit.jsonl"
        entry = json.loads(
            log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        )

        for key in (
            "post_button_selector_method",
            "post_button_candidate_count",
            "post_button_visible",
            "post_button_interactive",
        ):
            self.assertIn(key, entry)


# ---------------------------------------------------------------------------
# Phase 8E.2 — send-failure persistence into approval_queue.json
# ---------------------------------------------------------------------------


class FailurePersistenceTests(SenderTestBase):
    def test_send_failure_writes_error_and_keeps_status_approved(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_fails=True
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(SendClickFailedError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")
            self.assertIsNone(stored["sent_at"])
            self.assertIsNotNone(stored["error"])
            self.assertNotEqual(stored["status"], "failed")

    def test_post_button_not_found_also_persists_error(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_missing=True
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(PostButtonNotFoundError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")
            self.assertIsNotNone(stored["error"])

    def test_verification_failure_also_persists_error(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            sent_reply_text="nothing relevant here"
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(SendVerificationFailedError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "approved")
            self.assertIsNotNone(stored["error"])

    def test_successful_send_clears_a_previous_error(self) -> None:
        page, _reply_button, _composer = _build_success_scenario()
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(
                queue_dir, [_approved_record(error="a previous failure")]
            )

            result = asyncio.run(
                sender.send_reply(_approved_record(), approval_service=service)
            )

            self.assertTrue(result.sent)
            stored = service.get(REAL_APPROVAL_ID)
            self.assertEqual(stored["status"], "sent")
            self.assertIsNone(stored["error"])

    def test_failed_send_queue_write_is_atomic_and_leaves_no_tmp_file(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_fails=True
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(SendClickFailedError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            leftover_tmp_files = list(Path(queue_dir).glob("*.tmp"))
            self.assertEqual(leftover_tmp_files, [])

            data = json.loads(service.queue_path.read_text(encoding="utf-8"))
            self.assertIn("approvals", data)
            self.assertEqual(data["approvals"][0]["status"], "approved")

    def test_failed_send_never_writes_literal_failed_status(self) -> None:
        page, _reply_button, _composer = _build_success_scenario(
            post_button_missing=True
        )
        sender = self._sender(page)

        with tempfile.TemporaryDirectory() as queue_dir:
            service = _write_queue(queue_dir, [_approved_record()])

            with self.assertRaises(PostButtonNotFoundError):
                asyncio.run(
                    sender.send_reply(_approved_record(), approval_service=service)
                )

            data = json.loads(service.queue_path.read_text(encoding="utf-8"))
            statuses = {entry["status"] for entry in data["approvals"]}
            self.assertNotIn("failed", statuses)


if __name__ == "__main__":
    unittest.main()
