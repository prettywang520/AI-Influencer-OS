from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .approval_service import ApprovalService, build_approval_service
from .instagram_models import (
    ApprovalRecordInvalidError,
    ApprovalRecordNotApprovedError,
    ApprovalRecordNotFoundError,
    CommentNotFoundError,
    InstagramConfigError,
    InstagramSessionError,
    ReplyControlNotFoundError,
    ReplyControllerError,
    ReplyControllerLoginLostError,
    ReplyPostAccessError,
    ReplyTextboxNotFoundError,
)
from .instagram_session import (
    InstagramSession,
    PlaywrightTimeoutError,
    load_config as load_session_config,
)

# Phase 8D is a STRICT DRY-RUN PREVIEW ONLY.
#
# This module fills the reply textbox for a human to visually verify,
# then stops. It must never grow methods for any of the actions below,
# and never presses Enter or clicks Post/Send.
DISALLOWED_ACTIONS = (
    "send",
    "post",
    "press_enter",
    "like",
    "follow",
    "unfollow",
    "hide",
    "delete",
    "report",
    "send_dm",
)

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "social" / "instagram.yaml"

REQUIRED_APPROVAL_FIELDS = (
    "approval_id",
    "comment_id",
    "post_url",
    "proposed_reply",
)


# ---------------------------------------------------------------------------
# Local types (Phase 8D.1 adds fields that instagram_models.py's shared
# ReplyControllerSelectors/ReplyControllerConfig/ReplyPreviewResult do
# not have; this phase's file list does not include instagram_models.py,
# so these are defined locally here instead of extending the shared
# dataclasses).
# ---------------------------------------------------------------------------


class ReplyContextNotVerifiedError(ReplyControllerError):
    """
    Raised when, after clicking the row's Reply control, no reliable
    signal confirms the textbox is genuinely a threaded reply to the
    exact target comment (as opposed to the generic "Add a comment"
    field). The textbox is never filled in this case.
    """


@dataclass(slots=True)
class ReplyControllerSelectors:
    """
    Centralised CSS/Playwright selectors for the reply controller.

    comment_container / comment_item are reused from
    comment_reader.selectors (Phase 8B, validated against a real
    post). comment_permalink_template, reply_button, and
    reply_textbox are from Phase 8D and are untested placeholder
    defaults — tune against a real approved record using the
    screenshot and diagnostic log this module writes on failure.

    row_reply_composer, replying_to_text, and
    comment_context_attribute_template are Phase 8D.1 additions used
    to verify a genuine threaded-reply context before anything is
    filled; also untested placeholder defaults.
    """

    comment_container: list[str] = field(default_factory=list)
    comment_item: list[str] = field(default_factory=list)
    comment_permalink_template: str = "a[href*='/c/{comment_id}/']"
    reply_button: list[str] = field(default_factory=list)
    reply_textbox: list[str] = field(default_factory=list)
    row_reply_composer: list[str] = field(default_factory=list)
    replying_to_text: list[str] = field(default_factory=list)
    comment_context_attribute_template: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplyControllerConfig:
    """
    Fully-resolved runtime configuration for the reply controller.

    Built by load_reply_controller_config() from
    config/social/instagram.yaml.
    """

    selectors: ReplyControllerSelectors
    max_scroll_attempts: int
    scroll_pause_ms: int
    search_timeout_ms: int
    post_navigation_timeout_ms: int
    post_fill_wait_ms: int
    screenshots_dir: Path
    logs_dir: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["screenshots_dir"] = str(self.screenshots_dir)
        data["logs_dir"] = str(self.logs_dir)
        return data


@dataclass(slots=True)
class ReplyPreviewResult:
    """
    Summary of one instagram_reply_controller --dry-run run.

    sent is always False: this module never submits a reply.
    reply_context_verified/method/target_username are Phase 8D.1
    additions describing the threaded-reply verification outcome.
    existing_composer_text/final_composer_text/mention_preserved are
    Phase 8D.2 additions describing whether Instagram's auto-inserted
    @username mention was preserved rather than overwritten.
    """

    login_status: str
    approval_id: str
    comment_found: bool
    reply_control_clicked: bool
    textbox_filled: bool
    sent: bool
    reply_context_verified: bool = False
    reply_context_method: str | None = None
    reply_target_username: str | None = None
    existing_composer_text: str | None = None
    final_composer_text: str | None = None
    mention_preserved: bool = False
    screenshot_path: str | None = None
    selector_used: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _runtime_root() -> Path:
    """
    instagram_reply_controller.py location:

    10_apps/claude_runtime/src/social/instagram_reply_controller.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def load_reply_controller_config(
    *,
    config_path: str | Path | None = None,
) -> ReplyControllerConfig:
    """
    Load the reply_controller section from config/social/instagram.yaml.

    This is a deliberately independent loader: instagram_session.py is
    not modified by Phase 8D, so this function reads the same YAML
    file on its own rather than reaching into that module's loader
    (matching the pattern already used by instagram_comment_reader.py).
    """
    runtime_root = _runtime_root()

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (runtime_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve()
    )

    if not resolved_config_path.exists():
        raise InstagramConfigError(
            f"Instagram session config not found: {resolved_config_path}"
        )

    try:
        with resolved_config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise InstagramConfigError(
            f"Invalid YAML in {resolved_config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict) or not raw:
        raise InstagramConfigError(
            f"Instagram session config is empty or invalid: {resolved_config_path}"
        )

    section = raw.get("reply_controller") or {}
    selectors_section = section.get("selectors") or {}
    output_section = section.get("output") or {}
    timeouts_section = section.get("timeouts") or {}

    selectors = ReplyControllerSelectors(
        comment_container=list(selectors_section.get("comment_container", [])),
        comment_item=list(selectors_section.get("comment_item", [])),
        comment_permalink_template=str(
            selectors_section.get(
                "comment_permalink_template",
                "a[href*='/c/{comment_id}/']",
            )
        ),
        reply_button=list(selectors_section.get("reply_button", [])),
        reply_textbox=list(selectors_section.get("reply_textbox", [])),
        row_reply_composer=list(
            selectors_section.get("row_reply_composer", [])
        ),
        replying_to_text=list(selectors_section.get("replying_to_text", [])),
        comment_context_attribute_template=list(
            selectors_section.get("comment_context_attribute_template", [])
        ),
    )

    screenshots_dir = str(
        output_section.get("screenshots_dir", "output/community/screenshots")
    )
    logs_dir = str(output_section.get("logs_dir", "output/community/logs"))

    return ReplyControllerConfig(
        selectors=selectors,
        max_scroll_attempts=int(timeouts_section.get("max_scroll_attempts", 15)),
        scroll_pause_ms=int(timeouts_section.get("scroll_pause_ms", 1200)),
        search_timeout_ms=int(timeouts_section.get("search_timeout_ms", 60000)),
        post_navigation_timeout_ms=int(
            timeouts_section.get("post_navigation_timeout_ms", 30000)
        ),
        post_fill_wait_ms=int(timeouts_section.get("post_fill_wait_ms", 5000)),
        screenshots_dir=(runtime_root / screenshots_dir).resolve(),
        logs_dir=(runtime_root / logs_dir).resolve(),
    )


def load_approved_record(
    approval_id: str,
    *,
    approval_service: ApprovalService,
) -> dict[str, Any]:
    """
    Look up one approval record and validate it is ready to preview.

    Pure validation against the (read-only) approval queue — no
    browser is involved. Raises the specific ReplyControllerError
    subclass for each failure so callers can handle them clearly:
    approval ID not found, record not approved, or missing a required
    field.
    """
    record = approval_service.get(approval_id)

    if record is None:
        raise ApprovalRecordNotFoundError(
            f"No approval record found for: {approval_id}"
        )

    if record.get("status") != "approved":
        raise ApprovalRecordNotApprovedError(
            f"Approval record {approval_id!r} is not approved "
            f"(status={record.get('status')!r})."
        )

    missing = [
        field_name
        for field_name in REQUIRED_APPROVAL_FIELDS
        if not record.get(field_name)
    ]

    if missing:
        raise ApprovalRecordInvalidError(
            f"Approval record {approval_id!r} is missing required "
            f"field(s): {', '.join(missing)}"
        )

    return record


class InstagramReplyController:
    """
    Dry-run-only preview of filling an approved Aiko reply into
    Instagram's reply textbox for a specific comment.

    Strict safety: this class NEVER presses Enter, NEVER clicks
    Post/Send, NEVER writes to approval_queue.json, and NEVER likes,
    follows, unfollows, hides, deletes, reports, or sends direct
    messages. It only fills a textbox for a human to look at, waits,
    and takes a screenshot.
    """

    def __init__(
        self,
        *,
        session: InstagramSession,
        config: ReplyControllerConfig,
    ) -> None:
        self.session = session
        self.config = config

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- generic selector helpers (mirrors instagram_comment_reader.py) ------

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
    async def _find_all_first(root: Any, selectors: list[str]) -> list[Any]:
        for selector in selectors:
            try:
                elements = await root.query_selector_all(selector)
            except Exception:
                elements = []

            if elements:
                return elements

        return []

    async def _detect_login(self, page: Any) -> tuple[str, str]:
        """
        Reuse Phase 8A's classification logic against this page,
        without duplicating InstagramSession's private internals.
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

    # -- comment location -------------------------------------------------

    async def _find_comment_row(self, page: Any, *, comment_id: str) -> Any:
        """
        Locate the exact row for comment_id via its permalink,
        scrolling/expanding with bounded attempts if it is not yet
        visible.
        """
        permalink_selector = self.config.selectors.comment_permalink_template.format(
            comment_id=comment_id
        )

        attempts = 0

        while True:
            container = await self._find_first(
                page,
                self.config.selectors.comment_container,
            )

            if container is not None:
                rows = await self._find_all_first(
                    container,
                    self.config.selectors.comment_item,
                )

                for row in rows:
                    match = await self._find_first(row, [permalink_selector])

                    if match is not None:
                        return row

            if attempts >= self.config.max_scroll_attempts:
                return None

            try:
                await page.mouse.wheel(0, 1600)
            except Exception:
                pass

            await page.wait_for_timeout(self.config.scroll_pause_ms)
            attempts += 1

    # -- diagnostics ----------------------------------------------------------

    def _diagnostic_log_path(self) -> Path:
        return self.config.logs_dir / "reply_controller_diagnostics.jsonl"

    def _append_diagnostic_log(self, entry: dict[str, Any]) -> None:
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        with self._diagnostic_log_path().open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def _save_screenshot(
        self,
        page: Any,
        *,
        filename: str,
    ) -> str | None:
        try:
            self.config.screenshots_dir.mkdir(parents=True, exist_ok=True)

            path = self.config.screenshots_dir / filename

            await page.screenshot(path=str(path), full_page=True)
            return str(path)
        except Exception:
            # Screenshot capture is best-effort diagnostics only.
            return None

    def _log_result(
        self,
        *,
        record: dict[str, Any],
        comment_permalink: str | None = None,
        row_selector: str | None = None,
        reply_selector: str | None = None,
        composer_selector: str | None = None,
        reply_context_signal: str | None = None,
        target_username: str | None = None,
        existing_composer_text: str | None = None,
        final_composer_text: str | None = None,
        mention_preserved: bool | None = None,
        selector_used: str | None = None,
        screenshot_path: str | None = None,
        result: str,
        error: str | None,
    ) -> None:
        self._append_diagnostic_log(
            {
                "approval_id": record.get("approval_id"),
                "post_url": record.get("post_url"),
                "comment_id": record.get("comment_id"),
                "username": record.get("username"),
                "proposed_reply": record.get("proposed_reply"),
                "comment_permalink": comment_permalink,
                "row_selector": row_selector,
                "reply_selector": reply_selector,
                "composer_selector": composer_selector,
                "reply_context_signal": reply_context_signal,
                "target_username": target_username,
                "existing_composer_text": existing_composer_text,
                "final_composer_text": final_composer_text,
                "mention_preserved": mention_preserved,
                "selector_used": selector_used,
                "screenshot_path": screenshot_path,
                "result": result,
                "error": error,
                "created_at": self._now(),
            }
        )

    @staticmethod
    def _compose_final_text(
        *,
        existing_text: str,
        proposed_reply: str,
        target_username: str | None,
    ) -> tuple[str, bool]:
        """
        Preserve Instagram's auto-inserted @username mention when
        present, rather than overwriting it with a bare
        composer.fill(proposed_reply).

        existing_text is used verbatim as the prefix (not
        reconstructed as "@{target_username}") so whatever Instagram
        actually put in the composer is preserved exactly, only
        trimmed of surrounding whitespace. If the existing text does
        not start with the expected @username, the proposed reply is
        used unchanged.
        """
        existing_stripped = (existing_text or "").strip()

        if target_username:
            expected_prefix = f"@{target_username}".strip().lower()

            if existing_stripped.lower().startswith(expected_prefix):
                return f"{existing_stripped} {proposed_reply}", True

        return proposed_reply, False

    # -- threaded-reply context verification (Phase 8D.1) --------------------

    @staticmethod
    async def _composer_current_text(element: Any) -> str:
        """
        Read whatever text a composer currently holds, whether it is a
        <textarea>/<input> (input_value) or a contenteditable div
        (inner_text). Falls back to "" rather than raising.
        """
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

    async def _verify_reply_context(
        self,
        page: Any,
        row: Any,
        *,
        comment_id: str,
        target_username: str | None,
    ) -> tuple[bool, str | None, Any | None, str | None]:
        """
        Verify the page is genuinely in a threaded-reply state for
        this exact comment before anything is filled.

        Checks four signals, in order (first match wins):
        1. a DOM attribute tying a composer/container to comment_id
        2. a composer found INSIDE the comment row itself
        3. visible "Replying to @<target_username>" text
        4. "@<target_username>" already pre-filled into a composer

        A bare page-level "Add a comment" field with none of the
        above present does NOT verify — that is the exact gap this
        method exists to close.

        Returns (verified, method, composer_element_or_None,
        composer_selector_or_None). composer_element is the element
        to fill when this method already located one; callers must
        still search for a composer themselves when it is None but
        verified is True (signals 1 and 3 confirm context without
        necessarily locating the composer).
        """
        for template in self.config.selectors.comment_context_attribute_template:
            selector = template.format(comment_id=comment_id)
            match = await self._find_first(page, [selector])

            if match is not None:
                return True, "dom_attribute", None, None

        row_composer, row_composer_selector = None, None

        for selector in self.config.selectors.row_reply_composer:
            candidate = await self._find_first(row, [selector])

            if candidate is not None:
                row_composer = candidate
                row_composer_selector = selector
                break

        if row_composer is not None:
            return True, "row_scoped_composer", row_composer, row_composer_selector

        for selector in self.config.selectors.replying_to_text:
            candidate = await self._find_first(page, [selector])

            if candidate is None:
                continue

            text = await self._composer_current_text(candidate)

            if target_username and target_username.lower() in text.lower():
                return True, "replying_to_text", None, None

        for selector in self.config.selectors.reply_textbox:
            candidate = await self._find_first(page, [selector])

            if candidate is None:
                continue

            text = await self._composer_current_text(candidate)

            if target_username and f"@{target_username}".lower() in text.lower():
                return True, "at_mention_prefill", candidate, selector

        return False, None, None, None

    # -- orchestration ----------------------------------------------------

    async def preview_reply(self, record: dict[str, Any]) -> ReplyPreviewResult:
        """
        Verify a genuine threaded-reply context for the exact comment,
        then fill the proposed reply into the textbox and take a
        screenshot, then stop.

        If the reply context cannot be verified (Phase 8D.1), the
        textbox is never filled and a ReplyContextNotVerifiedError is
        raised instead.

        Never presses Enter, never clicks Post/Send, and never writes
        to approval_queue.json — this method only ever reads fields
        off the record dict it is given.
        """
        approval_id = str(record["approval_id"])
        post_url = str(record["post_url"])
        comment_id = str(record["comment_id"])
        proposed_reply = str(record["proposed_reply"])
        target_username = record.get("username")

        comment_permalink = self.config.selectors.comment_permalink_template.format(
            comment_id=comment_id
        )

        playwright, context = await self.session._open_context()

        try:
            page = (
                context.pages[0]
                if context.pages
                else await context.new_page()
            )

            try:
                await page.goto(
                    post_url,
                    timeout=self.config.post_navigation_timeout_ms,
                    wait_until="domcontentloaded",
                )
            except PlaywrightTimeoutError as exc:
                screenshot_path = await self._save_screenshot(
                    page, filename=f"reply_preview_{approval_id}.png"
                )
                self._log_result(
                    record=record,
                    comment_permalink=comment_permalink,
                    screenshot_path=screenshot_path,
                    result="failed",
                    error=f"post_open_timeout: {exc}",
                )
                raise ReplyPostAccessError(
                    f"Timed out opening post: {post_url}"
                ) from exc
            except Exception as exc:
                screenshot_path = await self._save_screenshot(
                    page, filename=f"reply_preview_{approval_id}.png"
                )
                self._log_result(
                    record=record,
                    comment_permalink=comment_permalink,
                    screenshot_path=screenshot_path,
                    result="failed",
                    error=f"post_open_failed: {exc}",
                )
                raise ReplyPostAccessError(
                    f"Unable to open post: {post_url}: {exc}"
                ) from exc

            await page.wait_for_timeout(1500)

            login_status, login_reason = await self._detect_login(page)

            if login_status != "logged_in":
                screenshot_path = await self._save_screenshot(
                    page, filename=f"reply_preview_{approval_id}.png"
                )
                self._log_result(
                    record=record,
                    comment_permalink=comment_permalink,
                    screenshot_path=screenshot_path,
                    result="failed",
                    error=(
                        f"login_lost: status={login_status} "
                        f"reason={login_reason}"
                    ),
                )
                raise ReplyControllerLoginLostError(
                    "Instagram session is not logged_in while previewing "
                    f"a reply (status={login_status})."
                )

            row = await self._find_comment_row(page, comment_id=comment_id)

            if row is None:
                screenshot_path = await self._save_screenshot(
                    page, filename=f"reply_preview_{approval_id}.png"
                )
                self._log_result(
                    record=record,
                    comment_permalink=comment_permalink,
                    screenshot_path=screenshot_path,
                    result="failed",
                    error="comment_not_found",
                )
                raise CommentNotFoundError(
                    f"Could not locate comment_id={comment_id} on {post_url}"
                )

            reply_button, reply_selector = None, None

            for selector in self.config.selectors.reply_button:
                candidate = await self._find_first(row, [selector])

                if candidate is not None:
                    reply_button = candidate
                    reply_selector = selector
                    break

            if reply_button is None:
                screenshot_path = await self._save_screenshot(
                    page, filename=f"reply_preview_{approval_id}.png"
                )
                self._log_result(
                    record=record,
                    comment_permalink=comment_permalink,
                    screenshot_path=screenshot_path,
                    result="failed",
                    error="reply_control_not_found",
                )
                raise ReplyControlNotFoundError(
                    f"Could not find the Reply control for comment_id={comment_id}"
                )

            # Two mandatory diagnostic screenshots bracketing the
            # click, regardless of what happens next.
            before_screenshot_path = await self._save_screenshot(
                page, filename=f"before_reply_click_{approval_id}.png"
            )

            try:
                await reply_button.click(timeout=5000)
            except Exception as exc:
                after_screenshot_path = await self._save_screenshot(
                    page, filename=f"after_reply_click_{approval_id}.png"
                )
                self._log_result(
                    record=record,
                    comment_permalink=comment_permalink,
                    reply_selector=reply_selector,
                    screenshot_path=after_screenshot_path,
                    result="failed",
                    error=f"reply_control_click_failed: {exc}",
                )
                raise ReplyControlNotFoundError(
                    f"Unable to click the Reply control: {exc}"
                ) from exc

            await page.wait_for_timeout(800)

            after_screenshot_path = await self._save_screenshot(
                page, filename=f"after_reply_click_{approval_id}.png"
            )

            (
                context_verified,
                context_method,
                composer,
                composer_selector,
            ) = await self._verify_reply_context(
                page,
                row,
                comment_id=comment_id,
                target_username=target_username,
            )

            if not context_verified:
                self._log_result(
                    record=record,
                    comment_permalink=comment_permalink,
                    reply_selector=reply_selector,
                    reply_context_signal=None,
                    target_username=target_username,
                    screenshot_path=after_screenshot_path,
                    result="failed",
                    error="reply_context_not_verified",
                )
                raise ReplyContextNotVerifiedError(
                    "Could not verify a genuine threaded-reply context for "
                    f"comment_id={comment_id} (a generic 'Add a comment' "
                    "field alone is not sufficient). Textbox was NOT filled. "
                    f"Screenshot: {after_screenshot_path}"
                )

            if composer is None:
                composer = await self._find_first(
                    row, self.config.selectors.row_reply_composer
                )

                if composer is None:
                    for selector in self.config.selectors.reply_textbox:
                        composer = await self._find_first(page, [selector])

                        if composer is not None:
                            composer_selector = selector
                            break

            if composer is None:
                self._log_result(
                    record=record,
                    comment_permalink=comment_permalink,
                    reply_selector=reply_selector,
                    reply_context_signal=context_method,
                    target_username=target_username,
                    screenshot_path=after_screenshot_path,
                    result="failed",
                    error="reply_textbox_not_found",
                )
                raise ReplyTextboxNotFoundError(
                    "Reply context was verified, but no composer element "
                    f"could be located to fill for comment_id={comment_id}"
                )

            # Read the composer's existing value BEFORE adding
            # anything, then construct the final text in one step, so
            # Instagram's auto-inserted @username mention (if present)
            # is preserved rather than overwritten by a bare fill.
            existing_composer_text = await self._composer_current_text(composer)

            final_composer_text, mention_preserved = self._compose_final_text(
                existing_text=existing_composer_text,
                proposed_reply=proposed_reply,
                target_username=target_username,
            )

            try:
                await composer.fill(final_composer_text)
            except Exception:
                await composer.click(timeout=2000)
                await composer.type(final_composer_text)

            # Strict dry run: wait, screenshot, and stop here. Never
            # press Enter, never click Post/Send.
            await page.wait_for_timeout(self.config.post_fill_wait_ms)

            screenshot_path = await self._save_screenshot(
                page, filename=f"reply_preview_{approval_id}.png"
            )

            self._log_result(
                record=record,
                comment_permalink=comment_permalink,
                reply_selector=reply_selector,
                composer_selector=composer_selector,
                reply_context_signal=context_method,
                target_username=target_username,
                existing_composer_text=existing_composer_text,
                final_composer_text=final_composer_text,
                mention_preserved=mention_preserved,
                selector_used=composer_selector,
                screenshot_path=screenshot_path,
                result="preview_success",
                error=None,
            )

            return ReplyPreviewResult(
                login_status=login_status,
                approval_id=approval_id,
                comment_found=True,
                reply_control_clicked=True,
                textbox_filled=True,
                sent=False,
                reply_context_verified=True,
                reply_context_method=context_method,
                reply_target_username=target_username,
                existing_composer_text=existing_composer_text,
                final_composer_text=final_composer_text,
                mention_preserved=mention_preserved,
                screenshot_path=screenshot_path,
                selector_used=composer_selector,
                error=None,
            )

        finally:
            await context.close()
            await playwright.stop()


def build_reply_controller(
    *,
    config_path: str | Path | None = None,
) -> InstagramReplyController:
    session_config = load_session_config(config_path=config_path)
    reply_config = load_reply_controller_config(config_path=config_path)
    session = InstagramSession(session_config)

    return InstagramReplyController(session=session, config=reply_config)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Instagram Reply Controller — Dry Run Preview (Phase 8D)"
        )
    )

    parser.add_argument(
        "--dry-run",
        metavar="APPROVAL_ID",
        required=True,
        help=(
            "Preview an approved reply for this approval_id. Fills the "
            "textbox and screenshots it — never sends, never updates "
            "approval_queue.json."
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate instagram.yaml config file.",
    )

    return parser.parse_args(argv)


def _print_result(result: ReplyPreviewResult) -> None:
    print()
    print("AIKO Instagram Reply Controller — Dry Run Preview")
    print("----------------------------------------------------")
    print(f"login status:          {result.login_status}")
    print(f"approval id:           {result.approval_id}")
    print(f"comment found:         {result.comment_found}")
    print(f"reply control clicked: {result.reply_control_clicked}")
    print(f"reply context verified:{result.reply_context_verified}")
    print(f"reply context method:  {result.reply_context_method}")
    print(f"reply target username: {result.reply_target_username}")
    print(f"mention preserved:     {result.mention_preserved}")
    print(f"existing composer text:{result.existing_composer_text!r}")
    print(f"final composer text:   {result.final_composer_text!r}")
    print(f"textbox filled:        {result.textbox_filled}")
    print(f"sent:                  {result.sent}")
    print(f"screenshot path:       {result.screenshot_path}")
    print()


async def _run(arguments: argparse.Namespace) -> int:
    approval_service = build_approval_service()

    try:
        record = load_approved_record(
            arguments.dry_run,
            approval_service=approval_service,
        )
    except ReplyControllerError as exc:
        print(f"[InstagramReplyController] {exc}")
        return 1

    controller = build_reply_controller(config_path=arguments.config)

    try:
        result = await controller.preview_reply(record)
    except InstagramSessionError as exc:
        print(f"[InstagramReplyController] preview failed: {exc}")
        return 1

    _print_result(result)
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
