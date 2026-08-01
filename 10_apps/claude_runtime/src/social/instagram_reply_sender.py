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
from .instagram_reply_controller import (
    InstagramReplyController,
    ReplyContextNotVerifiedError,
    ReplyControllerConfig,
    load_approved_record,
    load_reply_controller_config,
)
from .instagram_session import InstagramSession, load_config as load_session_config

# Phase 8E is the ONLY module in this codebase that actually sends
# anything to Instagram. It performs exactly one Post click per
# invocation, for exactly one approval_id, and only after reusing
# Phase 8D.1's unmodified threaded-reply verification. Phase 8E.1
# adds a --preflight mode that reaches the exact same verified,
# filled-composer state but never clicks Post. Phase 8E.2 tightens
# Post-button identification and persists send failures. Neither
# mode ever grows a method for any of the actions below, and neither
# ever falls back to pressing Enter to submit.
DISALLOWED_ACTIONS = (
    "like",
    "follow",
    "unfollow",
    "hide",
    "delete",
    "report",
    "send_dm",
    "press_enter",
    "press",
    "batch_send",
    "watch",
    "retry_send",
)

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "social" / "instagram.yaml"


def _runtime_root() -> Path:
    """
    instagram_reply_sender.py location:

    10_apps/claude_runtime/src/social/instagram_reply_sender.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


class PostButtonNotFoundError(ReplyControllerError):
    """Raised when no Post button candidate can be located at all."""


class PostButtonDisabledError(ReplyControllerError):
    """
    Raised when exactly one Post button candidate was found (via
    selector match) but it is not currently usable — not visible
    and/or not enabled.
    """


class PostButtonAmbiguousError(ReplyControllerError):
    """
    Raised when more than one visible, enabled Post button candidate
    is found and none can be safely preferred. Fails safely rather
    than guessing which one is the real composer's submit control.
    """


class SendClickFailedError(ReplyControllerError):
    """Raised when clicking the Post button raises an error."""


class SendVerificationFailedError(ReplyControllerError):
    """
    Raised when, after clicking Post, the reply cannot be confirmed as
    having actually rendered under the target comment. The approval
    record's status is left as "approved", not marked "sent", when
    this is raised.
    """


@dataclass(slots=True)
class ReplySenderSelectors:
    """
    Centralised CSS/Playwright selectors specific to sending, kept
    separate from ReplyControllerSelectors (Phase 8D) since sending is
    a distinct, higher-stakes capability living in its own module.

    post_button and composer_container_selector were validated live
    2026-07-31 (Phase 8E.2) — see the comments in
    config/social/instagram.yaml for how. sent_reply_item remains an
    untested placeholder: no live send has ever succeeded yet.
    """

    post_button: list[str] = field(default_factory=list)
    sent_reply_item: list[str] = field(default_factory=list)
    composer_container_selector: str = "form"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplySenderConfig:
    """
    Fully-resolved runtime configuration for the reply sender.

    Screenshots and logs reuse the wrapped ReplyControllerConfig's
    directories (via InstagramReplySender.config, inherited) — this
    only adds what's new: Post-button/sent-reply selectors and
    send-specific timeouts.
    """

    selectors: ReplySenderSelectors
    post_click_timeout_ms: int = 5000
    send_settle_wait_ms: int = 2000

    def to_dict(self) -> dict[str, Any]:
        return {
            "selectors": self.selectors.to_dict(),
            "post_click_timeout_ms": self.post_click_timeout_ms,
            "send_settle_wait_ms": self.send_settle_wait_ms,
        }


@dataclass(slots=True)
class ReplySendResult:
    """
    Summary of one instagram_reply_sender --send-approved run.

    sent is True only when the Post click AND the post-send
    verification both succeeded, which is also the only condition
    under which approval_queue.json's status/sent_at were updated.
    On any failure, error is instead persisted into the queue record
    (status stays "approved") — see InstagramReplySender._mark_failed.
    """

    login_status: str
    approval_id: str
    comment_found: bool
    reply_control_clicked: bool
    reply_context_verified: bool
    reply_context_method: str | None
    reply_target_username: str | None
    existing_composer_text: str | None
    final_reply_text: str
    mention_preserved: bool
    send_clicked: bool
    send_verified: bool
    sent: bool
    status: str
    post_button_selector_method: str | None = None
    post_button_candidate_count: int = 0
    post_button_visible: bool = False
    post_button_enabled: bool = False
    post_button_interactive: bool = False
    before_send_screenshot_path: str | None = None
    after_send_screenshot_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReplyPreflightResult:
    """
    Summary of one instagram_reply_sender --preflight run.

    sent is always False: preflight never clicks Post and never
    receives an ApprovalService reference, so it is structurally
    incapable of writing to approval_queue.json.
    """

    approval_id: str
    reply_target_username: str | None
    reply_context_verified: bool
    reply_context_method: str | None
    mention_preserved: bool
    final_reply_text: str
    post_button_found: bool
    post_button_enabled: bool
    sent_reply_container_found: bool
    post_button_selector_method: str | None = None
    post_button_candidate_count: int = 0
    post_button_visible: bool = False
    post_button_interactive: bool = False
    sent: bool = False
    screenshot_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _PostButtonLookup:
    """
    Internal result of _locate_post_button(). Not part of the public
    result dataclasses, but its fields map directly onto them.

    element is non-None only when exactly one visible, enabled
    candidate was found for some selector in
    sender_config.selectors.post_button, searched first within the
    verified composer's container (selector_method="composer_container")
    and only falling back to the whole page
    (selector_method="page_wide_fallback") if that container could not
    be located at all.
    """

    element: Any | None
    selector_method: str | None
    selector_string: str | None
    candidate_count: int
    visible: bool
    enabled: bool
    interactive: bool
    ambiguous: bool = False
    disabled_candidate: bool = False


def load_reply_sender_config(
    *,
    config_path: str | Path | None = None,
) -> ReplySenderConfig:
    """
    Load the reply_sender section from config/social/instagram.yaml.

    Independent loader, matching the pattern already used by
    instagram_comment_reader.py and instagram_reply_controller.py.
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

    section = raw.get("reply_sender") or {}
    selectors_section = section.get("selectors") or {}
    timeouts_section = section.get("timeouts") or {}

    selectors = ReplySenderSelectors(
        post_button=list(selectors_section.get("post_button", [])),
        sent_reply_item=list(selectors_section.get("sent_reply_item", [])),
        composer_container_selector=str(
            selectors_section.get("composer_container_selector", "form")
        ),
    )

    return ReplySenderConfig(
        selectors=selectors,
        post_click_timeout_ms=int(
            timeouts_section.get("post_click_timeout_ms", 5000)
        ),
        send_settle_wait_ms=int(
            timeouts_section.get("send_settle_wait_ms", 2000)
        ),
    )


class InstagramReplySender(InstagramReplyController):
    """
    Sends one approved Aiko reply to Instagram for real, or verifies
    the send is ready without sending anything (--preflight).

    Subclasses InstagramReplyController (Phase 8D) to reuse — not
    duplicate — its comment-location, login-detection, threaded-reply
    context verification (Phase 8D.1, unmodified), and @mention
    -preserving composition logic (Phase 8D.2, unmodified).

    send_reply() and preflight() both call the same private
    _reach_filled_composer() helper for everything through "composer
    filled with the final reply text", and the same
    _locate_post_button() for identifying the Post control —
    preflight is only a trustworthy predictor of what a real send
    would do because it is not a re-implementation of that path, it
    is the same code.

    Never presses Enter as a submit fallback. Never likes, follows,
    unfollows, hides, deletes, reports, or sends direct messages.
    Never performs more than one Post click per invocation. Never
    retries a failed send automatically. preflight() never clicks
    Post and never receives an ApprovalService reference at all.
    """

    def __init__(
        self,
        *,
        session: InstagramSession,
        controller_config: ReplyControllerConfig,
        sender_config: ReplySenderConfig,
    ) -> None:
        super().__init__(session=session, config=controller_config)
        self.sender_config = sender_config

    # -- audit log ----------------------------------------------------

    def _audit_log_path(self) -> Path:
        return self.config.logs_dir / "reply_sender_audit.jsonl"

    def _write_audit(self, entry: dict[str, Any]) -> None:
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        with self._audit_log_path().open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _audit_entry(
        self,
        *,
        record: dict[str, Any],
        result: str,
        error: str | None,
        final_reply_text: str | None = None,
        reply_context_method: str | None = None,
        before_screenshot: str | None = None,
        after_screenshot: str | None = None,
        screenshot: str | None = None,
        send_clicked: bool = False,
        send_verified: bool = False,
        sent_at: str | None = None,
        post_button_found: bool | None = None,
        post_button_enabled: bool | None = None,
        post_button_selector: str | None = None,
        post_button_selector_method: str | None = None,
        post_button_candidate_count: int | None = None,
        post_button_visible: bool | None = None,
        post_button_interactive: bool | None = None,
        sent_reply_container_found: bool | None = None,
        sent_reply_item_candidate_count: int | None = None,
    ) -> dict[str, Any]:
        return {
            "approval_id": record.get("approval_id"),
            "stable_id": record.get("stable_id"),
            "post_url": record.get("post_url"),
            "comment_id": record.get("comment_id"),
            "username": record.get("username"),
            "final_reply_text": final_reply_text,
            "reply_context_method": reply_context_method,
            "before_screenshot": before_screenshot,
            "after_screenshot": after_screenshot,
            "screenshot": screenshot,
            "send_clicked": send_clicked,
            "send_verified": send_verified,
            "post_button_found": post_button_found,
            "post_button_enabled": post_button_enabled,
            "post_button_selector": post_button_selector,
            "post_button_selector_method": post_button_selector_method,
            "post_button_candidate_count": post_button_candidate_count,
            "post_button_visible": post_button_visible,
            "post_button_interactive": post_button_interactive,
            "sent_reply_container_found": sent_reply_container_found,
            "sent_reply_item_candidate_count": sent_reply_item_candidate_count,
            "result": result,
            "error": error,
            "created_at": self._now(),
            "sent_at": sent_at,
        }

    # -- shared: reach a verified, filled composer ---------------------------

    async def _reach_filled_composer(
        self,
        page: Any,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Shared by send_reply() and preflight(): navigate to the post,
        verify login, find the exact comment row, click its row
        -scoped Reply control, verify a genuine threaded-reply context
        (Phase 8D.1, unmodified), locate the composer, and fill it
        with the @mention-preserving composed text (Phase 8D.2,
        unmodified).

        Returns a dict: login_status, row, reply_context_method,
        composer, existing_composer_text, final_reply_text,
        mention_preserved. Raises the same exceptions as before for
        each failure mode, writing an audit-log entry first.
        """
        post_url = str(record["post_url"])
        comment_id = str(record["comment_id"])
        proposed_reply = str(record["proposed_reply"])
        target_username = record.get("username")

        try:
            await page.goto(
                post_url,
                timeout=self.config.post_navigation_timeout_ms,
                wait_until="domcontentloaded",
            )
        except Exception as exc:
            self._write_audit(
                self._audit_entry(
                    record=record,
                    result="failed",
                    error=f"post_open_failed: {exc}",
                )
            )
            raise ReplyPostAccessError(
                f"Unable to open post: {post_url}: {exc}"
            ) from exc

        await page.wait_for_timeout(1500)

        login_status, login_reason = await self._detect_login(page)

        if login_status != "logged_in":
            self._write_audit(
                self._audit_entry(
                    record=record,
                    result="failed",
                    error=(
                        f"login_lost: status={login_status} "
                        f"reason={login_reason}"
                    ),
                )
            )
            raise ReplyControllerLoginLostError(
                "Instagram session is not logged_in "
                f"(status={login_status})."
            )

        row = await self._find_comment_row(page, comment_id=comment_id)

        if row is None:
            self._write_audit(
                self._audit_entry(
                    record=record,
                    result="failed",
                    error="comment_not_found",
                )
            )
            raise CommentNotFoundError(
                f"Could not locate comment_id={comment_id} on {post_url}"
            )

        reply_button = None

        for selector in self.config.selectors.reply_button:
            candidate = await self._find_first(row, [selector])

            if candidate is not None:
                reply_button = candidate
                break

        if reply_button is None:
            self._write_audit(
                self._audit_entry(
                    record=record,
                    result="failed",
                    error="reply_control_not_found",
                )
            )
            raise ReplyControlNotFoundError(
                f"Could not find the Reply control for comment_id={comment_id}"
            )

        try:
            await reply_button.click(timeout=5000)
        except Exception as exc:
            self._write_audit(
                self._audit_entry(
                    record=record,
                    result="failed",
                    error=f"reply_control_click_failed: {exc}",
                )
            )
            raise ReplyControlNotFoundError(
                f"Unable to click the Reply control: {exc}"
            ) from exc

        await page.wait_for_timeout(800)

        (
            context_verified,
            context_method,
            composer,
            _composer_selector,
        ) = await self._verify_reply_context(
            page,
            row,
            comment_id=comment_id,
            target_username=target_username,
        )

        if not context_verified:
            self._write_audit(
                self._audit_entry(
                    record=record,
                    result="failed",
                    error="reply_context_not_verified",
                )
            )
            raise ReplyContextNotVerifiedError(
                "Could not verify a genuine threaded-reply context for "
                f"comment_id={comment_id}. Refusing to proceed — no Post "
                "button was searched for, and nothing was clicked."
            )

        if composer is None:
            composer = await self._find_first(
                row, self.config.selectors.row_reply_composer
            )

            if composer is None:
                for selector in self.config.selectors.reply_textbox:
                    composer = await self._find_first(page, [selector])

                    if composer is not None:
                        break

        if composer is None:
            self._write_audit(
                self._audit_entry(
                    record=record,
                    reply_context_method=context_method,
                    result="failed",
                    error="reply_textbox_not_found",
                )
            )
            raise ReplyTextboxNotFoundError(
                "Reply context was verified, but no composer element "
                f"could be located to fill for comment_id={comment_id}"
            )

        existing_composer_text = await self._composer_current_text(composer)

        final_reply_text, mention_preserved = self._compose_final_text(
            existing_text=existing_composer_text,
            proposed_reply=proposed_reply,
            target_username=target_username,
        )

        try:
            await composer.fill(final_reply_text)
        except Exception:
            await composer.click(timeout=2000)
            await composer.type(final_reply_text)

        return {
            "login_status": login_status,
            "row": row,
            "reply_context_method": context_method,
            "composer": composer,
            "existing_composer_text": existing_composer_text,
            "final_reply_text": final_reply_text,
            "mention_preserved": mention_preserved,
        }

    # -- Post button identification (Phase 8E.2) -----------------------------

    async def _locate_composer_container(self, composer: Any) -> Any | None:
        """
        Find the element wrapping the verified composer (its <form> by
        default, per composer_container_selector), so the Post-button
        search below can be scoped to the EXACT SAME composer rather
        than the whole page. Returns None (triggering a page-wide
        fallback) if this cannot be determined — including in tests,
        where composer is a plain fake object with no real DOM.
        """
        tag = self.sender_config.selectors.composer_container_selector

        try:
            handle = await composer.evaluate_handle(
                f"el => el.closest('{tag}') || el.parentElement"
            )
        except Exception:
            return None

        try:
            element = handle.as_element()
        except Exception:
            return None

        return element

    async def _locate_post_button(
        self,
        page: Any,
        *,
        composer: Any,
    ) -> _PostButtonLookup:
        """
        Locate the Post button, scoped to the verified composer's
        container first, falling back to the whole page only if that
        container cannot be located. A candidate is only accepted when
        it is the SOLE visible, enabled match for a given selector —
        zero candidates, or more than one, both fail safely (no
        element is ever returned in those cases) rather than guessing.

        This is what fixes the original bug: an unquoted, page-wide
        text=Post selector previously matched hidden <title> elements
        and risked matching "Boost post". The default selector is now
        an EXACT accessible-name role query
        (role=button[name="Post"]), and even that is still filtered
        here for visibility/enabled state and uniqueness rather than
        trusted blindly.
        """
        container = await self._locate_composer_container(composer)
        search_root = container if container is not None else page
        selector_method = (
            "composer_container" if container is not None else "page_wide_fallback"
        )

        total_candidates = 0
        saw_single_unusable_candidate = False

        for selector in self.sender_config.selectors.post_button:
            try:
                candidates = await search_root.query_selector_all(selector)
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
                return _PostButtonLookup(
                    element=valid[0],
                    selector_method=selector_method,
                    selector_string=selector,
                    candidate_count=len(candidates),
                    visible=True,
                    enabled=True,
                    interactive=True,
                )

            if len(valid) > 1:
                return _PostButtonLookup(
                    element=None,
                    selector_method=selector_method,
                    selector_string=selector,
                    candidate_count=len(candidates),
                    visible=True,
                    enabled=True,
                    interactive=False,
                    ambiguous=True,
                )

            if len(candidates) == 1:
                saw_single_unusable_candidate = True

        return _PostButtonLookup(
            element=None,
            selector_method=selector_method,
            selector_string=None,
            candidate_count=total_candidates,
            visible=False,
            enabled=False,
            interactive=False,
            disabled_candidate=saw_single_unusable_candidate,
        )

    @staticmethod
    def _raise_for_failed_lookup(
        lookup: _PostButtonLookup,
        *,
        comment_id: str,
        context: str,
    ) -> None:
        if lookup.ambiguous:
            raise PostButtonAmbiguousError(
                f"Found {lookup.candidate_count} visible, enabled Post button "
                f"candidates for comment_id={comment_id} ({context}); refusing "
                "to guess which one is correct. Nothing was sent."
            )

        if lookup.disabled_candidate:
            raise PostButtonDisabledError(
                f"The Post button for comment_id={comment_id} was found but "
                f"is not usable (not visible and/or not enabled) ({context})."
            )

        raise PostButtonNotFoundError(
            f"Could not find a unique, visible, enabled Post button for "
            f"comment_id={comment_id} ({context}). Nothing was sent."
        )

    # -- post-send verification ----------------------------------------------

    async def _verify_send(
        self,
        row: Any,
        *,
        target_username: str | None,
        final_reply_text: str,
    ) -> bool:
        """
        Confirm the reply actually rendered under the target comment
        thread, rather than trusting a successful click alone.

        Checks target username against the row itself (confirms this
        is genuinely the right person's thread) and final_reply_text
        against the row or any nested reply item (confirms our
        specific content is present). Single check — no polling loop,
        since automatic retries are explicitly out of scope.
        """
        row_text = ((await self._composer_current_text(row)) or "").lower()
        expected_username = (target_username or "").strip().lower()

        if expected_username and expected_username not in row_text:
            return False

        expected_text = final_reply_text.strip().lower()

        if not expected_text:
            return False

        candidates = list(
            await self._find_all_first(row, self.sender_config.selectors.sent_reply_item)
        )
        candidates.append(row)

        for element in candidates:
            text = ((await self._composer_current_text(element)) or "").lower()

            if expected_text in text:
                return True

        return False

    # -- queue updates (reuse ApprovalService's own atomic write) ------------

    def _mark_sent(
        self,
        approval_service: ApprovalService,
        approval_id: str,
        *,
        sent_at: str,
    ) -> dict[str, Any]:
        queue = approval_service.load_queue()
        record: dict[str, Any] | None = None

        for entry in queue["approvals"]:
            if entry.get("approval_id") == approval_id:
                record = entry
                break

        if record is None:
            raise ApprovalRecordNotFoundError(
                f"No approval record found for: {approval_id}"
            )

        record["status"] = "sent"
        record["sent_at"] = sent_at
        record["error"] = None

        queue["updated_at"] = self._now()
        approval_service.save_queue(queue)

        return record

    def _mark_failed(
        self,
        approval_service: ApprovalService,
        approval_id: str,
        *,
        error: str,
    ) -> dict[str, Any] | None:
        """
        Persist a send failure into the approval record: error is set
        to the failure message, status stays "approved" (NEVER changed
        to "failed"), sent_at stays whatever it already was (null,
        unless a bug elsewhere set it — never touched here). Uses
        ApprovalService's own atomic load/save (tmp file + replace),
        the same as _mark_sent.

        Deliberately swallows its own failures (returns None) rather
        than raising: this is called from an exception handler, and a
        secondary failure here must never shadow the original error
        being reported to the caller.
        """
        try:
            queue = approval_service.load_queue()
        except Exception:
            return None

        record: dict[str, Any] | None = None

        for entry in queue.get("approvals", []):
            if entry.get("approval_id") == approval_id:
                record = entry
                break

        if record is None:
            return None

        record["error"] = error
        # Phase 8C's per-record schema has no updated_at field; only
        # the queue's own top-level updated_at is supported.
        queue["updated_at"] = self._now()

        try:
            approval_service.save_queue(queue)
        except Exception:
            return None

        return record

    # -- orchestration: real send -------------------------------------------

    async def send_reply(
        self,
        record: dict[str, Any],
        *,
        approval_service: ApprovalService,
    ) -> ReplySendResult:
        approval_id = str(record["approval_id"])
        comment_id = str(record["comment_id"])

        playwright, context = await self.session._open_context()

        try:
            page = (
                context.pages[0]
                if context.pages
                else await context.new_page()
            )

            try:
                target_username = record.get("username")

                if not target_username:
                    raise ApprovalRecordInvalidError(
                        f"Approval record {approval_id!r} has no target "
                        "username; refusing to send."
                    )

                prepared = await self._reach_filled_composer(page, record)
                row = prepared["row"]
                composer = prepared["composer"]
                context_method = prepared["reply_context_method"]
                final_reply_text = prepared["final_reply_text"]

                before_send_path = await self._save_screenshot(
                    page, filename=f"before_send_{approval_id}.png"
                )

                lookup = await self._locate_post_button(page, composer=composer)

                if lookup.element is None:
                    after_path = await self._save_screenshot(
                        page, filename=f"after_send_{approval_id}.png"
                    )
                    self._write_audit(
                        self._audit_entry(
                            record=record,
                            final_reply_text=final_reply_text,
                            reply_context_method=context_method,
                            before_screenshot=before_send_path,
                            after_screenshot=after_path,
                            send_clicked=False,
                            send_verified=False,
                            post_button_found=False,
                            post_button_selector_method=lookup.selector_method,
                            post_button_candidate_count=lookup.candidate_count,
                            post_button_visible=lookup.visible,
                            post_button_enabled=lookup.enabled,
                            post_button_interactive=lookup.interactive,
                            result="failed",
                            error=(
                                "post_button_ambiguous"
                                if lookup.ambiguous
                                else "post_button_not_found"
                            ),
                        )
                    )
                    self._raise_for_failed_lookup(
                        lookup, comment_id=comment_id, context="send"
                    )

                post_button = lookup.element

                # The one and only send action. No Enter-key fallback
                # exists anywhere in this module.
                try:
                    await post_button.click(
                        timeout=self.sender_config.post_click_timeout_ms
                    )
                except Exception as exc:
                    after_path = await self._save_screenshot(
                        page, filename=f"after_send_{approval_id}.png"
                    )
                    self._write_audit(
                        self._audit_entry(
                            record=record,
                            final_reply_text=final_reply_text,
                            reply_context_method=context_method,
                            before_screenshot=before_send_path,
                            after_screenshot=after_path,
                            send_clicked=False,
                            send_verified=False,
                            post_button_found=True,
                            post_button_selector_method=lookup.selector_method,
                            post_button_candidate_count=lookup.candidate_count,
                            post_button_visible=lookup.visible,
                            post_button_enabled=lookup.enabled,
                            post_button_interactive=lookup.interactive,
                            result="failed",
                            error=f"send_click_failed: {exc}",
                        )
                    )
                    raise SendClickFailedError(
                        f"Unable to click Post: {exc}"
                    ) from exc

                await page.wait_for_timeout(self.sender_config.send_settle_wait_ms)

                after_send_path = await self._save_screenshot(
                    page, filename=f"after_send_{approval_id}.png"
                )

                send_verified = await self._verify_send(
                    row,
                    target_username=target_username,
                    final_reply_text=final_reply_text,
                )

                if not send_verified:
                    self._write_audit(
                        self._audit_entry(
                            record=record,
                            final_reply_text=final_reply_text,
                            reply_context_method=context_method,
                            before_screenshot=before_send_path,
                            after_screenshot=after_send_path,
                            send_clicked=True,
                            send_verified=False,
                            post_button_found=True,
                            post_button_selector_method=lookup.selector_method,
                            post_button_candidate_count=lookup.candidate_count,
                            post_button_visible=lookup.visible,
                            post_button_enabled=lookup.enabled,
                            post_button_interactive=lookup.interactive,
                            result="failed",
                            error="send_verification_failed",
                        )
                    )
                    raise SendVerificationFailedError(
                        "Clicked Post, but could not verify the reply "
                        f"appeared under comment_id={comment_id}. Approval "
                        "status left as 'approved' — please check manually "
                        "before retrying."
                    )

                sent_at = self._now()
                self._mark_sent(approval_service, approval_id, sent_at=sent_at)

                self._write_audit(
                    self._audit_entry(
                        record=record,
                        final_reply_text=final_reply_text,
                        reply_context_method=context_method,
                        before_screenshot=before_send_path,
                        after_screenshot=after_send_path,
                        send_clicked=True,
                        send_verified=True,
                        post_button_found=True,
                        post_button_selector_method=lookup.selector_method,
                        post_button_candidate_count=lookup.candidate_count,
                        post_button_visible=lookup.visible,
                        post_button_enabled=lookup.enabled,
                        post_button_interactive=lookup.interactive,
                        result="sent",
                        error=None,
                        sent_at=sent_at,
                    )
                )

                return ReplySendResult(
                    login_status=prepared["login_status"],
                    approval_id=approval_id,
                    comment_found=True,
                    reply_control_clicked=True,
                    reply_context_verified=True,
                    reply_context_method=context_method,
                    reply_target_username=target_username,
                    existing_composer_text=prepared["existing_composer_text"],
                    final_reply_text=final_reply_text,
                    mention_preserved=prepared["mention_preserved"],
                    send_clicked=True,
                    send_verified=True,
                    sent=True,
                    status="sent",
                    post_button_selector_method=lookup.selector_method,
                    post_button_candidate_count=lookup.candidate_count,
                    post_button_visible=lookup.visible,
                    post_button_enabled=lookup.enabled,
                    post_button_interactive=lookup.interactive,
                    before_send_screenshot_path=before_send_path,
                    after_send_screenshot_path=after_send_path,
                    error=None,
                )

            except InstagramSessionError as exc:
                # Persist the failure so a human looking at
                # approval_queue.json sees why, without needing to
                # cross-reference the audit log. status stays
                # "approved" (never "failed") and sent_at stays null —
                # _mark_failed only ever touches the error field.
                self._mark_failed(approval_service, approval_id, error=str(exc))
                raise

        finally:
            await context.close()
            await playwright.stop()

    # -- orchestration: preflight (never sends) -----------------

    async def preflight(self, record: dict[str, Any]) -> ReplyPreflightResult:
        """
        Verify the send is ready — exact comment lookup, threaded
        -reply context, @mention-preserving composition, and a
        uniquely-identified, visible, enabled, interactive Post button
        scoped to the verified composer — WITHOUT clicking Post and
        WITHOUT ever touching approval_queue.json (no ApprovalService
        reference is accepted here at all).
        """
        approval_id = str(record["approval_id"])
        comment_id = str(record["comment_id"])
        target_username = record.get("username")

        if not target_username:
            raise ApprovalRecordInvalidError(
                f"Approval record {approval_id!r} has no target username; "
                "refusing preflight."
            )

        playwright, context = await self.session._open_context()

        try:
            page = (
                context.pages[0]
                if context.pages
                else await context.new_page()
            )

            prepared = await self._reach_filled_composer(page, record)
            row = prepared["row"]
            composer = prepared["composer"]
            context_method = prepared["reply_context_method"]
            final_reply_text = prepared["final_reply_text"]

            lookup = await self._locate_post_button(page, composer=composer)

            try:
                sent_reply_candidates = await self._find_all_first(
                    row, self.sender_config.selectors.sent_reply_item
                )
            except Exception:
                sent_reply_candidates = []

            sent_reply_container_found = row is not None

            await page.wait_for_timeout(self.sender_config.send_settle_wait_ms)

            screenshot_path = await self._save_screenshot(
                page, filename=f"preflight_{approval_id}.png"
            )

            if lookup.element is None:
                self._write_audit(
                    self._audit_entry(
                        record=record,
                        final_reply_text=final_reply_text,
                        reply_context_method=context_method,
                        screenshot=screenshot_path,
                        send_clicked=False,
                        post_button_found=False,
                        post_button_selector_method=lookup.selector_method,
                        post_button_candidate_count=lookup.candidate_count,
                        post_button_visible=lookup.visible,
                        post_button_enabled=lookup.enabled,
                        post_button_interactive=lookup.interactive,
                        sent_reply_container_found=sent_reply_container_found,
                        sent_reply_item_candidate_count=len(sent_reply_candidates),
                        result="preflight_failed",
                        error=(
                            "post_button_ambiguous"
                            if lookup.ambiguous
                            else "post_button_disabled"
                            if lookup.disabled_candidate
                            else "post_button_not_found"
                        ),
                    )
                )
                self._raise_for_failed_lookup(
                    lookup, comment_id=comment_id, context="preflight"
                )

            self._write_audit(
                self._audit_entry(
                    record=record,
                    final_reply_text=final_reply_text,
                    reply_context_method=context_method,
                    screenshot=screenshot_path,
                    send_clicked=False,
                    post_button_found=True,
                    post_button_selector=lookup.selector_string,
                    post_button_selector_method=lookup.selector_method,
                    post_button_candidate_count=lookup.candidate_count,
                    post_button_visible=lookup.visible,
                    post_button_enabled=lookup.enabled,
                    post_button_interactive=lookup.interactive,
                    sent_reply_container_found=sent_reply_container_found,
                    sent_reply_item_candidate_count=len(sent_reply_candidates),
                    result="preflight_success",
                    error=None,
                )
            )

            return ReplyPreflightResult(
                approval_id=approval_id,
                reply_target_username=target_username,
                reply_context_verified=True,
                reply_context_method=context_method,
                mention_preserved=prepared["mention_preserved"],
                final_reply_text=final_reply_text,
                post_button_found=True,
                post_button_enabled=True,
                sent_reply_container_found=sent_reply_container_found,
                post_button_selector_method=lookup.selector_method,
                post_button_candidate_count=lookup.candidate_count,
                post_button_visible=lookup.visible,
                post_button_interactive=lookup.interactive,
                sent=False,
                screenshot_path=screenshot_path,
                error=None,
            )

        finally:
            await context.close()
            await playwright.stop()


def build_reply_sender(
    *,
    config_path: str | Path | None = None,
) -> InstagramReplySender:
    session_config = load_session_config(config_path=config_path)
    controller_config = load_reply_controller_config(config_path=config_path)
    sender_config = load_reply_sender_config(config_path=config_path)
    session = InstagramSession(session_config)

    return InstagramReplySender(
        session=session,
        controller_config=controller_config,
        sender_config=sender_config,
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIKO Instagram Reply Sender (Phase 8E / 8E.1 / 8E.2). "
            "--send-approved sends exactly one approved reply for real. "
            "--preflight verifies the send is ready without sending anything."
        )
    )

    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument(
        "--send-approved",
        dest="send_approved",
        metavar="APPROVAL_ID",
        default=None,
        help="Send the approved reply for this approval_id. Requires --confirm.",
    )

    action.add_argument(
        "--preflight",
        dest="preflight",
        metavar="APPROVAL_ID",
        default=None,
        help=(
            "Verify the send is ready for this approval_id without "
            "sending anything. Does not require --confirm."
        ),
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "Mandatory explicit confirmation when using --send-approved. "
            "Not required (and has no effect) with --preflight."
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate instagram.yaml config file.",
    )

    arguments = parser.parse_args(argv)

    if arguments.send_approved and not arguments.confirm:
        parser.error("--confirm is required when using --send-approved")

    return arguments


def _print_result(result: ReplySendResult) -> None:
    print()
    print("AIKO Instagram Reply Sender — REAL SEND")
    print("--------------------------------------------")
    print(f"login status:           {result.login_status}")
    print(f"approval id:            {result.approval_id}")
    print(f"comment found:          {result.comment_found}")
    print(f"reply context verified: {result.reply_context_verified}")
    print(f"reply context method:   {result.reply_context_method}")
    print(f"mention preserved:      {result.mention_preserved}")
    print(f"final reply text:       {result.final_reply_text!r}")
    print(f"post button method:     {result.post_button_selector_method}")
    print(f"post button candidates: {result.post_button_candidate_count}")
    print(f"post button interactive:{result.post_button_interactive}")
    print(f"send clicked:           {result.send_clicked}")
    print(f"send verified:          {result.send_verified}")
    print(f"sent:                   {result.sent}")
    print(f"status:                 {result.status}")
    print(f"before screenshot:      {result.before_send_screenshot_path}")
    print(f"after screenshot:       {result.after_send_screenshot_path}")
    print()


def _print_preflight_result(result: ReplyPreflightResult) -> None:
    print()
    print("AIKO Instagram Reply Sender — Preflight (no send)")
    print("--------------------------------------------------")
    print(f"approval id:                             {result.approval_id}")
    print(f"target username:                         {result.reply_target_username}")
    print(f"reply context verified:                  {result.reply_context_verified}")
    print(f"reply context method:                    {result.reply_context_method}")
    print(f"mention preserved:                       {result.mention_preserved}")
    print(f"final reply text:                        {result.final_reply_text!r}")
    print(f"post button found:                       {result.post_button_found}")
    print(f"post button enabled:                     {result.post_button_enabled}")
    print(f"post button selector method:             {result.post_button_selector_method}")
    print(f"post button candidate count:             {result.post_button_candidate_count}")
    print(f"post button interactive:                 {result.post_button_interactive}")
    print(f"sent-reply verification container found: {result.sent_reply_container_found}")
    print(f"sent:                                    {result.sent}")
    print(f"screenshot path:                         {result.screenshot_path}")
    print()


async def _run(arguments: argparse.Namespace) -> int:
    approval_service = build_approval_service()
    target_approval_id = arguments.send_approved or arguments.preflight

    try:
        record = load_approved_record(
            target_approval_id,
            approval_service=approval_service,
        )
    except ReplyControllerError as exc:
        print(f"[InstagramReplySender] {exc}")
        return 1

    sender = build_reply_sender(config_path=arguments.config)

    if arguments.preflight:
        try:
            preflight_result = await sender.preflight(record)
        except InstagramSessionError as exc:
            print(f"[InstagramReplySender] preflight failed: {exc}")
            return 1

        _print_preflight_result(preflight_result)
        return 0

    try:
        result = await sender.send_reply(record, approval_service=approval_service)
    except InstagramSessionError as exc:
        print(f"[InstagramReplySender] send failed: {exc}")
        return 1

    _print_result(result)
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
