from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from .instagram_models import InstagramConfigError, InstagramSessionError
from .instagram_session import (
    InstagramSession,
    PlaywrightTimeoutError,
    load_config as load_session_config,
)

# Phase 8F is a read-only DM inbox reader + reply *preparation* step.
#
# This module intentionally does not implement, and must never grow,
# methods for: sending a direct message, marking a conversation read/
# unread, posting, replying to a comment, liking, following,
# unfollowing, hiding, deleting, or reporting. No DM is ever sent by
# this module.
DISALLOWED_ACTIONS = (
    "send_dm",
    "send_message",
    "mark_read",
    "mark_unread",
    "post",
    "reply",
    "like",
    "follow",
    "unfollow",
    "hide",
    "delete",
    "report",
)

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "social" / "instagram.yaml"
QUEUE_FILENAME = "dm_queue.json"

MessageType = Literal[
    "text",
    "emoji",
    "image",
    "video",
    "reel_share",
    "story_reply",
    "voice",
    "attachment",
    "system",
    "unknown",
]

DMMessageStatus = Literal["pending", "processed", "skipped", "error"]

_EMOJI_ONLY_PATTERN = (
    "[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000026FF"
    "\U00002700-\U000027BF\U0000FE0F\U0000200D]+"
)


class DMReaderError(InstagramSessionError):
    """Base error for the Instagram DM reader."""


class InboxAccessError(DMReaderError):
    """Raised when the DM inbox cannot be opened."""


class ConversationContainerNotFoundError(DMReaderError):
    """Raised when the conversation list cannot be located on the inbox page."""


class MessageThreadNotFoundError(DMReaderError):
    """Raised when a conversation's message thread cannot be parsed."""


class DMReaderLoginLostError(DMReaderError):
    """Raised when the session is not logged_in while reading DMs."""


@dataclass(slots=True)
class DMSelectors:
    """
    Centralised CSS/Playwright selectors for the DM reader.

    All selectors here are UNTESTED placeholder defaults — Phase 8F
    performs no live read (unlike Phase 8B/8D, which validated their
    initial selectors against a real logged-in page). Tune these
    against a real inbox using the screenshot and diagnostic log this
    module writes on failure, the same way comment_reader's and
    reply_controller's selectors were derived.
    """

    inbox_path: str = "direct/inbox/"
    conversation_container: list[str] = field(default_factory=list)
    conversation_item: list[str] = field(default_factory=list)
    conversation_link: list[str] = field(default_factory=list)
    conversation_username: list[str] = field(default_factory=list)
    conversation_display_name: list[str] = field(default_factory=list)
    conversation_unread_indicator: list[str] = field(default_factory=list)
    message_container: list[str] = field(default_factory=list)
    message_item: list[str] = field(default_factory=list)
    message_id_attributes: list[str] = field(default_factory=list)
    message_text: list[str] = field(default_factory=list)
    message_timestamp: list[str] = field(default_factory=list)
    message_own_indicator: list[str] = field(default_factory=list)
    message_image_indicator: list[str] = field(default_factory=list)
    message_video_indicator: list[str] = field(default_factory=list)
    message_voice_indicator: list[str] = field(default_factory=list)
    message_reel_share_indicator: list[str] = field(default_factory=list)
    message_story_reply_indicator: list[str] = field(default_factory=list)
    message_attachment_indicator: list[str] = field(default_factory=list)
    message_system_indicator: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DMReaderConfig:
    """
    Fully-resolved runtime configuration for the DM reader.

    Built by instagram_dm_reader.load_dm_reader_config() from
    config/social/instagram.yaml.
    """

    own_username: str
    selectors: DMSelectors
    default_limit_conversations: int
    default_limit_messages: int
    max_scroll_attempts: int
    scroll_pause_ms: int
    read_timeout_ms: int
    inbox_navigation_timeout_ms: int
    thread_navigation_timeout_ms: int
    queue_dir: Path
    logs_dir: Path
    screenshots_dir: Path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["queue_dir"] = str(self.queue_dir)
        data["logs_dir"] = str(self.logs_dir)
        data["screenshots_dir"] = str(self.screenshots_dir)
        return data


@dataclass(slots=True)
class DMMessageRecord:
    """
    One discovered Instagram DM from a fan, as stored in
    dm_queue.json. classification/proposed_reply/safety_route start
    null and are filled in later by
    dm_approval_service.py --prepare-replies.
    """

    source: str
    stable_id: str
    conversation_id: str | None
    thread_url: str | None
    username: str | None
    display_name: str | None
    message_id: str | None
    message_text: str | None
    timestamp_text: str | None
    message_type: MessageType
    status: DMMessageStatus
    discovered_at: str
    classification: str | None = None
    proposed_reply: str | None = None
    safety_route: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DMReadResult:
    """Summary of one instagram_dm_reader run, as reported by the CLI."""

    login_status: str
    conversations_visited: int
    messages_found: int
    new_messages_queued: int
    duplicates_skipped: int
    excluded_own_count: int
    queue_path: str
    screenshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _runtime_root() -> Path:
    """
    instagram_dm_reader.py location:

    10_apps/claude_runtime/src/social/instagram_dm_reader.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def load_dm_reader_config(
    *,
    config_path: str | Path | None = None,
) -> DMReaderConfig:
    """
    Load the dm_reader section (and instagram.own_username) from
    config/social/instagram.yaml.

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

    instagram_section = raw.get("instagram") or {}
    reader_section = raw.get("dm_reader") or {}

    defaults_section = reader_section.get("defaults") or {}
    output_section = reader_section.get("output") or {}
    selectors_section = reader_section.get("selectors") or {}

    own_username = str(instagram_section.get("own_username", "") or "").strip()

    selectors = DMSelectors(
        inbox_path=str(selectors_section.get("inbox_path", "direct/inbox/")),
        conversation_container=list(
            selectors_section.get("conversation_container", [])
        ),
        conversation_item=list(selectors_section.get("conversation_item", [])),
        conversation_link=list(selectors_section.get("conversation_link", [])),
        conversation_username=list(
            selectors_section.get("conversation_username", [])
        ),
        conversation_display_name=list(
            selectors_section.get("conversation_display_name", [])
        ),
        conversation_unread_indicator=list(
            selectors_section.get("conversation_unread_indicator", [])
        ),
        message_container=list(selectors_section.get("message_container", [])),
        message_item=list(selectors_section.get("message_item", [])),
        message_id_attributes=list(
            selectors_section.get("message_id_attributes", [])
        ),
        message_text=list(selectors_section.get("message_text", [])),
        message_timestamp=list(selectors_section.get("message_timestamp", [])),
        message_own_indicator=list(
            selectors_section.get("message_own_indicator", [])
        ),
        message_image_indicator=list(
            selectors_section.get("message_image_indicator", [])
        ),
        message_video_indicator=list(
            selectors_section.get("message_video_indicator", [])
        ),
        message_voice_indicator=list(
            selectors_section.get("message_voice_indicator", [])
        ),
        message_reel_share_indicator=list(
            selectors_section.get("message_reel_share_indicator", [])
        ),
        message_story_reply_indicator=list(
            selectors_section.get("message_story_reply_indicator", [])
        ),
        message_attachment_indicator=list(
            selectors_section.get("message_attachment_indicator", [])
        ),
        message_system_indicator=list(
            selectors_section.get("message_system_indicator", [])
        ),
    )

    queue_dir = str(output_section.get("queue_dir", "output/community/queues"))
    logs_dir = str(output_section.get("logs_dir", "output/community/logs"))
    screenshots_dir = str(
        output_section.get("screenshots_dir", "output/community/screenshots")
    )

    return DMReaderConfig(
        own_username=own_username,
        selectors=selectors,
        default_limit_conversations=int(
            defaults_section.get("limit_conversations", 10)
        ),
        default_limit_messages=int(defaults_section.get("limit_messages", 50)),
        max_scroll_attempts=int(defaults_section.get("max_scroll_attempts", 15)),
        scroll_pause_ms=int(defaults_section.get("scroll_pause_ms", 1200)),
        read_timeout_ms=int(defaults_section.get("read_timeout_ms", 60000)),
        inbox_navigation_timeout_ms=int(
            defaults_section.get("inbox_navigation_timeout_ms", 30000)
        ),
        thread_navigation_timeout_ms=int(
            defaults_section.get("thread_navigation_timeout_ms", 30000)
        ),
        queue_dir=(runtime_root / queue_dir).resolve(),
        logs_dir=(runtime_root / logs_dir).resolve(),
        screenshots_dir=(runtime_root / screenshots_dir).resolve(),
    )


# ---------------------------------------------------------------------------
# Pure parsing / dedup helpers — no Playwright, no I/O. Fully unit testable.
# ---------------------------------------------------------------------------


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def parse_thread_id(href: str | None) -> str | None:
    """
    Extract a conversation/thread id from a DM thread href, where
    Instagram exposes one: .../direct/t/<thread_id>/...
    """
    if not href:
        return None

    marker = "/direct/t/"
    index = href.find(marker)

    if index == -1:
        return None

    remainder = href[index + len(marker) :]
    thread_id = remainder.split("/")[0].split("?")[0].split("#")[0]

    return thread_id or None


def is_emoji_only_text(text: str) -> bool:
    if not text.strip():
        return False

    compiled = re.compile(_EMOJI_ONLY_PATTERN, flags=re.UNICODE)
    stripped = compiled.sub("", text)

    return not stripped.strip()


def determine_message_type(raw: dict[str, Any]) -> MessageType:
    """
    Pure classification of one raw extracted message dict into one of
    the 10 supported message_type values.

    Media-flag signals take priority over text content, since e.g. an
    image message can carry emoji-only caption text — the underlying
    media is what determines the type, not incidental caption text.
    """
    if raw.get("is_voice"):
        return "voice"

    if raw.get("is_video"):
        return "video"

    if raw.get("is_reel_share"):
        return "reel_share"

    if raw.get("is_story_reply"):
        return "story_reply"

    if raw.get("is_image"):
        return "image"

    if raw.get("is_attachment"):
        return "attachment"

    if raw.get("is_system"):
        return "system"

    text = _clean_text(raw.get("message_text"))

    if text is None:
        return "unknown"

    if is_emoji_only_text(text):
        return "emoji"

    return "text"


def is_aiko_message(raw: dict[str, Any], *, own_username: str | None = None) -> bool:
    """
    True when a raw extracted message was sent by Aiko herself
    (requirement 5: these are always excluded, never queued).

    "No signal found" resolves to False (message is KEPT) rather than
    True (message DROPPED): a false negative here means a genuine fan
    message survives for human review, which is the safer failure
    mode than silently discarding one.
    """
    if bool(raw.get("is_own_message")):
        return True

    sender_username = _clean_text(raw.get("sender_username"))

    if sender_username and own_username:
        return (
            sender_username.strip().lstrip("@").lower()
            == own_username.strip().lstrip("@").lower()
        )

    return False


def compute_dm_stable_id(
    *,
    message_id: str | None,
    conversation_id: str | None,
    username: str | None,
    message_text: str | None,
    timestamp_text: str | None,
) -> str:
    """
    message_id when available; otherwise
    hash(conversation_id + username + message_text + timestamp_text).
    """
    if message_id:
        return f"message_id:{message_id}"

    composite = "|".join(
        [
            (conversation_id or "").strip(),
            (username or "").strip().lower(),
            (message_text or "").strip(),
            (timestamp_text or "").strip(),
        ]
    )

    digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()
    return f"hash:{digest}"


def build_dm_message_record(
    raw: dict[str, Any],
    *,
    conversation_id: str | None,
    thread_url: str | None,
    discovered_at: str,
) -> DMMessageRecord:
    """
    Turn one raw extracted message dict into a DMMessageRecord.

    Pure: takes plain strings/booleans only, no Playwright objects.
    """
    username = _clean_text(raw.get("sender_username"))
    display_name = _clean_text(raw.get("sender_display_name"))
    message_text = _clean_text(raw.get("message_text"))
    timestamp_text = _clean_text(raw.get("timestamp_text"))
    message_id = _clean_text(raw.get("message_id"))

    stable_id = compute_dm_stable_id(
        message_id=message_id,
        conversation_id=conversation_id,
        username=username,
        message_text=message_text,
        timestamp_text=timestamp_text,
    )

    message_type = determine_message_type(raw)
    error = None if (message_text or message_type != "unknown") else "message_content_not_found"

    return DMMessageRecord(
        source="instagram_dm",
        stable_id=stable_id,
        conversation_id=conversation_id,
        thread_url=thread_url,
        username=username,
        display_name=display_name,
        message_id=message_id,
        message_text=message_text,
        timestamp_text=timestamp_text,
        message_type=message_type,
        status="pending",
        discovered_at=discovered_at,
        classification=None,
        proposed_reply=None,
        safety_route=None,
        error=error,
    )


class InstagramDMReader:
    """
    Read-only Instagram DM inbox reader.

    Responsibilities:
    - reuse the existing persistent InstagramSession (Phase 8A) to
      open a browser against the same logged-in profile
    - open the DM inbox, detect logged_in / logged_out before
      trusting anything on the page
    - list conversations (unread ones first), up to --limit-
      conversations
    - open each conversation's thread and extract fan messages, up to
      --limit-messages per conversation
    - exclude Aiko's own messages
    - deduplicate against previously queued messages and persist new
      ones to output/community/queues/dm_queue.json

    This class does not send a DM, mark anything read/unread, post,
    reply, like, follow, unfollow, hide, delete, or report, and never
    will.
    """

    def __init__(
        self,
        *,
        session: InstagramSession,
        config: DMReaderConfig,
    ) -> None:
        self.session = session
        self.config = config

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- generic selector helpers -----------------------------------------

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

    async def _text_of(self, root: Any, selectors: list[str]) -> str | None:
        element = await self._find_first(root, selectors)

        if element is None:
            return None

        try:
            return await element.inner_text()
        except Exception:
            return None

    async def _attr_of(self, root: Any, selectors: list[str], name: str) -> str | None:
        element = await self._find_first(root, selectors)

        if element is None:
            return None

        try:
            return await element.get_attribute(name)
        except Exception:
            return None

    async def _exists(self, root: Any, selectors: list[str]) -> bool:
        return await self._find_first(root, selectors) is not None

    async def _is_right_aligned(self, element: Any) -> bool:
        """
        True when `element`'s bounding box sits right of the viewport
        center — the standard chat-UI convention (confirmed live
        2026-07-31 against a real thread) for an outgoing (Aiko-sent)
        message, versus a left-aligned incoming (fan) one. Instagram's
        real DM markup carries no distinguishing class/attribute for
        this — see message_own_indicator's comment in
        config/social/instagram.yaml — so this is the PRIMARY own-vs-
        fan signal, not just a fallback.

        Returns False (message treated as NOT Aiko's own, i.e. kept)
        on any error or missing box — the same "no signal -> keep it"
        safe default used everywhere else in this module.
        """
        try:
            box = await element.bounding_box()
        except Exception:
            return False

        if not box:
            return False

        viewport_width = self.session.config.viewport.width
        center_x = box["x"] + box["width"] / 2

        return center_x > (viewport_width / 2)

    async def _message_id_of(self, element: Any) -> str | None:
        for attribute in self.config.selectors.message_id_attributes:
            try:
                value = await element.get_attribute(attribute)
            except Exception:
                value = None

            if value:
                return value

        return None

    async def _detect_login(self, page: Any) -> tuple[str, str]:
        """
        Reuse Phase 8A's classification logic against this page,
        without duplicating InstagramSession's private selector-scan
        internals.
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

    # -- diagnostics ----------------------------------------------------------

    def _diagnostic_log_path(self) -> Path:
        return self.config.logs_dir / "dm_reader_diagnostics.jsonl"

    def _append_diagnostic_log(self, entry: dict[str, Any]) -> None:
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        with self._diagnostic_log_path().open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def _save_diagnostic(
        self,
        page: Any,
        *,
        label: str,
        message: str,
        context_url: str | None = None,
    ) -> str | None:
        screenshot_path: str | None = None

        try:
            self.config.screenshots_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

            candidate_path = (
                self.config.screenshots_dir / f"dm_{label}_{timestamp}.png"
            )

            await page.screenshot(path=str(candidate_path), full_page=True)
            screenshot_path = str(candidate_path)
        except Exception:
            screenshot_path = None

        self._append_diagnostic_log(
            {
                "timestamp": self._now(),
                "label": label,
                "message": message,
                "context_url": context_url,
                "current_url": getattr(page, "url", None),
                "screenshot_path": screenshot_path,
            }
        )

        return screenshot_path

    # -- conversation extraction ----------------------------------------------

    async def _extract_conversation_item(self, element: Any) -> dict[str, Any]:
        selectors = self.config.selectors

        username = await self._text_of(element, selectors.conversation_username)
        display_name = await self._text_of(
            element, selectors.conversation_display_name
        )
        href = await self._attr_of(element, selectors.conversation_link, "href")
        is_unread = await self._exists(
            element, selectors.conversation_unread_indicator
        )

        return {
            "username": username,
            "display_name": display_name,
            "href": href,
            "is_unread": is_unread,
        }

    async def _extract_visible_conversations(
        self,
        page: Any,
    ) -> list[dict[str, Any]] | None:
        """
        Returns raw conversation dicts currently visible, or None if
        the conversation container itself could not be located (a
        distinct condition from "container found, zero conversations").
        """
        container = await self._find_first(
            page, self.config.selectors.conversation_container
        )

        if container is None:
            return None

        items = await self._find_all_first(
            container, self.config.selectors.conversation_item
        )

        return [await self._extract_conversation_item(item) for item in items]

    # -- message extraction -------------------------------------------------

    async def _extract_message_item(self, element: Any) -> dict[str, Any]:
        selectors = self.config.selectors

        text_element = await self._find_first(element, selectors.message_text)
        message_text: str | None = None
        is_own_by_position = False

        if text_element is not None:
            try:
                message_text = await text_element.inner_text()
            except Exception:
                message_text = None

            is_own_by_position = await self._is_right_aligned(text_element)

        timestamp_text = await self._text_of(element, selectors.message_timestamp)
        message_id = await self._message_id_of(element)

        is_own_message = (
            await self._exists(element, selectors.message_own_indicator)
            or is_own_by_position
        )

        return {
            "message_id": message_id,
            "message_text": message_text,
            "timestamp_text": timestamp_text,
            "is_own_message": is_own_message,
            "sender_username": None,
            "sender_display_name": None,
            "is_image": await self._exists(element, selectors.message_image_indicator),
            "is_video": await self._exists(element, selectors.message_video_indicator),
            "is_voice": await self._exists(element, selectors.message_voice_indicator),
            "is_reel_share": await self._exists(
                element, selectors.message_reel_share_indicator
            ),
            "is_story_reply": await self._exists(
                element, selectors.message_story_reply_indicator
            ),
            "is_attachment": await self._exists(
                element, selectors.message_attachment_indicator
            ),
            "is_system": await self._exists(element, selectors.message_system_indicator),
        }

    async def _extract_visible_messages(
        self,
        page: Any,
    ) -> list[dict[str, Any]] | None:
        container = await self._find_first(
            page, self.config.selectors.message_container
        )

        if container is None:
            return None

        items = await self._find_all_first(container, self.config.selectors.message_item)

        return [await self._extract_message_item(item) for item in items]

    # -- queue persistence ----------------------------------------------------

    def _queue_path(self) -> Path:
        return self.config.queue_dir / QUEUE_FILENAME

    def _load_queue(self) -> dict[str, Any]:
        path = self._queue_path()

        if not path.exists():
            return {"version": "1.0", "updated_at": None, "messages": []}

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise DMReaderError(f"Invalid JSON in DM queue: {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise DMReaderError(f"DM queue must be a JSON object: {path}")

        data.setdefault("version", "1.0")
        data.setdefault("messages", [])

        if not isinstance(data["messages"], list):
            raise DMReaderError(f"DM queue 'messages' must be a list: {path}")

        return data

    def _save_queue(self, queue: dict[str, Any]) -> None:
        path = self._queue_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_suffix(path.suffix + ".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(queue, file, ensure_ascii=False, indent=2)

        temporary_path.replace(path)

    # -- orchestration ----------------------------------------------------

    async def _collect_conversations(
        self,
        page: Any,
        *,
        inbox_url: str,
        limit_conversations: int,
        max_scroll_attempts: int,
        timeout_ms: int,
    ) -> list[dict[str, Any]]:
        collected: dict[str, dict[str, Any]] = {}
        attempts = 0
        stagnant_rounds = 0
        previous_count = -1
        start = datetime.now(timezone.utc)

        while True:
            raw_conversations = await self._extract_visible_conversations(page)

            if raw_conversations is None:
                screenshot_path = await self._save_diagnostic(
                    page,
                    label="conversation_container_not_found",
                    message="No conversation_container selector matched.",
                    context_url=inbox_url,
                )
                raise ConversationContainerNotFoundError(
                    "Could not locate the conversation list on the inbox. "
                    f"Screenshot: {screenshot_path}"
                )

            for raw in raw_conversations:
                thread_id = parse_thread_id(raw.get("href"))
                key = thread_id or raw.get("href") or f"{raw.get('username')}"
                collected[key] = raw

            elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if len(collected) >= limit_conversations:
                break

            if attempts >= max_scroll_attempts:
                break

            if elapsed_ms >= timeout_ms:
                break

            if len(collected) == previous_count:
                stagnant_rounds += 1

                if stagnant_rounds >= 2:
                    break
            else:
                stagnant_rounds = 0

            previous_count = len(collected)

            try:
                await page.mouse.wheel(0, 1600)
            except Exception:
                pass

            await page.wait_for_timeout(self.config.scroll_pause_ms)
            attempts += 1

        # Unread-first, otherwise preserve discovery order (stable sort).
        ordered = sorted(
            collected.values(),
            key=lambda conversation: 0 if conversation.get("is_unread") else 1,
        )

        return ordered[:limit_conversations]

    async def _open_conversation_by_click(
        self,
        page: Any,
        *,
        inbox_url: str,
        username: str | None,
    ) -> str | None:
        """
        Navigate into a specific conversation by re-opening the inbox
        and clicking the row matching `username`.

        Confirmed live 2026-07-31: Instagram's real conversation rows
        carry no href at all — only a client-side click handler — so
        this is the actual navigation path, not just a fallback.
        Element handles from the original conversation-list pass are
        never reused here: navigating to a different conversation and
        back invalidates them, so the list is re-extracted fresh each
        time and the target row is re-located by username. Matching
        by username assumes conversation usernames are unique in the
        visible list, which holds for Instagram (usernames are unique
        account identifiers) but would misfire on two distinct
        conversations that render an identical display string (e.g.
        two different "Instagram User" placeholder rows) — an
        accepted, documented limitation rather than a silent one.

        Returns the resulting thread URL, or None if the inbox can't
        be reopened or the row can't be re-located.
        """
        if not username:
            return None

        try:
            await page.goto(
                inbox_url,
                timeout=self.config.inbox_navigation_timeout_ms,
                wait_until="domcontentloaded",
            )
        except Exception:
            return None

        await page.wait_for_timeout(800)

        container = await self._find_first(
            page, self.config.selectors.conversation_container
        )

        if container is None:
            return None

        items = await self._find_all_first(
            container, self.config.selectors.conversation_item
        )

        for item in items:
            item_username = await self._text_of(
                item, self.config.selectors.conversation_username
            )

            if item_username is None:
                continue

            if item_username.strip() != username.strip():
                continue

            try:
                await item.click(timeout=3000)
            except Exception:
                return None

            await page.wait_for_timeout(1200)
            return page.url

        return None

    async def _collect_thread_messages(
        self,
        page: Any,
        *,
        thread_url: str,
        limit_messages: int,
        max_scroll_attempts: int,
        timeout_ms: int,
    ) -> list[dict[str, Any]]:
        collected: dict[int, dict[str, Any]] = {}
        attempts = 0
        stagnant_rounds = 0
        previous_count = -1
        start = datetime.now(timezone.utc)

        while True:
            raw_messages = await self._extract_visible_messages(page)

            if raw_messages is None:
                screenshot_path = await self._save_diagnostic(
                    page,
                    label="message_thread_not_found",
                    message="No message_container selector matched.",
                    context_url=thread_url,
                )
                raise MessageThreadNotFoundError(
                    f"Could not parse the message thread at {thread_url}. "
                    f"Screenshot: {screenshot_path}"
                )

            for index, raw in enumerate(raw_messages):
                collected[index] = raw

            elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            if len(collected) >= limit_messages:
                break

            if attempts >= max_scroll_attempts:
                break

            if elapsed_ms >= timeout_ms:
                break

            if len(collected) == previous_count:
                stagnant_rounds += 1

                if stagnant_rounds >= 2:
                    break
            else:
                stagnant_rounds = 0

            previous_count = len(collected)

            try:
                await page.mouse.wheel(0, 1600)
            except Exception:
                pass

            await page.wait_for_timeout(self.config.scroll_pause_ms)
            attempts += 1

        ordered_keys = sorted(collected.keys())
        return [collected[key] for key in ordered_keys][:limit_messages]

    async def read_messages(
        self,
        *,
        limit_conversations: int | None = None,
        limit_messages: int | None = None,
        max_scroll_attempts: int | None = None,
        timeout_ms: int | None = None,
    ) -> DMReadResult:
        effective_limit_conversations = (
            limit_conversations
            if limit_conversations is not None
            else self.config.default_limit_conversations
        )
        effective_limit_messages = (
            limit_messages
            if limit_messages is not None
            else self.config.default_limit_messages
        )
        effective_attempts = (
            max_scroll_attempts
            if max_scroll_attempts is not None
            else self.config.max_scroll_attempts
        )
        effective_timeout_ms = (
            timeout_ms if timeout_ms is not None else self.config.read_timeout_ms
        )

        base_url = self.session.config.base_url.rstrip("/")
        inbox_path = self.config.selectors.inbox_path.strip("/")
        inbox_url = f"{base_url}/{inbox_path}/"

        playwright, context = await self.session._open_context()

        try:
            page = context.pages[0] if context.pages else await context.new_page()

            try:
                await page.goto(
                    inbox_url,
                    timeout=self.config.inbox_navigation_timeout_ms,
                    wait_until="domcontentloaded",
                )
            except PlaywrightTimeoutError as exc:
                screenshot_path = await self._save_diagnostic(
                    page,
                    label="inbox_open_failed",
                    message=f"Timed out opening inbox: {exc}",
                    context_url=inbox_url,
                )
                raise InboxAccessError(
                    f"Timed out opening inbox: {inbox_url}. "
                    f"Screenshot: {screenshot_path}"
                ) from exc
            except Exception as exc:
                screenshot_path = await self._save_diagnostic(
                    page,
                    label="inbox_open_failed",
                    message=f"Unable to open inbox: {exc}",
                    context_url=inbox_url,
                )
                raise InboxAccessError(
                    f"Unable to open inbox: {inbox_url}: {exc}. "
                    f"Screenshot: {screenshot_path}"
                ) from exc

            await page.wait_for_timeout(1500)

            login_status, login_reason = await self._detect_login(page)

            if login_status != "logged_in":
                screenshot_path = await self._save_diagnostic(
                    page,
                    label="login_lost",
                    message=f"status={login_status} reason={login_reason}",
                    context_url=inbox_url,
                )
                raise DMReaderLoginLostError(
                    "Instagram session is not logged_in while reading DMs "
                    f"(status={login_status}, reason={login_reason}). "
                    f"Screenshot: {screenshot_path}"
                )

            conversations = await self._collect_conversations(
                page,
                inbox_url=inbox_url,
                limit_conversations=effective_limit_conversations,
                max_scroll_attempts=effective_attempts,
                timeout_ms=effective_timeout_ms,
            )

            queue = self._load_queue()
            existing_ids = {
                str(message.get("stable_id")) for message in queue.get("messages", [])
            }

            all_raw_messages: list[dict[str, Any]] = []
            new_records: list[DMMessageRecord] = []
            duplicates_skipped = 0
            excluded_own = 0

            for conversation in conversations:
                href = conversation.get("href")
                username = conversation.get("username")

                if href:
                    # Fast path, kept for forward-compatibility: if
                    # Instagram ever exposes a real href for a
                    # conversation row, use it directly instead of
                    # clicking. Confirmed live 2026-07-31 that real
                    # rows do NOT carry one today — see
                    # _open_conversation_by_click below, the actual
                    # path exercised in practice.
                    thread_url = (
                        href if href.startswith("http") else f"{base_url}{href}"
                    )

                    try:
                        await page.goto(
                            thread_url,
                            timeout=self.config.thread_navigation_timeout_ms,
                            wait_until="domcontentloaded",
                        )
                    except Exception as exc:
                        await self._save_diagnostic(
                            page,
                            label="thread_open_failed",
                            message=f"Unable to open conversation thread: {exc}",
                            context_url=thread_url,
                        )
                        continue

                    await page.wait_for_timeout(800)
                else:
                    thread_url = await self._open_conversation_by_click(
                        page,
                        inbox_url=inbox_url,
                        username=username,
                    )

                    if thread_url is None:
                        await self._save_diagnostic(
                            page,
                            label="thread_open_failed",
                            message=(
                                "Could not re-locate the conversation row for "
                                f"@{username} to click into it."
                            ),
                            context_url=inbox_url,
                        )
                        continue

                thread_id = parse_thread_id(thread_url)

                raw_messages = await self._collect_thread_messages(
                    page,
                    thread_url=thread_url,
                    limit_messages=effective_limit_messages,
                    max_scroll_attempts=effective_attempts,
                    timeout_ms=effective_timeout_ms,
                )

                if not raw_messages:
                    continue

                discovered_at = self._now()

                for raw in raw_messages:
                    all_raw_messages.append(raw)

                    if raw.get("sender_username") is None:
                        raw = {**raw, "sender_username": conversation.get("username")}

                    if raw.get("sender_display_name") is None:
                        raw = {
                            **raw,
                            "sender_display_name": conversation.get("display_name"),
                        }

                    if is_aiko_message(raw, own_username=self.config.own_username):
                        excluded_own += 1
                        continue

                    record = build_dm_message_record(
                        raw,
                        conversation_id=thread_id,
                        thread_url=thread_url,
                        discovered_at=discovered_at,
                    )

                    if record.stable_id in existing_ids:
                        duplicates_skipped += 1
                        continue

                    existing_ids.add(record.stable_id)
                    new_records.append(record)

            queue.setdefault("messages", []).extend(
                record.to_dict() for record in new_records
            )
            queue["version"] = queue.get("version", "1.0")
            queue["updated_at"] = self._now()

            self._save_queue(queue)

            return DMReadResult(
                login_status=login_status,
                conversations_visited=len(conversations),
                messages_found=len(all_raw_messages),
                new_messages_queued=len(new_records),
                duplicates_skipped=duplicates_skipped,
                excluded_own_count=excluded_own,
                queue_path=str(self._queue_path()),
            )

        finally:
            await context.close()
            await playwright.stop()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Instagram Fans DM Reader (Phase 8F)"
    )

    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to visit (default from config).",
    )

    parser.add_argument(
        "--limit-messages",
        type=int,
        default=None,
        help="Maximum number of messages to read per conversation (default from config).",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate instagram.yaml config file.",
    )

    return parser.parse_args(argv)


def _print_result(result: DMReadResult) -> None:
    print()
    print("AIKO Instagram DM Reader")
    print("--------------------------")
    print(f"login status:         {result.login_status}")
    print(f"conversations visited: {result.conversations_visited}")
    print(f"messages found:        {result.messages_found}")
    print(f"new messages queued:   {result.new_messages_queued}")
    print(f"duplicates skipped:    {result.duplicates_skipped}")
    print(f"excluded (own):        {result.excluded_own_count}")
    print(f"queue path:            {result.queue_path}")

    if result.screenshot_path:
        print(f"screenshot:            {result.screenshot_path}")

    print()


async def _run(arguments: argparse.Namespace) -> int:
    session_config = load_session_config(config_path=arguments.config)
    reader_config = load_dm_reader_config(config_path=arguments.config)

    session = InstagramSession(session_config)
    reader = InstagramDMReader(session=session, config=reader_config)

    try:
        result = await reader.read_messages(
            limit_conversations=arguments.limit_conversations,
            limit_messages=arguments.limit_messages,
        )
    except InstagramSessionError as exc:
        print(f"[InstagramDMReader] failed: {exc}")
        return 1

    _print_result(result)
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
