from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .instagram_models import (
    CommentContainerNotFoundError,
    CommentReaderConfig,
    CommentReaderError,
    CommentReaderLoginLostError,
    CommentReadResult,
    CommentRecord,
    CommentSelectors,
    InstagramConfigError,
    InstagramSessionError,
    PostAccessError,
)
from .instagram_session import (
    InstagramSession,
    PlaywrightTimeoutError,
    load_config as load_session_config,
)

# Phase 8B is a read-only comment reader.
#
# This module intentionally does not implement, and must never grow,
# methods for: posting, replying, liking, following, unfollowing,
# hiding, deleting, reporting, or sending direct messages.
DISALLOWED_ACTIONS = (
    "post",
    "reply",
    "like",
    "follow",
    "unfollow",
    "hide",
    "delete",
    "report",
    "send_dm",
)

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "social" / "instagram.yaml"
QUEUE_FILENAME = "comment_queue.json"

_COMMENT_ID_PATTERN = re.compile(r"/c/(?P<comment_id>\d+)/?")
_REPLY_ID_PATTERN = re.compile(r"/c/(?P<parent_id>\d+)/r/(?P<reply_id>\d+)/?")

_POST_URL_PATTERN = re.compile(
    r"^https?://(www\.)?instagram\.com/(p|reel|tv)/[^/?#]+/?",
    re.IGNORECASE,
)


def _runtime_root() -> Path:
    """
    instagram_comment_reader.py location:

    10_apps/claude_runtime/src/social/instagram_comment_reader.py

    parents[2] therefore resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def load_comment_reader_config(
    *,
    config_path: str | Path | None = None,
) -> CommentReaderConfig:
    """
    Load the comment_reader section (and instagram.own_username) from
    config/social/instagram.yaml.

    This is a deliberately independent loader: instagram_session.py is
    not modified by Phase 8B, so this function reads the same YAML
    file on its own rather than reaching into that module's loader.
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
    reader_section = raw.get("comment_reader") or {}

    defaults_section = reader_section.get("defaults") or {}
    output_section = reader_section.get("output") or {}
    selectors_section = reader_section.get("selectors") or {}

    own_username = str(instagram_section.get("own_username", "") or "").strip()

    selectors = CommentSelectors(
        comment_container=list(selectors_section.get("comment_container", [])),
        comment_item=list(selectors_section.get("comment_item", [])),
        comment_username=list(selectors_section.get("comment_username", [])),
        comment_text=list(selectors_section.get("comment_text", [])),
        comment_timestamp=list(selectors_section.get("comment_timestamp", [])),
        comment_permalink=list(selectors_section.get("comment_permalink", [])),
        reply_item=list(selectors_section.get("reply_item", [])),
        load_more_button=list(selectors_section.get("load_more_button", [])),
    )

    queue_dir = str(output_section.get("queue_dir", "output/community/queues"))
    logs_dir = str(output_section.get("logs_dir", "output/community/logs"))
    screenshots_dir = str(
        output_section.get("screenshots_dir", "output/community/screenshots")
    )

    return CommentReaderConfig(
        own_username=own_username,
        selectors=selectors,
        default_limit=int(defaults_section.get("limit", 20)),
        max_scroll_attempts=int(defaults_section.get("max_scroll_attempts", 15)),
        scroll_pause_ms=int(defaults_section.get("scroll_pause_ms", 1200)),
        read_timeout_ms=int(defaults_section.get("read_timeout_ms", 60000)),
        post_navigation_timeout_ms=int(
            defaults_section.get("post_navigation_timeout_ms", 30000)
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


def parse_comment_permalink(
    href: str | None,
) -> tuple[str | None, str | None, bool]:
    """
    Best-effort extraction of (comment_id, parent_comment_id, is_reply)
    from a comment permalink href, where Instagram exposes one.

    Instagram's comment DOM does not reliably expose stable per-comment
    identifiers. Both IDs are legitimately often None ("where
    available"); callers must fall back to compute_stable_id's
    composite hash in that case.
    """
    if not href:
        return None, None, False

    reply_match = _REPLY_ID_PATTERN.search(href)
    if reply_match:
        return (
            reply_match.group("reply_id"),
            reply_match.group("parent_id"),
            True,
        )

    comment_match = _COMMENT_ID_PATTERN.search(href)
    if comment_match:
        return comment_match.group("comment_id"), None, False

    return None, None, False


def compute_stable_id(
    *,
    comment_id: str | None,
    post_url: str,
    username: str | None,
    comment_text: str | None,
) -> str:
    """
    comment_id when available; otherwise hash(post_url + username + comment_text).
    """
    if comment_id:
        return f"comment_id:{comment_id}"

    composite = "|".join(
        [
            post_url.strip(),
            (username or "").strip().lower(),
            (comment_text or "").strip(),
        ]
    )

    digest = hashlib.sha256(composite.encode("utf-8")).hexdigest()
    return f"hash:{digest}"


def is_own_comment(username: str | None, own_username: str | None) -> bool:
    if not username or not own_username:
        return False

    return (
        username.strip().lstrip("@").lower()
        == own_username.strip().lstrip("@").lower()
    )


def build_comment_record(
    raw: dict[str, Any],
    *,
    post_url: str,
    discovered_at: str,
) -> CommentRecord:
    """
    Turn one raw extracted comment dict into a CommentRecord.

    Pure: takes plain strings/booleans only, no Playwright objects.
    """
    username = _clean_text(raw.get("username"))
    comment_text = _clean_text(raw.get("comment_text"))
    timestamp_text = _clean_text(raw.get("timestamp_text"))
    comment_id = _clean_text(raw.get("comment_id"))
    parent_comment_id = _clean_text(raw.get("parent_comment_id"))
    is_reply = bool(raw.get("is_reply", False))

    stable_id = compute_stable_id(
        comment_id=comment_id,
        post_url=post_url,
        username=username,
        comment_text=comment_text,
    )

    error = None if comment_text else "comment_text_not_found"

    return CommentRecord(
        source="instagram_post_comment",
        stable_id=stable_id,
        status="pending",
        discovered_at=discovered_at,
        post_url=post_url,
        username=username,
        comment_text=comment_text,
        timestamp_text=timestamp_text,
        parent_comment_id=parent_comment_id,
        is_reply=is_reply,
        error=error,
        comment_id=comment_id,
    )


def _validate_post_url(post_url: str) -> None:
    if not post_url or not _POST_URL_PATTERN.match(post_url.strip()):
        raise PostAccessError(
            f"'{post_url}' does not look like a valid Instagram post/reel URL "
            "(expected something like https://www.instagram.com/p/<code>/)"
        )


class InstagramCommentReader:
    """
    Read-only Instagram comment reader.

    Responsibilities:
    - reuse the existing persistent InstagramSession (Phase 8A) to open
      a browser against the same logged-in profile
    - open a user-supplied post/reel URL
    - detect logged_in / logged_out before trusting anything on the page
    - extract visible comments, loading more via scroll / "load more"
      up to a configurable limit, timeout, and bounded attempts
    - exclude Aiko's own comments
    - deduplicate against previously queued comments and persist new
      ones to output/community/queues/comment_queue.json

    This class does not post, reply, like, follow, unfollow, hide,
    delete, report, or send direct messages, and never will.
    """

    def __init__(
        self,
        *,
        session: InstagramSession,
        config: CommentReaderConfig,
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

    async def _attr_of(
        self,
        root: Any,
        selectors: list[str],
        name: str,
    ) -> str | None:
        element = await self._find_first(root, selectors)

        if element is None:
            return None

        try:
            return await element.get_attribute(name)
        except Exception:
            return None

    # -- comment extraction -------------------------------------------------

    async def _extract_comment_item(
        self,
        element: Any,
        *,
        parent_comment_id: str | None,
        is_reply: bool,
    ) -> list[dict[str, Any]]:
        selectors = self.config.selectors

        username = await self._text_of(element, selectors.comment_username)
        comment_text = await self._text_of(element, selectors.comment_text)
        timestamp_text = await self._text_of(element, selectors.comment_timestamp)
        comment_href = await self._attr_of(
            element,
            selectors.comment_permalink,
            "href",
        )

        own_comment_id, href_parent_id, href_is_reply = parse_comment_permalink(
            comment_href
        )

        resolved_is_reply = is_reply or href_is_reply
        resolved_parent_id = parent_comment_id or href_parent_id

        records: list[dict[str, Any]] = [
            {
                "username": username,
                "comment_text": comment_text,
                "timestamp_text": timestamp_text,
                "comment_id": own_comment_id,
                "parent_comment_id": resolved_parent_id,
                "is_reply": resolved_is_reply,
            }
        ]

        reply_elements = await self._find_all_first(element, selectors.reply_item)

        for reply_element in reply_elements:
            records.extend(
                await self._extract_comment_item(
                    reply_element,
                    parent_comment_id=own_comment_id,
                    is_reply=True,
                )
            )

        return records

    async def _extract_visible_comments(
        self,
        page: Any,
    ) -> list[dict[str, Any]] | None:
        """
        Returns the list of raw comment dicts currently visible, or
        None if the comment container itself could not be located
        (a distinct condition from "container found, zero comments").
        """
        container = await self._find_first(
            page,
            self.config.selectors.comment_container,
        )

        if container is None:
            return None

        top_level_items = await self._find_all_first(
            container,
            self.config.selectors.comment_item,
        )

        raw_comments: list[dict[str, Any]] = []

        for item in top_level_items:
            raw_comments.extend(
                await self._extract_comment_item(
                    item,
                    parent_comment_id=None,
                    is_reply=False,
                )
            )

        return raw_comments

    async def _click_load_more(self, page: Any) -> bool:
        element = await self._find_first(
            page,
            self.config.selectors.load_more_button,
        )

        if element is None:
            return False

        try:
            await element.click(timeout=2000)
            return True
        except Exception:
            return False

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
        return self.config.logs_dir / "comment_reader_diagnostics.jsonl"

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
        post_url: str,
    ) -> str | None:
        screenshot_path: str | None = None

        try:
            self.config.screenshots_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

            candidate_path = (
                self.config.screenshots_dir / f"{label}_{timestamp}.png"
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
                "post_url": post_url,
                "current_url": getattr(page, "url", None),
                "screenshot_path": screenshot_path,
            }
        )

        return screenshot_path

    # -- queue persistence ----------------------------------------------------

    def _queue_path(self) -> Path:
        return self.config.queue_dir / QUEUE_FILENAME

    def _load_queue(self) -> dict[str, Any]:
        path = self._queue_path()

        if not path.exists():
            return {"version": "1.0", "updated_at": None, "comments": []}

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise CommentReaderError(
                f"Invalid JSON in comment queue: {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise CommentReaderError(
                f"Comment queue must be a JSON object: {path}"
            )

        data.setdefault("version", "1.0")
        data.setdefault("comments", [])

        if not isinstance(data["comments"], list):
            raise CommentReaderError(
                f"Comment queue 'comments' must be a list: {path}"
            )

        return data

    def _save_queue(self, queue: dict[str, Any]) -> None:
        path = self._queue_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_suffix(path.suffix + ".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(queue, file, ensure_ascii=False, indent=2)

        temporary_path.replace(path)

    # -- orchestration ----------------------------------------------------

    async def read_comments(
        self,
        *,
        post_url: str,
        limit: int | None = None,
        max_scroll_attempts: int | None = None,
        timeout_ms: int | None = None,
    ) -> CommentReadResult:
        effective_limit = limit if limit is not None else self.config.default_limit
        effective_attempts = (
            max_scroll_attempts
            if max_scroll_attempts is not None
            else self.config.max_scroll_attempts
        )
        effective_timeout_ms = (
            timeout_ms if timeout_ms is not None else self.config.read_timeout_ms
        )

        _validate_post_url(post_url)

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
                screenshot_path = await self._save_diagnostic(
                    page,
                    label="post_open_failed",
                    message=f"Timed out opening post: {exc}",
                    post_url=post_url,
                )
                raise PostAccessError(
                    f"Timed out opening post: {post_url}. "
                    f"Screenshot: {screenshot_path}"
                ) from exc
            except Exception as exc:
                screenshot_path = await self._save_diagnostic(
                    page,
                    label="post_open_failed",
                    message=f"Unable to open post: {exc}",
                    post_url=post_url,
                )
                raise PostAccessError(
                    f"Unable to open post: {post_url}: {exc}. "
                    f"Screenshot: {screenshot_path}"
                ) from exc

            await page.wait_for_timeout(1500)

            login_status, login_reason = await self._detect_login(page)

            if login_status != "logged_in":
                screenshot_path = await self._save_diagnostic(
                    page,
                    label="login_lost",
                    message=f"status={login_status} reason={login_reason}",
                    post_url=post_url,
                )
                raise CommentReaderLoginLostError(
                    "Instagram session is not logged_in while reading "
                    f"comments (status={login_status}, reason={login_reason}). "
                    f"Screenshot: {screenshot_path}"
                )

            collected: dict[str, CommentRecord] = {}
            attempts = 0
            stagnant_rounds = 0
            previous_count = -1
            start = datetime.now(timezone.utc)

            while True:
                raw_comments = await self._extract_visible_comments(page)

                if raw_comments is None:
                    screenshot_path = await self._save_diagnostic(
                        page,
                        label="comment_container_not_found",
                        message="No comment_container selector matched.",
                        post_url=post_url,
                    )
                    raise CommentContainerNotFoundError(
                        "Could not locate the comment container on "
                        f"{post_url}. Screenshot: {screenshot_path}"
                    )

                discovered_at = self._now()

                for raw in raw_comments:
                    record = build_comment_record(
                        raw,
                        post_url=post_url,
                        discovered_at=discovered_at,
                    )
                    collected[record.stable_id] = record

                elapsed_ms = (
                    datetime.now(timezone.utc) - start
                ).total_seconds() * 1000

                if len(collected) >= effective_limit:
                    break

                if attempts >= effective_attempts:
                    break

                if elapsed_ms >= effective_timeout_ms:
                    break

                if len(collected) == previous_count:
                    stagnant_rounds += 1

                    if stagnant_rounds >= 2:
                        # Two consecutive rounds produced no new
                        # comments: further scrolling is not helping.
                        break
                else:
                    stagnant_rounds = 0

                previous_count = len(collected)

                clicked = await self._click_load_more(page)

                if not clicked:
                    try:
                        await page.mouse.wheel(0, 1600)
                    except Exception:
                        pass

                await page.wait_for_timeout(self.config.scroll_pause_ms)
                attempts += 1

            queue = self._load_queue()
            existing_ids = {
                str(comment.get("stable_id"))
                for comment in queue.get("comments", [])
            }

            new_records: list[CommentRecord] = []
            duplicates_skipped = 0
            excluded_own = 0

            for record in collected.values():
                if is_own_comment(record.username, self.config.own_username):
                    excluded_own += 1
                    continue

                if record.stable_id in existing_ids:
                    duplicates_skipped += 1
                    continue

                existing_ids.add(record.stable_id)
                new_records.append(record)

            queue.setdefault("comments", []).extend(
                record.to_dict() for record in new_records
            )
            queue["version"] = queue.get("version", "1.0")
            queue["updated_at"] = self._now()

            self._save_queue(queue)

            return CommentReadResult(
                login_status=login_status,
                post_url=post_url,
                comments_found=len(collected),
                new_comments_queued=len(new_records),
                duplicates_skipped=duplicates_skipped,
                excluded_own_count=excluded_own,
                queue_path=str(self._queue_path()),
            )

        finally:
            await context.close()
            await playwright.stop()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Instagram Comment Reader (Phase 8B)"
    )

    parser.add_argument(
        "--post-url",
        required=True,
        help="Instagram post or reel URL to read comments from.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of comments to collect (default from config).",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate instagram.yaml config file.",
    )

    return parser.parse_args(argv)


def _print_result(result: CommentReadResult) -> None:
    print()
    print("AIKO Instagram Comment Reader")
    print("------------------------------")
    print(f"login status:       {result.login_status}")
    print(f"post url:           {result.post_url}")
    print(f"comments found:     {result.comments_found}")
    print(f"new comments queued:{result.new_comments_queued}")
    print(f"duplicates skipped: {result.duplicates_skipped}")
    print(f"excluded (own):     {result.excluded_own_count}")
    print(f"queue path:         {result.queue_path}")

    if result.screenshot_path:
        print(f"screenshot:         {result.screenshot_path}")

    print()


async def _run(arguments: argparse.Namespace) -> int:
    session_config = load_session_config(config_path=arguments.config)
    reader_config = load_comment_reader_config(config_path=arguments.config)

    session = InstagramSession(session_config)
    reader = InstagramCommentReader(session=session, config=reader_config)

    try:
        result = await reader.read_comments(
            post_url=arguments.post_url,
            limit=arguments.limit,
        )
    except InstagramSessionError as exc:
        print(f"[InstagramCommentReader] failed: {exc}")
        return 1

    _print_result(result)
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
