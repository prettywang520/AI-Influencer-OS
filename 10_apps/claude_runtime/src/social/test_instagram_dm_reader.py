from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from .instagram_dm_reader import (
    ConversationContainerNotFoundError,
    DISALLOWED_ACTIONS,
    DMReaderError,
    DMReaderLoginLostError,
    InboxAccessError,
    InstagramDMReader,
    MessageThreadNotFoundError,
    build_dm_message_record,
    compute_dm_stable_id,
    determine_message_type,
    is_aiko_message,
    load_dm_reader_config,
    parse_arguments,
    parse_thread_id,
)
from .instagram_models import LoginDetectionConfig

# None of these tests open a real browser or contact Instagram.
# Playwright-shaped objects are hand-built fakes; everything else is
# pure functions or local tempfile I/O.


# ---------------------------------------------------------------------------
# Fake Playwright-shaped objects
# ---------------------------------------------------------------------------


class FakeElement:
    """
    Minimal stand-in for a Playwright ElementHandle, exposing only the
    async methods InstagramDMReader actually calls.
    """

    def __init__(
        self,
        *,
        text: str = "",
        attrs: dict[str, str] | None = None,
        children: dict[str, "FakeElement"] | None = None,
        child_lists: dict[str, list["FakeElement"]] | None = None,
        page: "FakePage | None" = None,
        click_navigates_to: str | None = None,
        fail_click: bool = False,
        bounding_box: dict | None = None,
    ) -> None:
        self._text = text
        self._attrs = attrs or {}
        self._children = children or {}
        self._child_lists = child_lists or {}
        self._page = page
        self._click_navigates_to = click_navigates_to
        self._fail_click = fail_click
        self._bounding_box = bounding_box
        self.clicked = False
        self.click_count = 0

    async def query_selector(self, selector: str):
        return self._children.get(selector)

    async def query_selector_all(self, selector: str):
        return self._child_lists.get(selector, [])

    async def inner_text(self) -> str:
        return self._text

    async def get_attribute(self, name: str):
        return self._attrs.get(name)

    async def bounding_box(self):
        return self._bounding_box

    async def click(self, timeout=None):
        """
        Simulates Instagram's real click-driven conversation navigation
        (no real <a href> exists — see _open_conversation_by_click in
        instagram_dm_reader.py): updates the bound FakePage's url/route
        exactly as a real click would, IF this element was built with
        page= and click_navigates_to= (i.e. it represents a
        conversation row). Otherwise a no-op click, same as any other
        element that happens to get click()'d.
        """
        if self._fail_click:
            raise RuntimeError("simulated click failure")

        self.clicked = True
        self.click_count += 1

        if self._page is not None and self._click_navigates_to is not None:
            self._page.url = self._click_navigates_to
            self._page._apply_route(self._click_navigates_to)


class FakePage:
    """
    Stand-in for a Playwright Page that can serve different content
    per URL (inbox vs. each conversation's thread), via a route table
    keyed by exact URL string.
    """

    def __init__(
        self,
        *,
        routes: dict[str, dict],
        start_url: str = "https://www.instagram.com/direct/inbox/",
    ) -> None:
        self._routes = routes
        self.url = start_url
        self._children: dict[str, FakeElement] = {}
        self._child_lists: dict[str, list[FakeElement]] = {}
        self._apply_route(start_url)
        self.screenshot_calls: list[str] = []

        outer = self

        class _Mouse:
            async def wheel(self, x, y):
                pass

        self.mouse = _Mouse()

    def _apply_route(self, url: str) -> None:
        route = self._routes.get(url, {})
        self._children = route.get("children", {})
        self._child_lists = route.get("child_lists", {})

    async def goto(self, url, timeout=None, wait_until=None):
        if url not in self._routes:
            raise RuntimeError(f"no fake route registered for {url}")

        self.url = url
        self._apply_route(url)

    async def wait_for_timeout(self, ms):
        pass

    async def query_selector(self, selector: str):
        return self._children.get(selector)

    async def query_selector_all(self, selector: str):
        return self._child_lists.get(selector, [])

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


class FakeViewportConfig:
    def __init__(self, width: int = 1280) -> None:
        self.width = width


class FakeSessionConfig:
    def __init__(self) -> None:
        self.base_url = "https://www.instagram.com/"
        self.viewport = FakeViewportConfig(width=1280)
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


def _reader_config():
    return load_dm_reader_config()


def _selectors():
    return _reader_config().selectors


def _make_reader(session: FakeSession, **config_overrides) -> InstagramDMReader:
    config = _reader_config()

    for key, value in config_overrides.items():
        object.__setattr__(config, key, value)

    return InstagramDMReader(session=session, config=config)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class ParseThreadIdTests(unittest.TestCase):
    def test_none_href_returns_none(self) -> None:
        self.assertIsNone(parse_thread_id(None))

    def test_extracts_thread_id(self) -> None:
        self.assertEqual(
            parse_thread_id("https://www.instagram.com/direct/t/17895512345/"),
            "17895512345",
        )

    def test_extracts_thread_id_relative_href(self) -> None:
        self.assertEqual(parse_thread_id("/direct/t/abc123/"), "abc123")

    def test_no_marker_returns_none(self) -> None:
        self.assertIsNone(parse_thread_id("https://www.instagram.com/p/ABC123/"))

    def test_strips_query_string(self) -> None:
        self.assertEqual(
            parse_thread_id("/direct/t/abc123/?a=1"), "abc123"
        )


class DetermineMessageTypeTests(unittest.TestCase):
    def test_plain_text(self) -> None:
        self.assertEqual(
            determine_message_type({"message_text": "hey there!"}), "text"
        )

    def test_emoji_only(self) -> None:
        self.assertEqual(
            determine_message_type({"message_text": "😍🌸"}), "emoji"
        )

    def test_image(self) -> None:
        self.assertEqual(
            determine_message_type({"message_text": None, "is_image": True}),
            "image",
        )

    def test_video_takes_priority_over_text(self) -> None:
        self.assertEqual(
            determine_message_type({"message_text": "check this out", "is_video": True}),
            "video",
        )

    def test_voice(self) -> None:
        self.assertEqual(determine_message_type({"is_voice": True}), "voice")

    def test_reel_share(self) -> None:
        self.assertEqual(
            determine_message_type({"is_reel_share": True}), "reel_share"
        )

    def test_story_reply(self) -> None:
        self.assertEqual(
            determine_message_type({"is_story_reply": True}), "story_reply"
        )

    def test_attachment(self) -> None:
        self.assertEqual(
            determine_message_type({"is_attachment": True}), "attachment"
        )

    def test_system(self) -> None:
        self.assertEqual(determine_message_type({"is_system": True}), "system")

    def test_unknown_media_attachment_with_no_flags_and_no_text(self) -> None:
        self.assertEqual(determine_message_type({}), "unknown")

    def test_media_flag_takes_priority_over_emoji_text(self) -> None:
        self.assertEqual(
            determine_message_type({"message_text": "🤍", "is_image": True}),
            "image",
        )


class IsAikoMessageTests(unittest.TestCase):
    def test_own_message_flag_true(self) -> None:
        self.assertTrue(is_aiko_message({"is_own_message": True}))

    def test_username_matches_own_username(self) -> None:
        self.assertTrue(
            is_aiko_message(
                {"sender_username": "@aikotraveldiary"},
                own_username="aikotraveldiary",
            )
        )

    def test_username_case_and_at_insensitive(self) -> None:
        self.assertTrue(
            is_aiko_message(
                {"sender_username": "AikoTravelDiary"},
                own_username="@aikotraveldiary",
            )
        )

    def test_fan_message_is_false(self) -> None:
        self.assertFalse(
            is_aiko_message(
                {"sender_username": "some_fan_123", "is_own_message": False},
                own_username="aikotraveldiary",
            )
        )

    def test_no_signal_defaults_to_false_message_kept(self) -> None:
        """
        A false negative here (keeping a genuine Aiko message) is
        preferable to a false positive (silently dropping a real fan
        message) — see instagram_dm_reader.is_aiko_message docstring.
        """
        self.assertFalse(is_aiko_message({}))


class ComputeDmStableIdTests(unittest.TestCase):
    def test_uses_message_id_when_available(self) -> None:
        stable_id = compute_dm_stable_id(
            message_id="998877",
            conversation_id="conv-1",
            username="fan_one",
            message_text="hi",
            timestamp_text="2h",
        )
        self.assertEqual(stable_id, "message_id:998877")

    def test_falls_back_to_composite_hash(self) -> None:
        stable_id = compute_dm_stable_id(
            message_id=None,
            conversation_id="conv-1",
            username="fan_one",
            message_text="hi there",
            timestamp_text="2h",
        )
        self.assertTrue(stable_id.startswith("hash:"))

    def test_hash_is_deterministic(self) -> None:
        kwargs = dict(
            message_id=None,
            conversation_id="conv-1",
            username="fan_one",
            message_text="hi there",
            timestamp_text="2h",
        )
        self.assertEqual(compute_dm_stable_id(**kwargs), compute_dm_stable_id(**kwargs))

    def test_hash_differs_for_different_text(self) -> None:
        base = dict(
            message_id=None, conversation_id="conv-1", username="fan_one", timestamp_text="2h"
        )
        first = compute_dm_stable_id(message_text="hi there", **base)
        second = compute_dm_stable_id(message_text="something else", **base)
        self.assertNotEqual(first, second)


class BuildDmMessageRecordTests(unittest.TestCase):
    def test_builds_pending_record_with_null_classification_fields(self) -> None:
        record = build_dm_message_record(
            {
                "message_id": "1",
                "message_text": "you're so beautiful!",
                "timestamp_text": "2h",
                "sender_username": "fan_one",
                "sender_display_name": "Fan One",
            },
            conversation_id="conv-1",
            thread_url="https://www.instagram.com/direct/t/conv-1/",
            discovered_at="2026-07-31T00:00:00Z",
        )

        self.assertEqual(record.source, "instagram_dm")
        self.assertEqual(record.status, "pending")
        self.assertEqual(record.message_type, "text")
        self.assertIsNone(record.classification)
        self.assertIsNone(record.proposed_reply)
        self.assertIsNone(record.safety_route)
        self.assertIsNone(record.error)
        self.assertEqual(record.stable_id, "message_id:1")

    def test_emoji_only_message_type(self) -> None:
        record = build_dm_message_record(
            {"message_text": "😍😍", "sender_username": "fan_one"},
            conversation_id="conv-1",
            thread_url="https://x/t/conv-1/",
            discovered_at="2026-07-31T00:00:00Z",
        )
        self.assertEqual(record.message_type, "emoji")

    def test_unknown_media_attachment_gets_error_flag(self) -> None:
        record = build_dm_message_record(
            {"message_text": None},
            conversation_id="conv-1",
            thread_url="https://x/t/conv-1/",
            discovered_at="2026-07-31T00:00:00Z",
        )
        self.assertEqual(record.message_type, "unknown")
        self.assertIsNotNone(record.error)


class LoadDMReaderConfigTests(unittest.TestCase):
    def test_loads_real_config(self) -> None:
        config = _reader_config()

        self.assertEqual(config.selectors.inbox_path, "direct/inbox/")
        self.assertGreater(len(config.selectors.conversation_container), 0)
        self.assertGreater(len(config.selectors.message_item), 0)
        self.assertEqual(config.default_limit_conversations, 10)
        self.assertEqual(config.default_limit_messages, 50)


class ExceptionHierarchyTests(unittest.TestCase):
    def test_all_dm_reader_errors_are_dm_reader_error(self) -> None:
        for exc_class in (
            InboxAccessError,
            ConversationContainerNotFoundError,
            MessageThreadNotFoundError,
            DMReaderLoginLostError,
        ):
            self.assertTrue(issubclass(exc_class, DMReaderError))


class DisallowedActionsTests(unittest.TestCase):
    def test_disallowed_actions_covers_dm_send(self) -> None:
        self.assertIn("send_dm", DISALLOWED_ACTIONS)
        self.assertIn("send_message", DISALLOWED_ACTIONS)
        self.assertIn("mark_read", DISALLOWED_ACTIONS)

    def test_reader_never_implements_a_send_method(self) -> None:
        for action in DISALLOWED_ACTIONS:
            self.assertFalse(
                hasattr(InstagramDMReader, action),
                f"InstagramDMReader must not implement '{action}'",
            )


class CliArgumentTests(unittest.TestCase):
    def test_defaults_are_none(self) -> None:
        arguments = parse_arguments([])
        self.assertIsNone(arguments.limit_conversations)
        self.assertIsNone(arguments.limit_messages)

    def test_parses_limits(self) -> None:
        arguments = parse_arguments(
            ["--limit-conversations", "10", "--limit-messages", "50"]
        )
        self.assertEqual(arguments.limit_conversations, 10)
        self.assertEqual(arguments.limit_messages, 50)


# ---------------------------------------------------------------------------
# Extraction tests (call reader methods directly against fake elements)
# ---------------------------------------------------------------------------


class ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = FakeSession(FakePage(routes={}))
        self.reader = _make_reader(self.session)
        self.selectors = _selectors()

    def test_extract_conversation_item(self) -> None:
        conv = FakeElement(
            children={
                self.selectors.conversation_username[0]: FakeElement(text="fan_one"),
                self.selectors.conversation_display_name[0]: FakeElement(text="Fan One"),
                self.selectors.conversation_link[0]: FakeElement(
                    attrs={"href": "/direct/t/conv-1/"}
                ),
                self.selectors.conversation_unread_indicator[0]: FakeElement(),
            }
        )

        raw = asyncio.run(self.reader._extract_conversation_item(conv))

        self.assertEqual(raw["username"], "fan_one")
        self.assertEqual(raw["href"], "/direct/t/conv-1/")
        self.assertTrue(raw["is_unread"])

    def test_extract_conversation_item_without_unread_indicator(self) -> None:
        conv = FakeElement(
            children={
                self.selectors.conversation_username[0]: FakeElement(text="fan_two"),
                self.selectors.conversation_link[0]: FakeElement(
                    attrs={"href": "/direct/t/conv-2/"}
                ),
            }
        )

        raw = asyncio.run(self.reader._extract_conversation_item(conv))
        self.assertFalse(raw["is_unread"])

    def test_extract_visible_conversations_container_missing(self) -> None:
        page = FakePage(routes={})
        result = asyncio.run(self.reader._extract_visible_conversations(page))
        self.assertIsNone(result)

    def test_extract_visible_conversations(self) -> None:
        conv = FakeElement(
            children={
                self.selectors.conversation_username[0]: FakeElement(text="fan_one"),
                self.selectors.conversation_link[0]: FakeElement(
                    attrs={"href": "/direct/t/conv-1/"}
                ),
            }
        )
        container = FakeElement(
            child_lists={self.selectors.conversation_item[0]: [conv]}
        )
        page = FakePage(
            routes={
                "https://www.instagram.com/direct/inbox/": {
                    "children": {self.selectors.conversation_container[0]: container},
                }
            }
        )

        result = asyncio.run(self.reader._extract_visible_conversations(page))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["username"], "fan_one")

    def test_extract_message_item_text(self) -> None:
        item = FakeElement(
            children={
                self.selectors.message_text[0]: FakeElement(text="hello there!"),
                self.selectors.message_timestamp[0]: FakeElement(text="2h"),
            },
            attrs={"data-message-id": "998"},
        )

        raw = asyncio.run(self.reader._extract_message_item(item))
        self.assertEqual(raw["message_text"], "hello there!")
        self.assertEqual(raw["message_id"], "998")
        self.assertFalse(raw["is_own_message"])
        self.assertFalse(raw["is_image"])

    def test_extract_message_item_own_message(self) -> None:
        item = FakeElement(
            children={
                self.selectors.message_text[0]: FakeElement(text="aww thank you 🤍"),
                self.selectors.message_own_indicator[0]: FakeElement(),
            }
        )

        raw = asyncio.run(self.reader._extract_message_item(item))
        self.assertTrue(raw["is_own_message"])

    def test_extract_message_item_own_message_by_right_alignment(self) -> None:
        """
        Confirmed live 2026-07-31: Instagram's real DM markup carries
        no own_indicator class/attribute at all — bounding-box
        position (right-of-center = Aiko's own, left-of-center = fan)
        is the actual signal used in practice.
        """
        item = FakeElement(
            children={
                self.selectors.message_text[0]: FakeElement(
                    text="follow my journey",
                    bounding_box={"x": 900, "y": 100, "width": 200, "height": 50},
                ),
            }
        )

        raw = asyncio.run(self.reader._extract_message_item(item))
        self.assertTrue(raw["is_own_message"])

    def test_extract_message_item_fan_message_left_aligned(self) -> None:
        item = FakeElement(
            children={
                self.selectors.message_text[0]: FakeElement(
                    text="hi aiko!",
                    bounding_box={"x": 50, "y": 100, "width": 200, "height": 50},
                ),
            }
        )

        raw = asyncio.run(self.reader._extract_message_item(item))
        self.assertFalse(raw["is_own_message"])

    def test_extract_message_item_no_bounding_box_defaults_to_fan(self) -> None:
        item = FakeElement(
            children={
                self.selectors.message_text[0]: FakeElement(text="hi aiko!"),
            }
        )

        raw = asyncio.run(self.reader._extract_message_item(item))
        self.assertFalse(raw["is_own_message"])

    def test_extract_message_item_image(self) -> None:
        item = FakeElement(
            children={self.selectors.message_image_indicator[0]: FakeElement()}
        )

        raw = asyncio.run(self.reader._extract_message_item(item))
        self.assertTrue(raw["is_image"])

    def test_extract_visible_messages_container_missing(self) -> None:
        page = FakePage(routes={})
        result = asyncio.run(self.reader._extract_visible_messages(page))
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Queue persistence tests
# ---------------------------------------------------------------------------


class QueuePersistenceTests(unittest.TestCase):
    def test_missing_queue_returns_empty_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = FakeSession(FakePage(routes={}))
            reader = _make_reader(session, queue_dir=Path(temp_dir))

            queue = reader._load_queue()
            self.assertEqual(queue["version"], "1.0")
            self.assertEqual(queue["messages"], [])

    def test_save_then_load_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = FakeSession(FakePage(routes={}))
            reader = _make_reader(session, queue_dir=Path(temp_dir))

            reader._save_queue(
                {"version": "1.0", "updated_at": "now", "messages": [{"stable_id": "x"}]}
            )

            queue = reader._load_queue()
            self.assertEqual(len(queue["messages"]), 1)

    def test_save_leaves_no_tmp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = FakeSession(FakePage(routes={}))
            reader = _make_reader(session, queue_dir=Path(temp_dir))

            reader._save_queue({"version": "1.0", "updated_at": "now", "messages": []})

            leftover = list(Path(temp_dir).glob("*.tmp"))
            self.assertEqual(leftover, [])


# ---------------------------------------------------------------------------
# Full read_messages() integration tests — mocked Playwright end to end
# ---------------------------------------------------------------------------


def _build_inbox_and_thread_routes(
    *,
    selectors,
    conversations: list[dict],
) -> dict[str, dict]:
    """
    conversations: list of dicts with keys username, href, is_unread,
    messages (list of raw message kwargs for FakeElement children).
    """
    conv_elements = []

    for conversation in conversations:
        conv_children = {
            selectors.conversation_username[0]: FakeElement(
                text=conversation["username"]
            ),
            selectors.conversation_link[0]: FakeElement(
                attrs={"href": conversation["href"]}
            ),
        }

        if conversation.get("is_unread"):
            conv_children[selectors.conversation_unread_indicator[0]] = FakeElement()

        conv_elements.append(FakeElement(children=conv_children))

    inbox_container = FakeElement(
        child_lists={selectors.conversation_item[0]: conv_elements}
    )

    routes = {
        "https://www.instagram.com/direct/inbox/": {
            "children": {
                "svg[aria-label='Home']": FakeElement(),
                selectors.conversation_container[0]: inbox_container,
            },
        }
    }

    for conversation in conversations:
        thread_url = "https://www.instagram.com" + conversation["href"]
        message_elements = []

        for message in conversation.get("messages", []):
            children = {}

            if message.get("text") is not None:
                children[selectors.message_text[0]] = FakeElement(text=message["text"])

            if message.get("is_own"):
                children[selectors.message_own_indicator[0]] = FakeElement()

            if message.get("is_image"):
                children[selectors.message_image_indicator[0]] = FakeElement()

            attrs = {}
            if message.get("message_id"):
                attrs["data-message-id"] = message["message_id"]

            message_elements.append(FakeElement(children=children, attrs=attrs))

        thread_container = FakeElement(
            child_lists={selectors.message_item[0]: message_elements}
        )

        routes[thread_url] = {
            "children": {selectors.message_container[0]: thread_container},
        }

    return routes


class ReadMessagesIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selectors = _selectors()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _reader_for(self, routes: dict[str, dict]) -> InstagramDMReader:
        page = FakePage(routes=routes)
        session = FakeSession(page)
        return _make_reader(session, queue_dir=Path(self.temp_dir.name))

    def test_new_fan_messages_are_queued(self) -> None:
        routes = _build_inbox_and_thread_routes(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "is_unread": True,
                    "messages": [
                        {"text": "you're so beautiful!", "message_id": "1"},
                    ],
                }
            ],
        )
        reader = self._reader_for(routes)

        result = asyncio.run(reader.read_messages())

        self.assertEqual(result.login_status, "logged_in")
        self.assertEqual(result.new_messages_queued, 1)
        self.assertEqual(result.excluded_own_count, 0)

        queue = json.loads(Path(reader._queue_path()).read_text(encoding="utf-8"))
        self.assertEqual(len(queue["messages"]), 1)
        self.assertEqual(queue["messages"][0]["message_text"], "you're so beautiful!")
        self.assertEqual(queue["messages"][0]["status"], "pending")

    def test_aikos_own_messages_are_excluded(self) -> None:
        routes = _build_inbox_and_thread_routes(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "messages": [
                        {"text": "hi aiko!", "message_id": "1"},
                        {"text": "thanks so much 🤍", "message_id": "2", "is_own": True},
                    ],
                }
            ],
        )
        reader = self._reader_for(routes)

        result = asyncio.run(reader.read_messages())

        self.assertEqual(result.new_messages_queued, 1)
        self.assertEqual(result.excluded_own_count, 1)

        queue = json.loads(Path(reader._queue_path()).read_text(encoding="utf-8"))
        self.assertEqual(len(queue["messages"]), 1)
        self.assertEqual(queue["messages"][0]["message_text"], "hi aiko!")

    def test_duplicate_prevention_across_runs(self) -> None:
        routes = _build_inbox_and_thread_routes(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "messages": [{"text": "you're so beautiful!", "message_id": "1"}],
                }
            ],
        )
        reader = self._reader_for(routes)

        first = asyncio.run(reader.read_messages())
        self.assertEqual(first.new_messages_queued, 1)

        second = asyncio.run(reader.read_messages())
        self.assertEqual(second.new_messages_queued, 0)
        self.assertEqual(second.duplicates_skipped, 1)

        queue = json.loads(Path(reader._queue_path()).read_text(encoding="utf-8"))
        self.assertEqual(len(queue["messages"]), 1)

    def test_unknown_media_attachment_is_queued_with_unknown_type(self) -> None:
        routes = _build_inbox_and_thread_routes(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "messages": [{"text": None, "message_id": "1"}],
                }
            ],
        )
        reader = self._reader_for(routes)

        result = asyncio.run(reader.read_messages())
        self.assertEqual(result.new_messages_queued, 1)

        queue = json.loads(Path(reader._queue_path()).read_text(encoding="utf-8"))
        self.assertEqual(queue["messages"][0]["message_type"], "unknown")

    def test_image_message_type_detected(self) -> None:
        routes = _build_inbox_and_thread_routes(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "messages": [
                        {"text": None, "message_id": "1", "is_image": True}
                    ],
                }
            ],
        )
        reader = self._reader_for(routes)

        result = asyncio.run(reader.read_messages())
        self.assertEqual(result.new_messages_queued, 1)

        queue = json.loads(Path(reader._queue_path()).read_text(encoding="utf-8"))
        self.assertEqual(queue["messages"][0]["message_type"], "image")

    def test_login_lost_raises_and_saves_diagnostic(self) -> None:
        routes = {
            "https://www.instagram.com/direct/inbox/": {
                "children": {"input[type='password']": FakeElement()},
            }
        }
        reader = self._reader_for(routes)

        with self.assertRaises(DMReaderLoginLostError):
            asyncio.run(reader.read_messages())

        log_path = reader._diagnostic_log_path()
        self.assertTrue(log_path.exists())

    def test_conversation_container_not_found_raises(self) -> None:
        routes = {
            "https://www.instagram.com/direct/inbox/": {
                "children": {"svg[aria-label='Home']": FakeElement()},
            }
        }
        reader = self._reader_for(routes)

        with self.assertRaises(ConversationContainerNotFoundError):
            asyncio.run(reader.read_messages())

    def test_inbox_open_failure_raises(self) -> None:
        session = FakeSession(FakePage(routes={}))
        reader = _make_reader(session, queue_dir=Path(self.temp_dir.name))

        with self.assertRaises(InboxAccessError):
            asyncio.run(reader.read_messages())

    def test_context_and_playwright_cleaned_up(self) -> None:
        routes = _build_inbox_and_thread_routes(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "messages": [{"text": "hi!", "message_id": "1"}],
                }
            ],
        )
        page = FakePage(routes=routes)
        session = FakeSession(page)
        reader = _make_reader(session, queue_dir=Path(self.temp_dir.name))

        asyncio.run(reader.read_messages())

        self.assertTrue(session._context.closed)
        self.assertTrue(session._playwright.stopped)

    def test_no_composer_interaction_method_is_ever_called(self) -> None:
        """
        FakeElement has no fill()/type()/press() capability at all —
        if read_messages() ever tried to compose or send anything,
        this would raise AttributeError rather than silently
        succeeding. click() IS legitimately used, but only to open a
        conversation for reading (_open_conversation_by_click) — never
        on a composer.
        """
        for forbidden in ("fill", "type", "press"):
            self.assertFalse(hasattr(FakeElement, forbidden))


def _build_click_based_routes_and_page(
    *,
    selectors,
    conversations: list[dict],
) -> FakePage:
    """
    Like _build_inbox_and_thread_routes, but conversation items carry
    NO href at all — matching the real Instagram DOM confirmed live
    2026-07-31 — forcing InstagramDMReader down the click-based
    _open_conversation_by_click() path. The returned FakePage is
    wired so that clicking a conversation row updates page.url/route
    exactly as a real click-driven navigation would.
    """
    inbox_url = "https://www.instagram.com/direct/inbox/"
    routes: dict[str, dict] = {}
    page = FakePage(routes=routes, start_url=inbox_url)

    conv_elements = []

    for conversation in conversations:
        thread_url = "https://www.instagram.com" + conversation["href"]

        conv_elements.append(
            FakeElement(
                children={
                    selectors.conversation_username[0]: FakeElement(
                        text=conversation["username"]
                    ),
                },
                page=page,
                click_navigates_to=thread_url,
                fail_click=conversation.get("fail_click", False),
            )
        )

        message_elements = []

        for message in conversation.get("messages", []):
            children = {}

            if message.get("text") is not None:
                children[selectors.message_text[0]] = FakeElement(text=message["text"])

            if message.get("is_own"):
                children[selectors.message_own_indicator[0]] = FakeElement()

            attrs = {}
            if message.get("message_id"):
                attrs["data-message-id"] = message["message_id"]

            message_elements.append(FakeElement(children=children, attrs=attrs))

        thread_container = FakeElement(
            child_lists={selectors.message_item[0]: message_elements}
        )

        routes[thread_url] = {
            "children": {selectors.message_container[0]: thread_container},
        }

    inbox_container = FakeElement(
        child_lists={selectors.conversation_item[0]: conv_elements}
    )

    routes[inbox_url] = {
        "children": {
            "svg[aria-label='Home']": FakeElement(),
            selectors.conversation_container[0]: inbox_container,
        },
    }

    return page


class ClickBasedNavigationTests(unittest.TestCase):
    """
    Covers the real navigation path confirmed live 2026-07-31:
    Instagram's conversation rows carry no href at all, so
    InstagramDMReader must click each row for real and capture the
    resulting page.url as thread_url.
    """

    def setUp(self) -> None:
        self.selectors = _selectors()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _reader_for(self, page: FakePage) -> InstagramDMReader:
        session = FakeSession(page)
        return _make_reader(session, queue_dir=Path(self.temp_dir.name))

    def test_messages_collected_via_click_navigation(self) -> None:
        page = _build_click_based_routes_and_page(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "messages": [{"text": "you're so beautiful!", "message_id": "1"}],
                }
            ],
        )
        reader = self._reader_for(page)

        result = asyncio.run(reader.read_messages())

        self.assertEqual(result.new_messages_queued, 1)

        queue = json.loads(Path(reader._queue_path()).read_text(encoding="utf-8"))
        self.assertEqual(queue["messages"][0]["message_text"], "you're so beautiful!")
        self.assertEqual(queue["messages"][0]["conversation_id"], "conv-1")

    def test_row_is_actually_clicked_not_just_read(self) -> None:
        page = _build_click_based_routes_and_page(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "messages": [{"text": "hi there", "message_id": "1"}],
                }
            ],
        )
        reader = self._reader_for(page)

        asyncio.run(reader.read_messages())

        container = page._routes["https://www.instagram.com/direct/inbox/"]["children"][
            self.selectors.conversation_container[0]
        ]
        row = container._child_lists[self.selectors.conversation_item[0]][0]
        self.assertTrue(row.clicked)
        self.assertEqual(row.click_count, 1)

    def test_multiple_conversations_each_re_located_by_username(self) -> None:
        page = _build_click_based_routes_and_page(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "messages": [{"text": "hi from fan one", "message_id": "1"}],
                },
                {
                    "username": "fan_two",
                    "href": "/direct/t/conv-2/",
                    "messages": [{"text": "hi from fan two", "message_id": "2"}],
                },
            ],
        )
        reader = self._reader_for(page)

        result = asyncio.run(reader.read_messages())

        self.assertEqual(result.conversations_visited, 2)
        self.assertEqual(result.new_messages_queued, 2)

        queue = json.loads(Path(reader._queue_path()).read_text(encoding="utf-8"))
        texts = {message["message_text"] for message in queue["messages"]}
        self.assertEqual(texts, {"hi from fan one", "hi from fan two"})

    def test_click_failure_is_skipped_gracefully(self) -> None:
        page = _build_click_based_routes_and_page(
            selectors=self.selectors,
            conversations=[
                {
                    "username": "fan_one",
                    "href": "/direct/t/conv-1/",
                    "fail_click": True,
                    "messages": [{"text": "hi there", "message_id": "1"}],
                }
            ],
        )
        reader = self._reader_for(page)

        result = asyncio.run(reader.read_messages())

        self.assertEqual(result.new_messages_queued, 0)

        log_path = reader._diagnostic_log_path()
        self.assertTrue(log_path.exists())


if __name__ == "__main__":
    unittest.main()
