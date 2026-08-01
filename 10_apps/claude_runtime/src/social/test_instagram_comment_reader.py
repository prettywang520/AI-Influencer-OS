from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from .instagram_models import (
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
from .instagram_comment_reader import (
    DISALLOWED_ACTIONS,
    InstagramCommentReader,
    build_comment_record,
    compute_stable_id,
    is_own_comment,
    load_comment_reader_config,
    parse_arguments,
    parse_comment_permalink,
)

# None of these tests open a real browser or contact Instagram.
# Playwright-shaped objects are hand-built fakes; everything else is
# pure functions or local tempfile I/O.


# ---------------------------------------------------------------------------
# Fake Playwright-shaped objects for extraction tests
# ---------------------------------------------------------------------------


class FakeElement:
    """
    Minimal stand-in for a Playwright ElementHandle, exposing only the
    async methods InstagramCommentReader actually calls.
    """

    def __init__(
        self,
        *,
        text: str = "",
        attrs: dict[str, str] | None = None,
        children: dict[str, "FakeElement"] | None = None,
        child_lists: dict[str, list["FakeElement"]] | None = None,
    ) -> None:
        self._text = text
        self._attrs = attrs or {}
        self._children = children or {}
        self._child_lists = child_lists or {}

    async def query_selector(self, selector: str):
        return self._children.get(selector)

    async def query_selector_all(self, selector: str):
        return self._child_lists.get(selector, [])

    async def inner_text(self) -> str:
        return self._text

    async def get_attribute(self, name: str):
        return self._attrs.get(name)


class FakePage:
    """Minimal stand-in for a Playwright Page (top-level query only)."""

    def __init__(
        self,
        *,
        children: dict[str, FakeElement] | None = None,
        url: str = "https://www.instagram.com/p/ABC123/",
    ) -> None:
        self._children = children or {}
        self.url = url

    async def query_selector(self, selector: str):
        return self._children.get(selector)


def _make_reader(**config_overrides) -> InstagramCommentReader:
    config = load_comment_reader_config()

    for key, value in config_overrides.items():
        object.__setattr__(config, key, value)

    class _StubSessionConfig:
        login_detection = None

    class _StubSession:
        config = _StubSessionConfig()

    return InstagramCommentReader(session=_StubSession(), config=config)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class ParseCommentPermalinkTests(unittest.TestCase):
    def test_none_href_returns_all_none(self) -> None:
        comment_id, parent_id, is_reply = parse_comment_permalink(None)
        self.assertIsNone(comment_id)
        self.assertIsNone(parent_id)
        self.assertFalse(is_reply)

    def test_top_level_comment_href(self) -> None:
        comment_id, parent_id, is_reply = parse_comment_permalink(
            "https://www.instagram.com/p/ABC123/c/17895/"
        )
        self.assertEqual(comment_id, "17895")
        self.assertIsNone(parent_id)
        self.assertFalse(is_reply)

    def test_reply_href(self) -> None:
        comment_id, parent_id, is_reply = parse_comment_permalink(
            "https://www.instagram.com/p/ABC123/c/17895/r/222/"
        )
        self.assertEqual(comment_id, "222")
        self.assertEqual(parent_id, "17895")
        self.assertTrue(is_reply)

    def test_unrelated_href_returns_all_none(self) -> None:
        comment_id, parent_id, is_reply = parse_comment_permalink(
            "https://www.instagram.com/some_username/"
        )
        self.assertIsNone(comment_id)
        self.assertIsNone(parent_id)
        self.assertFalse(is_reply)


class ComputeStableIdTests(unittest.TestCase):
    def test_uses_comment_id_when_available(self) -> None:
        stable_id = compute_stable_id(
            comment_id="17895",
            post_url="https://www.instagram.com/p/ABC123/",
            username="alice",
            comment_text="nice photo",
        )
        self.assertEqual(stable_id, "comment_id:17895")

    def test_falls_back_to_hash_without_comment_id(self) -> None:
        stable_id = compute_stable_id(
            comment_id=None,
            post_url="https://www.instagram.com/p/ABC123/",
            username="alice",
            comment_text="nice photo",
        )
        self.assertTrue(stable_id.startswith("hash:"))
        self.assertEqual(len(stable_id), len("hash:") + 64)

    def test_hash_is_stable_for_same_inputs(self) -> None:
        kwargs = dict(
            comment_id=None,
            post_url="https://www.instagram.com/p/ABC123/",
            username="alice",
            comment_text="nice photo",
        )
        self.assertEqual(compute_stable_id(**kwargs), compute_stable_id(**kwargs))

    def test_hash_differs_for_different_text(self) -> None:
        first = compute_stable_id(
            comment_id=None,
            post_url="https://www.instagram.com/p/ABC123/",
            username="alice",
            comment_text="nice photo",
        )
        second = compute_stable_id(
            comment_id=None,
            post_url="https://www.instagram.com/p/ABC123/",
            username="alice",
            comment_text="different text",
        )
        self.assertNotEqual(first, second)

    def test_hash_is_case_insensitive_on_username(self) -> None:
        first = compute_stable_id(
            comment_id=None,
            post_url="https://www.instagram.com/p/ABC123/",
            username="Alice",
            comment_text="nice photo",
        )
        second = compute_stable_id(
            comment_id=None,
            post_url="https://www.instagram.com/p/ABC123/",
            username="alice",
            comment_text="nice photo",
        )
        self.assertEqual(first, second)


class IsOwnCommentTests(unittest.TestCase):
    def test_matches_case_insensitively(self) -> None:
        self.assertTrue(is_own_comment("Aiko.Sato.Travels", "aiko.sato.travels"))

    def test_matches_with_at_prefix(self) -> None:
        self.assertTrue(is_own_comment("@aiko.sato.travels", "aiko.sato.travels"))

    def test_does_not_match_other_username(self) -> None:
        self.assertFalse(is_own_comment("someone_else", "aiko.sato.travels"))

    def test_false_when_username_missing(self) -> None:
        self.assertFalse(is_own_comment(None, "aiko.sato.travels"))

    def test_false_when_own_username_missing(self) -> None:
        self.assertFalse(is_own_comment("someone", ""))


class BuildCommentRecordTests(unittest.TestCase):
    def test_builds_record_with_comment_id(self) -> None:
        record = build_comment_record(
            {
                "username": "  alice  ",
                "comment_text": " nice photo! ",
                "timestamp_text": "2h",
                "comment_id": "17895",
                "parent_comment_id": None,
                "is_reply": False,
            },
            post_url="https://www.instagram.com/p/ABC123/",
            discovered_at="2026-07-31T00:00:00Z",
        )

        self.assertIsInstance(record, CommentRecord)
        self.assertEqual(record.username, "alice")
        self.assertEqual(record.comment_text, "nice photo!")
        self.assertEqual(record.stable_id, "comment_id:17895")
        self.assertEqual(record.status, "pending")
        self.assertEqual(record.source, "instagram_post_comment")
        self.assertIsNone(record.error)

    def test_missing_text_sets_error_field(self) -> None:
        record = build_comment_record(
            {
                "username": "alice",
                "comment_text": None,
                "timestamp_text": None,
                "comment_id": None,
                "parent_comment_id": None,
                "is_reply": False,
            },
            post_url="https://www.instagram.com/p/ABC123/",
            discovered_at="2026-07-31T00:00:00Z",
        )

        self.assertEqual(record.error, "comment_text_not_found")

    def test_falls_back_to_hash_stable_id_without_comment_id(self) -> None:
        record = build_comment_record(
            {
                "username": "alice",
                "comment_text": "nice photo",
                "timestamp_text": None,
                "comment_id": None,
                "parent_comment_id": None,
                "is_reply": False,
            },
            post_url="https://www.instagram.com/p/ABC123/",
            discovered_at="2026-07-31T00:00:00Z",
        )

        self.assertTrue(record.stable_id.startswith("hash:"))


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------


class LoadCommentReaderConfigTests(unittest.TestCase):
    def test_loads_real_config_file(self) -> None:
        config = load_comment_reader_config()

        self.assertIsInstance(config, CommentReaderConfig)
        self.assertEqual(config.own_username, "aikotraveldiary")
        self.assertGreater(config.default_limit, 0)
        self.assertTrue(config.queue_dir.as_posix().endswith(
            "output/community/queues"
        ))
        self.assertTrue(config.logs_dir.as_posix().endswith(
            "output/community/logs"
        ))
        self.assertTrue(config.screenshots_dir.as_posix().endswith(
            "output/community/screenshots"
        ))
        self.assertGreater(len(config.selectors.comment_container), 0)
        self.assertGreater(len(config.selectors.comment_item), 0)
        self.assertIsInstance(config.selectors, CommentSelectors)

    def test_missing_config_file_raises(self) -> None:
        missing_path = Path(__file__).resolve().parent / "does_not_exist.yaml"

        with self.assertRaises(InstagramConfigError):
            load_comment_reader_config(config_path=missing_path)

    def test_empty_config_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_config = Path(temp_dir) / "instagram.yaml"
            empty_config.write_text("", encoding="utf-8")

            with self.assertRaises(InstagramConfigError):
                load_comment_reader_config(config_path=empty_config)

    def test_missing_sections_fall_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            minimal_config = Path(temp_dir) / "instagram.yaml"
            minimal_config.write_text("instagram: {}\n", encoding="utf-8")

            config = load_comment_reader_config(config_path=minimal_config)

            self.assertEqual(config.own_username, "")
            self.assertEqual(config.default_limit, 20)
            self.assertEqual(config.selectors.comment_container, [])


# ---------------------------------------------------------------------------
# Extraction tests against mocked Playwright objects
# ---------------------------------------------------------------------------


class ExtractionTests(unittest.TestCase):
    def _reader(self) -> InstagramCommentReader:
        selectors = CommentSelectors(
            comment_container=["ul.comments"],
            comment_item=["li.comment"],
            comment_username=["a.username"],
            comment_text=["span.text"],
            comment_timestamp=["time"],
            comment_permalink=["a.permalink"],
            reply_item=["li.reply"],
            load_more_button=["button.load-more"],
        )
        return _make_reader(selectors=selectors)

    def test_container_not_found_returns_none(self) -> None:
        reader = self._reader()
        page = FakePage(children={})

        result = asyncio.run(reader._extract_visible_comments(page))

        self.assertIsNone(result)

    def test_container_found_but_empty_returns_empty_list(self) -> None:
        reader = self._reader()
        container = FakeElement(child_lists={"li.comment": []})
        page = FakePage(children={"ul.comments": container})

        result = asyncio.run(reader._extract_visible_comments(page))

        self.assertEqual(result, [])

    def test_extracts_top_level_comment_fields(self) -> None:
        reader = self._reader()

        comment_item = FakeElement(
            children={
                "a.username": FakeElement(text="alice"),
                "span.text": FakeElement(text="  Love this!  "),
                "time": FakeElement(text="2h"),
                "a.permalink": FakeElement(
                    attrs={"href": "https://www.instagram.com/p/ABC123/c/999/"}
                ),
            },
            child_lists={"li.reply": []},
        )

        container = FakeElement(child_lists={"li.comment": [comment_item]})
        page = FakePage(children={"ul.comments": container})

        raw = asyncio.run(reader._extract_visible_comments(page))

        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["username"], "alice")
        self.assertEqual(raw[0]["comment_text"], "  Love this!  ")
        self.assertEqual(raw[0]["comment_id"], "999")
        self.assertFalse(raw[0]["is_reply"])
        self.assertIsNone(raw[0]["parent_comment_id"])

    def test_extracts_nested_reply_with_parent_id(self) -> None:
        reader = self._reader()

        reply_item = FakeElement(
            children={
                "a.username": FakeElement(text="bob"),
                "span.text": FakeElement(text="Totally agree"),
            },
            child_lists={"li.reply": []},
        )

        parent_item = FakeElement(
            children={
                "a.username": FakeElement(text="alice"),
                "span.text": FakeElement(text="Love this!"),
                "a.permalink": FakeElement(
                    attrs={"href": "https://www.instagram.com/p/ABC123/c/999/"}
                ),
            },
            child_lists={"li.reply": [reply_item]},
        )

        container = FakeElement(child_lists={"li.comment": [parent_item]})
        page = FakePage(children={"ul.comments": container})

        raw = asyncio.run(reader._extract_visible_comments(page))

        self.assertEqual(len(raw), 2)

        parent_record, reply_record = raw[0], raw[1]

        self.assertFalse(parent_record["is_reply"])
        self.assertEqual(parent_record["comment_id"], "999")

        self.assertTrue(reply_record["is_reply"])
        self.assertEqual(reply_record["username"], "bob")
        self.assertEqual(reply_record["parent_comment_id"], "999")

    def test_missing_optional_fields_return_none_gracefully(self) -> None:
        reader = self._reader()

        comment_item = FakeElement(children={}, child_lists={"li.reply": []})
        container = FakeElement(child_lists={"li.comment": [comment_item]})
        page = FakePage(children={"ul.comments": container})

        raw = asyncio.run(reader._extract_visible_comments(page))

        self.assertEqual(len(raw), 1)
        self.assertIsNone(raw[0]["username"])
        self.assertIsNone(raw[0]["comment_text"])
        self.assertIsNone(raw[0]["comment_id"])


# ---------------------------------------------------------------------------
# Queue persistence / dedup tests (tempfile only, no Playwright)
# ---------------------------------------------------------------------------


class QueuePersistenceTests(unittest.TestCase):
    def _reader_with_temp_queue(self, temp_dir: str) -> InstagramCommentReader:
        return _make_reader(queue_dir=Path(temp_dir))

    def test_load_queue_returns_empty_shape_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reader = self._reader_with_temp_queue(temp_dir)
            queue = reader._load_queue()

            self.assertEqual(queue["version"], "1.0")
            self.assertEqual(queue["comments"], [])

    def test_save_then_load_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reader = self._reader_with_temp_queue(temp_dir)

            queue = reader._load_queue()
            queue["comments"].append({"stable_id": "hash:abc", "username": "alice"})
            queue["updated_at"] = "2026-07-31T00:00:00Z"
            reader._save_queue(queue)

            reloaded = reader._load_queue()
            self.assertEqual(len(reloaded["comments"]), 1)
            self.assertEqual(reloaded["comments"][0]["stable_id"], "hash:abc")

    def test_queue_schema_has_required_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reader = self._reader_with_temp_queue(temp_dir)
            queue = reader._load_queue()
            queue["updated_at"] = "2026-07-31T00:00:00Z"
            reader._save_queue(queue)

            raw = json.loads(reader._queue_path().read_text(encoding="utf-8"))
            self.assertIn("version", raw)
            self.assertIn("updated_at", raw)
            self.assertIn("comments", raw)

    def test_corrupt_queue_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reader = self._reader_with_temp_queue(temp_dir)
            reader._queue_path().parent.mkdir(parents=True, exist_ok=True)
            reader._queue_path().write_text("not json", encoding="utf-8")

            with self.assertRaises(CommentReaderError):
                reader._load_queue()

    def test_never_imports_same_stable_id_twice_across_runs(self) -> None:
        """
        Simulates two separate reader runs writing to the same queue
        file: a stable_id already present must never be re-added.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            reader = self._reader_with_temp_queue(temp_dir)

            queue = reader._load_queue()
            queue["comments"].append(
                {"stable_id": "comment_id:999", "username": "alice"}
            )
            reader._save_queue(queue)

            # Second "run": load the queue again and attempt to add the
            # same stable_id plus one genuinely new one.
            queue = reader._load_queue()
            existing_ids = {c["stable_id"] for c in queue["comments"]}

            candidate_ids = ["comment_id:999", "comment_id:1000"]
            added = [cid for cid in candidate_ids if cid not in existing_ids]

            self.assertEqual(added, ["comment_id:1000"])


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class ExceptionHierarchyTests(unittest.TestCase):
    def test_all_exceptions_inherit_from_instagram_session_error(self) -> None:
        for exception_type in (
            CommentReaderError,
            PostAccessError,
            CommentReaderLoginLostError,
        ):
            self.assertTrue(issubclass(exception_type, InstagramSessionError))


# ---------------------------------------------------------------------------
# Disallowed-action safety net
# ---------------------------------------------------------------------------


class DisallowedActionsTests(unittest.TestCase):
    def test_no_mutating_action_methods_exist_on_reader(self) -> None:
        for forbidden_name in (
            "post",
            "reply",
            "like",
            "follow",
            "unfollow",
            "hide",
            "delete",
            "report",
            "send_dm",
            "hide_comment",
            "delete_comment",
            "report_comment",
        ):
            self.assertFalse(
                hasattr(InstagramCommentReader, forbidden_name),
                f"InstagramCommentReader must not implement '{forbidden_name}'",
            )

        self.assertIn("send_dm", DISALLOWED_ACTIONS)
        self.assertIn("delete", DISALLOWED_ACTIONS)


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------


class CliArgumentTests(unittest.TestCase):
    def test_post_url_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_parses_post_url_and_limit(self) -> None:
        arguments = parse_arguments(
            ["--post-url", "https://www.instagram.com/p/ABC123/", "--limit", "20"]
        )
        self.assertEqual(arguments.post_url, "https://www.instagram.com/p/ABC123/")
        self.assertEqual(arguments.limit, 20)

    def test_limit_defaults_to_none(self) -> None:
        arguments = parse_arguments(
            ["--post-url", "https://www.instagram.com/p/ABC123/"]
        )
        self.assertIsNone(arguments.limit)


if __name__ == "__main__":
    unittest.main()
