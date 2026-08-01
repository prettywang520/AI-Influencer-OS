from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from .instagram_models import (
    InstagramConfigError,
    InstagramSessionError,
    LoginTimeoutError,
    ProfileDirectoryError,
    SessionDetectionError,
    SessionState,
)
from .instagram_session import (
    DISALLOWED_ACTIONS,
    InstagramSession,
    load_config,
    parse_arguments,
)

# None of these tests open a real browser or contact Instagram.
# InstagramSession.classify() is pure, and dry_run=True short-circuits
# check()/login() before Playwright is ever touched.


class LoadConfigTests(unittest.TestCase):
    def test_loads_real_config_file(self) -> None:
        config = load_config()

        self.assertEqual(config.base_url, "https://www.instagram.com/")
        self.assertTrue(config.login_url.endswith("/accounts/login/"))
        self.assertTrue(config.profile_dir.as_posix().endswith(
            "browser_profile/instagram"
        ))
        self.assertFalse(config.headless)
        self.assertFalse(config.dry_run)
        self.assertGreater(len(config.login_detection.logged_in_selectors), 0)
        self.assertGreater(len(config.login_detection.logged_out_url_markers), 0)
        self.assertGreater(len(config.login_detection.pending_url_markers), 0)
        self.assertIn("codeentry", config.login_detection.pending_url_markers)
        self.assertIn("post", config.disallowed_actions)
        self.assertIn("send_dm", config.disallowed_actions)

    def test_headless_override(self) -> None:
        config = load_config(headless_override=True)
        self.assertTrue(config.headless)

    def test_dry_run_override(self) -> None:
        config = load_config(dry_run_override=True)
        self.assertTrue(config.dry_run)

    def test_missing_config_file_raises(self) -> None:
        missing_path = Path(__file__).resolve().parent / "does_not_exist.yaml"

        with self.assertRaises(InstagramConfigError):
            load_config(config_path=missing_path)

    def test_empty_config_file_raises(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            empty_config = Path(temp_dir) / "instagram.yaml"
            empty_config.write_text("", encoding="utf-8")

            with self.assertRaises(InstagramConfigError):
                load_config(config_path=empty_config)

    def test_invalid_yaml_raises(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            bad_config = Path(temp_dir) / "instagram.yaml"
            bad_config.write_text("key: [unclosed", encoding="utf-8")

            with self.assertRaises(InstagramConfigError):
                load_config(config_path=bad_config)


class ClassifyTests(unittest.TestCase):
    def test_logged_out_url_marker_wins_even_with_logged_in_selector(self) -> None:
        status, reason = InstagramSession.classify(
            current_url="https://www.instagram.com/accounts/login/",
            found_logged_in_selectors=["svg[aria-label='Home']"],
            found_logged_out_selectors=[],
            logged_out_url_markers=["accounts/login"],
            pending_url_markers=[],
        )

        self.assertEqual(status, "logged_out")
        self.assertEqual(reason, "url_matched_logged_out_marker")

    def test_logged_in_selector_match(self) -> None:
        status, reason = InstagramSession.classify(
            current_url="https://www.instagram.com/",
            found_logged_in_selectors=["svg[aria-label='Home']"],
            found_logged_out_selectors=[],
            logged_out_url_markers=[],
            pending_url_markers=[],
        )

        self.assertEqual(status, "logged_in")
        self.assertEqual(reason, "matched_logged_in_selector")

    def test_logged_out_selector_match(self) -> None:
        status, reason = InstagramSession.classify(
            current_url="https://www.instagram.com/",
            found_logged_in_selectors=[],
            found_logged_out_selectors=["input[name='username']"],
            logged_out_url_markers=[],
            pending_url_markers=[],
        )

        self.assertEqual(status, "logged_out")
        self.assertEqual(reason, "matched_logged_out_selector")

    def test_pending_auth_marker_returns_unknown_not_logged_in(self) -> None:
        """
        Regression test: a real login run against Instagram landed on
        .../auth_platform/codeentry/?apc=... (a 2FA verification-code
        screen) and was previously misclassified as logged_in because
        any instagram.com/* URL was treated as a weak logged_in signal.
        This must now resolve to unknown, never logged_in.
        """
        status, reason = InstagramSession.classify(
            current_url=(
                "https://www.instagram.com/auth_platform/codeentry/"
                "?apc=AbcXyz123"
            ),
            found_logged_in_selectors=[],
            found_logged_out_selectors=[],
            logged_out_url_markers=["accounts/login", "accounts/onetap"],
            pending_url_markers=["auth_platform", "codeentry", "checkpoint"],
        )

        self.assertEqual(status, "unknown")
        self.assertEqual(reason, "url_matched_pending_auth_marker")
        self.assertNotEqual(status, "logged_in")

    def test_pending_auth_marker_wins_over_generic_url(self) -> None:
        status, reason = InstagramSession.classify(
            current_url="https://www.instagram.com/challenge/",
            found_logged_in_selectors=[],
            found_logged_out_selectors=[],
            logged_out_url_markers=[],
            pending_url_markers=["challenge"],
        )

        self.assertEqual(status, "unknown")
        self.assertEqual(reason, "url_matched_pending_auth_marker")

    def test_no_generic_url_can_ever_produce_logged_in(self) -> None:
        """
        There is deliberately no code path where a bare URL match can
        return logged_in — only an actual DOM selector match can.
        """
        status, reason = InstagramSession.classify(
            current_url="https://www.instagram.com/",
            found_logged_in_selectors=[],
            found_logged_out_selectors=[],
            logged_out_url_markers=[],
            pending_url_markers=[],
        )

        self.assertEqual(status, "unknown")
        self.assertEqual(reason, "no_selector_or_url_marker_matched")

    def test_unknown_when_nothing_matches(self) -> None:
        status, reason = InstagramSession.classify(
            current_url="https://www.instagram.com/some/other/page/",
            found_logged_in_selectors=[],
            found_logged_out_selectors=[],
            logged_out_url_markers=[],
            pending_url_markers=[],
        )

        self.assertEqual(status, "unknown")
        self.assertEqual(reason, "no_selector_or_url_marker_matched")


class DryRunTests(unittest.TestCase):
    def _config(self, **overrides):
        config = load_config(dry_run_override=True)

        for key, value in overrides.items():
            object.__setattr__(config, key, value)

        return config

    def test_check_dry_run_never_touches_playwright(self) -> None:
        session = InstagramSession(self._config())

        state = asyncio.run(session.check())

        self.assertIsInstance(state, SessionState)
        self.assertEqual(state.status, "logged_out")
        self.assertTrue(state.dry_run)
        self.assertEqual(state.reason, "dry_run_mode_no_browser_launched")
        self.assertIsNone(state.screenshot_path)

    def test_login_dry_run_never_touches_playwright(self) -> None:
        session = InstagramSession(self._config())

        state = asyncio.run(session.login())

        self.assertIsInstance(state, SessionState)
        self.assertTrue(state.dry_run)
        self.assertEqual(state.reason, "dry_run_mode_no_browser_launched")


class SessionStateTests(unittest.TestCase):
    def test_logged_in_property(self) -> None:
        state = SessionState(status="logged_in", checked_at="2026-07-31T00:00:00Z")
        self.assertTrue(state.logged_in)

        state = SessionState(status="logged_out", checked_at="2026-07-31T00:00:00Z")
        self.assertFalse(state.logged_in)

    def test_to_dict_round_trips_fields(self) -> None:
        state = SessionState(
            status="unknown",
            checked_at="2026-07-31T00:00:00Z",
            current_url="https://www.instagram.com/",
            reason="no_selector_or_url_marker_matched",
            screenshot_path="/tmp/shot.png",
            dry_run=False,
        )

        data = state.to_dict()

        self.assertEqual(data["status"], "unknown")
        self.assertEqual(data["screenshot_path"], "/tmp/shot.png")


class ExceptionHierarchyTests(unittest.TestCase):
    def test_all_exceptions_inherit_from_base(self) -> None:
        for exception_type in (
            InstagramConfigError,
            ProfileDirectoryError,
            LoginTimeoutError,
            SessionDetectionError,
        ):
            self.assertTrue(issubclass(exception_type, InstagramSessionError))

    def test_base_exception_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(InstagramSessionError, RuntimeError))


class DisallowedActionsTests(unittest.TestCase):
    def test_no_mutating_action_methods_exist_on_session(self) -> None:
        for action in DISALLOWED_ACTIONS:
            self.assertFalse(
                hasattr(InstagramSession, action),
                f"InstagramSession must not implement '{action}'",
            )

        for forbidden_name in (
            "post",
            "reply",
            "like",
            "follow",
            "unfollow",
            "send_dm",
            "publish_comment_reply",
            "hide_comment",
        ):
            self.assertFalse(
                hasattr(InstagramSession, forbidden_name),
                f"InstagramSession must not implement '{forbidden_name}'",
            )


class CliArgumentTests(unittest.TestCase):
    def test_check_flag_parses(self) -> None:
        arguments = parse_arguments(["--check"])
        self.assertTrue(arguments.check)
        self.assertFalse(arguments.login)
        self.assertFalse(arguments.dry_run)

    def test_login_flag_parses(self) -> None:
        arguments = parse_arguments(["--login"])
        self.assertTrue(arguments.login)
        self.assertFalse(arguments.check)

    def test_login_and_check_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--login", "--check"])

    def test_requires_one_action(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_dry_run_and_headless_flags_parse(self) -> None:
        arguments = parse_arguments(["--check", "--dry-run", "--headless"])
        self.assertTrue(arguments.dry_run)
        self.assertTrue(arguments.headless)


if __name__ == "__main__":
    unittest.main()
