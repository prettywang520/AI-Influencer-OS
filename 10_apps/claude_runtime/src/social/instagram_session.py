from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .instagram_models import (
    InstagramConfigError,
    InstagramSessionConfig,
    InstagramSessionError,
    LoginDetectionConfig,
    LoginTimeoutError,
    ProfileDirectoryError,
    SessionDetectionError,
    SessionState,
    SessionStatus,
    ViewportConfig,
)

try:
    from playwright.async_api import (
        Page,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )
except ImportError:  # pragma: no cover - exercised only when playwright is absent
    Page = Any  # type: ignore[assignment,misc]
    PlaywrightTimeoutError = TimeoutError  # type: ignore[assignment,misc]
    async_playwright = None


# Phase 8A is session infrastructure only.
#
# This module intentionally does not implement, and must never grow,
# methods for: posting, replying, liking, following, unfollowing, or
# sending direct messages. It only opens a browser, waits for a human
# to log in, and reports logged_in / logged_out state.
DISALLOWED_ACTIONS = (
    "post",
    "reply",
    "like",
    "follow",
    "unfollow",
    "send_dm",
)

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "social" / "instagram.yaml"


def _runtime_root() -> Path:
    """
    instagram_session.py location:

    10_apps/claude_runtime/src/social/instagram_session.py

    parents[2] therefore resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def load_config(
    *,
    config_path: str | Path | None = None,
    headless_override: bool | None = None,
    dry_run_override: bool | None = None,
) -> InstagramSessionConfig:
    """
    Load and fully resolve config/social/instagram.yaml.

    No username, password, or any other credential is read here or
    anywhere else in this module. Initial login is always manual.
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
    browser_section = raw.get("browser") or {}
    profile_section = raw.get("profile") or {}
    timeouts_section = raw.get("timeouts") or {}
    detection_section = raw.get("login_detection") or {}
    screenshot_section = raw.get("screenshots") or {}
    safety_section = raw.get("safety") or {}

    viewport_section = browser_section.get("viewport") or {}

    profile_directory = str(
        profile_section.get("directory", "browser_profile/instagram")
    )

    screenshot_directory = str(
        screenshot_section.get(
            "directory",
            "output/social/instagram/screenshots",
        )
    )

    headless = bool(browser_section.get("headless", False))
    if headless_override is not None:
        headless = headless_override

    dry_run = bool(safety_section.get("dry_run_default", False))
    if dry_run_override is not None:
        dry_run = dry_run_override

    return InstagramSessionConfig(
        base_url=str(
            instagram_section.get("base_url", "https://www.instagram.com/")
        ),
        login_url=str(
            instagram_section.get(
                "login_url",
                "https://www.instagram.com/accounts/login/",
            )
        ),
        profile_dir=(runtime_root / profile_directory).resolve(),
        headless=headless,
        dry_run=dry_run,
        locale=str(browser_section.get("locale", "en-US")),
        timezone_id=str(browser_section.get("timezone_id", "UTC")),
        viewport=ViewportConfig(
            width=int(viewport_section.get("width", 1280)),
            height=int(viewport_section.get("height", 900)),
        ),
        navigation_timeout_ms=int(timeouts_section.get("navigation_ms", 30000)),
        login_wait_timeout_ms=int(timeouts_section.get("login_wait_ms", 300000)),
        check_timeout_ms=int(timeouts_section.get("check_ms", 15000)),
        login_detection=LoginDetectionConfig(
            logged_in_selectors=list(
                detection_section.get("logged_in_selectors", [])
            ),
            logged_out_selectors=list(
                detection_section.get("logged_out_selectors", [])
            ),
            logged_out_url_markers=list(
                detection_section.get("logged_out_url_markers", [])
            ),
            pending_url_markers=list(
                detection_section.get("pending_url_markers", [])
            ),
        ),
        screenshot_dir=(runtime_root / screenshot_directory).resolve(),
        screenshot_on_detection_failure=bool(
            screenshot_section.get("on_detection_failure", True)
        ),
        disallowed_actions=list(
            safety_section.get("disallowed_actions", list(DISALLOWED_ACTIONS))
        ),
    )


class InstagramSession:
    """
    Read-only Instagram browser session infrastructure.

    Responsibilities:
    - open a persistent, visible-by-default Chromium profile
    - let a human complete login manually (no credentials are handled)
    - detect logged_in / logged_out / unknown state
    - save a screenshot when detection is ambiguous

    This class does not post, reply, like, follow, unfollow, or send
    direct messages, and never will as part of Phase 8A.
    """

    def __init__(self, config: InstagramSessionConfig) -> None:
        self.config = config

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _prepare_profile_dir(self) -> None:
        try:
            self.config.profile_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProfileDirectoryError(
                "Unable to create persistent browser profile directory: "
                f"{self.config.profile_dir}"
            ) from exc

    @staticmethod
    def classify(
        *,
        current_url: str,
        found_logged_in_selectors: list[str],
        found_logged_out_selectors: list[str],
        logged_out_url_markers: list[str],
        pending_url_markers: list[str],
    ) -> tuple[SessionStatus, str]:
        """
        Pure session-state classification.

        Takes plain strings/lists rather than a live browser page, so
        it can be unit tested without Playwright or real Instagram
        access. Priority order:

        1. A logged_out URL marker (e.g. accounts/login) is decisive.
        2. A pending-auth URL marker (2FA / checkpoint / consent) means
           login is in progress but not finished: unknown, not logged_in.
        3. A matched logged_in selector.
        4. A matched logged_out selector.
        5. Otherwise: unknown.

        There is deliberately no "any instagram.com/* URL means
        logged_in" fallback: that previously produced a false positive
        on the 2FA verification-code screen. logged_in must always be
        confirmed by an actual DOM selector.
        """
        lowered_url = current_url.lower()

        if any(
            marker.lower() in lowered_url
            for marker in logged_out_url_markers
        ):
            return "logged_out", "url_matched_logged_out_marker"

        if any(
            marker.lower() in lowered_url
            for marker in pending_url_markers
        ):
            return "unknown", "url_matched_pending_auth_marker"

        if found_logged_in_selectors:
            return "logged_in", "matched_logged_in_selector"

        if found_logged_out_selectors:
            return "logged_out", "matched_logged_out_selector"

        return "unknown", "no_selector_or_url_marker_matched"

    async def _detect_selectors(
        self,
        page: Page,
    ) -> tuple[list[str], list[str]]:
        found_logged_in: list[str] = []

        for selector in self.config.login_detection.logged_in_selectors:
            try:
                element = await page.query_selector(selector)
            except Exception:
                element = None

            if element is not None:
                found_logged_in.append(selector)

        found_logged_out: list[str] = []

        for selector in self.config.login_detection.logged_out_selectors:
            try:
                element = await page.query_selector(selector)
            except Exception:
                element = None

            if element is not None:
                found_logged_out.append(selector)

        return found_logged_in, found_logged_out

    async def _save_screenshot(
        self,
        page: Page,
        *,
        label: str,
    ) -> str | None:
        if not self.config.screenshot_on_detection_failure:
            return None

        try:
            self.config.screenshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )

            screenshot_path = (
                self.config.screenshot_dir
                / f"{label}_{timestamp}.png"
            )

            await page.screenshot(
                path=str(screenshot_path),
                full_page=True,
            )

            return str(screenshot_path)

        except Exception:
            # Screenshot capture is best-effort diagnostics only.
            # A failure here must never mask the underlying detection result.
            return None

    async def _open_context(self):
        if async_playwright is None:
            raise InstagramSessionError(
                "playwright is not installed. Install it with: "
                "pip install playwright && playwright install chromium"
            )

        self._prepare_profile_dir()

        playwright = await async_playwright().start()

        try:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.config.profile_dir),
                headless=self.config.headless,
                viewport={
                    "width": self.config.viewport.width,
                    "height": self.config.viewport.height,
                },
                locale=self.config.locale,
                timezone_id=self.config.timezone_id,
            )
        except Exception:
            await playwright.stop()
            raise

        return playwright, context

    def _dry_run_state(self) -> SessionState:
        return SessionState(
            status="logged_out",
            checked_at=self._now(),
            current_url=None,
            reason="dry_run_mode_no_browser_launched",
            screenshot_path=None,
            dry_run=True,
        )

    async def check(self) -> SessionState:
        """
        Navigate to Instagram and report logged_in / logged_out / unknown.

        Read-only. Never posts, replies, likes, follows, unfollows, or
        sends direct messages.
        """
        if self.config.dry_run:
            return self._dry_run_state()

        playwright, context = await self._open_context()

        try:
            page = (
                context.pages[0]
                if context.pages
                else await context.new_page()
            )

            try:
                await page.goto(
                    self.config.base_url,
                    timeout=self.config.check_timeout_ms,
                    wait_until="domcontentloaded",
                )
            except PlaywrightTimeoutError as exc:
                raise SessionDetectionError(
                    f"Timed out navigating to {self.config.base_url}"
                ) from exc

            await page.wait_for_timeout(1500)

            found_in, found_out = await self._detect_selectors(page)

            status, reason = self.classify(
                current_url=page.url,
                found_logged_in_selectors=found_in,
                found_logged_out_selectors=found_out,
                logged_out_url_markers=(
                    self.config.login_detection.logged_out_url_markers
                ),
                pending_url_markers=(
                    self.config.login_detection.pending_url_markers
                ),
            )

            screenshot_path = None

            if status == "unknown":
                screenshot_path = await self._save_screenshot(
                    page,
                    label="check_unknown",
                )

            return SessionState(
                status=status,
                checked_at=self._now(),
                current_url=page.url,
                reason=reason,
                screenshot_path=screenshot_path,
                dry_run=False,
            )

        finally:
            await context.close()
            await playwright.stop()

    async def login(self) -> SessionState:
        """
        Open Instagram and wait for a human to log in manually.

        This method never types a username or password. It only opens
        the login page and polls the page for a logged_in signal until
        the configured timeout elapses.
        """
        if self.config.dry_run:
            return self._dry_run_state()

        playwright, context = await self._open_context()

        try:
            page = (
                context.pages[0]
                if context.pages
                else await context.new_page()
            )

            await page.goto(
                self.config.login_url,
                timeout=self.config.navigation_timeout_ms,
                wait_until="domcontentloaded",
            )

            print("[InstagramSession] Browser opened. Please log in manually.")
            print(
                "[InstagramSession] Waiting up to "
                f"{self.config.login_wait_timeout_ms // 1000}s "
                "for login to complete..."
            )

            poll_interval_ms = 2000
            elapsed_ms = 0

            while elapsed_ms < self.config.login_wait_timeout_ms:
                await page.wait_for_timeout(poll_interval_ms)
                elapsed_ms += poll_interval_ms

                found_in, found_out = await self._detect_selectors(page)

                status, reason = self.classify(
                    current_url=page.url,
                    found_logged_in_selectors=found_in,
                    found_logged_out_selectors=found_out,
                    logged_out_url_markers=(
                        self.config.login_detection.logged_out_url_markers
                    ),
                    pending_url_markers=(
                        self.config.login_detection.pending_url_markers
                    ),
                )

                if status == "logged_in":
                    print("[InstagramSession] Login detected.")

                    return SessionState(
                        status="logged_in",
                        checked_at=self._now(),
                        current_url=page.url,
                        reason=reason,
                        screenshot_path=None,
                        dry_run=False,
                    )

            screenshot_path = await self._save_screenshot(
                page,
                label="login_timeout",
            )

            raise LoginTimeoutError(
                "Manual login was not detected within "
                f"{self.config.login_wait_timeout_ms}ms. "
                f"Screenshot saved at: {screenshot_path}"
            )

        finally:
            await context.close()
            await playwright.stop()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Instagram Browser Session (Phase 8A)"
    )

    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument(
        "--login",
        action="store_true",
        help="Open Instagram and wait for manual login.",
    )

    action.add_argument(
        "--check",
        action="store_true",
        help="Report whether the current session is logged_in or logged_out.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip launching a real browser; return a safe simulated "
            "session state."
        ),
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode (visible browser is the default).",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate instagram.yaml config file.",
    )

    return parser.parse_args(argv)


def _print_state(state: SessionState, *, title: str) -> None:
    print()
    print(title)
    print("-" * len(title))
    print(f"status:      {state.status}")
    print(f"checked_at:  {state.checked_at}")
    print(f"current_url: {state.current_url}")
    print(f"reason:      {state.reason}")
    print(f"screenshot:  {state.screenshot_path}")
    print(f"dry_run:     {state.dry_run}")
    print()


async def _run(arguments: argparse.Namespace) -> int:
    config = load_config(
        config_path=arguments.config,
        headless_override=True if arguments.headless else None,
        dry_run_override=True if arguments.dry_run else None,
    )

    session = InstagramSession(config)

    if arguments.check:
        try:
            state = await session.check()
        except InstagramSessionError as exc:
            print(f"[InstagramSession] check failed: {exc}")
            return 1

        _print_state(state, title="AIKO Instagram Session — Check")
        return 0

    try:
        state = await session.login()
    except InstagramSessionError as exc:
        print(f"[InstagramSession] login failed: {exc}")
        return 1

    _print_state(state, title="AIKO Instagram Session — Login")
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = asyncio.run(_run(arguments))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
