"""InstagramBrowserAdapter -- the injectable browser/session boundary.
`observer.py`'s business logic only ever talks to this Protocol, never
to Playwright (or anything network-shaped) directly, so it stays fully
testable without a browser. `PlaywrightInstagramBrowserAdapter` is the
real implementation (self-contained; does not import
src/social/instagram_session.py -- see docs/creator_research/
instagram_connector.md for why); `FakeBrowserAdapter` is the in-memory
test/demo double, defined here (not per-test-file) because simulating
a whole navigable fake page tree is substantial enough that
duplicating it across five-plus test files would be pure waste, and
the task itself names FakeBrowserAdapter as a first-class fixture.

Neither adapter has a method that types into a password field, reads a
cookie, solves a CAPTCHA, or retries past a challenge screen -- there
is no such capability anywhere in this file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .exceptions import InstagramConnectorError

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - exercised only when playwright is absent
    sync_playwright = None
    PlaywrightTimeoutError = TimeoutError


class InstagramBrowserAdapter(Protocol):
    def open_profile(self, url: str) -> None: ...

    def get_current_url(self) -> str: ...

    def wait_for_page_ready(self, timeout_seconds: float) -> bool: ...

    def query(self, selector_candidates: list[str]) -> Any | None: ...

    def query_all(self, selector_candidates: list[str]) -> list[Any]: ...

    def click(self, element: Any) -> None: ...

    def scroll(self, amount: str = "page") -> None: ...

    def get_text(self, element: Any) -> str | None: ...

    def get_attribute(self, element: Any, name: str) -> str | None: ...

    def get_screenshot_reference(self, label: str) -> str | None: ...

    def close(self) -> None: ...


class PlaywrightInstagramBrowserAdapter:
    """Real adapter. Opens a persistent Chromium context against its
    own profile directory (never the reply-production system's
    `browser_profile/instagram`, to avoid two Playwright processes
    locking the same user_data_dir). Login is always manual, by a
    human, in a visible browser -- this class has no method that
    accepts or reads a credential."""

    def __init__(
        self,
        profile_dir: str | Path,
        *,
        headless: bool = False,
        viewport_width: int = 1280,
        viewport_height: int = 900,
        locale: str = "en-US",
        timezone_id: str = "UTC",
        navigation_timeout_ms: int = 30000,
        selector_timeout_ms: int = 10000,
        screenshot_dir: str | Path | None = None,
    ) -> None:
        if sync_playwright is None:
            raise InstagramConnectorError(
                "playwright is not installed. Install it with: "
                "pip install playwright && playwright install chromium"
            )
        self._profile_dir = Path(profile_dir)
        self._headless = headless
        self._viewport = {"width": viewport_width, "height": viewport_height}
        self._locale = locale
        self._timezone_id = timezone_id
        self._navigation_timeout_ms = navigation_timeout_ms
        self._selector_timeout_ms = selector_timeout_ms
        self._screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self._playwright = None
        self._context = None
        self._page = None

    def _ensure_context(self) -> None:
        if self._context is not None:
            return
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self._profile_dir),
            headless=self._headless,
            viewport=self._viewport,
            locale=self._locale,
            timezone_id=self._timezone_id,
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()

    def open_profile(self, url: str) -> None:
        self._ensure_context()
        self._page.goto(url, timeout=self._navigation_timeout_ms, wait_until="domcontentloaded")

    def get_current_url(self) -> str:
        return self._page.url if self._page is not None else ""

    def wait_for_page_ready(self, timeout_seconds: float) -> bool:
        if self._page is None:
            return False
        try:
            self._page.wait_for_load_state("networkidle", timeout=timeout_seconds * 1000)
            return True
        except PlaywrightTimeoutError:
            return False

    def query(self, selector_candidates: list[str]) -> Any | None:
        for selector in selector_candidates:
            try:
                element = self._page.query_selector(selector)
            except Exception:
                element = None
            if element is not None:
                return element
        return None

    def query_all(self, selector_candidates: list[str]) -> list[Any]:
        for selector in selector_candidates:
            try:
                elements = self._page.query_selector_all(selector)
            except Exception:
                elements = []
            if elements:
                return elements
        return []

    def click(self, element: Any) -> None:
        try:
            element.click(timeout=self._selector_timeout_ms)
        except Exception:
            pass

    def scroll(self, amount: str = "page") -> None:
        delta = 2000 if amount == "page" else 500
        try:
            self._page.mouse.wheel(0, delta)
        except Exception:
            pass

    def get_text(self, element: Any) -> str | None:
        try:
            return element.text_content()
        except Exception:
            return None

    def get_attribute(self, element: Any, name: str) -> str | None:
        try:
            return element.get_attribute(name)
        except Exception:
            return None

    def get_screenshot_reference(self, label: str) -> str | None:
        if self._page is None or self._screenshot_dir is None:
            return None
        try:
            from datetime import datetime, timezone

            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = self._screenshot_dir / f"{label}_{timestamp}.png"
            self._page.screenshot(path=str(path), full_page=True)
            return str(path)
        except Exception:
            # Screenshot capture is best-effort diagnostics only -- a
            # failure here must never mask the underlying observation.
            return None

    def close(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()


class FakeElement:
    """In-memory stand-in for a Playwright element handle."""

    def __init__(self, text: str | None = None, attributes: dict[str, str] | None = None) -> None:
        self.text = text
        self.attributes = attributes or {}


class FakePage:
    """One simulated page's worth of registered fake elements, keyed
    by the exact selector string a test/demo registers them under.
    `reveal_batch` (if set) simulates pagination: query()/query_all()
    only see the first N elements until enough scroll() calls reveal
    more, letting tests exercise stop-after-no-new-items logic."""

    def __init__(self) -> None:
        self.elements: dict[str, list[FakeElement]] = {}
        self._reveal_counts: dict[str, int] = {}
        self._reveal_increment: dict[str, int] = {}
        self.ready = True

    def register(self, selector: str, elements: list[FakeElement], *, reveal_batch: int | None = None) -> None:
        self.elements[selector] = elements
        if reveal_batch is not None:
            self._reveal_counts[selector] = reveal_batch
            self._reveal_increment[selector] = reveal_batch
        else:
            self._reveal_counts[selector] = len(elements)

    def visible(self, selector: str) -> list[FakeElement]:
        all_elements = self.elements.get(selector, [])
        limit = self._reveal_counts.get(selector, len(all_elements))
        return all_elements[:limit]

    def reveal_more(self) -> None:
        for selector, increment in self._reveal_increment.items():
            total = len(self.elements.get(selector, []))
            self._reveal_counts[selector] = min(total, self._reveal_counts.get(selector, 0) + increment)


class FakeBrowserAdapter:
    """Test/demo-only implementation of InstagramBrowserAdapter. No
    real browser, no Playwright import required to use it."""

    def __init__(self) -> None:
        self.pages: dict[str, FakePage] = {}
        self.current_url: str = ""
        self.closed: bool = False
        self.click_log: list[Any] = []
        self.scroll_count: int = 0

    def add_page(self, url: str) -> FakePage:
        page = FakePage()
        self.pages[url] = page
        return page

    def _page(self) -> FakePage:
        return self.pages.setdefault(self.current_url, FakePage())

    def open_profile(self, url: str) -> None:
        self.current_url = url
        self.pages.setdefault(url, FakePage())

    def get_current_url(self) -> str:
        return self.current_url

    def wait_for_page_ready(self, timeout_seconds: float) -> bool:
        return self._page().ready

    def query(self, selector_candidates: list[str]) -> FakeElement | None:
        page = self._page()
        for selector in selector_candidates:
            visible = page.visible(selector)
            if visible:
                return visible[0]
        return None

    def query_all(self, selector_candidates: list[str]) -> list[FakeElement]:
        page = self._page()
        for selector in selector_candidates:
            visible = page.visible(selector)
            if visible:
                return visible
        return []

    def click(self, element: Any) -> None:
        self.click_log.append(element)

    def scroll(self, amount: str = "page") -> None:
        self.scroll_count += 1
        self._page().reveal_more()

    def get_text(self, element: FakeElement | None) -> str | None:
        return element.text if element is not None else None

    def get_attribute(self, element: FakeElement | None, name: str) -> str | None:
        return element.attributes.get(name) if element is not None else None

    def get_screenshot_reference(self, label: str) -> str | None:
        return f"fake-screenshot://{label}"

    def close(self) -> None:
        self.closed = True
