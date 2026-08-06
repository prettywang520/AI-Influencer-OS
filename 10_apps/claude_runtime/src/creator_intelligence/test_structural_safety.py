import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR.parents[1] / "config" / "creator_intelligence"

_SOURCE_FILES = sorted(p for p in PACKAGE_DIR.glob("*.py") if not p.name.startswith("test_"))
_SOURCE_TEXT_BY_FILE = {p: p.read_text(encoding="utf-8") for p in _SOURCE_FILES}
_ALL_SOURCE_TEXT = "\n".join(_SOURCE_TEXT_BY_FILE.values())

_CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))
_ALL_CONFIG_TEXT = "\n".join(p.read_text(encoding="utf-8") for p in _CONFIG_FILES)

_FORBIDDEN_IMPORT_PREFIXES = (
    "import requests",
    "from requests",
    "import httpx",
    "from httpx",
    "import playwright",
    "from playwright",
    "import selenium",
    "from selenium",
    "import browser",
    "from .browser",
    "from src.browser",
    "from .social",
    "from src.social",
    "from .publishing",
    "from src.publishing",
    "from .instagram_connector",
    "from src.instagram_connector",
    "from .03_personas",
)

_REAL_CREATOR_HANDLES = ("carlysuen112", "aitana_10_01")


class NoForbiddenImportsTests(unittest.TestCase):
    def test_no_forbidden_import_in_any_module(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for line in text.splitlines():
                stripped = line.strip()
                for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
                    self.assertFalse(
                        stripped.startswith(forbidden),
                        f"{path.name} contains forbidden import: {stripped!r}",
                    )

    def test_no_persona_directory_referenced(self):
        self.assertNotIn("03_personas", _ALL_SOURCE_TEXT)

    def test_no_reply_production_module_referenced(self):
        for forbidden in ("reply_agent", "reply_service", "reply_dispatcher", "reply_worker", "dm_reply_brain"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT)

    def test_no_rendering_module_referenced(self):
        for forbidden in ("video_engine", "renderer_execution_engine", "filter_graph_builder", "overlay_render_engine"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT)


class NoNetworkOrAutomationCodeTests(unittest.TestCase):
    def test_no_network_calls(self):
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download("):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT)

    def test_no_browser_automation_keywords(self):
        for forbidden in ("webdriver", "chromedriver", "playwright.sync_api", "playwright.async_api"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT.lower())

    def test_no_login_or_session_keywords(self):
        for forbidden in ("login(", "log_in(", "session_cookie", "auth_token"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT)

    def test_no_subprocess_usage(self):
        for line in _ALL_SOURCE_TEXT.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import subprocess"))
            self.assertFalse(stripped.startswith("from subprocess"))


class NoRealCreatorHandleTests(unittest.TestCase):
    def test_no_real_handle_in_source_code(self):
        for handle in _REAL_CREATOR_HANDLES:
            self.assertNotIn(handle, _ALL_SOURCE_TEXT)

    def test_no_real_handle_in_config(self):
        for handle in _REAL_CREATOR_HANDLES:
            self.assertNotIn(handle, _ALL_CONFIG_TEXT)

    def test_no_instagram_url_in_source_code(self):
        self.assertNotIn("instagram.com/", _ALL_SOURCE_TEXT)


class EvidenceTypeClosedSetTests(unittest.TestCase):
    def test_evidence_type_has_no_automated_collection_member(self):
        from src.creator_intelligence.evidence import EvidenceType

        forbidden = {"scraped", "api_fetch", "browser_automation", "login_session", "auto_collected"}
        self.assertFalse(forbidden.intersection(EvidenceType.ALL))


if __name__ == "__main__":
    unittest.main()
