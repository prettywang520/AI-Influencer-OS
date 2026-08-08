import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR.parents[2] / "config" / "creator_learning" / "engine.yaml"

_SOURCE_FILES = sorted(
    p for p in PACKAGE_DIR.rglob("*.py") if not p.name.startswith("test_") and p.name != "__init__.py"
)
_SOURCE_TEXT_BY_FILE = {p: p.read_text(encoding="utf-8") for p in _SOURCE_FILES}
_ALL_SOURCE_TEXT = "\n".join(_SOURCE_TEXT_BY_FILE.values())
_ALL_SOURCE_TEXT_LOWER = _ALL_SOURCE_TEXT.lower()
_CONFIG_TEXT = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.is_file() else ""

_FORBIDDEN_IMPORT_PREFIXES = (
    "import requests", "from requests", "import httpx", "from httpx",
    "import selenium", "from selenium", "import playwright", "from playwright",
    "import cv2", "from cv2",
    "import face_recognition", "from face_recognition",
    "import whisper", "from whisper",
    "import torch", "from torch",
)

_FORBIDDEN_METHOD_NAMES = (
    "def follow(", "def unfollow(", "def like(", "def unlike(", "def comment(",
    "def send_dm(", "def dm(", "def share(", "def save_post(", "def publish(", "def edit_profile(",
    "def post_comment(", "def send_message(", "def scrape(", "def scroll(",
)

# Confirmed absent from every phase this session -- real target-creator
# accounts must never appear in source or config, matching every prior
# creator_research/creator_intelligence phase's own structural safety test.
_TARGET_CREATOR_USERNAMES = ("carlysuen112", "aitana_10_01")


class NoNetworkOrBrowserInfrastructureTests(unittest.TestCase):
    def test_no_forbidden_imports(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for line in text.splitlines():
                stripped = line.strip()
                for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
                    self.assertFalse(stripped.startswith(forbidden), f"{path.name}: {stripped!r}")

    def test_no_http_client_calls(self):
        for forbidden in ("requests.get(", "requests.post(", "httpx.get(", "httpx.post(", "urllib.request.urlopen("):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT)

    def test_no_engagement_or_scraping_methods_defined(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for forbidden in _FORBIDDEN_METHOD_NAMES:
                self.assertNotIn(forbidden, text, f"{path.name} defines forbidden method {forbidden!r}")


class NoCredentialHandlingTests(unittest.TestCase):
    def test_no_password_assignment_or_handling(self):
        forbidden_patterns = ("password =", "password:", "password_value", ".fill(password", "type(password")
        for forbidden in forbidden_patterns:
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_cookie_reading_or_export(self):
        for forbidden in ("cookies.json", "export_cookies", "get_cookies", ".cookies(", "cookie_jar"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_session_token_or_auth_header_handling(self):
        for forbidden in ("session_token", "auth_header", "bearer ", "x-csrftoken"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_config_file_has_no_credential_key(self):
        forbidden_keys = ("password:", "username:", "cookie:", "token:", "secret:")
        for line in _CONFIG_TEXT.lower().splitlines():
            stripped = line.strip()
            for forbidden in forbidden_keys:
                self.assertFalse(stripped.startswith(forbidden), f"engine.yaml: {stripped!r}")


class NoTargetCreatorHardcodingTests(unittest.TestCase):
    def test_target_creator_usernames_absent_from_source(self):
        for username in _TARGET_CREATOR_USERNAMES:
            self.assertNotIn(username, _ALL_SOURCE_TEXT_LOWER)

    def test_target_creator_usernames_absent_from_config(self):
        for username in _TARGET_CREATOR_USERNAMES:
            self.assertNotIn(username, _CONFIG_TEXT.lower())


class NoPersonaOrPublishingCouplingTests(unittest.TestCase):
    def test_no_import_of_persona_or_publishing_packages(self):
        forbidden_import_fragments = (
            "03_personas", "from src.social", "import src.social",
            "from src.publishing", "import src.publishing",
        )
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for fragment in forbidden_import_fragments:
                self.assertNotIn(fragment, text, f"{path.name} references {fragment!r}")


if __name__ == "__main__":
    unittest.main()
