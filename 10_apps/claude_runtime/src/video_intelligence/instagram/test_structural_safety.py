import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR.parents[2] / "config" / "video_intelligence" / "instagram_reels.yaml"

_SOURCE_FILES = sorted(p for p in PACKAGE_DIR.glob("*.py") if not p.name.startswith("test_") and p.name != "__init__.py")
_SOURCE_TEXT_BY_FILE = {p: p.read_text(encoding="utf-8") for p in _SOURCE_FILES}
_ALL_SOURCE_TEXT = "\n".join(_SOURCE_TEXT_BY_FILE.values())
_ALL_SOURCE_TEXT_LOWER = _ALL_SOURCE_TEXT.lower()
_CONFIG_TEXT = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.is_file() else ""

_FORBIDDEN_IMPORT_PREFIXES = (
    "import requests", "from requests", "import httpx", "from httpx",
    "import selenium", "from selenium", "import playwright", "from playwright",
    "import whisper", "from whisper",
    "import openai", "from openai",
    "import cv2", "from cv2",
    "import torch", "from torch",
)

# This package must never import creator_research.instagram (or any
# other package that can transitively reach a browser/session) --
# only the stable, public creator_intelligence.evidence.Evidence type
# is imported by mapper.py, and only for type-shaped parsing. Checked
# against actual `import`/`from` statement lines only -- module
# docstrings legitimately discuss creator_research.instagram in prose
# (e.g. explaining what convention is parsed, or stating outright that
# it is never imported), which a bare substring check would misread
# as a violation.
_FORBIDDEN_IMPORT_LINE_PREFIXES = (
    "from src.creator_research", "import src.creator_research",
    "from creator_research", "import creator_research",
    "from src.social", "import src.social",
    "from src.publishing", "import src.publishing",
    "from src.production", "import src.production",
)

_FORBIDDEN_METHOD_NAMES = (
    "def follow(", "def unfollow(", "def like(", "def unlike(", "def comment(", "def reply(",
    "def send_dm(", "def dm(", "def share(", "def save_post(", "def publish(", "def edit_profile(",
    "def post_comment(", "def send_message(", "def scrape(", "def scroll(", "def login(",
)

# Confirmed absent from every phase this session -- real target-creator/
# persona names must never appear in source or config.
_FORBIDDEN_NAMES = (
    "carlysuen112", "aitana_10_01", "howdoyoulikedurian",
    "winwin", "yua lin", "yua_lin", "張晞晴", "林柔雅",
)


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

    def test_no_transcription_or_generation_apis(self):
        for forbidden in ("whisper.load_model", "openai.chat", "chatcompletion", "generativemodel"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_engagement_or_scraping_methods_defined(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for forbidden in _FORBIDDEN_METHOD_NAMES:
                self.assertNotIn(forbidden, text, f"{path.name} defines forbidden method {forbidden!r}")

    def test_no_coupling_to_creator_research_instagram_or_production_packages(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for line in text.splitlines():
                stripped = line.strip()
                for forbidden in _FORBIDDEN_IMPORT_LINE_PREFIXES:
                    self.assertFalse(stripped.startswith(forbidden), f"{path.name}: {stripped!r}")

    def test_no_03_personas_reference(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            self.assertNotIn("03_personas", text, f"{path.name} references 03_personas")


class NoCredentialHandlingTests(unittest.TestCase):
    def test_no_password_assignment_or_handling(self):
        for forbidden in ("password =", "password:", "password_value", ".fill(password", "type(password"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_cookie_reading_or_export(self):
        for forbidden in ("cookies.json", "export_cookies", "get_cookies", ".cookies(", "cookie_jar"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_session_token_or_auth_header_handling(self):
        for forbidden in ("session_token", "auth_header", "bearer ", "x-csrftoken"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)


class NoRealAccountOrPersonaMutationTests(unittest.TestCase):
    def test_forbidden_names_absent_from_source(self):
        for name in _FORBIDDEN_NAMES:
            self.assertNotIn(name.lower(), _ALL_SOURCE_TEXT_LOWER)

    def test_forbidden_names_absent_from_config(self):
        for name in _FORBIDDEN_NAMES:
            self.assertNotIn(name.lower(), _CONFIG_TEXT.lower())

    def test_no_creator_specific_configuration_in_yaml(self):
        # config/video_intelligence/instagram_reels.yaml must stay
        # creator-neutral -- shared across Aiko and any future persona.
        for forbidden_key in ("creator_id:", "creator_username:", "persona:", "aiko:"):
            self.assertNotIn(forbidden_key, _CONFIG_TEXT.lower())


if __name__ == "__main__":
    unittest.main()
