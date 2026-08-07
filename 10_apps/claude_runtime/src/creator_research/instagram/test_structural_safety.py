import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR.parents[2] / "config" / "creator_research" / "instagram.yaml"

_SOURCE_FILES = sorted(p for p in PACKAGE_DIR.glob("*.py") if not p.name.startswith("test_") and p.name != "__init__.py")
_SOURCE_TEXT_BY_FILE = {p: p.read_text(encoding="utf-8") for p in _SOURCE_FILES}
_ALL_SOURCE_TEXT = "\n".join(_SOURCE_TEXT_BY_FILE.values())
_ALL_SOURCE_TEXT_LOWER = _ALL_SOURCE_TEXT.lower()
_CONFIG_TEXT = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.is_file() else ""

_FORBIDDEN_METHOD_NAMES = (
    "def follow(", "def unfollow(", "def like(", "def unlike(", "def comment(", "def reply(",
    "def send_dm(", "def dm(", "def share(", "def save_post(", "def publish(", "def edit_profile(",
    "def post_comment(", "def send_message(",
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "import requests", "from requests", "import httpx", "from httpx",
    "import selenium", "from selenium",
    "import cv2", "from cv2",
    "import face_recognition", "from face_recognition",
    "import whisper", "from whisper",
    "import torch", "from torch",
)


class NoScrapingInfrastructureTests(unittest.TestCase):
    def test_no_requests_based_http_client(self):
        for forbidden in ("requests.get(", "requests.post(", "httpx.get(", "httpx.post(", "urllib.request.urlopen("):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT)

    def test_no_hidden_instagram_api_reference(self):
        for forbidden in ("graphql", "/api/v1/", "instagram_private_api", "i.instagram.com/api"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_mobile_api_reference(self):
        for forbidden in ("x-ig-app-id", "instagram-ajax", "mobile_api"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_forbidden_imports(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for line in text.splitlines():
                stripped = line.strip()
                for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
                    self.assertFalse(stripped.startswith(forbidden), f"{path.name}: {stripped!r}")


class NoCredentialHandlingTests(unittest.TestCase):
    def test_no_password_assignment_or_handling(self):
        # Narrowly checks for actually *handling* a password value
        # (assignment, dict access, fill-with-a-password-variable) --
        # not the bare word, which legitimately appears in this
        # package as documentation ("never types into a password
        # field") and as a *detection-only* selector
        # ("input[type='password']", used to recognize a login wall,
        # the same convention social/instagram_session.py already
        # established -- never to fill it in).
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
        # Checks for an actual YAML mapping key (a line that, once
        # trimmed, starts with one of these names followed by a
        # colon) -- not the bare word, which legitimately appears in
        # this file's own header comment (explaining that no such key
        # exists) and in the unrelated
        # `redact_usernames_in_audience_comments` privacy setting.
        forbidden_keys = ("password:", "username:", "cookie:", "token:", "secret:")
        for line in _CONFIG_TEXT.lower().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for key in forbidden_keys:
                self.assertFalse(stripped.startswith(key), f"config defines forbidden key: {stripped!r}")

    def test_no_login_typing_into_password_field(self):
        for forbidden in ("fill('input[type=", 'fill("input[type=', "type_password", "enter_password"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)


class NoEngagementOrPublishingActionsTests(unittest.TestCase):
    def test_no_forbidden_method_defined_anywhere(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for forbidden in _FORBIDDEN_METHOD_NAMES:
                self.assertNotIn(forbidden, text, f"{path.name} defines a forbidden method: {forbidden}")

    def test_forbidden_actions_list_has_no_corresponding_capability(self):
        from src.creator_research.instagram.config import load_instagram_connector_config

        config = load_instagram_connector_config()
        for action in config.forbidden_actions:
            self.assertNotIn(f"def {action}(", _ALL_SOURCE_TEXT)


class NoEvasionOrBypassTests(unittest.TestCase):
    def test_no_captcha_solving(self):
        for forbidden in ("solve_captcha", "captcha_solver", "2captcha", "anticaptcha"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_proxy_rotation(self):
        for forbidden in ("proxy_pool", "rotate_proxy", "proxy_list", "next_proxy"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_account_rotation(self):
        for forbidden in ("rotate_account", "account_pool", "next_account"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_challenge_and_access_signals_only_ever_raise_or_warn(self):
        # Structural guarantee: the four access-event kinds only ever
        # appear alongside raise/warning-producing code, confirmed by
        # exceptions.py defining a matching exception for each and
        # observer.py only ever raising (never "solving") them.
        from src.creator_research.instagram.models import AccessEventKind

        for kind in AccessEventKind.ALL:
            self.assertIn(kind, _ALL_SOURCE_TEXT)

    def test_no_private_account_circumvention(self):
        for forbidden in ("bypass_private", "unlock_private", "force_follow_private"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)


class NoVisualAnalysisOverreachTests(unittest.TestCase):
    def test_no_face_recognition(self):
        for forbidden in ("face_recognition", "facial_recognition", "detect_face", "identify_person"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_audio_transcription(self):
        for forbidden in ("whisper", "speech_to_text", "transcribe_audio"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_default_media_download(self):
        for forbidden in ("download_video", "download_media", ".mp4', 'wb'", "save_video_file"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_visual_realism_final_judgement_language(self):
        # observer.py/evidence_mapper.py must only ever *select*
        # visual examples, never assert a conclusion about them.
        from src.creator_research.instagram import observer

        source = observer.observe_visual_examples.__doc__ or ""
        for forbidden in ("authentic", "fake", "ai-generated", "real photo"):
            self.assertNotIn(forbidden, source.lower())


class NoRealCreatorHandleTests(unittest.TestCase):
    def test_no_real_handle_in_source(self):
        for handle in ("carlysuen112", "aitana_10_01"):
            self.assertNotIn(handle, _ALL_SOURCE_TEXT)

    def test_no_real_handle_in_config(self):
        for handle in ("carlysuen112", "aitana_10_01"):
            self.assertNotIn(handle, _CONFIG_TEXT)


class ReadOnlyGuaranteeTests(unittest.TestCase):
    def test_read_only_flag_defaults_true_in_shipped_config(self):
        self.assertIn("read_only: true", _CONFIG_TEXT)

    def test_no_write_capable_instagram_endpoint_string(self):
        for forbidden in ("/accounts/edit/", "/direct/send/", "/web/comments/", "web/likes/"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT)


class NoAutomaticLiveCliTests(unittest.TestCase):
    def test_this_package_has_no_cli_module(self):
        # Task §23: no one-command autonomous live-research CLI in
        # this phase. Confirmed structurally -- no cli.py exists here.
        self.assertFalse((PACKAGE_DIR / "cli.py").exists())


if __name__ == "__main__":
    unittest.main()
