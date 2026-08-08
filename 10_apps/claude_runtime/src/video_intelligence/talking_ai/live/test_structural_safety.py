import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR.parents[3] / "config" / "video_intelligence" / "talking_ai_live.yaml"

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
    "import dlib", "from dlib",
    "import mediapipe", "from mediapipe",
    "import pyannote", "from pyannote",
    "import resemblyzer", "from resemblyzer",
    "import face_recognition", "from face_recognition",
    "import tts", "from tts",
    "import elevenlabs", "from elevenlabs",
)

# This package must never import creator_research.instagram's own
# connector/observer/navigator (the only things in this codebase that
# can open a browser) -- only the stable, public
# creator_research.instagram.models.ReelRecord dataclass is imported
# (by observer_bridge.py), and only creator_intelligence.evidence.Evidence
# (by mapper.py). Also never coupled to creator_intelligence/
# creator_learning, rendering, publishing, production, or social.
_FORBIDDEN_IMPORT_LINE_PREFIXES = (
    "from src.creator_research.instagram.connector", "import src.creator_research.instagram.connector",
    "from creator_research.instagram.connector", "import creator_research.instagram.connector",
    "from src.creator_research.instagram.observer", "import src.creator_research.instagram.observer",
    "from creator_research.instagram.observer", "import creator_research.instagram.observer",
    "from src.creator_research.instagram.navigator", "import src.creator_research.instagram.navigator",
    "from creator_research.instagram.navigator", "import creator_research.instagram.navigator",
    "from src.creator_intelligence", "import src.creator_intelligence",
    "from creator_intelligence", "import creator_intelligence",
    "from src.creator_learning", "import src.creator_learning",
    "from creator_learning", "import creator_learning",
    "from src.social", "import src.social",
    "from src.publishing", "import src.publishing",
    "from src.production", "import src.production",
    "from src.rendering", "import src.rendering",
)

# mapper.py is explicitly allowed to import creator_intelligence.evidence.Evidence
# (a stable, public type, matching Phase 12D.1's own precedent) -- the
# blanket "from src.creator_intelligence" check above would otherwise
# false-positive on that one legitimate line, so it is checked
# separately, restricted to exactly the allowed submodule.
_ALLOWED_CREATOR_INTELLIGENCE_IMPORT = "from src.creator_intelligence.evidence import"

_FORBIDDEN_METHOD_NAMES = (
    "def follow(", "def unfollow(", "def like(", "def unlike(", "def comment(", "def reply(",
    "def send_dm(", "def dm(", "def share(", "def save_post(", "def publish(", "def edit_profile(",
    "def post_comment(", "def send_message(", "def scrape(", "def scroll(", "def login(",
    "def clone_voice(", "def synthesize_speech(", "def generate_video(", "def generate_lipsync(",
    "def open_profile(", "def get_screenshot_reference(",
)

_FORBIDDEN_NAMES = (
    "carlysuen112", "aitana_10_01", "howdoyoulikedurian",
    "winwin", "yua lin", "yua_lin", "張晞晴", "林柔雅", "aiko",
)

_BANNED_CLASSIFICATION_FIELD_NAMES = (
    "is_human", "is_ai", "is_ai_generated", "human_or_ai", "ai_generated_flag", "is_real_person",
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

    def test_no_transcription_speech_or_generation_apis(self):
        for forbidden in (
            "whisper.load_model", "openai.chat", "chatcompletion", "generativemodel",
            "text_to_speech", "voice_clone",
        ):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_computer_vision_or_face_landmark_apis(self):
        for forbidden in ("facemesh", "face_landmarks", "dlib.get_frontal_face", "cv2.videocapture", "mediapipe.solutions"):
            self.assertNotIn(forbidden, _ALL_SOURCE_TEXT_LOWER)

    def test_no_engagement_scraping_or_generation_methods_defined(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for forbidden in _FORBIDDEN_METHOD_NAMES:
                self.assertNotIn(forbidden, text, f"{path.name} defines forbidden method {forbidden!r}")

    def test_no_coupling_to_other_packages(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(_ALLOWED_CREATOR_INTELLIGENCE_IMPORT):
                    continue
                for forbidden in _FORBIDDEN_IMPORT_LINE_PREFIXES:
                    self.assertFalse(stripped.startswith(forbidden), f"{path.name}: {stripped!r}")

    def test_mapper_only_imports_the_stable_evidence_type_from_creator_intelligence(self):
        mapper_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "mapper.py"]
        creator_intelligence_lines = [
            line.strip() for line in mapper_text.splitlines()
            if "creator_intelligence" in line and (line.strip().startswith("from") or line.strip().startswith("import"))
        ]
        self.assertTrue(creator_intelligence_lines)
        for line in creator_intelligence_lines:
            self.assertTrue(line.startswith(_ALLOWED_CREATOR_INTELLIGENCE_IMPORT), line)

    def test_observer_bridge_only_imports_the_stable_reelrecord_type(self):
        observer_bridge_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "observer_bridge.py"]
        creator_research_lines = [
            line.strip() for line in observer_bridge_text.splitlines()
            if "creator_research" in line and (line.strip().startswith("from") or line.strip().startswith("import"))
        ]
        self.assertTrue(creator_research_lines)
        for line in creator_research_lines:
            self.assertEqual(line, "from src.creator_research.instagram.models import ReelRecord")

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


class NoRealAccountOrPersonaReferenceTests(unittest.TestCase):
    def test_forbidden_names_absent_from_source(self):
        for name in _FORBIDDEN_NAMES:
            self.assertNotIn(name.lower(), _ALL_SOURCE_TEXT_LOWER)

    def test_forbidden_names_absent_from_config(self):
        for name in _FORBIDDEN_NAMES:
            self.assertNotIn(name.lower(), _CONFIG_TEXT.lower())

    def test_no_creator_specific_configuration_in_yaml(self):
        # "preserve_creator_username: false" is a policy flag (task's
        # own §20 suggested shape) -- not a per-creator value, so it is
        # not itself a violation; the forbidden keys below would only
        # ever appear if this config named a specific creator directly.
        for forbidden_key in ("creator_id:", "\ncreator_username:", "persona:"):
            self.assertNotIn(forbidden_key, _CONFIG_TEXT.lower())

    def test_howdoyoulikedurian_absent_from_every_source_file_individually(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            self.assertNotIn("howdoyoulikedurian", text.lower(), f"{path.name} references howdoyoulikedurian")


class NoHumanAIClassificationTests(unittest.TestCase):
    def test_no_banned_field_names_anywhere_in_source(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for banned in _BANNED_CLASSIFICATION_FIELD_NAMES:
                self.assertNotIn(banned, text.lower(), f"{path.name} defines banned field {banned!r}")


class NoLiveRunInThisPhaseTests(unittest.TestCase):
    def test_no_playwright_dependency_reachable_from_workflow_cli(self):
        workflow_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "workflow.py"]
        self.assertNotIn("playwright", workflow_text.lower())
        for line in workflow_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from") or stripped.startswith("import"):
                self.assertNotIn("browser", stripped.lower())

    def test_config_defaults_to_no_live_run(self):
        self.assertIn("enabled_by_default: false", _CONFIG_TEXT.lower())
        self.assertIn("require_explicit_authorization: true", _CONFIG_TEXT.lower())

    def test_no_live_instagram_cli_subcommand(self):
        # the only CLI in this package is the offline/local --evidence
        # JSON path (task §22) -- no subcommand accepts a profile URL,
        # username, or "--live" flag.
        workflow_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "workflow.py"]
        for forbidden_flag in ("--live", "--profile-url", "--username", "add_argument(\"--login\")"):
            self.assertNotIn(forbidden_flag, workflow_text)


if __name__ == "__main__":
    unittest.main()
