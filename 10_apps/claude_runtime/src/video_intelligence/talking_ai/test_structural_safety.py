import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR.parents[2] / "config" / "video_intelligence" / "talking_ai.yaml"

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

# This package must never import creator_research, creator_intelligence,
# creator_learning, video_intelligence.instagram (the Instagram
# Connector), rendering, publishing, production, or social -- only the
# stable, public video_intelligence.models/evidence primitives named in
# the approved plan (ConfidenceLevel, ConfidenceThresholds,
# compute_confidence_level, min_confidence, video_id_for, VideoPlatform)
# are reused, and only workflow.py's optional bridge reads (never
# imports as a dependency edge into) video_intelligence.instagram's own
# record type for type-shaped, best-effort conversion.
_FORBIDDEN_IMPORT_LINE_PREFIXES = (
    "from src.creator_research", "import src.creator_research",
    "from creator_research", "import creator_research",
    "from src.creator_intelligence", "import src.creator_intelligence",
    "from creator_intelligence", "import creator_intelligence",
    "from src.creator_learning", "import src.creator_learning",
    "from creator_learning", "import creator_learning",
    "from src.social", "import src.social",
    "from src.publishing", "import src.publishing",
    "from src.production", "import src.production",
    "from src.rendering", "import src.rendering",
)

_FORBIDDEN_METHOD_NAMES = (
    "def follow(", "def unfollow(", "def like(", "def unlike(", "def comment(", "def reply(",
    "def send_dm(", "def dm(", "def share(", "def save_post(", "def publish(", "def edit_profile(",
    "def post_comment(", "def send_message(", "def scrape(", "def scroll(", "def login(",
    "def clone_voice(", "def synthesize_speech(", "def generate_video(", "def generate_lipsync(",
)

# Confirmed absent from every phase this session -- real target-creator/
# persona names must never appear in source or config, including the
# task's own explicitly-named future-only reference account.
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
        for forbidden in ("whisper.load_model", "openai.chat", "chatcompletion", "generativemodel", "text_to_speech", "voice_clone"):
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


class NoRealAccountOrPersonaReferenceTests(unittest.TestCase):
    def test_forbidden_names_absent_from_source(self):
        for name in _FORBIDDEN_NAMES:
            self.assertNotIn(name.lower(), _ALL_SOURCE_TEXT_LOWER)

    def test_forbidden_names_absent_from_config(self):
        for name in _FORBIDDEN_NAMES:
            self.assertNotIn(name.lower(), _CONFIG_TEXT.lower())

    def test_no_creator_specific_configuration_in_yaml(self):
        for forbidden_key in ("creator_id:", "creator_username:", "persona:"):
            self.assertNotIn(forbidden_key, _CONFIG_TEXT.lower())

    def test_howdoyoulikedurian_absent_from_every_source_file_individually(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            self.assertNotIn("howdoyoulikedurian", text.lower(), f"{path.name} references howdoyoulikedurian")


class NoHumanAIClassificationTests(unittest.TestCase):
    def test_no_banned_field_names_anywhere_in_source(self):
        for path, text in _SOURCE_TEXT_BY_FILE.items():
            for banned in _BANNED_CLASSIFICATION_FIELD_NAMES:
                self.assertNotIn(banned, text.lower(), f"{path.name} defines banned field {banned!r}")

    def test_naturalness_module_never_claims_definitive_human_or_ai_origin(self):
        naturalness_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "naturalness.py"].lower()
        for banned_phrase in ("this video is ai", "this video is human", "confirmed ai-generated", "confirmed human"):
            self.assertNotIn(banned_phrase, naturalness_text)


class SchemaHonestyTests(unittest.TestCase):
    # Both modules' docstrings legitimately *discuss* the banned phrases
    # in prose while explaining why they're never emitted (e.g. "so no
    # code path here could ever claim phoneme-level validation") -- a
    # bare substring check over the whole file would misread that
    # explanation as a violation. Functional coverage that no *rationale
    # string* the analyzer actually returns contains these phrases
    # already lives in test_lip_sync.py/test_speech_sync.py; here we
    # only check the rationale-construction code itself never
    # concatenates the banned phrase into an f-string/literal it emits.
    def test_lip_sync_source_never_constructs_a_phoneme_level_claim(self):
        lip_sync_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "lip_sync.py"]
        rationale_lines = [line for line in lip_sync_text.splitlines() if "rationale" in line.lower()]
        for line in rationale_lines:
            self.assertNotIn("phoneme-level validation", line)

    def test_pause_behavior_source_never_constructs_a_breathing_detected_claim(self):
        speech_sync_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "speech_sync.py"]
        rationale_lines = [line for line in speech_sync_text.splitlines() if "rationale" in line.lower()]
        for line in rationale_lines:
            self.assertNotIn("breathing detected", line.lower())


class SiblingNotExtensionTests(unittest.TestCase):
    def test_production_dna_never_imports_video_dna(self):
        production_dna_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "production_dna.py"]
        for line in production_dna_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from ") or stripped.startswith("import "):
                self.assertNotIn("VideoDNA", stripped, f"import line pulls in VideoDNA: {stripped!r}")

    def test_models_never_redefines_confidence_level(self):
        # Confidence vocabulary must be imported, never redefined --
        # a second, silently-diverging definition would violate the
        # "one confidence philosophy" invariant this whole codebase relies on.
        models_text = _SOURCE_TEXT_BY_FILE[PACKAGE_DIR / "models.py"]
        self.assertIn("from src.video_intelligence.models import", models_text)
        self.assertNotIn("class ConfidenceLevel", models_text)


if __name__ == "__main__":
    unittest.main()
