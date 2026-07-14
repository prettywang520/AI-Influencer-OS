"""
AI Influencer OS
Runtime Configuration
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ===========================
# Load .env
# ===========================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


class Config:

    APP_NAME = os.getenv("APP_NAME", "AI Influencer OS")

    VERSION = os.getenv("VERSION", "1.0.0")

    DEBUG = os.getenv("DEBUG", "True") == "True"

    LANGUAGE = os.getenv("LANGUAGE", "en")

    DEFAULT_PERSONA = os.getenv("DEFAULT_PERSONA", "maya")

    DEFAULT_WORKFLOW = os.getenv(
        "DEFAULT_WORKFLOW",
        "instagram_post"
    )

    OUTPUT_DIR = BASE_DIR / "output"

    LOG_DIR = BASE_DIR / "logs"

    PLUGIN_DIR = BASE_DIR.parent.parent / "11_plugins"

    PERSONA_DIR = BASE_DIR.parent.parent / "03_personas"

    WORKFLOW_DIR = BASE_DIR.parent.parent / "05_workflows"

    DATABASE_DIR = BASE_DIR.parent.parent / "07_database"

    USERS_DIR = BASE_DIR.parent.parent / "08_users"

    BRAIN_DIR = BASE_DIR.parent.parent / "06_brain"


config = Config()