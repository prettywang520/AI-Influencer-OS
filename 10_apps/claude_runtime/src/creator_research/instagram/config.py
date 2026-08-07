"""Configuration loading for the Instagram Research Connector. Matches
the exact `load_*_config()` convention used throughout this codebase:
raises a module-specific error on a missing/invalid *file*; individual
missing *keys* within a present file fall back to documented defaults.
No credential field exists anywhere in this module or the YAML it
reads -- checked structurally in test_structural_safety.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .exceptions import InstagramConfigError

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "creator_research" / "instagram.yaml"


def _runtime_root() -> Path:
    """
    config.py location: 10_apps/claude_runtime/src/creator_research/instagram/config.py
    parents[3] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return _runtime_root() / DEFAULT_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class InstagramConnectorConfig:
    connector_version: str
    read_only: bool
    access_limit_behavior: str
    allowed_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]

    max_posts_per_job: int
    max_reels_per_job: int
    max_comments_per_post: int
    max_creator_replies_per_post: int
    max_highlight_items: int
    max_scroll_rounds_per_section: int

    rate_limit_enabled: bool
    minimum_delay_seconds: float
    maximum_delay_seconds: float
    backoff_seconds: tuple[float, ...]
    max_retries_per_navigation: int

    headless: bool
    reuse_existing_profile: bool
    profile_directory: str
    page_ready_timeout_seconds: float
    selector_timeout_seconds: float
    viewport_width: int
    viewport_height: int
    locale: str
    timezone_id: str

    save_text: bool
    save_public_metrics: bool
    save_visual_annotations: bool
    preserve_source_url: bool
    preserve_published_at: bool
    preserve_capture_timestamp: bool

    checkpoint_enabled: bool
    checkpoint_every_items: int
    checkpoint_directory: str

    screenshots_on_error: bool
    save_dom_snippet_on_error: bool
    redact_usernames_in_audience_comments: bool
    diagnostics_directory: str
    screenshots_directory: str

    def resolved_profile_directory(self) -> Path:
        return _runtime_root() / self.profile_directory

    def resolved_checkpoint_directory(self) -> Path:
        return _runtime_root() / self.checkpoint_directory

    def resolved_diagnostics_directory(self) -> Path:
        return _runtime_root() / self.diagnostics_directory

    def resolved_screenshots_directory(self) -> Path:
        return _runtime_root() / self.screenshots_directory


def load_instagram_connector_config(config_path: str | Path | None = None) -> InstagramConnectorConfig:
    path = Path(config_path) if config_path else default_config_path()
    if not path.exists():
        raise InstagramConfigError(f"Instagram connector config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise InstagramConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise InstagramConfigError(f"Instagram connector config is empty or invalid: {path}")

    instagram_section = raw.get("instagram") or {}
    limits_section = raw.get("limits") or {}
    rate_limit_section = raw.get("rate_limit") or {}
    browser_section = raw.get("browser") or {}
    evidence_section = raw.get("evidence") or {}
    checkpoint_section = raw.get("checkpoint") or {}
    diagnostics_section = raw.get("diagnostics") or {}

    return InstagramConnectorConfig(
        connector_version=str(instagram_section.get("connector_version", "12B.2")),
        read_only=bool(instagram_section.get("read_only", True)),
        access_limit_behavior=str(instagram_section.get("access_limit_behavior", "skip")),
        allowed_actions=tuple(instagram_section.get("allowed_actions", [])),
        forbidden_actions=tuple(instagram_section.get("forbidden_actions", [])),
        max_posts_per_job=int(limits_section.get("max_posts_per_job", 100)),
        max_reels_per_job=int(limits_section.get("max_reels_per_job", 50)),
        max_comments_per_post=int(limits_section.get("max_comments_per_post", 100)),
        max_creator_replies_per_post=int(limits_section.get("max_creator_replies_per_post", 100)),
        max_highlight_items=int(limits_section.get("max_highlight_items", 100)),
        max_scroll_rounds_per_section=int(limits_section.get("max_scroll_rounds_per_section", 50)),
        rate_limit_enabled=bool(rate_limit_section.get("enabled", True)),
        minimum_delay_seconds=float(rate_limit_section.get("minimum_delay_seconds", 1.5)),
        maximum_delay_seconds=float(rate_limit_section.get("maximum_delay_seconds", 4.0)),
        backoff_seconds=tuple(float(v) for v in rate_limit_section.get("backoff_seconds", [10, 30, 60])),
        max_retries_per_navigation=int(rate_limit_section.get("max_retries_per_navigation", 2)),
        headless=bool(browser_section.get("headless", False)),
        reuse_existing_profile=bool(browser_section.get("reuse_existing_profile", True)),
        profile_directory=str(
            browser_section.get("profile_directory", "browser_profile/creator_research_instagram")
        ),
        page_ready_timeout_seconds=float(browser_section.get("page_ready_timeout_seconds", 20)),
        selector_timeout_seconds=float(browser_section.get("selector_timeout_seconds", 10)),
        viewport_width=int(browser_section.get("viewport_width", 1280)),
        viewport_height=int(browser_section.get("viewport_height", 900)),
        locale=str(browser_section.get("locale", "en-US")),
        timezone_id=str(browser_section.get("timezone_id", "UTC")),
        save_text=bool(evidence_section.get("save_text", True)),
        save_public_metrics=bool(evidence_section.get("save_public_metrics", True)),
        save_visual_annotations=bool(evidence_section.get("save_visual_annotations", False)),
        preserve_source_url=bool(evidence_section.get("preserve_source_url", True)),
        preserve_published_at=bool(evidence_section.get("preserve_published_at", True)),
        preserve_capture_timestamp=bool(evidence_section.get("preserve_capture_timestamp", True)),
        checkpoint_enabled=bool(checkpoint_section.get("enabled", True)),
        checkpoint_every_items=int(checkpoint_section.get("checkpoint_every_items", 10)),
        checkpoint_directory=str(
            checkpoint_section.get("directory", "output/creator_research/instagram/checkpoints")
        ),
        screenshots_on_error=bool(diagnostics_section.get("screenshots_on_error", True)),
        save_dom_snippet_on_error=bool(diagnostics_section.get("save_dom_snippet_on_error", False)),
        redact_usernames_in_audience_comments=bool(
            diagnostics_section.get("redact_usernames_in_audience_comments", True)
        ),
        diagnostics_directory=str(
            diagnostics_section.get("directory", "output/creator_research/instagram/diagnostics")
        ),
        screenshots_directory=str(
            diagnostics_section.get("screenshots_directory", "output/creator_research/instagram/screenshots")
        ),
    )
