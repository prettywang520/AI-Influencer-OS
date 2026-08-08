"""Subtitles Analyzer -- scores evidence coverage for a reference
video's on-screen captions: style, position, font feeling, timing,
highlight words, animation. Purely observational -- unrelated to, and
never importing, the production-side subtitle_engine.py/
subtitle_render_engine.py, which plan and render Aiko's own subtitles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags, evidence_for_video, observed_vocabulary
from .evidence import VideoIntelligenceError

TRAIT_NAME = "subtitles"
TAGS = ("subtitles", "subtitle_style", "subtitle_position", "font_feeling", "subtitle_timing", "highlight_words", "subtitle_animation")

DEFAULT_SUBTITLES_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "subtitles.yaml"


class SubtitlesConfigError(VideoIntelligenceError):
    """Raised when config/video_intelligence/subtitles.yaml is missing or invalid."""


def _runtime_root() -> Path:
    """
    subtitles.py location: 10_apps/claude_runtime/src/video_intelligence/subtitles.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_subtitles_config_path() -> Path:
    return _runtime_root() / DEFAULT_SUBTITLES_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class SubtitlesConfig:
    schema_version: str
    style_types: tuple[str, ...]
    position_types: tuple[str, ...]


def load_subtitles_config(config_path: str | Path | None = None) -> SubtitlesConfig:
    path = Path(config_path) if config_path else default_subtitles_config_path()
    if not path.exists():
        raise SubtitlesConfigError(f"Subtitles config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SubtitlesConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise SubtitlesConfigError(f"Subtitles config is empty or invalid: {path}")
    section = raw.get("subtitles") or {}
    return SubtitlesConfig(
        schema_version=str(section.get("schema_version", "1.0")),
        style_types=tuple(section.get("style_types", [])),
        position_types=tuple(section.get("position_types", [])),
    )


class SubtitlesAnalyzer:
    name = TRAIT_NAME

    def __init__(self, subtitles_config: SubtitlesConfig | None = None) -> None:
        self.subtitles_config = subtitles_config or load_subtitles_config()

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        result = analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Subtitle style/timing pattern observed across cited evidence.",
            rationale_without_evidence="No subtitle-tagged evidence supplied; confidence is unknown.",
        )
        own_video_evidence = evidence_for_video(context.evidence, context.video_id)
        relevant = evidence_for_tags(own_video_evidence, TAGS)
        observed_style = observed_vocabulary(relevant, self.subtitles_config.style_types)
        observed_position = observed_vocabulary(relevant, self.subtitles_config.position_types)
        if observed_style:
            result.trait_score.rationale += f" Style observed: {', '.join(observed_style)}."
        if observed_position:
            result.trait_score.rationale += f" Position observed: {', '.join(observed_position)}."
        return result
