"""Pacing Analyzer -- scores evidence coverage for overall rhythm/
tempo (a "meta" trait synthesized from shot-duration and rhythm
evidence). Complementary to editing.py, which catalogs the concrete
cut/zoom/transition techniques that produce a given tempo -- see
docs/video_intelligence/dna.md for why these are two analyzers, not one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags, evidence_for_video, observed_vocabulary
from .evidence import VideoIntelligenceError

TRAIT_NAME = "pacing"
TAGS = ("pacing", "tempo", "cut_frequency", "shot_duration", "rhythm")

DEFAULT_PACING_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "pacing.yaml"


class PacingConfigError(VideoIntelligenceError):
    """Raised when config/video_intelligence/pacing.yaml is missing or invalid."""


def _runtime_root() -> Path:
    """
    pacing.py location: 10_apps/claude_runtime/src/video_intelligence/pacing.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_pacing_config_path() -> Path:
    return _runtime_root() / DEFAULT_PACING_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class PacingConfig:
    schema_version: str
    tempo_types: tuple[str, ...]
    rhythm_types: tuple[str, ...]


def load_pacing_config(config_path: str | Path | None = None) -> PacingConfig:
    path = Path(config_path) if config_path else default_pacing_config_path()
    if not path.exists():
        raise PacingConfigError(f"Pacing config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise PacingConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise PacingConfigError(f"Pacing config is empty or invalid: {path}")
    section = raw.get("pacing") or {}
    return PacingConfig(
        schema_version=str(section.get("schema_version", "1.0")),
        tempo_types=tuple(section.get("tempo_types", [])),
        rhythm_types=tuple(section.get("rhythm_types", [])),
    )


class PacingAnalyzer:
    name = TRAIT_NAME

    def __init__(self, pacing_config: PacingConfig | None = None) -> None:
        self.pacing_config = pacing_config or load_pacing_config()

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        result = analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Pacing pattern observed across cited evidence.",
            rationale_without_evidence="No pacing-tagged evidence supplied; confidence is unknown.",
        )
        own_video_evidence = evidence_for_video(context.evidence, context.video_id)
        relevant = evidence_for_tags(own_video_evidence, TAGS)
        observed_tempo = observed_vocabulary(relevant, self.pacing_config.tempo_types)
        observed_rhythm = observed_vocabulary(relevant, self.pacing_config.rhythm_types)
        if observed_tempo:
            result.trait_score.rationale += f" Tempo observed: {', '.join(observed_tempo)}."
        if observed_rhythm:
            result.trait_score.rationale += f" Rhythm observed: {', '.join(observed_rhythm)}."
        return result
