"""Speech Analyzer -- scores evidence coverage for spoken delivery
(language, speed, pause timing, sentence rhythm, energy, warmth,
delivery confidence, question style). Never transcribes audio itself
-- language/wording are always operator-observed, exactly like every
other VideoEvidence item in this package (no Whisper, no external
API, ever).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags, evidence_for_video, observed_vocabulary
from .evidence import VideoIntelligenceError

TRAIT_NAME = "speech"
TAGS = (
    "speech", "language", "speech_speed", "pause_timing", "sentence_rhythm",
    "energy", "warmth", "delivery_confidence", "question_style",
)

DEFAULT_SPEECH_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "speech.yaml"


class SpeechConfigError(VideoIntelligenceError):
    """Raised when config/video_intelligence/speech.yaml is missing or invalid."""


def _runtime_root() -> Path:
    """
    speech.py location: 10_apps/claude_runtime/src/video_intelligence/speech.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_speech_config_path() -> Path:
    return _runtime_root() / DEFAULT_SPEECH_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class SpeechConfig:
    schema_version: str
    languages: tuple[str, ...]
    speed_types: tuple[str, ...]
    energy_types: tuple[str, ...]
    question_style_types: tuple[str, ...]


def load_speech_config(config_path: str | Path | None = None) -> SpeechConfig:
    path = Path(config_path) if config_path else default_speech_config_path()
    if not path.exists():
        raise SpeechConfigError(f"Speech config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SpeechConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise SpeechConfigError(f"Speech config is empty or invalid: {path}")
    section = raw.get("speech") or {}
    return SpeechConfig(
        schema_version=str(section.get("schema_version", "1.0")),
        languages=tuple(section.get("languages", [])),
        speed_types=tuple(section.get("speed_types", [])),
        energy_types=tuple(section.get("energy_types", [])),
        question_style_types=tuple(section.get("question_style_types", [])),
    )


class SpeechAnalyzer:
    name = TRAIT_NAME

    def __init__(self, speech_config: SpeechConfig | None = None) -> None:
        self.speech_config = speech_config or load_speech_config()

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        result = analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Speech delivery pattern observed across cited evidence.",
            rationale_without_evidence="No speech-tagged evidence supplied; confidence is unknown.",
        )
        own_video_evidence = evidence_for_video(context.evidence, context.video_id)
        relevant = evidence_for_tags(own_video_evidence, TAGS)
        observed_languages = observed_vocabulary(relevant, self.speech_config.languages)
        observed_speed = observed_vocabulary(relevant, self.speech_config.speed_types)
        observed_energy = observed_vocabulary(relevant, self.speech_config.energy_types)
        observed_question_style = observed_vocabulary(relevant, self.speech_config.question_style_types)
        if observed_languages:
            result.trait_score.rationale += f" Language(s) observed: {', '.join(observed_languages)}."
        if observed_speed:
            result.trait_score.rationale += f" Speed observed: {', '.join(observed_speed)}."
        if observed_energy:
            result.trait_score.rationale += f" Energy observed: {', '.join(observed_energy)}."
        if observed_question_style:
            result.trait_score.rationale += f" Question style observed: {', '.join(observed_question_style)}."
        return result
