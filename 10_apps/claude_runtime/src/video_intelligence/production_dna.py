"""VideoDNA -- the top-level aggregate record (the task's OUTPUT
section calls this "Video DNA" / "Production DNA" -- same record, not
separate classes; see docs/video_intelligence/dna.md for the full
facet mapping). build_video_dna() runs all 13 analyzers over one
video's evidence and aggregates their outputs, mirroring
creator_intelligence.summary_builder.build_creator_dna() exactly.
load_engine_config() reads config/video_intelligence/engine.yaml.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .analyzer import AnalyzerContext
from .audio import AudioAnalyzer
from .camera import CameraAnalyzer
from .cta import CTAAnalyzer
from .editing import EditingAnalyzer
from .emotion import EmotionAnalyzer
from .evidence import VideoEvidence, VideoIntelligenceError
from .framing import FramingAnalyzer
from .gesture import GestureAnalyzer
from .hook import HookAnalyzer
from .lipsync import LipsyncAnalyzer
from .models import ConfidenceLevel, CTAObservation, StoryBeat, TraitScore, VideoIntelligenceConfig, min_confidence
from .pacing import PacingAnalyzer
from .speech import SpeechAnalyzer
from .storytelling import StorytellingAnalyzer
from .subtitles import SubtitlesAnalyzer

DEFAULT_ENGINE_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "engine.yaml"

TRAIT_FIELDS = (
    "hook", "pacing", "speech", "lipsync", "gesture", "camera", "framing",
    "editing", "subtitles", "audio", "storytelling", "cta", "emotion",
)


class VideoIntelligenceConfigError(VideoIntelligenceError):
    """Raised when config/video_intelligence/engine.yaml is missing or invalid."""


def _runtime_root() -> Path:
    """
    production_dna.py location: 10_apps/claude_runtime/src/video_intelligence/production_dna.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_engine_config_path() -> Path:
    return _runtime_root() / DEFAULT_ENGINE_CONFIG_RELATIVE_PATH


def load_engine_config(config_path: str | Path | None = None) -> VideoIntelligenceConfig:
    path = Path(config_path) if config_path else default_engine_config_path()
    if not path.exists():
        raise VideoIntelligenceConfigError(f"Engine config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise VideoIntelligenceConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise VideoIntelligenceConfigError(f"Engine config is empty or invalid: {path}")

    engine_section = raw.get("video_intelligence") or {}
    confidence_section = raw.get("confidence") or {}
    evidence_section = raw.get("evidence") or {}
    output_section = raw.get("output") or {}
    weights_section = raw.get("analyzer_weights") or {}

    return VideoIntelligenceConfig(
        schema_version=str(engine_section.get("schema_version", "1.0")),
        confidence_min_evidence_for_medium=int(confidence_section.get("min_evidence_for_medium", 2)),
        confidence_min_evidence_for_high=int(confidence_section.get("min_evidence_for_high", 4)),
        confidence_min_corroboration_for_verified=int(confidence_section.get("min_corroboration_for_verified", 2)),
        excerpt_max_chars=int(evidence_section.get("excerpt_max_chars", 280)),
        output_root=str(output_section.get("root_directory", "output/video_intelligence")),
        analyzer_weights={str(key): float(value) for key, value in weights_section.items()},
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(payload: dict[str, Any], *, length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def _trait_score_to_payload(trait: TraitScore | None) -> dict | None:
    if trait is None:
        return None
    return {
        "trait_name": trait.trait_name,
        "score": trait.score,
        "confidence": trait.confidence,
        "evidence_ids": sorted(trait.evidence_ids),
        "rationale": trait.rationale,
    }


def _compute_dna_id(dna: VideoDNA) -> str:
    """Content-hash over every field except `generated_at` and the id
    itself, so re-serializing an unchanged record at a different time
    never mints a new id (mirrors
    creator_intelligence.models._compute_dna_id() exactly)."""
    payload: dict = {
        "video_id": dna.video_id,
        "subject_label": dna.subject_label,
        "schema_version": dna.schema_version,
        "evidence_index": sorted(dna.evidence_index),
        "story_beats": sorted(
            (
                {"beat_type": beat.beat_type, "description": beat.description, "evidence_ids": sorted(beat.evidence_ids)}
                for beat in dna.story_beats
            ),
            key=lambda beat: (beat["beat_type"], beat["description"]),
        ),
        "cta_observations": sorted(
            (
                {
                    "cta_type": obs.cta_type,
                    "timing_seconds": obs.timing_seconds,
                    "evidence_ids": sorted(obs.evidence_ids),
                }
                for obs in dna.cta_observations
            ),
            key=lambda obs: (obs["cta_type"], obs["timing_seconds"] or 0.0),
        ),
        "overall_confidence": dna.overall_confidence,
        "warnings": sorted(dna.warnings),
    }
    for name in TRAIT_FIELDS:
        payload[name] = _trait_score_to_payload(getattr(dna, name))
    return _content_hash(payload)


@dataclass(slots=True)
class VideoDNA:
    """Top-level aggregated result for one observed video -- the
    task's "Video DNA" / "Production DNA". `subject_label` is an
    operator-chosen free text label, same rule
    creator_intelligence.CreatorDNA already enforces."""

    video_id: str
    subject_label: str
    schema_version: str = "1.0"
    generated_at: str = ""
    evidence_index: list[str] = field(default_factory=list)

    hook: TraitScore | None = None
    pacing: TraitScore | None = None
    speech: TraitScore | None = None
    lipsync: TraitScore | None = None
    gesture: TraitScore | None = None
    camera: TraitScore | None = None
    framing: TraitScore | None = None
    editing: TraitScore | None = None
    subtitles: TraitScore | None = None
    audio: TraitScore | None = None
    storytelling: TraitScore | None = None
    cta: TraitScore | None = None
    emotion: TraitScore | None = None

    story_beats: list[StoryBeat] = field(default_factory=list)
    cta_observations: list[CTAObservation] = field(default_factory=list)

    overall_confidence: str = ConfidenceLevel.UNKNOWN
    warnings: list[str] = field(default_factory=list)

    dna_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.dna_id = _compute_dna_id(self)


def build_video_dna(
    video_id: str,
    subject_label: str,
    evidence: list[VideoEvidence],
    config: VideoIntelligenceConfig,
) -> VideoDNA:
    """Builds a VideoDNA record from operator-/test-supplied evidence
    only, scoped to `video_id` (evidence for other videos in the same
    list is ignored -- see analyzer.evidence_for_video())."""
    context = AnalyzerContext(video_id=video_id, evidence=evidence, config=config)

    hook_result = HookAnalyzer().analyze(context)
    pacing_result = PacingAnalyzer().analyze(context)
    speech_result = SpeechAnalyzer().analyze(context)
    lipsync_result = LipsyncAnalyzer().analyze(context)
    gesture_result = GestureAnalyzer().analyze(context)
    camera_result = CameraAnalyzer().analyze(context)
    framing_result = FramingAnalyzer().analyze(context)
    editing_result = EditingAnalyzer().analyze(context)
    subtitles_result = SubtitlesAnalyzer().analyze(context)
    audio_result = AudioAnalyzer().analyze(context)
    storytelling = StorytellingAnalyzer()
    storytelling_result = storytelling.analyze(context)
    story_beats = storytelling.extract_story_beats(context)
    cta_analyzer = CTAAnalyzer()
    cta_result = cta_analyzer.analyze(context)
    cta_observations = cta_analyzer.extract_cta_observations(context)
    emotion_result = EmotionAnalyzer().analyze(context)

    all_results = [
        hook_result, pacing_result, speech_result, lipsync_result, gesture_result,
        camera_result, framing_result, editing_result, subtitles_result, audio_result,
        storytelling_result, cta_result, emotion_result,
    ]

    warnings: list[str] = []
    evidence_ids: set[str] = set()
    for result in all_results:
        warnings.extend(result.warnings)
        evidence_ids.update(result.trait_score.evidence_ids)
    for beat in story_beats:
        evidence_ids.update(beat.evidence_ids)
    for observation in cta_observations:
        evidence_ids.update(observation.evidence_ids)

    overall_confidence = min_confidence([result.trait_score.confidence for result in all_results])

    return VideoDNA(
        video_id=video_id,
        subject_label=subject_label,
        schema_version=config.schema_version,
        generated_at=_now_iso(),
        evidence_index=sorted(evidence_ids),
        hook=hook_result.trait_score,
        pacing=pacing_result.trait_score,
        speech=speech_result.trait_score,
        lipsync=lipsync_result.trait_score,
        gesture=gesture_result.trait_score,
        camera=camera_result.trait_score,
        framing=framing_result.trait_score,
        editing=editing_result.trait_score,
        subtitles=subtitles_result.trait_score,
        audio=audio_result.trait_score,
        storytelling=storytelling_result.trait_score,
        cta=cta_result.trait_score,
        emotion=emotion_result.trait_score,
        story_beats=story_beats,
        cta_observations=cta_observations,
        overall_confidence=overall_confidence,
        warnings=warnings,
    )
