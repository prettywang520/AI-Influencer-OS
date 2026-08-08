"""Camera Analyzer -- scores evidence coverage for camera
movement/equipment feel: distance, angle, lens feeling, handheld/
tripod/tracking, selfie/friend shot. Composition/subject placement is
scored by the separate framing.py analyzer -- see
docs/video_intelligence/dna.md for why these are two analyzers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags, evidence_for_video, observed_vocabulary
from .evidence import VideoIntelligenceError

TRAIT_NAME = "camera"
TAGS = ("camera", "distance", "angle", "lens_feeling", "handheld", "tripod", "tracking", "selfie", "friend_shot")

DEFAULT_CAMERA_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "camera.yaml"


class CameraConfigError(VideoIntelligenceError):
    """Raised when config/video_intelligence/camera.yaml is missing or invalid."""


def _runtime_root() -> Path:
    """
    camera.py location: 10_apps/claude_runtime/src/video_intelligence/camera.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_camera_config_path() -> Path:
    return _runtime_root() / DEFAULT_CAMERA_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class CameraConfig:
    schema_version: str
    distance_types: tuple[str, ...]
    angle_types: tuple[str, ...]
    lens_feeling_types: tuple[str, ...]
    equipment_types: tuple[str, ...]


def load_camera_config(config_path: str | Path | None = None) -> CameraConfig:
    path = Path(config_path) if config_path else default_camera_config_path()
    if not path.exists():
        raise CameraConfigError(f"Camera config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise CameraConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise CameraConfigError(f"Camera config is empty or invalid: {path}")
    section = raw.get("camera") or {}
    return CameraConfig(
        schema_version=str(section.get("schema_version", "1.0")),
        distance_types=tuple(section.get("distance_types", [])),
        angle_types=tuple(section.get("angle_types", [])),
        lens_feeling_types=tuple(section.get("lens_feeling_types", [])),
        equipment_types=tuple(section.get("equipment_types", [])),
    )


class CameraAnalyzer:
    name = TRAIT_NAME

    def __init__(self, camera_config: CameraConfig | None = None) -> None:
        self.camera_config = camera_config or load_camera_config()

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        result = analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Camera pattern observed across cited evidence.",
            rationale_without_evidence="No camera-tagged evidence supplied; confidence is unknown.",
        )
        own_video_evidence = evidence_for_video(context.evidence, context.video_id)
        relevant = evidence_for_tags(own_video_evidence, TAGS)
        observed_distance = observed_vocabulary(relevant, self.camera_config.distance_types)
        observed_angle = observed_vocabulary(relevant, self.camera_config.angle_types)
        observed_lens = observed_vocabulary(relevant, self.camera_config.lens_feeling_types)
        observed_equipment = observed_vocabulary(relevant, self.camera_config.equipment_types)
        if observed_distance:
            result.trait_score.rationale += f" Distance observed: {', '.join(observed_distance)}."
        if observed_angle:
            result.trait_score.rationale += f" Angle observed: {', '.join(observed_angle)}."
        if observed_lens:
            result.trait_score.rationale += f" Lens feeling observed: {', '.join(observed_lens)}."
        if observed_equipment:
            result.trait_score.rationale += f" Equipment observed: {', '.join(observed_equipment)}."
        return result
