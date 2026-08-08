"""Hook Analyzer -- scores evidence coverage for a video's opening
(first frame / first second / first 3 seconds) and surfaces which
recognized hook types were observed. Never asserts a hook "worked" --
only what evidence shows was present.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag, evidence_for_tags, evidence_for_video, observed_vocabulary
from .evidence import VideoIntelligenceError

TRAIT_NAME = "hook"
TAGS = (
    "hook", "first_frame", "first_second", "first_3_seconds",
    "hook_type", "hook_wording", "hook_emotion", "hook_structure", "hook_timing",
)

DEFAULT_HOOK_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "hooks.yaml"


class HookConfigError(VideoIntelligenceError):
    """Raised when config/video_intelligence/hooks.yaml is missing or invalid."""


def _runtime_root() -> Path:
    """
    hook.py location: 10_apps/claude_runtime/src/video_intelligence/hook.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_hook_config_path() -> Path:
    return _runtime_root() / DEFAULT_HOOK_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class HookConfig:
    schema_version: str
    hook_types: tuple[str, ...]


def load_hook_config(config_path: str | Path | None = None) -> HookConfig:
    path = Path(config_path) if config_path else default_hook_config_path()
    if not path.exists():
        raise HookConfigError(f"Hook config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise HookConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise HookConfigError(f"Hook config is empty or invalid: {path}")
    section = raw.get("hook") or {}
    return HookConfig(
        schema_version=str(section.get("schema_version", "1.0")),
        hook_types=tuple(section.get("hook_types", [])),
    )


class HookAnalyzer:
    name = TRAIT_NAME

    def __init__(self, hook_config: HookConfig | None = None) -> None:
        self.hook_config = hook_config or load_hook_config()

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        result = analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Hook pattern observed across cited evidence.",
            rationale_without_evidence="No hook-tagged evidence supplied; confidence is unknown.",
        )
        own_video_evidence = evidence_for_video(context.evidence, context.video_id)
        relevant = evidence_for_tags(own_video_evidence, TAGS)
        observed_types = observed_vocabulary(relevant, self.hook_config.hook_types)
        if observed_types:
            result.trait_score.rationale += f" Hook type(s) observed: {', '.join(observed_types)}."
        return result
