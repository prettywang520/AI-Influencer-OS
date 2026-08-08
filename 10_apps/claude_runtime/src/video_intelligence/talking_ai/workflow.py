"""TalkingAIWorkflow.analyze_talking_video()/analyze_talking_videos() --
the orchestration entry points: evidence -> build_talking_ai_dna() ->
save -> report -> knowledge base ingest. Zero network, zero rendering,
zero live Instagram access -- every input is operator-supplied
TalkingAIEvidence/SpeechSegment already in memory. Also owns
load_talking_ai_config() (config/video_intelligence/talking_ai.yaml)
and talking_evidence_from_reel_record() -- an optional, lossy,
best-effort bridge from an already-analyzed
video_intelligence.instagram.InstagramReelLearningRecord (task §22).
The bridge only *reads* that record's output type; it never modifies
video_intelligence/instagram/ and never opens a browser or network
connection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .evidence import CameraMotionType, FacialExpression, GestureType, SpeechSegment, TalkingAIEvidence
from .exceptions import TalkingAICliError, TalkingAIConfigError
from .knowledge_base import TalkingAIKnowledgeBase, TalkingAIProductionPattern
from .models import ConfidenceLevel, TalkingAIConfig
from .production_dna import TalkingAIProductionDNA, build_talking_ai_dna
from .report import TalkingAIReport, build_talking_ai_report, render_talking_ai_report_markdown
from .serialization import load_talking_ai_dna, save_talking_ai_dna

PREFIX = "[TalkingAIIntelligence]"

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "talking_ai.yaml"


def _runtime_root() -> Path:
    """
    workflow.py location: 10_apps/claude_runtime/src/video_intelligence/talking_ai/workflow.py
    parents[3] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[3]


def default_talking_ai_config_path() -> Path:
    return _runtime_root() / DEFAULT_CONFIG_RELATIVE_PATH


def load_talking_ai_config(config_path: str | Path | None = None) -> TalkingAIConfig:
    path = Path(config_path) if config_path else default_talking_ai_config_path()
    if not path.exists():
        raise TalkingAIConfigError(f"Talking AI config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise TalkingAIConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise TalkingAIConfigError(f"Talking AI config is empty or invalid: {path}")

    talking_ai_section = raw.get("talking_ai") or {}
    timing_section = raw.get("timing") or {}
    sample_size_section = raw.get("sample_size") or {}
    naturalness_section = raw.get("naturalness") or {}
    knowledge_section = raw.get("knowledge") or {}
    output_section = raw.get("output") or {}
    confidence_section = raw.get("confidence") or {}

    return TalkingAIConfig(
        version=str(raw.get("version", "1.0")),
        schema_version=str(talking_ai_section.get("schema_version", "1.0")),
        engine_version=str(talking_ai_section.get("engine_version", "1.0.0")),
        maximum_reasonable_sync_latency_seconds=float(
            timing_section.get("maximum_reasonable_sync_latency_seconds", 0.5)
        ),
        pause_min_seconds=float(timing_section.get("pause_min_seconds", 0.3)),
        minimum_talking_videos=int(sample_size_section.get("minimum_talking_videos", 1)),
        recommended_talking_videos=int(sample_size_section.get("recommended_talking_videos", 5)),
        strong_sample=int(sample_size_section.get("strong_sample", 15)),
        naturalness_dimensions=tuple(naturalness_section.get("dimensions", ())),
        minimum_pattern_corroboration=int(knowledge_section.get("minimum_pattern_corroboration", 2)),
        deidentify_source=bool(knowledge_section.get("deidentify_source", True)),
        preserve_verbatim_transcripts=bool(knowledge_section.get("preserve_verbatim_transcripts", False)),
        output_root=str(output_section.get("root_directory", "output/video_intelligence/talking_ai")),
        confidence_min_evidence_for_medium=int(confidence_section.get("min_evidence_for_medium", 2)),
        confidence_min_evidence_for_high=int(confidence_section.get("min_evidence_for_high", 4)),
        confidence_min_corroboration_for_verified=int(confidence_section.get("min_corroboration_for_verified", 2)),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        Path(tmp_name).replace(path)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


@dataclass(slots=True)
class TalkingAIAnalysisResult:
    video_id: str
    dna: TalkingAIProductionDNA
    report: TalkingAIReport
    report_markdown: str
    patterns: list[TalkingAIProductionPattern] = field(default_factory=list)


class TalkingAIWorkflow:
    def __init__(self, config: TalkingAIConfig | None = None) -> None:
        self.config = config or load_talking_ai_config()

    def output_root(self) -> Path:
        return self.config.resolved_output_root()

    def _video_dir(self, video_id: str) -> Path:
        return self.output_root() / video_id

    def knowledge_base(self) -> TalkingAIKnowledgeBase:
        return TalkingAIKnowledgeBase(self.output_root() / "knowledge_base")

    def analyze_talking_video(
        self,
        video_id: str,
        subject_label: str,
        evidence: list[TalkingAIEvidence],
        speech_segments: list[SpeechSegment],
        *,
        ingest_into_knowledge_base: bool = True,
    ) -> TalkingAIAnalysisResult:
        """Builds a TalkingAIProductionDNA for `video_id` from
        `evidence`/`speech_segments` (items for other videos in the
        same lists are ignored -- scoping is by video_id), saves it,
        renders a report, and -- unless disabled -- folds its
        structured patterns into the de-identified knowledge base."""
        dna = build_talking_ai_dna(video_id, subject_label, evidence, speech_segments, self.config)

        video_dir = self._video_dir(video_id)
        save_talking_ai_dna(dna, video_dir / "talking_ai_dna.json")

        report = build_talking_ai_report(dna)
        markdown = render_talking_ai_report_markdown(report)
        _atomic_write_text(video_dir / "talking_ai_report.md", markdown)
        _atomic_write_text(
            video_dir / "talking_ai_report.json",
            json.dumps(
                {
                    "video_id": report.video_id,
                    "subject_label": report.subject_label,
                    "dna_id": report.dna_id,
                    "overall_confidence": report.overall_confidence,
                    "trait_summary": report.trait_summary,
                    "naturalness_summary": report.naturalness_summary,
                    "artifact_patterns": report.artifact_patterns,
                    "warnings": report.warnings,
                },
                indent=2,
                sort_keys=True,
            ),
        )

        patterns: list[TalkingAIProductionPattern] = []
        if ingest_into_knowledge_base:
            patterns = self.knowledge_base().ingest_talking_ai_dna(
                dna, evidence=evidence, speech_segments=speech_segments,
            )

        return TalkingAIAnalysisResult(
            video_id=video_id, dna=dna, report=report, report_markdown=markdown, patterns=patterns,
        )

    def analyze_talking_videos(
        self,
        videos: list[tuple[str, str, list[TalkingAIEvidence], list[SpeechSegment]]],
        *,
        ingest_into_knowledge_base: bool = True,
    ) -> list[TalkingAIAnalysisResult]:
        """Batch facade: one (video_id, subject_label, evidence,
        speech_segments) tuple per video -> one analyze_talking_video()
        call each."""
        return [
            self.analyze_talking_video(
                video_id, subject_label, evidence, speech_segments,
                ingest_into_knowledge_base=ingest_into_knowledge_base,
            )
            for video_id, subject_label, evidence, speech_segments in videos
        ]

    def load_dna(self, video_id: str) -> TalkingAIProductionDNA | None:
        path = self._video_dir(video_id) / "talking_ai_dna.json"
        if not path.exists():
            return None
        return load_talking_ai_dna(path)

    def load_report_markdown(self, video_id: str) -> str | None:
        path = self._video_dir(video_id) / "talking_ai_report.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")


# -- Optional bridge from the Instagram Reels Learning Adapter (§22) ---
#
# Best-effort and deliberately lossy: video_intelligence.instagram's own
# VideoEvidence carries a tag vocabulary (see e.g. lipsync.TAGS,
# gesture.TAGS, camera.TAGS), not a TalkingAIEvidence-shaped timeline --
# only tags with an unambiguous, non-magnitude meaning are bridged.
# Everything else on the resulting TalkingAIEvidence stays None -- never
# fabricated. Reads video_intelligence.instagram's own record type only;
# does not modify that package.

_TAG_TO_BOOLEAN_FIELD = {
    "blink_timing": "blink",
    "eye_contact": "eye_contact",
    "subtitles": "subtitle_visible",
}
_TAG_TO_CAMERA_MOTION = {
    "handheld": CameraMotionType.HANDHELD,
    "tripod": CameraMotionType.STATIC,
    "tracking": CameraMotionType.TRACKING,
}
_TAG_TO_FACIAL_EXPRESSION = {
    "smile_timing": FacialExpression.SMILE,
}
_TAG_TO_GESTURE_TYPE = {
    "finger_pointing": GestureType.POINT,
}


def talking_evidence_from_reel_record(record, *, collected_by: str = "operator") -> list[TalkingAIEvidence]:
    """Converts an already-analyzed InstagramReelLearningRecord's
    VideoEvidence into sparse TalkingAIEvidence, timestamp by
    timestamp, populating only fields a source tag can honestly imply.
    A VideoEvidence item with no timestamp, or whose tags map to
    nothing this bridge understands, contributes no TalkingAIEvidence
    at all -- never an all-unknown placeholder observation."""
    video_id = record.dna.video_id if record.dna is not None else record.reel_id
    bridged: list[TalkingAIEvidence] = []
    for item in record.video_evidence:
        if item.timestamp_seconds is None:
            continue
        kwargs: dict = {}
        matched_tags: list[str] = []
        for tag in item.tags:
            if tag in _TAG_TO_BOOLEAN_FIELD:
                kwargs[_TAG_TO_BOOLEAN_FIELD[tag]] = True
                matched_tags.append(tag)
            if tag in _TAG_TO_CAMERA_MOTION:
                kwargs["camera_motion"] = _TAG_TO_CAMERA_MOTION[tag]
                matched_tags.append(tag)
            if tag in _TAG_TO_FACIAL_EXPRESSION:
                kwargs["facial_expression"] = _TAG_TO_FACIAL_EXPRESSION[tag]
                matched_tags.append(tag)
            if tag in _TAG_TO_GESTURE_TYPE:
                kwargs["gesture_type"] = _TAG_TO_GESTURE_TYPE[tag]
                matched_tags.append(tag)
        if not kwargs:
            continue
        bridged.append(
            TalkingAIEvidence(
                video_id=video_id,
                timestamp_seconds=item.timestamp_seconds,
                collected_by=collected_by,
                confidence=ConfidenceLevel.LOW,
                notes=f"bridged from Instagram Reels adapter evidence tags: {', '.join(sorted(matched_tags))}",
                **kwargs,
            )
        )
    return bridged


# -- CLI -----------------------------------------------------------------


def _evidence_from_dict(payload: dict, video_id: str) -> TalkingAIEvidence:
    kwargs = {key: value for key, value in payload.items() if key not in {"video_id", "evidence_id"}}
    return TalkingAIEvidence(video_id=video_id, **kwargs)


def _segment_from_dict(payload: dict, video_id: str) -> SpeechSegment:
    kwargs = {key: value for key, value in payload.items() if key not in {"video_id", "segment_id"}}
    return SpeechSegment(video_id=video_id, **kwargs)


def _video_input_from_dict(payload: dict) -> tuple[str, str, list[TalkingAIEvidence], list[SpeechSegment]]:
    video_id = payload["video_id"]
    subject_label = payload.get("subject_label", video_id)
    evidence = [_evidence_from_dict(item, video_id) for item in payload.get("evidence", [])]
    speech_segments = [_segment_from_dict(item, video_id) for item in payload.get("speech_segments", [])]
    return video_id, subject_label, evidence, speech_segments


def _load_video_inputs(path: Path) -> list[tuple[str, str, list[TalkingAIEvidence], list[SpeechSegment]]]:
    if not path.exists():
        raise TalkingAICliError(f"Evidence file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TalkingAICliError(f"Invalid JSON in {path}: {exc}") from exc

    payloads = raw if isinstance(raw, list) else [raw]
    inputs = []
    for index, item in enumerate(payloads):
        if not isinstance(item, dict):
            raise TalkingAICliError(f"Evidence entry {index} is not an object")
        try:
            inputs.append(_video_input_from_dict(item))
        except KeyError as exc:
            raise TalkingAICliError(f"Evidence entry {index} missing required field: {exc}") from exc
    return inputs


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="video_intelligence.talking_ai.workflow", description="Talking AI Intelligence CLI",
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        config = load_talking_ai_config(args.config)
        video_inputs = _load_video_inputs(args.evidence)

        summary_path = args.output / "talking_ai_batch_summary.json"
        if summary_path.exists() and not args.force:
            raise TalkingAICliError(f"Output already exists (use --force to overwrite): {summary_path}")

        workflow = TalkingAIWorkflow(config)
        results = workflow.analyze_talking_videos(video_inputs)

        payload = {
            "total_videos": len(results),
            "video_dna_ids": {result.video_id: result.dna.dna_id for result in results},
            "overall_confidence": {result.video_id: result.dna.overall_confidence for result in results},
            "knowledge_patterns_touched": len({
                pattern.pattern_id for result in results for pattern in result.patterns
            }),
        }
        _atomic_write_text(summary_path, json.dumps(payload, indent=2, sort_keys=True))
        result = payload
    except (TalkingAICliError, TalkingAIConfigError) as exc:
        print(f"{PREFIX} Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print(f"{PREFIX} result:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
