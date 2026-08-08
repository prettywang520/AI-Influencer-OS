"""learn_live_talking_reel()/learn_live_talking_reels() -- the batch-
learning facade this whole adapter exists to provide (task §18), plus
its CLI (task §22). Composes adapter.py/sampler.py/validator.py/
diagnostics.py/report.py with the existing, unmodified
talking_ai.workflow.TalkingAIWorkflow -- no analyzer orchestration is
duplicated here. Local/offline evidence mode only: this CLI never
opens a browser or touches the network -- live acquisition remains
owned by creator_research/instagram, and any *future* live run is a
separate, explicitly-authorized action (see
docs/video_intelligence/talking_ai_live_runbook.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

from ..evidence import SpeechSegment, TalkingAIEvidence
from ..workflow import TalkingAIWorkflow
from .adapter import LiveAdaptedEvidence, live_talking_evidence_from_record
from .diagnostics import LiveLearningDiagnostics, build_live_learning_diagnostics, save_live_learning_diagnostics
from .exceptions import LiveAdapterCliError, LiveAdapterConfigError
from .models import LiveAdapterConfig, LiveBatchLearningReport, LiveTalkingLearningRecord, LiveTalkingReelRecord
from .report import build_live_learning_report, render_live_learning_report_markdown
from .sampler import sample_live_talking_reels
from .validator import validate_records

PREFIX = "[TalkingAILiveAdapter]"

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "talking_ai_live.yaml"


def _runtime_root() -> Path:
    """
    workflow.py location: 10_apps/claude_runtime/src/video_intelligence/talking_ai/live/workflow.py
    parents[4] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[4]


def default_talking_ai_live_config_path() -> Path:
    return _runtime_root() / DEFAULT_CONFIG_RELATIVE_PATH


def load_talking_ai_live_config(config_path: str | Path | None = None) -> LiveAdapterConfig:
    path = Path(config_path) if config_path else default_talking_ai_live_config_path()
    if not path.exists():
        raise LiveAdapterConfigError(f"Talking AI live adapter config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise LiveAdapterConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise LiveAdapterConfigError(f"Talking AI live adapter config is empty or invalid: {path}")

    adapter_section = raw.get("adapter") or {}
    sampling_section = raw.get("sampling") or {}
    evidence_section = raw.get("evidence") or {}
    knowledge_section = raw.get("knowledge") or {}
    live_section = raw.get("live") or {}
    output_section = raw.get("output") or {}

    return LiveAdapterConfig(
        version=str(raw.get("version", "1.0")),
        schema_version=str(adapter_section.get("schema_version", "1.0")),
        adapter_version=str(adapter_section.get("adapter_version", "12D.3")),
        source_platform=str(adapter_section.get("source_platform", "instagram_reels")),
        minimum_talking_reels=int(sampling_section.get("minimum_talking_reels", 5)),
        recommended_talking_reels=int(sampling_section.get("recommended_talking_reels", 15)),
        strong_sample=int(sampling_section.get("strong_sample", 30)),
        deterministic=bool(sampling_section.get("deterministic", True)),
        allow_sparse_timeline=bool(evidence_section.get("allow_sparse_timeline", True)),
        require_exact_timestamps=bool(evidence_section.get("require_exact_timestamps", False)),
        preserve_source_references=bool(evidence_section.get("preserve_source_references", True)),
        deidentify_source=bool(knowledge_section.get("deidentify_source", True)),
        preserve_verbatim_transcripts=bool(knowledge_section.get("preserve_verbatim_transcripts", False)),
        preserve_creator_username=bool(knowledge_section.get("preserve_creator_username", False)),
        live_enabled_by_default=bool(live_section.get("enabled_by_default", False)),
        live_require_explicit_authorization=bool(live_section.get("require_explicit_authorization", True)),
        output_root=str(output_section.get("root_directory", "output/video_intelligence/talking_ai/live")),
    )


def _learn_from_adapted(
    record: LiveTalkingReelRecord,
    adapted: LiveAdaptedEvidence,
    *,
    talking_ai_config=None,
    ingest_into_knowledge_base: bool = True,
) -> LiveTalkingLearningRecord:
    subject_label = record.metadata.get("creator_label", "") or record.reel_id
    result = TalkingAIWorkflow(talking_ai_config).analyze_talking_video(
        record.reel_id, subject_label, adapted.evidence, adapted.speech_segments,
        ingest_into_knowledge_base=ingest_into_knowledge_base,
    )
    return LiveTalkingLearningRecord(
        reel_id=record.reel_id,
        record_id=record.record_id,
        source_url=record.source_url,
        dna=result.dna,
        completeness=record.completeness,
        patterns_touched=[pattern.pattern_id for pattern in result.patterns],
        warnings=list(record.warnings) + adapted.warnings,
    )


def learn_live_talking_reel(
    record: LiveTalkingReelRecord,
    *,
    talking_ai_config=None,
    ingest_into_knowledge_base: bool = True,
) -> LiveTalkingLearningRecord:
    """Maps one record to TalkingAIEvidence/SpeechSegment and runs it
    through the existing, unmodified
    TalkingAIWorkflow.analyze_talking_video() -- the entire "build DNA /
    save / report / knowledge-base ingest" step is that one call;
    nothing is duplicated here."""
    adapted = live_talking_evidence_from_record(record)
    return _learn_from_adapted(
        record, adapted, talking_ai_config=talking_ai_config, ingest_into_knowledge_base=ingest_into_knowledge_base,
    )


def learn_live_talking_reels(
    records: list[LiveTalkingReelRecord],
    *,
    config: LiveAdapterConfig | None = None,
    talking_ai_config=None,
    sample: bool = True,
    ingest_into_knowledge_base: bool = True,
) -> tuple[LiveBatchLearningReport, list[LiveTalkingLearningRecord], LiveLearningDiagnostics]:
    """Batch flow: records -> validation -> (optional) deterministic
    sampling -> learn_live_talking_reel() per sampled record ->
    LiveBatchLearningReport + LiveLearningDiagnostics. Pure -- callers
    decide what (if anything) to persist (see main())."""
    config = config or load_talking_ai_live_config()

    valid_records, validation_errors = validate_records(records)
    sampled_records = sample_live_talking_reels(valid_records, config) if sample else valid_records

    adapted_by_reel_id = {record.reel_id: live_talking_evidence_from_record(record) for record in sampled_records}
    outcomes = [
        _learn_from_adapted(
            record, adapted_by_reel_id[record.reel_id],
            talking_ai_config=talking_ai_config, ingest_into_knowledge_base=ingest_into_knowledge_base,
        )
        for record in sampled_records
    ]

    report = build_live_learning_report(
        outcomes, total_reels_discovered=len(records), reels_sampled=len(sampled_records),
    )
    for reel_id, errors in sorted(validation_errors.items()):
        report.warnings.append(f"skipped reel_id={reel_id!r}: {errors}")

    diagnostics = build_live_learning_diagnostics(
        total_records=len(records), valid_records=valid_records, invalid_count=len(validation_errors),
        sampled_records=sampled_records, adapted_results=list(adapted_by_reel_id.values()),
    )

    return report, outcomes, diagnostics


# -- CLI -----------------------------------------------------------------


def _evidence_from_dict(payload: dict) -> TalkingAIEvidence:
    kwargs = {key: value for key, value in payload.items() if key not in {"video_id", "evidence_id"}}
    return TalkingAIEvidence(video_id="", **kwargs)


def _segment_from_dict(payload: dict) -> SpeechSegment:
    kwargs = {key: value for key, value in payload.items() if key not in {"video_id", "segment_id"}}
    return SpeechSegment(video_id="", **kwargs)


def _record_from_dict(payload: dict) -> LiveTalkingReelRecord:
    kwargs = {
        "source_url": payload["source_url"],
        "creator_label": payload.get("creator_label", ""),
        "source_evidence_ids": list(payload.get("source_evidence_ids", [])),
        "published_at": payload.get("published_at"),
        "duration_seconds": payload.get("duration_seconds"),
        "language_observations": list(payload.get("language_observations", [])),
        "timeline_observations": [_evidence_from_dict(item) for item in payload.get("timeline_observations", [])],
        "speech_segments": [_segment_from_dict(item) for item in payload.get("speech_segments", [])],
        "subtitle_observations": list(payload.get("subtitle_observations", [])),
        "camera_observations": list(payload.get("camera_observations", [])),
        "artifact_observations": list(payload.get("artifact_observations", [])),
        "metadata": dict(payload.get("metadata", {})),
    }
    return LiveTalkingReelRecord(**kwargs)


def _load_records(path: Path) -> list[LiveTalkingReelRecord]:
    if not path.exists():
        raise LiveAdapterCliError(f"Evidence file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LiveAdapterCliError(f"Invalid JSON in {path}: {exc}") from exc

    payloads = raw if isinstance(raw, list) else [raw]
    records = []
    for index, item in enumerate(payloads):
        if not isinstance(item, dict):
            raise LiveAdapterCliError(f"Evidence entry {index} is not an object")
        try:
            records.append(_record_from_dict(item))
        except KeyError as exc:
            raise LiveAdapterCliError(f"Evidence entry {index} missing required field: {exc}") from exc
    return records


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


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="video_intelligence.talking_ai.live.workflow", description="Talking AI Live Learning Adapter CLI",
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        config = load_talking_ai_live_config(args.config)
        records = _load_records(args.evidence)

        if args.validate_only:
            _, errors = validate_records(records)
            result = {"passed": not errors, "total_records": len(records), "errors": errors}
        else:
            report_path = args.output / "talking_ai_live_learning_report.json"
            if report_path.exists() and not args.force:
                raise LiveAdapterCliError(f"Output already exists (use --force to overwrite): {report_path}")

            report, _outcomes, diagnostics = learn_live_talking_reels(records, config=config)
            payload = {
                "total_reels_discovered": report.total_reels_discovered,
                "reels_sampled": report.reels_sampled,
                "reels_analyzed": report.reels_analyzed,
                "evidence_completeness": report.evidence_completeness,
                "production_dna_ids": report.production_dna_ids,
                "knowledge_patterns_touched": report.knowledge_patterns_touched,
                "insufficient_evidence_reel_ids": report.insufficient_evidence_reel_ids,
                "warnings": report.warnings,
                "generated_at": report.generated_at,
            }
            _atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True))
            _atomic_write_text(
                args.output / "talking_ai_live_learning_report.md", render_live_learning_report_markdown(report),
            )
            save_live_learning_diagnostics(diagnostics, args.output / f"diagnostics_{diagnostics.run_id}.json")
            result = payload
    except (LiveAdapterCliError, LiveAdapterConfigError) as exc:
        print(f"{PREFIX} Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print(f"{PREFIX} result:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    if isinstance(result, dict) and result.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
