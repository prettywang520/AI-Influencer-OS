"""learn_reel()/learn_reels() -- the batch-learning facade this whole
adapter exists to provide, plus its CLI. Composes mapper.py/sampler.py/
validator.py with the existing, unmodified
video_intelligence.workflow.VideoIntelligenceWorkflow -- no analyzer
orchestration is duplicated here (see docs/video_intelligence/
instagram_reels_adapter.md). Local evidence mode only: this CLI never
opens a browser or touches the network -- live acquisition remains
owned by creator_research/instagram.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType, VideoPlatform, VideoRecord
from src.video_intelligence.workflow import VideoIntelligenceWorkflow

from .exceptions import AdapterCliError, AdapterConfigError
from .mapper import packet_to_video_evidence
from .models import (
    BatchLearningReport,
    InstagramReelEvidencePacket,
    InstagramReelLearningRecord,
    InstagramReelsConfig,
    compute_completeness,
)
from .report import build_batch_report, render_batch_report_markdown
from .sampler import sample_reels
from .validator import validate_packets

PREFIX = "[InstagramReelsAdapter]"

DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "video_intelligence" / "instagram_reels.yaml"

# The task's own yaml spells this priority key "lip_sync"; every real
# trait name elsewhere (TRAIT_FIELDS, VideoDNA, TRAIT_TAGS) spells it
# "lipsync" -- normalized once, here, so the rest of the adapter never
# has to special-case the task's own yaml spelling.
_PRIORITY_KEY_ALIASES = {"lip_sync": "lipsync"}


def _runtime_root() -> Path:
    """
    workflow.py location: 10_apps/claude_runtime/src/video_intelligence/instagram/workflow.py
    parents[3] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[3]


def default_instagram_reels_config_path() -> Path:
    return _runtime_root() / DEFAULT_CONFIG_RELATIVE_PATH


def load_instagram_reels_config(config_path: str | Path | None = None) -> InstagramReelsConfig:
    path = Path(config_path) if config_path else default_instagram_reels_config_path()
    if not path.exists():
        raise AdapterConfigError(f"Instagram Reels adapter config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise AdapterConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise AdapterConfigError(f"Instagram Reels adapter config is empty or invalid: {path}")

    adapter_section = raw.get("adapter") or {}
    sampling_section = raw.get("sampling") or {}
    priorities_section = raw.get("priorities") or {}
    knowledge_section = raw.get("knowledge") or {}

    priorities = {
        _PRIORITY_KEY_ALIASES.get(str(key), str(key)): float(value) for key, value in priorities_section.items()
    }

    return InstagramReelsConfig(
        version=str(raw.get("version", "1.0")),
        schema_version=str(adapter_section.get("schema_version", "1.0")),
        adapter_version=str(adapter_section.get("adapter_version", "12D.1")),
        platform=str(adapter_section.get("platform", VideoPlatform.INSTAGRAM_REELS)),
        minimum_reels_for_dna=int(sampling_section.get("minimum_reels_for_dna", 10)),
        recommended_reels=int(sampling_section.get("recommended_reels", 30)),
        strong_sample=int(sampling_section.get("strong_sample", 50)),
        deterministic=bool(sampling_section.get("deterministic", True)),
        priorities=priorities,
        deidentify_creator=bool(knowledge_section.get("deidentify_creator", True)),
        preserve_verbatim_scripts=bool(knowledge_section.get("preserve_verbatim_scripts", False)),
        minimum_pattern_corroboration=int(knowledge_section.get("minimum_pattern_corroboration", 2)),
    )


def learn_reel(
    packet: InstagramReelEvidencePacket,
    *,
    video_config=None,
    ingest_into_knowledge_base: bool = True,
) -> InstagramReelLearningRecord:
    """Maps one packet to VideoEvidence and runs it through the
    existing, unmodified VideoIntelligenceWorkflow.analyze_video() --
    the entire "build DNA / save / report / knowledge-base ingest"
    step is that one call; nothing is duplicated here."""
    video_evidence = packet_to_video_evidence(packet)
    record = VideoRecord(
        platform=VideoPlatform.INSTAGRAM_REELS, creator_label=packet.creator_label, reference=packet.reel_url,
    )
    warnings: list[str] = []
    if packet.duration_seconds is None:
        warnings.append("duration_seconds unavailable")
    if not any("subtitles" in item.tags for item in packet.annotations):
        warnings.append(
            "text_overlays unavailable via reel_to_evidence()'s Evidence contract "
            "(see docs/video_intelligence/instagram_reels_adapter.md)"
        )

    result = VideoIntelligenceWorkflow(video_config).analyze_video(
        record, video_evidence, ingest_into_knowledge_base=ingest_into_knowledge_base,
    )

    return InstagramReelLearningRecord(
        reel_id=packet.reel_id,
        source_url=packet.reel_url,
        source_evidence_ids=list(packet.source_evidence_ids),
        published_at=packet.published_at,
        video_evidence=video_evidence,
        dna=result.dna,
        completeness=compute_completeness(video_evidence),
        patterns_touched=[pattern.pattern_id for pattern in result.patterns],
        warnings=warnings,
    )


def learn_reels(
    packets: list[InstagramReelEvidencePacket],
    *,
    config: InstagramReelsConfig | None = None,
    video_config=None,
    sample: bool = True,
    ingest_into_knowledge_base: bool = True,
) -> tuple[BatchLearningReport, list[InstagramReelLearningRecord]]:
    """Batch flow: packets -> validation -> (optional) deterministic
    sampling -> learn_reel() per valid packet -> BatchLearningReport."""
    config = config or load_instagram_reels_config()

    valid_packets, invalid_errors = validate_packets(packets)
    if sample:
        valid_packets = sample_reels(valid_packets, config)

    records = [
        learn_reel(packet, video_config=video_config, ingest_into_knowledge_base=ingest_into_knowledge_base)
        for packet in valid_packets
    ]

    # Distinct patterns touched (created or corroborated) across the
    # whole batch -- reuses each learn_reel() call's own
    # AnalysisResult.patterns rather than re-querying the knowledge
    # base (which can't distinguish "touched this batch" from
    # "already existed").
    knowledge_patterns_added = len({pattern_id for record in records for pattern_id in record.patterns_touched})

    report = build_batch_report(records, skipped=invalid_errors, knowledge_patterns_added=knowledge_patterns_added)
    return report, records


# -- CLI ---------------------------------------------------------------


def _packet_from_dict(payload: dict) -> InstagramReelEvidencePacket:
    annotations = []
    for item in payload.get("annotations", []):
        annotations.append(
            VideoEvidence(
                video_id="",  # rewritten by packet_to_video_evidence() at mapping time
                evidence_type=item.get("evidence_type", VideoEvidenceType.OPERATOR_OBSERVATION),
                source_description=item.get("source_description", ""),
                content_excerpt=item.get("content_excerpt", ""),
                timestamp_seconds=item.get("timestamp_seconds"),
                collected_by=item.get("collected_by", "operator"),
                tags=list(item.get("tags", [])),
            )
        )
    return InstagramReelEvidencePacket(
        reel_url=payload["reel_url"],
        published_at=payload.get("published_at"),
        duration_seconds=payload.get("duration_seconds"),
        views=payload.get("views"),
        likes=payload.get("likes"),
        comments=payload.get("comments"),
        caption=payload.get("caption"),
        creator_label=payload.get("creator_label", ""),
        annotations=annotations,
    )


def _load_packets(path: Path) -> list[InstagramReelEvidencePacket]:
    if not path.exists():
        raise AdapterCliError(f"Evidence file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterCliError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise AdapterCliError(f"Evidence file must contain a JSON list: {path}")

    packets = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AdapterCliError(f"Evidence entry {index} is not an object")
        try:
            packets.append(_packet_from_dict(item))
        except KeyError as exc:
            raise AdapterCliError(f"Evidence entry {index} missing required field: {exc}") from exc
    return packets


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
        prog="video_intelligence.instagram.workflow", description="Instagram Reels Learning Adapter CLI",
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
        config = load_instagram_reels_config(args.config)
        packets = _load_packets(args.evidence)

        if args.validate_only:
            _, errors = validate_packets(packets)
            result = {"passed": not errors, "total_reels": len(packets), "errors": errors}
        else:
            report_path = args.output / "instagram_reels_learning_report.json"
            if report_path.exists() and not args.force:
                raise AdapterCliError(f"Output already exists (use --force to overwrite): {report_path}")
            report, _records = learn_reels(packets, config=config)
            payload = {
                "total_reels": report.total_reels,
                "reels_analyzed": report.reels_analyzed,
                "reels_skipped": report.reels_skipped,
                "evidence_completeness": report.evidence_completeness,
                "analyzer_coverage": report.analyzer_coverage,
                "video_dna_ids": report.video_dna_ids,
                "knowledge_patterns_added": report.knowledge_patterns_added,
                "warnings": report.warnings,
                "missing_evidence": report.missing_evidence,
                "generated_at": report.generated_at,
            }
            _atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True))
            _atomic_write_text(args.output / "instagram_reels_learning_report.md", render_batch_report_markdown(report))
            result = payload
    except (AdapterCliError, AdapterConfigError) as exc:
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
