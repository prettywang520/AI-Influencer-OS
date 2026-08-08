"""VideoIntelligenceWorkflow.analyze_video() -- the one orchestration
entry point: evidence -> build_video_dna() -> save -> report ->
knowledge base ingest. Zero network, zero rendering -- every input is
operator-supplied VideoEvidence already in memory.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .evidence import VideoEvidence, VideoRecord
from .knowledge_base import ProductionPattern, VideoKnowledgeBase
from .models import VideoIntelligenceConfig
from .production_dna import VideoDNA, build_video_dna, load_engine_config
from .report import VideoReport, build_video_report, render_video_report_markdown
from .serialization import load_video_dna, save_video_dna


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
class AnalysisResult:
    video_id: str
    dna: VideoDNA
    report: VideoReport
    report_markdown: str
    patterns: list[ProductionPattern] = field(default_factory=list)


class VideoIntelligenceWorkflow:
    def __init__(self, config: VideoIntelligenceConfig | None = None) -> None:
        self.config = config or load_engine_config()

    def output_root(self) -> Path:
        return self.config.resolved_output_root()

    def _video_dir(self, video_id: str) -> Path:
        return self.output_root() / video_id

    def knowledge_base(self) -> VideoKnowledgeBase:
        return VideoKnowledgeBase(self.output_root() / "knowledge_base")

    def analyze_video(
        self,
        record: VideoRecord,
        evidence: list[VideoEvidence],
        *,
        ingest_into_knowledge_base: bool = True,
    ) -> AnalysisResult:
        """Builds a VideoDNA for `record` from `evidence` (evidence for
        other videos in the same list is ignored -- scoping is by
        video_id), saves it, renders a report, and -- unless disabled
        -- folds its structured patterns into the de-identified
        knowledge base."""
        dna = build_video_dna(record.video_id, record.creator_label or record.video_id, evidence, self.config)

        video_dir = self._video_dir(record.video_id)
        save_video_dna(dna, video_dir / "video_dna.json")

        report = build_video_report(dna)
        markdown = render_video_report_markdown(report)
        _atomic_write_text(video_dir / "report.md", markdown)

        patterns: list[ProductionPattern] = []
        if ingest_into_knowledge_base:
            patterns = self.knowledge_base().ingest_video_dna(dna, evidence)

        return AnalysisResult(
            video_id=record.video_id, dna=dna, report=report, report_markdown=markdown, patterns=patterns,
        )

    def load_dna(self, video_id: str) -> VideoDNA | None:
        path = self._video_dir(video_id) / "video_dna.json"
        if not path.exists():
            return None
        return load_video_dna(path)

    def load_report_markdown(self, video_id: str) -> str | None:
        path = self._video_dir(video_id) / "report.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
