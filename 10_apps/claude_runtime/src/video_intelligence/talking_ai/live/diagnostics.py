"""Local-run diagnostics for a batch of LiveTalkingReelRecords --
batch-processing statistics only (records attempted/valid/sampled/
processed, timing-bridge placeholder usage, average completeness per
domain). No live Instagram session exists in this phase to record
diagnostics *about* -- this module has no field for a password,
cookie, auth header, session token, access-limit event, or selector
failure (unlike creator_research.instagram.diagnostics.DiagnosticsRecorder,
which exists precisely to record those for a real browser session);
it is scoped only to this package's own offline batch-learning runs,
giving a future live-acquisition phase a consistent place to extend
without needing new code here.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .adapter import LiveAdaptedEvidence
from .models import COMPLETENESS_DOMAINS, LiveTalkingReelRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_live_run_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass(slots=True)
class LiveLearningDiagnostics:
    run_id: str
    records_attempted: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    records_sampled: int = 0
    records_processed: int = 0
    ordinal_timing_item_count: int = 0
    approximate_timing_item_count: int = 0
    completeness_summary: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None

    def finish(self) -> None:
        self.finished_at = _now_iso()

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "records_attempted": self.records_attempted,
            "records_valid": self.records_valid,
            "records_invalid": self.records_invalid,
            "records_sampled": self.records_sampled,
            "records_processed": self.records_processed,
            "ordinal_timing_item_count": self.ordinal_timing_item_count,
            "approximate_timing_item_count": self.approximate_timing_item_count,
            "completeness_summary": self.completeness_summary,
            "warnings": self.warnings,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def build_live_learning_diagnostics(
    *,
    run_id: str | None = None,
    total_records: int,
    valid_records: list[LiveTalkingReelRecord],
    invalid_count: int,
    sampled_records: list[LiveTalkingReelRecord],
    adapted_results: list[LiveAdaptedEvidence],
) -> LiveLearningDiagnostics:
    """Pure aggregation over an already-completed batch run -- this
    package's runs are local/offline and complete in one pass, so
    diagnostics are built once at the end rather than accumulated
    incrementally across an async session (contrast
    creator_research.instagram.diagnostics.DiagnosticsRecorder, which
    accumulates during a real, long-running browser session)."""
    diagnostics = LiveLearningDiagnostics(
        run_id=run_id or new_live_run_id(),
        records_attempted=total_records,
        records_valid=len(valid_records),
        records_invalid=invalid_count,
        records_sampled=len(sampled_records),
        records_processed=len(adapted_results),
    )

    diagnostics.ordinal_timing_item_count = sum(result.ordinal_timing_count for result in adapted_results)
    diagnostics.approximate_timing_item_count = sum(result.approximate_timing_count for result in adapted_results)
    for result in adapted_results:
        diagnostics.warnings.extend(result.warnings)

    if sampled_records:
        diagnostics.completeness_summary = {
            domain: mean(record.completeness.get(domain, 0.0) for record in sampled_records)
            for domain in COMPLETENESS_DOMAINS
        }

    diagnostics.finish()
    return diagnostics


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


def save_live_learning_diagnostics(diagnostics: LiveLearningDiagnostics, path: str | Path) -> Path:
    path = Path(path)
    _atomic_write_text(path, json.dumps(diagnostics.to_dict(), indent=2, sort_keys=True))
    return path
