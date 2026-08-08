"""CreatorKnowledgeBase -- durable, per-creator, on-disk evidence
store plus learning-session history/state/report files. Lives under
<knowledge_base_root>/<creator_id>/ (gitignored, matching every other
real-creator data directory this session has established). All writes
atomic (temp-file + Path.replace()); the evidence merge is a pure
dedup-by-evidence_id union that never drops or mutates an existing
entry, matching Evidence's own immutable content-hashed identity.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.creator_intelligence.evidence import Evidence
from src.creator_intelligence.models import CreatorDNA
from src.creator_intelligence.serialization import creator_dna_from_dict, creator_dna_to_dict

from .exceptions import KnowledgeBaseError


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


def evidence_to_dict(evidence: Evidence) -> dict:
    return {
        "evidence_id": evidence.evidence_id,
        "evidence_type": evidence.evidence_type,
        "source_description": evidence.source_description,
        "content_excerpt": evidence.content_excerpt,
        "collected_at": evidence.collected_at,
        "collected_by": evidence.collected_by,
        "tags": list(evidence.tags),
    }


def evidence_from_dict(payload: dict) -> Evidence:
    return Evidence(
        evidence_type=payload["evidence_type"],
        source_description=payload["source_description"],
        content_excerpt=payload.get("content_excerpt", ""),
        collected_at=payload.get("collected_at", ""),
        collected_by=payload.get("collected_by", "operator"),
        tags=list(payload.get("tags", [])),
    )


def _load_json(path: Path, *, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Invalid JSON in {path}: {exc}") from exc


class CreatorKnowledgeBase:
    """One creator's durable evidence store and session history,
    rooted at <root_directory>/<creator_id>/."""

    def __init__(self, root_directory: str | Path, creator_id: str) -> None:
        self.creator_id = creator_id
        self.root = Path(root_directory) / creator_id

    @property
    def evidence_store_path(self) -> Path:
        return self.root / "evidence_store.json"

    @property
    def learning_history_path(self) -> Path:
        return self.root / "learning_history.json"

    @property
    def learning_state_path(self) -> Path:
        return self.root / "learning_state.json"

    def dna_snapshot_path(self, session_id: str) -> Path:
        return self.root / "dna_snapshots" / f"{session_id}.json"

    def reports_directory(self, session_id: str) -> Path:
        return self.root / "reports" / session_id

    # -- evidence store --------------------------------------------------

    def load_evidence(self) -> list[Evidence]:
        raw = _load_json(self.evidence_store_path, default=[])
        return [evidence_from_dict(item) for item in raw]

    def merge_evidence(self, new_items: list[Evidence]) -> list[Evidence]:
        """Dedup-by-evidence_id union of the existing store with
        new_items. Never drops or mutates an existing entry. Returns
        the full merged store, not just the delta."""
        by_id = {item.evidence_id: item for item in self.load_evidence()}
        for item in new_items:
            by_id.setdefault(item.evidence_id, item)
        merged = sorted(by_id.values(), key=lambda item: item.evidence_id)
        payload = [evidence_to_dict(item) for item in merged]
        _atomic_write_text(self.evidence_store_path, json.dumps(payload, indent=2, sort_keys=True))
        return merged

    # -- CreatorDNA snapshots ---------------------------------------------

    def save_dna_snapshot(self, session_id: str, dna: CreatorDNA) -> Path:
        path = self.dna_snapshot_path(session_id)
        _atomic_write_text(path, json.dumps(creator_dna_to_dict(dna), indent=2, sort_keys=True))
        return path

    def load_dna_snapshot(self, session_id: str) -> CreatorDNA | None:
        path = self.dna_snapshot_path(session_id)
        if not path.exists():
            return None
        payload = _load_json(path, default=None)
        return creator_dna_from_dict(payload)

    def latest_session_id(self) -> str | None:
        return self.load_learning_state().get("last_session_id")

    def latest_dna(self) -> CreatorDNA | None:
        session_id = self.latest_session_id()
        if session_id is None:
            return None
        return self.load_dna_snapshot(session_id)

    # -- learning history ---------------------------------------------------

    def load_learning_history(self) -> list[dict]:
        return _load_json(self.learning_history_path, default=[])

    def append_learning_history(self, entry: dict) -> None:
        history = self.load_learning_history()
        history.append(entry)
        _atomic_write_text(self.learning_history_path, json.dumps(history, indent=2, sort_keys=True))

    # -- learning state (coverage/gaps, last session pointer) -----------------

    def load_learning_state(self) -> dict:
        return _load_json(self.learning_state_path, default={})

    def save_learning_state(self, state: dict) -> None:
        _atomic_write_text(self.learning_state_path, json.dumps(state, indent=2, sort_keys=True))

    # -- reports ------------------------------------------------------------

    def save_report(self, session_id: str, report_name: str, markdown_text: str) -> Path:
        path = self.reports_directory(session_id) / f"{report_name}.md"
        _atomic_write_text(path, markdown_text)
        return path

    def load_report(self, session_id: str, report_name: str) -> str | None:
        path = self.reports_directory(session_id) / f"{report_name}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
