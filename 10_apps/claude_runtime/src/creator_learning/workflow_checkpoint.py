"""Workflow checkpoint persistence -- one JSON file per workflow_id
under <knowledge_base_root>/<creator_id>/workflow/, plus a latest.json
pointer (same pattern as knowledge_base.py's learning_state.json
last_session_id pointer). Atomic JSON writes (temp-file +
Path.replace()), the same convention used throughout this codebase.
Loading always revalidates identity before reuse: a workflow_id/
creator_id mismatch is a hard error, never silently adopted (mirrors
creator_research/instagram/checkpoint.py exactly).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .workflow_exceptions import WorkflowCheckpointError, WorkflowCheckpointMismatchError
from .workflow_models import WorkflowCheckpoint
from .workflow_state import WorkflowState


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


def _workflow_dir(root_directory: str | Path, creator_id: str) -> Path:
    return Path(root_directory) / creator_id / "workflow"


def _checkpoint_path(root_directory: str | Path, creator_id: str, workflow_id: str) -> Path:
    return _workflow_dir(root_directory, creator_id) / f"{workflow_id}.json"


def _latest_pointer_path(root_directory: str | Path, creator_id: str) -> Path:
    return _workflow_dir(root_directory, creator_id) / "latest.json"


def save_checkpoint(root_directory: str | Path, checkpoint: WorkflowCheckpoint) -> Path:
    path = _checkpoint_path(root_directory, checkpoint.creator_id, checkpoint.workflow_id)
    _atomic_write_text(path, json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True))
    _atomic_write_text(
        _latest_pointer_path(root_directory, checkpoint.creator_id),
        json.dumps({"workflow_id": checkpoint.workflow_id, "state": checkpoint.state}, indent=2, sort_keys=True),
    )
    return path


def load_checkpoint(root_directory: str | Path, creator_id: str, workflow_id: str) -> WorkflowCheckpoint | None:
    path = _checkpoint_path(root_directory, creator_id, workflow_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowCheckpointError(f"Invalid JSON in {path}: {exc}") from exc
    return WorkflowCheckpoint.from_dict(payload)


def load_latest_checkpoint(root_directory: str | Path, creator_id: str) -> WorkflowCheckpoint | None:
    pointer_path = _latest_pointer_path(root_directory, creator_id)
    if not pointer_path.exists():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowCheckpointError(f"Invalid JSON in {pointer_path}: {exc}") from exc
    workflow_id = pointer.get("workflow_id")
    if not workflow_id:
        return None
    return load_checkpoint(root_directory, creator_id, workflow_id)


def list_checkpoints(root_directory: str | Path, creator_id: str) -> list[WorkflowCheckpoint]:
    directory = _workflow_dir(root_directory, creator_id)
    if not directory.is_dir():
        return []
    checkpoints: list[WorkflowCheckpoint] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "latest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkflowCheckpointError(f"Invalid JSON in {path}: {exc}") from exc
        checkpoints.append(WorkflowCheckpoint.from_dict(payload))
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.created_at)


def find_checkpoint(root_directory: str | Path, workflow_id: str) -> WorkflowCheckpoint | None:
    """Looks up a checkpoint by workflow_id alone, without knowing
    which creator it belongs to -- workflow_id is a random 16-hex-char
    token (~64 bits), so a collision across creators is not a
    practical concern. Used by the CLI (`--resume WORKFLOW_ID`,
    `--cancel WORKFLOW_ID`) which never asks the caller for a
    creator_id. Scans every creator directory under root_directory."""
    root = Path(root_directory)
    if not root.is_dir():
        return None
    matches = sorted(root.glob(f"*/workflow/{workflow_id}.json"))
    if not matches:
        return None
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    return WorkflowCheckpoint.from_dict(payload)


def list_all_checkpoints(root_directory: str | Path) -> list[WorkflowCheckpoint]:
    """Every checkpoint across every creator directory, oldest first.
    Read-only. Used by --status (no id given) and validate_checkpoints()."""
    root = Path(root_directory)
    if not root.is_dir():
        return []
    checkpoints: list[WorkflowCheckpoint] = []
    for path in sorted(root.glob("*/workflow/*.json")):
        if path.name == "latest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkflowCheckpointError(f"Invalid JSON in {path}: {exc}") from exc
        checkpoints.append(WorkflowCheckpoint.from_dict(payload))
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.created_at)


def validate_checkpoints(root_directory: str | Path, creator_id: str | None = None) -> tuple[bool, list[str]]:
    """Structural check over every checkpoint file (or just one
    creator's, if given): state is a recognized WorkflowState,
    last_completed_step (if set) is a recognized forward-progression
    state, and no workflow_id is duplicated across the scanned set.
    Read-only -- never writes (mirrors creator_research/queue.py's
    validate_queue())."""
    checkpoints = list_checkpoints(root_directory, creator_id) if creator_id else list_all_checkpoints(root_directory)
    errors: list[str] = []
    seen_ids: set[str] = set()
    for checkpoint in checkpoints:
        if checkpoint.state not in WorkflowState.ALL:
            errors.append(f"workflow {checkpoint.workflow_id!r} has unrecognized state {checkpoint.state!r}")
        if checkpoint.last_completed_step is not None and checkpoint.last_completed_step not in WorkflowState.ORDER:
            errors.append(
                f"workflow {checkpoint.workflow_id!r} has unrecognized last_completed_step "
                f"{checkpoint.last_completed_step!r}"
            )
        if checkpoint.workflow_id in seen_ids:
            errors.append(f"duplicate workflow_id: {checkpoint.workflow_id!r}")
        seen_ids.add(checkpoint.workflow_id)
    return not errors, errors


def validate_checkpoint(checkpoint: WorkflowCheckpoint, *, workflow_id: str, creator_id: str) -> None:
    """Raises WorkflowCheckpointMismatchError if workflow_id/creator_id
    don't match. Read-only, no return value on success."""
    if checkpoint.workflow_id != workflow_id or checkpoint.creator_id != creator_id:
        raise WorkflowCheckpointMismatchError(
            f"checkpoint identity mismatch: stored workflow_id={checkpoint.workflow_id!r}/"
            f"creator_id={checkpoint.creator_id!r} vs requested workflow_id={workflow_id!r}/creator_id={creator_id!r}"
        )
