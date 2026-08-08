"""WorkflowConfig -- config/creator_learning/workflow.yaml loading --
plus a thin `python3 -m src.creator_learning.workflow --creator-url
<url>` entry point that delegates to WorkflowRunner.start() in
zero-network mode. This matches the PURPOSE section literally: the
"final command" this phase targets. The full multi-action CLI
(--start/--resume/--cancel/--status/--validate) lives in
workflow_cli.py; this module's __main__ is intentionally a reduced
surface, not a duplicate of it.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .workflow_exceptions import WorkflowConfigError

DEFAULT_WORKFLOW_CONFIG_RELATIVE_PATH = Path("config") / "creator_learning" / "workflow.yaml"


def _runtime_root() -> Path:
    """
    workflow.py location: 10_apps/claude_runtime/src/creator_learning/workflow.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_workflow_config_path() -> Path:
    return _runtime_root() / DEFAULT_WORKFLOW_CONFIG_RELATIVE_PATH


@dataclass(slots=True)
class WorkflowConfig:
    schema_version: str
    max_retry_attempts: int
    generate_reports_on_completion: bool


def load_workflow_config(config_path: str | Path | None = None) -> WorkflowConfig:
    path = Path(config_path) if config_path else default_workflow_config_path()
    if not path.exists():
        raise WorkflowConfigError(f"Workflow config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise WorkflowConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise WorkflowConfigError(f"Workflow config is empty or invalid: {path}")

    section = raw.get("workflow") or {}
    return WorkflowConfig(
        schema_version=str(section.get("schema_version", "1.0")),
        max_retry_attempts=int(section.get("max_retry_attempts", 3)),
        generate_reports_on_completion=bool(section.get("generate_reports_on_completion", True)),
    )


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="creator_learning.workflow",
        description="Run the Creator Learning workflow to completion for one creator (zero-network mode only).",
    )
    parser.add_argument("--creator-url", required=True)
    parser.add_argument("--intake-evidence-bundle", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Deferred import: workflow_runner.py imports WorkflowConfig from
    # this module, so importing WorkflowRunner at module scope here
    # would create a circular import. Delaying it to call time (this
    # script's actual entry point) breaks the cycle cleanly.
    from .workflow_runner import WorkflowRunner
    from .workflow_exceptions import WorkflowError

    args = _parse_arguments(argv)
    runner = WorkflowRunner()
    try:
        result = runner.start(args.creator_url, intake_evidence_bundle_path=args.intake_evidence_bundle)
    except WorkflowError as exc:
        print(f"[CreatorLearningWorkflow] Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.checkpoint.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"[CreatorLearningWorkflow] workflow_id={result.workflow_id} state={result.state}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
