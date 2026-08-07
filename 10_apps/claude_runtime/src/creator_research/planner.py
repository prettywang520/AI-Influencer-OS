"""ResearchPlanner -- decides which sections to collect, in what
order, where to resume from, and the retry policy. This is the one
place the 10-section/8-state mismatch (interfaces.ConnectorSection has
10 members, state.JobStatus.WORKING_STATES has 8) is reconciled -- see
docs/creator_research/architecture.md for why collect_posts() folds
into GRID and collect_relationships() folds into VISUAL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .exceptions import PlannerError, ResearchConfigError
from .interfaces import ConnectorSection
from .jobs import ResearchJob
from .progress import ResearchProgress
from .state import JobStatus

DEFAULT_PLANNER_CONFIG_RELATIVE_PATH = Path("config") / "creator_research" / "planner.yaml"


def _runtime_root() -> Path:
    """
    planner.py location: 10_apps/claude_runtime/src/creator_research/planner.py
    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def default_planner_config_path() -> Path:
    return _runtime_root() / DEFAULT_PLANNER_CONFIG_RELATIVE_PATH

SECTION_STATE_MAP: dict[str, str] = {
    ConnectorSection.PROFILE: JobStatus.PROFILE,
    ConnectorSection.GRID: JobStatus.GRID,
    ConnectorSection.POSTS: JobStatus.GRID,
    ConnectorSection.CAPTIONS: JobStatus.CAPTIONS,
    ConnectorSection.COMMENTS: JobStatus.COMMENTS,
    ConnectorSection.CREATOR_REPLIES: JobStatus.REPLIES,
    ConnectorSection.REELS: JobStatus.REELS,
    ConnectorSection.HIGHLIGHTS: JobStatus.HIGHLIGHTS,
    ConnectorSection.RELATIONSHIPS: JobStatus.VISUAL,
    ConnectorSection.VISUAL_EXAMPLES: JobStatus.VISUAL,
}


def state_for_section(section: str) -> str:
    try:
        return SECTION_STATE_MAP[section]
    except KeyError as exc:
        raise PlannerError(f"Unrecognized section: {section!r}") from exc


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3


def load_retry_policy(config_path: str | Path | None = None) -> RetryPolicy:
    path = Path(config_path) if config_path else default_planner_config_path()
    if not path.exists():
        raise ResearchConfigError(f"Planner config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ResearchConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ResearchConfigError(f"Planner config is empty or invalid: {path}")
    retry_section = raw.get("retry") or {}
    return RetryPolicy(max_attempts=int(retry_section.get("max_attempts", 3)))


class ResearchPlanner:
    def __init__(self, retry_policy: RetryPolicy | None = None) -> None:
        self.retry_policy = retry_policy or RetryPolicy()

    def section_order(self, job: ResearchJob) -> tuple[str, ...]:
        """Requested sections, ordered by their mapped state's
        position in JobStatus.WORKING_STATES, with ties (sections
        sharing a state) broken by ConnectorSection.ALL's fixed order
        -- always deterministic, never dict/set order dependent."""
        requested = set(job.requested_sections)
        for section in requested:
            state_for_section(section)  # raises PlannerError early on a bad section name

        def sort_key(section: str) -> tuple[int, int]:
            state = SECTION_STATE_MAP[section]
            return (JobStatus.WORKING_STATES.index(state), ConnectorSection.ALL.index(section))

        return tuple(sorted(requested, key=sort_key))

    def resume_point(self, job: ResearchJob, progress: ResearchProgress) -> str:
        """First not-yet-completed working state in the fixed
        JobStatus.WORKING_STATES pipeline -- every job flows through
        every working state regardless of which sections it actually
        requested (a state with no requested sections simply collects
        nothing and completes immediately). STARTING if nothing has
        completed yet; COMPLETE if every working state is already
        done."""
        for state in JobStatus.WORKING_STATES:
            if state not in progress.completed_steps:
                return state
        return JobStatus.COMPLETE

    def state_order(self, job: ResearchJob) -> tuple[str, ...]:
        """The distinct working states (deduplicated, in canonical
        order) that job.section_order() will pass through."""
        seen: list[str] = []
        for section in self.section_order(job):
            state = SECTION_STATE_MAP[section]
            if state not in seen:
                seen.append(state)
        return tuple(seen)

    def sections_for_state(self, job: ResearchJob, state: str) -> tuple[str, ...]:
        return tuple(section for section in self.section_order(job) if SECTION_STATE_MAP[section] == state)
