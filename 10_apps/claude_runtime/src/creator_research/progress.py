"""ResearchProgress -- tracks how far a job has advanced through its
planned states. Percentage/estimated-remaining are step-count-based,
not wall-clock-time estimates: there is no real timing data to
estimate from without a real connector, and this package never
fabricates one.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ResearchProgress:
    total_steps: int
    current_step: str = ""
    completed_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, len(self.completed_steps) / self.total_steps)

    @property
    def estimated_remaining_steps(self) -> int:
        return max(0, self.total_steps - len(self.completed_steps))

    def mark_step_started(self, step: str) -> None:
        self.current_step = step

    def mark_step_completed(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        if self.current_step == step:
            self.current_step = ""

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
