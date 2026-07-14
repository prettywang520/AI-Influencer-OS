"""
Content plan loader for AI Influencer OS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ContentPlanLoader:
    """Load and validate one monthly content-plan YAML file."""

    def __init__(
        self,
        plan_path: str | Path,
    ) -> None:
        self.plan_path = Path(plan_path).resolve()

    def load(self) -> dict[str, Any]:
        if not self.plan_path.exists():
            raise FileNotFoundError(
                f"Content plan was not found: {self.plan_path}"
            )

        with self.plan_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Content plan must contain a YAML object."
            )

        schedule = data.get("schedule")

        if not isinstance(schedule, dict):
            raise ValueError(
                "Content plan must contain a schedule object."
            )

        return data

    def get_date(
        self,
        date_string: str,
    ) -> dict[str, Any]:
        plan = self.load()
        schedule = plan["schedule"]

        item = schedule.get(date_string)

        if not isinstance(item, dict):
            raise KeyError(
                f"No content is scheduled for {date_string}."
            )

        request_name = str(
            item.get("request", "")
        ).strip()

        if not request_name:
            raise ValueError(
                f"Scheduled item for {date_string} "
                "does not contain a request."
            )

        return item