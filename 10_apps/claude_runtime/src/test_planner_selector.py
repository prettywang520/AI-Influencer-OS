from __future__ import annotations

import tempfile
from pathlib import Path

from .planner_selector import PlannerSelector
from .planning_history import (
    PlanningHistory,
    PlanningHistoryEntry,
)
from .planning_loader import build_planning_loader


def main() -> None:
    loader = build_planning_loader()

    errors = loader.validate_required_files()

    if errors:
        print("Planning database validation failed:")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        history = PlanningHistory(
            Path(temp_dir)
            / "planning_history.json"
        )

        history.add_entry(
            PlanningHistoryEntry(
                production_date="2026-07-26",
                country="Hong Kong",
                city="Hong Kong",
                venue="Central Ferry Piers",
                theme="sightseeing",
            )
        )

        selector = PlannerSelector(
            loader=loader,
            history=history,
            random_seed=42,
        )

        scheduled = selector.select(
            production_date="2026-07-27"
        )

        assert scheduled.country == "Japan"
        assert scheduled.city == "Tokyo"
        assert scheduled.venue == "Daikanyama T-Site"
        assert scheduled.theme == "bookstore"
        assert scheduled.source == "calendar"

        fallback = selector.select(
            production_date="2026-08-10"
        )

        assert fallback.country
        assert fallback.city
        assert fallback.venue
        assert fallback.theme
        assert fallback.source == "location_rotation"

        print("Planner Selector test passed.")
        print()

        print("Scheduled selection")
        print("-------------------")
        print(f"Date:       {scheduled.production_date}")
        print(f"Country:    {scheduled.country}")
        print(f"City:       {scheduled.city}")
        print(f"Venue:      {scheduled.venue}")
        print(f"Theme:      {scheduled.theme}")
        print(f"Source:     {scheduled.source}")
        print()

        print("Fallback selection")
        print("------------------")
        print(f"Date:       {fallback.production_date}")
        print(f"Country:    {fallback.country}")
        print(f"City:       {fallback.city}")
        print(f"Venue:      {fallback.venue}")
        print(f"Theme:      {fallback.theme}")
        print(f"Source:     {fallback.source}")


if __name__ == "__main__":
    main()