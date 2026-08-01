from __future__ import annotations

import tempfile
from pathlib import Path

from .planning_history import (
    PlanningHistory,
    PlanningHistoryEntry,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        history_path = (
            Path(temp_dir)
            / "planning_history.json"
        )

        history = PlanningHistory(history_path)

        history.add_entry(
            PlanningHistoryEntry(
                production_date="2026-07-27",
                country="Japan",
                city="Tokyo",
                venue="Daikanyama T-Site",
                theme="bookstore",
                behaviors=[
                    "compare_books",
                    "write_travel_ideas",
                ],
                emotions=[
                    "inspired",
                    "thoughtful",
                ],
                interactions=[
                    "staff_recommends_book",
                    "barista_refills_water",
                ],
                cameras=[
                    "hidden_shelf",
                    "top_down_table",
                ],
                story_stages=[
                    "discovery",
                    "quiet_creation",
                ],
            )
        )

        history.add_entry(
            PlanningHistoryEntry(
                production_date="2026-07-28",
                country="Taiwan",
                city="Taipei",
                venue="Ningxia Night Market",
                theme="night_market",
                behaviors=[
                    "read_stall_menu",
                    "take_first_bite",
                ],
                emotions=[
                    "excited",
                    "satisfied",
                ],
                interactions=[
                    "vendor_explains_flavour",
                    "vendor_hands_food",
                ],
                cameras=[
                    "queue_side_documentary",
                    "friend_candid_closeup",
                ],
                story_stages=[
                    "queue",
                    "first_bite",
                ],
            )
        )

        recent = history.recent_context(
            reference_date="2026-07-29",
            days=7,
        )

        assert "tokyo" in recent.cities
        assert "taipei" in recent.cities
        assert "bookstore" in recent.themes
        assert "night_market" in recent.themes
        assert "compare_books" in recent.behaviors
        assert "take_first_bite" in recent.behaviors
        assert "hidden_shelf" in recent.cameras

        assert history.was_city_used_recently(
            "Tokyo",
            reference_date="2026-07-29",
            days=3,
        )

        assert history.was_venue_used_recently(
            "Ningxia Night Market",
            reference_date="2026-07-29",
            days=30,
        )

        assert history.was_theme_used_recently(
            "bookstore",
            reference_date="2026-07-29",
            days=7,
        )

        status = history.status()

        assert status["entry_count"] == 2
        assert status["latest_date"] == "2026-07-28"

        print("Planning History test passed.")
        print()
        print(f"Entries:     {status['entry_count']}")
        print(f"Latest date: {status['latest_date']}")
        print(
            "Recent cities: "
            + ", ".join(
                sorted(recent.cities)
            )
        )
        print(
            "Recent themes: "
            + ", ".join(
                sorted(recent.themes)
            )
        )


if __name__ == "__main__":
    main()