from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from .planning_history import PlanningHistory
from .planning_loader import PlanningLoader


class PlannerSelectorError(RuntimeError):
    """Raised when a valid daily planning selection cannot be created."""


@dataclass(slots=True)
class DailySelection:
    production_date: str
    country: str
    city: str
    district: str | None
    venue: str
    venue_id: str | None
    theme: str
    daypart: str
    weather_mode: str
    season: str
    story_direction: str
    timezone: str | None = None
    language_detail: str | None = None
    source: str = "calendar"
    indoor: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlannerSelector:
    """
    Selects the daily location and theme before StoryBuilder runs.

    Priority:

    1. Exact date entry in calendar.yaml
    2. Calendar fallback theme rotation
    3. Location rotation database
    4. Planning history anti-repeat rules
    """

    def __init__(
        self,
        *,
        loader: PlanningLoader,
        history: PlanningHistory,
        random_seed: int | None = None,
    ) -> None:
        self.loader = loader
        self.history = history
        self.random = random.Random(random_seed)

        self.calendar_data = loader.calendar()
        self.location_data = loader.locations()

    @staticmethod
    def _normalise(value: str | None) -> str:
        if not value:
            return ""

        return (
            value.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        return [value]

    @staticmethod
    def _parse_date(value: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise PlannerSelectorError(
                f"Invalid production date: {value}"
            ) from exc

    def _schedule(self) -> dict[str, Any]:
        schedule = self.calendar_data.get("schedule", {})

        if not isinstance(schedule, dict):
            raise PlannerSelectorError(
                "calendar.yaml must contain a 'schedule' mapping"
            )

        return schedule

    def _calendar_entry(
        self,
        production_date: str,
    ) -> dict[str, Any] | None:
        entry = self._schedule().get(production_date)

        if entry is None:
            return None

        if not isinstance(entry, dict):
            raise PlannerSelectorError(
                f"Calendar entry {production_date} must be a mapping"
            )

        return entry

    def _countries(self) -> dict[str, Any]:
        countries = self.location_data.get("countries", {})

        if not isinstance(countries, dict):
            raise PlannerSelectorError(
                "location_rotation.yaml must contain a 'countries' mapping"
            )

        return countries

    def _city_data(
        self,
        *,
        country: str,
        city: str,
    ) -> dict[str, Any] | None:
        country_data = self._countries().get(country)

        if not isinstance(country_data, dict):
            return None

        cities = country_data.get("cities", {})

        if not isinstance(cities, dict):
            return None

        city_data = cities.get(city)

        return city_data if isinstance(city_data, dict) else None

    def _find_venue_by_name(
        self,
        *,
        country: str,
        city: str,
        venue_name: str,
        theme: str,
    ) -> dict[str, Any] | None:
        city_data = self._city_data(
            country=country,
            city=city,
        )

        if not city_data:
            return None

        venues = city_data.get("venues", {})

        if not isinstance(venues, dict):
            return None

        theme_venues = venues.get(theme, [])

        for venue in self._as_list(theme_venues):
            if not isinstance(venue, dict):
                continue

            if self._normalise(str(venue.get("name", ""))) == (
                self._normalise(venue_name)
            ):
                return venue

        return None

    def _selection_from_calendar(
        self,
        *,
        production_date: str,
        entry: dict[str, Any],
    ) -> DailySelection:
        required_fields = [
            "country",
            "city",
            "venue",
            "theme",
        ]

        missing = [
            field
            for field in required_fields
            if not entry.get(field)
        ]

        if missing:
            raise PlannerSelectorError(
                f"Calendar entry {production_date} is missing: "
                + ", ".join(missing)
            )

        country = str(entry["country"])
        city = str(entry["city"])
        venue_name = str(entry["venue"])
        theme = self._normalise(str(entry["theme"]))

        city_data = self._city_data(
            country=country,
            city=city,
        ) or {}

        venue_data = self._find_venue_by_name(
            country=country,
            city=city,
            venue_name=venue_name,
            theme=theme,
        ) or {}

        theme_aliases = {
            "old_town": "sightseeing",
        }

        theme = theme_aliases.get(theme, theme)

        return DailySelection(
            production_date=production_date,
            country=country,
            city=city,
            district=(
                str(entry.get("district"))
                if entry.get("district")
                else venue_data.get("district")
            ),
            venue=venue_name,
            venue_id=(
                str(venue_data.get("id"))
                if venue_data.get("id")
                else None
            ),
            theme=theme,
            daypart=str(
                entry.get("daypart", "afternoon")
            ),
            weather_mode=self._normalise(
                str(entry.get("weather_mode", "clear"))
            ),
            season=self._normalise(
                str(entry.get("season", "summer"))
            ),
            story_direction=str(
                entry.get(
                    "story_direction",
                    f"Aiko explores {venue_name} through a real local experience.",
                )
            ).strip(),
            timezone=(
                str(city_data.get("timezone"))
                if city_data.get("timezone")
                else None
            ),
            language_detail=(
                str(city_data.get("language_detail"))
                if city_data.get("language_detail")
                else None
            ),
            source="calendar",
            indoor=(
                bool(venue_data.get("indoor"))
                if "indoor" in venue_data
                else None
            ),
        )

    def _theme_sequence(self) -> list[str]:
        fallback = self.calendar_data.get(
            "rotation_fallback",
            {},
        )

        if not isinstance(fallback, dict):
            return []

        sequence = fallback.get(
            "theme_sequence",
            [],
        )

        return [
            self._normalise(str(theme))
            for theme in self._as_list(sequence)
            if str(theme).strip()
        ]

    def _fallback_theme(
        self,
        production_date: str,
    ) -> str:
        sequence = self._theme_sequence()

        if not sequence:
            raise PlannerSelectorError(
                "No fallback theme sequence is configured"
            )

        target_date = self._parse_date(production_date)
        index = target_date.toordinal() % len(sequence)

        recent = self.history.recent_context(
            reference_date=production_date,
            days=7,
        )

        for offset in range(len(sequence)):
            candidate = sequence[
                (index + offset) % len(sequence)
            ]

            if candidate not in recent.themes:
                return candidate

        return sequence[index]

    def _venue_candidates(
        self,
        *,
        theme: str,
        daypart: str | None = None,
        weather_mode: str | None = None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        for country, country_data in self._countries().items():
            if not isinstance(country_data, dict):
                continue

            cities = country_data.get("cities", {})

            if not isinstance(cities, dict):
                continue

            for city, city_data in cities.items():
                if not isinstance(city_data, dict):
                    continue

                venues = city_data.get("venues", {})

                if not isinstance(venues, dict):
                    continue

                theme_venues = venues.get(theme, [])

                for venue in self._as_list(theme_venues):
                    if not isinstance(venue, dict):
                        continue

                    preferred_dayparts = [
                        self._normalise(str(value))
                        for value in self._as_list(
                            venue.get("preferred_dayparts")
                        )
                    ]

                    allowed_weather = [
                        self._normalise(str(value))
                        for value in self._as_list(
                            venue.get("allowed_weather")
                        )
                    ]

                    daypart_match = (
                        not daypart
                        or not preferred_dayparts
                        or self._normalise(daypart)
                        in preferred_dayparts
                    )

                    weather_match = (
                        not weather_mode
                        or not allowed_weather
                        or self._normalise(weather_mode)
                        in allowed_weather
                    )

                    candidates.append(
                        {
                            "country": str(country),
                            "city": str(city),
                            "timezone": city_data.get(
                                "timezone"
                            ),
                            "language_detail": city_data.get(
                                "language_detail"
                            ),
                            "venue": venue,
                            "daypart_match": daypart_match,
                            "weather_match": weather_match,
                        }
                    )

        return candidates

    def _weighted_choice(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not candidates:
            raise PlannerSelectorError(
                "No valid venue candidates found"
            )

        weights = [
            max(
                int(
                    candidate.get("venue", {}).get(
                        "weight",
                        1,
                    )
                ),
                1,
            )
            for candidate in candidates
        ]

        return self.random.choices(
            candidates,
            weights=weights,
            k=1,
        )[0]

    def _filter_recent(
        self,
        *,
        candidates: list[dict[str, Any]],
        production_date: str,
    ) -> list[dict[str, Any]]:
        rules = self.location_data.get(
            "selection_rules",
            {},
        )

        if not isinstance(rules, dict):
            rules = {}

        city_days = int(
            rules.get("avoid_recent_city_days", 3)
        )
        venue_days = int(
            rules.get("avoid_recent_venue_days", 30)
        )

        recent_cities = self.history.recent_context(
            reference_date=production_date,
            days=city_days,
        ).cities

        recent_venues = self.history.recent_context(
            reference_date=production_date,
            days=venue_days,
        ).venues

        filtered = []

        for candidate in candidates:
            venue = candidate["venue"]

            city_key = self._normalise(
                candidate["city"]
            )
            venue_key = self._normalise(
                str(venue.get("name", ""))
            )

            if city_key in recent_cities:
                continue

            if venue_key in recent_venues:
                continue

            filtered.append(candidate)

        return filtered or candidates

    def _selection_from_rotation(
        self,
        *,
        production_date: str,
        theme: str,
    ) -> DailySelection:
        theme_aliases = {
            "old_town": "sightseeing",
        }

        theme = self._normalise(theme)
        theme = theme_aliases.get(
            theme,
            theme,
        )

        candidates = self._venue_candidates(
            theme=theme,
    )

        candidates = self._filter_recent(
            candidates=candidates,
            production_date=production_date,
        )

        preferred = [
            candidate
            for candidate in candidates
            if candidate["daypart_match"]
            and candidate["weather_match"]
        ]

        selected = self._weighted_choice(
            preferred or candidates
        )

        venue = selected["venue"]

        default_dayparts = self._as_list(
            venue.get("preferred_dayparts")
        )

        daypart = (
            str(default_dayparts[0])
            if default_dayparts
            else "afternoon"
        )

        theme_aliases = {
            "old_town": "sightseeing",
        }

        theme = theme_aliases.get(theme, theme)

        return DailySelection(
            production_date=production_date,
            country=selected["country"],
            city=selected["city"],
            district=(
                str(venue.get("district"))
                if venue.get("district")
                else None
            ),
            venue=str(venue["name"]),
            venue_id=(
                str(venue.get("id"))
                if venue.get("id")
                else None
            ),
            theme=theme,
            daypart=daypart,
            weather_mode="clear",
            season=self._season_for_date(
                production_date
            ),
            story_direction=(
                f"Aiko explores {venue['name']} through a complete "
                f"{theme.replace('_', ' ')} story with real activity, "
                "social interaction and a clear emotional progression."
            ),
            timezone=(
                str(selected["timezone"])
                if selected.get("timezone")
                else None
            ),
            language_detail=(
                str(selected["language_detail"])
                if selected.get("language_detail")
                else None
            ),
            source="location_rotation",
            indoor=(
                bool(venue.get("indoor"))
                if "indoor" in venue
                else None
            ),
        )

    @staticmethod
    def _season_for_date(
        production_date: str,
    ) -> str:
        month = datetime.strptime(
            production_date,
            "%Y-%m-%d",
        ).month

        if month in {3, 4, 5}:
            return "spring"

        if month in {6, 7, 8}:
            return "summer"

        if month in {9, 10, 11}:
            return "autumn"

        return "winter"

    def select(
        self,
        *,
        production_date: str,
    ) -> DailySelection:
        self._parse_date(production_date)

        calendar_entry = self._calendar_entry(
            production_date
        )

        if calendar_entry:
            return self._selection_from_calendar(
                production_date=production_date,
                entry=calendar_entry,
            )

        theme = self._fallback_theme(
            production_date
        )

        return self._selection_from_rotation(
            production_date=production_date,
            theme=theme,
        )