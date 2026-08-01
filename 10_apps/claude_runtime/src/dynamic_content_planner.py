from __future__ import annotations

from datetime import date
from typing import Any

from .content_models import DailyContentPlan
from .planner_selector import DailySelection, PlannerSelector
from .story_builder import (
    ResolvedScene,
    StoryBuilder,
    StoryBuilderError,
)


class DynamicContentPlannerError(RuntimeError):
    """Raised when a dynamic daily content plan cannot be created."""


class DynamicContentPlanner:
    """
    Builds one complete AIKO daily content plan.

    Pipeline:

    production date
    -> PlannerSelector
    -> DailySelection
    -> StoryBuilder
    -> Feed x1
    -> Stories x4
    -> Reel scenes x5
    -> DailyContentPlan

    This planner does not generate prompts, captions or hashtags.
    Those tasks remain inside ContentService.
    """

    def __init__(
        self,
        *,
        selector: PlannerSelector,
        builder: StoryBuilder,
    ) -> None:
        self.selector = selector
        self.builder = builder

    @staticmethod
    def _normalise_key(value: str) -> str:
        return (
            value.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _title_from_id(value: str) -> str:
        return (
            value.replace("_", " ")
            .strip()
            .title()
        )

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        return [value]

    @staticmethod
    def _build_location(
        selection: DailySelection,
    ) -> str:
        location_parts = [
            selection.venue,
            selection.district,
            selection.city,
            selection.country,
        ]

        output: list[str] = []
        seen: set[str] = set()

        for part in location_parts:
            if not part:
                continue

            clean = str(part).strip()
            key = clean.lower()

            if not clean or key in seen:
                continue

            seen.add(key)
            output.append(clean)

        return ", ".join(output)

    def _theme_data(
        self,
        theme: str,
    ) -> dict[str, Any]:
        themes = self.builder.story_data.get(
            "themes",
            {},
        )

        if not isinstance(themes, dict):
            raise DynamicContentPlannerError(
                "story_templates.yaml must contain a "
                "'themes' mapping"
            )

        theme_key = self._normalise_key(theme)
        theme_data = themes.get(theme_key)

        if not isinstance(theme_data, dict):
            available = ", ".join(
                sorted(themes.keys())
            )

            raise DynamicContentPlannerError(
                f"Unknown theme '{theme_key}'. "
                f"Available themes: {available}"
            )

        return theme_data

    @staticmethod
    def _require_scene(
        *,
        stage: str,
        scenes: dict[str, ResolvedScene],
        content_label: str,
    ) -> ResolvedScene:
        stage_key = (
            stage.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        scene = scenes.get(stage_key)

        if scene is None:
            raise DynamicContentPlannerError(
                f"Missing resolved scene '{stage_key}' "
                f"for {content_label}"
            )

        return scene

    def _feed_stage(
        self,
        theme_data: dict[str, Any],
    ) -> str:
        stage = theme_data.get("feed_stage")

        if not stage:
            feed_data = theme_data.get("feed", {})

            if isinstance(feed_data, dict):
                stage = (
                    feed_data.get("stage")
                    or feed_data.get("story_stage")
                )

        if not stage:
            raise DynamicContentPlannerError(
                "Theme is missing feed_stage"
            )

        return self._normalise_key(str(stage))

    def _story_stages(
        self,
        theme_data: dict[str, Any],
    ) -> list[str]:
        stages = [
            self._normalise_key(str(stage))
            for stage in self._as_list(
                theme_data.get("stories")
            )
            if str(stage).strip()
        ]

        if len(stages) != 4:
            raise DynamicContentPlannerError(
                "Each theme must define exactly four "
                f"Story stages; received {len(stages)}"
            )

        if len(stages) != len(set(stages)):
            raise DynamicContentPlannerError(
                "Duplicate Story stages detected"
            )

        return stages

    def _reel_stages(
        self,
        theme_data: dict[str, Any],
    ) -> list[str]:
        stages = [
            self._normalise_key(str(stage))
            for stage in self._as_list(
                theme_data.get("reels")
            )
            if str(stage).strip()
        ]

        if len(stages) != 5:
            raise DynamicContentPlannerError(
                "Each theme must define exactly five "
                f"Reel stages; received {len(stages)}"
            )

        if len(stages) != len(set(stages)):
            raise DynamicContentPlannerError(
                "Duplicate Reel stages detected"
            )

        return stages

    @staticmethod
    def _story_stage_name(
        *,
        stage: str,
        story_index: int,
    ) -> str:
        """
        Preserve the required four-part Instagram Story progression
        while retaining the actual source stage inside the title.

        Story 1: arrival
        Story 2: discovery
        Story 3: quiet_moment
        Story 4: leaving
        """
        required_progression = [
            "arrival",
            "discovery",
            "quiet_moment",
            "leaving",
        ]

        if 0 <= story_index < len(
            required_progression
        ):
            return required_progression[
                story_index
            ]

        return stage

    def generate(
        self,
        *,
        target_date: str | None = None,
    ) -> DailyContentPlan:
        production_date = (
            target_date
            or date.today().isoformat()
        )

        selection = self.selector.select(
            production_date=production_date
        )

        theme_key = self._normalise_key(
            selection.theme
        )

        theme_data = self._theme_data(
            theme_key
        )

        location = self._build_location(
            selection
        )

        try:
            resolved_scenes = (
                self.builder.build_theme_scenes(
                    theme=theme_key,
                    weather_mode=(
                        selection.weather_mode
                    ),
                    season=selection.season,
                )
            )
        except StoryBuilderError as exc:
            raise DynamicContentPlannerError(
                f"Unable to resolve theme scenes: {exc}"
            ) from exc

        feed_stage = self._feed_stage(
            theme_data
        )

        story_stages = self._story_stages(
            theme_data
        )

        reel_stages = self._reel_stages(
            theme_data
        )

        feed_scene = self._require_scene(
            stage=feed_stage,
            scenes=resolved_scenes,
            content_label="Feed",
        )

        feed = self.builder.build_content_moment(
            content_id=(
                f"{production_date}-feed-01"
            ),
            content_type="feed",
            title=(
                f"{self._title_from_id(feed_stage)} "
                f"at {selection.venue}"
            ),
            location=location,
            scene=feed_scene,
            story_stage="hero_moment",
        )

        stories = []

        for index, stage in enumerate(
            story_stages,
            start=1,
        ):
            scene = self._require_scene(
                stage=stage,
                scenes=resolved_scenes,
                content_label=f"Story {index}",
            )

            stories.append(
                self.builder.build_content_moment(
                    content_id=(
                        f"{production_date}-story-"
                        f"{index:02d}"
                    ),
                    content_type="story",
                    title=self._title_from_id(
                        stage
                    ),
                    location=location,
                    scene=scene,
                    story_stage=(
                        self._story_stage_name(
                            stage=stage,
                            story_index=index - 1,
                        )
                    ),
                )
            )

        reel_scenes = []

        for index, stage in enumerate(
            reel_stages,
            start=1,
        ):
            scene = self._require_scene(
                stage=stage,
                scenes=resolved_scenes,
                content_label=(
                    f"Reel scene {index}"
                ),
            )

            reel_scenes.append(
                self.builder.build_reel_scene(
                    scene_number=index,
                    title=self._title_from_id(
                        stage
                    ),
                    scene=scene,
                    duration_seconds=2.5,
                )
            )

        theme_title = str(
            theme_data.get("title")
            or self._title_from_id(theme_key)
        )

        story_summary = (
            selection.story_direction.strip()
            if selection.story_direction
            else (
                f"Aiko experiences {selection.venue} "
                f"through a complete {theme_title} "
                "journey with real activities, natural "
                "interaction and emotional progression."
            )
        )

        return DailyContentPlan(
            date=production_date,
            country=selection.country,
            city=selection.city,
            venue=selection.venue,
            theme=theme_key,
            story_summary=story_summary,
            feed=feed,
            stories=stories,
            reel_scenes=reel_scenes,
        )
    
            