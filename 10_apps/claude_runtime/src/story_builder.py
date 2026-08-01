from __future__ import annotations

from datetime import date
from pathlib import Path
import random

from .motion_engine import MotionEngine

from dataclasses import dataclass
from typing import Any

from .content_models import ContentMoment, ReelScene
from .planning_loader import PlanningLoader


class StoryBuilderError(RuntimeError):
    """Raised when a dynamic story cannot be built safely."""

@dataclass(slots=True)

class ResolvedScene:
    stage: str
    purpose: str
    behavior_id: str
    behavior_text: str
    motion_category: str
    emotion_id: str
    emotion_text: str
    interaction_id: str
    interaction_text: str
    camera_id: str
    camera_text: str
    daily_details: list[str]


class StoryBuilder:
    """
    Builds Feed, Stories and Reel scenes from the Planning Database.

    Flow:

    Theme
    → Story Stage
    → Behavior
    → Emotion
    → Interaction
    → Camera
    → Weather Details
    → Seasonal Details
    → ContentMoment / ReelScene
    """

    def __init__(
        self,
        loader: PlanningLoader,
        *,
        random_seed: int | None = None,
    ) -> None:
        self.loader = loader
        self.random = random.Random(random_seed)

        self.project_root = Path(__file__).resolve().parents[3]

        self.output_root = (
            Path(__file__).resolve().parents[1]
            / "output"
        )

        self.story_data = loader.story_templates()
        self.behavior_data = loader.behaviors()
        self.camera_data = loader.cameras()
        self.emotion_data = loader.emotions()
        self.interaction_data = loader.interactions()
        self.weather_data = loader.weather_rules()
        self.seasonal_data = loader.seasonal_rules()

        self.motion_engine = MotionEngine(
            library_path=(
                self.project_root
                / "03_personas"
                / "aiko"
                / "motion"
                / "motion_library.yaml"
            ),
            rules_path=(
                self.project_root
                / "03_personas"
                / "aiko"
                / "motion"
                / "motion_rules.yaml"
            ),
            history_path=(
                self.output_root
                / "history"
                / "motion.json"
            ),
            random_seed=random_seed,
        )
        
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
        return value.replace("_", " ").strip().title()

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        return [value]

    def _themes(self) -> dict[str, Any]:
        themes = self.story_data.get("themes", {})

        if not isinstance(themes, dict):
            raise StoryBuilderError(
                "story_templates.yaml must contain a 'themes' mapping"
            )

        return themes

    def _get_theme(self, theme: str) -> dict[str, Any]:
        theme_key = self._normalise_key(theme)
        themes = self._themes()

        theme_data = themes.get(theme_key)

        if not isinstance(theme_data, dict):
            available = ", ".join(sorted(themes.keys()))

            raise StoryBuilderError(
                f"Unknown story theme: {theme_key}. "
                f"Available themes: {available}"
            )

        return theme_data

    @staticmethod
    def _database_items(
        database: dict[str, Any],
        primary_key: str,
    ) -> dict[str, Any]:
        """
        Support both:

        behaviors:
          compare_books: ...

        and:

        items:
          compare_books: ...
        """
        values = database.get(primary_key)

        if isinstance(values, dict):
            return values

        values = database.get("items")

        if isinstance(values, dict):
            return values

        return {}

    def _behaviors(self) -> dict[str, Any]:
        return self._database_items(
            self.behavior_data,
            "behaviors",
        )

    def _cameras(self) -> dict[str, Any]:
        return self._database_items(
            self.camera_data,
            "cameras",
        )

    def _emotions(self) -> dict[str, Any]:
        return self._database_items(
            self.emotion_data,
            "emotions",
        )

    def _interactions(self) -> dict[str, Any]:
        return self._database_items(
            self.interaction_data,
            "interactions",
        )

    def _choose_id(
        self,
        pool: list[str],
        *,
        database: dict[str, Any],
        excluded: set[str],
        label: str,
    ) -> str:
        """
        Select one item from a stage pool.

        Preference:
        1. Exists in database and not recently selected
        2. Exists in database
        3. Raw pool ID as fallback
        """
        if not pool:
            raise StoryBuilderError(
                f"Empty {label} pool in story template"
            )

        normalised_pool = [
            self._normalise_key(str(item))
            for item in pool
        ]

        available = [
            item
            for item in normalised_pool
            if item in database and item not in excluded
        ]

        if not available:
            available = [
                item
                for item in normalised_pool
                if item in database
            ]

        if not available:
            available = [
                item
                for item in normalised_pool
                if item not in excluded
            ]

        if not available:
            available = normalised_pool

        return self.random.choice(available)

    def _resolve_behavior_text(
        self,
        behavior_id: str,
    ) -> str:
        item = self._behaviors().get(behavior_id, {})

        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            return str(
                item.get("prompt_text")
                or item.get("description")
                or item.get("title")
                or self._title_from_id(behavior_id)
            )

        return self._title_from_id(behavior_id)

    def _resolve_camera_text(
        self,
        camera_id: str,
    ) -> str:
        item = self._cameras().get(camera_id, {})

        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            return str(
                item.get("prompt_text")
                or item.get("description")
                or item.get("title")
                or self._title_from_id(camera_id)
            )

        return self._title_from_id(camera_id)

    def _resolve_emotion_text(
        self,
        emotion_id: str,
    ) -> str:
        item = self._emotions().get(emotion_id, {})

        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            return str(
                item.get("prompt_text")
                or item.get("description")
                or item.get("title")
                or self._title_from_id(emotion_id)
            )

        return self._title_from_id(emotion_id)

    def _resolve_interaction_text(
        self,
        interaction_id: str,
    ) -> str:
        item = self._interactions().get(
            interaction_id,
            {},
        )

        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            actor = item.get("actor")
            action = (
                item.get("prompt_text")
                or item.get("action")
                or item.get("description")
            )

            if actor and action:
                return f"{actor} {action}"

            return str(
                action
                or item.get("title")
                or self._title_from_id(interaction_id)
            )

        return self._title_from_id(interaction_id)

    @staticmethod
    def _extract_rule_details(
        database: dict[str, Any],
        *,
        section_name: str,
        selected_key: str | None,
    ) -> list[str]:
        if not selected_key:
            return []

        section = database.get(section_name, {})

        if not isinstance(section, dict):
            return []

        key = (
            selected_key.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        item = section.get(key, {})

        if isinstance(item, list):
            return [str(value) for value in item]

        if isinstance(item, dict):
            details = item.get("details", [])

            if isinstance(details, list):
                return [str(value) for value in details]

        return []

    def _weather_details(
        self,
        weather_mode: str | None,
    ) -> list[str]:
        return self._extract_rule_details(
            self.weather_data,
            section_name="weather",
            selected_key=weather_mode,
        )

    def _seasonal_details(
        self,
        season: str | None,
    ) -> list[str]:
        return self._extract_rule_details(
            self.seasonal_data,
            section_name="seasons",
            selected_key=season,
        )

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()

        for value in values:
            clean = value.strip()

            if not clean:
                continue

            key = clean.lower()

            if key in seen:
                continue

            seen.add(key)
            output.append(clean)

        return output

    def _resolve_details(
        self,
        stage_data: dict[str, Any],
        *,
        weather_mode: str | None,
        season: str | None,
        minimum: int = 3,
        maximum: int = 6,
    ) -> list[str]:
        base_details = [
            str(value)
            for value in self._as_list(
                stage_data.get("detail_tags")
            )
        ]

        weather_details = self._weather_details(
            weather_mode
        )

        seasonal_details = self._seasonal_details(
            season
        )

        combined = self._deduplicate(
            [
                *base_details,
                *weather_details,
                *seasonal_details,
            ]
        )

        if len(combined) > maximum:
            base_count = min(
                len(base_details),
                maximum,
            )

            selected = self._deduplicate(
                base_details[:base_count]
            )

            remaining = [
                detail
                for detail in combined
                if detail.lower()
                not in {
                    value.lower()
                    for value in selected
                }
            ]

            slots = maximum - len(selected)

            if slots > 0:
                selected.extend(
                    self.random.sample(
                        remaining,
                        k=min(slots, len(remaining)),
                    )
                )

            combined = selected

        fallback_details = [
            "phone",
            "camera strap",
            "handbag",
            "natural hair movement",
            "small personal belonging",
        ]

        for fallback in fallback_details:
            if len(combined) >= minimum:
                break

            if fallback.lower() not in {
                item.lower()
                for item in combined
            }:
                combined.append(fallback)

        return combined
    
    def resolve_scene(
        self,
        *,
        production_date: str,
        theme: str,
        stage: str,
        stage_data: dict[str, Any],
        weather_mode: str,
        season: str,
        used_behaviors: set[str],
        used_emotions: set[str],
        used_interactions: set[str],
        used_cameras: set[str],
        previous_motion_category: str | None = None,
    ) -> ResolvedScene:
        emotion_pool = [
            str(item)
            for item in self._as_list(
                stage_data.get("emotion_tags")
                or stage_data.get("emotion_pool")
            )
        ]

        interaction_pool = [
            str(item)
            for item in self._as_list(
                stage_data.get("interaction_tags")
                or stage_data.get("interaction_pool")
            )
        ]

        camera_pool = [
            str(item)
            for item in self._as_list(
                stage_data.get("camera_tags")
                or stage_data.get("camera_pool")
            )
        ]

        daily_details = self._resolve_details(
            stage_data,
            weather_mode=weather_mode,
            season=season,
        )

        motion = self.motion_engine.choose(
            production_date=production_date,
            stage=stage,
            activity=theme,
            used_today=used_behaviors,
            previous_category=previous_motion_category,
        )

        behavior_id = motion["id"]
        behavior_text = motion["text"]

        emotion_id = self._choose_id(
            emotion_pool,
            database=self._emotions(),
            excluded=used_emotions,
            label="emotion",
        )

        interaction_id = self._choose_id(
            interaction_pool,
            database=self._interactions(),
            excluded=used_interactions,
            label="interaction",
        )

        camera_id = self._choose_id(
            camera_pool,
            database=self._cameras(),
            excluded=used_cameras,
            label="camera",
        )

        used_behaviors.add(behavior_id)
        used_emotions.add(emotion_id)
        used_interactions.add(interaction_id)
        used_cameras.add(camera_id)

        return ResolvedScene(
            stage=stage,
            purpose=str(stage_data.get("purpose", "")),
            behavior_id=behavior_id,
            behavior_text=behavior_text,
            motion_category=motion["category"],
            emotion_id=emotion_id,
            emotion_text=self._resolve_emotion_text(emotion_id),
            interaction_id=interaction_id,
            interaction_text=self._resolve_interaction_text(
                interaction_id
            ),
            camera_id=camera_id,
            camera_text=self._resolve_camera_text(camera_id),
            daily_details=daily_details,
        )

    def build_theme_scenes(
        self,
        *,
        theme: str,
        weather_mode: str,
        season: str,
        production_date: str | None = None,
    ) -> dict[str, ResolvedScene]:
        effective_date = production_date or date.today().isoformat()
        theme_data = self._get_theme(theme)

        stage_mapping = theme_data.get("stages", {})

        if not isinstance(stage_mapping, dict) or not stage_mapping:
            raise StoryBuilderError(
                f"Theme '{theme}' must contain a non-empty 'stages' mapping"
            )

        stages = [
            self._normalise_key(str(stage))
            for stage in stage_mapping.keys()
            if str(stage).strip()
        ]

        used_behaviors: set[str] = set()
        used_emotions: set[str] = set()
        used_interactions: set[str] = set()
        used_cameras: set[str] = set()

        resolved: dict[str, ResolvedScene] = {}
        previous_motion_category: str | None = None

        for stage in stages:
            stage_data = stage_mapping.get(stage)

            if not isinstance(stage_data, dict):
                raise StoryBuilderError(
                    f"Missing stage '{stage}' in theme '{theme}'"
                )

            scene = self.resolve_scene(
                production_date=effective_date,
                theme=theme,
                stage=stage,
                stage_data=stage_data,
                weather_mode=weather_mode,
                season=season,
                used_behaviors=used_behaviors,
                used_emotions=used_emotions,
                used_interactions=used_interactions,
                used_cameras=used_cameras,
                previous_motion_category=previous_motion_category,
            )

            resolved[stage] = scene
            previous_motion_category = scene.motion_category

        self.motion_engine.remember_day(
            production_date=effective_date,
            motions=[
                {
                    "id": scene.behavior_id,
                    "text": scene.behavior_text,
                    "category": scene.motion_category,
                }
                for scene in resolved.values()
            ],
        )

        return resolved

    @staticmethod
    def _build_real_life(
        *,
        scene: ResolvedScene,
        location: str,
    ) -> str:
        return (
            f"At {location}, Aiko is actively "
            f"{scene.behavior_text.rstrip('.').lower()}. "
            f"The moment advances the {scene.stage.replace('_', ' ')} "
            "stage of the travel story."
        )

    @staticmethod
    def _build_body_motion(
        scene: ResolvedScene,
    ) -> str:
        return (
            f"Her hands, posture and eye movement follow the action: "
            f"{scene.behavior_text.rstrip('.').lower()}. "
            "Her hair, clothing and personal belongings move naturally "
            "with the activity."
        )

    def build_content_moment(
        self,
        *,
        content_id: str,
        content_type: str,
        title: str,
        location: str,
        scene: ResolvedScene,
        story_stage: str,
    ) -> ContentMoment:
        return ContentMoment(
            content_id=content_id,
            content_type=content_type,  # type: ignore[arg-type]
            title=title,
            location=location,
            real_life=self._build_real_life(
                scene=scene,
                location=location,
            ),
            behavior=scene.behavior_text,
            emotion=scene.emotion_text,
            interaction=scene.interaction_text,
            body_motion=self._build_body_motion(scene),
            camera_story=scene.camera_text,
            daily_details=scene.daily_details,
            story_stage=story_stage,
        )

    def build_reel_scene(
        self,
        *,
        scene_number: int,
        title: str,
        scene: ResolvedScene,
        duration_seconds: float = 2.5,
    ) -> ReelScene:
        return ReelScene(
            scene_number=scene_number,
            title=title,
            behavior=scene.behavior_text,
            emotion=scene.emotion_text,
            interaction=scene.interaction_text,
            camera_story=scene.camera_text,
            body_motion=self._build_body_motion(scene),
            daily_details=scene.daily_details,
            duration_seconds=duration_seconds,
        )