from __future__ import annotations

from dataclasses import dataclass, field

from .content_models import DailyContentPlan


@dataclass(slots=True)
class ContentValidation:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ContentValidator:
    MINIMUM_DETAILS = 3

    def validate(
        self,
        plan: DailyContentPlan,
    ) -> ContentValidation:
        errors: list[str] = []
        warnings: list[str] = []

        moments = [
            plan.feed,
            *plan.stories,
        ]

        if len(plan.stories) != 4:
            errors.append("exactly_four_stories_required")

        if len(plan.reel_scenes) != 5:
            errors.append("exactly_five_reel_scenes_required")

        behaviors: list[str] = []
        cameras: list[str] = []
        emotions: list[str] = []
        interactions: list[str] = []

        for moment in moments:
            prefix = moment.content_id

            if not moment.real_life.strip():
                errors.append(f"{prefix}:missing_real_life")

            if not moment.behavior.strip():
                errors.append(f"{prefix}:missing_behavior")

            if not moment.emotion.strip():
                errors.append(f"{prefix}:missing_emotion")

            if not moment.interaction.strip():
                errors.append(f"{prefix}:missing_interaction")

            if not moment.camera_story.strip():
                errors.append(f"{prefix}:missing_camera_story")

            if len(moment.daily_details) < self.MINIMUM_DETAILS:
                errors.append(
                    f"{prefix}:minimum_three_daily_details_required"
                )

            forbidden_behaviors = {
                "standing",
                "posing",
                "smiling",
                "looking at camera",
            }

            if moment.behavior.strip().lower() in forbidden_behaviors:
                errors.append(
                    f"{prefix}:invalid_static_behavior"
                )

            behaviors.append(moment.behavior.lower())
            cameras.append(moment.camera_story.lower())
            emotions.append(moment.emotion.lower())
            interactions.append(moment.interaction.lower())

        if len(behaviors) != len(set(behaviors)):
            errors.append("duplicate_main_behavior_detected")

        if len(cameras) != len(set(cameras)):
            errors.append("duplicate_camera_story_detected")

        story_stages = [
            story.story_stage
            for story in plan.stories
        ]

        expected_stages = [
            "arrival",
            "discovery",
            "quiet_moment",
            "leaving",
        ]

        if story_stages != expected_stages:
            errors.append(
                "invalid_story_progression:"
                f"expected_{expected_stages}"
            )

        for scene in plan.reel_scenes:
            if len(scene.daily_details) < self.MINIMUM_DETAILS:
                errors.append(
                    f"reel_scene_{scene.scene_number}:"
                    "minimum_three_daily_details_required"
                )

            if not scene.interaction.strip():
                errors.append(
                    f"reel_scene_{scene.scene_number}:"
                    "missing_interaction"
                )

        if not plan.feed_caption:
            errors.append("missing_feed_caption")

        if len(plan.story_captions) != 4:
            errors.append("missing_story_captions")

        if not plan.reel_caption:
            errors.append("missing_reel_caption")

        if not plan.hashtags:
            warnings.append("hashtags_are_empty")

        return ContentValidation(
            passed=not errors,
            errors=errors,
            warnings=warnings,
        )