from __future__ import annotations

from .content_models import DailyContentPlan
from .prompt_engine import PromptEngine


class ReelsEngine:
    def __init__(
        self,
        prompt_engine: PromptEngine,
    ) -> None:
        self.prompt_engine = prompt_engine

    def apply(self, plan: DailyContentPlan) -> None:
        location = (
            f"{plan.venue}, {plan.city}, {plan.country}"
        )

        for scene in plan.reel_scenes:
            scene.prompt = (
                self.prompt_engine.build_reel_scene_prompt(
                    scene=scene,
                    location=location,
                )
            )