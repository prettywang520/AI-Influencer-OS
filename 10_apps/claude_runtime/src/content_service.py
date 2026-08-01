from __future__ import annotations

import argparse
from email import errors
import json
from datetime import date
from pathlib import Path

from .caption_engine import CaptionEngine
from .content_models import DailyContentPlan
from .content_validator import ContentValidator
from .dynamic_content_planner import DynamicContentPlanner
from .hashtag_engine import HashtagEngine
from .planner_selector import PlannerSelector
from .planning_history import PlanningHistory
from .planning_loader import PlanningLoader
from .prompt_engine import PromptEngine
from .reels_engine import ReelsEngine
from .story_builder import StoryBuilder


class ContentService:
    """
    AIKO Dynamic Content Service.

    Pipeline:

    Planning YAML
    -> PlanningLoader
    -> PlanningHistory
    -> PlannerSelector
    -> StoryBuilder
    -> DynamicContentPlanner
    -> Prompt Engine
    -> Caption Engine
    -> Hashtag Engine
    -> Reels Engine
    -> Content Validator
    -> Output
    """

    def __init__(
        self,
        *,
        output_root: str | Path,
        planner: DynamicContentPlanner,
        history: PlanningHistory,
    ) -> None:
        self.output_root = Path(
            output_root
        ).expanduser().resolve()

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.planner = planner
        self.history = history

        self.prompt_engine = PromptEngine()
        self.caption_engine = CaptionEngine()
        self.hashtag_engine = HashtagEngine()

        self.reels_engine = ReelsEngine(
            prompt_engine=self.prompt_engine
        )

        self.validator = ContentValidator()

    def generate(
        self,
        *,
        target_date: str | None = None,
        remember_plan: bool = True,
    ) -> DailyContentPlan:
        production_date = (
            target_date
            or date.today().isoformat()
        )

        # DynamicContentPlanner uses generate(), not create_plan().
        plan = self.planner.generate(
            target_date=production_date
        )

        # Feed prompt
        plan.feed.prompt = (
            self.prompt_engine.build_moment_prompt(
                plan.feed
            )
        )

        # Story prompts
        for story in plan.stories:
            story.prompt = (
                self.prompt_engine.build_moment_prompt(
                    story
                )
            )

        # Reel scene prompts
        self.reels_engine.apply(plan)

        # Captions
        self.caption_engine.apply(plan)

        # Hashtags
        self.hashtag_engine.apply(plan)

        # Final quality gate
        validation = self.validator.validate(plan)
        plan.validation = validation.to_dict()

        if not validation.passed:
            error_lines = "\n".join(
                validation.errors
            )

        gate_errors = validation.errors

        if not validation.passed:
            raise RuntimeError(
                "Content Quality Gate failed:\n"
                + "\n".join(validation.errors)
            )

        # Save output first.
        self.save(plan)

        # Store successful plan in anti-repeat history.
        if remember_plan:
            self.history.remember_plan(plan)

        return plan

    def save(
        self,
        plan: DailyContentPlan,
    ) -> Path:
        day_root = (
            self.output_root
            / plan.date
        )

        prompts_root = (
            day_root
            / "prompts"
        )

        captions_root = (
            day_root
            / "captions"
        )

        reels_root = (
            day_root
            / "reels"
        )

        prompts_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        captions_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        reels_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Feed prompt
        self._write_text(
            prompts_root / "feed_prompt.txt",
            plan.feed.prompt or "",
        )

        # Story prompts
        for index, story in enumerate(
            plan.stories,
            start=1,
        ):
            self._write_text(
                prompts_root
                / f"story_{index}_prompt.txt",
                story.prompt or "",
            )

        # Reel prompts
        for scene in plan.reel_scenes:
            self._write_text(
                reels_root
                / f"scene_{scene.scene_number}_prompt.txt",
                scene.prompt or "",
            )

        # Captions
        self._write_text(
            captions_root / "feed_caption.txt",
            plan.feed_caption or "",
        )

        story_caption_text = "\n\n".join(
            f"Story {index}\n{caption}"
            for index, caption in enumerate(
                plan.story_captions,
                start=1,
            )
        )

        self._write_text(
            captions_root / "story_captions.txt",
            story_caption_text,
        )

        self._write_text(
            captions_root / "reel_caption.txt",
            plan.reel_caption or "",
        )

        # Hashtags
        self._write_text(
            day_root / "hashtags.txt",
            " ".join(plan.hashtags),
        )

        # Full structured plan
        with (
            day_root / "content_plan.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                plan.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )

        self._write_dashboard(
            day_root=day_root,
            plan=plan,
        )

        return day_root

    @staticmethod
    def _write_text(
        path: Path,
        content: str,
    ) -> None:
        path.write_text(
            content.strip() + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_dashboard(
        *,
        day_root: Path,
        plan: DailyContentPlan,
    ) -> None:
        validation_status = (
            "PASSED"
            if plan.validation.get("passed")
            else "FAILED"
        )

        dashboard = f"""
# AIKO Daily Content

Date: {plan.date}

Country: {plan.country}

City: {plan.city}

Venue: {plan.venue}

Theme: {plan.theme}

## Story

{plan.story_summary}

## Output

- Feed prompts: 1
- Story prompts: {len(plan.stories)}
- Reel scenes: {len(plan.reel_scenes)}
- Feed caption: complete
- Story captions: {len(plan.story_captions)}
- Reel caption: complete
- Hashtags: {len(plan.hashtags)}

## Feed

- Title: {plan.feed.title}
- Behavior: {plan.feed.behavior}
- Emotion: {plan.feed.emotion}
- Interaction: {plan.feed.interaction}
- Camera: {plan.feed.camera_story}

## Quality Gate

Status: {validation_status}

Errors: {len(plan.validation.get("errors", []))}

Warnings: {len(plan.validation.get("warnings", []))}
""".strip()

        (
            day_root / "dashboard.md"
        ).write_text(
            dashboard + "\n",
            encoding="utf-8",
        )


def build_service() -> ContentService:
    """
    Build the complete Dynamic Content Service.
    """
    source_path = Path(__file__).resolve()

    runtime_root = source_path.parents[1]
    project_root = source_path.parents[3]

    planning_root = (
        project_root
        / "03_personas"
        / "aiko"
        / "planning"
    )

    output_root = (
        runtime_root
        / "output"
    )

    history_path = (
        output_root
        / "planning_history.json"
    )

    # 1. Load planning databases.
    loader = PlanningLoader(
        planning_root=planning_root
    )

    loader_errors = (
        loader.validate_required_files()
    )

    if loader_errors:
        raise RuntimeError(
            "Planning database validation failed:\n"
            + "\n".join(
                f"- {error}"
                for error in loader_errors
            )
        )

    # 2. Load anti-repeat history.
    history = PlanningHistory(
        history_path=history_path
    )

    # 3. Select date, city, venue and theme.
    selector = PlannerSelector(
        loader=loader,
        history=history,
    )

    # 4. Resolve behavior, emotion, interaction and camera.
    builder = StoryBuilder(
        loader=loader,
    )

    # 5. Build dynamic daily plan.
    planner = DynamicContentPlanner(
        selector=selector,
        builder=builder,
    )

    return ContentService(
        output_root=output_root,
        planner=planner,
        history=history,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Dynamic Content Service"
    )

    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Production date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Generate without writing planning history.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    service = build_service()

    plan = service.generate(
        target_date=arguments.date,
        remember_plan=not arguments.no_history,
    )

    print()
    print("AIKO Dynamic Content Service")
    print("----------------------------")
    print(f"date:          {plan.date}")
    print(f"country:       {plan.country}")
    print(f"city:          {plan.city}")
    print(f"venue:         {plan.venue}")
    print(f"theme:         {plan.theme}")
    print("feed:          1")
    print(f"stories:       {len(plan.stories)}")
    print(f"reel scenes:   {len(plan.reel_scenes)}")
    print(
        "quality gate:  "
        f"{'passed' if plan.validation.get('passed') else 'failed'}"
    )
    print(
        "history:       "
        f"{'saved' if not arguments.no_history else 'skipped'}"
    )
    print(
        "output:        "
        f"{service.output_root / plan.date}"
    )
    print()


if __name__ == "__main__":
    main()