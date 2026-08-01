from __future__ import annotations

from .planning_loader import build_planning_loader
from .story_builder import StoryBuilder


def main() -> None:
    loader = build_planning_loader()

    errors = loader.validate_required_files()

    if errors:
        print("Planning database validation failed:")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    builder = StoryBuilder(
        loader=loader,
        random_seed=42,
    )

    scenes = builder.build_theme_scenes(
        theme="bookstore",
        weather_mode="rain",
        season="summer",
    )

    assert scenes
    assert "discovery" in scenes
    assert "arrival" in scenes
    assert "checkout" in scenes

    print("Story Builder test passed.")
    print()

    for stage, scene in scenes.items():
        print(f"Stage:       {stage}")
        print(f"Behavior:    {scene.behavior_id}")
        print(f"Emotion:     {scene.emotion_id}")
        print(f"Interaction: {scene.interaction_id}")
        print(f"Camera:      {scene.camera_id}")
        print(
            "Details:     "
            + ", ".join(scene.daily_details)
        )
        print()


if __name__ == "__main__":
    main()