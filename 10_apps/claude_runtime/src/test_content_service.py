from __future__ import annotations

from .content_service import build_service


def main() -> None:
    service = build_service()

    plan = service.generate(
        target_date="2026-07-27"
    )

    assert plan.validation["passed"] is True
    assert len(plan.stories) == 4
    assert len(plan.reel_scenes) == 5
    assert plan.feed.prompt
    assert plan.feed_caption
    assert len(plan.story_captions) == 4
    assert plan.reel_caption
    assert plan.hashtags

    print("Content Service test passed.")
    print(f"Theme: {plan.theme}")
    print(f"Feed: {plan.feed.title}")

    for story in plan.stories:
        print(
            f"Story: {story.title} "
            f"({story.behavior})"
        )


if __name__ == "__main__":
    main()