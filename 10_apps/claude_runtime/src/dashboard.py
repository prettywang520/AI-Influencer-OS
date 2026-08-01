from __future__ import annotations

from typing import Any

from .content_models import DailyContentPlan

# Was previously an empty, unused file. Populated for Phase 9 (Auto
# Daily Director) — nothing else in the codebase imported this
# module before, so this is purely additive.


def build_dashboard_markdown(
    *,
    plan: DailyContentPlan,
    wardrobe: dict[str, Any],
    threads_post: str,
    hashtags_by_channel: dict[str, list[str]],
    quality_gate: dict[str, Any],
    production_status: str,
) -> str:
    validation_status = "PASSED" if quality_gate.get("passed") else "FAILED"

    outfit_line = ", ".join(
        str(value)
        for key, value in wardrobe.items()
        if value and key not in ("city", "weather", "season", "venue", "activity")
    )

    lines = [
        "# AIKO Daily Content",
        "",
        f"Date: {plan.date}",
        f"Production status: {production_status}",
        "",
        f"Country: {plan.country}",
        f"City: {plan.city}",
        f"Venue: {plan.venue}",
        f"Theme: {plan.theme}",
        "",
        "## Story",
        "",
        plan.story_summary,
        "",
        "## Output",
        "",
        "- Feed prompts: 1",
        f"- Story prompts: {len(plan.stories)}",
        f"- Reel scenes: {len(plan.reel_scenes)}",
        "- Feed caption: complete",
        f"- Story captions: {len(plan.story_captions)}",
        "- Reel caption: complete",
        "- Threads post: complete",
        f"- Feed hashtags: {len(hashtags_by_channel.get('feed', []))}",
        f"- Reel hashtags: {len(hashtags_by_channel.get('reel', []))}",
        f"- Threads hashtags: {len(hashtags_by_channel.get('threads', []))}",
        "",
        "## Feed",
        "",
        f"- Title: {plan.feed.title}",
        f"- Behavior: {plan.feed.behavior}",
        f"- Emotion: {plan.feed.emotion}",
        f"- Interaction: {plan.feed.interaction}",
        f"- Camera: {plan.feed.camera_story}",
        f"- Caption: {plan.feed_caption}",
        "",
        "## Wardrobe",
        "",
        f"- {outfit_line}" if outfit_line else "- (no wardrobe selected)",
        "",
        "## Threads",
        "",
        threads_post,
        "",
        "## Quality Gate",
        "",
        f"Status: {validation_status}",
        f"Errors: {len(quality_gate.get('errors', []))}",
        f"Warnings: {len(quality_gate.get('warnings', []))}",
    ]

    if quality_gate.get("errors"):
        lines.append("")
        lines.append("### Errors")
        lines.append("")
        lines.extend(f"- {error}" for error in quality_gate["errors"])

    return "\n".join(lines).strip() + "\n"
