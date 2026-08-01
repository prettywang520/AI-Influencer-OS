from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .caption_engine import CaptionEngine
from .content_models import DailyContentPlan
from .content_validator import ContentValidator
from .dashboard import build_dashboard_markdown
from .daily_run_models import (
    DailyDirectorConfig,
    DailyDirectorImageTask,
    DailyDirectorManifest,
    DailyDirectorReelTask,
    RunLogEntry,
    StatusReport,
    WardrobeSelection,
    WardrobeSelector,
    build_wardrobe_selector,
    generate_threads_post,
    load_daily_director_config,
    reel_real_life,
    reel_stage_label,
    run_supplementary_quality_gate,
    write_json_atomic,
    write_text_atomic,
)
from .dynamic_content_planner import DynamicContentPlanner
from .hashtag_engine import HashtagEngine
from .planner_selector import DailySelection, PlannerSelector
from .planning_history import PlanningHistory
from .planning_loader import PlanningLoader
from .prompt_engine import PromptEngine
from .reels_engine import ReelsEngine
from .story_builder import StoryBuilder

# ---------------------------------------------------------------------------
# Phase 9 — AIKO Auto Daily Director
# ---------------------------------------------------------------------------
#
# Orchestrates the existing (working) production pipeline into one
# command that prepares Feed/Stories/Reels/captions/hashtags/queues/
# manifest/dashboard as LOCAL FILES ONLY.
#
# Never publishes anything. Never generates an image through an API.
# Never sends a comment or DM. Never touches src/social/, config/
# social/, output/community/, or any Instagram session/comment/DM
# code — nothing in this module imports from those at all.
#
# Composes the genuinely-reusable lower-level engines directly
# (PlannerSelector, StoryBuilder, DynamicContentPlanner, PromptEngine,
# ReelsEngine, ContentValidator, and the now-fixed CaptionEngine/
# HashtagEngine) rather than calling ContentService.generate()/
# ProductionService.run() as black boxes — those write a different,
# older file layout as a side effect, and this way
# content_service.py/production_service.py/daily.py's existing CLIs
# stay completely unmodified in behavior.

DISALLOWED_ACTIONS = (
    "publish",
    "post",
    "upload",
    "send_dm",
    "send_comment",
    "reply",
    "generate_image",
    "generate_video",
)


class AutoDailyDirectorError(RuntimeError):
    """Base error for the Auto Daily Director."""


class ExistingOutputError(AutoDailyDirectorError):
    """Raised when output already exists for a date and --force was not used."""


class QualityGateFailedError(AutoDailyDirectorError):
    """Raised when the combined quality gate does not pass."""


def _runtime_root() -> Path:
    """
    auto_daily_director.py location:

    10_apps/claude_runtime/src/auto_daily_director.py

    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


class _SelectionCapturingSelector:
    """
    Thin duck-typed wrapper around PlannerSelector that records the
    DailySelection from the most recent select() call.

    DailyContentPlan (content_models.py, not editable) does not carry
    weather_mode/daypart — fields only DailySelection has, and only
    DynamicContentPlanner.generate() calls selector.select()
    internally. This wrapper makes that selection observable
    afterward without calling select() a second time (which would
    burn extra PlannerSelector.random state and could pick a
    DIFFERENT venue/theme for rotation-fallback dates) and without
    modifying planner_selector.py or dynamic_content_planner.py.
    Delegates everything; introduces no new randomness or logic.
    """

    def __init__(self, selector: PlannerSelector) -> None:
        self._selector = selector
        self.last_selection: DailySelection | None = None

    def select(self, *, production_date: str) -> DailySelection:
        selection = self._selector.select(production_date=production_date)
        self.last_selection = selection
        return selection


class AutoDailyDirector:
    """
    One command, local files only:

    production date
    -> PlannerSelector (location/theme)
    -> StoryBuilder + MotionEngine (behavior/emotion/interaction/camera)
    -> DynamicContentPlanner (Feed x1, Stories x4, Reel scenes x5)
    -> PromptEngine (image/reel prompts)
    -> ReelsEngine (reel prompts)
    -> CaptionEngine (Feed/Story/Reel captions)
    -> HashtagEngine (per-channel hashtags)
    -> WardrobeSelector (one outfit/day, season + anti-repeat aware)
    -> Threads post generator
    -> ContentValidator + supplementary quality gate
    -> content_plan.json / production_manifest.json / dashboard.md /
       prompts/ captions/ hashtags/ queues/ images/
    """

    def __init__(
        self,
        *,
        config: DailyDirectorConfig,
        output_root: Path,
        history_dir: Path,
        logs_dir: Path,
        motion_history_path: Path,
        loader: PlanningLoader,
        history: PlanningHistory,
        capturing_selector: _SelectionCapturingSelector,
        planner: DynamicContentPlanner,
        prompt_engine: PromptEngine,
        reels_engine: ReelsEngine,
        caption_engine: CaptionEngine,
        hashtag_engine: HashtagEngine,
        validator: ContentValidator,
        wardrobe_selector: WardrobeSelector,
    ) -> None:
        self.config = config
        self.output_root = output_root
        self.history_dir = history_dir
        self.logs_dir = logs_dir
        # Always the REAL runtime output/history/motion.json — this is
        # where StoryBuilder's own internal MotionEngine unconditionally
        # writes (hardcoded in story_builder.py, not configurable), so
        # snapshot/restore for --dry-run / --no-history must target
        # this exact path regardless of any output_root override used
        # for THIS module's own artifacts (see build_auto_daily_director).
        self.motion_history_path = motion_history_path

        self.loader = loader
        self.history = history
        self.capturing_selector = capturing_selector
        self.planner = planner
        self.prompt_engine = prompt_engine
        self.reels_engine = reels_engine
        self.caption_engine = caption_engine
        self.hashtag_engine = hashtag_engine
        self.validator = validator
        self.wardrobe_selector = wardrobe_selector

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- plan generation (no file writes) -----------------------------------

    def _generate_plan(
        self,
        *,
        target_date: str,
        services_called: list[str],
    ) -> dict[str, Any]:
        """
        Builds the full plan + wardrobe + captions + hashtags +
        Threads post + combined quality gate result, entirely in
        memory. services_called is mutated in place so a caller can
        see progress even if this raises partway through.

        Note: StoryBuilder.build_theme_scenes() (called transitively
        via self.planner.generate()) unconditionally writes motion
        history to output/history/motion.json as a side effect —
        motion_engine.py/story_builder.py have no dry-run concept at
        all. Callers needing a true dry-run must snapshot/restore
        that specific file around this call (see run()).
        """
        services_called.append("PlannerSelector")
        plan = self.planner.generate(target_date=target_date)
        services_called.extend(
            ["StoryBuilder", "MotionEngine", "DynamicContentPlanner"]
        )

        selection = self.capturing_selector.last_selection

        plan.feed.prompt = self.prompt_engine.build_moment_prompt(plan.feed)

        for story in plan.stories:
            story.prompt = self.prompt_engine.build_moment_prompt(story)

        services_called.append("PromptEngine")

        self.reels_engine.apply(plan)
        services_called.append("ReelsEngine")

        self.caption_engine.apply(plan)
        services_called.append("CaptionEngine")

        self.hashtag_engine.apply(plan)
        services_called.append("HashtagEngine")

        hashtags_by_channel = {
            "feed": self.hashtag_engine.feed_hashtags(
                plan, maximum=self.config.hashtags.feed_count
            ),
            "reel": self.hashtag_engine.reel_hashtags(
                plan, maximum=self.config.hashtags.reel_count
            ),
            "threads": self.hashtag_engine.threads_hashtags(
                plan, maximum=self.config.hashtags.threads_count
            ),
        }

        rng = random.Random(f"{plan.date}:threads:{plan.venue}")

        threads_post = generate_threads_post(
            plan=plan,
            feed_caption=plan.feed_caption or "",
            style=self.config.threads_style,
            rng=rng,
        )
        services_called.append("ThreadsPostGenerator")

        wardrobe = self.wardrobe_selector.choose(
            production_date=target_date,
            city=plan.city,
            weather=selection.weather_mode if selection else "unspecified",
            season=selection.season if selection else "summer",
            venue=plan.venue,
            activity=plan.theme,
        )
        services_called.append("WardrobeSelector")

        base_validation = self.validator.validate(plan)
        services_called.append("ContentValidator")

        supplementary_errors, supplementary_warnings = (
            run_supplementary_quality_gate(plan=plan, wardrobe=wardrobe)
        )
        services_called.append("SupplementaryQualityGate")

        combined_errors = list(base_validation.errors) + supplementary_errors
        combined_warnings = list(base_validation.warnings) + supplementary_warnings

        quality_gate = {
            "passed": not combined_errors,
            "errors": combined_errors,
            "warnings": combined_warnings,
        }

        plan.validation = quality_gate

        return {
            "plan": plan,
            "selection": selection,
            "wardrobe": wardrobe,
            "hashtags_by_channel": hashtags_by_channel,
            "threads_post": threads_post,
            "quality_gate": quality_gate,
        }

    def _build_content_plan_payload(
        self,
        *,
        plan: DailyContentPlan,
        wardrobe: WardrobeSelection,
        threads_post: str,
        hashtags_by_channel: dict[str, list[str]],
        quality_gate: dict[str, Any],
    ) -> dict[str, Any]:
        wardrobe_dict = wardrobe.to_dict()

        feed_item = {**plan.feed.to_dict(), "wardrobe": wardrobe_dict}

        story_items = [
            {**story.to_dict(), "wardrobe": wardrobe_dict}
            for story in plan.stories
        ]

        location = f"{plan.venue}, {plan.city}, {plan.country}"

        reel_items = [
            {
                **scene.to_dict(),
                "wardrobe": wardrobe_dict,
                "story_stage": reel_stage_label(scene.scene_number),
                "real_life": reel_real_life(scene=scene, location=location),
            }
            for scene in plan.reel_scenes
        ]

        return {
            "date": plan.date,
            "country": plan.country,
            "city": plan.city,
            "venue": plan.venue,
            "theme": plan.theme,
            "story_summary": plan.story_summary,
            "feed": feed_item,
            "stories": story_items,
            "reel_scenes": reel_items,
            "feed_caption": plan.feed_caption,
            "story_captions": plan.story_captions,
            "reel_caption": plan.reel_caption,
            "threads_post": threads_post,
            "hashtags": hashtags_by_channel,
            "wardrobe": wardrobe_dict,
            "validation": quality_gate,
        }

    # -- file writing (staging directory -> atomic commit) -------------------

    def _write_output(
        self,
        *,
        staging_root: Path,
        final_root: Path,
        result: dict[str, Any],
        production_id: str,
        created_at: str,
    ) -> tuple[DailyDirectorManifest, list[str]]:
        plan: DailyContentPlan = result["plan"]
        wardrobe: WardrobeSelection = result["wardrobe"]
        hashtags_by_channel: dict[str, list[str]] = result["hashtags_by_channel"]
        threads_post: str = result["threads_post"]
        quality_gate: dict[str, Any] = result["quality_gate"]

        output_cfg = self.config.output
        files_written: list[str] = []

        def paths(relative: str) -> tuple[Path, Path]:
            return staging_root / relative, final_root / relative

        for relative in (
            output_cfg.prompts_dir,
            output_cfg.reel_prompts_dir,
            output_cfg.captions_dir,
            output_cfg.hashtags_dir,
            output_cfg.queues_dir,
            f"{output_cfg.images_dir}/feed",
            f"{output_cfg.images_dir}/stories",
        ):
            physical, _ = paths(relative)
            physical.mkdir(parents=True, exist_ok=True)

        def write_text(relative: str, content: str) -> str:
            physical, logical = paths(relative)
            write_text_atomic(physical, (content or "").strip() + "\n")
            files_written.append(str(logical))
            return str(logical)

        def write_json(relative: str, payload: dict[str, Any]) -> str:
            physical, logical = paths(relative)
            write_json_atomic(physical, payload)
            files_written.append(str(logical))
            return str(logical)

        write_text(
            f"{output_cfg.prompts_dir}/{output_cfg.feed_prompt_file}",
            plan.feed.prompt or "",
        )

        for index, story in enumerate(plan.stories, start=1):
            write_text(
                f"{output_cfg.prompts_dir}/"
                f"{output_cfg.story_prompt_template.format(index=index)}",
                story.prompt or "",
            )

        for scene in plan.reel_scenes:
            write_text(
                f"{output_cfg.reel_prompts_dir}/"
                f"{output_cfg.reel_prompt_template.format(index=scene.scene_number)}",
                scene.prompt or "",
            )

        write_text(
            f"{output_cfg.captions_dir}/{output_cfg.feed_caption_file}",
            plan.feed_caption or "",
        )

        for index, caption in enumerate(plan.story_captions, start=1):
            write_text(
                f"{output_cfg.captions_dir}/"
                f"{output_cfg.story_caption_template.format(index=index)}",
                caption,
            )

        write_text(
            f"{output_cfg.captions_dir}/{output_cfg.reel_caption_file}",
            plan.reel_caption or "",
        )

        write_text(
            f"{output_cfg.captions_dir}/{output_cfg.threads_post_file}",
            threads_post,
        )

        write_text(
            f"{output_cfg.hashtags_dir}/{output_cfg.feed_hashtags_file}",
            " ".join(hashtags_by_channel["feed"]),
        )
        write_text(
            f"{output_cfg.hashtags_dir}/{output_cfg.reel_hashtags_file}",
            " ".join(hashtags_by_channel["reel"]),
        )
        write_text(
            f"{output_cfg.hashtags_dir}/{output_cfg.threads_hashtags_file}",
            " ".join(hashtags_by_channel["threads"]),
        )

        image_tasks: list[DailyDirectorImageTask] = []

        _, feed_prompt_logical = paths(
            f"{output_cfg.prompts_dir}/{output_cfg.feed_prompt_file}"
        )
        _, feed_image_logical = paths(f"{output_cfg.images_dir}/feed/feed_01.png")

        image_tasks.append(
            DailyDirectorImageTask(
                task_id=f"{plan.date}-image-feed-01",
                production_date=plan.date,
                content_id=plan.feed.content_id,
                content_type="feed",
                title=plan.feed.title,
                prompt_file=str(feed_prompt_logical),
                expected_output_file=str(feed_image_logical),
            )
        )

        for index, story in enumerate(plan.stories, start=1):
            _, prompt_logical = paths(
                f"{output_cfg.prompts_dir}/"
                f"{output_cfg.story_prompt_template.format(index=index)}"
            )
            _, image_logical = paths(
                f"{output_cfg.images_dir}/stories/story_{index:02d}.png"
            )

            image_tasks.append(
                DailyDirectorImageTask(
                    task_id=f"{plan.date}-image-story-{index:02d}",
                    production_date=plan.date,
                    content_id=story.content_id,
                    content_type="story",
                    title=story.title,
                    prompt_file=str(prompt_logical),
                    expected_output_file=str(image_logical),
                )
            )

        reel_tasks: list[DailyDirectorReelTask] = []

        for scene in plan.reel_scenes:
            _, prompt_logical = paths(
                f"{output_cfg.reel_prompts_dir}/"
                f"{output_cfg.reel_prompt_template.format(index=scene.scene_number)}"
            )
            _, video_logical = paths(
                f"{output_cfg.images_dir}/reels/shot_{scene.scene_number:02d}.mp4"
            )

            reel_tasks.append(
                DailyDirectorReelTask(
                    task_id=f"{plan.date}-reel-shot-{scene.scene_number:02d}",
                    production_date=plan.date,
                    scene_number=scene.scene_number,
                    title=scene.title,
                    duration_seconds=scene.duration_seconds,
                    prompt_file=str(prompt_logical),
                    expected_output_file=str(video_logical),
                )
            )

        write_json(
            f"{output_cfg.queues_dir}/{output_cfg.image_queue_file}",
            {
                "version": "1.0",
                "provider": "chatgpt_manual",
                "status": "pending",
                "task_count": len(image_tasks),
                "tasks": [task.to_dict() for task in image_tasks],
            },
        )

        write_json(
            f"{output_cfg.queues_dir}/{output_cfg.reel_queue_file}",
            {
                "version": "1.0",
                "provider": "runway_manual",
                "status": "pending",
                "task_count": len(reel_tasks),
                "tasks": [task.to_dict() for task in reel_tasks],
            },
        )

        content_plan_payload = self._build_content_plan_payload(
            plan=plan,
            wardrobe=wardrobe,
            threads_post=threads_post,
            hashtags_by_channel=hashtags_by_channel,
            quality_gate=quality_gate,
        )
        write_json(output_cfg.content_plan_file, content_plan_payload)

        manifest = DailyDirectorManifest(
            production_id=production_id,
            production_date=plan.date,
            created_at=created_at,
            country=plan.country,
            city=plan.city,
            venue=plan.venue,
            theme=plan.theme,
            story_summary=plan.story_summary,
            status="ready_for_images" if quality_gate["passed"] else "failed",
            feed_count=1,
            story_count=len(plan.stories),
            reel_scene_count=len(plan.reel_scenes),
            image_tasks=image_tasks,
            reel_tasks=reel_tasks,
            hashtags=hashtags_by_channel,
            quality_gate=quality_gate,
            wardrobe=wardrobe.to_dict(),
            output_directory=str(final_root),
        )

        write_json(output_cfg.production_manifest_file, manifest.to_dict())

        dashboard_markdown = build_dashboard_markdown(
            plan=plan,
            wardrobe=wardrobe.to_dict(),
            threads_post=threads_post,
            hashtags_by_channel=hashtags_by_channel,
            quality_gate=quality_gate,
            production_status=manifest.status,
        )
        write_text(output_cfg.dashboard_file, dashboard_markdown)

        return manifest, files_written

    @staticmethod
    def _commit_staging(*, staging_root: Path, day_root: Path) -> None:
        """
        Atomic-as-possible directory swap: the new content is fully
        written to staging_root first (every file inside it already
        individually atomic via write_text_atomic/write_json_atomic),
        THEN swapped into place. On --force over an existing day_root,
        the old directory is moved aside first and only removed after
        the new one is successfully in place, so a crash between the
        two renames still leaves a valid (old or new) directory rather
        than a partial one, and a failed commit restores the old
        directory rather than losing it.
        """
        if day_root.exists():
            backup_root = day_root.with_name(
                f".backup-{day_root.name}-{uuid4().hex[:8]}"
            )
            day_root.rename(backup_root)

            try:
                staging_root.rename(day_root)
            except Exception:
                backup_root.rename(day_root)
                raise

            shutil.rmtree(backup_root, ignore_errors=True)
        else:
            staging_root.rename(day_root)

    # -- history side-effect isolation (dry-run / --no-history) --------------

    @staticmethod
    def _snapshot(path: Path) -> bytes | None:
        if not path.exists():
            return None

        return path.read_bytes()

    @staticmethod
    def _restore(path: Path, snapshot: bytes | None) -> None:
        if snapshot is None:
            if path.exists():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(snapshot)

    # -- logging ----------------------------------------------------------

    def _log_path(self, production_date: str) -> Path:
        filename = self.config.output.log_filename_template.format(
            date=production_date
        )
        return self.logs_dir / filename

    def _write_log(self, entry: RunLogEntry) -> None:
        write_json_atomic(self._log_path(entry.date), entry.to_dict())

    # -- orchestration --------------------------------------------------

    def run(
        self,
        *,
        target_date: str,
        force: bool = False,
        no_history: bool = False,
        dry_run: bool = False,
    ) -> DailyDirectorManifest:
        date.fromisoformat(target_date)  # raises ValueError on bad format

        started_at = self._now()
        log_entry = RunLogEntry(started_at=started_at, date=target_date)

        day_root = self.output_root / target_date
        manifest_marker = day_root / self.config.output.production_manifest_file

        if not dry_run and manifest_marker.exists() and not force:
            log_entry.result = "refused_existing"
            log_entry.error = (
                f"Output already exists for {target_date} and --force was "
                "not used"
            )
            log_entry.finished_at = self._now()
            self._write_log(log_entry)

            raise ExistingOutputError(
                f"Output already exists for {target_date}: {day_root}. "
                "Use --force to rebuild it, or --status to inspect it."
            )

        skip_history = dry_run or no_history
        motion_snapshot = (
            self._snapshot(self.motion_history_path) if skip_history else None
        )

        services_called: list[str] = []

        try:
            result = self._generate_plan(
                target_date=target_date,
                services_called=services_called,
            )

            log_entry.services_called = list(services_called)
            log_entry.quality_gate = result["quality_gate"]
            log_entry.selection = (
                result["selection"].to_dict() if result["selection"] else {}
            )

            if not result["quality_gate"]["passed"]:
                log_entry.result = "failed"
                log_entry.error = "quality_gate_failed: " + "; ".join(
                    result["quality_gate"]["errors"]
                )
                log_entry.finished_at = self._now()
                self._write_log(log_entry)

                raise QualityGateFailedError(
                    "Quality gate failed for "
                    f"{target_date}:\n"
                    + "\n".join(result["quality_gate"]["errors"])
                )

            if dry_run:
                log_entry.result = "dry_run"
                log_entry.finished_at = self._now()
                self._write_log(log_entry)

                return self._summary_manifest(result, status="dry_run")

            production_id = f"aiko-daily-{target_date}"
            created_at = self._now()

            staging_root = (
                self.output_root / f".building-{target_date}-{uuid4().hex[:8]}"
            )
            staging_root.mkdir(parents=True, exist_ok=False)

            try:
                manifest, files_written = self._write_output(
                    staging_root=staging_root,
                    final_root=day_root,
                    result=result,
                    production_id=production_id,
                    created_at=created_at,
                )
            except Exception:
                shutil.rmtree(staging_root, ignore_errors=True)
                raise

            self._commit_staging(staging_root=staging_root, day_root=day_root)

            if not no_history:
                self.history.remember_plan(result["plan"])
                self.wardrobe_selector.save(
                    production_date=target_date,
                    outfit=result["wardrobe"],
                )
                services_called.append("PlanningHistory")
                services_called.append("WardrobeSelector.save")

            log_entry.services_called = list(services_called)
            log_entry.files_written = files_written
            log_entry.result = "success"
            log_entry.finished_at = self._now()
            self._write_log(log_entry)

            return manifest

        except (ExistingOutputError, QualityGateFailedError):
            raise
        except Exception as exc:
            log_entry.services_called = list(services_called)
            log_entry.result = "failed"
            log_entry.error = f"{type(exc).__name__}: {exc}"
            log_entry.finished_at = self._now()
            self._write_log(log_entry)
            raise
        finally:
            if skip_history:
                self._restore(self.motion_history_path, motion_snapshot)

    def _summary_manifest(
        self,
        result: dict[str, Any],
        *,
        status: str,
    ) -> DailyDirectorManifest:
        plan: DailyContentPlan = result["plan"]
        wardrobe: WardrobeSelection = result["wardrobe"]

        return DailyDirectorManifest(
            production_id=f"aiko-daily-{plan.date}",
            production_date=plan.date,
            created_at=self._now(),
            country=plan.country,
            city=plan.city,
            venue=plan.venue,
            theme=plan.theme,
            story_summary=plan.story_summary,
            status=status,
            feed_count=1,
            story_count=len(plan.stories),
            reel_scene_count=len(plan.reel_scenes),
            image_tasks=[],
            reel_tasks=[],
            hashtags=result["hashtags_by_channel"],
            quality_gate=result["quality_gate"],
            wardrobe=wardrobe.to_dict(),
            output_directory="",
        )

    # -- status -------------------------------------------------------

    def _expected_relative_files(self) -> list[str]:
        cfg = self.config.output

        return [
            cfg.content_plan_file,
            cfg.production_manifest_file,
            cfg.dashboard_file,
            f"{cfg.prompts_dir}/{cfg.feed_prompt_file}",
            *[
                f"{cfg.prompts_dir}/{cfg.story_prompt_template.format(index=i)}"
                for i in range(1, 5)
            ],
            *[
                f"{cfg.reel_prompts_dir}/"
                f"{cfg.reel_prompt_template.format(index=i)}"
                for i in range(1, 6)
            ],
            f"{cfg.captions_dir}/{cfg.feed_caption_file}",
            *[
                f"{cfg.captions_dir}/"
                f"{cfg.story_caption_template.format(index=i)}"
                for i in range(1, 5)
            ],
            f"{cfg.captions_dir}/{cfg.reel_caption_file}",
            f"{cfg.captions_dir}/{cfg.threads_post_file}",
            f"{cfg.hashtags_dir}/{cfg.feed_hashtags_file}",
            f"{cfg.hashtags_dir}/{cfg.reel_hashtags_file}",
            f"{cfg.hashtags_dir}/{cfg.threads_hashtags_file}",
            f"{cfg.queues_dir}/{cfg.image_queue_file}",
            f"{cfg.queues_dir}/{cfg.reel_queue_file}",
        ]

    def status(self, *, target_date: str) -> StatusReport:
        day_root = self.output_root / target_date
        manifest_path = day_root / self.config.output.production_manifest_file

        expected = self._expected_relative_files()

        if not manifest_path.exists():
            return StatusReport(
                production_date=target_date,
                exists=False,
                production_status="not_created",
                quality_gate_status="not_created",
                image_task_count=0,
                reel_task_count=0,
                files_present=[],
                files_missing=expected,
                next_recommended_command=(
                    f"python3 -u -m src.auto_daily_director --date {target_date}"
                ),
            )

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

        present = [rel for rel in expected if (day_root / rel).exists()]
        missing = [rel for rel in expected if not (day_root / rel).exists()]

        quality_gate = data.get("quality_gate", {}) or {}
        gate_passed = bool(quality_gate.get("passed"))

        next_command = (
            f"python3 -u -m src.production_worker --date {target_date} --next"
            if gate_passed and not missing
            else f"python3 -u -m src.auto_daily_director --date {target_date} --force"
        )

        return StatusReport(
            production_date=target_date,
            exists=True,
            production_status=str(data.get("status", "unknown")),
            quality_gate_status="passed" if gate_passed else "failed",
            image_task_count=len(data.get("image_tasks", [])),
            reel_task_count=len(data.get("reel_tasks", [])),
            files_present=present,
            files_missing=missing,
            next_recommended_command=next_command,
        )


def build_auto_daily_director(
    *,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    random_seed: int | None = None,
) -> AutoDailyDirector:
    """
    Build the fully-wired Auto Daily Director.

    output_root overrides where THIS module's own new artifacts go
    (content_plan.json/production_manifest.json/dashboard.md/prompts/
    captions/hashtags/queues/images under output_root/<date>/, plus
    output_root/history/wardrobe.json and output_root/logs/). Existing
    shared state — output/planning_history.json (read by PlanningHistory)
    and output/history/motion.json (written by StoryBuilder's own
    internal MotionEngine, hardcoded, not configurable) — always uses
    the REAL runtime output/ directory regardless of this override,
    since story_builder.py is not in Phase 9's editable file list.
    """
    source_path = Path(__file__).resolve()
    runtime_root = source_path.parents[1]
    project_root = source_path.parents[3]

    planning_root = project_root / "03_personas" / "aiko" / "planning"
    real_output_root = runtime_root / "output"

    resolved_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else real_output_root
    )

    config = load_daily_director_config(config_path=config_path)

    loader = PlanningLoader(planning_root=planning_root)
    loader_errors = loader.validate_required_files()

    if loader_errors:
        raise AutoDailyDirectorError(
            "Planning database validation failed:\n"
            + "\n".join(f"- {error}" for error in loader_errors)
        )

    history = PlanningHistory(
        history_path=real_output_root / "planning_history.json"
    )
    selector = PlannerSelector(loader=loader, history=history)
    capturing_selector = _SelectionCapturingSelector(selector)
    builder = StoryBuilder(loader=loader)
    planner = DynamicContentPlanner(
        selector=capturing_selector,  # type: ignore[arg-type]
        builder=builder,
    )

    prompt_engine = PromptEngine()
    reels_engine = ReelsEngine(prompt_engine=prompt_engine)
    caption_engine = CaptionEngine()
    hashtag_engine = HashtagEngine()
    validator = ContentValidator()

    wardrobe_selector = build_wardrobe_selector(
        config=config,
        random_seed=random_seed,
    )

    if output_root is not None:
        # Test isolation: point the wardrobe history file at the
        # overridden output root too (motion.json cannot be
        # redirected — see the docstring above).
        wardrobe_selector.history_path = (
            resolved_output_root
            / "history"
            / config.output.wardrobe_history_file
        )

    return AutoDailyDirector(
        config=config,
        output_root=resolved_output_root,
        history_dir=resolved_output_root / "history",
        logs_dir=resolved_output_root / "logs",
        motion_history_path=real_output_root / "history" / "motion.json",
        loader=loader,
        history=history,
        capturing_selector=capturing_selector,
        planner=planner,
        prompt_engine=prompt_engine,
        reels_engine=reels_engine,
        caption_engine=caption_engine,
        hashtag_engine=hashtag_engine,
        validator=validator,
        wardrobe_selector=wardrobe_selector,
    )


def _resolve_date(arguments: argparse.Namespace) -> str:
    if arguments.today:
        return date.today().isoformat()

    return arguments.date or date.today().isoformat()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Auto Daily Director (Phase 9) — local files only."
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Production date in YYYY-MM-DD format (default: today).",
    )

    parser.add_argument(
        "--today",
        action="store_true",
        help="Use today's date.",
    )

    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not update planning/motion/wardrobe history.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild this date's output even if it already exists.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show production status for this date instead of generating.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate in memory only; write nothing.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate daily_director.yaml config file.",
    )

    return parser.parse_args(argv)


def _print_manifest(manifest: DailyDirectorManifest, *, dry_run: bool) -> None:
    title = "AIKO Auto Daily Director" + (" — DRY RUN" if dry_run else "")

    print()
    print(title)
    print("-" * len(title))
    print(f"date:          {manifest.production_date}")
    print(f"country:       {manifest.country}")
    print(f"city:          {manifest.city}")
    print(f"venue:         {manifest.venue}")
    print(f"theme:         {manifest.theme}")
    print(f"status:        {manifest.status}")
    print("feed:          1")
    print(f"stories:       {manifest.story_count}")
    print(f"reel scenes:   {manifest.reel_scene_count}")
    print(f"image tasks:   {len(manifest.image_tasks)}")
    print(f"reel tasks:    {len(manifest.reel_tasks)}")
    print(
        "quality gate:  "
        f"{'passed' if manifest.quality_gate.get('passed') else 'failed'}"
    )

    if manifest.quality_gate.get("errors"):
        print("gate errors:")

        for error in manifest.quality_gate["errors"]:
            print(f"  - {error}")

    if not dry_run:
        print(f"output:        {manifest.output_directory}")

    print()


def _print_status(status: StatusReport) -> None:
    print()
    print("AIKO Auto Daily Director — Status")
    print("------------------------------------")
    print(f"date:                {status.production_date}")
    print(f"exists:              {status.exists}")
    print(f"production status:   {status.production_status}")
    print(f"quality gate status: {status.quality_gate_status}")
    print(f"image tasks:         {status.image_task_count}")
    print(f"reel tasks:          {status.reel_task_count}")
    print(f"files present:       {len(status.files_present)}")
    print(f"files missing:       {len(status.files_missing)}")

    for missing in status.files_missing:
        print(f"  missing: {missing}")

    print(f"next command:        {status.next_recommended_command}")
    print()


def _run(arguments: argparse.Namespace) -> int:
    target_date = _resolve_date(arguments)
    director = build_auto_daily_director(config_path=arguments.config)

    if arguments.status:
        _print_status(director.status(target_date=target_date))
        return 0

    try:
        manifest = director.run(
            target_date=target_date,
            force=arguments.force,
            no_history=arguments.no_history,
            dry_run=arguments.dry_run,
        )
    except ExistingOutputError as exc:
        print(f"[AutoDailyDirector] {exc}")
        _print_status(director.status(target_date=target_date))
        return 1
    except Exception as exc:
        print(f"[AutoDailyDirector] failed ({type(exc).__name__}): {exc}")
        return 1

    _print_manifest(manifest, dry_run=arguments.dry_run)
    return 0


def main() -> None:
    arguments = parse_arguments()
    exit_code = _run(arguments)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
