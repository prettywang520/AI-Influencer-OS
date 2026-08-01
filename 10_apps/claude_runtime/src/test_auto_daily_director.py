from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .auto_daily_director import (
    AutoDailyDirector,
    ExistingOutputError,
    QualityGateFailedError,
    build_auto_daily_director,
)
from .content_validator import ContentValidation

# None of these tests access Instagram or any external API. Planning
# data is read from the real, local 03_personas/aiko/planning/*.yaml
# files (matching the existing project-wide test convention for the
# non-social pipeline — e.g. test_content_service.py,
# test_dynamic_content_planner.py); only OUTPUT is redirected to a
# temp directory. The one real, hardcoded shared-state file this
# pipeline cannot redirect — output/history/motion.json, written
# internally by StoryBuilder's own MotionEngine — is snapshotted
# before and restored after every test that performs a real
# (non-dry-run, non-no-history) run, so the real file is never left
# modified by this suite.

CALENDAR_DATE = "2026-08-02"  # theme=sightseeing, confirmed valid
OTHER_CALENDAR_DATE = "2026-07-27"  # theme=bookstore, confirmed valid


class _AlwaysFailingValidation:
    passed = False
    errors = ["forced_failure_for_test"]
    warnings: list[str] = []

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class _AlwaysFailingValidator:
    def validate(self, plan) -> _AlwaysFailingValidation:
        return _AlwaysFailingValidation()


class DailyDirectorTestCase(unittest.TestCase):
    """
    Base class: builds a director pointed at a fresh temp output
    root, and always protects the real output/history/motion.json
    (see module docstring).
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name)

        self.real_motion_history_path = (
            Path(__file__).resolve().parents[1]
            / "output"
            / "history"
            / "motion.json"
        )
        self._motion_snapshot = (
            self.real_motion_history_path.read_bytes()
            if self.real_motion_history_path.exists()
            else None
        )

        self.director = build_auto_daily_director(
            output_root=self.output_root, random_seed=12345
        )

    def tearDown(self) -> None:
        if self._motion_snapshot is None:
            if self.real_motion_history_path.exists():
                self.real_motion_history_path.unlink()
        else:
            self.real_motion_history_path.write_bytes(self._motion_snapshot)

        self.temp_dir.cleanup()

    def _day_root(self, target_date: str) -> Path:
        return self.output_root / target_date


# ---------------------------------------------------------------------------
# 1 & 2 — normal run creates the complete structure with exact counts
# ---------------------------------------------------------------------------


class NormalRunStructureTests(DailyDirectorTestCase):
    def test_complete_required_structure_is_created(self) -> None:
        manifest = self.director.run(target_date=CALENDAR_DATE, no_history=True)

        self.assertEqual(manifest.status, "ready_for_images")
        day_root = self._day_root(CALENDAR_DATE)

        expected_files = [
            "content_plan.json",
            "production_manifest.json",
            "dashboard.md",
            "prompts/feed_prompt.txt",
            "prompts/story_1_prompt.txt",
            "prompts/story_2_prompt.txt",
            "prompts/story_3_prompt.txt",
            "prompts/story_4_prompt.txt",
            "prompts/reels/shot_01_prompt.txt",
            "prompts/reels/shot_02_prompt.txt",
            "prompts/reels/shot_03_prompt.txt",
            "prompts/reels/shot_04_prompt.txt",
            "prompts/reels/shot_05_prompt.txt",
            "captions/feed_caption.txt",
            "captions/story_1_caption.txt",
            "captions/story_2_caption.txt",
            "captions/story_3_caption.txt",
            "captions/story_4_caption.txt",
            "captions/reel_caption.txt",
            "captions/threads_post.txt",
            "hashtags/feed_hashtags.txt",
            "hashtags/reel_hashtags.txt",
            "hashtags/threads_hashtags.txt",
            "queues/image_queue.json",
            "queues/reel_queue.json",
        ]

        for relative in expected_files:
            self.assertTrue(
                (day_root / relative).exists(), f"missing: {relative}"
            )

        self.assertTrue((day_root / "images" / "feed").is_dir())
        self.assertTrue((day_root / "images" / "stories").is_dir())

    def test_exact_feed_story_reel_counts(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        content_plan = json.loads(
            (self._day_root(CALENDAR_DATE) / "content_plan.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertIsInstance(content_plan["feed"], dict)
        self.assertEqual(len(content_plan["stories"]), 4)
        self.assertEqual(len(content_plan["reel_scenes"]), 5)

        manifest = json.loads(
            (
                self._day_root(CALENDAR_DATE) / "production_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["feed_count"], 1)
        self.assertEqual(manifest["story_count"], 4)
        self.assertEqual(manifest["reel_scene_count"], 5)
        self.assertEqual(len(manifest["image_tasks"]), 5)
        self.assertEqual(len(manifest["reel_tasks"]), 5)

    def test_every_visual_item_contains_required_fields(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        content_plan = json.loads(
            (self._day_root(CALENDAR_DATE) / "content_plan.json").read_text(
                encoding="utf-8"
            )
        )

        required_moment_fields = {
            "real_life",
            "behavior",
            "body_motion",
            "emotion",
            "interaction",
            "camera_story",
            "daily_details",
            "wardrobe",
            "story_stage",
        }

        for item in [content_plan["feed"], *content_plan["stories"]]:
            self.assertTrue(required_moment_fields.issubset(item.keys()))
            self.assertTrue(item["wardrobe"])

        for item in content_plan["reel_scenes"]:
            self.assertTrue(required_moment_fields.issubset(item.keys()))
            self.assertTrue(item["wardrobe"])

        reel_stages = [
            item["story_stage"] for item in content_plan["reel_scenes"]
        ]
        self.assertEqual(
            reel_stages,
            ["beginning", "development", "interaction", "emotional_beat", "ending"],
        )


# ---------------------------------------------------------------------------
# 3 — dry-run writes nothing
# ---------------------------------------------------------------------------


class DryRunTests(DailyDirectorTestCase):
    def test_dry_run_writes_no_output_files(self) -> None:
        manifest = self.director.run(target_date=CALENDAR_DATE, dry_run=True)

        self.assertEqual(manifest.status, "dry_run")
        day_root = self._day_root(CALENDAR_DATE)
        self.assertFalse(day_root.exists())

    def test_dry_run_does_not_update_motion_history(self) -> None:
        before = (
            self.real_motion_history_path.read_bytes()
            if self.real_motion_history_path.exists()
            else None
        )

        self.director.run(target_date=CALENDAR_DATE, dry_run=True)

        after = (
            self.real_motion_history_path.read_bytes()
            if self.real_motion_history_path.exists()
            else None
        )

        self.assertEqual(before, after)

    def test_dry_run_leaves_no_staging_directories(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, dry_run=True)

        leftover = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".building-")
        ]
        self.assertEqual(leftover, [])


# ---------------------------------------------------------------------------
# 4 & 5 — idempotency and --force
# ---------------------------------------------------------------------------


class IdempotencyTests(DailyDirectorTestCase):
    def test_refuses_overwrite_without_force(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        with self.assertRaises(ExistingOutputError):
            self.director.run(target_date=CALENDAR_DATE, no_history=True)

    def test_force_rebuilds_only_the_requested_date(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)
        self.director.run(target_date=OTHER_CALENDAR_DATE, no_history=True)

        other_manifest_path = (
            self._day_root(OTHER_CALENDAR_DATE) / "production_manifest.json"
        )
        other_before = other_manifest_path.read_bytes()

        self.director.run(
            target_date=CALENDAR_DATE, no_history=True, force=True
        )

        other_after = other_manifest_path.read_bytes()

        self.assertEqual(other_before, other_after)
        self.assertTrue(self._day_root(CALENDAR_DATE).exists())
        self.assertTrue(self._day_root(OTHER_CALENDAR_DATE).exists())


# ---------------------------------------------------------------------------
# 6 — status reports missing files correctly
# ---------------------------------------------------------------------------


class StatusReportTests(DailyDirectorTestCase):
    def test_status_for_nonexistent_date(self) -> None:
        status = self.director.status(target_date="2099-01-01")

        self.assertFalse(status.exists)
        self.assertEqual(status.production_status, "not_created")
        self.assertGreater(len(status.files_missing), 0)
        self.assertEqual(status.files_present, [])

    def test_status_reports_missing_file_after_manual_deletion(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        missing_target = (
            self._day_root(CALENDAR_DATE) / "hashtags" / "threads_hashtags.txt"
        )
        missing_target.unlink()

        status = self.director.status(target_date=CALENDAR_DATE)

        self.assertTrue(status.exists)
        self.assertIn("hashtags/threads_hashtags.txt", status.files_missing)
        self.assertNotIn(
            "hashtags/threads_hashtags.txt", status.files_present
        )

    def test_status_reports_complete_run_with_next_command(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        status = self.director.status(target_date=CALENDAR_DATE)

        self.assertEqual(status.files_missing, [])
        self.assertEqual(status.quality_gate_status, "passed")
        self.assertIn("production_worker", status.next_recommended_command)


# ---------------------------------------------------------------------------
# 7 — quality gate failure exits non-zero (raises, in library terms)
# ---------------------------------------------------------------------------


class QualityGateFailureTests(DailyDirectorTestCase):
    def _director_with_failing_gate(self) -> AutoDailyDirector:
        # Reuse the same fully-wired director, only swap the reused
        # ContentValidator for one that always fails — proves the
        # combined quality gate result (base + supplementary) is
        # correctly propagated into a raised, logged failure.
        self.director.validator = _AlwaysFailingValidator()
        return self.director

    def test_failing_quality_gate_raises_and_writes_no_output(self) -> None:
        director = self._director_with_failing_gate()

        with self.assertRaises(QualityGateFailedError):
            director.run(target_date=CALENDAR_DATE, no_history=True)

        self.assertFalse(self._day_root(CALENDAR_DATE).exists())

    def test_failing_quality_gate_is_logged(self) -> None:
        director = self._director_with_failing_gate()

        with self.assertRaises(QualityGateFailedError):
            director.run(target_date=CALENDAR_DATE, no_history=True)

        log_path = (
            self.output_root / "logs" / f"auto_daily_director_{CALENDAR_DATE}.json"
        )
        self.assertTrue(log_path.exists())

        entry = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(entry["result"], "failed")
        self.assertIn("forced_failure_for_test", entry["error"])
        self.assertFalse(entry["quality_gate"]["passed"])

    def test_cli_wrapper_returns_non_zero_on_failure(self) -> None:
        """
        _run() (the CLI entry point main() calls) must catch any
        AutoDailyDirectorError-family exception and return a non-zero
        exit code rather than letting it propagate uncaught.
        """
        from unittest.mock import patch

        from .auto_daily_director import _run, parse_arguments

        arguments = parse_arguments(
            ["--date", CALENDAR_DATE, "--no-history"]
        )

        with patch(
            "src.auto_daily_director.build_auto_daily_director",
            return_value=self._director_with_failing_gate(),
        ):
            exit_code = _run(arguments)

        self.assertNotEqual(exit_code, 0)


# ---------------------------------------------------------------------------
# 8 & 9 — motion / wardrobe history updated on a successful real run
# ---------------------------------------------------------------------------


class HistoryUpdateTests(DailyDirectorTestCase):
    def test_motion_history_is_updated_on_successful_run(self) -> None:
        """
        Pre-existing, out-of-scope bug found while writing this test:
        DynamicContentPlanner.generate() (dynamic_content_planner.py,
        not editable in Phase 9) calls
        self.builder.build_theme_scenes(theme=..., weather_mode=...,
        season=...) WITHOUT passing production_date, so
        StoryBuilder.build_theme_scenes()'s effective_date silently
        defaults to date.today() rather than the requested
        target_date — motion history is always recorded under
        today's real date, not whatever --date was requested. This
        test verifies the mechanism actually fires and persists
        (requirement 22) against that real, observed behavior rather
        than asserting an incorrect expectation.
        """
        from datetime import date as _date

        self.director.run(target_date=CALENDAR_DATE)  # no no_history=True

        data = json.loads(
            self.real_motion_history_path.read_text(encoding="utf-8")
        )
        self.assertIn(_date.today().isoformat(), data.get("history", {}))

    def test_wardrobe_history_is_updated_on_successful_run(self) -> None:
        self.director.run(target_date=CALENDAR_DATE)

        wardrobe_history_path = self.output_root / "history" / "wardrobe.json"
        self.assertTrue(wardrobe_history_path.exists())

        data = json.loads(wardrobe_history_path.read_text(encoding="utf-8"))
        self.assertIn(CALENDAR_DATE, data)
        self.assertTrue(data[CALENDAR_DATE].get("top") or data[CALENDAR_DATE].get("dress"))

    def test_no_history_flag_skips_wardrobe_history(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        wardrobe_history_path = self.output_root / "history" / "wardrobe.json"
        self.assertFalse(wardrobe_history_path.exists())


# ---------------------------------------------------------------------------
# 10 — Threads post differs from Feed caption
# ---------------------------------------------------------------------------


class ThreadsPostTests(DailyDirectorTestCase):
    def test_threads_post_differs_from_feed_caption(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        day_root = self._day_root(CALENDAR_DATE)
        feed_caption = (day_root / "captions" / "feed_caption.txt").read_text(
            encoding="utf-8"
        ).strip()
        threads_post = (day_root / "captions" / "threads_post.txt").read_text(
            encoding="utf-8"
        ).strip()

        self.assertNotEqual(feed_caption.lower(), threads_post.lower())

    def test_threads_post_under_character_limit(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        day_root = self._day_root(CALENDAR_DATE)
        threads_post = (day_root / "captions" / "threads_post.txt").read_text(
            encoding="utf-8"
        ).strip()

        self.assertLessEqual(len(threads_post), 280)


# ---------------------------------------------------------------------------
# 11 — JSON writes are atomic
# ---------------------------------------------------------------------------


class AtomicWriteTests(DailyDirectorTestCase):
    def test_no_tmp_files_left_behind_after_real_run(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        leftover = list(self._day_root(CALENDAR_DATE).rglob("*.tmp"))
        self.assertEqual(leftover, [])

    def test_all_json_outputs_are_valid_json(self) -> None:
        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        day_root = self._day_root(CALENDAR_DATE)

        for relative in (
            "content_plan.json",
            "production_manifest.json",
            "queues/image_queue.json",
            "queues/reel_queue.json",
        ):
            json.loads((day_root / relative).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 12 — social files remain untouched
# ---------------------------------------------------------------------------


class SocialIsolationTests(DailyDirectorTestCase):
    def test_module_does_not_import_social_package(self) -> None:
        import ast

        source_root = Path(__file__).resolve().parent

        for module_name in ("auto_daily_director", "daily_run_models"):
            source = (source_root / f"{module_name}.py").read_text(
                encoding="utf-8"
            )
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertNotIn(
                        "social",
                        module,
                        f"{module_name}.py must not import from .social",
                    )

    def test_community_output_directory_is_untouched(self) -> None:
        community_dir = (
            Path(__file__).resolve().parents[1] / "output" / "community"
        )

        before = None
        if community_dir.exists():
            before = sorted(
                str(p.relative_to(community_dir))
                for p in community_dir.rglob("*")
            )

        self.director.run(target_date=CALENDAR_DATE, no_history=True)

        after = None
        if community_dir.exists():
            after = sorted(
                str(p.relative_to(community_dir))
                for p in community_dir.rglob("*")
            )

        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
