"""Covers workflow.py (learn_live_talking_reel()/learn_live_talking_reels()
+ CLI) AND report.py (build_live_learning_report()/
render_live_learning_report_markdown()) -- the task's own TESTS list
has no separate test_report.py, and workflow.py is what actually
drives report.py end-to-end (see test_mapper.py's own docstring for
the identical naming rationale applied to mapper.py/adapter.py).
"""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.live.exceptions import LiveAdapterConfigError
from src.video_intelligence.talking_ai.live.models import LiveTalkingLearningRecord, LiveTalkingReelRecord
from src.video_intelligence.talking_ai.live.report import build_live_learning_report, render_live_learning_report_markdown
from src.video_intelligence.talking_ai.live.workflow import (
    default_talking_ai_live_config_path,
    learn_live_talking_reel,
    learn_live_talking_reels,
    load_talking_ai_live_config,
    main,
)


def _record(index, **overrides):
    defaults = dict(
        source_url=f"https://www.instagram.com/reel/r{index}/",
        creator_label="ref",
        published_at=f"2026-01-{index + 1:02d}T00:00:00+00:00",
        duration_seconds=10.0,
        language_observations=["cantonese"],
        timeline_observations=[
            TalkingAIEvidence(video_id="", timestamp_seconds=0.1, mouth_open_ratio=0.3, blink=True, gaze_direction="direct_camera"),
            TalkingAIEvidence(video_id="", timestamp_seconds=1.0, mouth_open_ratio=0.28),
        ],
        speech_segments=[SpeechSegment(video_id="", start_seconds=0.0, end_seconds=2.0, language="cantonese", pause_before=0.0)],
        camera_observations=["static", "direct_to_camera"],
    )
    defaults.update(overrides)
    return LiveTalkingReelRecord(**defaults)


def _isolated_config():
    config = load_talking_ai_live_config()
    tmp = tempfile.mkdtemp()
    return dataclasses.replace(config, output_root=str(Path(tmp) / "live_out"))


class LoadConfigTests(unittest.TestCase):
    def test_default_config_path_exists_and_loads(self):
        path = default_talking_ai_live_config_path()
        self.assertTrue(path.is_file())
        config = load_talking_ai_live_config()
        self.assertEqual(config.adapter_version, "12D.3")
        self.assertEqual(config.recommended_talking_reels, 15)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LiveAdapterConfigError):
                load_talking_ai_live_config(Path(tmp) / "missing.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("adapter: [unclosed")
            with self.assertRaises(LiveAdapterConfigError):
                load_talking_ai_live_config(path)

    def test_no_live_run_flags_default_to_safe_values(self):
        config = load_talking_ai_live_config()
        self.assertFalse(config.live_enabled_by_default)
        self.assertTrue(config.live_require_explicit_authorization)


class LearnLiveTalkingReelTests(unittest.TestCase):
    def test_produces_dna_and_patterns(self):
        outcome = learn_live_talking_reel(_record(0))
        self.assertIsNotNone(outcome.dna)
        self.assertEqual(outcome.reel_id, _record(0).reel_id)

    def test_dna_id_stable_across_two_independent_runs(self):
        record = _record(0)
        first = learn_live_talking_reel(record)
        second = learn_live_talking_reel(record)
        self.assertEqual(first.dna.dna_id, second.dna.dna_id)

    def test_insufficient_evidence_record_still_produces_outcome(self):
        sparse = LiveTalkingReelRecord(source_url="https://www.instagram.com/reel/sparse/")
        outcome = learn_live_talking_reel(sparse)
        self.assertIsNotNone(outcome.dna)
        self.assertEqual(outcome.overall_confidence, "unknown")

    def test_knowledge_base_ingestion_can_be_disabled(self):
        outcome = learn_live_talking_reel(_record(0), ingest_into_knowledge_base=False)
        self.assertEqual(outcome.patterns_touched, [])

    def test_subject_label_never_a_raw_username(self):
        # No creator_label supplied in metadata -- subject_label falls back
        # to the opaque reel_id, never a real handle.
        record = _record(0)
        outcome = learn_live_talking_reel(record)
        self.assertEqual(outcome.dna.subject_label, record.reel_id)


class LearnLiveTalkingReelsBatchTests(unittest.TestCase):
    def test_single_record_batch(self):
        report, outcomes, diagnostics = learn_live_talking_reels([_record(0)])
        self.assertEqual(report.reels_analyzed, 1)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(diagnostics.records_attempted, 1)

    def test_five_record_batch(self):
        records = [_record(i) for i in range(5)]
        report, outcomes, _diagnostics = learn_live_talking_reels(records)
        self.assertEqual(report.reels_analyzed, 5)
        self.assertEqual(len(outcomes), 5)

    def test_fifteen_record_batch_respects_recommended_sample(self):
        config = _isolated_config()
        records = [_record(i) for i in range(20)]
        report, outcomes, _diagnostics = learn_live_talking_reels(records, config=config)
        self.assertEqual(report.total_reels_discovered, 20)
        self.assertLessEqual(report.reels_sampled, config.recommended_talking_reels)

    def test_partial_batch_with_some_invalid_records(self):
        good = _record(0)
        bad = LiveTalkingReelRecord(source_url="not-a-url")
        report, outcomes, _diagnostics = learn_live_talking_reels([good, bad])
        self.assertEqual(report.reels_analyzed, 1)
        self.assertTrue(any("not-a-url" in warning or "failed" in warning.lower() or bad.reel_id in warning for warning in report.warnings) or report.warnings)

    def test_malformed_batch_never_raises_only_skips(self):
        malformed = LiveTalkingReelRecord(source_url="https://x/", duration_seconds=-5.0)
        report, outcomes, _diagnostics = learn_live_talking_reels([malformed])
        self.assertEqual(report.reels_analyzed, 0)
        self.assertTrue(report.warnings)

    def test_sample_false_uses_every_valid_record(self):
        config = _isolated_config()
        records = [_record(i) for i in range(20)]
        report, _outcomes, _diagnostics = learn_live_talking_reels(records, config=config, sample=False)
        self.assertEqual(report.reels_analyzed, 20)


class DeidentificationTests(unittest.TestCase):
    def test_outcome_carries_no_creator_username_field(self):
        outcome = learn_live_talking_reel(_record(0))
        field_names = set(LiveTalkingLearningRecord.__slots__)
        self.assertNotIn("username", field_names)
        self.assertNotIn("creator_username", field_names)

    def test_dna_never_carries_raw_creator_label_as_username(self):
        outcome = learn_live_talking_reel(_record(0, creator_label="reference_creator"))
        # subject_label is the opaque reel_id, not the raw creator_label string
        self.assertNotEqual(outcome.dna.subject_label, "reference_creator")


class ReportBuildingTests(unittest.TestCase):
    def test_build_report_aggregates_completeness(self):
        _report, outcomes, _diagnostics = learn_live_talking_reels([_record(0), _record(1)])
        report = build_live_learning_report(outcomes, total_reels_discovered=2, reels_sampled=2)
        self.assertEqual(report.reels_analyzed, 2)
        self.assertIn("mouth", report.evidence_completeness)

    def test_insufficient_evidence_reel_ids_populated(self):
        sparse = LiveTalkingReelRecord(source_url="https://www.instagram.com/reel/sparse2/")
        outcome = learn_live_talking_reel(sparse)
        report = build_live_learning_report([outcome], total_reels_discovered=1, reels_sampled=1)
        self.assertIn(outcome.reel_id, report.insufficient_evidence_reel_ids)

    def test_render_markdown_contains_key_sections(self):
        _report, outcomes, _diagnostics = learn_live_talking_reels([_record(0)])
        report = build_live_learning_report(outcomes, total_reels_discovered=1, reels_sampled=1)
        markdown = render_live_learning_report_markdown(report)
        self.assertIn("# Talking AI Live Learning Report", markdown)
        self.assertIn("Production DNA IDs", markdown)
        self.assertIn("Insufficient evidence", markdown)

    def test_render_markdown_never_includes_verbatim_caption(self):
        record = _record(0, metadata={"caption": "a very unique secret caption string"})
        _report, outcomes, _diagnostics = learn_live_talking_reels([record])
        report = build_live_learning_report(outcomes, total_reels_discovered=1, reels_sampled=1)
        markdown = render_live_learning_report_markdown(report)
        self.assertNotIn("a very unique secret caption string", markdown)

    def test_empty_outcomes_report(self):
        report = build_live_learning_report([], total_reels_discovered=0, reels_sampled=0)
        self.assertEqual(report.reels_analyzed, 0)
        self.assertEqual(report.evidence_completeness, {})
        markdown = render_live_learning_report_markdown(report)
        self.assertIn("no Reels analyzed", markdown)


class CliTests(unittest.TestCase):
    def test_missing_evidence_file_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main(["--evidence", str(Path(tmp) / "missing.json"), "--output", tmp])
            self.assertEqual(exit_code, 1)

    def test_validate_only_reports_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text(json.dumps([{"source_url": "https://www.instagram.com/reel/cli1/"}]))
            exit_code = main(["--evidence", str(evidence_path), "--output", str(tmp_path), "--validate-only", "--json"])
            self.assertEqual(exit_code, 0)

    def test_end_to_end_cli_run_produces_report_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text(json.dumps([
                {
                    "source_url": "https://www.instagram.com/reel/cli2/",
                    "creator_label": "ref",
                    "duration_seconds": 8.0,
                    "language_observations": ["cantonese"],
                    "timeline_observations": [
                        {"timestamp_seconds": 0.1, "mouth_open_ratio": 0.3, "blink": True, "gaze_direction": "direct_camera"},
                        {"timestamp_seconds": 1.0, "mouth_open_ratio": 0.25},
                    ],
                    "speech_segments": [
                        {"start_seconds": 0.0, "end_seconds": 2.0, "language": "cantonese", "pause_before": 0.0},
                    ],
                }
            ]))
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            exit_code = main(["--evidence", str(evidence_path), "--output", str(output_dir), "--json"])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "talking_ai_live_learning_report.json").exists())
            self.assertTrue((output_dir / "talking_ai_live_learning_report.md").exists())
            self.assertEqual(len(list(output_dir.glob("diagnostics_*.json"))), 1)

    def test_existing_output_without_force_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text(json.dumps([{"source_url": "https://www.instagram.com/reel/cli3/"}]))
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            (output_dir / "talking_ai_live_learning_report.json").write_text("{}")
            exit_code = main(["--evidence", str(evidence_path), "--output", str(output_dir)])
            self.assertEqual(exit_code, 1)

    def test_malformed_json_input_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text("{not valid json")
            exit_code = main(["--evidence", str(evidence_path), "--output", str(tmp_path)])
            self.assertEqual(exit_code, 1)

    def test_missing_required_source_url_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text(json.dumps([{"creator_label": "ref"}]))
            exit_code = main(["--evidence", str(evidence_path), "--output", str(tmp_path)])
            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
