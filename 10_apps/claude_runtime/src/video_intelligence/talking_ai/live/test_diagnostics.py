import json
import tempfile
import unittest
from pathlib import Path

from src.video_intelligence.talking_ai.evidence import TalkingAIEvidence
from src.video_intelligence.talking_ai.live.adapter import LiveAdaptedEvidence
from src.video_intelligence.talking_ai.live.diagnostics import (
    build_live_learning_diagnostics,
    new_live_run_id,
    save_live_learning_diagnostics,
)
from src.video_intelligence.talking_ai.live.models import COMPLETENESS_DOMAINS, LiveTalkingReelRecord


def _record(index, **overrides):
    defaults = dict(source_url=f"https://www.instagram.com/reel/r{index}/")
    defaults.update(overrides)
    return LiveTalkingReelRecord(**defaults)


class NewLiveRunIdTests(unittest.TestCase):
    def test_generates_distinct_ids(self):
        self.assertNotEqual(new_live_run_id(), new_live_run_id())

    def test_generates_nonempty_hex_string(self):
        run_id = new_live_run_id()
        self.assertTrue(run_id)
        int(run_id, 16)  # raises if not valid hex


class BuildDiagnosticsTests(unittest.TestCase):
    def test_basic_counts(self):
        records = [_record(i) for i in range(5)]
        diagnostics = build_live_learning_diagnostics(
            total_records=7, valid_records=records, invalid_count=2,
            sampled_records=records[:3], adapted_results=[],
        )
        self.assertEqual(diagnostics.records_attempted, 7)
        self.assertEqual(diagnostics.records_valid, 5)
        self.assertEqual(diagnostics.records_invalid, 2)
        self.assertEqual(diagnostics.records_sampled, 3)
        self.assertEqual(diagnostics.records_processed, 0)

    def test_timing_counts_aggregated_from_adapted_results(self):
        adapted = [
            LiveAdaptedEvidence(ordinal_timing_count=2, approximate_timing_count=1),
            LiveAdaptedEvidence(ordinal_timing_count=3, approximate_timing_count=0),
        ]
        diagnostics = build_live_learning_diagnostics(
            total_records=2, valid_records=[], invalid_count=0, sampled_records=[], adapted_results=adapted,
        )
        self.assertEqual(diagnostics.ordinal_timing_item_count, 5)
        self.assertEqual(diagnostics.approximate_timing_item_count, 1)

    def test_warnings_collected_from_adapted_results(self):
        adapted = [LiveAdaptedEvidence(warnings=["w1"]), LiveAdaptedEvidence(warnings=["w2", "w3"])]
        diagnostics = build_live_learning_diagnostics(
            total_records=2, valid_records=[], invalid_count=0, sampled_records=[], adapted_results=adapted,
        )
        self.assertEqual(diagnostics.warnings, ["w1", "w2", "w3"])

    def test_completeness_summary_averages_sampled_records(self):
        record_a = _record(0, timeline_observations=[TalkingAIEvidence(video_id="", timestamp_seconds=0.0, blink=True)])
        record_b = _record(1)  # empty -- all zero completeness
        diagnostics = build_live_learning_diagnostics(
            total_records=2, valid_records=[record_a, record_b], invalid_count=0,
            sampled_records=[record_a, record_b], adapted_results=[],
        )
        self.assertEqual(set(diagnostics.completeness_summary.keys()), set(COMPLETENESS_DOMAINS))
        self.assertAlmostEqual(diagnostics.completeness_summary["blink"], 0.5)

    def test_empty_sampled_records_leaves_completeness_summary_empty(self):
        diagnostics = build_live_learning_diagnostics(
            total_records=0, valid_records=[], invalid_count=0, sampled_records=[], adapted_results=[],
        )
        self.assertEqual(diagnostics.completeness_summary, {})

    def test_finished_at_set(self):
        diagnostics = build_live_learning_diagnostics(
            total_records=0, valid_records=[], invalid_count=0, sampled_records=[], adapted_results=[],
        )
        self.assertIsNotNone(diagnostics.finished_at)

    def test_custom_run_id_used_when_supplied(self):
        diagnostics = build_live_learning_diagnostics(
            run_id="custom-run-id", total_records=0, valid_records=[], invalid_count=0,
            sampled_records=[], adapted_results=[],
        )
        self.assertEqual(diagnostics.run_id, "custom-run-id")


class SaveDiagnosticsTests(unittest.TestCase):
    def test_round_trips_to_disk(self):
        diagnostics = build_live_learning_diagnostics(
            total_records=1, valid_records=[_record(0)], invalid_count=0,
            sampled_records=[_record(0)], adapted_results=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics.json"
            save_live_learning_diagnostics(diagnostics, path)
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text())
            self.assertEqual(payload["run_id"], diagnostics.run_id)
            self.assertEqual(payload["records_attempted"], 1)

    def test_to_dict_is_json_serializable(self):
        diagnostics = build_live_learning_diagnostics(
            total_records=0, valid_records=[], invalid_count=0, sampled_records=[], adapted_results=[],
        )
        json.dumps(diagnostics.to_dict())  # raises if not serializable


if __name__ == "__main__":
    unittest.main()
