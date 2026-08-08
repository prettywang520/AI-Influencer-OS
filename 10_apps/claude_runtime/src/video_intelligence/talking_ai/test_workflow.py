import json
import tempfile
import unittest
from pathlib import Path

from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.exceptions import TalkingAIConfigError
from src.video_intelligence.talking_ai.models import NATURALNESS_DIMENSION_ALIASES, TalkingAIConfig
from src.video_intelligence.talking_ai.workflow import (
    TalkingAIWorkflow,
    default_talking_ai_config_path,
    load_talking_ai_config,
    main,
    talking_evidence_from_reel_record,
)


def _config(**overrides):
    defaults = dict(
        version="1.0", schema_version="1.0", engine_version="1.0.0",
        maximum_reasonable_sync_latency_seconds=0.5, pause_min_seconds=0.3,
        minimum_talking_videos=1, recommended_talking_videos=5, strong_sample=15,
        naturalness_dimensions=tuple(NATURALNESS_DIMENSION_ALIASES.values()),
        minimum_pattern_corroboration=2,
    )
    defaults.update(overrides)
    return TalkingAIConfig(**defaults)


def _make_video(video_id):
    segments = [SpeechSegment(video_id=video_id, start_seconds=0.0, end_seconds=2.0, pause_before=0.0)]
    evidence = [
        TalkingAIEvidence(
            video_id=video_id, timestamp_seconds=t, speech_active=(t < 2.0),
            mouth_open_ratio=0.3 if t < 2.0 else 0.02, blink=(t == 0.5),
            gaze_direction="direct_camera", camera_motion="static",
        )
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]
    ]
    return video_id, f"demo {video_id}", evidence, segments


class LoadTalkingAIConfigTests(unittest.TestCase):
    def test_default_config_path_exists_and_loads(self):
        path = default_talking_ai_config_path()
        self.assertTrue(path.is_file())
        config = load_talking_ai_config()
        self.assertEqual(config.schema_version, "1.0")
        self.assertIn("blink_variability", config.naturalness_dimensions)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TalkingAIConfigError):
                load_talking_ai_config(Path(tmp) / "missing.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("talking_ai: [unclosed")
            with self.assertRaises(TalkingAIConfigError):
                load_talking_ai_config(path)

    def test_confidence_section_defaults_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.yaml"
            path.write_text("talking_ai:\n  schema_version: \"1.0\"\n")
            config = load_talking_ai_config(path)
            self.assertEqual(config.confidence_min_evidence_for_medium, 2)


class WorkflowAnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = _config(output_root=str(Path(self.tmp.name) / "talking_ai_out"))
        self.workflow = TalkingAIWorkflow(self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyze_talking_video_produces_dna_and_report_files(self):
        video_id, label, evidence, segments = _make_video("vidA")
        result = self.workflow.analyze_talking_video(video_id, label, evidence, segments)
        video_dir = self.workflow._video_dir(video_id)
        self.assertTrue((video_dir / "talking_ai_dna.json").exists())
        self.assertTrue((video_dir / "talking_ai_report.md").exists())
        self.assertTrue((video_dir / "talking_ai_report.json").exists())
        self.assertEqual(result.dna.video_id, video_id)

    def test_load_dna_round_trips(self):
        video_id, label, evidence, segments = _make_video("vidA")
        result = self.workflow.analyze_talking_video(video_id, label, evidence, segments)
        loaded = self.workflow.load_dna(video_id)
        self.assertEqual(loaded.dna_id, result.dna.dna_id)

    def test_load_dna_for_unknown_video_is_none(self):
        self.assertIsNone(self.workflow.load_dna("never-analyzed"))

    def test_dna_id_is_stable_across_two_independent_runs(self):
        video_id, label, evidence, segments = _make_video("vidA")
        result_a = self.workflow.analyze_talking_video(video_id, label, evidence, segments)
        result_b = self.workflow.analyze_talking_video(video_id, label, evidence, segments)
        self.assertEqual(result_a.dna.dna_id, result_b.dna.dna_id)

    def test_analyze_talking_videos_batch(self):
        videos = [_make_video("vidA"), _make_video("vidB"), _make_video("vidC")]
        results = self.workflow.analyze_talking_videos(videos)
        self.assertEqual(len(results), 3)
        self.assertEqual({result.video_id for result in results}, {"vidA", "vidB", "vidC"})

    def test_knowledge_base_ingestion_can_be_disabled(self):
        video_id, label, evidence, segments = _make_video("vidA")
        result = self.workflow.analyze_talking_video(video_id, label, evidence, segments, ingest_into_knowledge_base=False)
        self.assertEqual(result.patterns, [])

    def test_load_report_markdown_round_trips(self):
        video_id, label, evidence, segments = _make_video("vidA")
        result = self.workflow.analyze_talking_video(video_id, label, evidence, segments)
        markdown = self.workflow.load_report_markdown(video_id)
        self.assertEqual(markdown, result.report_markdown)


class BridgeFromReelRecordTests(unittest.TestCase):
    class _FakeRecord:
        def __init__(self, video_evidence, dna=None, reel_id="reel123"):
            self.video_evidence = video_evidence
            self.dna = dna
            self.reel_id = reel_id

    def _evidence(self, timestamp, tags):
        return VideoEvidence(
            video_id="v1", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
            source_description="ops", content_excerpt="note", timestamp_seconds=timestamp,
            collected_by="operator", tags=tags,
        )

    def test_maps_recognized_boolean_tags(self):
        record = self._FakeRecord([self._evidence(1.0, ["blink_timing", "eye_contact"])])
        bridged = talking_evidence_from_reel_record(record)
        self.assertEqual(len(bridged), 1)
        self.assertTrue(bridged[0].blink)
        self.assertTrue(bridged[0].eye_contact)

    def test_maps_camera_motion_tags(self):
        record = self._FakeRecord([self._evidence(1.0, ["handheld"])])
        bridged = talking_evidence_from_reel_record(record)
        self.assertEqual(bridged[0].camera_motion, "handheld")

    def test_skips_evidence_with_no_timestamp(self):
        item = self._evidence(1.0, ["handheld"])
        item.timestamp_seconds = None
        record = self._FakeRecord([item])
        bridged = talking_evidence_from_reel_record(record)
        self.assertEqual(bridged, [])

    def test_skips_evidence_with_no_understood_tags(self):
        record = self._FakeRecord([self._evidence(1.0, ["delivery_confidence"])])
        bridged = talking_evidence_from_reel_record(record)
        self.assertEqual(bridged, [])

    def test_never_fabricates_unmapped_fields(self):
        record = self._FakeRecord([self._evidence(1.0, ["blink_timing"])])
        bridged = talking_evidence_from_reel_record(record)
        self.assertIsNone(bridged[0].mouth_open_ratio)
        self.assertIsNone(bridged[0].head_motion_intensity)

    def test_confidence_is_low_for_bridged_evidence(self):
        record = self._FakeRecord([self._evidence(1.0, ["blink_timing"])])
        bridged = talking_evidence_from_reel_record(record)
        self.assertEqual(bridged[0].confidence, "low")


class CliTests(unittest.TestCase):
    def test_missing_evidence_file_raises_cli_error_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main(["--evidence", str(Path(tmp) / "missing.json"), "--output", tmp])
            self.assertEqual(exit_code, 1)

    def test_end_to_end_cli_run_produces_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text(json.dumps([
                {
                    "video_id": "vidA",
                    "subject_label": "demo vidA",
                    "evidence": [
                        {"timestamp_seconds": 0.0, "mouth_open_ratio": 0.3, "camera_motion": "static"},
                        {"timestamp_seconds": 0.5, "blink": True, "gaze_direction": "direct_camera"},
                    ],
                    "speech_segments": [
                        {"start_seconds": 0.0, "end_seconds": 2.0, "pause_before": 0.0},
                    ],
                }
            ]))
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            config_path = default_talking_ai_config_path()
            exit_code = main([
                "--evidence", str(evidence_path), "--output", str(output_dir),
                "--config", str(config_path), "--json",
            ])
            self.assertEqual(exit_code, 0)
            summary_path = output_dir / "talking_ai_batch_summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text())
            self.assertEqual(payload["total_videos"], 1)
            self.assertIn("vidA", payload["video_dna_ids"])

    def test_existing_output_without_force_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text(json.dumps([
                {"video_id": "vidA", "subject_label": "demo", "evidence": [], "speech_segments": []}
            ]))
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            (output_dir / "talking_ai_batch_summary.json").write_text("{}")
            exit_code = main(["--evidence", str(evidence_path), "--output", str(output_dir)])
            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
