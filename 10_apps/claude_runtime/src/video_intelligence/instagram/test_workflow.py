import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType
from src.video_intelligence.production_dna import load_engine_config
from src.video_intelligence.workflow import VideoIntelligenceWorkflow
from src.video_intelligence.instagram.exceptions import AdapterCliError, AdapterConfigError
from src.video_intelligence.instagram.models import InstagramReelEvidencePacket, InstagramReelsConfig
from src.video_intelligence.instagram.workflow import (
    default_instagram_reels_config_path,
    learn_reel,
    learn_reels,
    load_instagram_reels_config,
    main,
    parse_arguments,
)


def _adapter_config(**overrides):
    defaults = dict(
        version="1.0", schema_version="1.0", adapter_version="12D.1", platform="instagram_reels",
        minimum_reels_for_dna=5, recommended_reels=30, strong_sample=50, deterministic=True,
        priorities={"hook": 1.0},
    )
    defaults.update(overrides)
    return InstagramReelsConfig(**defaults)


def _video_config(tmp_path):
    config = load_engine_config()
    config.output_root = str(Path(tmp_path) / "vi_output")
    return config


def _annotation(tags, timestamp=None):
    return VideoEvidence(
        video_id="", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="note",
        content_excerpt="x", timestamp_seconds=timestamp, tags=tags,
    )


def _packet(i, *, with_hook=True, malformed=False, creator_label="reference_creator", caption=None):
    if malformed:
        return InstagramReelEvidencePacket(reel_url="", creator_label=creator_label)
    annotations = [_annotation(["hook", "question"])] if with_hook else []
    return InstagramReelEvidencePacket(
        reel_url=f"https://www.instagram.com/reel/r{i}/", creator_label=creator_label,
        duration_seconds=10.0 + i, views=1000 + i, likes=50 + i, comments=5 + i, caption=caption,
        published_at=f"2026-08-{(i % 28) + 1:02d}T00:00:00Z", annotations=annotations,
    )


class LearnReelTests(unittest.TestCase):
    def test_produces_dna_and_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = learn_reel(_packet(0), video_config=_video_config(tmp))
            self.assertIsNotNone(record.dna)
            self.assertEqual(record.completeness["hook"], 1.0)

    def test_stable_reel_id_matches_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet = _packet(0)
            record = learn_reel(packet, video_config=_video_config(tmp))
            self.assertEqual(record.reel_id, packet.reel_id)

    def test_missing_duration_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet = InstagramReelEvidencePacket(reel_url="https://x/y", creator_label="ref")
            record = learn_reel(packet, video_config=_video_config(tmp))
            self.assertTrue(any("duration_seconds" in w for w in record.warnings))

    def test_missing_text_overlays_warns_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = learn_reel(_packet(0), video_config=_video_config(tmp))
            self.assertTrue(any("text_overlays" in w for w in record.warnings))


class LearnReelsBatchTests(unittest.TestCase):
    def test_10_reels(self):
        with tempfile.TemporaryDirectory() as tmp:
            packets = [_packet(i) for i in range(10)]
            report, records = learn_reels(
                packets, config=_adapter_config(recommended_reels=10), video_config=_video_config(tmp), sample=False,
            )
            self.assertEqual(report.reels_analyzed, 10)
            self.assertEqual(len(report.video_dna_ids), 10)

    def test_30_reels(self):
        with tempfile.TemporaryDirectory() as tmp:
            packets = [_packet(i) for i in range(30)]
            report, records = learn_reels(
                packets, config=_adapter_config(recommended_reels=30), video_config=_video_config(tmp), sample=False,
            )
            self.assertEqual(report.reels_analyzed, 30)
            self.assertEqual(len(records), 30)

    def test_partial_evidence_reels_still_analyzed(self):
        with tempfile.TemporaryDirectory() as tmp:
            packets = [_packet(i, with_hook=(i % 2 == 0)) for i in range(6)]
            report, records = learn_reels(
                packets, config=_adapter_config(recommended_reels=6), video_config=_video_config(tmp), sample=False,
            )
            self.assertEqual(report.reels_analyzed, 6)
            self.assertGreater(report.evidence_completeness["hook"], 0.0)
            self.assertLess(report.evidence_completeness["hook"], 1.0)

    def test_one_malformed_reel_skipped_others_proceed(self):
        with tempfile.TemporaryDirectory() as tmp:
            packets = [_packet(0), _packet(1), _packet(2, malformed=True)]
            report, records = learn_reels(
                packets, config=_adapter_config(recommended_reels=10), video_config=_video_config(tmp), sample=False,
            )
            self.assertEqual(report.reels_analyzed, 2)
            self.assertEqual(report.reels_skipped, 1)

    def test_video_dna_ids_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            packets = [_packet(i) for i in range(5)]
            report, _records = learn_reels(
                packets, config=_adapter_config(recommended_reels=5), video_config=_video_config(tmp), sample=False,
            )
            self.assertEqual(len(report.video_dna_ids), len(set(report.video_dna_ids)))

    def test_knowledge_base_receives_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_config = _video_config(tmp)
            packets = [_packet(i) for i in range(5)]
            report, _records = learn_reels(
                packets, config=_adapter_config(recommended_reels=5), video_config=video_config, sample=False,
            )
            self.assertGreater(report.knowledge_patterns_added, 0)
            kb_patterns = VideoIntelligenceWorkflow(video_config).knowledge_base().list_patterns()
            self.assertTrue(kb_patterns)

    def test_sampling_applied_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            packets = [_packet(i) for i in range(50)]
            report, _records = learn_reels(
                packets, config=_adapter_config(recommended_reels=10), video_config=_video_config(tmp), sample=True,
            )
            self.assertEqual(report.reels_analyzed, 10)

    def test_deterministic_dna_ids_across_independent_runs(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            report_a, _ = learn_reels(
                [_packet(i) for i in range(3)], config=_adapter_config(recommended_reels=3),
                video_config=_video_config(tmp_a), sample=False,
            )
            report_b, _ = learn_reels(
                [_packet(i) for i in range(3)], config=_adapter_config(recommended_reels=3),
                video_config=_video_config(tmp_b), sample=False,
            )
            self.assertEqual(report_a.video_dna_ids, report_b.video_dna_ids)

    def test_ingest_into_knowledge_base_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_config = _video_config(tmp)
            packets = [_packet(i) for i in range(3)]
            report, _records = learn_reels(
                packets, config=_adapter_config(recommended_reels=3), video_config=video_config, sample=False,
                ingest_into_knowledge_base=False,
            )
            self.assertEqual(report.knowledge_patterns_added, 0)
            self.assertEqual(VideoIntelligenceWorkflow(video_config).knowledge_base().list_patterns(), [])


class DeIdentificationTests(unittest.TestCase):
    def test_creator_label_never_enters_generalized_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_config = _video_config(tmp)
            unique_label = "very_unique_creator_handle_12345"
            packets = [_packet(i, creator_label=unique_label) for i in range(3)]
            learn_reels(packets, config=_adapter_config(recommended_reels=3), video_config=video_config, sample=False)
            patterns = VideoIntelligenceWorkflow(video_config).knowledge_base().list_patterns()
            self.assertTrue(patterns)
            for pattern in patterns:
                self.assertNotIn(unique_label, json.dumps(pattern.to_dict()))

    def test_verbatim_caption_never_enters_generalized_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_config = _video_config(tmp)
            unique_caption = "this is a very unique verbatim caption sentence xyz123"
            packet = _packet(0, caption=unique_caption)
            learn_reels([packet], config=_adapter_config(recommended_reels=1), video_config=video_config, sample=False)
            patterns = VideoIntelligenceWorkflow(video_config).knowledge_base().list_patterns()
            for pattern in patterns:
                self.assertNotIn(unique_caption, json.dumps(pattern.to_dict()))

    def test_production_pattern_has_no_creator_identifying_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_config = _video_config(tmp)
            packets = [_packet(i) for i in range(2)]
            learn_reels(packets, config=_adapter_config(recommended_reels=2), video_config=video_config, sample=False)
            patterns = VideoIntelligenceWorkflow(video_config).knowledge_base().list_patterns()
            for pattern in patterns:
                self.assertNotIn("creator_label", pattern.to_dict())
                self.assertNotIn("username", pattern.to_dict())


class LoadInstagramReelsConfigTests(unittest.TestCase):
    def test_default_config_path_exists_and_loads(self):
        self.assertTrue(default_instagram_reels_config_path().is_file())
        config = load_instagram_reels_config()
        self.assertEqual(config.platform, "instagram_reels")
        self.assertEqual(config.recommended_reels, 30)

    def test_lip_sync_priority_key_normalized_to_lipsync(self):
        config = load_instagram_reels_config()
        self.assertIn("lipsync", config.priorities)
        self.assertNotIn("lip_sync", config.priorities)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AdapterConfigError):
                load_instagram_reels_config(Path(tmp) / "missing.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instagram_reels.yaml"
            path.write_text("adapter: [unterminated\n")
            with self.assertRaises(AdapterConfigError):
                load_instagram_reels_config(path)

    def test_missing_individual_keys_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instagram_reels.yaml"
            path.write_text("version: '1.0'\n")
            config = load_instagram_reels_config(path)
            self.assertEqual(config.minimum_reels_for_dna, 10)
            self.assertEqual(config.recommended_reels, 30)
            self.assertTrue(config.deidentify_creator)


class CliTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.evidence_path = self.tmp_path / "evidence.json"
        self.output_dir = self.tmp_path / "output"
        self.video_config = _video_config(self.tmp_path)
        self._patcher = patch(
            "src.video_intelligence.instagram.workflow.VideoIntelligenceWorkflow",
            side_effect=lambda *_a, **_k: VideoIntelligenceWorkflow(self.video_config),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def _write_evidence(self, packets_payload):
        self.evidence_path.write_text(json.dumps(packets_payload))


class ParseArgumentsTests(unittest.TestCase):
    def test_requires_evidence_and_output(self):
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_no_browser_or_connector_flags_exist(self):
        args = parse_arguments(["--evidence", "e.json", "--output", "out"])
        arg_names = vars(args).keys()
        for forbidden in ("connector", "browser", "playwright", "session", "login"):
            matches = [name for name in arg_names if forbidden in name.lower()]
            self.assertEqual(matches, [], f"unexpected live-acquisition-shaped flag: {matches}")


class CliLocalEvidenceModeTests(CliTempTestCase):
    def test_analyze_from_local_evidence_writes_report(self):
        self._write_evidence(
            [
                {
                    "reel_url": "https://www.instagram.com/reel/cli1/", "creator_label": "cli_test",
                    "duration_seconds": 12.0, "views": 5000, "likes": 200, "comments": 20,
                    "annotations": [{"evidence_type": "operator_observation", "source_description": "hook",
                                      "content_excerpt": "x", "tags": ["hook", "question"]}],
                }
            ]
        )
        rc = main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir), "--json"])
        self.assertEqual(rc, 0)
        self.assertTrue((self.output_dir / "instagram_reels_learning_report.json").is_file())
        self.assertTrue((self.output_dir / "instagram_reels_learning_report.md").is_file())

    def test_missing_evidence_file_errors(self):
        rc = main(["--evidence", str(self.tmp_path / "nope.json"), "--output", str(self.output_dir)])
        self.assertEqual(rc, 1)

    def test_malformed_json_errors(self):
        self.evidence_path.write_text("{not valid json")
        rc = main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir)])
        self.assertEqual(rc, 1)

    def test_output_exists_without_force_errors(self):
        self._write_evidence([{"reel_url": "https://www.instagram.com/reel/cli1/", "duration_seconds": 5.0}])
        main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir)])
        rc = main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir)])
        self.assertEqual(rc, 1)

    def test_force_overwrites_existing_output(self):
        self._write_evidence([{"reel_url": "https://www.instagram.com/reel/cli1/", "duration_seconds": 5.0}])
        main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir)])
        rc = main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir), "--force"])
        self.assertEqual(rc, 0)


class CliValidateOnlyTests(CliTempTestCase):
    def test_validate_only_passes_on_clean_evidence(self):
        self._write_evidence([{"reel_url": "https://www.instagram.com/reel/cli1/", "duration_seconds": 5.0}])
        rc = main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir), "--validate-only", "--json"])
        self.assertEqual(rc, 0)

    def test_validate_only_fails_on_invalid_evidence(self):
        self._write_evidence([{"reel_url": "not-a-url"}])
        rc = main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir), "--validate-only"])
        self.assertEqual(rc, 1)

    def test_validate_only_does_not_write_report(self):
        self._write_evidence([{"reel_url": "https://www.instagram.com/reel/cli1/", "duration_seconds": 5.0}])
        main(["--evidence", str(self.evidence_path), "--output", str(self.output_dir), "--validate-only"])
        self.assertFalse((self.output_dir / "instagram_reels_learning_report.json").exists())


if __name__ == "__main__":
    unittest.main()
