import tempfile
import unittest
from pathlib import Path

from src.video_intelligence.analyzer import AnalyzerContext
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType, VideoPlatform, VideoRecord
from src.video_intelligence.hook import HookAnalyzer, HookConfig, HookConfigError, default_hook_config_path, load_hook_config
from src.video_intelligence.models import VideoIntelligenceConfig


def _config():
    return VideoIntelligenceConfig(
        schema_version="1.0", confidence_min_evidence_for_medium=2, confidence_min_evidence_for_high=4,
        confidence_min_corroboration_for_verified=2, excerpt_max_chars=280,
    )


def _record():
    return VideoRecord(platform=VideoPlatform.INSTAGRAM_REELS, creator_label="ref", reference="demo")


def _evidence(video_id, tags, timestamp=None, source_description="hook note"):
    return VideoEvidence(
        video_id=video_id, evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description=source_description,
        content_excerpt="excerpt", timestamp_seconds=timestamp, tags=tags,
    )


class DefaultHookConfigTests(unittest.TestCase):
    def test_default_config_path_exists_and_loads(self):
        path = default_hook_config_path()
        self.assertTrue(path.is_file())
        config = load_hook_config()
        self.assertEqual(config.schema_version, "1.0")
        self.assertIn("question", config.hook_types)


class LoadHookConfigTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(HookConfigError):
                load_hook_config(Path(tmp) / "does_not_exist.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hooks.yaml"
            path.write_text("hook: [unterminated\n")
            with self.assertRaises(HookConfigError):
                load_hook_config(path)

    def test_missing_individual_keys_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hooks.yaml"
            path.write_text("hook: {}\n")
            config = load_hook_config(path)
            self.assertEqual(config.schema_version, "1.0")
            self.assertEqual(config.hook_types, ())


class HookAnalyzerTests(unittest.TestCase):
    def test_zero_evidence_is_unknown_confidence(self):
        record = _record()
        analyzer = HookAnalyzer(HookConfig(schema_version="1.0", hook_types=("question",)))
        context = AnalyzerContext(video_id=record.video_id, evidence=[], config=_config())
        result = analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, "unknown")
        self.assertEqual(result.trait_score.score, 0.0)

    def test_scoped_evidence_increases_confidence(self):
        record = _record()
        analyzer = HookAnalyzer(HookConfig(schema_version="1.0", hook_types=("question",)))
        evidence = [_evidence(record.video_id, ["hook", "question"], timestamp=0.5)]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = analyzer.analyze(context)
        self.assertGreater(result.trait_score.score, 0.0)
        self.assertEqual(result.trait_score.evidence_ids, [evidence[0].evidence_id])

    def test_observed_hook_type_surfaced_in_rationale(self):
        record = _record()
        analyzer = HookAnalyzer(HookConfig(schema_version="1.0", hook_types=("question", "shock")))
        evidence = [_evidence(record.video_id, ["hook", "question"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = analyzer.analyze(context)
        self.assertIn("question", result.trait_score.rationale)
        self.assertNotIn("shock", result.trait_score.rationale)

    def test_evidence_from_other_video_is_ignored(self):
        record = _record()
        other_record = VideoRecord(platform=VideoPlatform.TIKTOK, creator_label="other", reference="other_ref")
        evidence = [_evidence(other_record.video_id, ["hook", "question"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = HookAnalyzer(HookConfig(schema_version="1.0", hook_types=("question",))).analyze(context)
        self.assertEqual(result.trait_score.confidence, "unknown")

    def test_unrelated_tag_does_not_leak_in(self):
        record = _record()
        evidence = [_evidence(record.video_id, ["camera", "close_up"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        result = HookAnalyzer(HookConfig(schema_version="1.0", hook_types=("question",))).analyze(context)
        self.assertEqual(result.trait_score.confidence, "unknown")

    def test_deterministic_across_repeated_runs(self):
        record = _record()
        analyzer = HookAnalyzer(HookConfig(schema_version="1.0", hook_types=("question",)))
        evidence = [_evidence(record.video_id, ["hook", "question"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        first = analyzer.analyze(context).trait_score
        second = analyzer.analyze(context).trait_score
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
