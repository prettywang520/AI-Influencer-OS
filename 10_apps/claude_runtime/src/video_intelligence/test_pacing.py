import tempfile
import unittest
from pathlib import Path

from src.video_intelligence.analyzer import AnalyzerContext
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType, VideoPlatform, VideoRecord
from src.video_intelligence.models import VideoIntelligenceConfig
from src.video_intelligence.pacing import PacingAnalyzer, PacingConfig, PacingConfigError, default_pacing_config_path, load_pacing_config


def _config():
    return VideoIntelligenceConfig(
        schema_version="1.0", confidence_min_evidence_for_medium=2, confidence_min_evidence_for_high=4,
        confidence_min_corroboration_for_verified=2, excerpt_max_chars=280,
    )


def _record():
    return VideoRecord(platform=VideoPlatform.INSTAGRAM_REELS, creator_label="ref", reference="demo")


def _evidence(video_id, tags):
    return VideoEvidence(
        video_id=video_id, evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="pacing note",
        content_excerpt="excerpt", tags=tags,
    )


class DefaultPacingConfigTests(unittest.TestCase):
    def test_default_config_path_exists_and_loads(self):
        self.assertTrue(default_pacing_config_path().is_file())
        config = load_pacing_config()
        self.assertIn("fast", config.tempo_types)
        self.assertIn("varied", config.rhythm_types)


class LoadPacingConfigTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PacingConfigError):
                load_pacing_config(Path(tmp) / "missing.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pacing.yaml"
            path.write_text("pacing: [unterminated\n")
            with self.assertRaises(PacingConfigError):
                load_pacing_config(path)


class PacingAnalyzerTests(unittest.TestCase):
    def test_zero_evidence_is_unknown(self):
        record = _record()
        analyzer = PacingAnalyzer(PacingConfig(schema_version="1.0", tempo_types=("fast",), rhythm_types=("varied",)))
        context = AnalyzerContext(video_id=record.video_id, evidence=[], config=_config())
        result = analyzer.analyze(context)
        self.assertEqual(result.trait_score.confidence, "unknown")

    def test_observed_tempo_and_rhythm_surfaced(self):
        record = _record()
        analyzer = PacingAnalyzer(PacingConfig(schema_version="1.0", tempo_types=("fast",), rhythm_types=("varied",)))
        evidence = [_evidence(record.video_id, ["pacing", "fast", "varied"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        rationale = analyzer.analyze(context).trait_score.rationale
        self.assertIn("fast", rationale)
        self.assertIn("varied", rationale)

    def test_deterministic(self):
        record = _record()
        analyzer = PacingAnalyzer(PacingConfig(schema_version="1.0", tempo_types=("fast",), rhythm_types=()))
        evidence = [_evidence(record.video_id, ["pacing", "fast"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        self.assertEqual(analyzer.analyze(context).trait_score, analyzer.analyze(context).trait_score)


if __name__ == "__main__":
    unittest.main()
