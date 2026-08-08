import tempfile
import unittest
from pathlib import Path

from src.video_intelligence.analyzer import AnalyzerContext
from src.video_intelligence.camera import CameraAnalyzer, CameraConfig, CameraConfigError, default_camera_config_path, load_camera_config
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType, VideoPlatform, VideoRecord
from src.video_intelligence.models import VideoIntelligenceConfig


def _config():
    return VideoIntelligenceConfig(
        schema_version="1.0", confidence_min_evidence_for_medium=2, confidence_min_evidence_for_high=4,
        confidence_min_corroboration_for_verified=2, excerpt_max_chars=280,
    )


def _record():
    return VideoRecord(platform=VideoPlatform.INSTAGRAM_REELS, creator_label="ref", reference="demo")


def _evidence(video_id, tags):
    return VideoEvidence(
        video_id=video_id, evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="camera note",
        content_excerpt="excerpt", tags=tags,
    )


class DefaultCameraConfigTests(unittest.TestCase):
    def test_default_config_path_exists_and_loads(self):
        self.assertTrue(default_camera_config_path().is_file())
        config = load_camera_config()
        self.assertIn("close_up", config.distance_types)
        self.assertIn("handheld", config.equipment_types)


class LoadCameraConfigTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CameraConfigError):
                load_camera_config(Path(tmp) / "missing.yaml")

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.yaml"
            path.write_text("camera: [unterminated\n")
            with self.assertRaises(CameraConfigError):
                load_camera_config(path)


class CameraAnalyzerTests(unittest.TestCase):
    def _camera_config(self):
        return CameraConfig(
            schema_version="1.0", distance_types=("close_up",), angle_types=("eye_level",),
            lens_feeling_types=("natural",), equipment_types=("handheld", "tripod"),
        )

    def test_zero_evidence_is_unknown(self):
        record = _record()
        analyzer = CameraAnalyzer(self._camera_config())
        context = AnalyzerContext(video_id=record.video_id, evidence=[], config=_config())
        self.assertEqual(analyzer.analyze(context).trait_score.confidence, "unknown")

    def test_observed_distance_and_equipment_surfaced(self):
        record = _record()
        analyzer = CameraAnalyzer(self._camera_config())
        evidence = [_evidence(record.video_id, ["camera", "close_up", "handheld"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        rationale = analyzer.analyze(context).trait_score.rationale
        self.assertIn("close_up", rationale)
        self.assertIn("handheld", rationale)
        self.assertNotIn("tripod", rationale)

    def test_deterministic(self):
        record = _record()
        analyzer = CameraAnalyzer(self._camera_config())
        evidence = [_evidence(record.video_id, ["camera", "close_up"])]
        context = AnalyzerContext(video_id=record.video_id, evidence=evidence, config=_config())
        self.assertEqual(analyzer.analyze(context).trait_score, analyzer.analyze(context).trait_score)


if __name__ == "__main__":
    unittest.main()
