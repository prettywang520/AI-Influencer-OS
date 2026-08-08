import tempfile
import unittest
from pathlib import Path

from src.video_intelligence.camera import CameraConfig
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType, VideoPlatform, VideoRecord
from src.video_intelligence.hook import HookConfig
from src.video_intelligence.knowledge_base import KnowledgeBaseError, ProductionPattern, VideoKnowledgeBase
from src.video_intelligence.pacing import PacingConfig
from src.video_intelligence.production_dna import load_engine_config, build_video_dna
from src.video_intelligence.speech import SpeechConfig
from src.video_intelligence.subtitles import SubtitlesConfig

_CONFIG = load_engine_config()

_HOOK_CONFIG = HookConfig(schema_version="1.0", hook_types=("question", "shock"))
_PACING_CONFIG = PacingConfig(schema_version="1.0", tempo_types=("fast",), rhythm_types=())
_SPEECH_CONFIG = SpeechConfig(schema_version="1.0", languages=("cantonese",), speed_types=(), energy_types=(), question_style_types=())
_CAMERA_CONFIG = CameraConfig(schema_version="1.0", distance_types=("close_up",), angle_types=(), lens_feeling_types=(), equipment_types=())
_SUBTITLES_CONFIG = SubtitlesConfig(schema_version="1.0", style_types=("bold_caption",), position_types=())


def _ingest_kwargs():
    return dict(
        hook_config=_HOOK_CONFIG, pacing_config=_PACING_CONFIG, speech_config=_SPEECH_CONFIG,
        camera_config=_CAMERA_CONFIG, subtitles_config=_SUBTITLES_CONFIG,
    )


def _video(creator_label, reference="ref", extra_tags=("hook", "question")):
    record = VideoRecord(platform=VideoPlatform.INSTAGRAM_REELS, creator_label=creator_label, reference=reference)
    evidence = [
        VideoEvidence(
            video_id=record.video_id, evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
            source_description="note", content_excerpt="excerpt", tags=list(extra_tags),
        )
    ]
    dna = build_video_dna(record.video_id, creator_label, evidence, _CONFIG)
    return dna, evidence


class ProductionPatternModelTests(unittest.TestCase):
    def test_pattern_id_is_deterministic_content_hash(self):
        a = ProductionPattern(category="hook", description="hook type: question")
        b = ProductionPattern(category="hook", description="hook type: question")
        self.assertEqual(a.pattern_id, b.pattern_id)

    def test_different_description_changes_id(self):
        a = ProductionPattern(category="hook", description="hook type: question")
        b = ProductionPattern(category="hook", description="hook type: shock")
        self.assertNotEqual(a.pattern_id, b.pattern_id)

    def test_to_dict_from_dict_round_trip(self):
        pattern = ProductionPattern(category="camera", description="distance: close_up", supporting_video_count=3, confidence="medium", example_video_ids=["v1", "v2"])
        restored = ProductionPattern.from_dict(pattern.to_dict())
        self.assertEqual(pattern.pattern_id, restored.pattern_id)
        self.assertEqual(restored.supporting_video_count, 3)

    def test_tampered_pattern_id_raises_on_load(self):
        pattern = ProductionPattern(category="camera", description="distance: close_up")
        payload = pattern.to_dict()
        payload["pattern_id"] = "tampered"
        with self.assertRaises(KnowledgeBaseError):
            ProductionPattern.from_dict(payload)

    def test_no_creator_label_field_exists(self):
        """Structural de-identification guarantee: ProductionPattern
        has no field that could carry a creator_label."""
        pattern = ProductionPattern(category="hook", description="hook type: question")
        self.assertNotIn("creator_label", pattern.to_dict())


class IngestVideoDnaTests(unittest.TestCase):
    def test_ingest_creates_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            dna, evidence = _video("creator_one")
            patterns = kb.ingest_video_dna(dna, evidence, **_ingest_kwargs())
            self.assertTrue(patterns)
            self.assertEqual(patterns[0].supporting_video_count, 1)

    def test_two_different_videos_increase_supporting_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            dna1, ev1 = _video("creator_one", reference="ref1")
            dna2, ev2 = _video("creator_two", reference="ref2")
            kb.ingest_video_dna(dna1, ev1, **_ingest_kwargs())
            patterns = kb.ingest_video_dna(dna2, ev2, **_ingest_kwargs())
            hook_pattern = next(p for p in patterns if p.category == "hook")
            self.assertEqual(hook_pattern.supporting_video_count, 2)

    def test_reingesting_same_video_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            dna, evidence = _video("creator_one")
            kb.ingest_video_dna(dna, evidence, **_ingest_kwargs())
            patterns = kb.ingest_video_dna(dna, evidence, **_ingest_kwargs())
            self.assertEqual(patterns[0].supporting_video_count, 1)

    def test_confidence_grows_with_corroboration(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            confidences = []
            for i in range(5):
                dna, evidence = _video(f"creator_{i}", reference=f"ref_{i}")
                patterns = kb.ingest_video_dna(dna, evidence, **_ingest_kwargs())
                hook_pattern = next(p for p in patterns if p.category == "hook")
                confidences.append(hook_pattern.confidence)
            # confidence should never regress as more independent videos corroborate
            from src.video_intelligence.models import confidence_rank
            ranks = [confidence_rank(c) for c in confidences]
            self.assertEqual(ranks, sorted(ranks))
            self.assertGreater(ranks[-1], ranks[0])

    def test_example_video_ids_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            for i in range(10):
                dna, evidence = _video(f"creator_{i}", reference=f"ref_{i}")
                patterns = kb.ingest_video_dna(dna, evidence, **_ingest_kwargs())
            hook_pattern = next(p for p in patterns if p.category == "hook")
            self.assertEqual(hook_pattern.supporting_video_count, 10)
            self.assertLessEqual(len(hook_pattern.example_video_ids), 5)

    def test_unrelated_video_does_not_pollute_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            dna1, ev1 = _video("creator_one", extra_tags=("hook", "question"))
            dna2, ev2 = _video("creator_two", reference="ref2", extra_tags=("camera", "close_up"))
            kb.ingest_video_dna(dna1, ev1, **_ingest_kwargs())
            patterns2 = kb.ingest_video_dna(dna2, ev2, **_ingest_kwargs())
            categories = {p.category for p in patterns2}
            self.assertEqual(categories, {"camera"})

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            dna, evidence = _video("creator_one")
            kb.ingest_video_dna(dna, evidence, **_ingest_kwargs())
            reloaded = VideoKnowledgeBase(Path(tmp))
            self.assertEqual(len(reloaded.load_patterns()), len(kb.load_patterns()))

    def test_list_patterns_filters_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            dna, evidence = _video("creator_one")
            kb.ingest_video_dna(dna, evidence, **_ingest_kwargs())
            hook_patterns = kb.list_patterns(category="hook")
            self.assertTrue(all(p.category == "hook" for p in hook_patterns))

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = VideoKnowledgeBase(Path(tmp))
            kb.patterns_path.parent.mkdir(parents=True, exist_ok=True)
            kb.patterns_path.write_text("{not valid")
            with self.assertRaises(KnowledgeBaseError):
                kb.load_patterns()


if __name__ == "__main__":
    unittest.main()
