import tempfile
import unittest

from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.exceptions import KnowledgeBaseError
from src.video_intelligence.talking_ai.knowledge_base import TalkingAIKnowledgeBase, TalkingAIProductionPattern
from src.video_intelligence.talking_ai.models import NATURALNESS_DIMENSION_ALIASES, TalkingAIConfig
from src.video_intelligence.talking_ai.production_dna import build_talking_ai_dna


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


def _make_video(video_id, *, gaze="direct_camera", camera="static"):
    segments = [
        SpeechSegment(video_id=video_id, start_seconds=0.0, end_seconds=2.0, pause_before=0.0, language="cantonese"),
        SpeechSegment(video_id=video_id, start_seconds=2.6, end_seconds=5.0, pause_before=0.6, language="cantonese"),
    ]
    evidence = [
        TalkingAIEvidence(
            video_id=video_id, timestamp_seconds=t, speech_active=(t < 2.0 or 2.6 <= t <= 5.0),
            mouth_open_ratio=0.3 if (t < 2.0 or 2.6 <= t <= 5.0) else 0.02, mouth_shape_category="open",
            blink=(t in (0.5, 1.0, 3.5)), gaze_direction=gaze, eye_contact=True,
            head_yaw=0.1 * t, head_motion_intensity=0.2, facial_expression="neutral", smile_intensity=0.1,
            left_hand_motion=0.2, right_hand_motion=0.2, gesture_type="still",
            subtitle_visible=True, subtitle_change=(t in (0.0, 2.6)), camera_motion=camera,
        )
        for t in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0]
    ]
    return evidence, segments


def _dna_for(video_id, config, **kwargs):
    evidence, segments = _make_video(video_id, **kwargs)
    return build_talking_ai_dna(video_id, f"demo {video_id}", evidence, segments, config), evidence, segments


class PatternIngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kb = TalkingAIKnowledgeBase(self.tmp.name)
        self.config = _config()

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_knowledge_base_has_no_patterns(self):
        self.assertEqual(self.kb.list_patterns(), [])

    def test_ingest_extracts_closed_vocabulary_patterns(self):
        dna, evidence, segments = _dna_for("vidA", self.config)
        patterns = self.kb.ingest_talking_ai_dna(dna, evidence=evidence, speech_segments=segments)
        categories = {pattern.category for pattern in patterns}
        self.assertIn("gaze", categories)
        self.assertIn("framing", categories)

    def test_breath_like_pause_pattern_uses_required_language(self):
        dna, evidence, segments = _dna_for("vidA", self.config)
        patterns = self.kb.ingest_talking_ai_dna(dna, evidence=evidence, speech_segments=segments)
        pause_patterns = [p for p in patterns if p.category == "pause_behavior"]
        self.assertTrue(pause_patterns)
        self.assertIn("breath-like pause behavior", pause_patterns[0].description)

    def test_corroboration_grows_supporting_video_count(self):
        dna1, ev1, seg1 = _dna_for("vidA", self.config)
        self.kb.ingest_talking_ai_dna(dna1, evidence=ev1, speech_segments=seg1)
        dna2, ev2, seg2 = _dna_for("vidB", self.config)
        self.kb.ingest_talking_ai_dna(dna2, evidence=ev2, speech_segments=seg2)
        gaze_patterns = self.kb.list_patterns(category="gaze")
        self.assertEqual(gaze_patterns[0].supporting_video_count, 2)

    def test_idempotent_reingest_never_double_counts(self):
        dna, evidence, segments = _dna_for("vidA", self.config)
        self.kb.ingest_talking_ai_dna(dna, evidence=evidence, speech_segments=segments)
        before = {p.pattern_id: p.supporting_video_count for p in self.kb.list_patterns()}
        self.kb.ingest_talking_ai_dna(dna, evidence=evidence, speech_segments=segments)
        after = {p.pattern_id: p.supporting_video_count for p in self.kb.list_patterns()}
        self.assertEqual(before, after)

    def test_evidence_coverage_is_mean_across_supporting_videos(self):
        dna, evidence, segments = _dna_for("vidA", self.config)
        patterns = self.kb.ingest_talking_ai_dna(dna, evidence=evidence, speech_segments=segments)
        for pattern in patterns:
            self.assertGreaterEqual(pattern.evidence_coverage, 0.0)
            self.assertLessEqual(pattern.evidence_coverage, 1.0)

    def test_example_video_ids_capped_at_five(self):
        for index in range(7):
            dna, evidence, segments = _dna_for(f"vid{index}", self.config)
            self.kb.ingest_talking_ai_dna(dna, evidence=evidence, speech_segments=segments)
        gaze_patterns = self.kb.list_patterns(category="gaze")
        self.assertLessEqual(len(gaze_patterns[0].example_video_ids), 5)
        self.assertEqual(gaze_patterns[0].supporting_video_count, 7)

    def test_distinct_gaze_directions_produce_distinct_patterns(self):
        dna_a, ev_a, seg_a = _dna_for("vidA", self.config, gaze="direct_camera")
        dna_b, ev_b, seg_b = _dna_for("vidB", self.config, gaze="away_left")
        self.kb.ingest_talking_ai_dna(dna_a, evidence=ev_a, speech_segments=seg_a)
        self.kb.ingest_talking_ai_dna(dna_b, evidence=ev_b, speech_segments=seg_b)
        gaze_patterns = self.kb.list_patterns(category="gaze")
        descriptions = {pattern.description for pattern in gaze_patterns}
        self.assertIn("gaze direction observed: direct_camera", descriptions)
        self.assertIn("gaze direction observed: away_left", descriptions)


class DeidentificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kb = TalkingAIKnowledgeBase(self.tmp.name)
        self.config = _config()

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_pattern_field_carries_a_creator_label_or_username(self):
        dna, evidence, segments = _dna_for("vidA", self.config)
        patterns = self.kb.ingest_talking_ai_dna(dna, evidence=evidence, speech_segments=segments)
        field_names = {f for f in TalkingAIProductionPattern.__slots__}
        self.assertNotIn("creator_label", field_names)
        self.assertNotIn("username", field_names)

    def test_no_description_reproduces_a_transcript(self):
        segments = [SpeechSegment(video_id="vidA", start_seconds=0.0, end_seconds=2.0, text_optional="a secret verbatim line")]
        evidence = [TalkingAIEvidence(video_id="vidA", timestamp_seconds=0.1, gaze_direction="direct_camera")]
        dna = build_talking_ai_dna("vidA", "demo", evidence, segments, self.config)
        patterns = self.kb.ingest_talking_ai_dna(dna, evidence=evidence, speech_segments=segments)
        for pattern in patterns:
            self.assertNotIn("a secret verbatim line", pattern.description)


class PatternSerializationIntegrityTests(unittest.TestCase):
    def test_pattern_id_recomputed_on_load(self):
        pattern = TalkingAIProductionPattern(category="gaze", description="gaze direction observed: direct_camera")
        payload = pattern.to_dict()
        reloaded = TalkingAIProductionPattern.from_dict(payload)
        self.assertEqual(reloaded.pattern_id, pattern.pattern_id)

    def test_tampered_pattern_id_raises(self):
        pattern = TalkingAIProductionPattern(category="gaze", description="gaze direction observed: direct_camera")
        payload = pattern.to_dict()
        payload["pattern_id"] = "deadbeefdeadbeef"
        with self.assertRaises(KnowledgeBaseError):
            TalkingAIProductionPattern.from_dict(payload)

    def test_invalid_json_on_disk_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = TalkingAIKnowledgeBase(tmp)
            kb.patterns_path.parent.mkdir(parents=True, exist_ok=True)
            kb.patterns_path.write_text("{not valid json")
            with self.assertRaises(KnowledgeBaseError):
                kb.load_patterns()


if __name__ == "__main__":
    unittest.main()
