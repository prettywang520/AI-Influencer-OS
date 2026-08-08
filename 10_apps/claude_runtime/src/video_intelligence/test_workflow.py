import tempfile
import unittest
from pathlib import Path

from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType, VideoPlatform, VideoRecord
from src.video_intelligence.production_dna import TRAIT_FIELDS, load_engine_config
from src.video_intelligence.workflow import VideoIntelligenceWorkflow


def _workflow(tmp_path):
    config = load_engine_config()
    config.output_root = str(Path(tmp_path) / "vi_output")
    return VideoIntelligenceWorkflow(config)


def _record(creator_label="reference_creator", reference="illustrative_example"):
    return VideoRecord(platform=VideoPlatform.INSTAGRAM_REELS, creator_label=creator_label, reference=reference)


def _note(video_id, tags, timestamp=None, description="note"):
    return VideoEvidence(
        video_id=video_id, evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description=description,
        content_excerpt="synthetic observation", timestamp_seconds=timestamp, tags=tags,
    )


def _full_evidence_set(video_id):
    """One synthetic observation per analysis domain -- exercises all
    13 analyzers plus story_beats/cta_observations end-to-end."""
    return [
        _note(video_id, ["hook", "first_second", "question"], timestamp=0.5, description="hook"),
        _note(video_id, ["pacing", "fast"], description="pacing"),
        _note(video_id, ["speech", "cantonese", "warm"], description="speech"),
        _note(video_id, ["lipsync", "naturalness"], description="lipsync"),
        _note(video_id, ["gesture", "hand_movement"], description="gesture"),
        _note(video_id, ["camera", "close_up", "handheld"], description="camera"),
        _note(video_id, ["framing", "rule_of_thirds"], description="framing"),
        _note(video_id, ["editing", "fast_cut"], description="editing"),
        _note(video_id, ["subtitles", "bold_caption"], description="subtitles"),
        _note(video_id, ["audio", "voice", "music"], description="audio"),
        _note(video_id, ["storytelling", "problem"], description="story beginning describes a common problem"),
        _note(video_id, ["cta", "comment"], timestamp=12.0, description="cta comment"),
        _note(video_id, ["emotion", "primary_emotion"], description="emotion"),
    ]


class AnalyzeVideoEndToEndTests(unittest.TestCase):
    def test_all_13_analyzers_receive_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = _workflow(tmp)
            record = _record()
            evidence = _full_evidence_set(record.video_id)
            result = workflow.analyze_video(record, evidence)
            for name in TRAIT_FIELDS:
                trait = getattr(result.dna, name)
                self.assertIsNotNone(trait, f"{name} trait should not be None")
                self.assertNotEqual(trait.confidence, "unknown", f"{name} should have evidenced confidence")

    def test_story_beats_and_cta_observations_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = _workflow(tmp)
            record = _record()
            evidence = _full_evidence_set(record.video_id)
            result = workflow.analyze_video(record, evidence)
            self.assertEqual(len(result.dna.story_beats), 1)
            self.assertEqual(result.dna.story_beats[0].beat_type, "problem")
            self.assertEqual(len(result.dna.cta_observations), 1)
            self.assertEqual(result.dna.cta_observations[0].cta_type, "comment")

    def test_dna_and_report_persisted_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = _workflow(tmp)
            record = _record()
            result = workflow.analyze_video(record, _full_evidence_set(record.video_id))
            loaded_dna = workflow.load_dna(record.video_id)
            self.assertEqual(loaded_dna.dna_id, result.dna.dna_id)
            markdown = workflow.load_report_markdown(record.video_id)
            self.assertIn(record.video_id, markdown)

    def test_knowledge_base_ingestion_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = _workflow(tmp)
            record = _record()
            result = workflow.analyze_video(record, _full_evidence_set(record.video_id), ingest_into_knowledge_base=False)
            self.assertEqual(result.patterns, [])
            self.assertEqual(workflow.knowledge_base().list_patterns(), [])

    def test_knowledge_base_ingestion_populates_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = _workflow(tmp)
            record = _record()
            result = workflow.analyze_video(record, _full_evidence_set(record.video_id))
            self.assertTrue(result.patterns)
            self.assertTrue(workflow.knowledge_base().list_patterns())

    def test_stable_video_id_and_deterministic_dna_id(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            record_a = _record()
            record_b = _record()
            self.assertEqual(record_a.video_id, record_b.video_id)

            result_a = _workflow(tmp_a).analyze_video(record_a, _full_evidence_set(record_a.video_id))
            result_b = _workflow(tmp_b).analyze_video(record_b, _full_evidence_set(record_b.video_id))
            self.assertEqual(result_a.dna.dna_id, result_b.dna.dna_id)

    def test_load_dna_for_unknown_video_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = _workflow(tmp)
            self.assertIsNone(workflow.load_dna("no_such_video"))
            self.assertIsNone(workflow.load_report_markdown("no_such_video"))

    def test_zero_evidence_video_still_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow = _workflow(tmp)
            record = _record()
            result = workflow.analyze_video(record, [])
            self.assertEqual(result.dna.overall_confidence, "unknown")
            self.assertEqual(result.patterns, [])


if __name__ == "__main__":
    unittest.main()
