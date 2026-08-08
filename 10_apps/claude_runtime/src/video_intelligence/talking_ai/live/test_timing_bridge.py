import unittest

from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.models import ConfidenceLevel
from src.video_intelligence.talking_ai.live.timing_bridge import (
    APPROXIMATE_TIMING_NOTE,
    ORDINAL_PLACEHOLDER_NOTE,
    assign_ordinal_speech_segments,
    assign_ordinal_timestamps,
    mark_approximate_speech_segment,
    mark_approximate_timing,
)


class MarkApproximateTimingTests(unittest.TestCase):
    def test_timestamp_unchanged(self):
        item = TalkingAIEvidence(video_id="v", timestamp_seconds=0.45, mouth_open_ratio=0.3)
        marked = mark_approximate_timing(item)
        self.assertEqual(marked.timestamp_seconds, 0.45)

    def test_unknown_confidence_bumped_to_low(self):
        item = TalkingAIEvidence(video_id="v", timestamp_seconds=0.45)
        marked = mark_approximate_timing(item)
        self.assertEqual(marked.confidence, ConfidenceLevel.LOW)

    def test_high_confidence_capped_to_low(self):
        item = TalkingAIEvidence(video_id="v", timestamp_seconds=0.45, confidence=ConfidenceLevel.VERIFIED)
        marked = mark_approximate_timing(item)
        self.assertEqual(marked.confidence, ConfidenceLevel.LOW)

    def test_note_appended(self):
        item = TalkingAIEvidence(video_id="v", timestamp_seconds=0.45, notes="operator note")
        marked = mark_approximate_timing(item)
        self.assertIn(APPROXIMATE_TIMING_NOTE, marked.notes)
        self.assertIn("operator note", marked.notes)

    def test_note_not_duplicated_if_already_present(self):
        item = TalkingAIEvidence(video_id="v", timestamp_seconds=0.45, notes=APPROXIMATE_TIMING_NOTE)
        marked = mark_approximate_timing(item)
        self.assertEqual(marked.notes.count(APPROXIMATE_TIMING_NOTE), 1)

    def test_evidence_id_recomputed_consistently(self):
        item = TalkingAIEvidence(video_id="v", timestamp_seconds=0.45)
        marked = mark_approximate_timing(item)
        # notes changed, so evidence_id (which hashes notes) must differ
        self.assertNotEqual(marked.evidence_id, item.evidence_id)
        # but is itself deterministic
        marked_again = mark_approximate_timing(item)
        self.assertEqual(marked.evidence_id, marked_again.evidence_id)

    def test_original_item_not_mutated(self):
        item = TalkingAIEvidence(video_id="v", timestamp_seconds=0.45)
        mark_approximate_timing(item)
        self.assertEqual(item.confidence, ConfidenceLevel.UNKNOWN)
        self.assertEqual(item.notes, "")


class MarkApproximateSpeechSegmentTests(unittest.TestCase):
    def test_confidence_capped_low(self):
        segment = SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=1.0, confidence=ConfidenceLevel.HIGH)
        marked = mark_approximate_speech_segment(segment)
        self.assertEqual(marked.confidence, ConfidenceLevel.LOW)

    def test_timing_unchanged(self):
        segment = SpeechSegment(video_id="v", start_seconds=0.4, end_seconds=1.2)
        marked = mark_approximate_speech_segment(segment)
        self.assertEqual((marked.start_seconds, marked.end_seconds), (0.4, 1.2))


class AssignOrdinalTimestampsTests(unittest.TestCase):
    def test_strictly_increasing_preserving_order(self):
        items = [
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, blink=True),
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, gaze_direction="direct_camera"),
            TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, gesture_type="wave"),
        ]
        assigned = assign_ordinal_timestamps(items, start_seconds=0.0, step_seconds=0.5)
        self.assertEqual([item.timestamp_seconds for item in assigned], [0.0, 0.5, 1.0])
        self.assertTrue(assigned[0].blink)
        self.assertEqual(assigned[1].gaze_direction, "direct_camera")
        self.assertEqual(assigned[2].gesture_type, "wave")

    def test_confidence_forced_low_regardless_of_input(self):
        items = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, confidence=ConfidenceLevel.VERIFIED)]
        assigned = assign_ordinal_timestamps(items)
        self.assertEqual(assigned[0].confidence, ConfidenceLevel.LOW)

    def test_note_marker_present(self):
        items = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0)]
        assigned = assign_ordinal_timestamps(items)
        self.assertIn(ORDINAL_PLACEHOLDER_NOTE, assigned[0].notes)

    def test_custom_start_and_step(self):
        items = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0) for _ in range(3)]
        assigned = assign_ordinal_timestamps(items, start_seconds=10.0, step_seconds=2.0)
        self.assertEqual([item.timestamp_seconds for item in assigned], [10.0, 12.0, 14.0])

    def test_non_positive_step_raises(self):
        items = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0)]
        with self.assertRaises(ValueError):
            assign_ordinal_timestamps(items, step_seconds=0.0)
        with self.assertRaises(ValueError):
            assign_ordinal_timestamps(items, step_seconds=-1.0)

    def test_empty_list(self):
        self.assertEqual(assign_ordinal_timestamps([]), [])

    def test_deterministic_across_two_calls(self):
        items = [TalkingAIEvidence(video_id="v", timestamp_seconds=0.0, blink=True)]
        first = assign_ordinal_timestamps(items)
        second = assign_ordinal_timestamps(items)
        self.assertEqual(first[0].evidence_id, second[0].evidence_id)


class AssignOrdinalSpeechSegmentsTests(unittest.TestCase):
    def test_non_overlapping_intervals_preserving_order(self):
        segments = [
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=0.0, language="cantonese"),
            SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=0.0, language="english"),
        ]
        assigned = assign_ordinal_speech_segments(segments, start_seconds=0.0, step_seconds=2.0)
        self.assertEqual((assigned[0].start_seconds, assigned[0].end_seconds), (0.0, 2.0))
        self.assertEqual((assigned[1].start_seconds, assigned[1].end_seconds), (2.0, 4.0))
        self.assertEqual(assigned[0].language, "cantonese")
        self.assertEqual(assigned[1].language, "english")

    def test_confidence_forced_low(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=0.0, confidence=ConfidenceLevel.HIGH)]
        assigned = assign_ordinal_speech_segments(segments)
        self.assertEqual(assigned[0].confidence, ConfidenceLevel.LOW)

    def test_non_positive_step_raises(self):
        segments = [SpeechSegment(video_id="v", start_seconds=0.0, end_seconds=0.0)]
        with self.assertRaises(ValueError):
            assign_ordinal_speech_segments(segments, step_seconds=0.0)

    def test_empty_list(self):
        self.assertEqual(assign_ordinal_speech_segments([]), [])


if __name__ == "__main__":
    unittest.main()
