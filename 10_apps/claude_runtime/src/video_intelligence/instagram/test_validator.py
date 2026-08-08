import unittest

from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType
from src.video_intelligence.instagram.exceptions import AdapterValidationError
from src.video_intelligence.instagram.models import InstagramReelEvidencePacket
from src.video_intelligence.instagram.validator import require_valid_packet, validate_packet, validate_packets


def _packet(**overrides):
    defaults = dict(reel_url="https://www.instagram.com/reel/abc/", creator_label="ref")
    defaults.update(overrides)
    return InstagramReelEvidencePacket(**defaults)


class ValidatePacketTests(unittest.TestCase):
    def test_valid_packet_has_no_errors(self):
        packet = _packet(duration_seconds=10.0, views=100, likes=5, comments=1, published_at="2026-08-01T00:00:00Z")
        self.assertEqual(validate_packet(packet), [])

    def test_unknown_optional_fields_are_allowed(self):
        packet = InstagramReelEvidencePacket(reel_url="https://x/y")
        self.assertEqual(validate_packet(packet), [])

    def test_empty_reel_url_fails(self):
        packet = _packet(reel_url="")
        errors = validate_packet(packet)
        self.assertTrue(any("reel_url" in e for e in errors))

    def test_non_url_reel_url_fails(self):
        packet = _packet(reel_url="not-a-url")
        errors = validate_packet(packet)
        self.assertTrue(any("does not look like an absolute URL" in e for e in errors))

    def test_negative_duration_fails(self):
        packet = _packet(duration_seconds=-1.0)
        errors = validate_packet(packet)
        self.assertTrue(any("duration_seconds" in e for e in errors))

    def test_negative_views_likes_comments_fail(self):
        for field_name, value in (("views", -1), ("likes", -1), ("comments", -1)):
            packet = _packet(**{field_name: value})
            errors = validate_packet(packet)
            self.assertTrue(any(field_name in e for e in errors), f"{field_name} should have failed")

    def test_malformed_published_at_fails(self):
        packet = _packet(published_at="not-a-real-timestamp")
        errors = validate_packet(packet)
        self.assertTrue(any("published_at" in e for e in errors))

    def test_iso_timestamp_with_z_suffix_accepted(self):
        packet = _packet(published_at="2026-08-01T00:00:00Z")
        self.assertEqual(validate_packet(packet), [])

    def test_duplicate_evidence_within_packet_fails(self):
        annotation = VideoEvidence(
            video_id="x", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="same",
            content_excerpt="same", tags=["hook"],
        )
        packet = _packet(annotations=[annotation, annotation])
        errors = validate_packet(packet)
        self.assertTrue(any("duplicate VideoEvidence" in e for e in errors))

    def test_hand_built_annotation_with_placeholder_video_id_is_valid(self):
        # Regression: mapper.packet_to_video_evidence() always rewrites
        # video_id to the packet's own reel_id, so a hand-built
        # annotation with a placeholder/empty video_id must not fail
        # validation (this used to be incorrectly flagged).
        annotation = VideoEvidence(
            video_id="", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="hook",
            content_excerpt="", tags=["hook", "question"],
        )
        packet = _packet(annotations=[annotation])
        self.assertEqual(validate_packet(packet), [])

    def test_negative_annotation_timestamp_fails(self):
        annotation = VideoEvidence(
            video_id="", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="hook",
            content_excerpt="", timestamp_seconds=-1.0, tags=["hook"],
        )
        packet = _packet(annotations=[annotation])
        errors = validate_packet(packet)
        self.assertTrue(any("timestamp_seconds" in e for e in errors))

    def test_non_serializable_metadata_fails(self):
        packet = _packet(metadata={"bad": object()})
        errors = validate_packet(packet)
        self.assertTrue(any("metadata" in e for e in errors))


class RequireValidPacketTests(unittest.TestCase):
    def test_valid_packet_does_not_raise(self):
        require_valid_packet(_packet())

    def test_invalid_packet_raises_with_details(self):
        with self.assertRaises(AdapterValidationError) as ctx:
            require_valid_packet(_packet(reel_url=""))
        self.assertIn("reel_url", str(ctx.exception))


class ValidatePacketsBatchTests(unittest.TestCase):
    def test_all_valid_packets_pass_through(self):
        packets = [_packet(reel_url=f"https://x/{i}") for i in range(5)]
        valid, errors = validate_packets(packets)
        self.assertEqual(len(valid), 5)
        self.assertEqual(errors, {})

    def test_invalid_packet_excluded_and_reported(self):
        good = _packet(reel_url="https://x/good")
        bad = _packet(reel_url="")
        valid, errors = validate_packets([good, bad])
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].reel_url, "https://x/good")
        self.assertIn(bad.reel_id, errors)

    def test_duplicate_reel_id_first_occurrence_kept_second_reported(self):
        first = _packet(reel_url="https://www.instagram.com/reel/same/")
        second = _packet(reel_url="https://www.instagram.com/reel/same/")
        valid, errors = validate_packets([first, second])
        self.assertEqual(len(valid), 1)
        self.assertIn(second.reel_id, errors)
        self.assertTrue(any("duplicate reel_id" in e for e in errors[second.reel_id]))

    def test_batch_validation_is_order_independent_for_duplicate_detection(self):
        # sorted by reel_url internally, so the "first" kept is always
        # the same regardless of input order
        a = _packet(reel_url="https://www.instagram.com/reel/aaa/")
        b = _packet(reel_url="https://www.instagram.com/reel/aaa/")
        valid1, _ = validate_packets([a, b])
        valid2, _ = validate_packets([b, a])
        self.assertEqual(valid1[0].reel_url, valid2[0].reel_url)

    def test_empty_batch(self):
        valid, errors = validate_packets([])
        self.assertEqual(valid, [])
        self.assertEqual(errors, {})


if __name__ == "__main__":
    unittest.main()
