import unittest

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType
from src.video_intelligence.instagram.mapper import (
    _kind_of,
    _parse_source_description,
    packet_to_video_evidence,
    reel_evidence_from_creator_research,
)
from src.video_intelligence.instagram.models import InstagramReelEvidencePacket


def _reel_evidence(url="https://www.instagram.com/reel/abc/", published_at="2026-08-01T00:00:00Z",
                    views="15000", likes="1200", comments="80", duration="14.5", caption="a coffee shop routine"):
    return Evidence(
        evidence_type=EvidenceType.OPERATOR_OBSERVATION,
        source_description=(
            f"instagram reel | url={url} | published_at={published_at} | views={views} "
            f"| likes={likes} | comments={comments} | duration_seconds={duration}"
        ),
        content_excerpt=caption,
        collected_by="instagram_research_connector",
        tags=["instagram", "reels", "reliability:medium"],
    )


def _linked_evidence(kind, reference, tags, content_excerpt="linked note"):
    return Evidence(
        evidence_type=EvidenceType.TEXT_EXCERPT,
        source_description=f"instagram {kind} | post={reference} | published_at=2026-08-01T00:00:00Z",
        content_excerpt=content_excerpt,
        collected_by="instagram_research_connector",
        tags=tags,
    )


class ParseSourceDescriptionTests(unittest.TestCase):
    def test_parses_key_value_pairs(self):
        parsed = _parse_source_description("instagram reel | url=https://x/y | views=100")
        self.assertEqual(parsed["url"], "https://x/y")
        self.assertEqual(parsed["views"], "100")

    def test_empty_string_yields_empty_dict(self):
        self.assertEqual(_parse_source_description(""), {})

    def test_segment_without_equals_is_ignored(self):
        parsed = _parse_source_description("instagram reel | url=https://x/y | malformed_segment")
        self.assertEqual(parsed, {"url": "https://x/y"})

    def test_kind_of_extracts_prefix(self):
        self.assertEqual(_kind_of("instagram reel | url=https://x/y"), "instagram reel")
        self.assertEqual(_kind_of(""), "")


class ReelEvidenceFromCreatorResearchTests(unittest.TestCase):
    def test_platform_mapping_produces_one_packet_per_reel(self):
        packets = reel_evidence_from_creator_research([_reel_evidence()])
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].reel_url, "https://www.instagram.com/reel/abc/")

    def test_metrics_parsed_as_numbers(self):
        packets = reel_evidence_from_creator_research([_reel_evidence()])
        packet = packets[0]
        self.assertEqual(packet.views, 15000)
        self.assertEqual(packet.likes, 1200)
        self.assertEqual(packet.comments, 80)
        self.assertAlmostEqual(packet.duration_seconds, 14.5)

    def test_timestamps_preserved(self):
        packets = reel_evidence_from_creator_research([_reel_evidence(published_at="2026-07-15T08:00:00Z")])
        self.assertEqual(packets[0].published_at, "2026-07-15T08:00:00Z")

    def test_caption_preserved_as_content_excerpt(self):
        packets = reel_evidence_from_creator_research([_reel_evidence(caption="come with me to get coffee")])
        self.assertEqual(packets[0].caption, "come with me to get coffee")

    def test_python_none_placeholder_parsed_as_none(self):
        packets = reel_evidence_from_creator_research(
            [_reel_evidence(published_at="None", views="None", likes="None", comments="None", duration="None")]
        )
        packet = packets[0]
        self.assertIsNone(packet.published_at)
        self.assertIsNone(packet.views)
        self.assertIsNone(packet.duration_seconds)

    def test_reel_without_url_is_skipped_not_guessed(self):
        broken = Evidence(
            evidence_type=EvidenceType.OPERATOR_OBSERVATION,
            source_description="instagram reel | published_at=2026-08-01T00:00:00Z",
            content_excerpt="", collected_by="instagram_research_connector", tags=["instagram", "reels"],
        )
        packets = reel_evidence_from_creator_research([broken])
        self.assertEqual(packets, [])

    def test_non_reel_evidence_is_ignored(self):
        profile_evidence = Evidence(
            evidence_type=EvidenceType.OPERATOR_OBSERVATION,
            source_description="instagram profile | username=someone | followers=100",
            content_excerpt="", collected_by="instagram_research_connector", tags=["instagram", "profile"],
        )
        packets = reel_evidence_from_creator_research([profile_evidence])
        self.assertEqual(packets, [])

    def test_evidence_linkage_by_matching_url(self):
        reel = _reel_evidence()
        linked = _linked_evidence("caption", "https://www.instagram.com/reel/abc/", tags=["instagram", "caption", "cta", "follow"])
        packets = reel_evidence_from_creator_research([reel, linked])
        packet = packets[0]
        self.assertIn(linked.evidence_id, packet.source_evidence_ids)
        self.assertEqual(len(packet.annotations), 1)
        self.assertEqual(packet.annotations[0].tags, ["instagram", "caption", "cta", "follow"])

    def test_unlinked_evidence_for_different_reel_not_folded_in(self):
        reel = _reel_evidence(url="https://www.instagram.com/reel/abc/")
        linked = _linked_evidence("caption", "https://www.instagram.com/reel/other/", tags=["instagram", "caption", "cta"])
        packets = reel_evidence_from_creator_research([reel, linked])
        self.assertEqual(packets[0].annotations, [])

    def test_linked_evidence_without_domain_tag_not_folded_into_annotations(self):
        reel = _reel_evidence()
        linked = _linked_evidence(
            "caption", "https://www.instagram.com/reel/abc/", tags=["instagram", "caption", "hashtag:coffee"]
        )
        packets = reel_evidence_from_creator_research([reel, linked])
        packet = packets[0]
        self.assertIn(linked.evidence_id, packet.source_evidence_ids)  # still traceable
        self.assertEqual(packet.annotations, [])  # but not handed to any analyzer

    def test_creator_label_passed_through_never_inferred(self):
        packets = reel_evidence_from_creator_research([_reel_evidence()], creator_label="reference_creator")
        self.assertEqual(packets[0].creator_label, "reference_creator")

    def test_evidence_linkage_via_creator_reply(self):
        reel = _reel_evidence()
        reply = _linked_evidence("creator reply", "https://www.instagram.com/reel/abc/", tags=["instagram", "reply", "creator_reply", "cta"])
        packets = reel_evidence_from_creator_research([reel, reply])
        self.assertEqual(len(packets[0].annotations), 1)


class PacketToVideoEvidenceTests(unittest.TestCase):
    def test_structural_summary_item_tagged_pacing_when_duration_known(self):
        packet = InstagramReelEvidencePacket(reel_url="https://x/y", duration_seconds=10.0)
        items = packet_to_video_evidence(packet)
        self.assertEqual(len(items), 1)
        self.assertIn("pacing", items[0].tags)
        self.assertIn("instagram_reels", items[0].tags)

    def test_no_pacing_tag_when_duration_unknown(self):
        packet = InstagramReelEvidencePacket(reel_url="https://x/y", views=100)
        items = packet_to_video_evidence(packet)
        self.assertEqual(len(items), 1)
        self.assertNotIn("pacing", items[0].tags)

    def test_no_structural_item_when_nothing_known(self):
        packet = InstagramReelEvidencePacket(reel_url="https://x/y")
        self.assertEqual(packet_to_video_evidence(packet), [])

    def test_all_items_scoped_to_packet_video_id(self):
        annotation = VideoEvidence(
            video_id="placeholder", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
            source_description="hook", content_excerpt="x", tags=["hook", "question"],
        )
        packet = InstagramReelEvidencePacket(reel_url="https://x/y", duration_seconds=5.0, annotations=[annotation])
        items = packet_to_video_evidence(packet)
        self.assertTrue(all(item.video_id == packet.reel_id for item in items))

    def test_annotation_tags_and_timestamp_preserved(self):
        annotation = VideoEvidence(
            video_id="placeholder", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
            source_description="camera note", content_excerpt="close up shot", timestamp_seconds=3.5,
            tags=["camera", "close_up"],
        )
        packet = InstagramReelEvidencePacket(reel_url="https://x/y", annotations=[annotation])
        items = packet_to_video_evidence(packet)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].tags, ["camera", "close_up"])
        self.assertEqual(items[0].timestamp_seconds, 3.5)

    def test_multiple_annotations_each_become_their_own_item(self):
        annotations = [
            VideoEvidence(video_id="placeholder", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
                          source_description="a", content_excerpt="", tags=["hook", "question"]),
            VideoEvidence(video_id="placeholder", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
                          source_description="b", content_excerpt="", tags=["cta", "follow"]),
        ]
        packet = InstagramReelEvidencePacket(reel_url="https://x/y", annotations=annotations)
        items = packet_to_video_evidence(packet)
        self.assertEqual(len(items), 2)

    def test_end_to_end_from_creator_research_evidence(self):
        reel = _reel_evidence()
        linked = _linked_evidence("caption", "https://www.instagram.com/reel/abc/", tags=["instagram", "caption", "storytelling", "problem"])
        packets = reel_evidence_from_creator_research([reel, linked])
        items = packet_to_video_evidence(packets[0])
        tags_seen = {tag for item in items for tag in item.tags}
        self.assertIn("pacing", tags_seen)
        self.assertIn("storytelling", tags_seen)


# -- Per-domain "does the tag survive the round trip" coverage (task §27) --


class DomainEvidencePassThroughTests(unittest.TestCase):
    """Each of these constructs an annotation carrying that domain's
    own recognized tags and confirms it survives mapping unchanged --
    the adapter never re-tags, reinterprets, or drops a recognized
    domain annotation."""

    def _round_trip(self, tags, timestamp=None):
        annotation = VideoEvidence(
            video_id="placeholder", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
            source_description="note", content_excerpt="observation", timestamp_seconds=timestamp, tags=tags,
        )
        packet = InstagramReelEvidencePacket(reel_url="https://x/y", annotations=[annotation])
        items = packet_to_video_evidence(packet)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].tags, tags)
        return items[0]

    def test_hook_first_frame_first_second_first_3_seconds(self):
        self._round_trip(["hook", "first_frame"], timestamp=0.0)
        self._round_trip(["hook", "first_second"], timestamp=0.9)
        self._round_trip(["hook", "first_3_seconds"], timestamp=2.5)

    def test_speech_cantonese_mandarin_english_mixed(self):
        self._round_trip(["speech", "cantonese"])
        self._round_trip(["speech", "mandarin"])
        self._round_trip(["speech", "english"])
        self._round_trip(["speech", "mixed"])

    def test_speech_insufficient_evidence_case(self):
        # No speech-tagged annotation at all -> zero speech evidence,
        # never a fabricated neutral score (verified at analyzer level
        # in test_speech.py; here we confirm the adapter emits nothing).
        packet = InstagramReelEvidencePacket(reel_url="https://x/y", duration_seconds=5.0)
        items = packet_to_video_evidence(packet)
        self.assertFalse(any("speech" in item.tags for item in items))

    def test_lipsync_mouth_timing_pause_blink(self):
        self._round_trip(["lipsync", "mouth_timing"])
        self._round_trip(["lipsync", "lip_pause"])
        self._round_trip(["lipsync", "blink_timing"])

    def test_gesture_hand_head_gaze(self):
        self._round_trip(["gesture", "hand_movement"])
        self._round_trip(["gesture", "head_motion"])
        self._round_trip(["gesture", "eye_contact"])

    def test_camera_framing_direct_to_camera_selfie_tripod_friend_shot(self):
        self._round_trip(["camera", "close_up"])
        self._round_trip(["camera", "selfie"])
        self._round_trip(["camera", "tripod"])
        self._round_trip(["camera", "friend_shot"])
        self._round_trip(["framing", "rule_of_thirds"])

    def test_pacing_editing_shot_duration_cuts_jump_cuts_broll_loop(self):
        self._round_trip(["pacing", "fast"])
        self._round_trip(["editing", "jump_cut"])
        self._round_trip(["editing", "zoom"])

    def test_subtitles_presence_position_highlighting(self):
        self._round_trip(["subtitles", "bold_caption"])

    def test_audio_voice_music_ambient(self):
        self._round_trip(["audio", "voice"])
        self._round_trip(["audio", "music"])
        self._round_trip(["audio", "ambient"])

    def test_storytelling_hook_setup_payoff_cta(self):
        self._round_trip(["storytelling", "problem"])
        self._round_trip(["storytelling", "payoff"])

    def test_cta_follow_comment_share_save_question(self):
        self._round_trip(["cta", "follow"])
        self._round_trip(["cta", "comment"])
        self._round_trip(["cta", "share"])
        self._round_trip(["cta", "save"])
        self._round_trip(["cta", "question"])

    def test_cta_no_cta_case_produces_nothing(self):
        packet = InstagramReelEvidencePacket(reel_url="https://x/y", duration_seconds=5.0)
        items = packet_to_video_evidence(packet)
        self.assertFalse(any("cta" in item.tags for item in items))

    def test_emotion_primary_and_transition(self):
        self._round_trip(["emotion", "primary_emotion"])
        self._round_trip(["emotion", "emotion_change"])


if __name__ == "__main__":
    unittest.main()
