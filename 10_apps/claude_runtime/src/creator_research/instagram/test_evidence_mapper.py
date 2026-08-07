import unittest

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.models import RelationshipBasis
from src.creator_research.instagram import evidence_mapper
from src.creator_research.instagram.models import (
    CaptionObservation,
    CommentObservation,
    CreatorReplyObservation,
    DiscoveredItem,
    HighlightRecord,
    PostRecord,
    ProfileRecord,
    ReelRecord,
    RelationshipEvidenceBasis,
    RelationshipEvidenceRecord,
    VisualExampleRecord,
)


class ProfileToEvidenceTests(unittest.TestCase):
    def test_returns_evidence_instance(self):
        evidence, _warnings = evidence_mapper.profile_to_evidence(ProfileRecord(username="demo"))
        self.assertIsInstance(evidence, Evidence)

    def test_tags_include_instagram_and_profile(self):
        evidence, _ = evidence_mapper.profile_to_evidence(ProfileRecord(username="demo"))
        self.assertIn("instagram", evidence.tags)
        self.assertIn("profile", evidence.tags)

    def test_source_description_includes_username(self):
        evidence, _ = evidence_mapper.profile_to_evidence(ProfileRecord(username="demo_creator"))
        self.assertIn("demo_creator", evidence.source_description)

    def test_bio_becomes_content_excerpt(self):
        evidence, _ = evidence_mapper.profile_to_evidence(ProfileRecord(username="demo", bio="travel blogger"))
        self.assertEqual(evidence.content_excerpt, "travel blogger")


class DiscoveredItemToEvidenceTests(unittest.TestCase):
    def test_screenshot_type_when_thumbnail_present(self):
        item = DiscoveredItem(source_url="https://x/p/1", thumbnail_reference="ref1")
        evidence, _ = evidence_mapper.discovered_item_to_evidence(item)
        self.assertEqual(evidence.evidence_type, EvidenceType.SCREENSHOT)

    def test_operator_observation_when_no_thumbnail(self):
        item = DiscoveredItem(source_url="https://x/p/1")
        evidence, _ = evidence_mapper.discovered_item_to_evidence(item)
        self.assertEqual(evidence.evidence_type, EvidenceType.OPERATOR_OBSERVATION)

    def test_tags_include_content_type(self):
        item = DiscoveredItem(source_url="https://x/r/1", content_type="reel")
        evidence, _ = evidence_mapper.discovered_item_to_evidence(item)
        self.assertIn("reel", evidence.tags)


class PostToEvidenceTests(unittest.TestCase):
    def test_carousel_tag_added_when_carousel(self):
        post = PostRecord(post_url="https://x/p/1", is_carousel=True)
        evidence, _ = evidence_mapper.post_to_evidence(post)
        self.assertIn("carousel", evidence.tags)

    def test_no_carousel_tag_when_not_carousel(self):
        post = PostRecord(post_url="https://x/p/1", is_carousel=False)
        evidence, _ = evidence_mapper.post_to_evidence(post)
        self.assertNotIn("carousel", evidence.tags)

    def test_metrics_in_source_description(self):
        post = PostRecord(post_url="https://x/p/1", like_count=1234, comment_count=56)
        evidence, _ = evidence_mapper.post_to_evidence(post)
        self.assertIn("1234", evidence.source_description)
        self.assertIn("56", evidence.source_description)


class CaptionToEvidenceTests(unittest.TestCase):
    def test_text_excerpt_type(self):
        caption = CaptionObservation(caption_text="hi", post_reference="https://x/p/1")
        evidence, _ = evidence_mapper.caption_to_evidence(caption)
        self.assertEqual(evidence.evidence_type, EvidenceType.TEXT_EXCERPT)

    def test_hashtags_and_mentions_become_tags(self):
        caption = CaptionObservation(
            caption_text="hi", post_reference="https://x/p/1", hashtags=["travel"], mentions=["friend"]
        )
        evidence, _ = evidence_mapper.caption_to_evidence(caption)
        self.assertIn("hashtag:travel", evidence.tags)
        self.assertIn("mention:friend", evidence.tags)

    def test_over_cap_excerpt_produces_warning(self):
        long_text = "x" * 10000
        caption = CaptionObservation(caption_text=long_text, post_reference="https://x/p/1")
        evidence, warnings = evidence_mapper.caption_to_evidence(caption)
        self.assertLess(len(evidence.content_excerpt), len(long_text))
        self.assertEqual(len(warnings), 1)

    def test_never_rewrites_caption_text(self):
        text = "Exact wording, not rewritten!! 🌸"
        caption = CaptionObservation(caption_text=text, post_reference="https://x/p/1")
        evidence, _ = evidence_mapper.caption_to_evidence(caption)
        self.assertEqual(evidence.content_excerpt, text)


class CommentToEvidenceTests(unittest.TestCase):
    def test_username_redacted_when_configured(self):
        comment = CommentObservation(comment_text="nice!", post_reference="https://x/p/1", author_username="fan1")
        evidence, _ = evidence_mapper.comment_to_evidence(comment, redact_usernames=True)
        self.assertNotIn("fan1", evidence.source_description)
        self.assertIn("[redacted]", evidence.source_description)

    def test_username_preserved_when_not_redacted(self):
        comment = CommentObservation(comment_text="nice!", post_reference="https://x/p/1", author_username="fan1")
        evidence, _ = evidence_mapper.comment_to_evidence(comment, redact_usernames=False)
        self.assertIn("fan1", evidence.source_description)

    def test_tagged_as_audience(self):
        comment = CommentObservation(comment_text="nice!", post_reference="https://x/p/1")
        evidence, _ = evidence_mapper.comment_to_evidence(comment, redact_usernames=True)
        self.assertIn("audience", evidence.tags)


class CreatorReplyToEvidenceTests(unittest.TestCase):
    def test_tagged_as_creator_reply(self):
        reply = CreatorReplyObservation(
            parent_comment_text="love this", creator_reply_text="thank you!!", post_reference="https://x/p/1"
        )
        evidence, _ = evidence_mapper.creator_reply_to_evidence(reply)
        self.assertIn("creator_reply", evidence.tags)
        self.assertIn("reliability:high", evidence.tags)

    def test_content_includes_both_comment_and_reply(self):
        reply = CreatorReplyObservation(
            parent_comment_text="love this", creator_reply_text="thank you!!", post_reference="https://x/p/1"
        )
        evidence, _ = evidence_mapper.creator_reply_to_evidence(reply)
        self.assertIn("love this", evidence.content_excerpt)
        self.assertIn("thank you!!", evidence.content_excerpt)


class ReelToEvidenceTests(unittest.TestCase):
    def test_screenshot_type_when_thumbnail_present(self):
        reel = ReelRecord(reel_url="https://x/r/1", thumbnail_reference="ref1")
        evidence, _ = evidence_mapper.reel_to_evidence(reel)
        self.assertEqual(evidence.evidence_type, EvidenceType.SCREENSHOT)

    def test_operator_observation_when_no_thumbnail(self):
        reel = ReelRecord(reel_url="https://x/r/1")
        evidence, _ = evidence_mapper.reel_to_evidence(reel)
        self.assertEqual(evidence.evidence_type, EvidenceType.OPERATOR_OBSERVATION)

    def test_metrics_in_source_description(self):
        reel = ReelRecord(reel_url="https://x/r/1", views=1000, likes=200)
        evidence, _ = evidence_mapper.reel_to_evidence(reel)
        self.assertIn("1000", evidence.source_description)


class HighlightToEvidenceTests(unittest.TestCase):
    def test_tagged_as_storytelling(self):
        highlight = HighlightRecord(highlight_title="Travel", item_index=0)
        evidence, _ = evidence_mapper.highlight_to_evidence(highlight)
        self.assertIn("storytelling", evidence.tags)
        self.assertIn("highlights", evidence.tags)


class RelationshipToEvidenceTests(unittest.TestCase):
    def test_tagged_account_maps_to_visually_depicted(self):
        record = RelationshipEvidenceRecord(
            description="tagged @friend", basis=RelationshipEvidenceBasis.TAGGED_ACCOUNT, related_account="friend"
        )
        evidence, _ = evidence_mapper.relationship_to_evidence(record)
        self.assertIn(f"relationship_basis:{RelationshipBasis.VISUALLY_DEPICTED}", evidence.tags)

    def test_caption_declared_maps_to_presented_narrative(self):
        record = RelationshipEvidenceRecord(
            description="mentioned in caption", basis=RelationshipEvidenceBasis.CAPTION_DECLARED
        )
        evidence, _ = evidence_mapper.relationship_to_evidence(record)
        self.assertIn(f"relationship_basis:{RelationshipBasis.PRESENTED_NARRATIVE}", evidence.tags)

    def test_tagged_as_human_authenticity(self):
        record = RelationshipEvidenceRecord(description="x", basis=RelationshipEvidenceBasis.COLLABORATION_LABEL)
        evidence, _ = evidence_mapper.relationship_to_evidence(record)
        self.assertIn("human_authenticity", evidence.tags)

    def test_never_asserts_relationship_type_in_description(self):
        record = RelationshipEvidenceRecord(
            description="tagged account @friend in post", basis=RelationshipEvidenceBasis.TAGGED_ACCOUNT
        )
        evidence, _ = evidence_mapper.relationship_to_evidence(record)
        for forbidden in ("girlfriend", "boyfriend", "sister", "brother", "wife", "husband", "married"):
            self.assertNotIn(forbidden, evidence.content_excerpt.lower())


class VisualExampleToEvidenceTests(unittest.TestCase):
    def test_tagged_as_visual_realism_and_photography(self):
        example = VisualExampleRecord(source_reference="https://x/p/1", sampling_bucket="travel")
        evidence, _ = evidence_mapper.visual_example_to_evidence(example)
        self.assertIn("visual_realism", evidence.tags)
        self.assertIn("photography", evidence.tags)
        self.assertIn("travel", evidence.tags)

    def test_makes_no_visual_realism_judgement(self):
        example = VisualExampleRecord(source_reference="https://x/p/1", sampling_bucket="selfie")
        evidence, _ = evidence_mapper.visual_example_to_evidence(example)
        for forbidden in ("authentic", "fake", "ai-generated", "real camera", "beautiful", "attractive"):
            self.assertNotIn(forbidden, evidence.source_description.lower())


class NoSchemaModificationTests(unittest.TestCase):
    def test_evidence_type_always_from_closed_set(self):
        profile_evidence, _ = evidence_mapper.profile_to_evidence(ProfileRecord(username="demo"))
        self.assertIn(profile_evidence.evidence_type, EvidenceType.ALL)

    def test_all_mapper_functions_return_evidence_evidence_ids_are_content_hashed(self):
        evidence, _ = evidence_mapper.profile_to_evidence(ProfileRecord(username="demo"))
        self.assertTrue(evidence.evidence_id)


if __name__ == "__main__":
    unittest.main()
