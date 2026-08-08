import unittest

from src.creator_learning.reports.markdown import (
    RENDERERS,
    render_all_reports,
    render_caption_report,
    render_learning_report,
    render_relationship_report,
    render_reply_report,
    render_style_report,
    render_visual_report,
)
from src.creator_learning.reports.models import (
    CaptionReport,
    LearningReport,
    RelationshipReport,
    ReplyReport,
    StyleReport,
    VisualReport,
)


class RenderLearningReportTests(unittest.TestCase):
    def test_renders_header_and_gaps(self):
        report = LearningReport(
            creator_id="c1", session_id="s1", generated_at="t", evidence_count=2,
            overall_confidence="low", session_count=1, confidence_trend=[("s1", "low")],
            gaps={"storytelling": ("coffee", "travel")},
        )
        text = render_learning_report(report)
        self.assertIn("# Learning Report -- c1", text)
        self.assertIn("storytelling", text)
        self.assertIn("coffee", text)

    def test_no_gaps_renders_placeholder(self):
        report = LearningReport(
            creator_id="c1", session_id="s1", generated_at="t", evidence_count=0,
            overall_confidence="unknown", session_count=0,
        )
        text = render_learning_report(report)
        self.assertIn("No tracked learning-target gaps.", text)


class RenderVisualReportTests(unittest.TestCase):
    def test_renders_scores_and_excerpts(self):
        report = VisualReport(
            creator_id="c1", session_id="s1", visual_realism_score=0.5, visual_realism_confidence="low",
            visual_realism_rationale="observed", photography_score=None, photography_confidence=None,
            photography_rationale="", cited_excerpts=["a warm shot"],
        )
        text = render_visual_report(report)
        self.assertIn("0.50", text)
        self.assertIn("a warm shot", text)

    def test_empty_excerpts_renders_placeholder(self):
        report = VisualReport(
            creator_id="c1", session_id="s1", visual_realism_score=None, visual_realism_confidence=None,
            visual_realism_rationale="", photography_score=None, photography_confidence=None, photography_rationale="",
        )
        text = render_visual_report(report)
        self.assertIn("(no excerpts cited yet)", text)


class RenderRelationshipReportTests(unittest.TestCase):
    def test_renders_claims_with_basis(self):
        report = RelationshipReport(
            creator_id="c1", session_id="s1", relationships_score=0.3, relationships_confidence="low",
            claims=[{"description": "has a sibling", "basis": "presented_narrative", "evidence_ids": ["e1"]}],
        )
        text = render_relationship_report(report)
        self.assertIn("has a sibling", text)
        self.assertIn("presented_narrative", text)

    def test_no_claims_renders_placeholder(self):
        report = RelationshipReport(creator_id="c1", session_id="s1", relationships_score=None, relationships_confidence=None)
        text = render_relationship_report(report)
        self.assertIn("(no relationship claims yet)", text)


class RenderReplyReportTests(unittest.TestCase):
    def test_never_generates_a_reply_disclaimer_present(self):
        report = ReplyReport(creator_id="c1", session_id="s1", replies_score=None, replies_confidence=None, rationale="")
        text = render_reply_report(report)
        self.assertIn("never generates a reply", text)


class RenderCaptionReportTests(unittest.TestCase):
    def test_renders_hashtag_and_mention_frequency(self):
        report = CaptionReport(
            creator_id="c1", session_id="s1", captions_score=0.4, captions_confidence="low", rationale="",
            hashtag_frequency={"coffee": 2}, mention_frequency={"friend": 1},
        )
        text = render_caption_report(report)
        self.assertIn("#coffee: 2", text)
        self.assertIn("@friend: 1", text)


class RenderStyleReportTests(unittest.TestCase):
    def test_baseline_renders_baseline_heading(self):
        report = StyleReport(
            creator_id="c1", session_id="s1", is_baseline=True, new_relationship_claim_count=0,
            changed_relationship_claim_count=0, new_warning_count=0, trait_summary=["persona: new baseline score 0.50 (low)"],
        )
        text = render_style_report(report)
        self.assertIn("## Baseline trait scores", text)
        self.assertIn("new baseline score", text)

    def test_non_baseline_renders_change_heading(self):
        report = StyleReport(
            creator_id="c1", session_id="s1", is_baseline=False, new_relationship_claim_count=1,
            changed_relationship_claim_count=0, new_warning_count=0, trait_summary=[],
        )
        text = render_style_report(report)
        self.assertIn("## Factual changes since last session", text)
        self.assertIn("No trait changes observed.", text)


class RenderAllReportsTests(unittest.TestCase):
    def test_renders_every_registered_report_type(self):
        reports = {
            "learning": LearningReport(creator_id="c1", session_id="s1", generated_at="t", evidence_count=0, overall_confidence="unknown", session_count=0),
            "visual": VisualReport(creator_id="c1", session_id="s1", visual_realism_score=None, visual_realism_confidence=None, visual_realism_rationale="", photography_score=None, photography_confidence=None, photography_rationale=""),
            "relationship": RelationshipReport(creator_id="c1", session_id="s1", relationships_score=None, relationships_confidence=None),
            "reply": ReplyReport(creator_id="c1", session_id="s1", replies_score=None, replies_confidence=None, rationale=""),
            "caption": CaptionReport(creator_id="c1", session_id="s1", captions_score=None, captions_confidence=None, rationale=""),
            "style": StyleReport(creator_id="c1", session_id="s1", is_baseline=True, new_relationship_claim_count=0, changed_relationship_claim_count=0, new_warning_count=0),
        }
        rendered = render_all_reports(reports)
        self.assertEqual(set(rendered.keys()), set(RENDERERS.keys()))
        for text in rendered.values():
            self.assertIsInstance(text, str)
            self.assertTrue(text.strip())

    def test_unknown_report_name_is_skipped_not_errored(self):
        rendered = render_all_reports({"not_a_real_report": object()})
        self.assertEqual(rendered, {})


if __name__ == "__main__":
    unittest.main()
