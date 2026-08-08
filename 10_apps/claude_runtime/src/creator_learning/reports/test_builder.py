import unittest

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.models import CreatorDNA, RelationshipClaim, TraitScore
from src.creator_learning.learning_history import LearningHistoryEntry
from src.creator_learning.reports.builder import (
    build_all_reports,
    build_caption_report,
    build_learning_report,
    build_relationship_report,
    build_reply_report,
    build_style_report,
    build_visual_report,
)
from src.creator_learning.style_evolution import diff_creator_dna


def _evidence(text, tags=None):
    return Evidence(evidence_type=EvidenceType.TEXT_EXCERPT, source_description=text, content_excerpt=text, tags=tags or [])


class BuildLearningReportTests(unittest.TestCase):
    def test_counts_and_confidence_reflected(self):
        e1 = _evidence("note one")
        dna = CreatorDNA(subject_label="c1", overall_confidence="low")
        report = build_learning_report(creator_id="c1", session_id="s1", dna=dna, evidence=[e1], history=[])
        self.assertEqual(report.evidence_count, 1)
        self.assertEqual(report.overall_confidence, "low")
        self.assertEqual(report.session_count, 0)

    def test_confidence_trend_reflects_history_order(self):
        history = [
            LearningHistoryEntry(session_id="s1", triggered_at="t1", evidence_count_before=0, evidence_count_after=1,
                                  new_evidence_count=1, dna_id="d1", overall_confidence="low"),
            LearningHistoryEntry(session_id="s2", triggered_at="t2", evidence_count_before=1, evidence_count_after=2,
                                  new_evidence_count=1, dna_id="d2", overall_confidence="medium"),
        ]
        dna = CreatorDNA(subject_label="c1")
        report = build_learning_report(creator_id="c1", session_id="s2", dna=dna, evidence=[], history=history)
        self.assertEqual(report.confidence_trend, [("s1", "low"), ("s2", "medium")])

    def test_gaps_include_uncovered_tags(self):
        dna = CreatorDNA(subject_label="c1")
        report = build_learning_report(creator_id="c1", session_id="s1", dna=dna, evidence=[], history=[])
        self.assertIn("coffee", report.gaps["storytelling"])


class BuildVisualReportTests(unittest.TestCase):
    def test_pulls_trait_scores_and_excerpts(self):
        evidence_item = Evidence(evidence_type=EvidenceType.TEXT_EXCERPT, source_description="s", content_excerpt="warm natural light")
        evidence = [evidence_item]
        trait = TraitScore(
            trait_name="visual_realism", score=0.6, confidence="medium",
            evidence_ids=[evidence_item.evidence_id], rationale="observed lighting",
        )
        dna = CreatorDNA(subject_label="c1", visual_realism=trait)
        report = build_visual_report(creator_id="c1", session_id="s1", dna=dna, evidence=evidence)
        self.assertEqual(report.visual_realism_score, 0.6)
        self.assertEqual(report.visual_realism_confidence, "medium")
        self.assertEqual(report.visual_realism_rationale, "observed lighting")

    def test_absent_trait_yields_none_scores(self):
        dna = CreatorDNA(subject_label="c1")
        report = build_visual_report(creator_id="c1", session_id="s1", dna=dna, evidence=[])
        self.assertIsNone(report.visual_realism_score)
        self.assertIsNone(report.photography_score)
        self.assertEqual(report.cited_excerpts, [])


class BuildRelationshipReportTests(unittest.TestCase):
    def test_claims_carried_through_as_plain_dicts(self):
        claim = RelationshipClaim(description="has a sibling", basis="presented_narrative", evidence_ids=["e1"])
        dna = CreatorDNA(subject_label="c1", relationship_claims=[claim])
        report = build_relationship_report(creator_id="c1", session_id="s1", dna=dna)
        self.assertEqual(report.claims, [{"description": "has a sibling", "basis": "presented_narrative", "evidence_ids": ["e1"]}])

    def test_no_claims_yields_empty_list(self):
        dna = CreatorDNA(subject_label="c1")
        report = build_relationship_report(creator_id="c1", session_id="s1", dna=dna)
        self.assertEqual(report.claims, [])


class BuildReplyReportTests(unittest.TestCase):
    def test_absent_trait_yields_none_score(self):
        dna = CreatorDNA(subject_label="c1")
        report = build_reply_report(creator_id="c1", session_id="s1", dna=dna, evidence=[])
        self.assertIsNone(report.replies_score)
        self.assertEqual(report.language_mix_covered, ())


class BuildCaptionReportTests(unittest.TestCase):
    def test_hashtag_and_mention_frequency_counted(self):
        excerpt = "loving this #coffee shop today #coffee @friend_of_creator"
        evidence_item = Evidence(evidence_type=EvidenceType.TEXT_EXCERPT, source_description="caption", content_excerpt=excerpt)
        trait = TraitScore(trait_name="captions", score=0.5, confidence="low", evidence_ids=[evidence_item.evidence_id])
        dna = CreatorDNA(subject_label="c1", captions=trait)
        report = build_caption_report(creator_id="c1", session_id="s1", dna=dna, evidence=[evidence_item])
        self.assertEqual(report.hashtag_frequency, {"coffee": 2})
        self.assertEqual(report.mention_frequency, {"friend_of_creator": 1})

    def test_no_captions_trait_yields_empty_frequencies(self):
        dna = CreatorDNA(subject_label="c1")
        report = build_caption_report(creator_id="c1", session_id="s1", dna=dna, evidence=[])
        self.assertEqual(report.hashtag_frequency, {})
        self.assertEqual(report.mention_frequency, {})


class BuildStyleReportTests(unittest.TestCase):
    def test_baseline_session_produces_baseline_lines(self):
        dna = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.4, confidence="low"))
        evolution = diff_creator_dna(None, dna)
        report = build_style_report(creator_id="c1", session_id="s1", evolution=evolution)
        self.assertTrue(report.is_baseline)
        self.assertTrue(any("new baseline" in line for line in report.trait_summary))

    def test_real_diff_reports_score_and_confidence_changes(self):
        previous = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.4, confidence="low"))
        current = CreatorDNA(subject_label="c1", persona=TraitScore(trait_name="persona", score=0.7, confidence="high"))
        evolution = diff_creator_dna(previous, current)
        report = build_style_report(creator_id="c1", session_id="s1", evolution=evolution)
        self.assertFalse(report.is_baseline)
        self.assertTrue(any("increased" in line for line in report.trait_summary))
        self.assertTrue(any("confidence changed" in line for line in report.trait_summary))


class BuildAllReportsTests(unittest.TestCase):
    def test_returns_all_six_report_types(self):
        dna = CreatorDNA(subject_label="c1")
        evolution = diff_creator_dna(None, dna)
        reports = build_all_reports(creator_id="c1", session_id="s1", dna=dna, evidence=[], history=[], evolution=evolution)
        self.assertEqual(
            set(reports.keys()), {"learning", "visual", "relationship", "reply", "caption", "style"}
        )


if __name__ == "__main__":
    unittest.main()
