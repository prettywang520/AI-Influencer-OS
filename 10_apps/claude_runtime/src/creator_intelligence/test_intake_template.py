import unittest

from src.creator_intelligence.intake_template import EVIDENCE_CATEGORIES, NO_ANALYSIS_BANNER, build_intake_checklist


class BuildIntakeChecklistTests(unittest.TestCase):
    def test_includes_no_analysis_banner(self):
        checklist = build_intake_checklist(["Creator A"])
        self.assertIn(NO_ANALYSIS_BANNER, checklist)

    def test_includes_every_subject_label(self):
        checklist = build_intake_checklist(["Creator A", "Creator B"])
        self.assertIn("Creator A", checklist)
        self.assertIn("Creator B", checklist)

    def test_includes_every_evidence_category_per_subject(self):
        checklist = build_intake_checklist(["Creator A"])
        for tag in EVIDENCE_CATEGORIES:
            self.assertIn(f"`{tag}`", checklist)

    def test_empty_subject_list_still_includes_banner(self):
        checklist = build_intake_checklist([])
        self.assertIn(NO_ANALYSIS_BANNER, checklist)

    def test_never_includes_a_score_or_confidence_value(self):
        checklist = build_intake_checklist(["Creator A"])
        for forbidden in ("score:", "confidence:", "verified", "dna_id"):
            self.assertNotIn(forbidden, checklist.lower())

    def test_checkbox_markdown_present(self):
        checklist = build_intake_checklist(["Creator A"])
        self.assertIn("- [ ]", checklist)

    def test_multiple_subjects_each_get_a_full_section(self):
        checklist = build_intake_checklist(["Creator A", "Creator B"])
        self.assertEqual(checklist.count("## Subject:"), 2)


if __name__ == "__main__":
    unittest.main()
