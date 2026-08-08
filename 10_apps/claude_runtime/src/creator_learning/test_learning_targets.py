import unittest

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_learning.learning_targets import (
    LEARNING_TARGETS,
    all_target_tags,
    compute_learning_target_coverage,
    missing_tags_by_trait,
)


def _note(tags):
    return Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="note", tags=tags)


class AllTargetTagsTests(unittest.TestCase):
    def test_returns_every_unique_tag_across_traits(self):
        tags = all_target_tags()
        self.assertIn("coffee", tags)
        self.assertIn("film_grain", tags)
        self.assertEqual(len(tags), len(set(tags)))


class ComputeLearningTargetCoverageTests(unittest.TestCase):
    def test_zero_evidence_means_all_missing(self):
        coverage = compute_learning_target_coverage([])
        for trait_name, tags in LEARNING_TARGETS.items():
            trait_coverage = coverage[trait_name]
            self.assertEqual(set(trait_coverage.missing_tags), set(tags))
            self.assertEqual(trait_coverage.covered_tags, ())

    def test_matching_tag_increments_count_under_every_relevant_trait(self):
        evidence = [_note(["family"])]
        coverage = compute_learning_target_coverage(evidence)
        # "family" is catalogued under both human_authenticity and relationships
        self.assertIn("family", coverage["human_authenticity"].covered_tags)
        self.assertIn("family", coverage["relationships"].covered_tags)
        self.assertEqual(coverage["human_authenticity"].tag_counts["family"], 1)

    def test_unrelated_tag_does_not_affect_coverage(self):
        evidence = [_note(["totally_unrelated_tag"])]
        coverage = compute_learning_target_coverage(evidence)
        for trait_coverage in coverage.values():
            self.assertEqual(trait_coverage.covered_tags, ())

    def test_multiple_evidence_items_accumulate_counts(self):
        evidence = [_note(["coffee"]), _note(["coffee"]), _note(["travel"])]
        coverage = compute_learning_target_coverage(evidence)
        self.assertEqual(coverage["storytelling"].tag_counts["coffee"], 2)
        self.assertEqual(coverage["storytelling"].tag_counts["travel"], 1)


class MissingTagsByTraitTests(unittest.TestCase):
    def test_covered_tag_is_excluded_from_missing(self):
        evidence = [_note(["coffee"])]
        missing = missing_tags_by_trait(evidence)
        self.assertNotIn("coffee", missing["storytelling"])
        self.assertIn("travel", missing["storytelling"])


if __name__ == "__main__":
    unittest.main()
