import unittest

from src.creator_intelligence.evidence import Evidence, EvidenceType, InvalidEvidenceTypeError


def _evidence(**overrides):
    defaults = dict(
        evidence_type=EvidenceType.MANUAL_NOTE,
        source_description="operator note",
        content_excerpt="observed pattern",
        collected_at="2026-08-07T00:00:00+00:00",
        collected_by="operator_a",
        tags=["persona"],
    )
    defaults.update(overrides)
    return Evidence(**defaults)


class EvidenceTypeTests(unittest.TestCase):
    def test_all_contains_every_named_constant(self):
        self.assertEqual(
            set(EvidenceType.ALL),
            {
                EvidenceType.SCREENSHOT,
                EvidenceType.MANUAL_NOTE,
                EvidenceType.TEXT_EXCERPT,
                EvidenceType.EXPORTED_DATA,
                EvidenceType.OPERATOR_OBSERVATION,
            },
        )

    def test_no_automated_collection_type_present(self):
        forbidden = {"scraped", "api_fetch", "browser_automation", "login_session"}
        self.assertFalse(forbidden.intersection(EvidenceType.ALL))


class EvidenceConstructionTests(unittest.TestCase):
    def test_valid_construction_succeeds(self):
        evidence = _evidence()
        self.assertEqual(evidence.evidence_type, EvidenceType.MANUAL_NOTE)

    def test_invalid_evidence_type_raises(self):
        with self.assertRaises(InvalidEvidenceTypeError):
            _evidence(evidence_type="scraped")

    def test_evidence_id_is_populated(self):
        evidence = _evidence()
        self.assertTrue(evidence.evidence_id)
        self.assertIsInstance(evidence.evidence_id, str)

    def test_default_tags_is_empty_list_not_shared(self):
        a = Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="a")
        b = Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="b")
        a.tags.append("x")
        self.assertEqual(b.tags, [])


class EvidenceIdStabilityTests(unittest.TestCase):
    def test_identical_content_yields_identical_id(self):
        first = _evidence()
        second = _evidence()
        self.assertEqual(first.evidence_id, second.evidence_id)

    def test_different_collected_at_does_not_change_id(self):
        first = _evidence(collected_at="2026-01-01T00:00:00+00:00")
        second = _evidence(collected_at="2026-12-31T23:59:59+00:00")
        self.assertEqual(first.evidence_id, second.evidence_id)

    def test_different_content_excerpt_changes_id(self):
        first = _evidence(content_excerpt="pattern a")
        second = _evidence(content_excerpt="pattern b")
        self.assertNotEqual(first.evidence_id, second.evidence_id)

    def test_different_source_description_changes_id(self):
        first = _evidence(source_description="source a")
        second = _evidence(source_description="source b")
        self.assertNotEqual(first.evidence_id, second.evidence_id)

    def test_different_collected_by_changes_id(self):
        first = _evidence(collected_by="alice")
        second = _evidence(collected_by="bob")
        self.assertNotEqual(first.evidence_id, second.evidence_id)

    def test_tag_order_does_not_change_id(self):
        first = _evidence(tags=["a", "b"])
        second = _evidence(tags=["b", "a"])
        self.assertEqual(first.evidence_id, second.evidence_id)

    def test_different_tags_change_id(self):
        first = _evidence(tags=["a"])
        second = _evidence(tags=["a", "b"])
        self.assertNotEqual(first.evidence_id, second.evidence_id)


if __name__ == "__main__":
    unittest.main()
