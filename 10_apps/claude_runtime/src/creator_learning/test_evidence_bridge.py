import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_learning.evidence_bridge import (
    evidence_from_intake_workspace,
    evidence_from_research_session,
    merge_into_knowledge_base,
)
from src.creator_learning.exceptions import KnowledgeBaseError
from src.creator_learning.knowledge_base import CreatorKnowledgeBase, evidence_to_dict


def _note(text):
    return Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description=text, content_excerpt=text)


class _FakeQueue:
    def __init__(self, items):
        self._items = items

    def all_items(self):
        return list(self._items)


class _FakeSession:
    def __init__(self, items):
        self.evidence_queue = _FakeQueue(items)


class EvidenceFromResearchSessionTests(unittest.TestCase):
    def test_extracts_all_items_from_session_queue(self):
        items = [_note("a"), _note("b")]
        session = _FakeSession(items)
        extracted = evidence_from_research_session(session)
        self.assertEqual([item.evidence_id for item in extracted], [item.evidence_id for item in items])


class EvidenceFromIntakeWorkspaceTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KnowledgeBaseError):
                evidence_from_intake_workspace(Path(tmp) / "evidence_bundle.json")

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence_bundle.json"
            path.write_text("{not valid")
            with self.assertRaises(KnowledgeBaseError):
                evidence_from_intake_workspace(path)

    def test_non_list_payload_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence_bundle.json"
            path.write_text(json.dumps({"not": "a list"}))
            with self.assertRaises(KnowledgeBaseError):
                evidence_from_intake_workspace(path)

    def test_loads_evidence_from_valid_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence_bundle.json"
            note = _note("intake note")
            path.write_text(json.dumps([evidence_to_dict(note)]))
            loaded = evidence_from_intake_workspace(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].evidence_id, note.evidence_id)


class MergeIntoKnowledgeBaseTests(unittest.TestCase):
    def test_merges_from_session_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            session = _FakeSession([_note("from session")])
            merged = merge_into_knowledge_base(kb, session=session)
            self.assertEqual(len(merged), 1)

    def test_merges_from_intake_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "evidence_bundle.json"
            bundle_path.write_text(json.dumps([evidence_to_dict(_note("from intake"))]))
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            merged = merge_into_knowledge_base(kb, intake_evidence_bundle_path=bundle_path)
            self.assertEqual(len(merged), 1)

    def test_merges_from_both_sources_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = _note("shared between both sources")
            bundle_path = Path(tmp) / "evidence_bundle.json"
            bundle_path.write_text(json.dumps([evidence_to_dict(shared)]))
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            session = _FakeSession([shared, _note("only in session")])
            merged = merge_into_knowledge_base(kb, session=session, intake_evidence_bundle_path=bundle_path)
            self.assertEqual(len(merged), 2)

    def test_neither_source_supplied_merges_nothing_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = CreatorKnowledgeBase(Path(tmp), "creator1")
            kb.merge_evidence([_note("already present")])
            merged = merge_into_knowledge_base(kb)
            self.assertEqual(len(merged), 1)


if __name__ == "__main__":
    unittest.main()
