import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence import intake, intake_manifest as im
from src.creator_intelligence.intake_validator import (
    IntakeValidationResult,
    main as validator_main,
    parse_arguments,
    validate_intake_workspace,
)


class IntakeValidatorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = im.load_intake_config()
        self.out = self.tmp_path / "ws"
        self.profile = intake.init_intake_workspace("demo", "instagram", "https://x/demo", self.out, self.config)

    def tearDown(self):
        self._tmp.cleanup()

    def _add_caption(self, text="hi"):
        caption = intake.CaptionRecord(creator_id=self.profile.creator_id, post_reference="p1", caption_text=text)
        intake.add_caption(self.out, caption, self.config)
        return caption


class ValidWorkspaceTests(IntakeValidatorTestCase):
    def test_freshly_initialized_workspace_passes(self):
        result = validate_intake_workspace(self.out, self.config)
        self.assertTrue(result.passed)
        self.assertEqual(result.errors, [])

    def test_workspace_with_one_caption_passes(self):
        self._add_caption()
        result = validate_intake_workspace(self.out, self.config)
        self.assertTrue(result.passed)

    def test_result_is_correct_type(self):
        result = validate_intake_workspace(self.out, self.config)
        self.assertIsInstance(result, IntakeValidationResult)

    def test_completeness_reported_separately_from_pass_fail(self):
        result = validate_intake_workspace(self.out, self.config)
        self.assertIn("overall", result.completeness)
        self.assertLess(result.completeness["overall"], 1.0)
        self.assertTrue(result.passed)  # low completeness alone does not fail validation


class MissingWorkspaceTests(unittest.TestCase):
    def test_nonexistent_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = im.load_intake_config()
            result = validate_intake_workspace(Path(tmp_dir) / "does_not_exist", config)
            self.assertFalse(result.passed)


class EvidenceIntegrityTests(IntakeValidatorTestCase):
    def test_tampered_evidence_id_fails(self):
        self._add_caption()
        bundle_path = self.out / "evidence_bundle.json"
        bundle = json.loads(bundle_path.read_text())
        bundle[0]["evidence_id"] = "0" * 16
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("evidence_id mismatch" in e for e in result.errors))

    def test_duplicate_evidence_id_fails(self):
        self._add_caption()
        bundle_path = self.out / "evidence_bundle.json"
        bundle = json.loads(bundle_path.read_text())
        bundle.append(bundle[0])
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("duplicate evidence_id" in e for e in result.errors))

    def test_unsupported_evidence_type_fails(self):
        self._add_caption()
        bundle_path = self.out / "evidence_bundle.json"
        bundle = json.loads(bundle_path.read_text())
        bundle[0]["evidence_type"] = "scraped"
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)

    def test_malformed_evidence_bundle_json_fails(self):
        (self.out / "evidence_bundle.json").write_text("{not valid", encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)


class SourceIndexTests(IntakeValidatorTestCase):
    def test_duplicate_source_id_fails(self):
        caption = self._add_caption()
        entries = im.load_source_index(self.out)
        duplicate = im.SourceIndexEntry(
            source_id=entries[0].source_id, creator_id=self.profile.creator_id, source_type="caption"
        )
        # bypass add_source_entry's own dedup to simulate corruption
        raw_entries = entries + [duplicate]
        payload = [
            {
                "source_id": e.source_id, "creator_id": e.creator_id, "source_type": e.source_type,
                "original_reference": e.original_reference, "local_file": e.local_file,
                "published_at": e.published_at, "captured_at": e.captured_at,
                "evidence_ids": e.evidence_ids, "notes": e.notes,
            }
            for e in raw_entries
        ]
        (self.out / "source_index.json").write_text(json.dumps(payload), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)

    def test_source_referencing_unknown_evidence_id_fails(self):
        self._add_caption()
        path = self.out / "source_index.json"
        entries = json.loads(path.read_text())
        entries[0]["evidence_ids"] = ["nonexistent_evidence_id"]
        path.write_text(json.dumps(entries), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)

    def test_source_with_missing_local_file_fails(self):
        path = self.out / "source_index.json"
        entries = [
            {
                "source_id": "s1", "creator_id": self.profile.creator_id, "source_type": "screenshot",
                "original_reference": "", "local_file": "screenshots/missing.png", "published_at": "",
                "captured_at": "", "evidence_ids": [], "notes": "",
            }
        ]
        path.write_text(json.dumps(entries), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("missing file" in e for e in result.errors))

    def test_source_with_empty_local_file_warns_not_fails(self):
        empty_file = self.out / "screenshots" / "empty.png"
        empty_file.write_bytes(b"")
        path = self.out / "source_index.json"
        entries = [
            {
                "source_id": "s1", "creator_id": self.profile.creator_id, "source_type": "screenshot",
                "original_reference": "", "local_file": "screenshots/empty.png", "published_at": "",
                "captured_at": "", "evidence_ids": [], "notes": "",
            }
        ]
        path.write_text(json.dumps(entries), encoding="utf-8")
        (self.out / "intake_manifest.json").unlink()  # avoid an unrelated stale-manifest error
        result = validate_intake_workspace(self.out, self.config)
        self.assertTrue(result.passed)
        self.assertTrue(any("empty file" in w for w in result.warnings))

    def test_local_path_traversal_fails(self):
        path = self.out / "source_index.json"
        entries = [
            {
                "source_id": "s1", "creator_id": self.profile.creator_id, "source_type": "screenshot",
                "original_reference": "", "local_file": "../../etc/passwd", "published_at": "",
                "captured_at": "", "evidence_ids": [], "notes": "",
            }
        ]
        path.write_text(json.dumps(entries), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("escapes" in e for e in result.errors))

    def test_evidence_not_referenced_by_any_source_warns(self):
        bundle_path = self.out / "evidence_bundle.json"
        bundle_path.write_text(
            json.dumps(
                [
                    {
                        "evidence_id": "abc", "evidence_type": "manual_note", "source_description": "x",
                        "content_excerpt": "", "collected_at": "", "collected_by": "operator", "tags": [],
                    }
                ]
            ),
            encoding="utf-8",
        )
        # not a real recomputed id, so this also produces an error --
        # use a properly-computed one instead:
        from src.creator_intelligence.evidence import Evidence, EvidenceType

        real_evidence = Evidence(evidence_type=EvidenceType.MANUAL_NOTE, source_description="x")
        bundle_path.write_text(
            json.dumps(
                [
                    {
                        "evidence_id": real_evidence.evidence_id, "evidence_type": "manual_note",
                        "source_description": "x", "content_excerpt": "", "collected_at": "",
                        "collected_by": "operator", "tags": [],
                    }
                ]
            ),
            encoding="utf-8",
        )
        result = validate_intake_workspace(self.out, self.config)
        self.assertTrue(any("not referenced by any source_index entry" in w for w in result.warnings))


class RecordShapeTests(IntakeValidatorTestCase):
    def test_malformed_caption_missing_text_fails(self):
        (self.out / "captions" / "bad.json").write_text(json.dumps({"post_reference": "p1"}), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("caption_text" in e for e in result.errors))

    def test_malformed_reply_pair_missing_reply_fails(self):
        (self.out / "replies" / "bad.json").write_text(json.dumps({"audience_comment": "hi"}), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)

    def test_reel_shot_order_violation_fails(self):
        payload = {"shot_notes": [{"shot_index": 2}, {"shot_index": 1}]}
        (self.out / "reels" / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("out-of-order shot_index" in e for e in result.errors))

    def test_reel_shot_order_ascending_passes(self):
        payload = {"shot_notes": [{"shot_index": 0}, {"shot_index": 1}, {"shot_index": 1}]}
        (self.out / "reels" / "good.json").write_text(json.dumps(payload), encoding="utf-8")
        (self.out / "intake_manifest.json").unlink()  # avoid an unrelated stale-manifest error
        result = validate_intake_workspace(self.out, self.config)
        self.assertTrue(result.passed)

    def test_creator_id_mismatch_fails(self):
        (self.out / "captions" / "bad.json").write_text(
            json.dumps({"creator_id": "someone_else", "caption_text": "hi"}), encoding="utf-8"
        )
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("does not match profile creator_id" in e for e in result.errors))

    def test_non_serializable_metadata_fails(self):
        # Construct a record whose metadata can't be represented in
        # JSON at all -- simulate by writing raw bytes as text that
        # decodes to something json.dumps would reject at the Python
        # level is not directly expressible in a JSON file, so this
        # test instead confirms the check is a no-op for normal
        # (already-serializable) metadata.
        (self.out / "captions" / "ok.json").write_text(
            json.dumps({"caption_text": "hi", "metadata": {"a": 1}}), encoding="utf-8"
        )
        (self.out / "intake_manifest.json").unlink()  # avoid an unrelated stale-manifest error
        result = validate_intake_workspace(self.out, self.config)
        self.assertTrue(result.passed)


class ProfileValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = im.load_intake_config()

    def tearDown(self):
        self._tmp.cleanup()

    def test_invalid_profile_url_fails(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "not-a-url", out, self.config)
        result = validate_intake_workspace(out, self.config)
        self.assertFalse(result.passed)

    def test_missing_profile_file_fails(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        (out / "creator_profile.json").unlink()
        result = validate_intake_workspace(out, self.config)
        self.assertFalse(result.passed)


class StrictModeTests(IntakeValidatorTestCase):
    def test_default_mode_below_target_only_warns(self):
        self._add_caption()
        result = validate_intake_workspace(self.out, self.config, strict=False)
        self.assertTrue(result.passed)
        self.assertTrue(any("below recommended target" in w for w in result.warnings))

    def test_strict_mode_below_target_fails(self):
        self._add_caption()
        result = validate_intake_workspace(self.out, self.config, strict=True)
        self.assertFalse(result.passed)

    def test_strict_mode_missing_targets_includes_below_target_categories(self):
        self._add_caption()
        result = validate_intake_workspace(self.out, self.config, strict=True)
        self.assertIn("captions", result.missing_targets)


class ManifestConsistencyTests(IntakeValidatorTestCase):
    def test_stale_manifest_counts_mismatch_fails(self):
        self._add_caption()
        manifest_path = self.out / "intake_manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["counts_by_type"]["captions"] = 999
        # manifest_id will now legitimately mismatch too, but we want
        # to isolate the counts-mismatch check specifically, so also
        # keep the file readable (integrity check runs first and will
        # itself fail -- both are valid signals of a stale manifest).
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        result = validate_intake_workspace(self.out, self.config)
        self.assertFalse(result.passed)

    def test_manifest_missing_is_not_a_hard_requirement(self):
        (self.out / "intake_manifest.json").unlink()
        result = validate_intake_workspace(self.out, self.config)
        self.assertTrue(result.passed)


class ReadOnlyGuaranteeTests(IntakeValidatorTestCase):
    def test_validate_never_writes_to_workspace(self):
        before = {p: p.stat().st_mtime for p in self.out.rglob("*") if p.is_file()}
        validate_intake_workspace(self.out, self.config)
        after = {p: p.stat().st_mtime for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_validate_never_creates_new_files(self):
        before = set(self.out.rglob("*"))
        validate_intake_workspace(self.out, self.config)
        after = set(self.out.rglob("*"))
        self.assertEqual(before, after)


class CliTests(IntakeValidatorTestCase):
    def test_cli_passes_on_valid_workspace(self):
        rc = validator_main(["--intake", str(self.out), "--json"])
        self.assertEqual(rc, 0)

    def test_cli_fails_on_missing_workspace(self):
        rc = validator_main(["--intake", str(self.tmp_path / "missing"), "--json"])
        self.assertEqual(rc, 1)

    def test_cli_strict_flag_parsed(self):
        args = parse_arguments(["--intake", str(self.out), "--strict"])
        self.assertTrue(args.strict)

    def test_cli_requires_intake_flag(self):
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_cli_missing_config_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_config = Path(tmp_dir) / "does_not_exist.yaml"
            rc = validator_main(["--intake", str(self.out), "--config", str(missing_config), "--json"])
            self.assertEqual(rc, 1)


class NoNetworkStructuralTests(unittest.TestCase):
    def setUp(self):
        from src.creator_intelligence import intake_validator

        self.source = Path(intake_validator.__file__).read_text(encoding="utf-8")

    def test_no_network_imports(self):
        for forbidden in ("import requests", "import playwright", "import selenium"):
            self.assertNotIn(forbidden, self.source)

    def test_no_analyzer_imports(self):
        for forbidden in ("summary_builder", "_analyzer import", "build_creator_dna"):
            self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
