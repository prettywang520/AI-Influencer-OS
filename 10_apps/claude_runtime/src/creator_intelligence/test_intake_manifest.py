import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence import intake_manifest as im


class IntakeTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class LoadIntakeConfigTests(unittest.TestCase):
    def test_loads_real_default_config(self):
        config = im.load_intake_config()
        self.assertEqual(config.schema_version, "1.0")

    def test_default_path_exists_on_disk(self):
        self.assertTrue(im.default_intake_config_path().is_file())

    def test_targets_cover_all_category_names(self):
        config = im.load_intake_config()
        self.assertEqual(set(config.targets.keys()), set(im.CATEGORY_NAMES))

    def test_package_directories_non_empty(self):
        config = im.load_intake_config()
        self.assertTrue(config.package_directories)

    def test_target_for_unknown_category_returns_zero(self):
        config = im.load_intake_config()
        target = config.target_for("not_a_real_category")
        self.assertEqual(target.minimum, 0)
        self.assertEqual(target.target, 0)

    def test_missing_config_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "does_not_exist.yaml"
            with self.assertRaises(im.IntakeConfigError):
                im.load_intake_config(missing)

    def test_empty_config_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty = Path(tmp_dir) / "empty.yaml"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(im.IntakeConfigError):
                im.load_intake_config(empty)

    def test_config_without_targets_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad = Path(tmp_dir) / "bad.yaml"
            bad.write_text("intake:\n  schema_version: '1.0'\n", encoding="utf-8")
            with self.assertRaises(im.IntakeConfigError):
                im.load_intake_config(bad)

    def test_custom_config_target_without_explicit_target_defaults_to_minimum(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom = Path(tmp_dir) / "custom.yaml"
            custom.write_text("targets:\n  profile:\n    minimum: 2\n", encoding="utf-8")
            config = im.load_intake_config(custom)
            self.assertEqual(config.targets["profile"].minimum, 2)
            self.assertEqual(config.targets["profile"].target, 2)


class SourceIndexTests(IntakeTempTestCase):
    def _entry(self, source_id="s1", creator_id="c1"):
        return im.SourceIndexEntry(source_id=source_id, creator_id=creator_id, source_type="caption")

    def test_add_source_entry_appends(self):
        entries = im.add_source_entry([], self._entry("s1"))
        self.assertEqual(len(entries), 1)

    def test_add_duplicate_source_id_raises(self):
        entries = im.add_source_entry([], self._entry("s1"))
        with self.assertRaises(im.DuplicateSourceIdError):
            im.add_source_entry(entries, self._entry("s1"))

    def test_entries_are_deterministically_ordered(self):
        entries = im.add_source_entry([], self._entry("s2"))
        entries = im.add_source_entry(entries, self._entry("s1"))
        self.assertEqual([e.source_id for e in entries], ["s1", "s2"])

    def test_save_and_load_round_trip(self):
        entries = im.add_source_entry([], self._entry("s1"))
        im.save_source_index(self.tmp_path, entries)
        loaded = im.load_source_index(self.tmp_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].source_id, "s1")

    def test_load_missing_source_index_returns_empty_list(self):
        self.assertEqual(im.load_source_index(self.tmp_path), [])

    def test_save_is_atomic_no_temp_file_left_behind(self):
        entries = im.add_source_entry([], self._entry("s1"))
        im.save_source_index(self.tmp_path, entries)
        remaining = list(self.tmp_path.iterdir())
        self.assertEqual(remaining, [self.tmp_path / "source_index.json"])

    def test_evidence_ids_round_trip(self):
        entry = im.SourceIndexEntry(source_id="s1", creator_id="c1", source_type="caption", evidence_ids=["e1", "e2"])
        im.save_source_index(self.tmp_path, [entry])
        loaded = im.load_source_index(self.tmp_path)
        self.assertEqual(loaded[0].evidence_ids, ["e1", "e2"])


class ComputeCompletenessTests(unittest.TestCase):
    def _config(self):
        return im.IntakeConfig(
            schema_version="1.0",
            targets={
                "profile": im.EvidenceTarget(minimum=1, target=1),
                "captions": im.EvidenceTarget(minimum=20, target=50),
            },
        )

    def test_zero_counts_yield_zero_completeness(self):
        per_category, overall = im.compute_completeness({}, self._config())
        self.assertEqual(per_category["captions"], 0.0)

    def test_partial_counts_yield_partial_completeness(self):
        per_category, _ = im.compute_completeness({"captions": 25}, self._config())
        self.assertAlmostEqual(per_category["captions"], 0.5)

    def test_over_target_counts_cap_at_one(self):
        per_category, _ = im.compute_completeness({"captions": 999}, self._config())
        self.assertEqual(per_category["captions"], 1.0)

    def test_category_without_configured_target_is_fully_complete(self):
        per_category, _ = im.compute_completeness({}, self._config())
        self.assertEqual(per_category["relationships"], 1.0)

    def test_overall_is_unweighted_mean(self):
        per_category, overall = im.compute_completeness({"profile": 1, "captions": 25}, self._config())
        expected = sum(per_category.values()) / len(per_category)
        self.assertAlmostEqual(overall, expected)

    def test_completeness_is_not_a_confidence_or_quality_value(self):
        # Structural guarantee: compute_completeness never imports or
        # references confidence.ConfidenceLevel / models.TraitScore.
        import inspect

        source = inspect.getsource(im.compute_completeness)
        self.assertNotIn("ConfidenceLevel", source)
        self.assertNotIn("TraitScore", source)


class MissingRecommendedEvidenceTests(unittest.TestCase):
    def _config(self):
        return im.IntakeConfig(
            schema_version="1.0",
            targets={"profile": im.EvidenceTarget(minimum=1, target=1), "captions": im.EvidenceTarget(minimum=20, target=50)},
        )

    def test_below_minimum_is_missing(self):
        missing = im.missing_recommended_evidence({"captions": 5}, self._config())
        self.assertIn("captions", missing)

    def test_at_minimum_is_not_missing(self):
        missing = im.missing_recommended_evidence({"profile": 1, "captions": 20}, self._config())
        self.assertNotIn("captions", missing)
        self.assertNotIn("profile", missing)


class CountCategoriesTests(IntakeTempTestCase):
    def _write(self, subdir, name, payload):
        directory = self.tmp_path / subdir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_empty_workspace_all_zero_except_profile(self):
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["captions"], 0)

    def test_profile_counted_when_file_present(self):
        (self.tmp_path / "creator_profile.json").write_text("{}", encoding="utf-8")
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["profile"], 1)

    def test_grid_screenshot_counted(self):
        self._write("screenshots", "s1", {"source_type": "grid_screenshot"})
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["grid"], 1)

    def test_post_and_carousel_screenshots_counted_as_posts(self):
        self._write("screenshots", "s1", {"source_type": "post_screenshot"})
        self._write("screenshots", "s2", {"source_type": "carousel_screenshot"})
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["posts"], 2)

    def test_visual_realism_counted_by_annotation_presence(self):
        self._write("screenshots", "s1", {"source_type": "post_screenshot", "visual_annotations": {"skin_texture_visible": "yes"}})
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["visual_realism"], 1)

    def test_captions_replies_reels_highlights_counted_by_directory(self):
        self._write("captions", "c1", {})
        self._write("replies", "r1", {})
        self._write("reels", "reel1", {})
        self._write("highlights", "h1", {})
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["captions"], 1)
        self.assertEqual(counts["creator_replies"], 1)
        self.assertEqual(counts["reels"], 1)
        self.assertEqual(counts["highlights"], 1)

    def test_relationship_records_count_toward_relationships_and_human_authenticity(self):
        self._write("relationships", "a1", {"role": "family_depiction"})
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["relationships"], 1)
        self.assertEqual(counts["human_authenticity"], 1)

    def test_human_authenticity_tagged_note_counted(self):
        self._write("notes", "n1", {"tags": ["childhood_or_old_photo"]})
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["human_authenticity"], 1)

    def test_malformed_json_file_is_skipped_not_raised(self):
        directory = self.tmp_path / "captions"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bad.json").write_text("{not valid json", encoding="utf-8")
        counts = im.count_categories(self.tmp_path)
        self.assertEqual(counts["captions"], 0)


class ManifestIdentityTests(unittest.TestCase):
    def _manifest(self, **overrides):
        defaults = dict(
            schema_version="1.0", creator_id="c1", username="demo", platform="instagram", profile_url="https://x/demo"
        )
        defaults.update(overrides)
        return im.IntakeManifest(**defaults)

    def test_manifest_id_is_populated(self):
        self.assertTrue(self._manifest().manifest_id)

    def test_identical_content_yields_identical_id(self):
        first = self._manifest()
        second = self._manifest()
        self.assertEqual(first.manifest_id, second.manifest_id)

    def test_different_created_at_does_not_change_id(self):
        first = self._manifest(created_at="2026-01-01T00:00:00+00:00")
        second = self._manifest(created_at="2026-12-31T00:00:00+00:00")
        self.assertEqual(first.manifest_id, second.manifest_id)

    def test_different_updated_at_does_not_change_id(self):
        first = self._manifest(updated_at="2026-01-01T00:00:00+00:00")
        second = self._manifest(updated_at="2026-12-31T00:00:00+00:00")
        self.assertEqual(first.manifest_id, second.manifest_id)

    def test_different_evidence_count_changes_id(self):
        first = self._manifest(evidence_count=1)
        second = self._manifest(evidence_count=2)
        self.assertNotEqual(first.manifest_id, second.manifest_id)

    def test_different_counts_by_type_changes_id(self):
        first = self._manifest(counts_by_type={"captions": 1})
        second = self._manifest(counts_by_type={"captions": 2})
        self.assertNotEqual(first.manifest_id, second.manifest_id)


class BuildIntakeManifestTests(IntakeTempTestCase):
    def _config(self):
        return im.load_intake_config()

    def test_counts_reflect_on_disk_state(self):
        (self.tmp_path / "creator_profile.json").write_text("{}", encoding="utf-8")
        (self.tmp_path / "captions").mkdir()
        (self.tmp_path / "captions" / "c1.json").write_text("{}", encoding="utf-8")
        manifest = im.build_intake_manifest(
            creator_id="c1", username="demo", platform="instagram", profile_url="https://x/demo",
            intake_dir=self.tmp_path, intake_config=self._config(),
        )
        self.assertEqual(manifest.counts_by_type["captions"], 1)
        self.assertEqual(manifest.counts_by_type["profile"], 1)

    def test_evidence_count_reads_evidence_bundle(self):
        (self.tmp_path / "evidence_bundle.json").write_text(json.dumps([{"evidence_id": "e1"}, {"evidence_id": "e2"}]), encoding="utf-8")
        manifest = im.build_intake_manifest(
            creator_id="c1", username="demo", platform="instagram", profile_url="https://x/demo",
            intake_dir=self.tmp_path, intake_config=self._config(),
        )
        self.assertEqual(manifest.evidence_count, 2)

    def test_source_count_reads_source_index(self):
        im.save_source_index(self.tmp_path, [im.SourceIndexEntry(source_id="s1", creator_id="c1", source_type="caption")])
        manifest = im.build_intake_manifest(
            creator_id="c1", username="demo", platform="instagram", profile_url="https://x/demo",
            intake_dir=self.tmp_path, intake_config=self._config(),
        )
        self.assertEqual(manifest.source_count, 1)


class SaveLoadManifestTests(IntakeTempTestCase):
    def _manifest(self):
        return im.IntakeManifest(
            schema_version="1.0", creator_id="c1", username="demo", platform="instagram", profile_url="https://x/demo"
        )

    def test_save_creates_file(self):
        im.save_intake_manifest(self.tmp_path, self._manifest())
        self.assertTrue((self.tmp_path / "intake_manifest.json").is_file())

    def test_load_round_trips_manifest_id(self):
        manifest = self._manifest()
        im.save_intake_manifest(self.tmp_path, manifest)
        loaded = im.load_intake_manifest(self.tmp_path)
        self.assertEqual(loaded.manifest_id, manifest.manifest_id)

    def test_load_missing_file_raises(self):
        with self.assertRaises(im.IntakeManifestError):
            im.load_intake_manifest(self.tmp_path)

    def test_load_tampered_manifest_id_raises(self):
        manifest = self._manifest()
        path = im.save_intake_manifest(self.tmp_path, manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["manifest_id"] = "0" * 16
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(im.IntakeManifestIntegrityError):
            im.load_intake_manifest(self.tmp_path)


class Sha256FileTests(IntakeTempTestCase):
    def test_hash_is_deterministic(self):
        path = self.tmp_path / "f.bin"
        path.write_bytes(b"hello world")
        first = im.sha256_file(path)
        second = im.sha256_file(path)
        self.assertEqual(first, second)

    def test_different_content_different_hash(self):
        path_a = self.tmp_path / "a.bin"
        path_b = self.tmp_path / "b.bin"
        path_a.write_bytes(b"aaaa")
        path_b.write_bytes(b"bbbb")
        self.assertNotEqual(im.sha256_file(path_a), im.sha256_file(path_b))

    def test_matches_hashlib_reference(self):
        import hashlib

        path = self.tmp_path / "f.bin"
        content = b"x" * 200000
        path.write_bytes(content)
        self.assertEqual(im.sha256_file(path), hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
