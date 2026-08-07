import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence import intake, intake_manifest as im
from src.creator_intelligence.models import RelationshipBasis


class IntakeTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.config = im.load_intake_config()

    def tearDown(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CreatorIdTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(intake.creator_id("instagram", "demo"), intake.creator_id("instagram", "demo"))

    def test_different_username_different_id(self):
        self.assertNotEqual(intake.creator_id("instagram", "a"), intake.creator_id("instagram", "b"))

    def test_different_platform_different_id(self):
        self.assertNotEqual(intake.creator_id("instagram", "demo"), intake.creator_id("tiktok", "demo"))


class CreatorProfileTests(unittest.TestCase):
    def test_creator_id_matches_helper(self):
        profile = intake.CreatorProfile(platform="instagram", username="demo", profile_url="https://x/demo")
        self.assertEqual(profile.creator_id, intake.creator_id("instagram", "demo"))

    def test_unknown_fields_default_to_none(self):
        profile = intake.CreatorProfile(platform="instagram", username="demo", profile_url="https://x/demo")
        self.assertIsNone(profile.display_name)
        self.assertIsNone(profile.bio)

    def test_no_sensitive_inference_fields_exist(self):
        # Structural guarantee: age/relationship-status/religion/politics/
        # legal-identity are simply not fields on this model.
        field_names = {f for f in intake.CreatorProfile.__dataclass_fields__}
        for forbidden in ("age", "relationship_status", "religion", "politics", "legal_identity", "residence"):
            self.assertNotIn(forbidden, field_names)


class VisualRealismAnnotationTests(unittest.TestCase):
    def test_default_construction_is_all_unknown(self):
        annotation = intake.VisualRealismAnnotation()
        self.assertEqual(annotation.skin_texture_visible, intake.TriState.UNKNOWN)
        self.assertEqual(annotation.editing_strength, intake.Intensity.UNKNOWN)

    def test_valid_tri_state_value_accepted(self):
        annotation = intake.VisualRealismAnnotation(pores_visible="yes")
        self.assertEqual(annotation.pores_visible, "yes")

    def test_invalid_tri_state_value_rejected(self):
        with self.assertRaises(intake.InvalidAnnotationValueError):
            intake.VisualRealismAnnotation(pores_visible="definitely")

    def test_invalid_intensity_value_rejected(self):
        with self.assertRaises(intake.InvalidAnnotationValueError):
            intake.VisualRealismAnnotation(makeup_level="a_lot")

    def test_free_text_field_over_cap_rejected(self):
        with self.assertRaises(intake.InvalidAnnotationValueError):
            intake.VisualRealismAnnotation(white_balance="x" * 200)

    def test_never_asserts_an_actual_camera_model(self):
        field_names = {f for f in intake.VisualRealismAnnotation.__dataclass_fields__}
        self.assertNotIn("camera_model", field_names)
        self.assertNotIn("device_model", field_names)


class RelationshipAnnotationTests(unittest.TestCase):
    def test_valid_construction(self):
        annotation = intake.RelationshipAnnotation(
            role=intake.RelationshipRole.FAMILY_DEPICTION, basis=RelationshipBasis.PRESENTED_NARRATIVE, confidence="low"
        )
        self.assertTrue(annotation.annotation_id)

    def test_invalid_role_rejected(self):
        with self.assertRaises(intake.InvalidAnnotationValueError):
            intake.RelationshipAnnotation(role="best_friend_forever", basis=RelationshipBasis.INFERRED, confidence="low")

    def test_invalid_basis_rejected(self):
        with self.assertRaises(intake.InvalidAnnotationValueError):
            intake.RelationshipAnnotation(role=intake.RelationshipRole.UNKNOWN_PERSON, basis="definitely_true", confidence="low")

    def test_invalid_confidence_rejected(self):
        with self.assertRaises(intake.InvalidAnnotationValueError):
            intake.RelationshipAnnotation(
                role=intake.RelationshipRole.UNKNOWN_PERSON, basis=RelationshipBasis.UNKNOWN, confidence="extremely_high"
            )

    def test_reuses_relationship_basis_from_models_module(self):
        # Structural guarantee: no duplicate enum was created.
        annotation = intake.RelationshipAnnotation(
            role=intake.RelationshipRole.COLLABORATION, basis=RelationshipBasis.VERIFIED, confidence="high"
        )
        self.assertIn(annotation.basis, RelationshipBasis.ALL)

    def test_depicted_does_not_equal_verified(self):
        depicted = intake.RelationshipAnnotation(
            role=intake.RelationshipRole.FAMILY_DEPICTION, basis=RelationshipBasis.VISUALLY_DEPICTED, confidence="medium"
        )
        self.assertNotEqual(depicted.basis, RelationshipBasis.VERIFIED)


class RecordIdStabilityTests(unittest.TestCase):
    def test_screenshot_id_stable_for_identical_content(self):
        a = intake.ScreenshotEvidence(creator_id="c1", source_type="grid_screenshot", manual_notes="note")
        b = intake.ScreenshotEvidence(creator_id="c1", source_type="grid_screenshot", manual_notes="note")
        self.assertEqual(a.screenshot_id, b.screenshot_id)

    def test_screenshot_id_excludes_captured_at(self):
        a = intake.ScreenshotEvidence(creator_id="c1", source_type="grid_screenshot", captured_at="2026-01-01")
        b = intake.ScreenshotEvidence(creator_id="c1", source_type="grid_screenshot", captured_at="2026-12-31")
        self.assertEqual(a.screenshot_id, b.screenshot_id)

    def test_caption_id_stable_for_identical_content(self):
        a = intake.CaptionRecord(creator_id="c1", post_reference="p1", caption_text="hi")
        b = intake.CaptionRecord(creator_id="c1", post_reference="p1", caption_text="hi")
        self.assertEqual(a.caption_id, b.caption_id)

    def test_caption_id_differs_for_different_text(self):
        a = intake.CaptionRecord(creator_id="c1", post_reference="p1", caption_text="hi")
        b = intake.CaptionRecord(creator_id="c1", post_reference="p1", caption_text="bye")
        self.assertNotEqual(a.caption_id, b.caption_id)

    def test_reply_pair_id_stable(self):
        a = intake.ReplyPair(creator_id="c1", post_reference="p1", audience_comment="hi", creator_reply="hello")
        b = intake.ReplyPair(creator_id="c1", post_reference="p1", audience_comment="hi", creator_reply="hello")
        self.assertEqual(a.reply_pair_id, b.reply_pair_id)

    def test_reel_id_stable(self):
        a = intake.ReelEvidence(creator_id="c1", source_reference="r1", hook_notes="zoom")
        b = intake.ReelEvidence(creator_id="c1", source_reference="r1", hook_notes="zoom")
        self.assertEqual(a.reel_id, b.reel_id)

    def test_highlight_id_stable(self):
        a = intake.HighlightItem(creator_id="c1", highlight_name="Travel", item_index=0)
        b = intake.HighlightItem(creator_id="c1", highlight_name="Travel", item_index=0)
        self.assertEqual(a.highlight_id, b.highlight_id)

    def test_highlight_id_differs_by_item_index(self):
        a = intake.HighlightItem(creator_id="c1", highlight_name="Travel", item_index=0)
        b = intake.HighlightItem(creator_id="c1", highlight_name="Travel", item_index=1)
        self.assertNotEqual(a.highlight_id, b.highlight_id)


# ---------------------------------------------------------------------------
# to_evidence() conversions
# ---------------------------------------------------------------------------


class ToEvidenceTests(unittest.TestCase):
    def test_screenshot_becomes_screenshot_evidence_type(self):
        screenshot = intake.ScreenshotEvidence(creator_id="c1", source_type="grid_screenshot", manual_notes="note")
        evidence, warnings = intake.to_evidence_from_screenshot(screenshot, 280)
        self.assertEqual(evidence.evidence_type, "screenshot")
        self.assertIn("grid_screenshot", evidence.tags)
        self.assertEqual(warnings, [])

    def test_caption_becomes_text_excerpt_with_caption_tag(self):
        caption = intake.CaptionRecord(creator_id="c1", post_reference="p1", caption_text="hi")
        evidence, _warnings = intake.to_evidence_from_caption(caption, 280)
        self.assertEqual(evidence.evidence_type, "text_excerpt")
        self.assertIn("caption", evidence.tags)

    def test_caption_over_cap_is_truncated_with_warning(self):
        caption = intake.CaptionRecord(creator_id="c1", post_reference="p1", caption_text="x" * 500)
        evidence, warnings = intake.to_evidence_from_caption(caption, 280)
        self.assertEqual(len(evidence.content_excerpt), 280)
        self.assertEqual(len(warnings), 1)

    def test_reply_pair_becomes_text_excerpt_with_reply_tag(self):
        pair = intake.ReplyPair(creator_id="c1", post_reference="p1", audience_comment="hi", creator_reply="hello")
        evidence, _warnings = intake.to_evidence_from_reply_pair(pair, 280)
        self.assertEqual(evidence.evidence_type, "text_excerpt")
        self.assertIn("reply", evidence.tags)

    def test_reply_pair_never_requires_audience_username(self):
        pair = intake.ReplyPair(creator_id="c1", post_reference="p1", audience_comment="hi", creator_reply="hello")
        self.assertIsNone(pair.audience_username_optional)

    def test_reel_becomes_operator_observation_with_reels_tag(self):
        reel = intake.ReelEvidence(creator_id="c1", source_reference="r1", hook_notes="zoom in")
        evidence, _warnings = intake.to_evidence_from_reel(reel, 280)
        self.assertEqual(evidence.evidence_type, "operator_observation")
        self.assertIn("reels", evidence.tags)

    def test_highlight_becomes_operator_observation_with_storytelling_tag(self):
        highlight = intake.HighlightItem(creator_id="c1", highlight_name="Travel", item_index=0, manual_summary="beach")
        evidence, _warnings = intake.to_evidence_from_highlight(highlight, 280)
        self.assertEqual(evidence.evidence_type, "operator_observation")
        self.assertIn("storytelling", evidence.tags)

    def test_relationship_annotation_becomes_relationship_and_human_authenticity_tagged(self):
        annotation = intake.RelationshipAnnotation(
            role=intake.RelationshipRole.FAMILY_DEPICTION, basis=RelationshipBasis.PRESENTED_NARRATIVE, confidence="low",
            description="childhood photo",
        )
        evidence, _warnings = intake.to_evidence_from_relationship_annotation(annotation, "c1", 280)
        self.assertIn("relationship", evidence.tags)
        self.assertIn("human_authenticity", evidence.tags)
        self.assertIn(intake.RelationshipRole.FAMILY_DEPICTION, evidence.tags)


# ---------------------------------------------------------------------------
# Workspace init
# ---------------------------------------------------------------------------


class InitIntakeWorkspaceTests(IntakeTempTestCase):
    def test_creates_every_package_directory(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        for dirname in self.config.package_directories:
            self.assertTrue((out / dirname).is_dir(), dirname)

    def test_writes_ownership_marker(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        self.assertTrue((out / intake.MARKER_FILENAME).is_file())

    def test_writes_empty_evidence_bundle_and_source_index(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        self.assertEqual(json.loads((out / "evidence_bundle.json").read_text()), [])
        self.assertEqual(json.loads((out / "source_index.json").read_text()), [])

    def test_copies_templates_into_underscore_templates(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        templates_dir = out / "_templates"
        self.assertTrue(templates_dir.is_dir())
        self.assertTrue(any(templates_dir.iterdir()))

    def test_writes_first_manifest(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        manifest = im.load_intake_manifest(out)
        self.assertEqual(manifest.evidence_count, 0)

    def test_refuses_unmarked_non_empty_directory(self):
        out = self.tmp_path / "ws"
        out.mkdir()
        (out / "some_file.txt").write_text("not ours", encoding="utf-8")
        with self.assertRaises(intake.UnsafeIntakeWorkspaceError):
            intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)

    def test_refuses_reinit_without_force(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        with self.assertRaises(intake.IntakeWorkspaceOwnershipError):
            intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)

    def test_reinit_with_force_succeeds(self):
        out = self.tmp_path / "ws"
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        profile = intake.init_intake_workspace(
            "demo", "instagram", "https://x/demo", out, self.config, display_name="Demo", force=True
        )
        self.assertEqual(profile.display_name, "Demo")

    def test_reinit_with_force_never_deletes_existing_evidence(self):
        out = self.tmp_path / "ws"
        profile = intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        caption = intake.CaptionRecord(creator_id=profile.creator_id, post_reference="p1", caption_text="hi")
        intake.add_caption(out, caption, self.config)
        intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config, force=True)
        self.assertTrue((out / "captions" / f"{caption.caption_id}.json").is_file())

    def test_creator_id_is_stable_across_reinit(self):
        out = self.tmp_path / "ws"
        first = intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config)
        second = intake.init_intake_workspace("demo", "instagram", "https://x/demo", out, self.config, force=True)
        self.assertEqual(first.creator_id, second.creator_id)


# ---------------------------------------------------------------------------
# add_* functions
# ---------------------------------------------------------------------------


class AddFunctionsTestCase(IntakeTempTestCase):
    def setUp(self):
        super().setUp()
        self.out = self.tmp_path / "ws"
        self.profile = intake.init_intake_workspace("demo", "instagram", "https://x/demo", self.out, self.config)


class AddScreenshotTests(AddFunctionsTestCase):
    def test_writes_record_file(self):
        screenshot = intake.ScreenshotEvidence(creator_id=self.profile.creator_id, source_type="grid_screenshot")
        intake.add_screenshot(self.out, screenshot, self.config)
        self.assertTrue((self.out / "screenshots" / f"{screenshot.screenshot_id}.json").is_file())

    def test_appends_evidence_to_bundle(self):
        screenshot = intake.ScreenshotEvidence(creator_id=self.profile.creator_id, source_type="grid_screenshot")
        intake.add_screenshot(self.out, screenshot, self.config)
        bundle = json.loads((self.out / "evidence_bundle.json").read_text())
        self.assertEqual(len(bundle), 1)

    def test_relationship_annotations_write_their_own_files(self):
        annotation = intake.RelationshipAnnotation(
            role=intake.RelationshipRole.FAMILY_DEPICTION, basis=RelationshipBasis.PRESENTED_NARRATIVE, confidence="low"
        )
        screenshot = intake.ScreenshotEvidence(
            creator_id=self.profile.creator_id, source_type="grid_screenshot", relationship_annotations=[annotation]
        )
        intake.add_screenshot(self.out, screenshot, self.config)
        self.assertTrue((self.out / "relationships" / f"{annotation.annotation_id}.json").is_file())

    def test_duplicate_without_force_raises(self):
        screenshot = intake.ScreenshotEvidence(creator_id=self.profile.creator_id, source_type="grid_screenshot")
        intake.add_screenshot(self.out, screenshot, self.config)
        with self.assertRaises(intake.IntakeRecordExistsError):
            intake.add_screenshot(self.out, screenshot, self.config)

    def test_duplicate_with_force_succeeds(self):
        screenshot = intake.ScreenshotEvidence(
            creator_id=self.profile.creator_id, source_type="grid_screenshot", manual_notes="v1"
        )
        intake.add_screenshot(self.out, screenshot, self.config)
        intake.add_screenshot(self.out, screenshot, self.config, force=True)

    def test_missing_local_file_raises(self):
        screenshot = intake.ScreenshotEvidence(
            creator_id=self.profile.creator_id, source_type="grid_screenshot", local_path="screenshots/missing.png"
        )
        with self.assertRaises(intake.MissingLocalFileError):
            intake.add_screenshot(self.out, screenshot, self.config)

    def test_path_traversal_local_path_rejected(self):
        screenshot = intake.ScreenshotEvidence(
            creator_id=self.profile.creator_id, source_type="grid_screenshot", local_path="../../etc/passwd"
        )
        with self.assertRaises(intake.UnsafeLocalPathError):
            intake.add_screenshot(self.out, screenshot, self.config)

    def test_existing_local_file_within_workspace_accepted(self):
        real_file = self.out / "screenshots" / "real.png"
        real_file.write_bytes(b"fake-png-bytes")
        screenshot = intake.ScreenshotEvidence(
            creator_id=self.profile.creator_id, source_type="grid_screenshot", local_path="screenshots/real.png"
        )
        manifest = intake.add_screenshot(self.out, screenshot, self.config)
        self.assertEqual(manifest.counts_by_type["grid"], 1)

    def test_requires_ownership_marker(self):
        stray_dir = self.tmp_path / "not_a_workspace"
        stray_dir.mkdir()
        screenshot = intake.ScreenshotEvidence(creator_id="c1", source_type="grid_screenshot")
        with self.assertRaises(intake.IntakeWorkspaceOwnershipError):
            intake.add_screenshot(stray_dir, screenshot, self.config)

    def test_manifest_rebuilt_after_add(self):
        screenshot = intake.ScreenshotEvidence(creator_id=self.profile.creator_id, source_type="grid_screenshot")
        manifest = intake.add_screenshot(self.out, screenshot, self.config)
        self.assertEqual(manifest.counts_by_type["grid"], 1)


class AddCaptionTests(AddFunctionsTestCase):
    def test_writes_record_and_evidence(self):
        caption = intake.CaptionRecord(creator_id=self.profile.creator_id, post_reference="p1", caption_text="hi")
        manifest = intake.add_caption(self.out, caption, self.config)
        self.assertTrue((self.out / "captions" / f"{caption.caption_id}.json").is_file())
        self.assertEqual(manifest.counts_by_type["captions"], 1)

    def test_preserves_exact_caption_text(self):
        text = "Exact text, not rewritten!! 🌸"
        caption = intake.CaptionRecord(creator_id=self.profile.creator_id, post_reference="p1", caption_text=text)
        intake.add_caption(self.out, caption, self.config)
        stored = json.loads((self.out / "captions" / f"{caption.caption_id}.json").read_text())
        self.assertEqual(stored["caption_text"], text)

    def test_requires_ownership_marker(self):
        stray_dir = self.tmp_path / "stray"
        stray_dir.mkdir()
        caption = intake.CaptionRecord(creator_id="c1", post_reference="p1", caption_text="hi")
        with self.assertRaises(intake.IntakeWorkspaceOwnershipError):
            intake.add_caption(stray_dir, caption, self.config)


class AddReplyPairTests(AddFunctionsTestCase):
    def test_writes_record_and_evidence(self):
        pair = intake.ReplyPair(creator_id=self.profile.creator_id, post_reference="p1", audience_comment="hi", creator_reply="hello")
        manifest = intake.add_reply_pair(self.out, pair, self.config)
        self.assertEqual(manifest.counts_by_type["creator_replies"], 1)

    def test_redacted_username_allowed(self):
        pair = intake.ReplyPair(
            creator_id=self.profile.creator_id, post_reference="p1", audience_comment="hi", creator_reply="hello",
            audience_username_optional="***redacted***",
        )
        intake.add_reply_pair(self.out, pair, self.config)
        stored = json.loads((self.out / "replies" / f"{pair.reply_pair_id}.json").read_text())
        self.assertEqual(stored["audience_username_optional"], "***redacted***")


class AddReelNoteTests(AddFunctionsTestCase):
    def test_writes_record_with_shot_notes(self):
        reel = intake.ReelEvidence(
            creator_id=self.profile.creator_id, source_reference="r1",
            shot_notes=[intake.ShotNote(shot_index=0, subject_action="walk"), intake.ShotNote(shot_index=1, subject_action="turn")],
        )
        manifest = intake.add_reel_note(self.out, reel, self.config)
        self.assertEqual(manifest.counts_by_type["reels"], 1)
        stored = json.loads((self.out / "reels" / f"{reel.reel_id}.json").read_text())
        self.assertEqual(len(stored["shot_notes"]), 2)

    def test_never_downloads_video(self):
        # Structural guarantee at the function level: no video bytes
        # are ever fetched/written by add_reel_note.
        import inspect

        source = inspect.getsource(intake.add_reel_note)
        for forbidden in ("requests.", "urllib.", "yt_dlp", "ffmpeg"):
            self.assertNotIn(forbidden, source)


class AddHighlightNoteTests(AddFunctionsTestCase):
    def test_writes_record_and_evidence(self):
        highlight = intake.HighlightItem(creator_id=self.profile.creator_id, highlight_name="Travel", item_index=0, manual_summary="beach")
        manifest = intake.add_highlight_note(self.out, highlight, self.config)
        self.assertEqual(manifest.counts_by_type["highlights"], 1)

    def test_does_not_infer_unsupported_category(self):
        highlight = intake.HighlightItem(creator_id=self.profile.creator_id, highlight_name="Misc", item_index=0)
        self.assertIsNone(highlight.story_type)


class AddManualNoteTests(AddFunctionsTestCase):
    def test_writes_record_and_evidence(self):
        manifest = intake.add_manual_note(
            self.out, creator_id_value=self.profile.creator_id, source_description="general observation",
            content_excerpt="warm tone", intake_config=self.config,
        )
        self.assertGreaterEqual(manifest.evidence_count, 1)

    def test_invalid_evidence_type_never_reachable(self):
        # add_manual_note always constructs EvidenceType.MANUAL_NOTE --
        # there is no way to pass an arbitrary evidence_type through it.
        import inspect

        signature = inspect.signature(intake.add_manual_note)
        self.assertNotIn("evidence_type", signature.parameters)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliInitTests(IntakeTempTestCase):
    def test_init_creates_workspace(self):
        out = self.tmp_path / "ws"
        rc = intake.main(
            ["--init", "--username", "demo", "--platform", "instagram", "--profile-url", "https://x/demo", "--output", str(out), "--json"]
        )
        self.assertEqual(rc, 0)
        self.assertTrue((out / intake.MARKER_FILENAME).is_file())

    def test_init_without_username_fails_argparse(self):
        out = self.tmp_path / "ws"
        with self.assertRaises(SystemExit):
            intake.main(["--init", "--platform", "instagram", "--profile-url", "https://x/demo", "--output", str(out)])

    def test_init_twice_without_force_exits_nonzero(self):
        out = self.tmp_path / "ws"
        intake.main(["--init", "--username", "demo", "--platform", "instagram", "--profile-url", "https://x/demo", "--output", str(out), "--json"])
        rc = intake.main(["--init", "--username", "demo", "--platform", "instagram", "--profile-url", "https://x/demo", "--output", str(out), "--json"])
        self.assertEqual(rc, 1)


class CliAddTests(IntakeTempTestCase):
    def setUp(self):
        super().setUp()
        self.out = self.tmp_path / "ws"
        intake.main(["--init", "--username", "demo", "--platform", "instagram", "--profile-url", "https://x/demo", "--output", str(self.out), "--json"])

    def _write_json(self, name, payload):
        path = self.tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_add_caption_via_cli(self):
        path = self._write_json("cap.json", {"post_reference": "p1", "caption_text": "hi"})
        rc = intake.main(["--add-caption", str(path), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 0)

    def test_add_caption_missing_file_exits_nonzero(self):
        rc = intake.main(["--add-caption", str(self.tmp_path / "missing.json"), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 1)

    def test_add_reply_pair_via_cli(self):
        path = self._write_json("reply.json", {"post_reference": "p1", "audience_comment": "hi", "creator_reply": "hello"})
        rc = intake.main(["--add-reply-pair", str(path), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 0)

    def test_add_reel_note_via_cli(self):
        path = self._write_json(
            "reel.json", {"source_reference": "r1", "hook_notes": "zoom", "shot_notes": [{"shot_index": 0, "subject_action": "walk"}]}
        )
        rc = intake.main(["--add-reel-note", str(path), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 0)

    def test_add_highlight_note_via_cli(self):
        path = self._write_json("hi.json", {"highlight_name": "Travel", "item_index": 0, "manual_summary": "beach"})
        rc = intake.main(["--add-highlight-note", str(path), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 0)

    def test_add_manual_note_via_cli(self):
        path = self._write_json("note.json", {"source_description": "note", "content_excerpt": "warm", "tags": ["persona"]})
        rc = intake.main(["--add-manual-note", str(path), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 0)

    def test_add_screenshot_via_cli_with_relationship_annotation(self):
        path = self._write_json(
            "shot.json",
            {
                "source_type": "grid_screenshot",
                "manual_notes": "grid",
                "relationship_annotations": [
                    {"role": "family_depiction", "basis": "presented_narrative", "confidence": "low", "description": "childhood photo"}
                ],
            },
        )
        rc = intake.main(["--add-screenshot", str(path), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 0)

    def test_add_screenshot_with_comment_keys_stripped(self):
        path = self._write_json(
            "shot2.json", {"_comment": "ignored", "source_type": "grid_screenshot", "manual_notes": "grid"}
        )
        rc = intake.main(["--add-screenshot", str(path), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 0)

    def test_malformed_json_file_exits_nonzero(self):
        path = self.tmp_path / "bad.json"
        path.write_text("{not valid", encoding="utf-8")
        rc = intake.main(["--add-caption", str(path), "--output", str(self.out), "--json"])
        self.assertEqual(rc, 1)


class CliRequiresExactlyOneActionTests(unittest.TestCase):
    def test_no_action_fails_argparse(self):
        with self.assertRaises(SystemExit):
            intake.parse_arguments(["--output", "/tmp/x"])


# ---------------------------------------------------------------------------
# Structural safety / scope boundaries
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(intake.__file__).read_text(encoding="utf-8")

    def test_no_network_or_browser_imports(self):
        for line in self.source.splitlines():
            stripped = line.strip()
            for forbidden in ("import requests", "from requests", "import httpx", "import playwright", "from playwright", "import selenium", "from selenium"):
                self.assertFalse(stripped.startswith(forbidden), stripped)

    def test_no_persona_social_publishing_reference(self):
        for forbidden in ("03_personas", "from .social", "from src.social", "from .publishing", "from src.publishing"):
            self.assertNotIn(forbidden, self.source)

    def test_no_network_calls(self):
        for forbidden in ("requests.", "urllib.request", "http.client", "socket."):
            self.assertNotIn(forbidden, self.source)

    def test_no_real_creator_handle_in_source(self):
        for handle in ("carlysuen112", "aitana_10_01"):
            self.assertNotIn(handle, self.source)

    def test_no_subprocess_usage(self):
        for line in self.source.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import subprocess"))
            self.assertFalse(stripped.startswith("from subprocess"))


class NoActiveAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(intake.__file__).read_text(encoding="utf-8")

    def test_does_not_import_any_analyzer(self):
        for forbidden in (
            "persona_analyzer", "visual_realism_analyzer", "photography_analyzer", "human_authenticity_analyzer",
            "caption_analyzer", "reply_analyzer", "storytelling_analyzer", "reels_analyzer", "relationship_analyzer",
            "summary_builder", "analyzer_base",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_does_not_call_build_creator_dna(self):
        self.assertNotIn("build_creator_dna", self.source)


if __name__ == "__main__":
    unittest.main()
