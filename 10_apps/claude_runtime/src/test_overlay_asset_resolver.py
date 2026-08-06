from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from . import overlay_asset_resolver as oar
from . import overlay_plan_engine, renderer_plan_engine, timeline_engine

MODULE_PATH = Path(__file__).resolve().parent / "overlay_asset_resolver.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeInspector:
    """Injectable stand-in for a real image inspector. Never touches
    Pillow. `overrides` maps str(path) -> OverlayAssetInspection for
    per-path customization; otherwise `default` is returned."""

    def __init__(self, *, default: oar.OverlayAssetInspection | None = None,
                 overrides: dict[str, oar.OverlayAssetInspection] | None = None,
                 raise_exception: Exception | None = None):
        self.default = default or oar.OverlayAssetInspection(
            width=200, height=100, format="PNG", color_mode="RGBA", has_alpha=True,
            alpha_mode=oar.AlphaMode.STRAIGHT, animated=False, frame_count=1,
        )
        self.overrides = overrides or {}
        self.raise_exception = raise_exception
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> oar.OverlayAssetInspection:
        self.calls.append(path)
        if self.raise_exception is not None:
            raise self.raise_exception
        return self.overrides.get(str(path), self.default)


class OverlayAssetResolverTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="overlay_asset_resolver_test_"))
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.config = oar.load_overlay_asset_config()

    def _png(self, name: str = "logo.png", content: bytes = b"fake-png-bytes") -> Path:
        path = self.temp_dir / name
        path.write_bytes(content)
        return path

    def _webp(self, name: str = "sticker.webp", content: bytes = b"fake-webp-bytes") -> Path:
        path = self.temp_dir / name
        path.write_bytes(content)
        return path

    def _jpeg(self, name: str = "photo.jpg", content: bytes = b"fake-jpeg-bytes") -> Path:
        path = self.temp_dir / name
        path.write_bytes(content)
        return path

    def _gif(self, name: str = "anim.gif", content: bytes = b"fake-gif-bytes") -> Path:
        path = self.temp_dir / name
        path.write_bytes(content)
        return path

    def _reference(self, **overrides) -> oar.OverlayAssetReference:
        raw_path = overrides.pop("raw_path", None) or str(self._png())
        defaults = dict(
            reference_id="ref1", source_plan_id="plan1", source_overlay_id="ov1", source_asset_id="",
            logical_role=oar.OverlayLogicalRole.LOGO, asset_type=oar.OverlayAssetType.PNG,
            raw_path=raw_path, required=True, requires_alpha=True, metadata={},
        )
        defaults.update(overrides)
        return oar.OverlayAssetReference(**defaults)

    def _overlay_plan_with(self, overlay_type: str, asset_reference: str | None, *, enabled=True,
                            content: str = "", position=None) -> overlay_plan_engine.OverlayPlan:
        scene = self.temp_dir / "scene.mp4"
        if not scene.exists():
            scene.write_bytes(b"x" * 16)
        # x=540/y=1400 sits inside the default action-safe zone for a
        # 1080x1920 canvas ([120, 960] x [220, 1500]) -- an overlay
        # outside that zone fails Overlay Plan Engine's own validation
        # before this module is ever reached.
        metadata = {"position": position or {"x": 540, "y": 1400}}
        if asset_reference is not None:
            metadata["asset"] = {"asset_reference": asset_reference}
        overlay_clip = timeline_engine.OverlayClip(
            clip_id="ov1", track_id="track_overlay", source_path="", overlay_type=overlay_type,
            start=0.0, end=3.0, duration_seconds=3.0, source_out=3.0, content=content, metadata=metadata,
        )
        if not enabled:
            overlay_clip = dataclasses.replace(overlay_clip, enabled=False)
        timeline = timeline_engine.Timeline(
            timeline_id="tl1", duration_seconds=5.0,
            tracks=[
                timeline_engine.TimelineTrack(
                    track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0,
                    clips=[timeline_engine.VideoClip(
                        clip_id="c1", track_id="track_video", source_path=str(scene), start=0.0, end=5.0,
                        duration_seconds=5.0, source_out=5.0, scene_number=1,
                    )],
                ),
                timeline_engine.TimelineTrack(
                    track_id="track_overlay", track_type=timeline_engine.TrackType.OVERLAY, order=1,
                    clips=[overlay_clip],
                ),
            ],
        )
        timeline_path = self.temp_dir / f"timeline_{overlay_type}.json"
        timeline_engine.save_timeline(timeline, timeline_path, force=True)
        overlay_config = overlay_plan_engine.load_overlay_plan_config()
        return overlay_plan_engine.build_overlay_plan(timeline_path, overlay_config)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationTests(unittest.TestCase):
    def test_default_config_loads(self):
        config = oar.load_overlay_asset_config()
        self.assertIn(".png", config.supported_extensions_enabled)
        self.assertIn(".gif", config.supported_extensions_disabled)
        self.assertTrue(config.fail_on_missing_required_asset)
        self.assertFalse(config.fail_on_invalid_optional_asset)

    def test_missing_config_raises(self):
        with self.assertRaises(oar.OverlayAssetConfigError):
            oar.load_overlay_asset_config("/nonexistent/overlay_assets.yaml")

    def test_empty_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(oar.OverlayAssetConfigError):
                oar.load_overlay_asset_config(path)

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("resolver: [unclosed", encoding="utf-8")
            with self.assertRaises(oar.OverlayAssetConfigError):
                oar.load_overlay_asset_config(path)

    def test_extension_policy_loads(self):
        config = oar.load_overlay_asset_config()
        self.assertEqual(set(config.supported_extensions_enabled), {".png", ".webp", ".jpg", ".jpeg"})
        self.assertEqual(set(config.supported_extensions_disabled), {".gif", ".svg"})

    def test_alpha_policies_load(self):
        config = oar.load_overlay_asset_config()
        self.assertIn("logo", config.alpha_roles_requiring_alpha)
        self.assertEqual(config.alpha_missing_alpha_policy["logo"], "error")
        self.assertEqual(config.alpha_missing_alpha_policy["cta"], "warning")

    def test_dimension_policies_load(self):
        config = oar.load_overlay_asset_config()
        logo_policy = config.dimension_policy("logo")
        self.assertEqual(logo_policy.minimum_width, 128)
        default_policy = config.dimension_policy("unknown_role")
        self.assertEqual(default_policy.minimum_width, 32)

    def test_dimension_policy_merges_default_and_role(self):
        config = oar.load_overlay_asset_config()
        logo_policy = config.dimension_policy("logo")
        # maximum_megapixels comes from default, not overridden by logo.
        self.assertEqual(logo_policy.maximum_megapixels, 40)

    def test_checksum_settings_load(self):
        config = oar.load_overlay_asset_config()
        self.assertTrue(config.checksum_enabled)
        self.assertEqual(config.checksum_algorithm, "sha256")
        self.assertEqual(config.checksum_chunk_size_bytes, 1048576)

    def test_duplicate_policies_load(self):
        config = oar.load_overlay_asset_config()
        self.assertEqual(config.duplicate_same_path_same_role, "merge_references")
        self.assertEqual(config.duplicate_same_path_different_role, "separate_assets")
        self.assertEqual(config.duplicate_conflicting_requirements, "error")
        self.assertFalse(config.duplicate_include_source_overlay_in_asset_id)

    def test_default_config_path_resolves(self):
        path = oar.default_overlay_asset_config_path()
        self.assertTrue(str(path).endswith("config/video/overlay_assets.yaml"))


# ---------------------------------------------------------------------------
# Reference collection
# ---------------------------------------------------------------------------


class ReferenceCollectionTests(OverlayAssetResolverTempTestCase):
    def test_logo_reference_collected(self):
        logo = self._png()
        plan = self._overlay_plan_with("logo", str(logo))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].logical_role, "logo")
        self.assertEqual(refs[0].raw_path, str(logo))
        self.assertTrue(refs[0].requires_alpha)

    def test_watermark_reference_collected(self):
        watermark = self._png("watermark.png")
        plan = self._overlay_plan_with("watermark", str(watermark))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs[0].logical_role, "watermark")

    def test_sticker_reference_collected(self):
        sticker = self._webp()
        plan = self._overlay_plan_with("sticker", str(sticker))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs[0].logical_role, "sticker")

    def test_cta_reference_collected(self):
        cta = self._png("cta.png")
        plan = self._overlay_plan_with("cta", str(cta))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs[0].logical_role, "cta")
        self.assertFalse(refs[0].requires_alpha)

    def test_location_reference_collected(self):
        location = self._png("location.png")
        plan = self._overlay_plan_with("location", str(location))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs[0].logical_role, "location")

    def test_custom_reference_collected(self):
        custom = self._png("custom.png")
        plan = self._overlay_plan_with("custom", str(custom))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(len(refs), 1)

    def test_renderer_asset_reference_collected(self):
        asset = renderer_plan_engine.RendererAsset(
            asset_id="asset_image_0001", asset_type=renderer_plan_engine.AssetType.IMAGE,
            source_path=str(self._png()), logical_role="logo",
        )
        plan = renderer_plan_engine.RendererPlan(renderer_plan_id="rp1", assets=[asset])
        refs = oar.collect_overlay_asset_references(renderer_plan=plan, config=self.config)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].source_asset_id, "asset_image_0001")
        self.assertEqual(refs[0].logical_role, "logo")

    def test_renderer_non_image_asset_skipped(self):
        asset = renderer_plan_engine.RendererAsset(
            asset_id="asset_video_0001", asset_type=renderer_plan_engine.AssetType.VIDEO,
            source_path=str(self._png()), logical_role="scene_video",
        )
        plan = renderer_plan_engine.RendererPlan(renderer_plan_id="rp1", assets=[asset])
        refs = oar.collect_overlay_asset_references(renderer_plan=plan, config=self.config)
        self.assertEqual(refs, [])

    def test_missing_raw_path_rejected_not_invented(self):
        plan = self._overlay_plan_with("logo", None)
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs, [])

    def test_subtitle_overlay_never_collected(self):
        plan = self._overlay_plan_with("subtitle", str(self._png()), content="hello there")
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs, [])

    def test_disabled_overlay_never_collected(self):
        plan = self._overlay_plan_with("logo", str(self._png()), enabled=False)
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs, [])

    def test_source_ids_preserved(self):
        logo = self._png()
        plan = self._overlay_plan_with("logo", str(logo))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs[0].source_plan_id, plan.plan_id)
        self.assertEqual(refs[0].source_overlay_id, "overlay_ov1")

    def test_metadata_preserved(self):
        logo = self._png()
        plan = self._overlay_plan_with("logo", str(logo))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertIn("cue_metadata", refs[0].metadata) if refs[0].metadata else None  # tolerate empty
        self.assertIsInstance(refs[0].metadata, dict)

    def test_no_auto_discovery_of_unreferenced_files(self):
        # A PNG sitting in the temp dir that no overlay ever references
        # must never appear as a collected reference.
        self._png("unreferenced.png")
        plan = self._overlay_plan_with("logo", str(self._png("actual_logo.png")))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(len(refs), 1)
        self.assertNotIn("unreferenced.png", refs[0].raw_path)

    def test_content_fallback_used_when_path_like(self):
        logo = self._png()
        plan = self._overlay_plan_with("logo", None, content=str(logo))
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].raw_path, str(logo))

    def test_content_not_used_when_not_path_like(self):
        plan = self._overlay_plan_with("cta", None, content="Shop Now")
        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        self.assertEqual(refs, [])

    def test_combined_overlay_and_renderer_plan(self):
        logo = self._png()
        overlay_plan = self._overlay_plan_with("logo", str(logo))
        asset = renderer_plan_engine.RendererAsset(
            asset_id="asset_image_0001", asset_type=renderer_plan_engine.AssetType.IMAGE,
            source_path=str(self._png("watermark2.png")), logical_role="watermark",
        )
        renderer_plan = renderer_plan_engine.RendererPlan(renderer_plan_id="rp1", assets=[asset])
        refs = oar.collect_overlay_asset_references(overlay_plan=overlay_plan, renderer_plan=renderer_plan, config=self.config)
        self.assertEqual(len(refs), 2)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class PathSafetyTests(OverlayAssetResolverTempTestCase):
    def test_valid_relative_path_resolves(self):
        logo = self._png()
        config = dataclasses.replace(self.config, project_root=str(self.temp_dir))
        reference = self._reference(raw_path="logo.png")
        asset = oar.resolve_overlay_asset(reference, config, inspector=FakeInspector())
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)

    def test_valid_absolute_path_resolves(self):
        logo = self._png()
        reference = self._reference(raw_path=str(logo))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)

    def test_absolute_path_rejected_when_disallowed(self):
        logo = self._png()
        config = dataclasses.replace(self.config, allow_absolute_paths=False)
        reference = self._reference(raw_path=str(logo))
        with self.assertRaises(oar.UnsafeOverlayAssetPathError):
            oar.resolve_overlay_asset(reference, config, inspector=FakeInspector())

    def test_missing_file_fails(self):
        reference = self._reference(raw_path=str(self.temp_dir / "missing.png"))
        with self.assertRaises(oar.OverlayAssetNotFoundError):
            oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())

    def test_directory_fails(self):
        directory = self.temp_dir / "adir.png"
        directory.mkdir()
        reference = self._reference(raw_path=str(directory))
        with self.assertRaises(oar.UnsafeOverlayAssetPathError):
            oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())

    def test_empty_file_fails(self):
        empty = self._png("empty.png", content=b"")
        reference = self._reference(raw_path=str(empty))
        with self.assertRaises(oar.OverlayAssetEmptyError):
            oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())

    def test_broken_symlink_fails(self):
        link = self.temp_dir / "broken.png"
        link.symlink_to(self.temp_dir / "does_not_exist.png")
        reference = self._reference(raw_path=str(link))
        with self.assertRaises(oar.OverlayAssetNotFoundError):
            oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())

    def test_valid_symlink_resolves(self):
        target = self._png("target.png")
        link = self.temp_dir / "link.png"
        link.symlink_to(target)
        reference = self._reference(raw_path=str(link))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)

    def test_symlink_rejected_when_not_followed(self):
        target = self._png("target2.png")
        link = self.temp_dir / "link2.png"
        link.symlink_to(target)
        config = dataclasses.replace(self.config, follow_symlinks=False)
        reference = self._reference(raw_path=str(link))
        with self.assertRaises(oar.UnsafeOverlayAssetPathError):
            oar.resolve_overlay_asset(reference, config, inspector=FakeInspector())

    def test_duplicate_resolved_path_detected_in_batch(self):
        logo = self._png()
        ref_a = self._reference(reference_id="a", raw_path=str(logo))
        ref_b = self._reference(reference_id="b", raw_path=str(logo))
        result = oar.resolve_overlay_assets([ref_a, ref_b], self.config, inspector=FakeInspector())
        self.assertEqual(len(result.assets), 1)
        self.assertEqual(sorted(result.assets[0].reference_ids), ["a", "b"])

    def test_unsupported_extension_fails(self):
        bmp = self.temp_dir / "image.bmp"
        bmp.write_bytes(b"fake-bmp")
        reference = self._reference(raw_path=str(bmp))
        with self.assertRaises(oar.UnsupportedOverlayAssetError):
            oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())

    def test_gif_rejected_by_default(self):
        gif = self._gif()
        reference = self._reference(raw_path=str(gif))
        with self.assertRaises(oar.UnsupportedOverlayAssetError):
            oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())

    def test_svg_rejected_by_default(self):
        svg = self.temp_dir / "icon.svg"
        svg.write_text("<svg></svg>", encoding="utf-8")
        reference = self._reference(raw_path=str(svg))
        with self.assertRaises(oar.UnsupportedOverlayAssetError):
            oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())

    def test_hidden_file_policy_rejects_when_enabled(self):
        hidden = self.temp_dir / ".hidden.png"
        hidden.write_bytes(b"fake-png")
        config = dataclasses.replace(self.config, reject_hidden_files=True)
        reference = self._reference(raw_path=str(hidden))
        with self.assertRaises(oar.UnsafeOverlayAssetPathError):
            oar.resolve_overlay_asset(reference, config, inspector=FakeInspector())

    def test_hidden_file_policy_allows_by_default(self):
        hidden = self.temp_dir / ".hidden2.png"
        hidden.write_bytes(b"fake-png")
        reference = self._reference(raw_path=str(hidden))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)

    def test_source_equal_to_manifest_output_rejected_in_cli(self):
        logo = self._png()
        plan = self._overlay_plan_with("logo", str(logo))
        overlay_plan_path = self.temp_dir / "overlay_plan_cli.json"
        overlay_plan_engine.save_overlay_plan(plan, overlay_plan_path, force=True)
        import contextlib
        import io
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            oar.main(["--overlay-plan", str(overlay_plan_path), "--output", str(overlay_plan_path)])
        self.assertIn("[OverlayAssetResolver]", buffer.getvalue())


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


class InspectionTests(OverlayAssetResolverTempTestCase):
    def test_png_metadata_accepted(self):
        png = self._png()
        reference = self._reference(raw_path=str(png))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        self.assertEqual(asset.asset_type, "png")
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)

    def test_webp_accepted(self):
        webp = self._webp()
        reference = self._reference(raw_path=str(webp), logical_role=oar.OverlayLogicalRole.STICKER)
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        self.assertEqual(asset.asset_type, "webp")

    def test_jpeg_accepted(self):
        jpeg = self._jpeg()
        reference = self._reference(raw_path=str(jpeg), logical_role=oar.OverlayLogicalRole.CTA, requires_alpha=False)
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector(default=oar.OverlayAssetInspection(width=300, height=150, has_alpha=False, alpha_mode=oar.AlphaMode.NONE)))
        self.assertEqual(asset.asset_type, "jpeg")

    def test_missing_inspector_fails_clearly(self):
        png = self._png()
        reference = self._reference(raw_path=str(png))
        with self.assertRaises(oar.OverlayAssetInspectorUnavailableError):
            oar.resolve_overlay_asset(reference, self.config, inspector=None)

    def test_missing_inspector_ok_when_not_required(self):
        png = self._png()
        # CTA's default alpha policy is "warning", not "error", so the
        # unknown-alpha outcome (no inspector, no data) does not raise.
        config = dataclasses.replace(self.config, require_inspector=False)
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.CTA, requires_alpha=False)
        asset = oar.resolve_overlay_asset(reference, config, inspector=None)
        # No dimension/alpha data at all without an inspector -- but the
        # file itself still resolves structurally.
        self.assertIsNone(asset.width)

    def test_inspector_exception_wrapped(self):
        png = self._png()
        reference = self._reference(raw_path=str(png))
        inspector = FakeInspector(raise_exception=ValueError("corrupt file"))
        with self.assertRaises(oar.OverlayAssetInspectionError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_animated_asset_rejected(self):
        webp = self._webp()
        reference = self._reference(raw_path=str(webp), logical_role=oar.OverlayLogicalRole.STICKER)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=100, height=100, has_alpha=True, animated=True))
        with self.assertRaises(oar.OverlayAssetAnimatedError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_animated_asset_allowed_when_configured(self):
        webp = self._webp()
        config = dataclasses.replace(self.config, reject_animated_assets=False)
        reference = self._reference(raw_path=str(webp), logical_role=oar.OverlayLogicalRole.STICKER)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=100, height=100, has_alpha=True, animated=True))
        asset = oar.resolve_overlay_asset(reference, config, inspector=inspector)
        self.assertTrue(asset.animated)

    def test_unknown_format_handled(self):
        png = self._png()
        reference = self._reference(raw_path=str(png))
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=10, height=10, format=None, has_alpha=True))
        asset = oar.resolve_overlay_asset(reference, dataclasses.replace(self.config, fail_on_missing_required_asset=False), inspector=inspector)
        self.assertIsNotNone(asset)

    def test_width_height_preserved(self):
        png = self._png()
        reference = self._reference(raw_path=str(png))
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=321, height=654, has_alpha=True))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=inspector)
        self.assertEqual(asset.width, 321)
        self.assertEqual(asset.height, 654)

    def test_color_mode_preserved(self):
        png = self._png()
        reference = self._reference(raw_path=str(png))
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=True, color_mode="RGBA"))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=inspector)
        self.assertEqual(asset.color_mode, "RGBA")

    def test_alpha_metadata_preserved(self):
        png = self._png()
        reference = self._reference(raw_path=str(png))
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=True, alpha_mode=oar.AlphaMode.PREMULTIPLIED))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=inspector)
        self.assertEqual(asset.alpha_mode, oar.AlphaMode.PREMULTIPLIED)


# ---------------------------------------------------------------------------
# Alpha validation
# ---------------------------------------------------------------------------


class AlphaTests(OverlayAssetResolverTempTestCase):
    def test_logo_with_alpha_passes(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)

    def test_logo_without_alpha_fails(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=False, alpha_mode=oar.AlphaMode.NONE))
        with self.assertRaises(oar.OverlayAssetAlphaError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_watermark_without_alpha_fails(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.WATERMARK)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=False))
        with self.assertRaises(oar.OverlayAssetAlphaError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_sticker_without_alpha_fails(self):
        webp = self._webp()
        reference = self._reference(raw_path=str(webp), logical_role=oar.OverlayLogicalRole.STICKER)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=100, height=100, has_alpha=False))
        with self.assertRaises(oar.OverlayAssetAlphaError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_cta_without_alpha_warns(self):
        jpeg = self._jpeg()
        reference = self._reference(raw_path=str(jpeg), logical_role=oar.OverlayLogicalRole.CTA, requires_alpha=False)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=300, height=150, has_alpha=False))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=inspector)
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)
        self.assertTrue(any("alpha" in w for w in asset.warnings))

    def test_unknown_alpha_follows_config_error_policy(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=None))
        with self.assertRaises(oar.OverlayAssetAlphaError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_unknown_alpha_allowed_when_configured(self):
        png = self._png()
        config = dataclasses.replace(self.config, allow_unknown_alpha=True)
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=None))
        asset = oar.resolve_overlay_asset(reference, config, inspector=inspector)
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)
        self.assertTrue(asset.warnings)

    def test_jpeg_alpha_required_policy_raises_for_logo(self):
        jpeg = self._jpeg()
        reference = self._reference(raw_path=str(jpeg), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=False))
        with self.assertRaises(oar.OverlayAssetAlphaError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_role_with_no_configured_policy_skips_check_entirely(self):
        png = self._png()
        config = dataclasses.replace(self.config, alpha_missing_alpha_policy={}, alpha_roles_requiring_alpha=())
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.DECORATIVE, requires_alpha=False)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=False))
        asset = oar.resolve_overlay_asset(reference, config, inspector=inspector)
        self.assertEqual(asset.warnings, [])

    def test_decorative_without_alpha_warns_by_default(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.DECORATIVE, requires_alpha=False)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=False))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=inspector)
        self.assertTrue(any("alpha" in w for w in asset.warnings))

    def test_no_alpha_synthesis(self):
        for forbidden in ("add_alpha_channel(", "synthesize_alpha(", "convert('RGBA')", 'convert("RGBA")'):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_image_conversion(self):
        self.assertNotIn(".save(", MODULE_SOURCE)
        self.assertNotIn(".convert(", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Dimension validation
# ---------------------------------------------------------------------------


class DimensionTests(OverlayAssetResolverTempTestCase):
    def test_valid_dimensions_pass(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=200, height=100, has_alpha=True))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=inspector)
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)

    def test_too_small_fails_for_logo(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=10, height=10, has_alpha=True))
        with self.assertRaises(oar.OverlayAssetDimensionError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_too_large_fails(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=20000, height=20000, has_alpha=True))
        with self.assertRaises(oar.OverlayAssetDimensionError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_megapixel_limit_enforced(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        # 7000x7000 = 49MP > default max 40MP, but within 8192 bounds.
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=7000, height=7000, has_alpha=True))
        with self.assertRaises(oar.OverlayAssetDimensionError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_logo_aspect_ratio_enforced(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=2000, height=100, has_alpha=True))  # ratio 20 > max 8
        with self.assertRaises(oar.OverlayAssetDimensionError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_watermark_aspect_ratio_enforced(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.WATERMARK)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=2000, height=50, has_alpha=True))  # ratio 40 > max 10
        with self.assertRaises(oar.OverlayAssetDimensionError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_sticker_preference_warning_not_error(self):
        webp = self._webp()
        reference = self._reference(raw_path=str(webp), logical_role=oar.OverlayLogicalRole.STICKER)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=300, height=100, has_alpha=True))  # ratio 3.0 > preferred max 2.0
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=inspector)
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)
        self.assertTrue(any("preferred_aspect_ratio" in w for w in asset.warnings))

    def test_cta_minimum_size_enforced(self):
        jpeg = self._jpeg()
        reference = self._reference(raw_path=str(jpeg), logical_role=oar.OverlayLogicalRole.CTA, requires_alpha=False)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=50, height=50, has_alpha=False))
        with self.assertRaises(oar.OverlayAssetDimensionError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_zero_dimension_rejected(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=0, height=100, has_alpha=True))
        with self.assertRaises(oar.OverlayAssetDimensionError):
            oar.resolve_overlay_asset(reference, self.config, inspector=inspector)

    def test_no_dimensions_from_inspector_skips_dimension_check(self):
        png = self._png()
        reference = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        inspector = FakeInspector(default=oar.OverlayAssetInspection(width=None, height=None, has_alpha=True))
        asset = oar.resolve_overlay_asset(reference, self.config, inspector=inspector)
        self.assertEqual(asset.status, oar.AssetResolutionStatus.RESOLVED)

    def test_no_resize_performed(self):
        self.assertNotIn(".resize(", MODULE_SOURCE)
        self.assertNotIn(".thumbnail(", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


class ChecksumTests(OverlayAssetResolverTempTestCase):
    def test_deterministic_checksum(self):
        png = self._png(content=b"stable-content")
        reference = self._reference(raw_path=str(png))
        asset_a = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        asset_b = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        self.assertEqual(asset_a.checksum, asset_b.checksum)

    def test_chunked_reading_small_chunk_size(self):
        png = self._png(content=b"x" * 5000)
        config = dataclasses.replace(self.config, checksum_chunk_size_bytes=16)
        reference = self._reference(raw_path=str(png))
        asset = oar.resolve_overlay_asset(reference, config, inspector=FakeInspector())
        self.assertIsNotNone(asset.checksum)
        self.assertEqual(len(asset.checksum), 64)  # sha256 hex length

    def test_checksum_algorithm_config(self):
        png = self._png()
        config = dataclasses.replace(self.config, checksum_algorithm="md5")
        reference = self._reference(raw_path=str(png))
        asset = oar.resolve_overlay_asset(reference, config, inspector=FakeInspector())
        self.assertEqual(len(asset.checksum), 32)  # md5 hex length

    def test_checksum_disabled(self):
        png = self._png()
        config = dataclasses.replace(self.config, checksum_enabled=False)
        reference = self._reference(raw_path=str(png))
        asset = oar.resolve_overlay_asset(reference, config, inspector=FakeInspector())
        self.assertIsNone(asset.checksum)

    def test_checksum_changes_when_file_changes(self):
        png_a = self._png("a.png", content=b"content-a")
        png_b = self._png("b.png", content=b"content-b")
        asset_a = oar.resolve_overlay_asset(self._reference(raw_path=str(png_a)), self.config, inspector=FakeInspector())
        asset_b = oar.resolve_overlay_asset(self._reference(raw_path=str(png_b)), self.config, inspector=FakeInspector())
        self.assertNotEqual(asset_a.checksum, asset_b.checksum)

    def test_checksum_failure_wrapped(self):
        png = self._png()
        config = dataclasses.replace(self.config, checksum_algorithm="not-a-real-algorithm")
        with self.assertRaises(oar.OverlayAssetChecksumError):
            oar.resolve_overlay_asset(self._reference(raw_path=str(png)), config, inspector=FakeInspector())

    def test_binary_content_never_logged(self):
        # The module never reads a whole asset into a variable it might
        # log/print -- only chunked hashing and Pillow's own I/O.
        self.assertNotIn("print(chunk", MODULE_SOURCE)
        self.assertNotIn(".read())", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Duplicates / conflicts
# ---------------------------------------------------------------------------


class DuplicatesConflictsTests(OverlayAssetResolverTempTestCase):
    def test_same_path_same_role_merges_references(self):
        logo = self._png()
        ref_a = self._reference(reference_id="a", source_overlay_id="ov_a", raw_path=str(logo))
        ref_b = self._reference(reference_id="b", source_overlay_id="ov_b", raw_path=str(logo))
        result = oar.resolve_overlay_assets([ref_a, ref_b], self.config, inspector=FakeInspector())
        self.assertEqual(len(result.assets), 1)
        self.assertEqual(sorted(result.assets[0].reference_ids), ["a", "b"])
        self.assertEqual(sorted(result.assets[0].source_overlay_ids), ["ov_a", "ov_b"])

    def test_same_path_different_role_stays_separate(self):
        logo = self._png()
        ref_a = self._reference(reference_id="a", raw_path=str(logo), logical_role=oar.OverlayLogicalRole.LOGO)
        ref_b = self._reference(reference_id="b", raw_path=str(logo), logical_role=oar.OverlayLogicalRole.WATERMARK)
        result = oar.resolve_overlay_assets([ref_a, ref_b], self.config, inspector=FakeInspector())
        self.assertEqual(len(result.assets), 2)

    def test_merge_disabled_keeps_separate_assets(self):
        logo = self._png()
        config = dataclasses.replace(self.config, duplicate_same_path_same_role="duplicate_per_role")
        ref_a = self._reference(reference_id="a", raw_path=str(logo))
        ref_b = self._reference(reference_id="b", raw_path=str(logo))
        result = oar.resolve_overlay_assets([ref_a, ref_b], config, inspector=FakeInspector())
        self.assertEqual(len(result.assets), 2)

    def test_conflicting_alpha_requirement_fails(self):
        logo = self._png()
        ref_a = self._reference(reference_id="a", raw_path=str(logo), requires_alpha=True)
        ref_b = self._reference(reference_id="b", raw_path=str(logo), requires_alpha=False)
        with self.assertRaises(oar.OverlayAssetConflictError):
            oar.resolve_overlay_assets([ref_a, ref_b], self.config, inspector=FakeInspector())

    def test_conflicting_requirements_ignored_when_configured(self):
        logo = self._png()
        config = dataclasses.replace(self.config, duplicate_conflicting_requirements="ignore")
        ref_a = self._reference(reference_id="a", raw_path=str(logo), requires_alpha=True)
        ref_b = self._reference(reference_id="b", raw_path=str(logo), requires_alpha=False)
        result = oar.resolve_overlay_assets([ref_a, ref_b], config, inspector=FakeInspector())
        self.assertEqual(len(result.assets), 1)

    def test_no_reference_silently_dropped_on_error(self):
        missing = self._reference(reference_id="missing_ref", raw_path=str(self.temp_dir / "missing.png"))
        present = self._reference(reference_id="present_ref", raw_path=str(self._png()))
        result = oar.resolve_overlay_assets([missing, present], self.config, inspector=FakeInspector())
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(result.assets), 1)

    def test_deterministic_ordering(self):
        logo_a = self._png("a.png")
        logo_b = self._png("b.png")
        refs = [self._reference(reference_id="a", raw_path=str(logo_a)), self._reference(reference_id="b", raw_path=str(logo_b))]
        result_1 = oar.resolve_overlay_assets(refs, self.config, inspector=FakeInspector())
        result_2 = oar.resolve_overlay_assets(refs, self.config, inspector=FakeInspector())
        self.assertEqual([a.asset_id for a in result_1.assets], [a.asset_id for a in result_2.assets])


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityTests(OverlayAssetResolverTempTestCase):
    def test_stable_asset_id(self):
        png = self._png()
        reference = self._reference(raw_path=str(png))
        asset_a = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        asset_b = oar.resolve_overlay_asset(reference, self.config, inspector=FakeInspector())
        self.assertEqual(asset_a.asset_id, asset_b.asset_id)

    def test_stable_manifest_id(self):
        png = self._png()
        refs = [self._reference(raw_path=str(png))]
        result_a = oar.resolve_overlay_assets(refs, self.config, inspector=FakeInspector())
        result_b = oar.resolve_overlay_assets(refs, self.config, inspector=FakeInspector())
        self.assertEqual(result_a.manifest_id, result_b.manifest_id)

    def test_timestamp_excluded_from_manifest_id(self):
        png = self._png()
        refs = [self._reference(raw_path=str(png))]
        result = oar.resolve_overlay_assets(refs, self.config, inspector=FakeInspector())
        manifest_a = oar.build_overlay_asset_manifest(result, refs, self.config, source_plan_ids=["p1"])
        manifest_b = dataclasses.replace(manifest_a, created_at="2000-01-01T00:00:00+00:00")
        self.assertEqual(manifest_a.manifest_id, manifest_b.manifest_id)

    def test_role_changes_asset_id(self):
        png = self._png()
        ref_logo = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.LOGO)
        ref_watermark = self._reference(raw_path=str(png), logical_role=oar.OverlayLogicalRole.WATERMARK)
        asset_logo = oar.resolve_overlay_asset(ref_logo, self.config, inspector=FakeInspector())
        asset_watermark = oar.resolve_overlay_asset(ref_watermark, self.config, inspector=FakeInspector())
        self.assertNotEqual(asset_logo.asset_id, asset_watermark.asset_id)

    def test_checksum_changes_asset_id(self):
        png_a = self._png("a.png", content=b"content-a")
        png_b = self._png("b.png", content=b"content-b")
        asset_a = oar.resolve_overlay_asset(self._reference(raw_path=str(png_a)), self.config, inspector=FakeInspector())
        asset_b = oar.resolve_overlay_asset(self._reference(raw_path=str(png_b)), self.config, inspector=FakeInspector())
        self.assertNotEqual(asset_a.asset_id, asset_b.asset_id)

    def test_source_overlay_changes_asset_id_when_configured(self):
        png = self._png()
        config = dataclasses.replace(self.config, duplicate_include_source_overlay_in_asset_id=True)
        ref_a = self._reference(raw_path=str(png), source_overlay_id="ov_a")
        ref_b = self._reference(raw_path=str(png), source_overlay_id="ov_b")
        asset_a = oar.resolve_overlay_asset(ref_a, config, inspector=FakeInspector())
        asset_b = oar.resolve_overlay_asset(ref_b, config, inspector=FakeInspector())
        self.assertNotEqual(asset_a.asset_id, asset_b.asset_id)

    def test_source_overlay_does_not_change_asset_id_by_default(self):
        png = self._png()
        ref_a = self._reference(raw_path=str(png), source_overlay_id="ov_a")
        ref_b = self._reference(raw_path=str(png), source_overlay_id="ov_b")
        asset_a = oar.resolve_overlay_asset(ref_a, self.config, inspector=FakeInspector())
        asset_b = oar.resolve_overlay_asset(ref_b, self.config, inspector=FakeInspector())
        self.assertEqual(asset_a.asset_id, asset_b.asset_id)

    def test_order_independent_collection_produces_deterministic_output(self):
        png_a = self._png("a.png")
        png_b = self._png("b.png")
        ref_a = self._reference(reference_id="ra", raw_path=str(png_a))
        ref_b = self._reference(reference_id="rb", raw_path=str(png_b))
        result_forward = oar.resolve_overlay_assets([ref_a, ref_b], self.config, inspector=FakeInspector())
        result_reverse = oar.resolve_overlay_assets([ref_b, ref_a], self.config, inspector=FakeInspector())
        self.assertEqual(
            sorted(a.asset_id for a in result_forward.assets),
            sorted(a.asset_id for a in result_reverse.assets),
        )

    def test_asset_id_is_16_hex_chars(self):
        png = self._png()
        asset = oar.resolve_overlay_asset(self._reference(raw_path=str(png)), self.config, inspector=FakeInspector())
        self.assertEqual(len(asset.asset_id), 16)
        int(asset.asset_id, 16)

    def test_manifest_id_is_16_hex_chars(self):
        png = self._png()
        result = oar.resolve_overlay_assets([self._reference(raw_path=str(png))], self.config, inspector=FakeInspector())
        self.assertEqual(len(result.manifest_id), 16)
        int(result.manifest_id, 16)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class ManifestTests(OverlayAssetResolverTempTestCase):
    def _manifest(self):
        png = self._png()
        refs = [self._reference(raw_path=str(png))]
        result = oar.resolve_overlay_assets(refs, self.config, inspector=FakeInspector())
        return oar.build_overlay_asset_manifest(result, refs, self.config, source_plan_ids=["p1"])

    def test_round_trip(self):
        manifest = self._manifest()
        data = oar.overlay_asset_manifest_to_dict(manifest)
        restored = oar.overlay_asset_manifest_from_dict(data)
        self.assertEqual(restored.manifest_id, manifest.manifest_id)
        self.assertEqual(len(restored.assets), len(manifest.assets))

    def test_atomic_write(self):
        manifest = self._manifest()
        path = self.temp_dir / "manifest.json"
        oar.save_overlay_asset_manifest(manifest, path)
        self.assertTrue(path.exists())
        self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_overwrite_refused(self):
        manifest = self._manifest()
        path = self.temp_dir / "manifest.json"
        oar.save_overlay_asset_manifest(manifest, path)
        with self.assertRaises(oar.OverlayAssetManifestExistsError):
            oar.save_overlay_asset_manifest(manifest, path)

    def test_force_overwrites(self):
        manifest = self._manifest()
        path = self.temp_dir / "manifest.json"
        oar.save_overlay_asset_manifest(manifest, path)
        oar.save_overlay_asset_manifest(manifest, path, force=True)

    def test_parent_created(self):
        manifest = self._manifest()
        path = self.temp_dir / "nested" / "dir" / "manifest.json"
        oar.save_overlay_asset_manifest(manifest, path)
        self.assertTrue(path.exists())

    def test_malformed_json_fails(self):
        path = self.temp_dir / "bad.json"
        path.write_text("{not valid", encoding="utf-8")
        with self.assertRaises(oar.OverlayAssetManifestJSONError):
            oar.load_overlay_asset_manifest(path)

    def test_missing_manifest_fails(self):
        with self.assertRaises(oar.OverlayAssetManifestJSONError):
            oar.load_overlay_asset_manifest(self.temp_dir / "missing.json")

    def test_non_object_root_fails(self):
        path = self.temp_dir / "list.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaises(oar.OverlayAssetManifestJSONError):
            oar.load_overlay_asset_manifest(path)

    def test_unknown_metadata_preserved(self):
        manifest = self._manifest()
        data = oar.overlay_asset_manifest_to_dict(manifest)
        data["metadata"]["future_field"] = "kept"
        restored = oar.overlay_asset_manifest_from_dict(data)
        self.assertEqual(restored.metadata.get("future_field"), "kept")

    def test_counts_correct(self):
        manifest = self._manifest()
        self.assertEqual(manifest.asset_count, 1)
        self.assertEqual(manifest.resolved_count, 1)
        self.assertEqual(manifest.missing_count, 0)
        self.assertEqual(manifest.invalid_count, 0)

    def test_source_plan_ids_included(self):
        manifest = self._manifest()
        self.assertEqual(manifest.source_plan_ids, ["p1"])

    def test_source_plan_hashes_unchanged_after_manifest_write(self):
        import hashlib

        logo = self._png()
        plan = self._overlay_plan_with("logo", str(logo))
        overlay_plan_path = self.temp_dir / "overlay_plan.json"
        overlay_plan_engine.save_overlay_plan(plan, overlay_plan_path)
        before = hashlib.sha256(overlay_plan_path.read_bytes()).hexdigest()

        refs = oar.collect_overlay_asset_references(overlay_plan=plan, config=self.config)
        result = oar.resolve_overlay_assets(refs, self.config, inspector=FakeInspector())
        manifest = oar.build_overlay_asset_manifest(result, refs, self.config, source_plan_ids=[plan.plan_id])
        oar.save_overlay_asset_manifest(manifest, self.temp_dir / "manifest.json")

        after = hashlib.sha256(overlay_plan_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_to_dict_json_serializable(self):
        manifest = self._manifest()
        json.dumps(oar.overlay_asset_manifest_to_dict(manifest))

    def test_overlay_render_asset_to_from_dict(self):
        manifest = self._manifest()
        asset = manifest.assets[0]
        data = oar.overlay_render_asset_to_dict(asset)
        restored = oar.overlay_render_asset_from_dict(data)
        self.assertEqual(restored.asset_id, asset.asset_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(OverlayAssetResolverTempTestCase):
    def _save_overlay_plan(self, overlay_type="cta") -> Path:
        # "cta" does not require alpha by default, keeping these CLI
        # tests independent of real image decoding (see _no_inspector_config).
        logo = self._png()
        plan = self._overlay_plan_with(overlay_type, str(logo))
        path = self.temp_dir / "overlay_plan.json"
        overlay_plan_engine.save_overlay_plan(plan, path, force=True)
        return path

    def _no_inspector_config(self) -> Path:
        # main() has no FakeInspector injection point (by design -- the
        # CLI always uses the real default backend). These CLI tests
        # must not require Pillow to be able to decode a real image, so
        # they point --config at a minimal override that disables the
        # inspector requirement entirely, exercising the CLI's real
        # load -> collect -> resolve -> manifest path end to end.
        path = self.temp_dir / "no_inspector_config.yaml"
        path.write_text("inspection:\n  require_inspector: false\n", encoding="utf-8")
        return path

    def test_no_source_fails(self):
        with self.assertRaises(SystemExit):
            oar.parse_arguments(["--output", "o.json"])

    def test_missing_output_without_validate_only_fails(self):
        with self.assertRaises(SystemExit):
            oar.parse_arguments(["--overlay-plan", "o.json"])

    def test_output_and_validate_only_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            oar.parse_arguments(["--overlay-plan", "o.json", "--output", "out.json", "--validate-only"])

    def test_overlay_plan_mode_accepted(self):
        args = oar.parse_arguments(["--overlay-plan", "o.json", "--output", "out.json"])
        self.assertEqual(args.overlay_plan, "o.json")

    def test_renderer_plan_mode_accepted(self):
        args = oar.parse_arguments(["--renderer-plan", "r.json", "--output", "out.json"])
        self.assertEqual(args.renderer_plan, "r.json")

    def test_overlay_plan_mode_works_end_to_end(self):
        overlay_plan_path = self._save_overlay_plan()
        output_path = self.temp_dir / "manifest.json"
        oar.main([
            "--overlay-plan", str(overlay_plan_path), "--output", str(output_path),
            "--config", str(self._no_inspector_config()),
        ])
        self.assertTrue(output_path.exists())

    def test_combined_mode_works(self):
        overlay_plan_path = self._save_overlay_plan()
        renderer_plan_config = renderer_plan_engine.load_renderer_plan_config()
        # Reuse the timeline behind the overlay plan for a real RendererPlan.
        timeline_path = self.temp_dir / "timeline_cta.json"
        renderer_plan = renderer_plan_engine.build_renderer_plan(timeline_path, overlay_plan_path, renderer_plan_config)
        renderer_plan_path = self.temp_dir / "renderer_plan.json"
        renderer_plan_engine.save_renderer_plan(renderer_plan, renderer_plan_path, force=True)
        output_path = self.temp_dir / "manifest_combined.json"
        oar.main([
            "--overlay-plan", str(overlay_plan_path), "--renderer-plan", str(renderer_plan_path),
            "--output", str(output_path), "--config", str(self._no_inspector_config()),
        ])
        self.assertTrue(output_path.exists())

    def test_validate_only_writes_nothing(self):
        overlay_plan_path = self._save_overlay_plan()
        output_path = self.temp_dir / "manifest.json"
        oar.main(["--overlay-plan", str(overlay_plan_path), "--validate-only", "--config", str(self._no_inspector_config())])
        self.assertFalse(output_path.exists())

    def test_json_output_valid(self):
        overlay_plan_path = self._save_overlay_plan()
        output_path = self.temp_dir / "manifest.json"
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            oar.main([
                "--overlay-plan", str(overlay_plan_path), "--output", str(output_path), "--json",
                "--config", str(self._no_inspector_config()),
            ])
        payload = json.loads(buffer.getvalue())
        self.assertIn("manifest_id", payload)

    def test_missing_plan_fails_no_traceback(self):
        import contextlib
        import io
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            oar.main(["--overlay-plan", str(self.temp_dir / "missing.json"), "--output", str(self.temp_dir / "o.json")])
        self.assertIn("[OverlayAssetResolver]", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())

    def test_malformed_plan_fails_no_traceback(self):
        bad_path = self.temp_dir / "bad_plan.json"
        bad_path.write_text("{not valid", encoding="utf-8")
        import contextlib
        import io
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            oar.main(["--overlay-plan", str(bad_path), "--output", str(self.temp_dir / "o.json")])
        self.assertNotIn("Traceback", buffer.getvalue())


# ---------------------------------------------------------------------------
# Structural safety
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def test_no_playwright_or_publish_reference(self):
        for forbidden in ("playwright", "Playwright", "instagram", "Instagram", "publish_reel", "upload_reel"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_publishing_or_social_import(self):
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from .publishing"))
            self.assertFalse(stripped.startswith("from .social"))

    def test_no_ffmpeg_reference(self):
        self.assertNotIn("ffmpeg_binary", MODULE_SOURCE)
        self.assertNotIn("subprocess.run(", MODULE_SOURCE)
        self.assertNotIn("import subprocess", MODULE_SOURCE)

    def test_no_direct_ffprobe_invocation(self):
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import ffprobe"))
        self.assertNotIn("ffprobe_binary", MODULE_SOURCE)
        self.assertNotIn("media_inspector", MODULE_SOURCE)

    def test_no_rendering_or_filter_graph_construction(self):
        for forbidden in ("filter_graph_builder", "filter_graph_serializer", "build_filter_graph", "FilterSpec"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_image_mutation_resize_convert(self):
        for forbidden in (".save(", ".resize(", ".thumbnail(", ".convert(", ".rotate(", ".crop("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_download_network(self):
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_automatic_asset_selection(self):
        for forbidden in ("pick_best_asset", "auto_select", "choose_replacement", "fallback_asset"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_package_install(self):
        for forbidden in ("pip install", "subprocess.call", "os.system("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_pillow_import_is_lazy(self):
        # No module-level (unindented) PIL import -- only a lazy,
        # function-scoped one inside default_image_inspector().
        for line in MODULE_SOURCE.splitlines():
            if line.startswith("import PIL") or line.startswith("from PIL"):
                self.fail(f"PIL must not be imported at module level: {line!r}")
        self.assertIn("from PIL import Image", MODULE_SOURCE)

    def test_no_video_timeline_import(self):
        for forbidden in ("timeline_engine", "video_engine", "subtitle_render_engine", "renderer_execution_engine"):
            for line in MODULE_SOURCE.splitlines():
                stripped = line.strip()
                if stripped.startswith("from . import") or stripped.startswith("from ."):
                    self.assertNotIn(forbidden, stripped, f"{forbidden} should not be imported by overlay_asset_resolver.py")

    def test_repo_config_has_no_ffmpeg_reference(self):
        content = (Path(__file__).resolve().parents[1] / "config" / "video" / "overlay_assets.yaml").read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn("ffmpeg", stripped)
            self.assertNotIn("ffprobe", stripped)


if __name__ == "__main__":
    unittest.main()
