from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from . import filter_graph_builder as fgb
from . import overlay_asset_resolver as oar
from . import overlay_plan_engine, renderer_plan_engine, subtitle_render_engine, timeline_engine

MODULE_PATH = Path(__file__).resolve().parent / "filter_graph_builder.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FilterGraphTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="filter_graph_test_"))
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.config = fgb.FilterGraphConfig()
        self.overlay_config = overlay_plan_engine.load_overlay_plan_config()
        self.renderer_plan_config = renderer_plan_engine.load_renderer_plan_config()

    def _video_clip(self, clip_id="clip_1", *, start=0.0, end=10.0, source_path=None, scene_number=1) -> timeline_engine.VideoClip:
        path = source_path or (self.temp_dir / "scene.mp4")
        if source_path is None and not path.exists():
            path.write_bytes(b"x" * 16)
        return timeline_engine.VideoClip(
            clip_id=clip_id, track_id="track_video", source_path=str(path),
            start=start, end=end, duration_seconds=end - start, source_out=end - start, scene_number=scene_number,
        )

    def _subtitle_clip(self, clip_id="sub_1", *, start=0.0, end=2.0, content="hello there", enabled=True) -> timeline_engine.OverlayClip:
        clip = timeline_engine.OverlayClip(
            clip_id=clip_id, track_id="track_subtitles", source_path="", overlay_type="subtitle",
            start=start, end=end, duration_seconds=end - start, source_out=end - start, content=content,
            metadata={"position": {"x": 540, "y": 1450}, "style": {"style_id": "default", "font_family": "Body"}, "language": "en"},
        )
        if not enabled:
            clip = dataclasses.replace(clip, enabled=False)
        return clip

    def _overlay_clip(self, clip_id="ov_1", *, start=0.0, end=3.0, opacity=1.0, z_index=None) -> timeline_engine.OverlayClip:
        metadata = {"position": {"x": 540, "y": 1400}, "opacity": opacity}
        if z_index is not None:
            metadata["z_index"] = z_index
        return timeline_engine.OverlayClip(
            clip_id=clip_id, track_id="track_overlay", source_path="", overlay_type="cta",
            start=start, end=end, duration_seconds=end - start, source_out=end - start, content="Shop now",
            metadata=metadata,
        )

    def _overlay_asset_manifest(
        self, overlay_plan, *, has_alpha=False, required=True, status=None, shared_path=None,
    ) -> oar.OverlayAssetManifest:
        """Builds a fixture OverlayAssetManifest (Phase 11F.5) with one
        fake-but-resolved OverlayRenderAsset per active, non-subtitle
        overlay in `overlay_plan` -- bypasses the real resolver
        entirely (no Pillow, no real image files) since this test only
        exercises filter_graph_builder's manifest-consumption logic."""
        assets = []
        for overlay in overlay_plan.overlays:
            if overlay.overlay_type == overlay_plan_engine.OverlayType.SUBTITLE or not overlay.enabled:
                continue
            resolved_path = str(shared_path) if shared_path else str(self.temp_dir / f"{overlay.overlay_id}.png")
            assets.append(
                oar.OverlayRenderAsset(
                    asset_id=f"asset_{overlay.overlay_id}",
                    reference_ids=[f"ref_{overlay.overlay_id}"],
                    logical_role=overlay.overlay_type,
                    asset_type=oar.OverlayAssetType.PNG,
                    source_path=resolved_path,
                    resolved_path=resolved_path,
                    filename=Path(resolved_path).name,
                    extension=".png",
                    file_size_bytes=100,
                    width=200,
                    height=100,
                    aspect_ratio=2.0,
                    has_alpha=has_alpha,
                    alpha_mode=oar.AlphaMode.STRAIGHT if has_alpha else oar.AlphaMode.NONE,
                    color_mode="RGBA" if has_alpha else "RGB",
                    animated=False,
                    checksum="deadbeefcafe",
                    source_plan_ids=[overlay_plan.plan_id],
                    source_overlay_ids=[overlay.overlay_id],
                    required=required,
                    status=status or oar.AssetResolutionStatus.RESOLVED,
                    warnings=[],
                    metadata={},
                )
            )
        return oar.OverlayAssetManifest(schema_version="1.0", manifest_id="manifest_test", assets=assets)

    def _build_timeline(
        self, *, with_subtitle=False, with_overlay=False, duration=10.0, subtitle_clips=None, overlay_clips=None,
    ) -> tuple[timeline_engine.Timeline, Path]:
        video_path = self.temp_dir / "scene.mp4"
        if not video_path.exists():
            video_path.write_bytes(b"x" * 16)
        tracks = [
            timeline_engine.TimelineTrack(
                track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0,
                clips=[self._video_clip(end=duration, source_path=video_path)],
            )
        ]
        if with_subtitle:
            tracks.append(
                timeline_engine.TimelineTrack(
                    track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=1,
                    clips=subtitle_clips or [self._subtitle_clip()],
                )
            )
        if with_overlay:
            tracks.append(
                timeline_engine.TimelineTrack(
                    track_id="track_overlay", track_type=timeline_engine.TrackType.OVERLAY, order=2,
                    clips=overlay_clips or [self._overlay_clip()],
                )
            )
        timeline = timeline_engine.Timeline(timeline_id="tl1", duration_seconds=duration, tracks=tracks)
        return timeline, video_path

    def _build_plans(
        self, *, with_subtitle=False, with_overlay=False, duration=10.0, subtitle_clips=None, overlay_clips=None,
        renderer_plan_config=None, video_paths=None,
    ) -> dict:
        timeline, video_path = self._build_timeline(
            with_subtitle=with_subtitle, with_overlay=with_overlay, duration=duration,
            subtitle_clips=subtitle_clips, overlay_clips=overlay_clips,
        )
        if video_paths:
            # Multi-clip override: replace the single video track's clips.
            clips = []
            cursor = 0.0
            for index, path in enumerate(video_paths):
                path.write_bytes(b"x" * 16) if not path.exists() else None
                clip_end = cursor + 5.0
                clips.append(self._video_clip(clip_id=f"c{index}", start=cursor, end=clip_end, source_path=path, scene_number=index + 1))
                cursor = clip_end
            timeline.tracks[0].clips = clips
            timeline.duration_seconds = cursor

        self._plan_counter = getattr(self, "_plan_counter", 0) + 1
        suffix = f"_{self._plan_counter}"

        timeline_path = self.temp_dir / f"timeline{suffix}.json"
        timeline_engine.save_timeline(timeline, timeline_path, force=True)

        overlay_plan = overlay_plan_engine.build_overlay_plan(timeline_path, self.overlay_config)
        overlay_plan_path = self.temp_dir / f"overlay_plan{suffix}.json"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path, force=True)

        effective_renderer_plan_config = renderer_plan_config or self.renderer_plan_config
        renderer_plan = renderer_plan_engine.build_renderer_plan(
            timeline_path, overlay_plan_path, effective_renderer_plan_config
        )
        renderer_plan_path = self.temp_dir / f"renderer_plan{suffix}.json"
        renderer_plan_engine.save_renderer_plan(renderer_plan, renderer_plan_path, force=True)

        return {
            "timeline": timeline, "timeline_path": timeline_path,
            "overlay_plan": overlay_plan, "overlay_plan_path": overlay_plan_path,
            "renderer_plan": renderer_plan, "renderer_plan_path": renderer_plan_path,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationTests(unittest.TestCase):
    def test_default_config_loads(self):
        config = fgb.load_filter_graph_config()
        self.assertEqual(config.video_label_prefix, "v")
        self.assertEqual(config.final_video_label, "outv")
        self.assertEqual(config.initial_input_label, "0:v")
        self.assertEqual(set(config.supported_filter_types), set(fgb.FilterType.ALL))

    def test_missing_config_raises(self):
        with self.assertRaises(fgb.FilterGraphConfigError):
            fgb.load_filter_graph_config("/nonexistent/filter_graph.yaml")

    def test_empty_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(fgb.FilterGraphConfigError):
                fgb.load_filter_graph_config(path)

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("filter_graph: [unclosed", encoding="utf-8")
            with self.assertRaises(fgb.FilterGraphConfigError):
                fgb.load_filter_graph_config(path)

    def test_custom_config_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.yaml"
            path.write_text(
                "labels:\n  video_label_prefix: 'f'\n  final_video_label: 'final'\n"
                "validation:\n  z_order_max: 50\n",
                encoding="utf-8",
            )
            config = fgb.load_filter_graph_config(path)
            self.assertEqual(config.video_label_prefix, "f")
            self.assertEqual(config.final_video_label, "final")
            self.assertEqual(config.z_order_max, 50)

    def test_default_config_path_resolves(self):
        path = fgb.default_filter_graph_config_path()
        self.assertTrue(str(path).endswith("config/video/filter_graph.yaml"))


# ---------------------------------------------------------------------------
# Label allocation
# ---------------------------------------------------------------------------


class LabelAllocationTests(unittest.TestCase):
    def test_sequential_labels(self):
        allocator = fgb._LabelAllocator("v")
        self.assertEqual(allocator.next(), "v0")
        self.assertEqual(allocator.next(), "v1")
        self.assertEqual(allocator.next(), "v2")

    def test_custom_prefix(self):
        allocator = fgb._LabelAllocator("f")
        self.assertEqual(allocator.next(), "f0")

    def test_ordered_dedup_preserves_first_occurrence_order(self):
        result = fgb._ordered_dedup(["a", "b", "a", "c", "b"])
        self.assertEqual(result, ["a", "b", "c"])

    def test_ordered_dedup_empty(self):
        self.assertEqual(fgb._ordered_dedup([]), [])


# ---------------------------------------------------------------------------
# Video pass derivation
# ---------------------------------------------------------------------------


class VideoPassDerivationTests(FilterGraphTempTestCase):
    def test_copy_hint_produces_no_filters(self):
        plans = self._build_plans()
        filters, label = fgb._derive_video_pass_filters(
            plans["renderer_plan"], renderer_plan_engine.RendererHintType.COPY, self.config, fgb._LabelAllocator("v")
        )
        self.assertEqual(filters, [])
        self.assertEqual(label, self.config.initial_input_label)

    def test_none_hint_produces_no_filters(self):
        plans = self._build_plans()
        filters, label = fgb._derive_video_pass_filters(
            plans["renderer_plan"], None, self.config, fgb._LabelAllocator("v")
        )
        self.assertEqual(filters, [])

    def test_normalize_hint_single_clip_produces_scale_filter(self):
        plans = self._build_plans()
        filters, label = fgb._derive_video_pass_filters(
            plans["renderer_plan"], renderer_plan_engine.RendererHintType.NORMALIZE, self.config, fgb._LabelAllocator("v")
        )
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0].filter_type, fgb.FilterType.SCALE)
        self.assertEqual(label, filters[0].label_out[0])

    def test_transcode_hint_produces_scale_filter(self):
        plans = self._build_plans()
        filters, label = fgb._derive_video_pass_filters(
            plans["renderer_plan"], renderer_plan_engine.RendererHintType.TRANSCODE, self.config, fgb._LabelAllocator("v")
        )
        self.assertEqual(len(filters), 1)

    def test_normalize_hint_multi_clip_adds_concat_filter(self):
        video_paths = [self.temp_dir / "s1.mp4", self.temp_dir / "s2.mp4"]
        plans = self._build_plans(video_paths=video_paths)
        filters, label = fgb._derive_video_pass_filters(
            plans["renderer_plan"], renderer_plan_engine.RendererHintType.NORMALIZE, self.config, fgb._LabelAllocator("v")
        )
        self.assertEqual(len(filters), 3)  # 2 scale + 1 concat
        self.assertEqual(filters[-1].filter_type, fgb.FilterType.CONCAT)
        self.assertEqual(filters[-1].parameters["n"], 2)
        self.assertEqual(label, filters[-1].label_out[0])

    def test_scale_filter_uses_canvas_dimensions(self):
        plans = self._build_plans()
        filters, _ = fgb._derive_video_pass_filters(
            plans["renderer_plan"], renderer_plan_engine.RendererHintType.NORMALIZE, self.config, fgb._LabelAllocator("v")
        )
        self.assertEqual(filters[0].parameters["width"], plans["renderer_plan"].canvas_width)
        self.assertEqual(filters[0].parameters["height"], plans["renderer_plan"].canvas_height)

    def test_scale_filter_label_in_is_raw_indexed_input(self):
        plans = self._build_plans()
        filters, _ = fgb._derive_video_pass_filters(
            plans["renderer_plan"], renderer_plan_engine.RendererHintType.NORMALIZE, self.config, fgb._LabelAllocator("v")
        )
        self.assertEqual(filters[0].label_in, ["0:v"])

    def test_no_video_tracks_produces_no_filters(self):
        empty_plan = renderer_plan_engine.RendererPlan()
        filters, label = fgb._derive_video_pass_filters(
            empty_plan, renderer_plan_engine.RendererHintType.NORMALIZE, self.config, fgb._LabelAllocator("v")
        )
        self.assertEqual(filters, [])


# ---------------------------------------------------------------------------
# Subtitle pass derivation
# ---------------------------------------------------------------------------


class SubtitlePassDerivationTests(FilterGraphTempTestCase):
    def test_drawtext_from_overlay_plan(self):
        plans = self._build_plans(with_subtitle=True)
        filters, label = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.DRAWTEXT,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0].filter_type, fgb.FilterType.DRAWTEXT)
        self.assertEqual(filters[0].parameters["text"], "hello there")
        self.assertEqual(label, filters[0].label_out[0])

    def test_drawtext_multiple_cues_chained(self):
        clips = [
            self._subtitle_clip(clip_id="s1", start=0.0, end=2.0, content="first"),
            self._subtitle_clip(clip_id="s2", start=2.0, end=4.0, content="second"),
        ]
        plans = self._build_plans(with_subtitle=True, subtitle_clips=clips)
        filters, label = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.DRAWTEXT,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        self.assertEqual(len(filters), 2)
        self.assertEqual(filters[0].parameters["text"], "first")
        self.assertEqual(filters[1].parameters["text"], "second")
        self.assertEqual(filters[1].label_in, filters[0].label_out)

    def test_disabled_overlay_excluded(self):
        clips = [
            self._subtitle_clip(clip_id="s1", start=0.0, end=2.0, content="visible", enabled=True),
            self._subtitle_clip(clip_id="s2", start=2.0, end=4.0, content="hidden", enabled=False),
        ]
        plans = self._build_plans(with_subtitle=True, subtitle_clips=clips)
        filters, _ = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.DRAWTEXT,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        texts = [f.parameters["text"] for f in filters]
        self.assertIn("visible", texts)
        self.assertNotIn("hidden", texts)

    def test_drawtext_enable_expression_matches_timing(self):
        plans = self._build_plans(with_subtitle=True)
        filters, _ = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.DRAWTEXT,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        self.assertEqual(filters[0].enable_expression, "between(t,0.0,2.0)")

    def test_ass_hint_no_asset_yields_none_ass_path(self):
        plans = self._build_plans(with_subtitle=True)
        filters, _ = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.ASS,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0].filter_type, fgb.FilterType.ASS)
        self.assertIsNone(filters[0].parameters["ass_path"])

    def test_ass_hint_with_subtitle_file_asset(self):
        plans = self._build_plans(with_subtitle=True)
        plan = plans["renderer_plan"]
        ass_asset = renderer_plan_engine.RendererAsset(
            asset_id="asset_subtitle_file_0001", asset_type=renderer_plan_engine.AssetType.SUBTITLE_FILE,
            source_path=str(self.temp_dir / "subs.ass"),
        )
        plan.assets.append(ass_asset)
        plan.subtitle_tracks[0].asset_ids.append(ass_asset.asset_id)
        filters, _ = fgb._derive_subtitle_pass_filters(
            plan, plans["overlay_plan"], renderer_plan_engine.RendererHintType.ASS,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        self.assertEqual(filters[0].parameters["ass_path"], str(self.temp_dir / "subs.ass"))

    def test_ass_hint_uses_injected_subtitle_request_when_provided(self):
        plans = self._build_plans(with_subtitle=True)
        video = self.temp_dir / "in.mp4"
        video.write_bytes(b"x" * 16)
        ass = self.temp_dir / "req.ass"
        ass.write_text("[Script Info]\n", encoding="utf-8")
        fontsdir = self.temp_dir / "fonts"
        fontsdir.mkdir()
        sre_config = subtitle_render_engine.SubtitleRenderConfig()
        request = subtitle_render_engine.build_subtitle_render_request(
            mode=subtitle_render_engine.SubtitleRenderMode.ASS, video_path=video,
            output_path=self.temp_dir / "out.mp4", ass_path=ass, fontsdir=fontsdir, config=sre_config,
        )
        filters, _ = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.ASS,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle",
            subtitle_requests={"pass_subtitle": request},
        )
        self.assertEqual(filters[0].parameters["ass_path"], str(ass))
        self.assertEqual(filters[0].parameters["fontsdir"], str(fontsdir))

    def test_drawtext_uses_injected_subtitle_request_cues(self):
        plans = self._build_plans(with_subtitle=True)
        video = self.temp_dir / "in.mp4"
        video.write_bytes(b"x" * 16)
        font = self.temp_dir / "font.ttf"
        font.write_bytes(b"fake")
        sre_config = subtitle_render_engine.SubtitleRenderConfig()
        cues = [subtitle_render_engine.DrawTextCue(text="from request", start_seconds=0.0, end_seconds=1.0, x=10, y=10, font_path=str(font))]
        request = subtitle_render_engine.build_subtitle_render_request(
            mode=subtitle_render_engine.SubtitleRenderMode.DRAWTEXT, video_path=video,
            output_path=self.temp_dir / "out.mp4", cues=cues, config=sre_config,
        )
        filters, _ = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.DRAWTEXT,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle",
            subtitle_requests={"pass_subtitle": request},
        )
        self.assertEqual(filters[0].parameters["text"], "from request")

    def test_no_subtitle_tracks_produces_no_filters(self):
        plans = self._build_plans()  # no subtitle
        filters, label = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.DRAWTEXT,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        self.assertEqual(filters, [])
        self.assertEqual(label, "0:v")

    def test_unsupported_hint_produces_no_filters(self):
        plans = self._build_plans(with_subtitle=True)
        filters, label = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "copy",
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        self.assertEqual(filters, [])
        self.assertEqual(label, "0:v")

    def test_semantic_params_not_escaped(self):
        clip = self._subtitle_clip(content="it's 50% off")
        plans = self._build_plans(with_subtitle=True, subtitle_clips=[clip])
        filters, _ = fgb._derive_subtitle_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], renderer_plan_engine.RendererHintType.DRAWTEXT,
            "0:v", fgb._LabelAllocator("v"), pass_id="pass_subtitle", subtitle_requests=None,
        )
        self.assertEqual(filters[0].parameters["text"], "it's 50% off")


# ---------------------------------------------------------------------------
# Overlay pass derivation
# ---------------------------------------------------------------------------


class OverlayPassDerivationTests(FilterGraphTempTestCase):
    def test_single_overlay_produces_overlay_filter(self):
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        filters, label, extra_inputs, skipped = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0].filter_type, fgb.FilterType.OVERLAY)
        self.assertEqual(label, filters[0].label_out[0])
        self.assertEqual(skipped, [])

    def test_overlay_filter_has_two_inputs(self):
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        filters, _, extra_inputs, _ = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        self.assertEqual(len(filters[0].label_in), 2)
        self.assertEqual(filters[0].label_in[0], "v0")
        self.assertEqual(len(extra_inputs), 1)

    def test_overlay_z_order_matches_z_index(self):
        plans = self._build_plans(with_overlay=True)
        overlay = plans["overlay_plan"].overlays[0]
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        filters, _, _, _ = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        self.assertEqual(filters[0].z_order, overlay.z_index)

    def test_multiple_overlays_ordered_by_z_index(self):
        # Non-overlapping times (Timeline Engine rejects overlapping
        # clips on the same track) and distinct z_index values so
        # sorting is meaningfully exercised.
        clips = [
            self._overlay_clip(clip_id="ov_a", start=0.0, end=3.0, z_index=5),
            self._overlay_clip(clip_id="ov_b", start=3.0, end=6.0, z_index=1),
        ]
        plans = self._build_plans(with_overlay=True, overlay_clips=clips)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        filters, _, _, _ = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        overlay_filters = [f for f in filters if f.filter_type == fgb.FilterType.OVERLAY]
        z_orders = [f.z_order for f in overlay_filters]
        self.assertEqual(z_orders, sorted(z_orders))

    def test_low_opacity_adds_alpha_filter(self):
        clip = self._overlay_clip(opacity=0.5)
        plans = self._build_plans(with_overlay=True, overlay_clips=[clip])
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        filters, _, _, _ = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        self.assertEqual(filters[0].filter_type, fgb.FilterType.ALPHA)
        self.assertEqual(filters[0].parameters["opacity"], 0.5)

    def test_full_opacity_skips_alpha_filter(self):
        clip = self._overlay_clip(opacity=1.0)
        plans = self._build_plans(with_overlay=True, overlay_clips=[clip])
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        filters, _, _, _ = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        self.assertTrue(all(f.filter_type != fgb.FilterType.ALPHA for f in filters))

    def test_no_overlay_tracks_produces_no_filters(self):
        plans = self._build_plans()  # no overlay
        filters, label, extra_inputs, skipped = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
        )
        self.assertEqual(filters, [])
        self.assertEqual(label, "v0")
        self.assertEqual(extra_inputs, [])
        self.assertEqual(skipped, [])

    def test_overlay_asset_label_format(self):
        # Phase 11F.5: overlay filters wire to a real, ffmpeg-consumable
        # numbered stream label backed by graph.extra_inputs -- never
        # the old synthetic "overlay_asset:{overlay_id}" placeholder.
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        filters, _, extra_inputs, _ = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        self.assertEqual(filters[0].label_in[1], "1:v")
        self.assertEqual(extra_inputs[0].label, "1:v")
        self.assertEqual(extra_inputs[0].resolved_path, manifest.assets[0].resolved_path)

    def test_manifest_required_when_overlays_active(self):
        plans = self._build_plans(with_overlay=True)
        with self.assertRaises(fgb.FilterGraphOverlayAssetManifestRequiredError):
            fgb._derive_overlay_pass_filters(
                plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            )

    def test_unresolved_required_asset_raises(self):
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"], status=oar.AssetResolutionStatus.MISSING)
        with self.assertRaises(fgb.FilterGraphOverlayAssetUnresolvedError):
            fgb._derive_overlay_pass_filters(
                plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
                overlay_asset_manifest=manifest,
            )

    def test_unresolved_optional_asset_skipped_not_raised(self):
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(
            plans["overlay_plan"], status=oar.AssetResolutionStatus.MISSING, required=False,
        )
        overlay_id = plans["overlay_plan"].overlays[0].overlay_id
        filters, label, extra_inputs, skipped = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        self.assertEqual(filters, [])
        self.assertEqual(label, "v0")
        self.assertEqual(extra_inputs, [])
        self.assertEqual(skipped, [overlay_id])

    def test_reused_asset_deduplicates_extra_input(self):
        shared = self.temp_dir / "shared_logo.png"
        clips = [
            self._overlay_clip(clip_id="ov_a", start=0.0, end=3.0),
            self._overlay_clip(clip_id="ov_b", start=3.0, end=6.0),
        ]
        plans = self._build_plans(with_overlay=True, overlay_clips=clips)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"], shared_path=shared)
        filters, _, extra_inputs, _ = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest,
        )
        overlay_filters = [f for f in filters if f.filter_type == fgb.FilterType.OVERLAY]
        self.assertEqual(len(overlay_filters), 2)
        self.assertEqual(len(extra_inputs), 1)
        self.assertEqual(overlay_filters[0].label_in[1], overlay_filters[1].label_in[1])

    def test_extra_input_start_index_respected(self):
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        filters, _, extra_inputs, _ = fgb._derive_overlay_pass_filters(
            plans["renderer_plan"], plans["overlay_plan"], "v0", fgb._LabelAllocator("v"),
            overlay_asset_manifest=manifest, extra_input_start_index=3,
        )
        self.assertEqual(extra_inputs[0].label, "3:v")
        self.assertEqual(filters[0].label_in[1], "3:v")


# ---------------------------------------------------------------------------
# Standalone build_subtitle_filter_spec()
# ---------------------------------------------------------------------------


class StandaloneSubtitleFilterSpecTests(FilterGraphTempTestCase):
    def _ass_request(self) -> subtitle_render_engine.SubtitleRenderRequest:
        video = self.temp_dir / "video.mp4"
        video.write_bytes(b"x" * 16)
        ass = self.temp_dir / "subs.ass"
        ass.write_text("[Script Info]\n", encoding="utf-8")
        sre_config = subtitle_render_engine.SubtitleRenderConfig()
        return subtitle_render_engine.build_subtitle_render_request(
            mode=subtitle_render_engine.SubtitleRenderMode.ASS, video_path=video,
            output_path=self.temp_dir / "out.mp4", ass_path=ass, config=sre_config,
        )

    def _drawtext_request(self, cue_count=2) -> subtitle_render_engine.SubtitleRenderRequest:
        video = self.temp_dir / "video.mp4"
        video.write_bytes(b"x" * 16)
        font = self.temp_dir / "font.ttf"
        font.write_bytes(b"fake")
        sre_config = subtitle_render_engine.SubtitleRenderConfig()
        cues = [
            subtitle_render_engine.DrawTextCue(text=f"cue {i}", start_seconds=float(i), end_seconds=float(i + 1), x=10, y=10, font_path=str(font))
            for i in range(cue_count)
        ]
        return subtitle_render_engine.build_subtitle_render_request(
            mode=subtitle_render_engine.SubtitleRenderMode.DRAWTEXT, video_path=video,
            output_path=self.temp_dir / "out.mp4", cues=cues, config=sre_config,
        )

    def test_ass_request_produces_one_filter(self):
        filters = fgb.build_subtitle_filter_spec(self._ass_request(), self.config)
        self.assertEqual(len(filters), 1)
        self.assertEqual(filters[0].filter_type, fgb.FilterType.ASS)

    def test_drawtext_request_produces_one_filter_per_cue(self):
        filters = fgb.build_subtitle_filter_spec(self._drawtext_request(cue_count=3), self.config)
        self.assertEqual(len(filters), 3)

    def test_drawtext_filters_chained(self):
        filters = fgb.build_subtitle_filter_spec(self._drawtext_request(cue_count=2), self.config)
        self.assertEqual(filters[1].label_in, filters[0].label_out)

    def test_drawtext_empty_cues_raises(self):
        video = self.temp_dir / "video.mp4"
        video.write_bytes(b"x" * 16)
        request = subtitle_render_engine.SubtitleRenderRequest(
            mode=subtitle_render_engine.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "o.mp4",
        )
        with self.assertRaises(fgb.FilterGraphStructureError):
            fgb.build_subtitle_filter_spec(request, self.config)

    def test_unsupported_mode_raises(self):
        video = self.temp_dir / "video.mp4"
        video.write_bytes(b"x" * 16)
        request = subtitle_render_engine.SubtitleRenderRequest(mode="karaoke", video_path=video, output_path=self.temp_dir / "o.mp4")
        with self.assertRaises(fgb.UnsupportedFilterTypeError):
            fgb.build_subtitle_filter_spec(request, self.config)

    def test_custom_label_in_respected(self):
        filters = fgb.build_subtitle_filter_spec(self._ass_request(), self.config, label_in="v5")
        self.assertEqual(filters[0].label_in, ["v5"])

    def test_default_label_in_is_initial_input(self):
        filters = fgb.build_subtitle_filter_spec(self._ass_request(), self.config)
        self.assertEqual(filters[0].label_in, [self.config.initial_input_label])

    def test_parameters_are_semantic_not_escaped(self):
        filters = fgb.build_subtitle_filter_spec(self._drawtext_request(cue_count=1), self.config)
        self.assertEqual(filters[0].parameters["text"], "cue 0")


# ---------------------------------------------------------------------------
# build_filter_graph() — full integration
# ---------------------------------------------------------------------------


class BuildFilterGraphTests(FilterGraphTempTestCase):
    def test_video_only_graph_has_no_filters(self):
        plans = self._build_plans()
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(graph.filters, [])
        self.assertEqual(graph.outputs, [self.config.initial_input_label])

    def test_subtitle_graph_has_drawtext_filter(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(len(graph.filters), 1)
        self.assertEqual(graph.filters[0].filter_type, fgb.FilterType.DRAWTEXT)

    def test_final_filter_output_renamed_to_outv(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(graph.outputs, ["outv"])
        self.assertEqual(graph.filters[-1].label_out, ["outv"])

    def test_graph_has_one_pass_per_renderer_pass(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(len(graph.passes), len(plans["renderer_plan"].passes))

    def test_graph_ids_populated(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(graph.renderer_plan_id, plans["renderer_plan"].renderer_plan_id)
        self.assertEqual(graph.overlay_plan_id, plans["overlay_plan"].plan_id)

    def test_graph_validation_populated_and_passed(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertTrue(graph.validation.passed)
        self.assertEqual(graph.validation.errors, [])

    def test_dependencies_mirror_renderer_plan(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        for pass_ in plans["renderer_plan"].passes:
            self.assertEqual(graph.dependencies[pass_.pass_id], pass_.dependencies)

    def test_metadata_includes_timeline_and_canvas(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(graph.metadata["timeline_id"], plans["renderer_plan"].timeline_id)
        self.assertEqual(graph.metadata["canvas_width"], plans["renderer_plan"].canvas_width)

    def test_mismatched_overlay_plan_id_raises(self):
        plans = self._build_plans(with_subtitle=True)
        other_overlay_plan = overlay_plan_engine.build_overlay_plan(plans["timeline_path"], self.overlay_config)
        other_overlay_plan.plan_id = "different"
        with self.assertRaises(fgb.FilterGraphPlanMismatchError):
            fgb.build_filter_graph(plans["renderer_plan"], other_overlay_plan, self.config)

    def test_video_and_subtitle_labels_chain(self):
        custom_config = dataclasses.replace(self.renderer_plan_config, video_hint="normalize")
        plans = self._build_plans(with_subtitle=True, renderer_plan_config=custom_config)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        scale_filter = next(f for f in graph.filters if f.filter_type == fgb.FilterType.SCALE)
        drawtext_filter = next(f for f in graph.filters if f.filter_type == fgb.FilterType.DRAWTEXT)
        self.assertEqual(drawtext_filter.label_in, scale_filter.label_out)

    def test_renderer_plan_and_overlay_plan_not_mutated(self):
        plans = self._build_plans(with_subtitle=True)
        before_passes = len(plans["renderer_plan"].passes)
        before_overlays = len(plans["overlay_plan"].overlays)
        fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(len(plans["renderer_plan"].passes), before_passes)
        self.assertEqual(len(plans["overlay_plan"].overlays), before_overlays)

    def test_labels_list_deterministic_order(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(graph.labels, fgb._ordered_dedup(graph.labels))

    def test_inputs_excludes_produced_labels(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        produced = {label for f in graph.filters for label in f.label_out}
        self.assertTrue(set(graph.inputs).isdisjoint(produced))

    def test_unknown_subtitle_hint_yields_no_subtitle_filters_but_still_valid_structure(self):
        custom_config = dataclasses.replace(self.renderer_plan_config, subtitle_hint="copy")
        plans = self._build_plans(with_subtitle=True, renderer_plan_config=custom_config)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        subtitle_pass = next(p for p in graph.passes if p.pass_type == "subtitle")
        self.assertEqual(subtitle_pass.filters, [])

    def test_overlay_pass_produces_overlay_filter_in_full_graph(self):
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        graph = fgb.build_filter_graph(
            plans["renderer_plan"], plans["overlay_plan"], self.config, overlay_asset_manifest=manifest,
        )
        overlay_filters = [f for f in graph.filters if f.filter_type == fgb.FilterType.OVERLAY]
        self.assertEqual(len(overlay_filters), 1)
        self.assertEqual(graph.outputs, ["outv"])
        self.assertEqual(len(graph.extra_inputs), 1)
        self.assertTrue(graph.validation.passed)

    def test_subtitle_and_overlay_chain_together(self):
        plans = self._build_plans(with_subtitle=True, with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        graph = fgb.build_filter_graph(
            plans["renderer_plan"], plans["overlay_plan"], self.config, overlay_asset_manifest=manifest,
        )
        drawtext_filter = next(f for f in graph.filters if f.filter_type == fgb.FilterType.DRAWTEXT)
        overlay_filter = next(f for f in graph.filters if f.filter_type == fgb.FilterType.OVERLAY)
        self.assertEqual(overlay_filter.label_in[0], drawtext_filter.label_out[0])
        self.assertEqual(overlay_filter.label_out, ["outv"])

    def test_overlay_pass_without_manifest_raises(self):
        plans = self._build_plans(with_overlay=True)
        with self.assertRaises(fgb.FilterGraphOverlayAssetManifestRequiredError):
            fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)

    def test_empty_graph_no_passes_at_all(self):
        empty_renderer_plan = renderer_plan_engine.RendererPlan(renderer_plan_id="rp_empty", overlay_plan_id="op_empty")
        empty_overlay_plan = overlay_plan_engine.OverlayPlan(plan_id="op_empty")
        graph = fgb.build_filter_graph(empty_renderer_plan, empty_overlay_plan, self.config)
        self.assertEqual(graph.passes, [])
        self.assertEqual(graph.filters, [])
        self.assertEqual(graph.outputs, [self.config.initial_input_label])
        self.assertTrue(graph.validation.passed)


# ---------------------------------------------------------------------------
# build_filter_graph_from_filters() (Phase 11F.3) — wraps an
# already-built filter list into a single-pass FilterGraph without a
# RendererPlan/OverlayPlan.
# ---------------------------------------------------------------------------


class BuildFilterGraphFromFiltersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = fgb.FilterGraphConfig()

    def test_single_filter_wraps_into_one_pass(self):
        spec = fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.ASS, label_in=["0:v"], label_out=["v0"], parameters={"ass_path": "/a.ass"})
        graph = fgb.build_filter_graph_from_filters([spec], pass_id="pass_subtitle", pass_type="subtitle", hint_type="ass", config=self.config)
        self.assertEqual(len(graph.passes), 1)
        self.assertEqual(graph.passes[0].pass_id, "pass_subtitle")
        self.assertEqual(graph.passes[0].pass_type, "subtitle")
        self.assertEqual(graph.passes[0].hint_type, "ass")

    def test_final_output_is_last_filter_output(self):
        spec1 = fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.DRAWTEXT, label_in=["0:v"], label_out=["v0"], parameters={"text": "a", "x": 1, "y": 1})
        spec2 = fgb.FilterSpec(filter_id="f2", filter_type=fgb.FilterType.DRAWTEXT, label_in=["v0"], label_out=["v1"], parameters={"text": "b", "x": 1, "y": 1})
        graph = fgb.build_filter_graph_from_filters([spec1, spec2], pass_id="p", pass_type="subtitle", config=self.config)
        self.assertEqual(graph.outputs, ["v1"])

    def test_empty_filters_uses_initial_input_label(self):
        graph = fgb.build_filter_graph_from_filters([], pass_id="p", pass_type="subtitle", config=self.config)
        self.assertEqual(graph.outputs, [self.config.initial_input_label])
        self.assertEqual(graph.passes[0].filters, [])

    def test_never_mutates_or_reorders_input_filters(self):
        spec1 = fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.SCALE, label_in=["0:v"], label_out=["v0"], parameters={"width": 10, "height": 20})
        spec2 = fgb.FilterSpec(filter_id="f2", filter_type=fgb.FilterType.FORMAT, label_in=["v0"], label_out=["v1"], parameters={})
        filters = [spec1, spec2]
        graph = fgb.build_filter_graph_from_filters(filters, pass_id="p", pass_type="video", config=self.config)
        self.assertEqual([f.filter_id for f in graph.filters], ["f1", "f2"])
        self.assertEqual(filters, [spec1, spec2])

    def test_graph_id_and_validation_populated(self):
        spec = fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.ASS, label_in=["0:v"], label_out=["v0"], parameters={"ass_path": "/a.ass"})
        graph = fgb.build_filter_graph_from_filters([spec], pass_id="p", pass_type="subtitle", config=self.config)
        self.assertTrue(graph.graph_id)
        self.assertTrue(graph.validation.passed)

    def test_optional_renderer_and_overlay_plan_ids(self):
        spec = fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.ASS, label_in=["0:v"], label_out=["v0"], parameters={"ass_path": "/a.ass"})
        graph = fgb.build_filter_graph_from_filters(
            [spec], pass_id="p", pass_type="subtitle", renderer_plan_id="rp1", overlay_plan_id="op1", config=self.config,
        )
        self.assertEqual(graph.renderer_plan_id, "rp1")
        self.assertEqual(graph.overlay_plan_id, "op1")

    def test_metadata_merged_with_builder_version(self):
        spec = fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.ASS, label_in=["0:v"], label_out=["v0"], parameters={"ass_path": "/a.ass"})
        graph = fgb.build_filter_graph_from_filters(
            [spec], pass_id="p", pass_type="subtitle", metadata={"canvas_width": 1080}, config=self.config,
        )
        self.assertEqual(graph.metadata["canvas_width"], 1080)
        self.assertIn("builder_version", graph.metadata)

    def test_dependencies_default_empty(self):
        spec = fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.ASS, label_in=["0:v"], label_out=["v0"], parameters={"ass_path": "/a.ass"})
        graph = fgb.build_filter_graph_from_filters([spec], pass_id="p", pass_type="subtitle", config=self.config)
        self.assertEqual(graph.dependencies, {"p": []})

    def test_does_not_affect_build_filter_graph(self):
        # Structural sanity: the additive function must not share
        # mutable state with build_filter_graph()'s own label allocator.
        spec = fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.ASS, label_in=["0:v"], label_out=["v0"], parameters={"ass_path": "/a.ass"})
        graph_a = fgb.build_filter_graph_from_filters([spec], pass_id="p", pass_type="subtitle", config=self.config)
        graph_b = fgb.build_filter_graph_from_filters([spec], pass_id="p", pass_type="subtitle", config=self.config)
        self.assertEqual(graph_a.graph_id, graph_b.graph_id)


# ---------------------------------------------------------------------------
# Deterministic graph_id
# ---------------------------------------------------------------------------


class GraphIdDeterminismTests(FilterGraphTempTestCase):
    def test_graph_id_stable_across_repeated_builds(self):
        plans = self._build_plans(with_subtitle=True)
        graph_a = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        graph_b = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(graph_a.graph_id, graph_b.graph_id)

    def test_graph_id_changes_with_subtitle_text(self):
        clip_a = self._subtitle_clip(content="Hello")
        clip_b = self._subtitle_clip(content="Goodbye")
        plans_a = self._build_plans(with_subtitle=True, subtitle_clips=[clip_a])
        plans_b = self._build_plans(with_subtitle=True, subtitle_clips=[clip_b])
        graph_a = fgb.build_filter_graph(plans_a["renderer_plan"], plans_a["overlay_plan"], self.config)
        graph_b = fgb.build_filter_graph(plans_b["renderer_plan"], plans_b["overlay_plan"], self.config)
        self.assertNotEqual(graph_a.graph_id, graph_b.graph_id)

    def test_graph_id_excludes_created_at(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        mutated = dataclasses.replace(graph, created_at="2000-01-01T00:00:00+00:00")
        self.assertEqual(fgb._compute_graph_id(graph), fgb._compute_graph_id(mutated))

    def test_graph_id_is_16_hex_chars(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        self.assertEqual(len(graph.graph_id), 16)
        int(graph.graph_id, 16)  # must not raise

    def test_graph_id_stable_across_repeated_builds_with_overlay_manifest(self):
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        graph_a = fgb.build_filter_graph(
            plans["renderer_plan"], plans["overlay_plan"], self.config, overlay_asset_manifest=manifest,
        )
        graph_b = fgb.build_filter_graph(
            plans["renderer_plan"], plans["overlay_plan"], self.config, overlay_asset_manifest=manifest,
        )
        self.assertEqual(graph_a.graph_id, graph_b.graph_id)

    def test_graph_id_changes_with_different_resolved_asset_path(self):
        plans = self._build_plans(with_overlay=True)
        manifest_a = self._overlay_asset_manifest(plans["overlay_plan"], shared_path=self.temp_dir / "logo_a.png")
        manifest_b = self._overlay_asset_manifest(plans["overlay_plan"], shared_path=self.temp_dir / "logo_b.png")
        graph_a = fgb.build_filter_graph(
            plans["renderer_plan"], plans["overlay_plan"], self.config, overlay_asset_manifest=manifest_a,
        )
        graph_b = fgb.build_filter_graph(
            plans["renderer_plan"], plans["overlay_plan"], self.config, overlay_asset_manifest=manifest_b,
        )
        self.assertNotEqual(graph_a.graph_id, graph_b.graph_id)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _base_graph() -> fgb.FilterGraph:
    filters = [
        fgb.FilterSpec(filter_id="f1", filter_type=fgb.FilterType.DRAWTEXT, label_in=["0:v"], label_out=["v0"], parameters={}, enable_expression="between(t,0,1)"),
    ]
    passes = [
        fgb.FilterGraphPass(pass_id="pass_video", pass_type="video", hint_type="copy", filters=[], dependencies=[]),
        fgb.FilterGraphPass(pass_id="pass_subtitle", pass_type="subtitle", hint_type="drawtext", filters=filters, dependencies=["pass_video"]),
    ]
    return fgb.FilterGraph(
        renderer_plan_id="rp1", overlay_plan_id="op1", passes=passes, filters=filters,
        labels=["0:v", "v0"], inputs=["0:v"], outputs=["v0"],
        dependencies={"pass_video": [], "pass_subtitle": ["pass_video"]},
    )


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = fgb.FilterGraphConfig()

    def test_valid_graph_passes(self):
        result = fgb.validate_filter_graph(_base_graph(), self.config)
        self.assertTrue(result.passed)
        self.assertEqual(result.errors, [])

    def test_duplicate_filter_id(self):
        graph = _base_graph()
        graph.filters.append(dataclasses.replace(graph.filters[0], label_out=["v1"]))
        graph.passes[1].filters.append(graph.filters[-1])
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("duplicate filter_id" in e for e in result.errors))

    def test_duplicate_produced_label(self):
        graph = _base_graph()
        duplicate = dataclasses.replace(graph.filters[0], filter_id="f2")
        graph.filters.append(duplicate)
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("duplicate produced label" in e for e in result.errors))

    def test_duplicate_declared_label(self):
        graph = _base_graph()
        graph.labels.append("v0")
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("graph.labels: duplicate" in e for e in result.errors))

    def test_missing_label(self):
        graph = _base_graph()
        graph.filters[0].label_in = ["never_produced"]
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("missing label" in e for e in result.errors))

    def test_label_in_from_declared_inputs_is_valid(self):
        graph = _base_graph()
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertTrue(result.passed)  # "0:v" is in graph.inputs

    def test_cycle_detected(self):
        graph = _base_graph()
        graph.passes[0].dependencies = ["pass_subtitle"]
        graph.dependencies["pass_video"] = ["pass_subtitle"]
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("cycle" in e for e in result.errors))

    def test_no_cycle_in_linear_chain(self):
        result = fgb.validate_filter_graph(_base_graph(), self.config)
        self.assertTrue(result.passed)

    def test_invalid_enable_expression_empty(self):
        graph = _base_graph()
        graph.filters[0].enable_expression = "   "
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("invalid enable expression" in e for e in result.errors))

    def test_invalid_enable_expression_unbalanced_parens(self):
        graph = _base_graph()
        graph.filters[0].enable_expression = "between(t,0,1"
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)

    def test_valid_enable_expression_none_is_fine(self):
        graph = _base_graph()
        graph.filters[0].enable_expression = None
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertTrue(result.passed)

    def test_unsupported_filter_type(self):
        graph = _base_graph()
        graph.filters[0].filter_type = "crop"
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("unsupported filter" in e for e in result.errors))

    def test_supported_filter_type_passes(self):
        graph = _base_graph()
        graph.filters[0].filter_type = fgb.FilterType.SCALE
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertTrue(result.passed)

    def test_unknown_renderer_hint(self):
        graph = _base_graph()
        graph.passes[1].hint_type = "karaoke"
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("unknown renderer hint" in e for e in result.errors))

    def test_unknown_hint_ignored_when_not_required(self):
        graph = _base_graph()
        graph.passes[1].hint_type = "karaoke"
        config = dataclasses.replace(self.config, require_known_renderer_hints=False)
        result = fgb.validate_filter_graph(graph, config)
        self.assertTrue(result.passed)

    def test_invalid_z_order_negative(self):
        graph = _base_graph()
        graph.filters[0].z_order = -1
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("invalid z-order" in e for e in result.errors))

    def test_invalid_z_order_non_integer(self):
        graph = _base_graph()
        graph.filters[0].z_order = 1.5
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)

    def test_invalid_z_order_above_max(self):
        graph = _base_graph()
        graph.filters[0].z_order = 99999
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)

    def test_valid_z_order_passes(self):
        graph = _base_graph()
        graph.filters[0].z_order = 10
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertTrue(result.passed)

    def test_broken_dependency_on_pass(self):
        graph = _base_graph()
        graph.passes[1].dependencies = ["nonexistent_pass"]
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("broken dependency" in e for e in result.errors))

    def test_broken_dependency_in_dependencies_dict_key(self):
        graph = _base_graph()
        graph.dependencies["nonexistent_pass"] = []
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)

    def test_broken_dependency_in_dependencies_dict_value(self):
        graph = _base_graph()
        graph.dependencies["pass_subtitle"] = ["nonexistent_pass"]
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)

    def test_unused_output_is_warning_not_error(self):
        graph = _base_graph()
        extra = fgb.FilterSpec(filter_id="f_extra", filter_type=fgb.FilterType.SCALE, label_in=["0:v"], label_out=["v_unused"], parameters={})
        graph.filters.append(extra)
        graph.passes[0].filters.append(extra)
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertTrue(result.passed)  # still passes -- warning only
        self.assertTrue(any("unused output" in w for w in result.warnings))

    def test_zero_final_outputs(self):
        graph = _base_graph()
        graph.outputs = []
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("multiple final outputs" in e for e in result.errors))

    def test_multiple_final_outputs(self):
        graph = _base_graph()
        graph.outputs = ["v0", "v1"]
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)

    def test_exactly_one_output_passes(self):
        graph = _base_graph()
        self.assertEqual(len(graph.outputs), 1)
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertTrue(result.passed)

    def test_validation_counts(self):
        graph = _base_graph()
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertEqual(result.filter_count, len(graph.filters))
        self.assertEqual(result.label_count, len(graph.labels))
        self.assertEqual(result.pass_count, len(graph.passes))

    def test_validation_summary_contains_graph_id(self):
        graph = _base_graph()
        graph.graph_id = "abc123"
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertIn("abc123", result.summary)

    def test_validate_filter_graph_idempotent(self):
        graph = _base_graph()
        result_a = fgb.validate_filter_graph(graph, self.config)
        result_b = fgb.validate_filter_graph(graph, self.config)
        self.assertEqual(result_a.passed, result_b.passed)
        self.assertEqual(result_a.errors, result_b.errors)

    # -- extra_inputs / overlay wiring (Phase 11F.5) --

    def _graph_with_overlay(self) -> fgb.FilterGraph:
        graph = _base_graph()
        overlay_filter = fgb.FilterSpec(
            filter_id="f_overlay", filter_type=fgb.FilterType.OVERLAY,
            label_in=["v0", "1:v"], label_out=["v_overlay"], parameters={"x": 0, "y": 0},
        )
        # _base_graph()'s pass_subtitle.filters IS graph.filters (same list
        # object) -- appending once already updates both; see
        # test_duplicate_filter_id for the same aliasing used deliberately.
        graph.filters.append(overlay_filter)
        graph.inputs.append("1:v")  # mirrors build_filter_graph()'s own inputs computation
        graph.labels.append("1:v")
        graph.outputs = ["v_overlay"]
        graph.extra_inputs = [fgb.ExtraInputSpec(label="1:v", resolved_path="/tmp/logo.png", asset_id="a1", logical_role="logo")]
        return graph

    def test_valid_graph_with_overlay_and_extra_input_passes(self):
        result = fgb.validate_filter_graph(self._graph_with_overlay(), self.config)
        self.assertTrue(result.passed)

    def test_duplicate_extra_input_label_errors(self):
        graph = self._graph_with_overlay()
        graph.extra_inputs.append(fgb.ExtraInputSpec(label="1:v", resolved_path="/tmp/other.png", asset_id="a2", logical_role="cta"))
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("graph.extra_inputs: duplicate label" in e for e in result.errors))

    def test_malformed_extra_input_label_errors(self):
        graph = self._graph_with_overlay()
        graph.extra_inputs[0].label = "not-a-stream-label"
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("invalid extra input label" in e for e in result.errors))

    def test_overlay_label_not_in_extra_inputs_errors(self):
        graph = self._graph_with_overlay()
        graph.extra_inputs = []
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("overlay image input not registered" in e for e in result.errors))

    def test_unused_extra_input_is_warning_not_error(self):
        graph = self._graph_with_overlay()
        graph.extra_inputs.append(fgb.ExtraInputSpec(label="2:v", resolved_path="/tmp/unused.png", asset_id="a3", logical_role="cta"))
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertTrue(result.passed)
        self.assertTrue(any("unused extra input" in w for w in result.warnings))

    def test_skipped_overlays_errors_when_require_all_resolved(self):
        graph = _base_graph()
        graph.metadata["skipped_overlays"] = ["ov_1"]
        result = fgb.validate_filter_graph(graph, self.config)
        self.assertFalse(result.passed)
        self.assertTrue(any("skipped_overlays" in e for e in result.errors))

    def test_skipped_overlays_tolerated_when_config_disabled(self):
        graph = _base_graph()
        graph.metadata["skipped_overlays"] = ["ov_1"]
        config = dataclasses.replace(self.config, overlay_require_all_resolved=False)
        result = fgb.validate_filter_graph(graph, config)
        self.assertTrue(result.passed)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class JSONSerializationTests(FilterGraphTempTestCase):
    def test_round_trip_preserves_content(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        data = fgb.filter_graph_to_dict(graph)
        restored = fgb.filter_graph_from_dict(data)
        self.assertEqual(restored.graph_id, graph.graph_id)
        self.assertEqual(len(restored.filters), len(graph.filters))
        self.assertEqual(restored.outputs, graph.outputs)

    def test_to_dict_is_json_serializable(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        json.dumps(fgb.filter_graph_to_dict(graph))  # must not raise

    def test_save_and_load(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        path = self.temp_dir / "graph.json"
        fgb.save_filter_graph(graph, path)
        loaded = fgb.load_filter_graph(path)
        self.assertEqual(loaded.graph_id, graph.graph_id)

    def test_save_atomic_no_tmp_left_behind(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        path = self.temp_dir / "graph.json"
        fgb.save_filter_graph(graph, path)
        self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_save_refuses_overwrite_without_force(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        path = self.temp_dir / "graph.json"
        fgb.save_filter_graph(graph, path)
        with self.assertRaises(fgb.FilterGraphOutputExistsError):
            fgb.save_filter_graph(graph, path)

    def test_save_overwrite_with_force(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        path = self.temp_dir / "graph.json"
        fgb.save_filter_graph(graph, path)
        fgb.save_filter_graph(graph, path, force=True)  # must not raise

    def test_save_rejects_directory_output(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        with self.assertRaises(fgb.UnsafeFilterGraphOutputError):
            fgb.save_filter_graph(graph, self.temp_dir)

    def test_load_missing_file_raises(self):
        with self.assertRaises(fgb.FilterGraphJSONError):
            fgb.load_filter_graph(self.temp_dir / "missing.json")

    def test_load_invalid_json_raises(self):
        path = self.temp_dir / "bad.json"
        path.write_text("{not valid", encoding="utf-8")
        with self.assertRaises(fgb.FilterGraphJSONError):
            fgb.load_filter_graph(path)

    def test_load_non_object_root_raises(self):
        path = self.temp_dir / "list.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaises(fgb.FilterGraphJSONError):
            fgb.load_filter_graph(path)

    def test_validation_field_round_trips(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        data = fgb.filter_graph_to_dict(graph)
        restored = fgb.filter_graph_from_dict(data)
        self.assertEqual(restored.validation.passed, graph.validation.passed)
        self.assertEqual(restored.validation.summary, graph.validation.summary)

    def test_dependencies_round_trip(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        data = fgb.filter_graph_to_dict(graph)
        restored = fgb.filter_graph_from_dict(data)
        self.assertEqual(restored.dependencies, graph.dependencies)

    def test_filter_graph_from_dict_non_dict_pass_entry_raises_cleanly(self):
        # A pass entry that isn't an object at all (e.g. a bare string)
        # must surface as FilterGraphJSONError, never a raw AttributeError.
        with self.assertRaises(fgb.FilterGraphJSONError):
            fgb.filter_graph_from_dict({"passes": ["not-an-object"]})

    def test_filter_graph_from_dict_non_dict_filter_entry_raises_cleanly(self):
        with self.assertRaises(fgb.FilterGraphJSONError):
            fgb.filter_graph_from_dict({"filters": ["not-an-object"]})

    def test_filter_graph_from_dict_missing_passes_defaults_empty(self):
        graph = fgb.filter_graph_from_dict({"renderer_plan_id": "rp1"})
        self.assertEqual(graph.passes, [])
        self.assertEqual(graph.renderer_plan_id, "rp1")

    def test_unknown_extra_fields_ignored_on_load(self):
        plans = self._build_plans(with_subtitle=True)
        graph = fgb.build_filter_graph(plans["renderer_plan"], plans["overlay_plan"], self.config)
        data = fgb.filter_graph_to_dict(graph)
        data["some_future_field"] = "ignored"
        restored = fgb.filter_graph_from_dict(data)
        self.assertEqual(restored.graph_id, graph.graph_id)

    def test_extra_inputs_round_trip(self):
        plans = self._build_plans(with_overlay=True)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        graph = fgb.build_filter_graph(
            plans["renderer_plan"], plans["overlay_plan"], self.config, overlay_asset_manifest=manifest,
        )
        data = fgb.filter_graph_to_dict(graph)
        restored = fgb.filter_graph_from_dict(data)
        self.assertEqual(len(restored.extra_inputs), len(graph.extra_inputs))
        self.assertEqual(restored.extra_inputs[0].label, graph.extra_inputs[0].label)
        self.assertEqual(restored.extra_inputs[0].resolved_path, graph.extra_inputs[0].resolved_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(FilterGraphTempTestCase):
    def test_validate_only_and_output_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            fgb.parse_arguments(["--renderer-plan", "r.json", "--overlay-plan", "o.json", "--output", "out.json", "--validate-only"])

    def test_output_required_without_validate_only(self):
        with self.assertRaises(SystemExit):
            fgb.parse_arguments(["--renderer-plan", "r.json", "--overlay-plan", "o.json"])

    def test_validate_only_accepted_without_output(self):
        args = fgb.parse_arguments(["--renderer-plan", "r.json", "--overlay-plan", "o.json", "--validate-only"])
        self.assertTrue(args.validate_only)

    def test_output_accepted(self):
        args = fgb.parse_arguments(["--renderer-plan", "r.json", "--overlay-plan", "o.json", "--output", "out.json"])
        self.assertEqual(args.output, "out.json")

    def test_force_flag(self):
        args = fgb.parse_arguments(["--renderer-plan", "r.json", "--overlay-plan", "o.json", "--output", "out.json", "--force"])
        self.assertTrue(args.force)

    def test_cli_build_writes_output(self):
        plans = self._build_plans(with_subtitle=True)
        output_path = self.temp_dir / "graph.json"
        fgb.main([
            "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
            "--output", str(output_path),
        ])
        self.assertTrue(output_path.is_file())

    def test_cli_validate_only_writes_nothing(self):
        plans = self._build_plans(with_subtitle=True)
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            fgb.main([
                "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
                "--validate-only",
            ])
        self.assertFalse((self.temp_dir / "graph.json").exists())

    def test_cli_missing_renderer_plan_fails_cleanly(self):
        plans = self._build_plans(with_subtitle=True)
        import contextlib
        import io
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            fgb.main([
                "--renderer-plan", str(self.temp_dir / "missing.json"), "--overlay-plan", str(plans["overlay_plan_path"]),
                "--output", str(self.temp_dir / "graph.json"),
            ])
        self.assertIn("[FilterGraphBuilder]", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())

    def test_cli_json_output(self):
        plans = self._build_plans(with_subtitle=True)
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            fgb.main([
                "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
                "--output", str(self.temp_dir / "graph.json"), "--json",
            ])
        payload = json.loads(buffer.getvalue())
        self.assertIn("graph_id", payload)

    def test_cli_output_same_as_renderer_plan_rejected(self):
        plans = self._build_plans(with_subtitle=True)
        import contextlib
        import io
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            fgb.main([
                "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
                "--output", str(plans["renderer_plan_path"]),
            ])

    def test_cli_overwrite_refused_without_force(self):
        plans = self._build_plans(with_subtitle=True)
        output_path = self.temp_dir / "graph.json"
        output_path.write_bytes(b"existing")
        import contextlib
        import io
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            fgb.main([
                "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
                "--output", str(output_path),
            ])

    def test_cli_custom_config_path_used(self):
        plans = self._build_plans(with_subtitle=True)
        config_path = self.temp_dir / "custom_filter_graph.yaml"
        config_path.write_text("labels:\n  final_video_label: 'custom_final'\n", encoding="utf-8")
        output_path = self.temp_dir / "graph.json"
        fgb.main([
            "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
            "--output", str(output_path), "--config", str(config_path),
        ])
        data = json.loads(output_path.read_text())
        self.assertEqual(data["outputs"], ["custom_final"])

    def test_cli_missing_overlay_plan_fails_cleanly(self):
        plans = self._build_plans(with_subtitle=True)
        import contextlib
        import io
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            fgb.main([
                "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(self.temp_dir / "missing.json"),
                "--output", str(self.temp_dir / "graph.json"),
            ])
        self.assertIn("[FilterGraphBuilder]", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())


# ---------------------------------------------------------------------------
# Structural safety
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def test_no_subprocess_import(self):
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import subprocess"))
            self.assertFalse(stripped.startswith("from subprocess"))
        self.assertNotIn("subprocess.run(", MODULE_SOURCE)
        self.assertNotIn("subprocess.Popen(", MODULE_SOURCE)

    def test_no_ffmpeg_ffprobe_binary_reference(self):
        self.assertNotIn("ffmpeg_binary", MODULE_SOURCE)
        self.assertNotIn("ffprobe_binary", MODULE_SOURCE)

    def test_no_forbidden_module_imports(self):
        for forbidden in ("video_engine", "renderer_execution_engine", "music_mixer", "media_inspector", "reel_builder", "timeline_engine"):
            for line in MODULE_SOURCE.splitlines():
                stripped = line.strip()
                if stripped.startswith("from . import") or stripped.startswith("from ."):
                    self.assertNotIn(forbidden, stripped, f"{forbidden} should not be imported by filter_graph_builder.py")

    def test_no_playwright_or_publish_reference(self):
        for forbidden in ("playwright", "Playwright", "instagram", "Instagram", "publish_reel", "upload_reel"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_media_probing(self):
        for forbidden in ("inspect_file(", "MediaInfo("):
            self.assertNotIn(forbidden, MODULE_SOURCE)
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import ffprobe"))
            self.assertNotIn("ffprobe(", stripped)

    def test_no_real_filter_string_escaping_helpers(self):
        # This module must not reimplement subtitle_render_engine.py's
        # ffmpeg-string escaping -- it stores semantic parameters only.
        self.assertNotIn("_escape_ffmpeg_quoted_value", MODULE_SOURCE)
        self.assertNotIn("shlex", MODULE_SOURCE)

    def test_no_fakerunner_reference(self):
        self.assertNotIn("FakeRunner", MODULE_SOURCE)
        self.assertNotIn("ProcessResult", MODULE_SOURCE)

    def test_build_never_raises_for_policy_only_structural(self):
        # build_filter_graph()'s only raise (besides the explicit
        # overlay/renderer plan_id mismatch pre-check) must be the
        # structural pass_type guard -- not a policy judgment.
        self.assertIn("FilterGraphStructureError", MODULE_SOURCE)

    def test_config_defaults_reasonable(self):
        config = fgb.FilterGraphConfig()
        self.assertEqual(config.z_order_min, 0)
        self.assertTrue(config.overwrite_requires_force)

    def test_repo_config_does_not_reference_ffmpeg(self):
        # No functional ffmpeg/ffprobe binary configuration -- explanatory
        # prose comments describing what this module never does are fine.
        content = (Path(__file__).resolve().parents[1] / "config" / "video" / "filter_graph.yaml").read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn("ffmpeg", stripped)
            self.assertNotIn("ffprobe", stripped)
        self.assertNotIn("binary:", content)

    def test_no_direct_video_engine_class_usage(self):
        self.assertNotIn("VideoEngine(", MODULE_SOURCE)
        self.assertNotIn("ConcatPlan", MODULE_SOURCE)


if __name__ == "__main__":
    unittest.main()
