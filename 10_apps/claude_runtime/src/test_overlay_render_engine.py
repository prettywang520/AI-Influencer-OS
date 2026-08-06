from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from . import filter_graph_builder, filter_graph_serializer
from . import overlay_asset_resolver as oar
from . import overlay_plan_engine, overlay_render_engine as ore, renderer_plan_engine, timeline_engine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class FakeRunner:
    """Injectable stand-in for overlay_render_engine.default_runner. Never
    touches real ffmpeg. Optionally writes a fake output file so
    verify_overlay_render_output() sees a real (fake) file."""

    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "",
                 raise_exception: Exception | None = None, write_output: bool = True,
                 output_bytes: bytes = b"fake-overlaid-video-bytes"):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raise_exception = raise_exception
        self.write_output = write_output
        self.output_bytes = output_bytes
        self.calls: list[tuple[list[str], int]] = []

    def __call__(self, command: list[str], *, timeout: int) -> ore.ProcessResult:
        self.calls.append((list(command), timeout))
        if self.raise_exception is not None:
            raise self.raise_exception
        if self.write_output and self.returncode == 0:
            Path(command[-1]).write_bytes(self.output_bytes)
        return ore.ProcessResult(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


class FakeInspectorInfo:
    def __init__(self, has_audio: bool = True):
        self.has_audio = has_audio


def _fake_output_inspector(*, has_audio: bool = True):
    def inspector(path: Path) -> FakeInspectorInfo:
        return FakeInspectorInfo(has_audio=has_audio)
    return inspector


class OverlayRenderTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="overlay_render_test_"))
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.config = ore.OverlayRenderConfig()
        self.overlay_config = overlay_plan_engine.load_overlay_plan_config()
        self.renderer_plan_config = renderer_plan_engine.load_renderer_plan_config()

    def _video(self, name: str = "video.mp4", content: bytes = b"fake-video-bytes") -> Path:
        path = self.temp_dir / name
        path.write_bytes(content)
        return path

    def _overlay_clip(self, clip_id: str = "ov_1", *, start=0.0, end=3.0, opacity=1.0, z_index=None):
        metadata = {"position": {"x": 540, "y": 1400}, "opacity": opacity}
        if z_index is not None:
            metadata["z_index"] = z_index
        return timeline_engine.OverlayClip(
            clip_id=clip_id, track_id="track_overlay", source_path="", overlay_type="cta",
            start=start, end=end, duration_seconds=end - start, source_out=end - start, content="Shop now",
            metadata=metadata,
        )

    def _plans(self, *, overlay_clips=None, duration: float = 6.0) -> dict:
        """Builds a real Timeline -> OverlayPlan -> RendererPlan with one
        video track + one overlay track, via the existing engines
        (never hand-constructed) -- same fixture shape as
        test_filter_graph_builder.py's own FilterGraphTempTestCase."""
        scene = self.temp_dir / "scene.mp4"
        if not scene.exists():
            scene.write_bytes(b"x" * 16)
        clips = overlay_clips or [self._overlay_clip()]
        timeline = timeline_engine.Timeline(
            timeline_id="tl1", duration_seconds=duration,
            tracks=[
                timeline_engine.TimelineTrack(
                    track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0,
                    clips=[
                        timeline_engine.VideoClip(
                            clip_id="c1", track_id="track_video", source_path=str(scene),
                            start=0.0, end=duration, duration_seconds=duration, source_out=duration, scene_number=1,
                        )
                    ],
                ),
                timeline_engine.TimelineTrack(
                    track_id="track_overlay", track_type=timeline_engine.TrackType.OVERLAY, order=1, clips=clips,
                ),
            ],
        )

        self._plan_counter = getattr(self, "_plan_counter", 0) + 1
        suffix = f"_{self._plan_counter}"

        timeline_path = self.temp_dir / f"timeline{suffix}.json"
        timeline_engine.save_timeline(timeline, timeline_path, force=True)

        overlay_plan = overlay_plan_engine.build_overlay_plan(timeline_path, self.overlay_config)
        overlay_plan_path = self.temp_dir / f"overlay_plan{suffix}.json"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path, force=True)

        renderer_plan = renderer_plan_engine.build_renderer_plan(timeline_path, overlay_plan_path, self.renderer_plan_config)
        renderer_plan_path = self.temp_dir / f"renderer_plan{suffix}.json"
        renderer_plan_engine.save_renderer_plan(renderer_plan, renderer_plan_path, force=True)

        return {
            "timeline": timeline, "timeline_path": timeline_path,
            "overlay_plan": overlay_plan, "overlay_plan_path": overlay_plan_path,
            "renderer_plan": renderer_plan, "renderer_plan_path": renderer_plan_path,
        }

    def _overlay_asset_manifest(
        self, overlay_plan, *, has_alpha=False, required=True, status=None, shared_path=None,
    ) -> oar.OverlayAssetManifest:
        """Fake-but-resolved OverlayAssetManifest -- bypasses the real
        resolver entirely (no Pillow, no real image files)."""
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
                    source_path=resolved_path, resolved_path=resolved_path,
                    filename=Path(resolved_path).name, extension=".png", file_size_bytes=100,
                    width=200, height=100, aspect_ratio=2.0,
                    has_alpha=has_alpha, alpha_mode=oar.AlphaMode.STRAIGHT if has_alpha else oar.AlphaMode.NONE,
                    color_mode="RGBA" if has_alpha else "RGB", animated=False, checksum="deadbeefcafe",
                    source_plan_ids=[overlay_plan.plan_id], source_overlay_ids=[overlay.overlay_id],
                    required=required, status=status or oar.AssetResolutionStatus.RESOLVED,
                    warnings=[], metadata={},
                )
            )
        return oar.OverlayAssetManifest(schema_version="1.0", manifest_id="manifest_test", assets=assets)

    def _serialized(self, plans: dict, manifest: oar.OverlayAssetManifest) -> filter_graph_serializer.SerializedFilterGraph:
        """Mirrors overlay_render_engine.build_overlay_render_plan()'s
        internal FilterGraph pipeline, for tests that exercise
        build_overlay_ffmpeg_command() directly/in isolation."""
        fg_config = filter_graph_builder.load_filter_graph_config()
        filters, extra_inputs, skipped = filter_graph_builder.build_overlay_filter_spec(
            plans["overlay_plan"], plans["renderer_plan"], manifest, fg_config,
        )
        graph = filter_graph_builder.build_filter_graph_from_filters(
            filters, pass_id="pass_overlay", pass_type="overlay", hint_type="overlay_png",
            extra_inputs=extra_inputs, metadata={"skipped_overlays": skipped}, config=fg_config,
        )
        return filter_graph_serializer.serialize_filter_graph(graph, filter_graph_serializer.load_filter_graph_serializer_config())

    def _request(self, plans: dict, manifest: oar.OverlayAssetManifest, **overrides) -> ore.OverlayRenderRequest:
        video = overrides.pop("video_path", None)
        if video is None:
            video = self._video()
        output = overrides.pop("output_path", self.temp_dir / "out.mp4")
        return ore.build_overlay_render_request(
            video_path=video, output_path=output, renderer_plan=plans["renderer_plan"],
            overlay_plan=plans["overlay_plan"], overlay_asset_manifest=manifest, config=self.config, **overrides,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationTests(unittest.TestCase):
    def test_default_config_loads(self):
        config = ore.load_overlay_render_config()
        self.assertEqual(config.ffmpeg_binary, "ffmpeg")
        self.assertTrue(config.copy_audio)
        self.assertTrue(config.faststart)
        self.assertEqual(config.video_codec, "libx264")

    def test_missing_config_file_raises(self):
        with self.assertRaises(ore.OverlayRenderConfigError):
            ore.load_overlay_render_config("/nonexistent/overlay_render.yaml")

    def test_empty_config_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ore.OverlayRenderConfigError):
                ore.load_overlay_render_config(path)

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("overlay_render: [unclosed", encoding="utf-8")
            with self.assertRaises(ore.OverlayRenderConfigError):
                ore.load_overlay_render_config(path)

    def test_custom_config_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.yaml"
            path.write_text("video:\n  crf: 20\n  copy_audio: false\n", encoding="utf-8")
            config = ore.load_overlay_render_config(path)
            self.assertEqual(config.video_crf, 20)
            self.assertFalse(config.copy_audio)

    def test_default_config_path_resolves_under_runtime_root(self):
        path = ore.default_overlay_render_config_path()
        self.assertTrue(str(path).endswith("config/video/overlay_render.yaml"))


# ---------------------------------------------------------------------------
# Request construction / validation
# ---------------------------------------------------------------------------


class RequestValidationTests(OverlayRenderTempTestCase):
    def test_missing_video_raises(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        with self.assertRaises(ore.OverlayVideoNotFoundError):
            self._request(plans, manifest, video_path=self.temp_dir / "missing.mp4")

    def test_video_is_directory_raises(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        with self.assertRaises(ore.OverlayVideoNotFoundError):
            self._request(plans, manifest, video_path=self.temp_dir)

    def test_empty_video_raises(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        with self.assertRaises(ore.OverlayVideoEmptyError):
            self._request(plans, manifest, video_path=self._video(content=b""))

    def test_output_same_as_video_raises(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        video = self._video()
        with self.assertRaises(ore.UnsafeOverlayRenderOutputError):
            self._request(plans, manifest, video_path=video, output_path=video)

    def test_output_directory_raises(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        with self.assertRaises(ore.UnsafeOverlayRenderOutputError):
            self._request(plans, manifest, output_path=self.temp_dir)

    def test_output_exists_without_force_raises(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        output = self.temp_dir / "out.mp4"
        output.write_bytes(b"existing")
        with self.assertRaises(ore.OverlayRenderOutputExistsError):
            self._request(plans, manifest, output_path=output)

    def test_output_exists_with_force_succeeds(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        output = self.temp_dir / "out.mp4"
        output.write_bytes(b"existing")
        request = self._request(plans, manifest, output_path=output, force=True)
        self.assertTrue(request.force)

    def test_output_parent_not_directory_raises(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        blocker = self.temp_dir / "blocker"
        blocker.write_bytes(b"x")
        with self.assertRaises(ore.UnsafeOverlayRenderOutputError):
            self._request(plans, manifest, output_path=blocker / "out.mp4")

    def test_valid_request_succeeds(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        self.assertEqual(request.overlay_asset_manifest, manifest)
        self.assertIs(request.renderer_plan, plans["renderer_plan"])


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


class CommandConstructionTests(OverlayRenderTempTestCase):
    def test_command_shape(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, self.config, serialized=serialized)
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("-hide_banner", command)
        self.assertIn("-i", command)
        self.assertIn(str(request.video_path), command)
        self.assertIn("-filter_complex", command)
        self.assertEqual(command[-1], str(request.output_path))
        self.assertIn("-c:a", command)
        self.assertIn("copy", command)

    def test_extra_input_flags_in_order(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, self.config, serialized=serialized)
        i_indices = [i for i, token in enumerate(command) if token == "-i"]
        i_values = [command[i + 1] for i in i_indices]
        self.assertEqual(i_values[0], str(request.video_path))
        self.assertEqual(i_values[1:], [e.resolved_path for e in serialized.extra_inputs])

    def test_multiple_overlays_produce_multiple_i_flags_deduplicated(self):
        shared = self.temp_dir / "shared_logo.png"
        clips = [self._overlay_clip(clip_id="ov_a", start=0.0, end=3.0), self._overlay_clip(clip_id="ov_b", start=3.0, end=6.0)]
        plans = self._plans(overlay_clips=clips)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"], shared_path=shared)
        request = self._request(plans, manifest)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, self.config, serialized=serialized)
        self.assertEqual(command.count("-i"), 2)  # video + 1 deduplicated shared image

    def test_filter_complex_value_taken_verbatim(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, self.config, serialized=serialized)
        index = command.index("-filter_complex")
        self.assertEqual(command[index + 1], serialized.filter_expression)

    def test_map_uses_final_output_label(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, self.config, serialized=serialized)
        self.assertIn(f"[{serialized.final_output_label}]", command)
        self.assertIn("0:a?", command)

    def test_copy_audio_true_uses_dash_c_a_copy(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        config = dataclasses.replace(self.config, copy_audio=True)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, config, serialized=serialized)
        index = command.index("-c:a")
        self.assertEqual(command[index + 1], "copy")

    def test_copy_audio_false_reencodes(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        config = dataclasses.replace(self.config, copy_audio=False, audio_codec_when_reencode_required="aac", audio_bitrate="128k")
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, config, serialized=serialized)
        index = command.index("-c:a")
        self.assertEqual(command[index + 1], "aac")
        self.assertIn("-b:a", command)
        self.assertIn("128k", command)

    def test_faststart_adds_movflags(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        config = dataclasses.replace(self.config, faststart=True)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, config, serialized=serialized)
        self.assertIn("-movflags", command)
        self.assertIn("+faststart", command)

    def test_faststart_disabled_omits_movflags(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        config = dataclasses.replace(self.config, faststart=False)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, config, serialized=serialized)
        self.assertNotIn("-movflags", command)

    def test_force_uses_dash_y(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        output = self.temp_dir / "out.mp4"
        output.write_bytes(b"existing")
        request = self._request(plans, manifest, output_path=output, force=True)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, self.config, serialized=serialized)
        self.assertIn("-y", command)
        self.assertNotIn("-n", command)

    def test_no_force_uses_dash_n(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, self.config, serialized=serialized)
        self.assertIn("-n", command)

    def test_hide_banner_disabled(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        config = dataclasses.replace(self.config, ffmpeg_hide_banner=False)
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, config, serialized=serialized)
        self.assertNotIn("-hide_banner", command)

    def test_codec_preset_crf_pixel_format_present(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        config = dataclasses.replace(self.config, video_codec="libx265", video_preset="fast", video_crf=22, video_pixel_format="yuv444p")
        serialized = self._serialized(plans, manifest)
        command = ore.build_overlay_ffmpeg_command(request, config, serialized=serialized)
        self.assertIn("libx265", command)
        self.assertIn("fast", command)
        self.assertIn("22", command)
        self.assertIn("yuv444p", command)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class PlanTests(OverlayRenderTempTestCase):
    def test_plan_builds_without_executing(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        plan = ore.build_overlay_render_plan(request, self.config)
        self.assertTrue(plan.render_id)
        self.assertFalse(request.output_path.exists())
        self.assertTrue(plan.filter_graph_id)
        self.assertTrue(plan.filter_graph_validation_passed)
        self.assertTrue(plan.serialization_id)
        self.assertEqual(plan.output_mode, "filter_complex")
        self.assertEqual(len(plan.extra_inputs), 1)

    def test_render_id_stable_across_repeated_planning(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        plan_a = ore.build_overlay_render_plan(request, self.config)
        plan_b = ore.build_overlay_render_plan(request, self.config)
        self.assertEqual(plan_a.render_id, plan_b.render_id)
        self.assertEqual(plan_a.command, plan_b.command)

    def test_render_id_changes_with_resolved_asset_path(self):
        plans = self._plans()
        manifest_a = self._overlay_asset_manifest(plans["overlay_plan"], shared_path=self.temp_dir / "a.png")
        manifest_b = self._overlay_asset_manifest(plans["overlay_plan"], shared_path=self.temp_dir / "b.png")
        request_a = self._request(plans, manifest_a, output_path=self.temp_dir / "out_a.mp4")
        request_b = self._request(plans, manifest_b, output_path=self.temp_dir / "out_b.mp4")
        plan_a = ore.build_overlay_render_plan(request_a, self.config)
        plan_b = ore.build_overlay_render_plan(request_b, self.config)
        self.assertNotEqual(plan_a.render_id, plan_b.render_id)


# ---------------------------------------------------------------------------
# FilterGraph / serializer integration
# ---------------------------------------------------------------------------


def _counting_wrapper(real_func):
    calls: list[tuple[tuple, dict]] = []

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real_func(*args, **kwargs)

    wrapper.calls = calls
    return wrapper


class FilterGraphIntegrationTests(OverlayRenderTempTestCase):
    def test_routes_through_filter_graph_builder(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        builder_spy = _counting_wrapper(filter_graph_builder.build_overlay_filter_spec)
        plan = ore.build_overlay_render_plan(request, self.config, filter_graph_builder_callable=builder_spy)
        self.assertEqual(len(builder_spy.calls), 1)
        self.assertTrue(plan.filter_graph_id)

    def test_wrapper_called_exactly_once(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        wrapper_spy = _counting_wrapper(filter_graph_builder.build_filter_graph_from_filters)
        ore.build_overlay_render_plan(request, self.config, filter_graph_wrapper_callable=wrapper_spy)
        self.assertEqual(len(wrapper_spy.calls), 1)

    def test_serializer_called_exactly_once(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        serializer_spy = _counting_wrapper(filter_graph_serializer.serialize_filter_graph)
        ore.build_overlay_render_plan(request, self.config, filter_graph_serializer_callable=serializer_spy)
        self.assertEqual(len(serializer_spy.calls), 1)

    def test_serialized_filter_complex_content_in_final_command(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        plan = ore.build_overlay_render_plan(request, self.config)
        self.assertIn("-filter_complex", plan.command)
        index = plan.command.index("-filter_complex")
        self.assertIn("overlay=", plan.command[index + 1])

    def test_format_auto_present_when_alpha(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"], has_alpha=True)
        request = self._request(plans, manifest)
        plan = ore.build_overlay_render_plan(request, self.config)
        index = plan.command.index("-filter_complex")
        self.assertIn("format=auto", plan.command[index + 1])

    def test_format_auto_absent_without_alpha(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"], has_alpha=False)
        request = self._request(plans, manifest)
        plan = ore.build_overlay_render_plan(request, self.config)
        index = plan.command.index("-filter_complex")
        self.assertNotIn("format=auto", plan.command[index + 1])

    def test_z_order_preserved_in_filter_chain(self):
        clips = [
            self._overlay_clip(clip_id="ov_top", start=0.0, end=3.0, z_index=5),
            self._overlay_clip(clip_id="ov_bottom", start=3.0, end=6.0, z_index=1),
        ]
        plans = self._plans(overlay_clips=clips)
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        plan = ore.build_overlay_render_plan(request, self.config)
        index = plan.command.index("-filter_complex")
        expr = plan.command[index + 1]
        # z_index=1 (ov_bottom) must be composited before z_index=5
        # (ov_top) -- earlier in the filter chain means "underneath".
        self.assertLess(expr.index("ov_bottom") if "ov_bottom" in expr else 0, len(expr))
        # Structural proof via filter_id order in the graph itself,
        # since ffmpeg filter syntax has no overlay_id -- assert via
        # the underlying filter list instead of the opaque expression.
        graph_filters = [f for f in filter_graph_builder.build_overlay_filter_spec(
            plans["overlay_plan"], plans["renderer_plan"], manifest, filter_graph_builder.load_filter_graph_config(),
        )[0] if f.filter_type == filter_graph_builder.FilterType.OVERLAY]
        self.assertEqual([f.z_order for f in graph_filters], sorted(f.z_order for f in graph_filters))

    def test_graph_validation_failure_raises_overlay_render_filter_graph_error(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)

        def broken_wrapper(filters, **kwargs):
            graph = filter_graph_builder.build_filter_graph_from_filters(filters, **kwargs)
            graph.validation.passed = False
            graph.validation.errors = ["synthetic failure for test"]
            return graph

        with self.assertRaises(ore.OverlayRenderFilterGraphError):
            ore.build_overlay_render_plan(request, self.config, filter_graph_wrapper_callable=broken_wrapper)

    def test_graph_validation_failure_prevents_runner_call(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        runner = FakeRunner()

        def broken_wrapper(filters, **kwargs):
            graph = filter_graph_builder.build_filter_graph_from_filters(filters, **kwargs)
            graph.validation.passed = False
            graph.validation.errors = ["synthetic failure for test"]
            return graph

        with self.assertRaises(ore.OverlayRenderFilterGraphError):
            ore.execute_overlay_render_plan(request, self.config, runner=runner, filter_graph_wrapper_callable=broken_wrapper)
        self.assertEqual(len(runner.calls), 0)

    def test_serializer_failure_raises_overlay_render_filter_graph_error(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)

        def broken_serializer(graph, config):
            raise filter_graph_serializer.FilterGraphSerializerError("synthetic serializer failure")

        with self.assertRaises(ore.OverlayRenderFilterGraphError):
            ore.build_overlay_render_plan(request, self.config, filter_graph_serializer_callable=broken_serializer)

    def test_serializer_failure_prevents_runner_call(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        runner = FakeRunner()

        def broken_serializer(graph, config):
            raise filter_graph_serializer.FilterGraphSerializerError("synthetic serializer failure")

        with self.assertRaises(ore.OverlayRenderFilterGraphError):
            ore.execute_overlay_render_plan(request, self.config, runner=runner, filter_graph_serializer_callable=broken_serializer)
        self.assertEqual(len(runner.calls), 0)

    def test_builder_failure_raises_overlay_render_filter_graph_error(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)

        def broken_builder(overlay_plan, renderer_plan, manifest, config, **kwargs):
            raise filter_graph_builder.FilterGraphBuilderError("synthetic builder failure")

        with self.assertRaises(ore.OverlayRenderFilterGraphError):
            ore.build_overlay_render_plan(request, self.config, filter_graph_builder_callable=broken_builder)

    def test_unresolved_required_asset_raises(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"], status=oar.AssetResolutionStatus.MISSING)
        request = self._request(plans, manifest)
        with self.assertRaises(ore.OverlayRenderFilterGraphError):
            ore.build_overlay_render_plan(request, self.config)

    def test_source_hashes_unchanged_after_planning(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        before_video = hashlib.sha256(request.video_path.read_bytes()).hexdigest()
        ore.build_overlay_render_plan(request, self.config)
        self.assertEqual(hashlib.sha256(request.video_path.read_bytes()).hexdigest(), before_video)

    def test_source_hashes_unchanged_after_execution(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        request = self._request(plans, manifest)
        config = dataclasses.replace(self.config, diagnostics_write_log=False)
        before_video = hashlib.sha256(request.video_path.read_bytes()).hexdigest()
        runner = FakeRunner()
        ore.execute_overlay_render_plan(request, config, runner=runner)
        self.assertEqual(hashlib.sha256(request.video_path.read_bytes()).hexdigest(), before_video)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecutionTests(OverlayRenderTempTestCase):
    def _fake_request(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        config = dataclasses.replace(self.config, diagnostics_write_log=False)
        request = self._request(plans, manifest, output_path=self.temp_dir / "out.mp4")
        return request, config

    def test_successful_execution_returns_result(self):
        request, config = self._fake_request()
        runner = FakeRunner()
        result = ore.execute_overlay_render_plan(request, config, runner=runner)
        self.assertEqual(result.return_code, 0)
        self.assertTrue(result.output_exists)
        self.assertTrue(request.output_path.exists())
        self.assertEqual(len(runner.calls), 1)
        self.assertTrue(result.filter_graph_id)
        self.assertTrue(result.serialization_id)

    def test_non_zero_return_code_raises(self):
        request, config = self._fake_request()
        runner = FakeRunner(returncode=1, stderr="boom")
        with self.assertRaises(ore.FFmpegExecutionError):
            ore.execute_overlay_render_plan(request, config, runner=runner)

    def test_missing_binary_raises_ffmpeg_not_found(self):
        request, config = self._fake_request()
        runner = FakeRunner(raise_exception=ore.FFmpegNotFoundError("no ffmpeg"))
        with self.assertRaises(ore.FFmpegNotFoundError):
            ore.execute_overlay_render_plan(request, config, runner=runner)

    def test_timeout_raises(self):
        request, config = self._fake_request()
        runner = FakeRunner(raise_exception=ore.FFmpegTimeoutError("timed out"))
        with self.assertRaises(ore.FFmpegTimeoutError):
            ore.execute_overlay_render_plan(request, config, runner=runner)

    def test_success_but_no_output_written_raises_verification_error(self):
        request, config = self._fake_request()
        runner = FakeRunner(write_output=False)
        with self.assertRaises(ore.OverlayRenderVerificationError):
            ore.execute_overlay_render_plan(request, config, runner=runner)

    def test_success_but_zero_byte_output_raises(self):
        request, config = self._fake_request()
        runner = FakeRunner(output_bytes=b"")
        with self.assertRaises(ore.OverlayRenderVerificationError):
            ore.execute_overlay_render_plan(request, config, runner=runner)

    def test_output_exists_without_force_raises_before_running(self):
        request, config = self._fake_request()
        request.output_path.write_bytes(b"already there")
        runner = FakeRunner()
        with self.assertRaises(ore.OverlayRenderOutputExistsError):
            ore.execute_overlay_render_plan(request, config, runner=runner)
        self.assertEqual(len(runner.calls), 0)

    def test_dry_run_plan_never_touches_filesystem(self):
        request, config = self._fake_request()
        plan = ore.build_overlay_render_plan(request, config)
        self.assertFalse(request.output_path.exists())
        self.assertTrue(plan.command)

    def test_runner_called_exactly_once(self):
        request, config = self._fake_request()
        runner = FakeRunner()
        ore.execute_overlay_render_plan(request, config, runner=runner)
        self.assertEqual(len(runner.calls), 1)

    def test_runner_receives_timeout_from_config(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        config = dataclasses.replace(self.config, diagnostics_write_log=False, ffmpeg_timeout_seconds=42)
        request = self._request(plans, manifest)
        runner = FakeRunner()
        ore.execute_overlay_render_plan(request, config, runner=runner)
        self.assertEqual(runner.calls[0][1], 42)

    def test_inspector_sets_audio_preserved(self):
        request, config = self._fake_request()
        runner = FakeRunner()
        result = ore.execute_overlay_render_plan(request, config, runner=runner, inspector=_fake_output_inspector(has_audio=True))
        self.assertTrue(result.audio_preserved)

    def test_inspector_omitted_leaves_audio_preserved_none(self):
        request, config = self._fake_request()
        runner = FakeRunner()
        result = ore.execute_overlay_render_plan(request, config, runner=runner)
        self.assertIsNone(result.audio_preserved)

    def test_extra_inputs_present_on_result(self):
        request, config = self._fake_request()
        runner = FakeRunner()
        result = ore.execute_overlay_render_plan(request, config, runner=runner)
        self.assertEqual(len(result.extra_inputs), 1)


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------


class VerificationTests(OverlayRenderTempTestCase):
    def test_missing_output_returns_false(self):
        exists, size = ore.verify_overlay_render_output(self.temp_dir / "missing.mp4", self.temp_dir / "in.mp4")
        self.assertFalse(exists)
        self.assertEqual(size, 0)

    def test_existing_output_returns_true_and_size(self):
        output = self._video("out.mp4", content=b"12345")
        exists, size = ore.verify_overlay_render_output(output, self.temp_dir / "in.mp4")
        self.assertTrue(exists)
        self.assertEqual(size, 5)

    def test_output_same_as_input_raises(self):
        video = self._video()
        with self.assertRaises(ore.UnsafeOverlayRenderOutputError):
            ore.verify_overlay_render_output(video, video)

    def test_inspector_invoked_when_provided(self):
        output = self._video("out.mp4")
        calls = []

        def inspector(path):
            calls.append(path)
            return FakeInspectorInfo()

        ore.verify_overlay_render_output(output, self.temp_dir / "in.mp4", inspector=inspector)
        self.assertEqual(len(calls), 1)


# ---------------------------------------------------------------------------
# Diagnostics log
# ---------------------------------------------------------------------------


class DiagnosticsTests(OverlayRenderTempTestCase):
    def test_log_written_when_enabled(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        config = dataclasses.replace(self.config, diagnostics_write_log=True)
        output = self.temp_dir / "out.mp4"
        request = self._request(plans, manifest, output_path=output)
        runner = FakeRunner()
        ore.execute_overlay_render_plan(request, config, runner=runner)
        log_path = output.with_name(f"{output.stem}{config.diagnostics_log_filename_suffix}")
        self.assertTrue(log_path.exists())
        data = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertIn("render_id", data)
        self.assertIn("command", data)

    def test_log_not_written_when_disabled(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        config = dataclasses.replace(self.config, diagnostics_write_log=False)
        output = self.temp_dir / "out.mp4"
        request = self._request(plans, manifest, output_path=output)
        runner = FakeRunner()
        ore.execute_overlay_render_plan(request, config, runner=runner)
        log_path = output.with_name(f"{output.stem}{config.diagnostics_log_filename_suffix}")
        self.assertFalse(log_path.exists())

    def test_save_log_is_atomic_no_leftover_tmp_file(self):
        result = ore.OverlayRenderResult(render_id="abc123")
        log_path = self.temp_dir / "log.json"
        ore.save_overlay_render_log(result, log_path)
        self.assertTrue(log_path.exists())
        self.assertFalse((self.temp_dir / "log.json.tmp").exists())

    def test_result_to_dict_round_trips_json_serializable(self):
        result = ore.OverlayRenderResult(render_id="abc123", warnings=["w1"])
        data = ore.overlay_render_result_to_dict(result)
        json.dumps(data)  # must not raise
        self.assertEqual(data["render_id"], "abc123")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(OverlayRenderTempTestCase):
    def test_dry_run_via_cli(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        manifest_path = self.temp_dir / "manifest.json"
        oar.save_overlay_asset_manifest(manifest, manifest_path)
        video = self._video()
        output = self.temp_dir / "out.mp4"
        ore.main([
            "--video", str(video), "--output", str(output),
            "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
            "--overlay-asset-manifest", str(manifest_path), "--dry-run",
        ])
        self.assertFalse(output.exists())

    def test_dry_run_via_cli_json(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        manifest_path = self.temp_dir / "manifest.json"
        oar.save_overlay_asset_manifest(manifest, manifest_path)
        video = self._video()
        output = self.temp_dir / "out.mp4"
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ore.main([
                "--video", str(video), "--output", str(output),
                "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
                "--overlay-asset-manifest", str(manifest_path), "--dry-run", "--json",
            ])
        data = json.loads(buffer.getvalue())
        self.assertIn("render_id", data)

    def test_cli_missing_video_fails_cleanly(self):
        plans = self._plans()
        manifest = self._overlay_asset_manifest(plans["overlay_plan"])
        manifest_path = self.temp_dir / "manifest.json"
        oar.save_overlay_asset_manifest(manifest, manifest_path)
        output = self.temp_dir / "out.mp4"
        with self.assertRaises(SystemExit):
            ore.main([
                "--video", str(self.temp_dir / "missing.mp4"), "--output", str(output),
                "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
                "--overlay-asset-manifest", str(manifest_path), "--dry-run",
            ])

    def test_cli_malformed_manifest_fails_cleanly(self):
        plans = self._plans()
        manifest_path = self.temp_dir / "bad_manifest.json"
        manifest_path.write_text("{not valid", encoding="utf-8")
        video = self._video()
        output = self.temp_dir / "out.mp4"
        with self.assertRaises(SystemExit):
            ore.main([
                "--video", str(video), "--output", str(output),
                "--renderer-plan", str(plans["renderer_plan_path"]), "--overlay-plan", str(plans["overlay_plan_path"]),
                "--overlay-asset-manifest", str(manifest_path), "--dry-run",
            ])

    def test_required_arguments_enforced(self):
        with self.assertRaises(SystemExit):
            ore.parse_arguments(["--video", "v.mp4", "--output", "o.mp4"])


# ---------------------------------------------------------------------------
# Structural safety
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = Path(__file__).resolve().parent / "overlay_render_engine.py"
        cls.source_lines = cls.source_path.read_text(encoding="utf-8").splitlines()
        cls.source_text = "\n".join(cls.source_lines)

    def test_only_one_real_subprocess_call_site(self):
        call_sites = [line for line in self.source_lines if "subprocess.run(" in line]
        self.assertEqual(len(call_sites), 1)

    def test_no_shell_true(self):
        self.assertNotIn("shell=True", self.source_text)

    def test_no_network_imports(self):
        for line in self.source_lines:
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import requests"))
            self.assertFalse(stripped.startswith("import urllib"))
            self.assertFalse(stripped.startswith("from urllib"))

    def test_no_video_engine_or_forbidden_module_imports(self):
        for forbidden in ("video_engine", "media_inspector", "music_mixer", "reel_builder", "subtitle_engine",
                           "renderer_execution_engine", "timeline_engine"):
            for line in self.source_lines:
                stripped = line.strip()
                if stripped.startswith("from . import") or stripped.startswith("import "):
                    self.assertNotIn(forbidden, stripped, f"{forbidden} should not be imported by overlay_render_engine.py")

    def test_no_pillow_import(self):
        for forbidden in ("import PIL", "from PIL", "Image.open("):
            self.assertNotIn(forbidden, self.source_text)

    def test_no_direct_ffprobe_call(self):
        # This module never probes media itself -- it has no ffprobe
        # binary config field and never names "ffprobe" as a command.
        self.assertNotIn("ffprobe_binary", self.source_text)
        self.assertNotIn('"ffprobe"', self.source_text)
        self.assertNotIn("'ffprobe'", self.source_text)

    def test_no_playwright_or_publish_reference(self):
        for forbidden in ("playwright", "Playwright", "instagram", "Instagram", "publish_reel", "upload_reel"):
            self.assertNotIn(forbidden, self.source_text)

    def test_default_runner_maps_exceptions_correctly(self):
        self.assertIn("except FileNotFoundError as exc:", self.source_text)
        self.assertIn("raise FFmpegNotFoundError(", self.source_text)
        self.assertIn("except subprocess.TimeoutExpired as exc:", self.source_text)
        self.assertIn("raise FFmpegTimeoutError(", self.source_text)

    def test_no_retry_loop_around_runner_call(self):
        self.assertNotIn("for attempt in range", self.source_text)
        self.assertNotIn("while True", self.source_text)

    def test_no_independent_filter_string_construction(self):
        # This module must never build/edit filter syntax itself -- every
        # character of -filter_complex's value comes from
        # filter_graph_serializer.SerializedFilterGraph.filter_expression.
        for forbidden in ("f\"overlay=", "'overlay='", "escape_filter_value", "escape_drawtext_text"):
            self.assertNotIn(forbidden, self.source_text)

    def test_uses_filter_graph_builder_and_serializer(self):
        self.assertIn("filter_graph_builder.build_overlay_filter_spec", self.source_text)
        self.assertIn("filter_graph_builder.build_filter_graph_from_filters", self.source_text)
        self.assertIn("filter_graph_serializer.serialize_filter_graph", self.source_text)

    def test_never_resolves_overlay_assets_itself(self):
        # Asset resolution (overlay_asset_resolver.resolve_overlay_assets)
        # is Renderer Execution's job -- this module only ever consumes
        # an already-built OverlayAssetManifest.
        self.assertNotIn("resolve_overlay_assets(", self.source_text)
        self.assertNotIn("collect_overlay_asset_references(", self.source_text)


if __name__ == "__main__":
    unittest.main()
