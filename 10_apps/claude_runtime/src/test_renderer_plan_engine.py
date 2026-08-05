from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import overlay_plan_engine, timeline_engine
from .renderer_plan_engine import (
    AssetType,
    RenderPassType,
    RendererAsset,
    RendererAudioTrack,
    RendererHint,
    RendererHintType,
    RendererMusicMetadataError,
    RendererMusicTrack,
    RendererOverlayPlanLoadError,
    RendererOverlayPlanValidationError,
    RendererOverlayTrack,
    RendererPass,
    RendererPlan,
    RendererPlanConfig,
    RendererPlanJSONError,
    RendererPlanMismatchError,
    RendererPlanOutputExistsError,
    RendererSubtitleTrack,
    RendererTimelineLoadError,
    RendererTimelineValidationError,
    RendererVideoTrack,
    UnsafeRendererPlanOutputError,
    build_renderer_plan,
    load_music_metadata,
    load_renderer_plan,
    load_renderer_plan_config,
    parse_arguments,
    renderer_plan_from_dict,
    renderer_plan_to_dict,
    save_renderer_plan,
    validate_renderer_plan,
)

MODULE_PATH = Path(__file__).resolve().parent / "renderer_plan_engine.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _video_clip(clip_id="clip_1", *, start=0.0, end=10.0, source_path="/scenes/s1.mp4") -> timeline_engine.VideoClip:
    return timeline_engine.VideoClip(
        clip_id=clip_id, track_id="track_video", source_path=source_path,
        start=start, end=end, duration_seconds=end - start, source_out=end - start, scene_number=1,
    )


def _subtitle_overlay_clip(
    clip_id="sub_1", *, start=0.0, end=2.5, text="a quiet morning", position=None,
) -> timeline_engine.OverlayClip:
    return timeline_engine.OverlayClip(
        clip_id=clip_id, track_id="track_subtitles", source_path="", overlay_type="subtitle",
        start=start, end=end, duration_seconds=end - start, source_out=end - start, content=text,
        metadata={
            "position": position if position is not None else {"x": 540, "y": 1450, "anchor": "bottom_center"},
            "style": {"style_id": "default", "font_family": "Arial"},
            "language": "en",
        },
    )


def _overlay_clip(
    clip_id, overlay_type, *, start=0.0, end=3.0, content="", position=None, asset=None, track_id="track_overlay",
) -> timeline_engine.OverlayClip:
    metadata = {}
    if position is not None:
        metadata["position"] = position
    if asset is not None:
        metadata["asset"] = asset
    return timeline_engine.OverlayClip(
        clip_id=clip_id, track_id=track_id, source_path="", overlay_type=overlay_type,
        start=start, end=end, duration_seconds=end - start, source_out=end - start, content=content, metadata=metadata,
    )


def _build_timeline(*, timeline_id="tl1", duration=10.0, with_subtitle=True, with_cta=False, with_watermark=False) -> timeline_engine.Timeline:
    tracks = [
        timeline_engine.TimelineTrack(track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0, clips=[_video_clip(end=duration)])
    ]
    order = 1
    if with_subtitle:
        tracks.append(timeline_engine.TimelineTrack(track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=order, clips=[_subtitle_overlay_clip()]))
        order += 1
    if with_cta:
        tracks.append(timeline_engine.TimelineTrack(track_id="track_cta", track_type=timeline_engine.TrackType.OVERLAY, order=order, clips=[
            _overlay_clip("cta_1", "cta", start=1.0, end=4.0, content="Follow us!", position={"x": 540, "y": 1000, "width": 300, "height": 80}, track_id="track_cta"),
        ]))
        order += 1
    if with_watermark:
        tracks.append(timeline_engine.TimelineTrack(track_id="track_watermark", track_type=timeline_engine.TrackType.OVERLAY, order=order, clips=[
            _overlay_clip("wm_1", "watermark", start=0.0, end=duration, position={"x": 980, "y": 100, "width": 80, "height": 40}, asset={"asset_reference": "brand-logo.png"}, track_id="track_watermark"),
        ]))
        order += 1
    return timeline_engine.Timeline(timeline_id=timeline_id, duration_seconds=duration, tracks=tracks)


class RendererPlanTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = load_renderer_plan_config()
        self.overlay_config = overlay_plan_engine.load_overlay_plan_config()

    def _write_timeline(self, timeline: timeline_engine.Timeline, *, name: str = "timeline.json") -> Path:
        path = self.temp_dir / name
        timeline_engine.save_timeline(timeline, path)
        return path

    def _write_overlay_plan(self, timeline_path: Path, *, name: str = "overlay_plan.json") -> tuple[Path, overlay_plan_engine.OverlayPlan]:
        plan = overlay_plan_engine.build_overlay_plan(timeline_path, self.overlay_config)
        path = self.temp_dir / name
        overlay_plan_engine.save_overlay_plan(plan, path)
        return path, plan

    def _standard_paths(self, **timeline_kwargs) -> tuple[Path, Path, timeline_engine.Timeline, overlay_plan_engine.OverlayPlan]:
        timeline = _build_timeline(**timeline_kwargs)
        timeline_path = self._write_timeline(timeline)
        overlay_plan_path, overlay_plan = self._write_overlay_plan(timeline_path)
        return timeline_path, overlay_plan_path, timeline, overlay_plan

    def _write_music_plan(self, **overrides) -> Path:
        data = {"music_path": "/music/bg.mp3", "volume": 0.2, "mode": "loop"}
        data.update(overrides)
        path = self.temp_dir / "music_plan.json"
        path.write_text(json.dumps(data))
        return path


# ---------------------------------------------------------------------------
# Valid plans / loading
# ---------------------------------------------------------------------------


class ValidPlanTests(RendererPlanTempTestCase):
    def test_valid_full_plan_builds(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True, with_cta=True, with_watermark=True)
        music_plan_path = self._write_music_plan()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        self.assertTrue(plan.video_tracks)
        self.assertTrue(plan.subtitle_tracks)
        self.assertTrue(plan.overlay_tracks)
        self.assertTrue(plan.music_tracks)

    def test_empty_overlay_plan_produces_valid_plan(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=False)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan.subtitle_tracks, [])
        self.assertEqual(plan.overlay_tracks, [])
        self.assertTrue(any(w.code == "empty_overlay_plan" for w in plan.warnings))

    def test_subtitle_only_plan(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True, with_cta=False, with_watermark=False)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(len(plan.subtitle_tracks), 1)
        self.assertEqual(plan.overlay_tracks, [])
        self.assertEqual(plan.music_tracks, [])

    def test_no_music_plan_means_no_music_track(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan.music_tracks, [])

    def test_music_plan_creates_music_track(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        music_plan_path = self._write_music_plan(volume=0.5, mode="trim")
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        self.assertEqual(len(plan.music_tracks), 1)
        self.assertEqual(plan.music_tracks[0].volume, 0.5)
        self.assertEqual(plan.music_tracks[0].mode, "trim")

    def test_malformed_timeline_json_fails(self) -> None:
        timeline_path = self.temp_dir / "timeline.json"
        timeline_path.write_text("{not valid json")
        overlay_plan_path = self.temp_dir / "overlay_plan.json"
        overlay_plan_path.write_text("{}")
        with self.assertRaises(RendererTimelineLoadError):
            build_renderer_plan(timeline_path, overlay_plan_path, self.config)

    def test_invalid_timeline_fails(self) -> None:
        bad_timeline = timeline_engine.Timeline(timeline_id="bad", duration_seconds=0.0, tracks=[])
        timeline_path = self._write_timeline(bad_timeline)
        overlay_plan_path = self.temp_dir / "overlay_plan.json"
        overlay_plan_path.write_text("{}")
        with self.assertRaises(RendererTimelineValidationError):
            build_renderer_plan(timeline_path, overlay_plan_path, self.config)

    def test_malformed_overlay_plan_json_fails(self) -> None:
        timeline_path = self._write_timeline(_build_timeline())
        overlay_plan_path = self.temp_dir / "overlay_plan.json"
        overlay_plan_path.write_text("{not valid json")
        with self.assertRaises(RendererOverlayPlanLoadError):
            build_renderer_plan(timeline_path, overlay_plan_path, self.config)

    def test_overlay_plan_timeline_id_mismatch_fails(self) -> None:
        timeline_path, overlay_plan_path, _t, overlay_plan = self._standard_paths()
        overlay_plan.timeline_id = "different-timeline"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path, force=True)
        with self.assertRaises(RendererPlanMismatchError):
            build_renderer_plan(timeline_path, overlay_plan_path, self.config)

    def test_source_files_unchanged(self) -> None:
        import hashlib

        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        before_t = hashlib.sha256(timeline_path.read_bytes()).hexdigest()
        before_o = hashlib.sha256(overlay_plan_path.read_bytes()).hexdigest()
        build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(hashlib.sha256(timeline_path.read_bytes()).hexdigest(), before_t)
        self.assertEqual(hashlib.sha256(overlay_plan_path.read_bytes()).hexdigest(), before_o)


# ---------------------------------------------------------------------------
# Music metadata
# ---------------------------------------------------------------------------


class MusicMetadataTests(RendererPlanTempTestCase):
    def test_valid_music_metadata_loads(self) -> None:
        path = self._write_music_plan()
        data = load_music_metadata(path)
        self.assertEqual(data["music_path"], "/music/bg.mp3")

    def test_missing_music_path_fails(self) -> None:
        path = self.temp_dir / "music.json"
        path.write_text(json.dumps({"volume": 0.5}))
        with self.assertRaises(RendererMusicMetadataError):
            load_music_metadata(path)

    def test_malformed_music_json_fails(self) -> None:
        path = self.temp_dir / "music.json"
        path.write_text("{not valid json")
        with self.assertRaises(RendererMusicMetadataError):
            load_music_metadata(path)

    def test_missing_music_file_fails(self) -> None:
        with self.assertRaises(RendererMusicMetadataError):
            load_music_metadata(self.temp_dir / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Asset references
# ---------------------------------------------------------------------------


class AssetReferenceTests(RendererPlanTempTestCase):
    def test_video_source_path_preserved_as_asset(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        video_assets = [a for a in plan.assets if a.asset_type == AssetType.VIDEO]
        self.assertEqual(video_assets[0].source_path, "/scenes/s1.mp4")

    def test_overlay_asset_reference_preserved(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_watermark=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        image_assets = [a for a in plan.assets if a.asset_type == AssetType.IMAGE]
        self.assertEqual(image_assets[0].source_path, "brand-logo.png")

    def test_music_path_preserved_as_asset(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        music_plan_path = self._write_music_plan()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        music_assets = [a for a in plan.assets if a.asset_type == AssetType.MUSIC]
        self.assertEqual(music_assets[0].source_path, "/music/bg.mp3")

    def test_duplicate_source_paths_deduplicated(self) -> None:
        clip1 = _video_clip("c1", start=0.0, end=5.0, source_path="/scenes/same.mp4")
        clip2 = _video_clip("c2", start=5.0, end=10.0, source_path="/scenes/same.mp4")
        clip2.scene_number = 2
        timeline = timeline_engine.Timeline(
            timeline_id="tl1", duration_seconds=10.0,
            tracks=[timeline_engine.TimelineTrack(track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0, clips=[clip1, clip2])],
        )
        timeline_path = self._write_timeline(timeline)
        overlay_plan_path, _o = self._write_overlay_plan(timeline_path)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        video_assets = [a for a in plan.assets if a.asset_type == AssetType.VIDEO]
        self.assertEqual(len(video_assets), 1)

    def test_no_asset_opened_or_verified(self) -> None:
        # "download" appears only in this module's own explanatory
        # comments ("never downloads any asset/font") -- checked via
        # actual call patterns, not a bare substring match.
        for forbidden in ("os.path.exists", ".read_bytes()", "download(", "urlretrieve"):
            self.assertNotIn(forbidden, MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Renderer hints
# ---------------------------------------------------------------------------


class HintTests(RendererPlanTempTestCase):
    def test_video_pass_gets_configured_hint(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        hint = next(h for h in plan.hints if h.target_id == "pass_video")
        self.assertEqual(hint.hint_type, self.config.video_hint)

    def test_subtitle_pass_gets_drawtext_hint(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        hint = next(h for h in plan.hints if h.target_id == "pass_subtitle")
        self.assertEqual(hint.hint_type, RendererHintType.DRAWTEXT)

    def test_overlay_pass_gets_overlay_png_hint(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_cta=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        hint = next(h for h in plan.hints if h.target_id == "pass_overlay")
        self.assertEqual(hint.hint_type, RendererHintType.OVERLAY_PNG)

    def test_music_pass_gets_normalize_hint(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        music_plan_path = self._write_music_plan()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        hint = next(h for h in plan.hints if h.target_id == "pass_music")
        self.assertEqual(hint.hint_type, RendererHintType.NORMALIZE)

    def test_unknown_hint_rejected_by_default(self) -> None:
        plan = RendererPlan(
            timeline_id="tl1", passes=[RendererPass(pass_id="pass_video", pass_type=RenderPassType.VIDEO)],
            video_tracks=[RendererVideoTrack(track_id="rt_video", asset_ids=["asset_video_0001"])],
            assets=[RendererAsset(asset_id="asset_video_0001", asset_type=AssetType.VIDEO, source_path="/x.mp4")],
            hints=[RendererHint(hint_id="h1", hint_type="not_a_real_hint", target_id="pass_video")],
        )
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_unknown_hint_allowed_when_configured(self) -> None:
        plan = RendererPlan(
            timeline_id="tl1", passes=[RendererPass(pass_id="pass_video", pass_type=RenderPassType.VIDEO)],
            video_tracks=[RendererVideoTrack(track_id="rt_video", asset_ids=["asset_video_0001"])],
            assets=[RendererAsset(asset_id="asset_video_0001", asset_type=AssetType.VIDEO, source_path="/x.mp4")],
            hints=[RendererHint(hint_id="h1", hint_type="not_a_real_hint", target_id="pass_video")],
        )
        config = RendererPlanConfig(allow_unknown_renderer_hints=True)
        result = validate_renderer_plan(plan, config)
        self.assertTrue(result.passed)

    def test_hints_never_executed(self) -> None:
        for forbidden in ("subprocess", "os.system", "eval(", "exec("):
            self.assertNotIn(forbidden, MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityTests(RendererPlanTempTestCase):
    def test_stable_renderer_plan_id(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan1 = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        plan2 = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan1.renderer_plan_id, plan2.renderer_plan_id)

    def test_created_at_excluded_from_id(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan1 = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        plan2 = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertNotEqual(plan1.created_at, plan2.created_at)
        self.assertEqual(plan1.renderer_plan_id, plan2.renderer_plan_id)

    def test_timeline_change_changes_id(self) -> None:
        timeline_path1, overlay_plan_path1, _t1, _o1 = self._standard_paths(timeline_id="tl1")
        with tempfile.TemporaryDirectory() as tmp2:
            timeline2 = _build_timeline(timeline_id="tl2")
            timeline_path2 = Path(tmp2) / "timeline.json"
            timeline_engine.save_timeline(timeline2, timeline_path2)
            overlay_plan2 = overlay_plan_engine.build_overlay_plan(timeline_path2, self.overlay_config)
            overlay_plan_path2 = Path(tmp2) / "overlay_plan.json"
            overlay_plan_engine.save_overlay_plan(overlay_plan2, overlay_plan_path2)

            plan1 = build_renderer_plan(timeline_path1, overlay_plan_path1, self.config)
            plan2 = build_renderer_plan(timeline_path2, overlay_plan_path2, self.config)
            self.assertNotEqual(plan1.renderer_plan_id, plan2.renderer_plan_id)

    def test_music_presence_changes_id(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan_no_music = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        music_plan_path = self._write_music_plan()
        plan_with_music = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        self.assertNotEqual(plan_no_music.renderer_plan_id, plan_with_music.renderer_plan_id)

    def test_config_change_changes_id(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan1 = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        config2 = RendererPlanConfig(renderer_version="different-version")
        plan2 = build_renderer_plan(timeline_path, overlay_plan_path, config2)
        self.assertNotEqual(plan1.renderer_plan_id, plan2.renderer_plan_id)


# ---------------------------------------------------------------------------
# Duplicate ids / dependency graph
# ---------------------------------------------------------------------------


class DependencyGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_renderer_plan_config()

    def _minimal_plan(self, **overrides) -> RendererPlan:
        base = dict(
            timeline_id="tl1",
            video_tracks=[RendererVideoTrack(track_id="rt_video", asset_ids=["asset_video_0001"])],
            assets=[RendererAsset(asset_id="asset_video_0001", asset_type=AssetType.VIDEO, source_path="/x.mp4")],
            passes=[RendererPass(pass_id="pass_video", pass_type=RenderPassType.VIDEO, outputs=["pass_video_output"])],
        )
        base.update(overrides)
        return RendererPlan(**base)

    def test_duplicate_asset_ids_fail(self) -> None:
        plan = self._minimal_plan(assets=[
            RendererAsset(asset_id="dup", asset_type=AssetType.VIDEO, source_path="/a.mp4"),
            RendererAsset(asset_id="dup", asset_type=AssetType.VIDEO, source_path="/b.mp4"),
        ], video_tracks=[RendererVideoTrack(track_id="rt_video", asset_ids=["dup"])])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_duplicate_pass_ids_fail(self) -> None:
        plan = self._minimal_plan(passes=[
            RendererPass(pass_id="dup", pass_type=RenderPassType.VIDEO, outputs=["o1"]),
            RendererPass(pass_id="dup", pass_type=RenderPassType.FINAL_ENCODE, outputs=["o2"]),
        ])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_duplicate_track_ids_fail(self) -> None:
        plan = self._minimal_plan(
            video_tracks=[RendererVideoTrack(track_id="dup", asset_ids=["asset_video_0001"])],
            audio_tracks=[RendererAudioTrack(track_id="dup")],
        )
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_dangling_dependency_fails(self) -> None:
        plan = self._minimal_plan(passes=[
            RendererPass(pass_id="pass_video", pass_type=RenderPassType.VIDEO, outputs=["o1"], dependencies=["pass_missing"]),
        ])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_cycle_detected(self) -> None:
        plan = self._minimal_plan(passes=[
            RendererPass(pass_id="pass_a", pass_type=RenderPassType.VIDEO, outputs=["oa"], dependencies=["pass_b"]),
            RendererPass(pass_id="pass_b", pass_type=RenderPassType.SUBTITLE, outputs=["ob"], dependencies=["pass_a"]),
        ])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_no_cycle_in_linear_chain(self) -> None:
        plan = self._minimal_plan(passes=[
            RendererPass(pass_id="pass_video", pass_type=RenderPassType.VIDEO, outputs=["o1"]),
            RendererPass(pass_id="pass_final_encode", pass_type=RenderPassType.FINAL_ENCODE, outputs=["o2"], dependencies=["pass_video"]),
        ])
        result = validate_renderer_plan(plan, self.config)
        self.assertTrue(result.passed)

    def test_duplicate_outputs_across_passes_fail(self) -> None:
        plan = self._minimal_plan(passes=[
            RendererPass(pass_id="pass_video", pass_type=RenderPassType.VIDEO, outputs=["same_output"]),
            RendererPass(pass_id="pass_final_encode", pass_type=RenderPassType.FINAL_ENCODE, outputs=["same_output"], dependencies=["pass_video"]),
        ])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_invalid_pass_type_fails(self) -> None:
        plan = self._minimal_plan(passes=[
            RendererPass(pass_id="pass_video", pass_type="not_a_real_type", outputs=["o1"]),
        ])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_orphan_asset_fails_by_default(self) -> None:
        plan = self._minimal_plan(assets=[
            RendererAsset(asset_id="asset_video_0001", asset_type=AssetType.VIDEO, source_path="/x.mp4"),
            RendererAsset(asset_id="orphan", asset_type=AssetType.IMAGE, source_path="/y.png"),
        ])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_orphan_asset_allowed_when_configured(self) -> None:
        plan = self._minimal_plan(assets=[
            RendererAsset(asset_id="asset_video_0001", asset_type=AssetType.VIDEO, source_path="/x.mp4"),
            RendererAsset(asset_id="orphan", asset_type=AssetType.IMAGE, source_path="/y.png"),
        ])
        config = RendererPlanConfig(allow_orphan_assets=True)
        result = validate_renderer_plan(plan, config)
        self.assertTrue(result.passed)

    def test_missing_tracks_fails(self) -> None:
        plan = self._minimal_plan(video_tracks=[])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_music_track_missing_asset_id_fails(self) -> None:
        plan = self._minimal_plan(music_tracks=[RendererMusicTrack(track_id="rt_music", asset_id=None)])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_hint_with_invalid_target_fails(self) -> None:
        plan = self._minimal_plan(hints=[RendererHint(hint_id="h1", hint_type=RendererHintType.COPY, target_id="does_not_exist")])
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_non_serializable_metadata_fails(self) -> None:
        plan = self._minimal_plan(metadata={"bad": object()})
        result = validate_renderer_plan(plan, self.config)
        self.assertFalse(result.passed)


# ---------------------------------------------------------------------------
# Pass ordering
# ---------------------------------------------------------------------------


class PassOrderingTests(RendererPlanTempTestCase):
    def test_deterministic_pass_order_full(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True, with_cta=True)
        music_plan_path = self._write_music_plan()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        self.assertEqual(
            [p.pass_type for p in plan.passes],
            [RenderPassType.VIDEO, RenderPassType.SUBTITLE, RenderPassType.OVERLAY, RenderPassType.MUSIC, RenderPassType.FINAL_ENCODE],
        )

    def test_missing_stages_skipped(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=False)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual([p.pass_type for p in plan.passes], [RenderPassType.VIDEO, RenderPassType.FINAL_ENCODE])

    def test_pass_dependencies_chain_linearly(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        video_pass = next(p for p in plan.passes if p.pass_type == RenderPassType.VIDEO)
        subtitle_pass = next(p for p in plan.passes if p.pass_type == RenderPassType.SUBTITLE)
        self.assertEqual(video_pass.dependencies, [])
        self.assertEqual(subtitle_pass.dependencies, [video_pass.pass_id])

    def test_final_encode_always_last(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True, with_cta=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan.passes[-1].pass_type, RenderPassType.FINAL_ENCODE)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class JsonTests(RendererPlanTempTestCase):
    def test_round_trip_preserved(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True, with_cta=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(renderer_plan_to_dict(rebuilt), renderer_plan_to_dict(plan))

    def test_atomic_write_no_tmp_left_behind(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        out = self.temp_dir / "renderer_plan.json"
        save_renderer_plan(plan, out)
        self.assertTrue(out.is_file())
        self.assertFalse((self.temp_dir / "renderer_plan.json.tmp").exists())

    def test_overwrite_refused_without_force(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        out = self.temp_dir / "renderer_plan.json"
        save_renderer_plan(plan, out)
        with self.assertRaises(RendererPlanOutputExistsError):
            save_renderer_plan(plan, out)

    def test_force_overwrites(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        out = self.temp_dir / "renderer_plan.json"
        save_renderer_plan(plan, out)
        save_renderer_plan(plan, out, force=True)
        self.assertTrue(out.is_file())

    def test_output_path_directory_rejected(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        out_dir = self.temp_dir / "a_directory"
        out_dir.mkdir()
        with self.assertRaises(UnsafeRendererPlanOutputError):
            save_renderer_plan(plan, out_dir)

    def test_output_same_as_timeline_rejected_via_cli(self) -> None:
        from .renderer_plan_engine import main as cli_main

        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        with self.assertRaises(SystemExit):
            cli_main(["--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path), "--output", str(timeline_path)])

    def test_output_same_as_overlay_plan_rejected_via_cli(self) -> None:
        from .renderer_plan_engine import main as cli_main

        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        with self.assertRaises(SystemExit):
            cli_main(["--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path), "--output", str(overlay_plan_path)])

    def test_malformed_renderer_plan_json_fails(self) -> None:
        bad = self.temp_dir / "bad.json"
        bad.write_text("{not valid json")
        with self.assertRaises(RendererPlanJSONError):
            load_renderer_plan(bad)

    def test_unknown_metadata_preserved(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        plan.metadata["custom"] = {"nested": True}
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(rebuilt.metadata["custom"], {"nested": True})

    def test_parent_directory_created(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        out = self.temp_dir / "nested" / "dir" / "renderer_plan.json"
        save_renderer_plan(plan, out)
        self.assertTrue(out.is_file())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(RendererPlanTempTestCase):
    def test_timeline_overlay_output_works(self) -> None:
        from .renderer_plan_engine import main as cli_main

        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        out = self.temp_dir / "renderer_plan.json"
        cli_main(["--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path), "--output", str(out)])
        self.assertTrue(out.is_file())

    def test_validate_only_writes_nothing(self) -> None:
        from .renderer_plan_engine import main as cli_main

        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        before = sorted(p.name for p in self.temp_dir.iterdir())
        cli_main(["--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path), "--validate-only"])
        after = sorted(p.name for p in self.temp_dir.iterdir())
        self.assertEqual(before, after)

    def test_json_flag_valid(self) -> None:
        import contextlib
        import io

        from .renderer_plan_engine import main as cli_main

        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        out = self.temp_dir / "renderer_plan.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli_main(["--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path), "--output", str(out), "--json"])
        data = json.loads(buffer.getvalue())
        self.assertIn("renderer_plan_id", data)

    def test_music_plan_flag_accepted(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        music_plan_path = self._write_music_plan()
        args = parse_arguments([
            "--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path),
            "--output", "o.json", "--music-plan", str(music_plan_path),
        ])
        self.assertEqual(args.music_plan, str(music_plan_path))

    def test_force_flag_works(self) -> None:
        args = parse_arguments(["--timeline", "t.json", "--overlay-plan", "o.json", "--output", "out.json", "--force"])
        self.assertTrue(args.force)

    def test_missing_timeline_fails(self) -> None:
        from .renderer_plan_engine import main as cli_main

        with self.assertRaises(SystemExit):
            cli_main([
                "--timeline", str(self.temp_dir / "does_not_exist.json"),
                "--overlay-plan", str(self.temp_dir / "op.json"),
                "--output", str(self.temp_dir / "o.json"),
            ])

    def test_output_required_unless_validate_only(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--timeline", "t.json", "--overlay-plan", "o.json"])

    def test_output_conflicts_with_validate_only(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--timeline", "t.json", "--overlay-plan", "o.json", "--output", "out.json", "--validate-only"])

    def test_expected_failure_no_traceback(self) -> None:
        import contextlib
        import io

        from .renderer_plan_engine import main as cli_main

        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            cli_main([
                "--timeline", str(self.temp_dir / "does_not_exist.json"),
                "--overlay-plan", str(self.temp_dir / "op.json"),
                "--output", str(self.temp_dir / "o.json"),
            ])
        self.assertIn("[RendererPlanEngine]", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())


# ---------------------------------------------------------------------------
# Validation (build-time cross-checks)
# ---------------------------------------------------------------------------


class ValidationCrossCheckTests(RendererPlanTempTestCase):
    def test_valid_plan_passes_with_supplied_timeline_and_overlay_plan(self) -> None:
        timeline_path, overlay_plan_path, timeline, overlay_plan = self._standard_paths(with_subtitle=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        result = validate_renderer_plan(plan, self.config, timeline=timeline, overlay_plan=overlay_plan)
        self.assertTrue(result.passed)

    def test_timeline_mismatch_detected(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        other_timeline = _build_timeline(timeline_id="other")
        result = validate_renderer_plan(plan, self.config, timeline=other_timeline)
        self.assertFalse(result.passed)

    def test_overlay_plan_mismatch_detected(self) -> None:
        timeline_path, overlay_plan_path, timeline, overlay_plan = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        overlay_plan.plan_id = "different-id"
        result = validate_renderer_plan(plan, self.config, overlay_plan=overlay_plan)
        self.assertFalse(result.passed)

    def test_counts_correct(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True, with_cta=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        result = validate_renderer_plan(plan, self.config)
        self.assertEqual(result.pass_count, len(plan.passes))
        self.assertEqual(result.asset_count, len(plan.assets))


# ---------------------------------------------------------------------------
# Structural safety
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def test_no_playwright_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import playwright"))
            self.assertFalse(stripped.startswith("from playwright"))

    def test_no_instagram_session_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from src.social"))
            self.assertFalse(stripped.startswith("from .social"))
        self.assertNotIn("InstagramSession(", MODULE_SOURCE)

    def test_no_publishing_or_social_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from src.publishing"))
            self.assertFalse(stripped.startswith("from .publishing"))

    def test_no_ffmpeg_ffprobe_or_subprocess(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import subprocess"))
            self.assertFalse(stripped.startswith("from subprocess"))
        for forbidden in ("subprocess.run(", "subprocess.Popen(", "ffmpeg_binary", "ffprobe_binary"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_does_not_import_other_engines(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from .subtitle_engine"))
            self.assertFalse(stripped.startswith("from .subtitle_timeline_integration"))
            self.assertFalse(stripped.startswith("from .music_mixer"))
            self.assertFalse(stripped.startswith("from .video_engine"))
            self.assertFalse(stripped.startswith("from .reel_builder"))

    def test_no_whisper_or_speech_recognition(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip().lower()
            self.assertFalse(stripped.startswith("import whisper"))
            self.assertFalse(stripped.startswith("from whisper"))
            self.assertFalse(stripped.startswith("import speech_recognition"))

    def test_no_font_loading(self) -> None:
        for forbidden in (".ttf", ".otf", ".woff", "font_path", "FontFile"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_network_or_download_code(self) -> None:
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_rendering_code(self) -> None:
        for forbidden in ("cv2.", "PIL.", "moviepy", "draw_text(", "render_frame("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_publishing_upload_code(self) -> None:
        for forbidden in ("requests.post", "upload_to", "publish_to"):
            self.assertNotIn(forbidden, MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Track / asset field content
# ---------------------------------------------------------------------------


class TrackFieldTests(RendererPlanTempTestCase):
    def test_video_track_duration_sums_enabled_clips(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan.video_tracks[0].duration_seconds, 10.0)

    def test_video_track_canvas_fields_from_timeline(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan.video_tracks[0].width, plan.canvas_width)
        self.assertEqual(plan.video_tracks[0].height, plan.canvas_height)
        self.assertEqual(plan.video_tracks[0].frame_rate, plan.frame_rate)

    def test_video_track_source_clip_ids_preserved(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan.video_tracks[0].source_clip_ids, ["clip_1"])

    def test_audio_track_derived_when_present(self) -> None:
        timeline = _build_timeline(with_subtitle=False)
        timeline.tracks.append(
            timeline_engine.TimelineTrack(
                track_id="track_audio", track_type=timeline_engine.TrackType.AUDIO, order=1,
                clips=[timeline_engine.AudioClip(clip_id="a1", track_id="track_audio", source_path="/audio/voice.wav", start=0.0, end=10.0, duration_seconds=10.0, source_out=10.0)],
            )
        )
        timeline_path = self._write_timeline(timeline)
        overlay_plan_path, _o = self._write_overlay_plan(timeline_path)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(len(plan.audio_tracks), 1)
        self.assertEqual(plan.audio_tracks[0].source_clip_ids, ["a1"])
        audio_assets = [a for a in plan.assets if a.asset_type == AssetType.AUDIO]
        self.assertEqual(audio_assets[0].source_path, "/audio/voice.wav")

    def test_video_asset_logical_role(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        video_asset = next(a for a in plan.assets if a.asset_type == AssetType.VIDEO)
        self.assertEqual(video_asset.logical_role, "scene_video")

    def test_music_asset_logical_role(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        music_plan_path = self._write_music_plan()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        music_asset = next(a for a in plan.assets if a.asset_type == AssetType.MUSIC)
        self.assertEqual(music_asset.logical_role, "background_music")

    def test_music_track_fade_and_ducking_preserved(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        music_plan_path = self._write_music_plan(fade_in_seconds=0.5, fade_out_seconds=1.0, ducking_mode="fixed")
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        track = plan.music_tracks[0]
        self.assertEqual(track.fade_in_seconds, 0.5)
        self.assertEqual(track.fade_out_seconds, 1.0)
        self.assertEqual(track.ducking_mode, "fixed")

    def test_music_metadata_extra_fields_preserved(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        music_plan_path = self._write_music_plan(metadata={"license": "royalty-free"})
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config, music_plan_path=music_plan_path)
        self.assertEqual(plan.music_tracks[0].metadata, {"license": "royalty-free"})

    def test_subtitle_track_language_captured(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan.subtitle_tracks[0].language, "en")

    def test_overlay_tracks_grouped_by_layer(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_cta=True, with_watermark=True)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        layer_ids = {t.layer_id for t in plan.overlay_tracks}
        self.assertEqual(len(plan.overlay_tracks), len(layer_ids))

    def test_disabled_overlay_excluded_from_renderer_tracks(self) -> None:
        timeline = _build_timeline(with_subtitle=False)
        clip = _subtitle_overlay_clip()
        clip.enabled = False
        timeline.tracks.append(
            timeline_engine.TimelineTrack(track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=1, clips=[clip])
        )
        timeline_path = self._write_timeline(timeline)
        overlay_plan_path, _o = self._write_overlay_plan(timeline_path)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(plan.subtitle_tracks, [])

    def test_disabled_overlay_included_by_include_disabled_still_excluded_from_renderer(self) -> None:
        timeline = _build_timeline(with_subtitle=False)
        clip = _subtitle_overlay_clip()
        clip.enabled = False
        timeline.tracks.append(
            timeline_engine.TimelineTrack(track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=1, clips=[clip])
        )
        timeline_path = self._write_timeline(timeline)
        overlay_plan = overlay_plan_engine.build_overlay_plan(timeline_path, self.overlay_config, include_disabled=True)
        overlay_plan_path = self.temp_dir / "overlay_plan.json"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path)
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        # overlay_plan.overlays includes the disabled one (enabled=False),
        # but Renderer Plan Engine must still never treat it as active.
        self.assertEqual(plan.subtitle_tracks, [])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_loads_from_repo(self) -> None:
        config = load_renderer_plan_config()
        self.assertEqual(config.renderer_version, "11E.1")
        self.assertIn(RenderPassType.VIDEO, config.pass_order)

    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(Exception):
            load_renderer_plan_config("/nonexistent/renderer_plan.yaml")

    def test_config_defaults_reasonable(self) -> None:
        config = RendererPlanConfig()
        self.assertEqual(config.video_hint, RendererHintType.COPY)
        self.assertEqual(config.subtitle_hint, RendererHintType.DRAWTEXT)
        self.assertFalse(config.allow_orphan_assets)
        self.assertFalse(config.allow_unknown_renderer_hints)


# ---------------------------------------------------------------------------
# Per-dataclass JSON round trip
# ---------------------------------------------------------------------------


class DataclassRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_renderer_plan_config()

    def test_asset_round_trip(self) -> None:
        plan = RendererPlan(
            timeline_id="tl1",
            assets=[RendererAsset(asset_id="a1", asset_type=AssetType.IMAGE, source_path="/x.png", logical_role="logo", checksum="abc123", metadata={"note": "x"})],
        )
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(rebuilt.assets[0], plan.assets[0])

    def test_hint_round_trip(self) -> None:
        plan = RendererPlan(timeline_id="tl1", hints=[RendererHint(hint_id="h1", hint_type=RendererHintType.ASS, target_id="pass_subtitle", metadata={"x": 1})])
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(rebuilt.hints[0], plan.hints[0])

    def test_pass_round_trip(self) -> None:
        plan = RendererPlan(
            timeline_id="tl1",
            passes=[RendererPass(pass_id="pass_video", pass_type=RenderPassType.VIDEO, order=0, inputs=["a"], outputs=["b"], dependencies=[], metadata={"k": "v"})],
        )
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(rebuilt.passes[0], plan.passes[0])

    def test_music_track_round_trip(self) -> None:
        plan = RendererPlan(timeline_id="tl1", music_tracks=[RendererMusicTrack(track_id="rt_music", asset_id="asset_music_0001", volume=0.3, mode="trim")])
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(rebuilt.music_tracks[0], plan.music_tracks[0])

    def test_subtitle_track_round_trip(self) -> None:
        plan = RendererPlan(timeline_id="tl1", subtitle_tracks=[RendererSubtitleTrack(track_id="rt_sub", overlay_ids=["o1", "o2"], language="ja")])
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(rebuilt.subtitle_tracks[0], plan.subtitle_tracks[0])

    def test_overlay_track_round_trip(self) -> None:
        plan = RendererPlan(timeline_id="tl1", overlay_tracks=[RendererOverlayTrack(track_id="rt_overlay", overlay_ids=["o1"], layer_id="layer_cta")])
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(rebuilt.overlay_tracks[0], plan.overlay_tracks[0])

    def test_output_profile_round_trip(self) -> None:
        plan = RendererPlan(timeline_id="tl1", output_profile={"container": "mov", "video_codec_hint": "prores"})
        rebuilt = renderer_plan_from_dict(renderer_plan_to_dict(plan))
        self.assertEqual(rebuilt.output_profile, plan.output_profile)

    def test_load_missing_renderer_plan_file_raises(self) -> None:
        with self.assertRaises(RendererPlanJSONError):
            load_renderer_plan("/nonexistent/renderer_plan.json")


# ---------------------------------------------------------------------------
# CLI validate-only failure path
# ---------------------------------------------------------------------------


class CliValidateOnlyTests(RendererPlanTempTestCase):
    def test_validate_only_reports_failure_for_mismatched_overlay_plan(self) -> None:
        from .renderer_plan_engine import main as cli_main

        timeline_path, overlay_plan_path, _t, overlay_plan = self._standard_paths()
        overlay_plan.timeline_id = "wrong-id"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path, force=True)

        with self.assertRaises(SystemExit):
            cli_main(["--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path), "--validate-only"])

    def test_validate_only_json_output_valid(self) -> None:
        import contextlib
        import io

        from .renderer_plan_engine import main as cli_main

        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli_main(["--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path), "--validate-only", "--json"])
        data = json.loads(buffer.getvalue())
        self.assertIn("passed", data)
        self.assertTrue(data["passed"])

    def test_validate_only_passing_plan_exits_cleanly(self) -> None:
        from .renderer_plan_engine import main as cli_main

        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        cli_main(["--timeline", str(timeline_path), "--overlay-plan", str(overlay_plan_path), "--validate-only"])


# ---------------------------------------------------------------------------
# Additional identity coverage
# ---------------------------------------------------------------------------


class MoreIdentityTests(RendererPlanTempTestCase):
    def test_overlay_change_changes_id(self) -> None:
        timeline_path, overlay_plan_path1, _t, _o = self._standard_paths(with_subtitle=True)
        plan1 = build_renderer_plan(timeline_path, overlay_plan_path1, self.config)

        timeline2 = _build_timeline(with_subtitle=True, with_cta=True)
        timeline2.timeline_id = "tl1"
        timeline_path2 = self.temp_dir / "timeline2.json"
        timeline_engine.save_timeline(timeline2, timeline_path2)
        overlay_plan2 = overlay_plan_engine.build_overlay_plan(timeline_path2, self.overlay_config)
        overlay_plan_path2 = self.temp_dir / "overlay_plan2.json"
        overlay_plan_engine.save_overlay_plan(overlay_plan2, overlay_plan_path2)
        plan2 = build_renderer_plan(timeline_path2, overlay_plan_path2, self.config)

        self.assertNotEqual(plan1.renderer_plan_id, plan2.renderer_plan_id)

    def test_hint_config_change_changes_id(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths(with_subtitle=True)
        plan1 = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        config2 = RendererPlanConfig(subtitle_hint=RendererHintType.ASS)
        plan2 = build_renderer_plan(timeline_path, overlay_plan_path, config2)
        self.assertNotEqual(plan1.renderer_plan_id, plan2.renderer_plan_id)

    def test_output_profile_change_changes_id(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan1 = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        config2 = RendererPlanConfig(output_container="mov")
        plan2 = build_renderer_plan(timeline_path, overlay_plan_path, config2)
        self.assertNotEqual(plan1.renderer_plan_id, plan2.renderer_plan_id)

    def test_renderer_plan_id_is_16_hex_chars(self) -> None:
        timeline_path, overlay_plan_path, _t, _o = self._standard_paths()
        plan = build_renderer_plan(timeline_path, overlay_plan_path, self.config)
        self.assertEqual(len(plan.renderer_plan_id), 16)
        int(plan.renderer_plan_id, 16)  # must not raise


if __name__ == "__main__":
    unittest.main()
