from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import end_to_end_render_pipeline as e2e
from . import overlay_plan_engine, renderer_plan_engine, subtitle_render_engine, timeline_engine

MODULE_PATH = Path(__file__).resolve().parent / "end_to_end_render_pipeline.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class ProcessResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _video_probe_json(*, duration: float = 10.0) -> dict:
    return {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920,
             "r_frame_rate": "30/1", "avg_frame_rate": "30/1", "duration": str(duration), "pix_fmt": "yuv420p",
             "tags": {}, "disposition": {"default": 1, "attached_pic": 0}},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2,
             "duration": str(duration), "tags": {}, "disposition": {"default": 1, "attached_pic": 0}},
        ],
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": str(duration), "bit_rate": "8000000"},
        "chapters": [],
    }


class FakeRunner:
    """Replaces ffprobe/ffmpeg across every delegated engine
    (media_inspector/video_engine/subtitle_render_engine/
    overlay_render_engine/music_mixer) simultaneously. probes maps a
    resolved absolute path (str) to a probe spec dict. fail_on_ffmpeg_call
    (1-indexed) makes exactly one real ffmpeg call fail, to exercise
    partial-failure/resume behavior deterministically."""

    def __init__(self, probes: dict[str, dict] | None = None, *, fail_on_ffmpeg_call: int | None = None) -> None:
        self.probes = probes or {}
        self.fail_on_ffmpeg_call = fail_on_ffmpeg_call
        self.ffmpeg_calls = 0
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, timeout: int) -> ProcessResult:
        self.calls.append(list(command))
        binary_name = Path(command[0]).name

        if binary_name == "ffprobe":
            path = command[-1]
            spec = self.probes.get(path)
            if spec is None:
                return ProcessResult(returncode=1, stderr=f"no such file: {path}")
            return ProcessResult(returncode=0, stdout=json.dumps(spec))

        self.ffmpeg_calls += 1
        if self.fail_on_ffmpeg_call is not None and self.ffmpeg_calls == self.fail_on_ffmpeg_call:
            return ProcessResult(returncode=1, stderr="simulated ffmpeg failure")

        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"fake-mp4-bytes-{self.ffmpeg_calls}".encode())
        return ProcessResult(returncode=0)


class FakeMediaInfo:
    def __init__(self, has_audio=True, has_video=True, duration_seconds=10.0, is_vertical=True):
        self.has_audio = has_audio
        self.has_video = has_video
        self.duration_seconds = duration_seconds
        self.is_vertical = is_vertical


class FakeValidationResult:
    def __init__(self, passed=True, score=100, summary="PASSED", failed_checks=None, warnings=None):
        self.passed = passed
        self.score = score
        self.summary = summary
        self.failed_checks = failed_checks or []
        self.warnings = warnings or []


def _fake_inspector(**overrides):
    def inspector(path: Path) -> FakeMediaInfo:
        return FakeMediaInfo(**overrides)
    return inspector


def _fake_validator(*, passed=True, failed_checks=None):
    def validator(media_info, profile):
        return FakeValidationResult(passed=passed, failed_checks=failed_checks or [])
    return validator


class EndToEndRenderTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = e2e.load_end_to_end_render_config()
        self.overlay_config = overlay_plan_engine.load_overlay_plan_config()
        self.renderer_plan_config = renderer_plan_engine.load_renderer_plan_config()

    def _font(self, name: str = "font.ttf") -> Path:
        path = self.temp_dir / name
        if not path.exists():
            path.write_bytes(b"fake-font-bytes")
        return path

    def _subtitle_render_config(self, **overrides) -> subtitle_render_engine.SubtitleRenderConfig:
        font = self._font()
        base = subtitle_render_engine.SubtitleRenderConfig(font_family_map={"Body": str(font)}, diagnostics_write_log=False)
        return dataclasses.replace(base, **overrides)

    def _video_clip(self, clip_id="clip_1", *, start=0.0, end=10.0, source_path=None) -> timeline_engine.VideoClip:
        path = source_path or (self.temp_dir / "scene.mp4")
        if source_path is None and not path.exists():
            path.write_bytes(b"x" * 16)
        return timeline_engine.VideoClip(
            clip_id=clip_id, track_id="track_video", source_path=str(path),
            start=start, end=end, duration_seconds=end - start, source_out=end - start, scene_number=1,
        )

    def _subtitle_clip(self, clip_id="sub_1", *, start=0.0, end=2.0, content="hello there") -> timeline_engine.OverlayClip:
        return timeline_engine.OverlayClip(
            clip_id=clip_id, track_id="track_subtitles", source_path="", overlay_type="subtitle",
            start=start, end=end, duration_seconds=end - start, source_out=end - start, content=content,
            metadata={"position": {"x": 540, "y": 1450}, "style": {"style_id": "default", "font_family": "Body"}, "language": "en"},
        )

    def _build_timeline(self, *, with_subtitle=False, duration=10.0, video_path=None, timeline_id="tl1") -> tuple[timeline_engine.Timeline, Path]:
        video_path = video_path or (self.temp_dir / "scene.mp4")
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
                    clips=[self._subtitle_clip()],
                )
            )
        timeline = timeline_engine.Timeline(timeline_id=timeline_id, duration_seconds=duration, tracks=tracks)
        return timeline, video_path

    def _build_chain(self, *, with_subtitle=False, duration=10.0, suffix="", timeline_id="tl1") -> dict:
        timeline, video_path = self._build_timeline(with_subtitle=with_subtitle, duration=duration, timeline_id=timeline_id)
        timeline_path = self.temp_dir / f"timeline{suffix}.json"
        timeline_engine.save_timeline(timeline, timeline_path)

        overlay_plan = overlay_plan_engine.build_overlay_plan(timeline_path, self.overlay_config)
        overlay_plan_path = self.temp_dir / f"overlay_plan{suffix}.json"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path)

        renderer_plan = renderer_plan_engine.build_renderer_plan(timeline_path, overlay_plan_path, self.renderer_plan_config)
        renderer_plan_path = self.temp_dir / f"renderer_plan{suffix}.json"
        renderer_plan_engine.save_renderer_plan(renderer_plan, renderer_plan_path)

        probes = {str(video_path.resolve()): _video_probe_json(duration=duration)}

        return {
            "timeline": timeline, "timeline_path": timeline_path,
            "overlay_plan": overlay_plan, "overlay_plan_path": overlay_plan_path,
            "renderer_plan": renderer_plan, "renderer_plan_path": renderer_plan_path,
            "video_path": video_path, "probes": probes,
        }

    def _request(self, chain, **overrides) -> e2e.EndToEndRenderRequest:
        defaults = dict(
            timeline_path=chain["timeline_path"], renderer_plan_path=chain["renderer_plan_path"],
            output_path=self.temp_dir / "reel_final.mp4", overlay_plan_path=chain["overlay_plan_path"],
            dry_run=False, workspace_path=self.temp_dir / "workspace",
        )
        defaults.update(overrides)
        return e2e.EndToEndRenderRequest(**defaults)

    def _run(self, chain, request=None, *, runner=None, **kwargs):
        request = request or self._request(chain)
        runner = runner or FakeRunner(chain["probes"])
        kwargs.setdefault("subtitle_render_config", self._subtitle_render_config())
        kwargs.setdefault("final_inspector", _fake_inspector())
        kwargs.setdefault("final_validator", _fake_validator())
        return e2e.run_end_to_end_render(request, self.config, runner=runner, **kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationTests(unittest.TestCase):
    def test_default_config_loads(self):
        config = e2e.load_end_to_end_render_config()
        self.assertEqual(config.supported_pass_order, ("video", "subtitle", "overlay", "music", "final_encode"))
        self.assertFalse(config.allow_duplicate_pass_types)
        self.assertTrue(config.fail_on_unsupported_pass)

    def test_workspace_settings_load(self):
        config = e2e.load_end_to_end_render_config()
        self.assertEqual(config.workspace_default_root, "output/render_workspaces")
        self.assertEqual(config.workspace_ownership_marker, ".ai_influencer_render_workspace")
        self.assertEqual(config.workspace_intermediates_directory, "intermediates")

    def test_resume_policy_loads(self):
        config = e2e.load_end_to_end_render_config()
        self.assertTrue(config.resume_enabled)
        self.assertTrue(config.resume_require_checksum_match)
        self.assertTrue(config.resume_require_command_fingerprint_match)

    def test_promotion_policy_loads(self):
        config = e2e.load_end_to_end_render_config()
        self.assertTrue(config.promotion_atomic_required)
        self.assertFalse(config.promotion_allow_cross_device_copy)

    def test_verification_policy_loads(self):
        config = e2e.load_end_to_end_render_config()
        self.assertEqual(config.verification_validator_profile, "instagram_reel")
        self.assertAlmostEqual(config.verification_duration_tolerance_seconds, 0.25)

    def test_missing_config_file_raises(self):
        with self.assertRaises(e2e.EndToEndRenderConfigError):
            e2e.load_end_to_end_render_config("/nonexistent/end_to_end_render.yaml")

    def test_empty_config_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(e2e.EndToEndRenderConfigError):
                e2e.load_end_to_end_render_config(path)

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("pipeline: [unclosed", encoding="utf-8")
            with self.assertRaises(e2e.EndToEndRenderConfigError):
                e2e.load_end_to_end_render_config(path)

    def test_custom_config_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.yaml"
            path.write_text(
                "resume:\n  enabled: false\nverification:\n  duration_tolerance_seconds: 1.0\n", encoding="utf-8",
            )
            config = e2e.load_end_to_end_render_config(path)
            self.assertFalse(config.resume_enabled)
            self.assertAlmostEqual(config.verification_duration_tolerance_seconds, 1.0)

    def test_default_config_path_resolves_under_runtime_root(self):
        path = e2e.default_end_to_end_render_config_path()
        self.assertTrue(str(path).endswith("config/video/end_to_end_render.yaml"))


# ---------------------------------------------------------------------------
# Request / input validation
# ---------------------------------------------------------------------------


class RequestValidationTests(EndToEndRenderTempTestCase):
    def test_valid_request_succeeds(self):
        chain = self._build_chain()
        result = self._run(chain)
        self.assertEqual(result.status, "promoted")

    def test_missing_timeline_raises(self):
        chain = self._build_chain()
        request = self._request(chain, renderer_plan_path=chain["renderer_plan_path"])
        request.timeline_path.unlink()
        with self.assertRaises(e2e.EndToEndRenderError):
            self._run(chain, request)

    def test_missing_renderer_plan_raises(self):
        chain = self._build_chain()
        request = self._request(chain)
        chain["renderer_plan_path"].unlink()
        with self.assertRaises(e2e.EndToEndRenderRequestError):
            self._run(chain, request)

    def test_malformed_renderer_plan_json_raises(self):
        chain = self._build_chain()
        chain["renderer_plan_path"].write_text("{not valid json")
        with self.assertRaises(e2e.EndToEndRenderRequestError):
            self._run(chain)

    def test_overlay_plan_required_when_referenced(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain, overlay_plan_path=None)
        with self.assertRaises(e2e.EndToEndRenderRequestError):
            self._run(chain, request)

    def test_overlay_asset_manifest_optional_when_no_overlay_pass(self):
        chain = self._build_chain()
        result = self._run(chain)  # no overlay_asset_manifest_path given
        self.assertEqual(result.status, "promoted")

    def test_output_same_as_timeline_rejected(self):
        chain = self._build_chain()
        request = self._request(chain, output_path=chain["timeline_path"])
        with self.assertRaises(e2e.UnsafeFinalOutputError):
            self._run(chain, request)

    def test_output_same_as_renderer_plan_rejected(self):
        chain = self._build_chain()
        request = self._request(chain, output_path=chain["renderer_plan_path"])
        with self.assertRaises(e2e.UnsafeFinalOutputError):
            self._run(chain, request)

    def test_existing_output_refused_without_force(self):
        chain = self._build_chain()
        output_path = self.temp_dir / "reel_final.mp4"
        output_path.write_bytes(b"existing")
        request = self._request(chain, output_path=output_path)
        with self.assertRaises(e2e.FinalOutputExistsError):
            self._run(chain, request)

    def test_existing_output_succeeds_with_force(self):
        chain = self._build_chain()
        output_path = self.temp_dir / "reel_final.mp4"
        output_path.write_bytes(b"existing")
        request = self._request(chain, output_path=output_path, force=True)
        result = self._run(chain, request)
        self.assertEqual(result.status, "promoted")

    def test_resume_and_restart_mutually_exclusive(self):
        chain = self._build_chain()
        request = self._request(chain, resume=True, restart=True)
        with self.assertRaises(e2e.EndToEndRenderRequestError):
            self._run(chain, request)

    def test_dry_run_performs_no_media_writes(self):
        chain = self._build_chain()
        request = self._request(chain, dry_run=True)
        result = self._run(chain, request)
        self.assertFalse(request.output_path.exists())
        self.assertIsNone(result.final_output)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class WorkspaceTests(EndToEndRenderTempTestCase):
    def test_deterministic_workspace_id(self):
        chain = self._build_chain()
        request_a = self._request(chain, output_path=self.temp_dir / "out_a.mp4", workspace_path=None)
        request_b = self._request(chain, output_path=self.temp_dir / "out_a.mp4", workspace_path=None)
        pid_a = e2e._compute_pipeline_id(
            chain["renderer_plan"].renderer_plan_id, chain["renderer_plan"].timeline_id,
            chain["renderer_plan"].overlay_plan_id, None, str(request_a.output_path.resolve()), self.config.schema_version,
        )
        pid_b = e2e._compute_pipeline_id(
            chain["renderer_plan"].renderer_plan_id, chain["renderer_plan"].timeline_id,
            chain["renderer_plan"].overlay_plan_id, None, str(request_b.output_path.resolve()), self.config.schema_version,
        )
        self.assertEqual(pid_a, pid_b)

    def test_deterministic_paths_layout(self):
        chain = self._build_chain()
        result = self._run(chain)
        workspace = e2e._resolve_workspace(
            e2e._normalize_request(self._request(chain)), self.config, result.pipeline_id,
        )
        self.assertEqual(workspace.intermediates_dir.name, "intermediates")
        self.assertEqual(workspace.final_dir.name, "final")
        self.assertEqual(workspace.logs_dir.name, "logs")
        self.assertEqual(workspace.diagnostics_dir.name, "diagnostics")

    def test_ownership_marker_created(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)
        marker = request.workspace_path / self.config.workspace_ownership_marker
        self.assertTrue(marker.is_file())
        data = json.loads(marker.read_text())
        self.assertIn("pipeline_id", data)

    def test_unsafe_workspace_path_is_file_rejected(self):
        chain = self._build_chain()
        blocker = self.temp_dir / "blocker_workspace"
        blocker.write_bytes(b"not a directory")
        request = self._request(chain, workspace_path=blocker)
        with self.assertRaises(e2e.UnsafeRenderWorkspaceError):
            self._run(chain, request)

    def test_existing_foreign_workspace_rejected(self):
        chain = self._build_chain()
        foreign = self.temp_dir / "foreign_workspace"
        foreign.mkdir()
        (foreign / "some_file.txt").write_text("not ours")
        request = self._request(chain, workspace_path=foreign)
        with self.assertRaises(e2e.RenderWorkspaceOwnershipError):
            self._run(chain, request)

    def test_existing_owned_workspace_without_resume_or_restart_rejected(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)  # creates+owns the workspace, promotes
        request2 = self._request(chain, output_path=self.temp_dir / "reel_final_2.mp4")
        # second call reuses the SAME workspace_path but neither resume
        # nor restart is set
        with self.assertRaises(e2e.RenderResumeError):
            self._run(chain, request2)

    def test_workspace_parent_created(self):
        chain = self._build_chain()
        nested = self.temp_dir / "a" / "b" / "c" / "workspace"
        request = self._request(chain, workspace_path=nested)
        self._run(chain, request)
        self.assertTrue(nested.is_dir())

    def test_default_workspace_root_under_runtime_root_when_not_overridden(self):
        chain = self._build_chain()
        request = self._request(chain, workspace_path=None)
        workspace = e2e._resolve_workspace(e2e._normalize_request(request), self.config, "abc123")
        self.assertIn("output/render_workspaces", str(workspace.workspace_path))
        self.assertIn("abc123", str(workspace.workspace_path))


# ---------------------------------------------------------------------------
# Pass order
# ---------------------------------------------------------------------------


class PassOrderTests(EndToEndRenderTempTestCase):
    def test_full_chain_video_subtitle(self):
        chain = self._build_chain(with_subtitle=True)
        result = self._run(chain)
        self.assertEqual([c.pass_type for c in result.passes], ["video", "subtitle"])

    def test_video_only_chain(self):
        chain = self._build_chain()
        result = self._run(chain)
        self.assertEqual([c.pass_type for c in result.passes], ["video"])

    def test_unsupported_pass_type_fails_before_write(self):
        chain = self._build_chain()
        config = dataclasses.replace(self.config, supported_pass_order=("video",), fail_on_unsupported_pass=True)
        request = self._request(chain)
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(e2e.RenderPassOrderError):
            e2e.run_end_to_end_render(
                request, config, runner=runner, subtitle_render_config=self._subtitle_render_config(),
            )
        self.assertEqual(runner.ffmpeg_calls, 0)

    def test_duplicate_pass_type_fails(self):
        chain = self._build_chain()
        plan = chain["renderer_plan"]
        duplicate_pass = dataclasses.replace(plan.passes[0], pass_id="pass_video_dup", outputs=["video_dup_output"])
        plan.passes.append(duplicate_pass)
        renderer_plan_engine.save_renderer_plan(plan, chain["renderer_plan_path"], force=True)
        with self.assertRaises(e2e.RenderPassOrderError):
            self._run(chain)

    def test_duplicate_pass_type_allowed_by_config(self):
        chain = self._build_chain()
        plan = chain["renderer_plan"]
        duplicate_pass = dataclasses.replace(plan.passes[0], pass_id="pass_video_dup", outputs=["video_dup_output"])
        plan.passes.append(duplicate_pass)
        renderer_plan_engine.save_renderer_plan(plan, chain["renderer_plan_path"], force=True)
        config = dataclasses.replace(self.config, allow_duplicate_pass_types=True)
        request = self._request(chain)
        runner = FakeRunner(chain["probes"])
        # still fails ordering (dup video after final_encode reorders
        # nothing, but two "video" entries violate the strict
        # subsequence-index check) -- config only disables the
        # duplicate-detection error, ordering is still enforced.
        with self.assertRaises(e2e.RenderPassOrderError):
            e2e.run_end_to_end_render(
                request, config, runner=runner, subtitle_render_config=self._subtitle_render_config(),
            )

    def test_unsupported_subtitle_hint_fails_before_write(self):
        custom_plan_config = dataclasses.replace(self.renderer_plan_config, subtitle_hint="copy")
        timeline, video_path = self._build_timeline(with_subtitle=True)
        timeline_path = self.temp_dir / "timeline.json"
        timeline_engine.save_timeline(timeline, timeline_path)
        overlay_plan = overlay_plan_engine.build_overlay_plan(timeline_path, self.overlay_config)
        overlay_plan_path = self.temp_dir / "overlay_plan.json"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path)
        renderer_plan = renderer_plan_engine.build_renderer_plan(timeline_path, overlay_plan_path, custom_plan_config)
        renderer_plan_path = self.temp_dir / "renderer_plan.json"
        renderer_plan_engine.save_renderer_plan(renderer_plan, renderer_plan_path)

        probes = {str(video_path.resolve()): _video_probe_json()}
        request = e2e.EndToEndRenderRequest(
            timeline_path=timeline_path, renderer_plan_path=renderer_plan_path,
            output_path=self.temp_dir / "reel_final.mp4", overlay_plan_path=overlay_plan_path,
            dry_run=False, workspace_path=self.temp_dir / "workspace",
        )
        runner = FakeRunner(probes)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner, subtitle_render_config=self._subtitle_render_config())
        self.assertEqual(runner.ffmpeg_calls, 0)
        self.assertFalse(request.output_path.exists())


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecutionTests(EndToEndRenderTempTestCase):
    def test_all_passes_succeed(self):
        chain = self._build_chain(with_subtitle=True)
        result = self._run(chain)
        self.assertTrue(all(c.status == "succeeded" for c in result.passes))

    def test_output_chaining_correct(self):
        chain = self._build_chain(with_subtitle=True)
        self._run(chain)

    def test_deterministic_intermediate_names(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        result = self._run(chain, request)
        video_checkpoint = next(c for c in result.passes if c.pass_type == "video")
        self.assertIn(self.config.intermediate_video_filename, video_checkpoint.output_path)

    def test_dry_run_writes_no_media(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain, dry_run=True)
        self._run(chain, request)
        self.assertFalse((request.workspace_path / "intermediates" / self.config.intermediate_video_filename).exists())

    def test_runner_failure_propagates(self):
        chain = self._build_chain()
        request = self._request(chain)
        runner = FakeRunner(chain["probes"], fail_on_ffmpeg_call=1)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner, subtitle_render_config=self._subtitle_render_config())

    def test_subtitle_failure_stops_before_promotion(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner, subtitle_render_config=self._subtitle_render_config())
        self.assertFalse(request.output_path.exists())

    def test_final_encode_never_dispatched_as_a_real_pass(self):
        # final_encode is handled entirely by _verify_candidate(), never
        # by execute_single_pass() -- no checkpoint should ever exist
        # for it.
        chain = self._build_chain()
        result = self._run(chain)
        self.assertNotIn("final_encode", [c.pass_type for c in result.passes])


# ---------------------------------------------------------------------------
# Partial output protection
# ---------------------------------------------------------------------------


class PartialProtectionTests(EndToEndRenderTempTestCase):
    def test_requested_final_untouched_until_promotion(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner, subtitle_render_config=self._subtitle_render_config())
        self.assertFalse(request.output_path.exists())

    def test_failure_creates_no_final(self):
        chain = self._build_chain()
        request = self._request(chain)
        runner = FakeRunner(chain["probes"], fail_on_ffmpeg_call=1)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner, subtitle_render_config=self._subtitle_render_config())
        self.assertFalse(request.output_path.exists())

    def test_existing_final_unchanged_on_failure(self):
        chain = self._build_chain(with_subtitle=True)
        output_path = self.temp_dir / "reel_final.mp4"
        output_path.write_bytes(b"pre-existing-final-content")
        before_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
        request = self._request(chain, output_path=output_path, force=True)
        runner = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner, subtitle_render_config=self._subtitle_render_config())
        self.assertEqual(hashlib.sha256(output_path.read_bytes()).hexdigest(), before_hash)

    def test_no_pass_writes_directly_to_final_path(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        result = self._run(chain, request)
        for checkpoint in result.passes:
            self.assertNotEqual(checkpoint.output_path, str(request.output_path))


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


class ResumeTests(EndToEndRenderTempTestCase):
    def test_valid_video_reused_after_subtitle_failure(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner1 = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner1, subtitle_render_config=self._subtitle_render_config())

        resume_request = self._request(chain, resume=True)
        runner2 = FakeRunner(chain["probes"])
        result = self._run(chain, resume_request, runner=runner2)
        self.assertEqual(result.status, "promoted")
        self.assertEqual(result.reused_pass_count, 1)
        self.assertEqual(result.executed_pass_count, 1)
        video_checkpoint = next(c for c in result.passes if c.pass_type == "video")
        self.assertTrue(video_checkpoint.reusable)
        self.assertEqual(runner2.ffmpeg_calls, 1)  # only subtitle re-rendered

    def test_all_passes_reused_when_nothing_changed(self):
        # Both passes execute successfully in run 1, but a verification
        # failure blocks promotion -- output_path never gets created, so
        # a resume run can reuse EVERY pass and just retry verification.
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        with self.assertRaises(e2e.FinalCandidateVerificationError):
            self._run(chain, request, final_inspector=_fake_inspector(has_video=False))
        self.assertFalse(request.output_path.exists())

        resume_request = self._request(chain, resume=True)
        runner2 = FakeRunner(chain["probes"])
        result = self._run(chain, resume_request, runner=runner2)
        self.assertEqual(result.status, "promoted")
        self.assertEqual(result.reused_pass_count, 2)
        self.assertEqual(result.executed_pass_count, 0)
        self.assertEqual(runner2.ffmpeg_calls, 0)

    def test_changed_source_invalidates_pass(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner1 = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner1, subtitle_render_config=self._subtitle_render_config())

        # change the source video content -- video pass's dry-run
        # fingerprint depends on probe data, which we change here.
        chain["video_path"].write_bytes(b"y" * 32)
        changed_probes = dict(chain["probes"])
        changed_probes[str(chain["video_path"].resolve())] = _video_probe_json(duration=12.0)

        resume_request = self._request(chain, resume=True)
        runner2 = FakeRunner(changed_probes)
        result = self._run(chain, resume_request, runner=runner2)
        # video pass invalidated (probe/duration changed) -> both passes re-executed
        self.assertEqual(result.reused_pass_count, 0)
        self.assertEqual(result.executed_pass_count, 2)

    def test_missing_intermediate_invalidates(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner1 = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner1, subtitle_render_config=self._subtitle_render_config())

        video_intermediate = request.workspace_path / "intermediates" / self.config.intermediate_video_filename
        video_intermediate.unlink()

        resume_request = self._request(chain, resume=True)
        runner2 = FakeRunner(chain["probes"])
        result = self._run(chain, resume_request, runner=runner2)
        self.assertEqual(result.reused_pass_count, 0)
        self.assertEqual(result.executed_pass_count, 2)

    def test_zero_byte_intermediate_invalidates(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner1 = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner1, subtitle_render_config=self._subtitle_render_config())

        video_intermediate = request.workspace_path / "intermediates" / self.config.intermediate_video_filename
        video_intermediate.write_bytes(b"")

        resume_request = self._request(chain, resume=True)
        runner2 = FakeRunner(chain["probes"])
        result = self._run(chain, resume_request, runner=runner2)
        self.assertEqual(result.reused_pass_count, 0)

    def test_checksum_mismatch_invalidates(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner1 = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner1, subtitle_render_config=self._subtitle_render_config())

        video_intermediate = request.workspace_path / "intermediates" / self.config.intermediate_video_filename
        video_intermediate.write_bytes(b"tampered-content-different-checksum")

        resume_request = self._request(chain, resume=True)
        runner2 = FakeRunner(chain["probes"])
        result = self._run(chain, resume_request, runner=runner2)
        self.assertEqual(result.reused_pass_count, 0)

    def test_invalid_state_fails_safely(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)  # creates workspace + state.json

        state_path = request.workspace_path / self.config.workspace_state_filename
        state_path.write_text("{not valid json", encoding="utf-8")

        resume_request = self._request(chain, output_path=self.temp_dir / "reel_final_2.mp4", resume=True)
        with self.assertRaises(e2e.RenderStateError):
            self._run(chain, resume_request)

    def test_resume_identity_mismatch_fails(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)

        other_chain = self._build_chain(suffix="_other", timeline_id="tl_different")
        resume_request = e2e.EndToEndRenderRequest(
            timeline_path=other_chain["timeline_path"], renderer_plan_path=other_chain["renderer_plan_path"],
            output_path=request.output_path, overlay_plan_path=other_chain["overlay_plan_path"],
            dry_run=False, workspace_path=request.workspace_path, resume=True, force=True,
        )
        with self.assertRaises(e2e.RenderResumeError):
            self._run(other_chain, resume_request)

    def test_no_reuse_based_only_on_filename_existence(self):
        # A file sitting at the expected intermediate path with no
        # matching checkpoint in state.json must never be reused.
        chain = self._build_chain()
        request = self._request(chain, restart=False)
        workspace_path = request.workspace_path
        workspace_path.mkdir(parents=True)
        (workspace_path / self.config.workspace_ownership_marker).write_text(json.dumps({"pipeline_id": "x"}), encoding="utf-8")
        (workspace_path / "intermediates").mkdir()
        fake_stray_file = workspace_path / "intermediates" / self.config.intermediate_candidate_filename
        fake_stray_file.write_bytes(b"stray-unverified-content")

        resume_request = self._request(chain, resume=True)
        runner = FakeRunner(chain["probes"])
        result = self._run(chain, resume_request, runner=runner)
        # no state.json existed, so nothing is "resumable" -- pipeline
        # must execute for real, not adopt the stray file.
        self.assertEqual(result.reused_pass_count, 0)
        self.assertEqual(result.executed_pass_count, 1)


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------


class RestartTests(EndToEndRenderTempTestCase):
    def test_owned_workspace_cleared_safely(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)
        intermediates_dir = request.workspace_path / "intermediates"
        self.assertTrue(any(intermediates_dir.iterdir())) if intermediates_dir.exists() else None

        restart_request = self._request(chain, output_path=self.temp_dir / "reel_final_2.mp4", restart=True)
        result = self._run(chain, restart_request)
        self.assertEqual(result.status, "promoted")
        self.assertEqual(result.reused_pass_count, 0)

    def test_foreign_workspace_untouched_by_restart(self):
        # _open_or_create_workspace() itself refuses a directory with no
        # ownership marker before --restart ever gets a chance to run --
        # the earliest possible, most specific rejection.
        chain = self._build_chain()
        foreign = self.temp_dir / "foreign"
        foreign.mkdir()
        (foreign / "important.txt").write_text("do not touch", encoding="utf-8")
        restart_request = self._request(chain, workspace_path=foreign, restart=True)
        with self.assertRaises(e2e.RenderWorkspaceOwnershipError):
            self._run(chain, restart_request)
        self.assertTrue((foreign / "important.txt").is_file())

    def test_source_files_untouched_by_restart(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)
        before_hash = hashlib.sha256(chain["timeline_path"].read_bytes()).hexdigest()

        restart_request = self._request(chain, output_path=self.temp_dir / "reel_final_2.mp4", restart=True)
        self._run(chain, restart_request)
        self.assertEqual(hashlib.sha256(chain["timeline_path"].read_bytes()).hexdigest(), before_hash)

    def test_existing_promoted_final_untouched_by_restart_alone(self):
        chain = self._build_chain()
        request = self._request(chain)
        result1 = self._run(chain, request)
        before_hash = hashlib.sha256(Path(result1.final_output).read_bytes()).hexdigest()

        # restart the SAME workspace but target a DIFFERENT output --
        # the original promoted final must remain untouched.
        restart_request = self._request(chain, output_path=self.temp_dir / "reel_final_new.mp4", restart=True)
        self._run(chain, restart_request)
        self.assertEqual(hashlib.sha256(Path(result1.final_output).read_bytes()).hexdigest(), before_hash)

    def test_all_checkpoints_reset_after_restart(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)

        restart_request = self._request(chain, output_path=self.temp_dir / "reel_final_2.mp4", restart=True)
        result = self._run(chain, restart_request)
        self.assertTrue(all(not c.reusable for c in result.passes))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class VerificationTests(EndToEndRenderTempTestCase):
    def test_valid_final_passes(self):
        chain = self._build_chain()
        result = self._run(chain)
        self.assertTrue(result.final_verified)

    def test_missing_video_stream_fails(self):
        chain = self._build_chain()
        request = self._request(chain)
        with self.assertRaises(e2e.FinalCandidateVerificationError):
            self._run(chain, request, final_inspector=_fake_inspector(has_video=False))

    def test_duration_mismatch_fails(self):
        chain = self._build_chain()
        request = self._request(chain)
        with self.assertRaises(e2e.FinalCandidateVerificationError):
            self._run(chain, request, final_inspector=_fake_inspector(duration_seconds=999.0))

    def test_wrong_orientation_fails(self):
        chain = self._build_chain()
        request = self._request(chain)
        with self.assertRaises(e2e.FinalCandidateVerificationError):
            self._run(chain, request, final_inspector=_fake_inspector(is_vertical=False))

    def test_hard_validator_failure_fails(self):
        chain = self._build_chain()
        request = self._request(chain)
        with self.assertRaises(e2e.FinalCandidateVerificationError):
            self._run(chain, request, final_validator=_fake_validator(passed=False))

    def test_planned_audio_missing_fails(self):
        chain = self._build_chain()
        # Inject a fake (structurally valid -- non-empty asset_id, a
        # matching RendererAsset) music track onto the plan so audio
        # becomes "planned" per _verify_candidate()'s own check, without
        # adding an actual "music" pass_type to plan.passes.
        plan = chain["renderer_plan"]
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"z" * 16)
        plan.assets.append(
            renderer_plan_engine.RendererAsset(
                asset_id="asset_music_0001", asset_type=renderer_plan_engine.AssetType.MUSIC, source_path=str(music_path),
            )
        )
        plan.music_tracks.append(renderer_plan_engine.RendererMusicTrack(track_id="music_1", asset_id="asset_music_0001"))
        renderer_plan_engine.save_renderer_plan(plan, chain["renderer_plan_path"], force=True)
        request = self._request(chain)
        with self.assertRaises(e2e.FinalCandidateVerificationError):
            self._run(chain, request, final_inspector=_fake_inspector(has_audio=False))

    def test_verification_disabled_skips_checks(self):
        chain = self._build_chain()
        config = dataclasses.replace(self.config, verification_enabled=False)
        request = self._request(chain)
        runner = FakeRunner(chain["probes"])
        result = e2e.run_end_to_end_render(
            request, config, runner=runner, subtitle_render_config=self._subtitle_render_config(),
            final_inspector=_fake_inspector(has_video=False),  # would normally fail
        )
        self.assertEqual(result.status, "promoted")

    def test_source_hash_unchanged_after_verification(self):
        chain = self._build_chain()
        before_hash = hashlib.sha256(chain["video_path"].read_bytes()).hexdigest()
        self._run(chain)
        self.assertEqual(hashlib.sha256(chain["video_path"].read_bytes()).hexdigest(), before_hash)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


class PromotionTests(EndToEndRenderTempTestCase):
    def test_atomic_replace_strategy(self):
        chain = self._build_chain()
        result = self._run(chain)
        self.assertEqual(result.promotion.strategy, "atomic_replace")
        self.assertTrue(result.promotion.promoted)

    def test_checksum_match_after_promotion(self):
        chain = self._build_chain()
        result = self._run(chain)
        self.assertEqual(result.promotion.candidate_checksum, result.promotion.final_checksum)

    def test_final_path_correct(self):
        chain = self._build_chain()
        request = self._request(chain)
        result = self._run(chain, request)
        self.assertEqual(result.promotion.final_path, str(request.output_path))
        self.assertEqual(result.final_output, str(request.output_path))

    def test_promotion_creates_output_parent(self):
        chain = self._build_chain()
        nested_output = self.temp_dir / "a" / "b" / "reel_final.mp4"
        request = self._request(chain, output_path=nested_output)
        self._run(chain, request)
        self.assertTrue(nested_output.is_file())


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class CleanupTests(EndToEndRenderTempTestCase):
    def test_default_removes_intermediates_after_success(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        result = self._run(chain, request)
        self.assertTrue(len(result.cleanup.deleted) >= 1)
        for deleted_path in result.cleanup.deleted:
            self.assertFalse(Path(deleted_path).exists())

    def test_keep_intermediates_retains_everything(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain, keep_intermediates=True)
        result = self._run(chain, request)
        self.assertEqual(result.cleanup.deleted, [])
        self.assertTrue(len(result.cleanup.retained) >= 1)

    def test_failure_preserves_workspace(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner, subtitle_render_config=self._subtitle_render_config())
        video_intermediate = request.workspace_path / "intermediates" / self.config.intermediate_video_filename
        self.assertTrue(video_intermediate.is_file())

    def test_only_owned_files_deleted(self):
        chain = self._build_chain()
        request = self._request(chain)
        result = self._run(chain, request)
        # cleanup only ever iterates checkpoint output_paths -- never
        # an arbitrary directory walk.
        self.assertTrue(result.cleanup.ownership_verified)

    def test_reports_retained_after_cleanup(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)
        self.assertTrue((request.workspace_path / "logs" / self.config.reporting_detailed_log_filename).is_file())
        self.assertTrue((request.workspace_path / "diagnostics" / self.config.reporting_report_filename).is_file())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class ReportingTests(EndToEndRenderTempTestCase):
    def test_detailed_log_written(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)
        log_path = request.workspace_path / "logs" / self.config.reporting_detailed_log_filename
        self.assertTrue(log_path.is_file())
        data = json.loads(log_path.read_text())
        self.assertIn("passes", data)

    def test_high_level_report_written(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)
        report_path = request.workspace_path / "diagnostics" / self.config.reporting_report_filename
        self.assertTrue(report_path.is_file())
        data = json.loads(report_path.read_text())
        for field_name in ("pipeline_id", "renderer_plan_id", "status", "passes", "promotion", "cleanup"):
            self.assertIn(field_name, data)

    def test_atomic_json_no_tmp_left_behind(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)
        report_path = request.workspace_path / "diagnostics" / self.config.reporting_report_filename
        self.assertFalse(Path(str(report_path) + ".tmp").exists())

    def test_pass_summaries_correct(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        result = self._run(chain, request)
        report_path = request.workspace_path / "diagnostics" / self.config.reporting_report_filename
        data = json.loads(report_path.read_text())
        self.assertEqual(len(data["passes"]), len(result.passes))

    def test_reused_and_executed_counts_correct(self):
        chain = self._build_chain(with_subtitle=True)
        request = self._request(chain)
        runner1 = FakeRunner(chain["probes"], fail_on_ffmpeg_call=2)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner1, subtitle_render_config=self._subtitle_render_config())
        resume_request = self._request(chain, resume=True)
        result = self._run(chain, resume_request, runner=FakeRunner(chain["probes"]))
        self.assertEqual(result.reused_pass_count, 1)
        self.assertEqual(result.executed_pass_count, 1)

    def test_error_report_correct_on_failure(self):
        chain = self._build_chain()
        request = self._request(chain)
        runner = FakeRunner(chain["probes"], fail_on_ffmpeg_call=1)
        with self.assertRaises(e2e.RenderPassExecutionError):
            e2e.run_end_to_end_render(request, self.config, runner=runner, subtitle_render_config=self._subtitle_render_config())
        report_path = request.workspace_path / "diagnostics" / self.config.reporting_report_filename
        data = json.loads(report_path.read_text())
        self.assertEqual(data["status"], "failed")
        self.assertIsNotNone(data["error"])

    def test_result_round_trips_json_serializable(self):
        chain = self._build_chain()
        result = self._run(chain)
        data = e2e.end_to_end_render_result_to_dict(result)
        json.dumps(data)  # must not raise

    def test_no_secrets_or_publishing_data_in_report(self):
        chain = self._build_chain()
        request = self._request(chain)
        self._run(chain, request)
        report_path = request.workspace_path / "diagnostics" / self.config.reporting_report_filename
        content = report_path.read_text()
        for forbidden in ("password", "api_key", "instagram", "browser_profile", "token"):
            self.assertNotIn(forbidden, content.lower())


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityTests(EndToEndRenderTempTestCase):
    def test_stable_pipeline_id_across_runs(self):
        args = ("rp1", "tl1", "op1", None, "/tmp/out.mp4", "1.0")
        self.assertEqual(e2e._compute_pipeline_id(*args), e2e._compute_pipeline_id(*args))

    def test_timestamps_excluded_from_pipeline_id(self):
        # _compute_pipeline_id() takes no timestamp/PID/hostname
        # parameter at all -- structurally impossible to include them.
        import inspect
        signature = inspect.signature(e2e._compute_pipeline_id)
        for forbidden in ("started_at", "timestamp", "pid", "hostname"):
            self.assertNotIn(forbidden, signature.parameters)

    def test_renderer_plan_change_changes_pipeline_id(self):
        id_a = e2e._compute_pipeline_id("rp1", "tl1", "op1", None, "/tmp/out.mp4", "1.0")
        id_b = e2e._compute_pipeline_id("rp2", "tl1", "op1", None, "/tmp/out.mp4", "1.0")
        self.assertNotEqual(id_a, id_b)

    def test_overlay_manifest_change_changes_pipeline_id(self):
        id_a = e2e._compute_pipeline_id("rp1", "tl1", "op1", "manifest_a", "/tmp/out.mp4", "1.0")
        id_b = e2e._compute_pipeline_id("rp1", "tl1", "op1", "manifest_b", "/tmp/out.mp4", "1.0")
        self.assertNotEqual(id_a, id_b)

    def test_output_target_change_changes_pipeline_id(self):
        id_a = e2e._compute_pipeline_id("rp1", "tl1", "op1", None, "/tmp/out_a.mp4", "1.0")
        id_b = e2e._compute_pipeline_id("rp1", "tl1", "op1", None, "/tmp/out_b.mp4", "1.0")
        self.assertNotEqual(id_a, id_b)

    def test_stable_command_fingerprint(self):
        fp_a = e2e._compute_command_fingerprint("p1", "video", ["ffmpeg", "-i", "a.mp4"], "1.0")
        fp_b = e2e._compute_command_fingerprint("p1", "video", ["ffmpeg", "-i", "a.mp4"], "1.0")
        self.assertEqual(fp_a, fp_b)

    def test_command_fingerprint_changes_with_command(self):
        fp_a = e2e._compute_command_fingerprint("p1", "video", ["ffmpeg", "-i", "a.mp4"], "1.0")
        fp_b = e2e._compute_command_fingerprint("p1", "video", ["ffmpeg", "-i", "b.mp4"], "1.0")
        self.assertNotEqual(fp_a, fp_b)

    def test_pipeline_id_stable_across_two_full_runs(self):
        chain = self._build_chain()
        request_a = self._request(chain, output_path=self.temp_dir / "reel.mp4", workspace_path=self.temp_dir / "ws_a", dry_run=True)
        request_b = self._request(chain, output_path=self.temp_dir / "reel.mp4", workspace_path=self.temp_dir / "ws_b", dry_run=True)
        result_a = self._run(chain, request_a)
        result_b = self._run(chain, request_b)
        self.assertEqual(result_a.pipeline_id, result_b.pipeline_id)


# ---------------------------------------------------------------------------
# Structural safety
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def test_no_playwright_or_publish_reference(self):
        # "instagram_reel" is a legitimate video_validator.py profile
        # NAME (config default), not a publishing/social reference --
        # excluded explicitly rather than matched as a bare substring.
        source_without_profile_name = MODULE_SOURCE.replace("instagram_reel", "")
        for forbidden in ("playwright", "Playwright", "instagram", "Instagram", "publish_reel", "upload_reel"):
            self.assertNotIn(forbidden, source_without_profile_name)

    def test_no_instagram_session_import(self):
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from .social"))
            self.assertFalse(stripped.startswith("from src.social"))

    def test_no_direct_ffmpeg_syntax_construction(self):
        for forbidden in ('"-filter_complex"', "'-filter_complex'", '"-vf"', "'-vf'", "f\"drawtext=", "f\"overlay="):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_duplicate_pass_execution_logic(self):
        # This module must never import the engines
        # renderer_execution_engine.py already delegates to -- it only
        # ever reaches them through prepare_execution_context()/
        # execute_single_pass().
        for forbidden in ("video_engine", "subtitle_render_engine", "overlay_render_engine", "music_mixer", "filter_graph_builder", "filter_graph_serializer"):
            for line in MODULE_SOURCE.splitlines():
                stripped = line.strip()
                if stripped.startswith("from . import") or stripped.startswith("import "):
                    self.assertNotIn(forbidden, stripped, f"{forbidden} should not be imported by end_to_end_render_pipeline.py")

    def test_no_network_or_download_code(self):
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_package_installation(self):
        for forbidden in ("pip install", "subprocess.run([\"pip\"", "pip.main"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_unsafe_recursive_cleanup(self):
        self.assertNotIn("shutil.rmtree", MODULE_SOURCE)
        self.assertNotIn("rmtree(", MODULE_SOURCE)

    def test_uses_renderer_execution_engine_seam(self):
        self.assertIn("renderer_execution_engine.prepare_execution_context", MODULE_SOURCE)
        self.assertIn("renderer_execution_engine.execute_single_pass", MODULE_SOURCE)

    def test_uses_video_validator_and_media_inspector_read_only(self):
        self.assertIn("video_validator.validate_media", MODULE_SOURCE)
        self.assertIn("media_inspector.inspect_file", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(EndToEndRenderTempTestCase):
    def test_resume_and_restart_mutually_exclusive_at_cli(self):
        with self.assertRaises(SystemExit):
            e2e.parse_arguments([
                "--timeline", "t.json", "--renderer-plan", "r.json", "--output", "o.mp4", "--resume", "--restart",
            ])

    def test_required_arguments_enforced(self):
        with self.assertRaises(SystemExit):
            e2e.parse_arguments(["--timeline", "t.json"])

    def test_dry_run_flag_parsed(self):
        args = e2e.parse_arguments([
            "--timeline", "t.json", "--renderer-plan", "r.json", "--output", "o.mp4", "--dry-run",
        ])
        self.assertTrue(args.dry_run)

    def test_overlay_plan_optional_at_cli(self):
        args = e2e.parse_arguments([
            "--timeline", "t.json", "--renderer-plan", "r.json", "--output", "o.mp4",
        ])
        self.assertIsNone(args.overlay_plan)

    def test_cli_missing_timeline_fails_cleanly(self):
        import contextlib
        import io

        chain = self._build_chain()
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            e2e.main([
                "--timeline", str(self.temp_dir / "missing.json"), "--renderer-plan", str(chain["renderer_plan_path"]),
                "--overlay-plan", str(chain["overlay_plan_path"]), "--output", str(self.temp_dir / "out.mp4"),
                "--dry-run", "--workspace", str(self.temp_dir / "cli_workspace"),
            ])
        self.assertIn("[EndToEndRenderPipeline]", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
