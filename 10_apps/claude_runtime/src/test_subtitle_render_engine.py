from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from . import filter_graph_builder, filter_graph_serializer
from . import subtitle_render_engine as sre


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class FakeRunner:
    """Injectable stand-in for subtitle_render_engine.default_runner. Never
    touches real ffmpeg. Optionally writes a fake output file so
    verify_subtitle_render_output() sees a real (fake) file."""

    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "",
                 raise_exception: Exception | None = None, write_output: bool = True,
                 output_bytes: bytes = b"fake-rendered-video-bytes"):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raise_exception = raise_exception
        self.write_output = write_output
        self.output_bytes = output_bytes
        self.calls: list[tuple[list[str], int]] = []

    def __call__(self, command: list[str], *, timeout: int) -> sre.ProcessResult:
        self.calls.append((list(command), timeout))
        if self.raise_exception is not None:
            raise self.raise_exception
        if self.write_output and self.returncode == 0:
            Path(command[-1]).write_bytes(self.output_bytes)
        return sre.ProcessResult(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


class FakeInspectorInfo:
    def __init__(self, has_audio: bool = True, has_video: bool = True, duration_seconds: float = 5.0):
        self.has_audio = has_audio
        self.has_video = has_video
        self.duration_seconds = duration_seconds


def _fake_inspector(*, has_audio: bool = True):
    def inspector(path: Path) -> FakeInspectorInfo:
        return FakeInspectorInfo(has_audio=has_audio)
    return inspector


class SubtitleRenderTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="subtitle_render_test_"))
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.config = sre.SubtitleRenderConfig()

    def _video(self, name: str = "video.mp4", content: bytes = b"fake-video-bytes") -> Path:
        path = self.temp_dir / name
        path.write_bytes(content)
        return path

    def _ass(self, name: str = "subs.ass", content: str = "[Script Info]\nScriptType: v4.00+\n") -> Path:
        path = self.temp_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def _font(self, name: str = "font.ttf", content: bytes = b"fake-font-bytes") -> Path:
        path = self.temp_dir / name
        path.write_bytes(content)
        return path

    def _fontsdir(self, name: str = "fonts") -> Path:
        path = self.temp_dir / name
        path.mkdir()
        return path

    def _cue(self, **overrides) -> sre.DrawTextCue:
        defaults = dict(
            text="Hello world",
            start_seconds=0.0,
            end_seconds=2.0,
            font_family="Body",
            x=100.0,
            y=200.0,
        )
        defaults.update(overrides)
        return sre.DrawTextCue(**defaults)

    def _build_serialized(
        self, request: sre.SubtitleRenderRequest, config: sre.SubtitleRenderConfig, *, serializer_config=None,
    ) -> filter_graph_serializer.SerializedFilterGraph:
        """Mirrors sre.build_subtitle_render_plan()'s internal FilterGraph
        pipeline, for tests that exercise build_subtitle_ffmpeg_command()
        directly/in isolation."""
        fg_config = filter_graph_builder.load_filter_graph_config()
        filters = filter_graph_builder.build_subtitle_filter_spec(request, fg_config)
        if request.mode == sre.SubtitleRenderMode.DRAWTEXT:
            for index, cue in enumerate(request.cues):
                resolved_font = sre._resolve_cue_font(cue, config)
                if resolved_font is not None and resolved_font.font_path:
                    filters[index].parameters["font_file"] = resolved_font.font_path
        graph = filter_graph_builder.build_filter_graph_from_filters(
            filters, pass_id="pass_subtitle", pass_type="subtitle", hint_type=request.mode, config=fg_config,
        )
        effective_serializer_config = serializer_config or filter_graph_serializer.load_filter_graph_serializer_config()
        return filter_graph_serializer.serialize_filter_graph(graph, effective_serializer_config)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationTests(unittest.TestCase):
    def test_default_config_loads(self):
        config = sre.load_subtitle_render_config()
        self.assertEqual(config.supported_modes, (sre.SubtitleRenderMode.ASS, sre.SubtitleRenderMode.DRAWTEXT))
        self.assertEqual(config.default_mode, sre.SubtitleRenderMode.ASS)
        self.assertFalse(config.allow_mode_fallback)
        self.assertEqual(config.ffmpeg_binary, "ffmpeg")
        self.assertTrue(config.copy_audio)
        self.assertTrue(config.drawtext_require_font_file)
        self.assertEqual(config.drawtext_anchor_margin_pixels, 40.0)

    def test_missing_config_file_raises(self):
        with self.assertRaises(sre.SubtitleRenderConfigError):
            sre.load_subtitle_render_config("/nonexistent/subtitle_render.yaml")

    def test_empty_config_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(sre.SubtitleRenderConfigError):
                sre.load_subtitle_render_config(path)

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("subtitle_render: [unclosed", encoding="utf-8")
            with self.assertRaises(sre.SubtitleRenderConfigError):
                sre.load_subtitle_render_config(path)

    def test_custom_config_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.yaml"
            path.write_text(
                "subtitle_render:\n  default_mode: drawtext\n"
                "video:\n  crf: 20\n  copy_audio: false\n"
                "drawtext:\n  anchor_margin_pixels: 60\n",
                encoding="utf-8",
            )
            config = sre.load_subtitle_render_config(path)
            self.assertEqual(config.default_mode, "drawtext")
            self.assertEqual(config.video_crf, 20)
            self.assertFalse(config.copy_audio)
            self.assertEqual(config.drawtext_anchor_margin_pixels, 60.0)

    def test_default_config_path_resolves_under_runtime_root(self):
        path = sre.default_subtitle_render_config_path()
        self.assertTrue(str(path).endswith("config/video/subtitle_render.yaml"))


# ---------------------------------------------------------------------------
# Font resolution
# ---------------------------------------------------------------------------


class FontResolutionTests(SubtitleRenderTempTestCase):
    def test_explicit_font_path_resolves(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=font, font_family=None, config=self.config)
        self.assertEqual(resolved.source, "explicit")
        self.assertEqual(resolved.font_path, str(font))

    def test_family_map_resolves(self):
        font = self._font()
        config = dataclasses.replace(self.config, font_family_map={"Body": str(font)})
        resolved = sre.resolve_font(explicit_font_path=None, font_family="Body", config=config)
        self.assertEqual(resolved.source, "family_map")

    def test_fallback_resolves(self):
        font = self._font()
        config = dataclasses.replace(self.config, font_fallback_path=str(font))
        resolved = sre.resolve_font(explicit_font_path=None, font_family=None, config=config)
        self.assertEqual(resolved.source, "fallback")

    def test_explicit_beats_family_map_and_fallback(self):
        explicit_font = self._font("explicit.ttf")
        mapped_font = self._font("mapped.ttf")
        config = dataclasses.replace(
            self.config, font_family_map={"Body": str(mapped_font)}, font_fallback_path=str(mapped_font)
        )
        resolved = sre.resolve_font(explicit_font_path=explicit_font, font_family="Body", config=config)
        self.assertEqual(resolved.font_path, str(explicit_font))

    def test_family_map_beats_fallback(self):
        mapped_font = self._font("mapped.ttf")
        fallback_font = self._font("fallback.ttf")
        config = dataclasses.replace(
            self.config, font_family_map={"Body": str(mapped_font)}, font_fallback_path=str(fallback_font)
        )
        resolved = sre.resolve_font(explicit_font_path=None, font_family="Body", config=config)
        self.assertEqual(resolved.font_path, str(mapped_font))

    def test_no_resolution_raises(self):
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre.resolve_font(explicit_font_path=None, font_family="Unknown", config=self.config)

    def test_missing_explicit_font_file_raises(self):
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre.resolve_font(
                explicit_font_path=self.temp_dir / "missing.ttf", font_family=None, config=self.config
            )

    def test_directory_as_font_path_raises(self):
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre.resolve_font(explicit_font_path=self.temp_dir, font_family=None, config=self.config)

    def test_zero_byte_font_raises(self):
        font = self._font(content=b"")
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre.resolve_font(explicit_font_path=font, font_family=None, config=self.config)

    def test_unsupported_font_extension_raises(self):
        font = self._font("font.woff")
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre.resolve_font(explicit_font_path=font, font_family=None, config=self.config)

    def test_cue_font_resolution_required_raises_when_unresolvable(self):
        cue = self._cue(font_family=None, font_path=None)
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre._resolve_cue_font(cue, self.config)

    def test_cue_font_resolution_optional_returns_none(self):
        config = dataclasses.replace(self.config, drawtext_require_font_file=False)
        cue = self._cue(font_family=None, font_path=None)
        self.assertIsNone(sre._resolve_cue_font(cue, config))

    def test_cue_font_resolution_optional_still_resolves_when_available(self):
        font = self._font()
        config = dataclasses.replace(self.config, drawtext_require_font_file=False)
        cue = self._cue(font_path=str(font))
        resolved = sre._resolve_cue_font(cue, config)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.font_path, str(font))


# ---------------------------------------------------------------------------
# Request construction / validation
# ---------------------------------------------------------------------------


class RequestValidationTests(SubtitleRenderTempTestCase):
    def test_unsupported_mode_raises(self):
        video = self._video()
        with self.assertRaises(sre.UnsupportedSubtitleRenderModeError):
            sre.build_subtitle_render_request(
                mode="karaoke", video_path=video, output_path=self.temp_dir / "out.mp4", config=self.config
            )

    def test_mode_disabled_by_config_raises(self):
        video = self._video()
        config = dataclasses.replace(self.config, supported_modes=(sre.SubtitleRenderMode.DRAWTEXT,))
        ass = self._ass()
        with self.assertRaises(sre.UnsupportedSubtitleRenderModeError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
                ass_path=ass, config=config,
            )

    def test_missing_video_raises(self):
        with self.assertRaises(sre.SubtitleAssetNotFoundError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=self.temp_dir / "missing.mp4",
                output_path=self.temp_dir / "out.mp4", cues=[self._cue()], config=self.config,
            )

    def test_video_is_directory_raises(self):
        with self.assertRaises(sre.SubtitleAssetNotFoundError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=self.temp_dir,
                output_path=self.temp_dir / "out.mp4", cues=[self._cue()], config=self.config,
            )

    def test_empty_video_raises(self):
        video = self._video(content=b"")
        with self.assertRaises(sre.SubtitleAssetEmptyError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
                cues=[self._cue()], config=self.config,
            )

    def test_ass_mode_requires_ass_path(self):
        video = self._video()
        with self.assertRaises(sre.SubtitleRenderRequestError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
                config=self.config,
            )

    def test_ass_missing_file_raises(self):
        video = self._video()
        with self.assertRaises(sre.SubtitleAssetNotFoundError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
                ass_path=self.temp_dir / "missing.ass", config=self.config,
            )

    def test_ass_empty_file_raises(self):
        video = self._video()
        ass = self._ass(content="")
        with self.assertRaises(sre.SubtitleAssetEmptyError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
                ass_path=ass, config=self.config,
            )

    def test_ass_wrong_extension_raises(self):
        video = self._video()
        srt = self.temp_dir / "subs.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
        with self.assertRaises(sre.UnsupportedSubtitleAssetError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
                ass_path=srt, config=self.config,
            )

    def test_drawtext_mode_requires_cues(self):
        video = self._video()
        with self.assertRaises(sre.SubtitleRenderRequestError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
                cues=[], config=self.config,
            )

    def test_fontsdir_not_a_directory_raises(self):
        video = self._video()
        ass = self._ass()
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
                ass_path=ass, fontsdir=video, config=self.config,
            )

    def test_output_same_as_video_raises(self):
        video = self._video()
        with self.assertRaises(sre.UnsafeSubtitleRenderOutputError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=video,
                cues=[self._cue()], config=self.config,
            )

    def test_output_same_as_ass_raises(self):
        video = self._video()
        ass = self._ass()
        with self.assertRaises(sre.UnsafeSubtitleRenderOutputError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=ass, ass_path=ass, config=self.config,
            )

    def test_output_directory_raises(self):
        video = self._video()
        with self.assertRaises(sre.UnsafeSubtitleRenderOutputError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir,
                cues=[self._cue()], config=self.config,
            )

    def test_output_exists_without_force_raises(self):
        video = self._video()
        output = self.temp_dir / "out.mp4"
        output.write_bytes(b"existing")
        with self.assertRaises(sre.SubtitleRenderOutputExistsError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=output,
                cues=[self._cue()], config=self.config,
            )

    def test_output_exists_with_force_succeeds(self):
        video = self._video()
        output = self.temp_dir / "out.mp4"
        output.write_bytes(b"existing")
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=output,
            cues=[self._cue()], force=True, config=self.config,
        )
        self.assertTrue(request.force)

    def test_output_parent_not_directory_raises(self):
        video = self._video()
        blocker = self.temp_dir / "blocker"
        blocker.write_bytes(b"x")
        with self.assertRaises(sre.UnsafeSubtitleRenderOutputError):
            sre.build_subtitle_render_request(
                mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=blocker / "out.mp4",
                cues=[self._cue()], config=self.config,
            )

    def test_valid_ass_request_succeeds(self):
        video = self._video()
        ass = self._ass()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
            ass_path=ass, config=self.config,
        )
        self.assertEqual(request.mode, sre.SubtitleRenderMode.ASS)
        self.assertEqual(request.ass_path, ass)

    def test_valid_drawtext_request_succeeds(self):
        video = self._video()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue()], config=self.config,
        )
        self.assertEqual(len(request.cues), 1)


# ---------------------------------------------------------------------------
# Command construction — Phase 11F.3: the -vf/-filter_complex portion
# now comes from a serialized FilterGraph (see FilterGraphIntegrationTests
# and test_filter_graph_serializer.py for the ASS/drawtext filter-shape/
# escaping/anchor-math coverage that used to live in this file directly
# against the since-removed build_ass_filter()/build_drawtext_filter()).
# ---------------------------------------------------------------------------


class CommandConstructionTests(SubtitleRenderTempTestCase):
    def test_ass_command_shape(self):
        video = self._video()
        ass = self._ass()
        fontsdir = self._fontsdir()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
            ass_path=ass, fontsdir=fontsdir, config=self.config,
        )
        serialized = self._build_serialized(request, self.config)
        command = sre.build_subtitle_ffmpeg_command(request, self.config, serialized=serialized)
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("-hide_banner", command)
        self.assertIn("-i", command)
        self.assertIn(str(video), command)
        self.assertIn("-vf", command)
        self.assertEqual(command[-1], str(request.output_path))
        self.assertIn("-c:a", command)
        self.assertIn("copy", command)

    def test_drawtext_command_joins_multiple_cues_in_order(self):
        video = self._video()
        font = self._font()
        cues = [
            self._cue(text="First", font_path=str(font)),
            self._cue(text="Second", start_seconds=2.0, end_seconds=4.0, font_path=str(font)),
        ]
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=cues, config=self.config,
        )
        serialized = self._build_serialized(request, self.config)
        command = sre.build_subtitle_ffmpeg_command(request, self.config, serialized=serialized)
        vf_index = command.index("-vf")
        filter_value = command[vf_index + 1]
        self.assertLess(filter_value.index("First"), filter_value.index("Second"))
        self.assertEqual(filter_value.count("drawtext="), 2)

    def test_copy_audio_true_uses_dash_c_a_copy(self):
        video = self._video()
        config = dataclasses.replace(self.config, copy_audio=True, drawtext_require_font_file=False)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=None, font_family=None)], config=config,
        )
        serializer_config = dataclasses.replace(
            filter_graph_serializer.load_filter_graph_serializer_config(), drawtext_require_font_file=False
        )
        serialized = self._build_serialized(request, config, serializer_config=serializer_config)
        command = sre.build_subtitle_ffmpeg_command(request, config, serialized=serialized)
        index = command.index("-c:a")
        self.assertEqual(command[index + 1], "copy")

    def test_copy_audio_false_reencodes(self):
        video = self._video()
        config = dataclasses.replace(
            self.config, copy_audio=False, drawtext_require_font_file=False,
            audio_codec_when_reencode_required="aac", audio_bitrate="128k",
        )
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=None, font_family=None)], config=config,
        )
        serializer_config = dataclasses.replace(
            filter_graph_serializer.load_filter_graph_serializer_config(), drawtext_require_font_file=False
        )
        serialized = self._build_serialized(request, config, serializer_config=serializer_config)
        command = sre.build_subtitle_ffmpeg_command(request, config, serialized=serialized)
        index = command.index("-c:a")
        self.assertEqual(command[index + 1], "aac")
        self.assertIn("-b:a", command)
        self.assertIn("128k", command)

    def test_faststart_adds_movflags(self):
        video = self._video()
        font = self._font()
        config = dataclasses.replace(self.config, faststart=True)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=config,
        )
        serialized = self._build_serialized(request, config)
        command = sre.build_subtitle_ffmpeg_command(request, config, serialized=serialized)
        self.assertIn("-movflags", command)
        self.assertIn("+faststart", command)

    def test_faststart_disabled_omits_movflags(self):
        video = self._video()
        font = self._font()
        config = dataclasses.replace(self.config, faststart=False)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=config,
        )
        serialized = self._build_serialized(request, config)
        command = sre.build_subtitle_ffmpeg_command(request, config, serialized=serialized)
        self.assertNotIn("-movflags", command)

    def test_force_uses_dash_y(self):
        video = self._video()
        font = self._font()
        output = self.temp_dir / "out.mp4"
        output.write_bytes(b"existing")
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=output,
            cues=[self._cue(font_path=str(font))], force=True, config=self.config,
        )
        serialized = self._build_serialized(request, self.config)
        command = sre.build_subtitle_ffmpeg_command(request, self.config, serialized=serialized)
        self.assertIn("-y", command)
        self.assertNotIn("-n", command)

    def test_no_force_uses_dash_n(self):
        video = self._video()
        font = self._font()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=self.config,
        )
        serialized = self._build_serialized(request, self.config)
        command = sre.build_subtitle_ffmpeg_command(request, self.config, serialized=serialized)
        self.assertIn("-n", command)

    def test_hide_banner_disabled(self):
        video = self._video()
        font = self._font()
        config = dataclasses.replace(self.config, ffmpeg_hide_banner=False)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=config,
        )
        serialized = self._build_serialized(request, config)
        command = sre.build_subtitle_ffmpeg_command(request, config, serialized=serialized)
        self.assertNotIn("-hide_banner", command)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class PlanTests(SubtitleRenderTempTestCase):
    def test_ass_plan_builds_without_executing(self):
        video = self._video()
        ass = self._ass()
        fontsdir = self._fontsdir()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
            ass_path=ass, fontsdir=fontsdir, config=self.config,
        )
        plan = sre.build_subtitle_render_plan(request, self.config)
        self.assertEqual(plan.mode, sre.SubtitleRenderMode.ASS)
        self.assertTrue(plan.render_id)
        self.assertFalse((self.temp_dir / "out.mp4").exists())
        self.assertEqual(plan.subtitle_asset.asset_type, "ass")
        self.assertTrue(plan.filter_graph_id)
        self.assertTrue(plan.filter_graph_validation_passed)
        self.assertTrue(plan.serialization_id)
        self.assertEqual(plan.output_mode, "simple_vf")

    def test_drawtext_plan_builds(self):
        video = self._video()
        font = self._font()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=self.config,
        )
        plan = sre.build_subtitle_render_plan(request, self.config)
        self.assertEqual(plan.mode, sre.SubtitleRenderMode.DRAWTEXT)
        self.assertEqual(plan.subtitle_asset.asset_type, "drawtext")
        self.assertEqual(len(plan.font_resolution), 1)

    def test_render_id_stable_across_repeated_planning(self):
        video = self._video()
        font = self._font()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=self.config,
        )
        plan_a = sre.build_subtitle_render_plan(request, self.config)
        plan_b = sre.build_subtitle_render_plan(request, self.config)
        self.assertEqual(plan_a.render_id, plan_b.render_id)
        self.assertEqual(plan_a.command, plan_b.command)

    def test_render_id_changes_with_cue_text(self):
        video = self._video()
        font = self._font()
        request_a = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(text="Hello", font_path=str(font))], config=self.config,
        )
        request_b = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(text="Goodbye", font_path=str(font))], config=self.config,
        )
        plan_a = sre.build_subtitle_render_plan(request_a, self.config)
        plan_b = sre.build_subtitle_render_plan(request_b, self.config)
        self.assertNotEqual(plan_a.render_id, plan_b.render_id)


# ---------------------------------------------------------------------------
# FilterGraph / serializer integration (Phase 11F.3)
# ---------------------------------------------------------------------------


def _counting_wrapper(real_func):
    calls: list[tuple[tuple, dict]] = []

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real_func(*args, **kwargs)

    wrapper.calls = calls
    return wrapper


class FilterGraphIntegrationTests(SubtitleRenderTempTestCase):
    def _drawtext_request(self):
        video = self._video()
        font = self._font()
        return sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=self.config,
        )

    def test_drawtext_routes_through_filter_graph_builder(self):
        request = self._drawtext_request()
        builder_spy = _counting_wrapper(filter_graph_builder.build_subtitle_filter_spec)
        plan = sre.build_subtitle_render_plan(request, self.config, filter_graph_builder_callable=builder_spy)
        self.assertEqual(len(builder_spy.calls), 1)
        self.assertTrue(plan.filter_graph_id)

    def test_ass_routes_through_filter_graph_builder(self):
        video = self._video()
        ass = self._ass()
        fontsdir = self._fontsdir()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
            ass_path=ass, fontsdir=fontsdir, config=self.config,
        )
        builder_spy = _counting_wrapper(filter_graph_builder.build_subtitle_filter_spec)
        plan = sre.build_subtitle_render_plan(request, self.config, filter_graph_builder_callable=builder_spy)
        self.assertEqual(len(builder_spy.calls), 1)
        self.assertTrue(plan.filter_graph_id)

    def test_validator_called_exactly_once(self):
        request = self._drawtext_request()
        validator_spy = _counting_wrapper(filter_graph_builder.build_filter_graph_from_filters)
        sre.build_subtitle_render_plan(request, self.config, filter_graph_validator_callable=validator_spy)
        self.assertEqual(len(validator_spy.calls), 1)

    def test_serializer_called_exactly_once(self):
        request = self._drawtext_request()
        serializer_spy = _counting_wrapper(filter_graph_serializer.serialize_filter_graph)
        sre.build_subtitle_render_plan(request, self.config, filter_graph_serializer_callable=serializer_spy)
        self.assertEqual(len(serializer_spy.calls), 1)

    def test_serialized_vf_content_in_final_command(self):
        request = self._drawtext_request()
        plan = sre.build_subtitle_render_plan(request, self.config)
        self.assertIn("-vf", plan.command)
        vf_index = plan.command.index("-vf")
        self.assertIn("drawtext=", plan.command[vf_index + 1])
        self.assertIn("Hello world", plan.command[vf_index + 1])

    def test_filter_complex_mode_honored_when_forced(self):
        request = self._drawtext_request()
        forced_config = dataclasses.replace(
            filter_graph_serializer.load_filter_graph_serializer_config(), force_filter_complex=True
        )
        plan = sre.build_subtitle_render_plan(request, self.config, serializer_config=forced_config)
        self.assertEqual(plan.output_mode, "filter_complex")
        self.assertIn("-filter_complex", plan.command)
        self.assertNotIn("-vf", plan.command)

    def test_graph_validation_failure_raises_subtitle_render_filter_graph_error(self):
        request = self._drawtext_request()

        def broken_validator(filters, **kwargs):
            graph = filter_graph_builder.build_filter_graph_from_filters(filters, **kwargs)
            graph.validation.passed = False
            graph.validation.errors = ["synthetic failure for test"]
            return graph

        with self.assertRaises(sre.SubtitleRenderFilterGraphError):
            sre.build_subtitle_render_plan(request, self.config, filter_graph_validator_callable=broken_validator)

    def test_graph_validation_failure_prevents_runner_call(self):
        request = self._drawtext_request()
        runner = FakeRunner()

        def broken_validator(filters, **kwargs):
            graph = filter_graph_builder.build_filter_graph_from_filters(filters, **kwargs)
            graph.validation.passed = False
            graph.validation.errors = ["synthetic failure for test"]
            return graph

        with self.assertRaises(sre.SubtitleRenderFilterGraphError):
            sre.execute_subtitle_render_plan(
                request, self.config, runner=runner, filter_graph_validator_callable=broken_validator
            )
        self.assertEqual(len(runner.calls), 0)

    def test_serializer_failure_raises_subtitle_render_filter_graph_error(self):
        request = self._drawtext_request()

        def broken_serializer(graph, config):
            raise filter_graph_serializer.FilterGraphSerializerError("synthetic serializer failure")

        with self.assertRaises(sre.SubtitleRenderFilterGraphError):
            sre.build_subtitle_render_plan(request, self.config, filter_graph_serializer_callable=broken_serializer)

    def test_serializer_failure_prevents_runner_call(self):
        request = self._drawtext_request()
        runner = FakeRunner()

        def broken_serializer(graph, config):
            raise filter_graph_serializer.FilterGraphSerializerError("synthetic serializer failure")

        with self.assertRaises(sre.SubtitleRenderFilterGraphError):
            sre.execute_subtitle_render_plan(
                request, self.config, runner=runner, filter_graph_serializer_callable=broken_serializer
            )
        self.assertEqual(len(runner.calls), 0)

    def test_builder_failure_raises_subtitle_render_filter_graph_error(self):
        request = self._drawtext_request()

        def broken_builder(request, config):
            raise filter_graph_builder.FilterGraphBuilderError("synthetic builder failure")

        with self.assertRaises(sre.SubtitleRenderFilterGraphError):
            sre.build_subtitle_render_plan(request, self.config, filter_graph_builder_callable=broken_builder)

    def test_dry_run_reports_graph_and_serialization_diagnostics(self):
        request = self._drawtext_request()
        plan = sre.build_subtitle_render_plan(request, self.config)
        self.assertFalse(request.output_path.exists())
        self.assertTrue(plan.filter_graph_id)
        self.assertTrue(plan.serialization_id)
        self.assertTrue(plan.filter_graph_validation_passed)

    def test_successful_fake_render_unchanged(self):
        request = self._drawtext_request()
        config = dataclasses.replace(self.config, diagnostics_write_log=False)
        runner = FakeRunner()
        result = sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertEqual(result.return_code, 0)
        self.assertTrue(result.output_exists)
        self.assertEqual(len(runner.calls), 1)
        self.assertTrue(result.filter_graph_id)
        self.assertTrue(result.serialization_id)

    def test_font_resolution_unchanged(self):
        video = self._video()
        font = self._font()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=self.config,
        )
        plan = sre.build_subtitle_render_plan(request, self.config)
        self.assertEqual(len(plan.font_resolution), 1)
        self.assertEqual(plan.font_resolution[0].font_path, str(font))
        self.assertEqual(plan.font_resolution[0].source, "explicit")

    def test_audio_mapping_unchanged(self):
        request = self._drawtext_request()
        config = dataclasses.replace(self.config, copy_audio=False, audio_codec_when_reencode_required="aac", audio_bitrate="128k")
        plan = sre.build_subtitle_render_plan(request, config)
        index = plan.command.index("-c:a")
        self.assertEqual(plan.command[index + 1], "aac")

    def test_output_verification_unchanged(self):
        request = self._drawtext_request()
        config = dataclasses.replace(self.config, diagnostics_write_log=False)
        runner = FakeRunner()
        result = sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertTrue(result.output_exists)
        self.assertGreater(result.output_size_bytes, 0)

    def test_source_hashes_unchanged_after_planning(self):
        import hashlib

        video = self._video()
        font = self._font()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue(font_path=str(font))], config=self.config,
        )
        before_video = hashlib.sha256(video.read_bytes()).hexdigest()
        before_font = hashlib.sha256(font.read_bytes()).hexdigest()
        sre.build_subtitle_render_plan(request, self.config)
        self.assertEqual(hashlib.sha256(video.read_bytes()).hexdigest(), before_video)
        self.assertEqual(hashlib.sha256(font.read_bytes()).hexdigest(), before_font)

    def test_source_hashes_unchanged_after_execution(self):
        import hashlib

        request = self._drawtext_request()
        config = dataclasses.replace(self.config, diagnostics_write_log=False)
        before_video = hashlib.sha256(request.video_path.read_bytes()).hexdigest()
        runner = FakeRunner()
        sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertEqual(hashlib.sha256(request.video_path.read_bytes()).hexdigest(), before_video)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecutionTests(SubtitleRenderTempTestCase):
    def _drawtext_request(self, **overrides):
        video = self._video()
        font = self._font()
        cues = overrides.pop("cues", [self._cue(font_path=str(font))])
        output = overrides.pop("output", self.temp_dir / "out.mp4")
        config = overrides.pop("config", dataclasses.replace(self.config, diagnostics_write_log=False))
        return sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=output,
            cues=cues, config=config, **overrides
        ), config

    def test_successful_execution_returns_result(self):
        request, config = self._drawtext_request()
        runner = FakeRunner()
        result = sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertEqual(result.return_code, 0)
        self.assertTrue(result.output_exists)
        self.assertTrue(request.output_path.exists())
        self.assertEqual(len(runner.calls), 1)

    def test_non_zero_return_code_raises(self):
        request, config = self._drawtext_request()
        runner = FakeRunner(returncode=1, stderr="boom")
        with self.assertRaises(sre.FFmpegExecutionError):
            sre.execute_subtitle_render_plan(request, config, runner=runner)

    def test_missing_binary_raises_ffmpeg_not_found(self):
        request, config = self._drawtext_request()
        runner = FakeRunner(raise_exception=sre.FFmpegNotFoundError("no ffmpeg"))
        with self.assertRaises(sre.FFmpegNotFoundError):
            sre.execute_subtitle_render_plan(request, config, runner=runner)

    def test_timeout_raises(self):
        request, config = self._drawtext_request()
        runner = FakeRunner(raise_exception=sre.FFmpegTimeoutError("timed out"))
        with self.assertRaises(sre.FFmpegTimeoutError):
            sre.execute_subtitle_render_plan(request, config, runner=runner)

    def test_success_but_no_output_written_raises_verification_error(self):
        request, config = self._drawtext_request()
        runner = FakeRunner(write_output=False)
        with self.assertRaises(sre.SubtitleRenderVerificationError):
            sre.execute_subtitle_render_plan(request, config, runner=runner)

    def test_success_but_zero_byte_output_raises(self):
        request, config = self._drawtext_request()
        runner = FakeRunner(output_bytes=b"")
        with self.assertRaises(sre.SubtitleRenderVerificationError):
            sre.execute_subtitle_render_plan(request, config, runner=runner)

    def test_output_exists_without_force_raises_before_running(self):
        request, config = self._drawtext_request()
        request.output_path.write_bytes(b"already there")
        runner = FakeRunner()
        with self.assertRaises(sre.SubtitleRenderOutputExistsError):
            sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertEqual(len(runner.calls), 0)

    def test_dry_run_plan_never_touches_filesystem(self):
        request, config = self._drawtext_request()
        plan = sre.build_subtitle_render_plan(request, config)
        self.assertFalse(request.output_path.exists())
        self.assertTrue(plan.command)

    def test_runner_called_exactly_once(self):
        request, config = self._drawtext_request()
        runner = FakeRunner()
        sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertEqual(len(runner.calls), 1)

    def test_runner_receives_timeout_from_config(self):
        request, config = self._drawtext_request(config=dataclasses.replace(
            self.config, diagnostics_write_log=False, ffmpeg_timeout_seconds=42
        ))
        runner = FakeRunner()
        sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertEqual(runner.calls[0][1], 42)

    def test_inspector_sets_audio_preserved(self):
        request, config = self._drawtext_request()
        runner = FakeRunner()
        result = sre.execute_subtitle_render_plan(
            request, config, runner=runner, inspector=_fake_inspector(has_audio=True)
        )
        self.assertTrue(result.audio_preserved)

    def test_inspector_omitted_leaves_audio_preserved_none(self):
        request, config = self._drawtext_request()
        runner = FakeRunner()
        result = sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertIsNone(result.audio_preserved)

    def test_ass_mode_executes(self):
        video = self._video()
        ass = self._ass()
        fontsdir = self._fontsdir()
        config = dataclasses.replace(self.config, diagnostics_write_log=False)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
            ass_path=ass, fontsdir=fontsdir, config=config,
        )
        runner = FakeRunner()
        result = sre.execute_subtitle_render_plan(request, config, runner=runner)
        self.assertEqual(result.mode, sre.SubtitleRenderMode.ASS)
        self.assertEqual(result.subtitle_asset, str(ass))


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------


class VerificationTests(SubtitleRenderTempTestCase):
    def test_missing_output_returns_false(self):
        exists, size = sre.verify_subtitle_render_output(self.temp_dir / "missing.mp4", self.temp_dir / "in.mp4")
        self.assertFalse(exists)
        self.assertEqual(size, 0)

    def test_existing_output_returns_true_and_size(self):
        output = self._video("out.mp4", content=b"12345")
        exists, size = sre.verify_subtitle_render_output(output, self.temp_dir / "in.mp4")
        self.assertTrue(exists)
        self.assertEqual(size, 5)

    def test_output_same_as_input_raises(self):
        video = self._video()
        with self.assertRaises(sre.UnsafeSubtitleRenderOutputError):
            sre.verify_subtitle_render_output(video, video)

    def test_inspector_invoked_when_provided(self):
        output = self._video("out.mp4")
        calls = []

        def inspector(path):
            calls.append(path)
            return FakeInspectorInfo()

        sre.verify_subtitle_render_output(output, self.temp_dir / "in.mp4", inspector=inspector)
        self.assertEqual(len(calls), 1)


# ---------------------------------------------------------------------------
# Diagnostics log
# ---------------------------------------------------------------------------


class DiagnosticsTests(SubtitleRenderTempTestCase):
    def test_log_written_when_enabled(self):
        video = self._video()
        font = self._font()
        config = dataclasses.replace(self.config, diagnostics_write_log=True)
        output = self.temp_dir / "out.mp4"
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=output,
            cues=[self._cue(font_path=str(font))], config=config,
        )
        runner = FakeRunner()
        sre.execute_subtitle_render_plan(request, config, runner=runner)
        log_path = output.with_name(f"{output.stem}{config.diagnostics_log_filename_suffix}")
        self.assertTrue(log_path.exists())
        data = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertIn("render_id", data)
        self.assertIn("command", data)

    def test_log_not_written_when_disabled(self):
        video = self._video()
        font = self._font()
        config = dataclasses.replace(self.config, diagnostics_write_log=False)
        output = self.temp_dir / "out.mp4"
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=output,
            cues=[self._cue(font_path=str(font))], config=config,
        )
        runner = FakeRunner()
        sre.execute_subtitle_render_plan(request, config, runner=runner)
        log_path = output.with_name(f"{output.stem}{config.diagnostics_log_filename_suffix}")
        self.assertFalse(log_path.exists())

    def test_save_log_is_atomic_no_leftover_tmp_file(self):
        result = sre.SubtitleRenderResult(render_id="abc123", mode="ass")
        log_path = self.temp_dir / "log.json"
        sre.save_subtitle_render_log(result, log_path)
        self.assertTrue(log_path.exists())
        self.assertFalse((self.temp_dir / "log.json.tmp").exists())

    def test_result_to_dict_round_trips_json_serializable(self):
        result = sre.SubtitleRenderResult(render_id="abc123", mode="drawtext", warnings=["w1"])
        data = sre.subtitle_render_result_to_dict(result)
        json.dumps(data)  # must not raise
        self.assertEqual(data["render_id"], "abc123")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(SubtitleRenderTempTestCase):
    def test_ass_and_drawtext_json_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            sre.parse_arguments(["--video", "v.mp4", "--output", "o.mp4", "--ass", "s.ass", "--drawtext-json", "c.json"])

    def test_missing_mode_argument_required(self):
        with self.assertRaises(SystemExit):
            sre.parse_arguments(["--video", "v.mp4", "--output", "o.mp4"])

    def test_dry_run_ass_via_cli(self):
        video = self._video()
        ass = self._ass()
        fontsdir = self._fontsdir()
        output = self.temp_dir / "out.mp4"
        config_path = self.temp_dir / "config.yaml"
        config_path.write_text(sre.default_subtitle_render_config_path().read_text(encoding="utf-8"), encoding="utf-8")
        sre.main([
            "--video", str(video), "--output", str(output), "--ass", str(ass),
            "--fontsdir", str(fontsdir), "--config", str(config_path), "--dry-run",
        ])
        self.assertFalse(output.exists())

    def test_dry_run_drawtext_via_cli_json(self):
        video = self._video()
        font = self._font()
        output = self.temp_dir / "out.mp4"
        cues_path = self.temp_dir / "cues.json"
        cues_path.write_text(
            json.dumps([{"text": "Hi", "start_seconds": 0.0, "end_seconds": 1.0, "x": 10, "y": 10, "font_path": str(font)}]),
            encoding="utf-8",
        )
        config_path = self.temp_dir / "config.yaml"
        config_path.write_text(sre.default_subtitle_render_config_path().read_text(encoding="utf-8"), encoding="utf-8")
        sre.main([
            "--video", str(video), "--output", str(output), "--drawtext-json", str(cues_path),
            "--config", str(config_path), "--dry-run",
        ])
        self.assertFalse(output.exists())

    def test_cli_missing_video_fails_cleanly(self):
        output = self.temp_dir / "out.mp4"
        ass = self._ass()
        config_path = self.temp_dir / "config.yaml"
        config_path.write_text(sre.default_subtitle_render_config_path().read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaises(SystemExit):
            sre.main([
                "--video", str(self.temp_dir / "missing.mp4"), "--output", str(output), "--ass", str(ass),
                "--config", str(config_path), "--dry-run",
            ])

    def test_load_drawtext_cues_missing_file_raises(self):
        with self.assertRaises(sre.SubtitleRenderRequestError):
            sre._load_drawtext_cues(self.temp_dir / "missing.json")

    def test_load_drawtext_cues_invalid_json_raises(self):
        path = self.temp_dir / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with self.assertRaises(sre.SubtitleRenderRequestError):
            sre._load_drawtext_cues(path)

    def test_load_drawtext_cues_non_list_root_raises(self):
        path = self.temp_dir / "obj.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(sre.SubtitleRenderRequestError):
            sre._load_drawtext_cues(path)

    def test_load_drawtext_cues_non_object_entry_raises(self):
        path = self.temp_dir / "entries.json"
        path.write_text(json.dumps(["not an object"]), encoding="utf-8")
        with self.assertRaises(sre.SubtitleRenderRequestError):
            sre._load_drawtext_cues(path)

    def test_load_drawtext_cues_parses_known_fields_only(self):
        path = self.temp_dir / "cues.json"
        path.write_text(json.dumps([{"text": "Hi", "start_seconds": 0.0, "end_seconds": 1.0, "unknown_field": "ignored"}]), encoding="utf-8")
        cues = sre._load_drawtext_cues(path)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "Hi")


# ---------------------------------------------------------------------------
# Structural safety
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_path = Path(__file__).resolve().parent / "subtitle_render_engine.py"
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

    def test_no_import_of_other_pipeline_modules(self):
        for forbidden in ("renderer_plan_engine", "overlay_plan_engine", "timeline_engine", "media_inspector",
                          "video_engine", "music_mixer", "reel_builder", "subtitle_engine"):
            for line in self.source_lines:
                stripped = line.strip()
                if stripped.startswith("from . import") or stripped.startswith("import "):
                    self.assertNotIn(forbidden, stripped, f"{forbidden} should not be imported by subtitle_render_engine.py")

    def test_no_playwright_or_publish_reference(self):
        for forbidden in ("playwright", "Playwright", "instagram", "Instagram", "publish_reel", "upload_reel"):
            self.assertNotIn(forbidden, self.source_text)

    def test_no_ass_file_content_read(self):
        # This module must never open/parse .ass file contents -- only
        # existence/size/extension checks on the path itself.
        for line in self.source_lines:
            stripped = line.strip()
            self.assertFalse(".ass" in stripped and ("read_text(" in stripped or "open(" in stripped))

    def test_default_runner_maps_exceptions_correctly(self):
        self.assertIn("except FileNotFoundError as exc:", self.source_text)
        self.assertIn("raise FFmpegNotFoundError(", self.source_text)
        self.assertIn("except subprocess.TimeoutExpired as exc:", self.source_text)
        self.assertIn("raise FFmpegTimeoutError(", self.source_text)

    def test_no_retry_loop_around_runner_call(self):
        self.assertNotIn("for attempt in range", self.source_text)
        self.assertNotIn("while True", self.source_text)

    def test_mode_is_never_reassigned_after_request_construction(self):
        # allow_mode_fallback exists in config but must never be read to
        # switch a request's mode -- no fallback logic implemented.
        import re
        self.assertIsNone(re.search(r"request\.mode\s*=(?!=)", self.source_text))
        self.assertNotIn("if config.allow_mode_fallback", self.source_text)

    def test_no_independent_filter_string_escaping(self):
        # Phase 11F.3: escaping/filter-string construction moved to
        # filter_graph_serializer.py -- this module must not define its
        # own copies anymore.
        for forbidden in (
            "_escape_ffmpeg_quoted_value", "_escape_drawtext_percent",
            "_ANCHOR_MAP", "_anchor_expression", "SubtitleFilterFragment",
            "def build_ass_filter(", "def build_drawtext_filter(",
        ):
            self.assertNotIn(forbidden, self.source_text)

    def test_uses_filter_graph_builder_and_serializer(self):
        self.assertIn("filter_graph_builder.build_subtitle_filter_spec", self.source_text)
        self.assertIn("filter_graph_builder.build_filter_graph_from_filters", self.source_text)
        self.assertIn("filter_graph_serializer.serialize_filter_graph", self.source_text)


if __name__ == "__main__":
    unittest.main()
