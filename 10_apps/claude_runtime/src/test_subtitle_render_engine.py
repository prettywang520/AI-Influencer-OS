from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path

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
# ASS filter
# ---------------------------------------------------------------------------


class AssFilterTests(SubtitleRenderTempTestCase):
    def _request(self, *, fontsdir=None, config=None) -> sre.SubtitleRenderRequest:
        video = self._video()
        ass = self._ass()
        return sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
            ass_path=ass, fontsdir=fontsdir, config=config or self.config,
        )

    def test_basic_ass_filter_with_fontsdir(self):
        fontsdir = self._fontsdir()
        request = self._request(fontsdir=fontsdir)
        fragment = sre.build_ass_filter(request, self.config)
        self.assertTrue(fragment.filter_string.startswith("subtitles=filename="))
        self.assertIn("fontsdir=", fragment.filter_string)

    def test_ass_filter_without_fontsdir_raises_by_default(self):
        request = self._request()
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre.build_ass_filter(request, self.config)

    def test_ass_filter_without_fontsdir_ok_when_fontconfig_allowed(self):
        config = dataclasses.replace(self.config, ass_allow_fontconfig_resolution=True)
        request = self._request(config=config)
        fragment = sre.build_ass_filter(request, config)
        self.assertNotIn("fontsdir=", fragment.filter_string)

    def test_ass_filter_without_fontsdir_ok_when_not_required(self):
        config = dataclasses.replace(self.config, ass_require_fontsdir_when_custom_fonts=False)
        request = self._request(config=config)
        fragment = sre.build_ass_filter(request, config)
        self.assertNotIn("fontsdir=", fragment.filter_string)

    def test_ass_filter_escapes_single_quote_in_path(self):
        tricky_dir = self.temp_dir / "it's tricky"
        tricky_dir.mkdir()
        ass = tricky_dir / "subs.ass"
        ass.write_text("[Script Info]\n", encoding="utf-8")
        video = self._video()
        fontsdir = self._fontsdir()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4",
            ass_path=ass, fontsdir=fontsdir, config=self.config,
        )
        fragment = sre.build_ass_filter(request, self.config)
        self.assertIn("'\\''", fragment.filter_string)
        self.assertNotIn("it's tricky'", fragment.filter_string.split("fontsdir")[0].split("'\\''")[0] + "tricky'")

    def test_build_ass_filter_requires_ass_path(self):
        video = self._video()
        request = sre.SubtitleRenderRequest(mode=sre.SubtitleRenderMode.ASS, video_path=video, output_path=self.temp_dir / "out.mp4")
        with self.assertRaises(sre.SubtitleFilterGraphError):
            sre.build_ass_filter(request, self.config)


# ---------------------------------------------------------------------------
# Drawtext filter
# ---------------------------------------------------------------------------


class DrawTextFilterTests(SubtitleRenderTempTestCase):
    def test_basic_filter_with_explicit_xy(self):
        font = self._font()
        cue = self._cue(font_path=str(font))
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        fragment = sre.build_drawtext_filter(cue, resolved, self.config)
        self.assertTrue(fragment.filter_string.startswith("drawtext="))
        self.assertIn("x=100.0", fragment.filter_string)
        self.assertIn("y=200.0", fragment.filter_string)

    def test_all_nine_anchors_resolve(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        for anchor in sre._ANCHOR_MAP:
            cue = self._cue(x=None, y=None, anchor=anchor)
            fragment = sre.build_drawtext_filter(cue, resolved, self.config)
            self.assertIn("x=", fragment.filter_string)
            self.assertIn("y=", fragment.filter_string)

    def test_bottom_center_uses_margin(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(x=None, y=None, anchor="bottom_center")
        fragment = sre.build_drawtext_filter(cue, resolved, self.config)
        self.assertIn("x=(w-text_w)/2", fragment.filter_string)
        self.assertIn(f"y=h-text_h-{self.config.drawtext_anchor_margin_pixels}", fragment.filter_string)

    def test_unsupported_anchor_raises(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(x=None, y=None, anchor="middle_of_nowhere")
        with self.assertRaises(sre.SubtitleDrawTextCueError):
            sre.build_drawtext_filter(cue, resolved, self.config)

    def test_missing_position_raises(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(x=None, y=None, anchor=None)
        with self.assertRaises(sre.SubtitleDrawTextCueError):
            sre.build_drawtext_filter(cue, resolved, self.config)

    def test_x_without_y_raises(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(x=100.0, y=None)
        with self.assertRaises(sre.SubtitleDrawTextCueError):
            sre.build_drawtext_filter(cue, resolved, self.config)

    def test_non_numeric_x_raises(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(x="(w-text_w)/2", y=100.0)
        with self.assertRaises(sre.SubtitleDrawTextCueError):
            sre.build_drawtext_filter(cue, resolved, self.config)

    def test_out_of_canvas_bounds_raises_when_strict(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(x=5000.0, y=200.0)
        with self.assertRaises(sre.SubtitleDrawTextCueError):
            sre.build_drawtext_filter(cue, resolved, self.config, canvas_width=1080, canvas_height=1920)

    def test_out_of_canvas_bounds_ignored_without_canvas_dims(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(x=5000.0, y=200.0)
        fragment = sre.build_drawtext_filter(cue, resolved, self.config)
        self.assertIn("x=5000.0", fragment.filter_string)

    def test_out_of_canvas_bounds_ignored_when_not_strict(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        config = dataclasses.replace(self.config, drawtext_strict_canvas_bounds=False)
        cue = self._cue(x=5000.0, y=200.0)
        fragment = sre.build_drawtext_filter(cue, resolved, config, canvas_width=1080, canvas_height=1920)
        self.assertIn("x=5000.0", fragment.filter_string)

    def test_invalid_timing_raises(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(start_seconds=5.0, end_seconds=2.0)
        with self.assertRaises(sre.SubtitleDrawTextCueError):
            sre.build_drawtext_filter(cue, resolved, self.config)

    def test_empty_text_raises(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(text="")
        with self.assertRaises(sre.SubtitleDrawTextCueError):
            sre.build_drawtext_filter(cue, resolved, self.config)

    def test_missing_font_raises_when_required(self):
        cue = self._cue()
        with self.assertRaises(sre.SubtitleFontResolutionError):
            sre.build_drawtext_filter(cue, None, self.config)

    def test_missing_font_ok_when_not_required(self):
        config = dataclasses.replace(self.config, drawtext_require_font_file=False)
        cue = self._cue()
        fragment = sre.build_drawtext_filter(cue, None, config)
        self.assertNotIn("fontfile=", fragment.filter_string)

    def test_percent_in_text_is_escaped(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(text="50% off today")
        fragment = sre.build_drawtext_filter(cue, resolved, self.config)
        self.assertIn("50%% off today", fragment.filter_string)

    def test_apostrophe_in_text_is_escaped(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(text="it's a test")
        fragment = sre.build_drawtext_filter(cue, resolved, self.config)
        self.assertIn("'\\''", fragment.filter_string)

    def test_colon_and_comma_pass_through_inside_quotes(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(text="Chapter 1: Intro, welcome!")
        fragment = sre.build_drawtext_filter(cue, resolved, self.config)
        self.assertIn("Chapter 1: Intro, welcome!", fragment.filter_string)

    def test_box_enabled_adds_box_options(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(box_enabled=True, box_color="#000000AA")
        fragment = sre.build_drawtext_filter(cue, resolved, self.config)
        self.assertIn("box=1", fragment.filter_string)
        self.assertIn("boxcolor=", fragment.filter_string)

    def test_box_default_from_config_when_cue_unset(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        config = dataclasses.replace(self.config, drawtext_box_enabled_default=True)
        cue = self._cue(box_enabled=None)
        fragment = sre.build_drawtext_filter(cue, resolved, config)
        self.assertIn("box=1", fragment.filter_string)

    def test_enable_expression_present(self):
        font = self._font()
        resolved = sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)
        cue = self._cue(start_seconds=1.5, end_seconds=3.25)
        fragment = sre.build_drawtext_filter(cue, resolved, self.config)
        self.assertIn("between(t,1.5,3.25)", fragment.filter_string)


# ---------------------------------------------------------------------------
# Command construction
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
        command = sre.build_subtitle_ffmpeg_command(request, self.config, resolved_fonts=[])
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
        cues = [self._cue(text="First"), self._cue(text="Second", start_seconds=2.0, end_seconds=4.0)]
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=cues, config=self.config,
        )
        resolved = [sre.resolve_font(explicit_font_path=str(font), font_family=None, config=self.config)] * 2
        command = sre.build_subtitle_ffmpeg_command(request, self.config, resolved_fonts=resolved)
        vf_index = command.index("-vf")
        filter_value = command[vf_index + 1]
        self.assertLess(filter_value.index("First"), filter_value.index("Second"))
        self.assertEqual(filter_value.count("drawtext="), 2)

    def test_copy_audio_true_uses_dash_c_a_copy(self):
        video = self._video()
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue()], config=dataclasses.replace(self.config, drawtext_require_font_file=False),
        )
        config = dataclasses.replace(self.config, copy_audio=True, drawtext_require_font_file=False)
        command = sre.build_subtitle_ffmpeg_command(request, config, resolved_fonts=[None])
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
            cues=[self._cue()], config=config,
        )
        command = sre.build_subtitle_ffmpeg_command(request, config, resolved_fonts=[None])
        index = command.index("-c:a")
        self.assertEqual(command[index + 1], "aac")
        self.assertIn("-b:a", command)
        self.assertIn("128k", command)

    def test_faststart_adds_movflags(self):
        video = self._video()
        config = dataclasses.replace(self.config, faststart=True, drawtext_require_font_file=False)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue()], config=config,
        )
        command = sre.build_subtitle_ffmpeg_command(request, config, resolved_fonts=[None])
        self.assertIn("-movflags", command)
        self.assertIn("+faststart", command)

    def test_faststart_disabled_omits_movflags(self):
        video = self._video()
        config = dataclasses.replace(self.config, faststart=False, drawtext_require_font_file=False)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue()], config=config,
        )
        command = sre.build_subtitle_ffmpeg_command(request, config, resolved_fonts=[None])
        self.assertNotIn("-movflags", command)

    def test_force_uses_dash_y(self):
        video = self._video()
        config = dataclasses.replace(self.config, drawtext_require_font_file=False)
        output = self.temp_dir / "out.mp4"
        output.write_bytes(b"existing")
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=output,
            cues=[self._cue()], force=True, config=config,
        )
        command = sre.build_subtitle_ffmpeg_command(request, config, resolved_fonts=[None])
        self.assertIn("-y", command)
        self.assertNotIn("-n", command)

    def test_no_force_uses_dash_n(self):
        video = self._video()
        config = dataclasses.replace(self.config, drawtext_require_font_file=False)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue()], config=config,
        )
        command = sre.build_subtitle_ffmpeg_command(request, config, resolved_fonts=[None])
        self.assertIn("-n", command)

    def test_hide_banner_disabled(self):
        video = self._video()
        config = dataclasses.replace(self.config, ffmpeg_hide_banner=False, drawtext_require_font_file=False)
        request = sre.build_subtitle_render_request(
            mode=sre.SubtitleRenderMode.DRAWTEXT, video_path=video, output_path=self.temp_dir / "out.mp4",
            cues=[self._cue()], config=config,
        )
        command = sre.build_subtitle_ffmpeg_command(request, config, resolved_fonts=[None])
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


if __name__ == "__main__":
    unittest.main()
