from __future__ import annotations

import dataclasses
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from .media_inspector import AudioStreamInfo, MediaInfo, ProcessResult, VideoStreamInfo
from .video_validator import (
    DISALLOWED_ACTIONS,
    UnknownProfileError,
    _rule_audio,
    _rule_moov_atom,
    default_validator_config_path,
    detect_moov_atom_position,
    format_human_report,
    load_validator_config,
    parse_arguments,
    validate_file,
    validate_media,
    _run,
)

MODULE_PATH = Path(__file__).resolve().parent / "video_validator.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthetic ISO-BMFF box builders (no real video content needed — only
# top-level box headers matter for detect_moov_atom_position()).
# ---------------------------------------------------------------------------


def _box(box_type: bytes, payload: bytes = b"") -> bytes:
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + box_type + payload


def _write_mp4_moov_at_start(path: Path) -> None:
    data = (
        _box(b"ftyp", b"isom" + b"\x00" * 12)
        + _box(b"moov", b"x" * 40)
        + _box(b"mdat", b"y" * 200)
    )
    path.write_bytes(data)


def _write_mp4_moov_at_end(path: Path) -> None:
    data = (
        _box(b"ftyp", b"isom" + b"\x00" * 12)
        + _box(b"mdat", b"y" * 200)
        + _box(b"moov", b"x" * 40)
    )
    path.write_bytes(data)


# ---------------------------------------------------------------------------
# MediaInfo fixture builders (mocked MediaInfo — no ffprobe, no real
# media files needed except where the moov_atom rule specifically
# requires a real file on disk to read box headers from).
# ---------------------------------------------------------------------------


def _default_video_stream(**overrides) -> VideoStreamInfo:
    defaults = dict(
        stream_index=0,
        codec_name="h264",
        codec_long_name="H.264 / AVC",
        profile="High",
        width=1080,
        height=1920,
        fps=30.0,
        average_fps=30.0,
        duration_seconds=15.0,
        bitrate=8_000_000,
        frame_count=450,
        rotation_degrees=0,
        disposition_default=True,
        pixel_format="yuv420p",
    )
    defaults.update(overrides)
    return VideoStreamInfo(**defaults)


def _default_audio_stream(**overrides) -> AudioStreamInfo:
    defaults = dict(
        stream_index=1,
        codec_name="aac",
        sample_rate=48000,
        channels=2,
        channel_layout="stereo",
        bitrate=128000,
        duration_seconds=15.0,
        disposition_default=True,
    )
    defaults.update(overrides)
    return AudioStreamInfo(**defaults)


def _valid_reel_media(path, *, video_streams=None, audio_streams=None, **overrides) -> MediaInfo:
    resolved_path = Path(path)
    videos = video_streams if video_streams is not None else [_default_video_stream()]
    audios = audio_streams if audio_streams is not None else [_default_audio_stream()]

    defaults = dict(
        path=str(resolved_path),
        filename=resolved_path.name,
        extension=resolved_path.suffix.lower(),
        file_size_bytes=10_000_000,
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        format_long_name="QuickTime / MOV",
        duration_seconds=15.0,
        overall_bitrate=8_000_000,
        start_time_seconds=0.0,
        stream_count=len(videos) + len(audios),
        video_streams=videos,
        audio_streams=audios,
        chapters=[],
        has_video=len(videos) > 0,
        has_audio=len(audios) > 0,
        primary_video_stream_index=videos[0].stream_index if videos else None,
        primary_audio_stream_index=audios[0].stream_index if audios else None,
        warnings=[],
    )
    defaults.update(overrides)
    return MediaInfo(**defaults)


class ValidatorTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = load_validator_config()
        self.profile = self.config.profiles["instagram_reel"]

    def _reel_path(self, name: str = "reel.mp4", *, moov_at_start: bool = True) -> Path:
        path = self.temp_dir / name
        if moov_at_start:
            _write_mp4_moov_at_start(path)
        else:
            _write_mp4_moov_at_end(path)
        return path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_path_exists(self) -> None:
        self.assertTrue(default_validator_config_path().is_file())

    def test_real_config_loads_instagram_reel_profile(self) -> None:
        config = load_validator_config()
        self.assertIn("instagram_reel", config.profiles)
        profile = config.profiles["instagram_reel"]
        self.assertEqual(profile.duration_min_seconds, 3)
        self.assertEqual(profile.duration_max_seconds, 90)
        self.assertAlmostEqual(profile.aspect_ratio_target, 9 / 16, places=4)
        self.assertIn("h264", profile.video_codec_allowed)
        self.assertTrue(profile.audio_required)
        self.assertTrue(profile.vertical_required)
        self.assertEqual(config.starting_score, 100)
        self.assertEqual(config.minimum_score, 0)


# ---------------------------------------------------------------------------
# Full pipeline: valid reel + one-field-wrong variants
# ---------------------------------------------------------------------------


class FullPipelineTests(ValidatorTempTestCase):
    def test_valid_reel_passes_with_score_100(self) -> None:
        path = self._reel_path(moov_at_start=True)
        media = _valid_reel_media(path)
        result = validate_media(media, self.profile, starting_score=100, minimum_score=0)
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 100)
        self.assertEqual(result.failed_checks, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(len(result.checks), 13)

    def test_wrong_duration_fails(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(path, duration_seconds=200.0)  # over max 90s
        result = validate_media(media, self.profile)
        self.assertFalse(result.passed)
        self.assertTrue(any(c.rule == "duration" for c in result.failed_checks))
        self.assertLess(result.score, 100)

    def test_wrong_resolution_fails(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(
            path, video_streams=[_default_video_stream(width=320, height=568)]
        )
        result = validate_media(media, self.profile)
        self.assertFalse(result.passed)
        self.assertTrue(any(c.rule == "resolution" for c in result.failed_checks))

    def test_wrong_fps_fails(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(
            path, video_streams=[_default_video_stream(fps=15.0, average_fps=15.0)]
        )
        result = validate_media(media, self.profile)
        self.assertFalse(result.passed)
        self.assertTrue(any(c.rule == "fps" for c in result.failed_checks))

    def test_wrong_codec_fails(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(path, video_streams=[_default_video_stream(codec_name="vp9")])
        result = validate_media(media, self.profile)
        self.assertFalse(result.passed)
        self.assertTrue(any(c.rule == "video_codec" for c in result.failed_checks))

    def test_wrong_aspect_ratio_fails(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(
            path, video_streams=[_default_video_stream(width=1920, height=1080)]
        )
        result = validate_media(media, self.profile)
        self.assertFalse(result.passed)
        self.assertTrue(any(c.rule == "aspect_ratio" for c in result.failed_checks))
        # 1920x1080 is also horizontal, not vertical.
        self.assertTrue(any(c.rule == "vertical_orientation" for c in result.failed_checks))

    def test_missing_audio_fails(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(path, audio_streams=[], has_audio=False, primary_audio_stream_index=None)
        result = validate_media(media, self.profile)
        self.assertFalse(result.passed)
        audio_check = next(c for c in result.checks if c.rule == "audio")
        self.assertEqual(audio_check.severity, "fail")
        self.assertIn(audio_check, result.failed_checks)

    def test_rotation_produces_warning_not_fail(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(
            path,
            video_streams=[_default_video_stream(rotation_degrees=90, width=1920, height=1080)],
        )
        result = validate_media(media, self.profile)
        rotation_check = next(c for c in result.checks if c.rule == "rotation")
        self.assertEqual(rotation_check.severity, "warning")
        self.assertIn(rotation_check, result.warnings)
        self.assertNotIn(rotation_check, result.failed_checks)

    def test_bitrate_out_of_range_produces_warning(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(path, overall_bitrate=1)  # far below min_bps
        result = validate_media(media, self.profile)
        bitrate_check = next(c for c in result.checks if c.rule == "bitrate")
        self.assertEqual(bitrate_check.severity, "warning")
        self.assertTrue(result.passed)  # warnings never fail the result

    def test_moov_at_end_produces_warning(self) -> None:
        path = self._reel_path(moov_at_start=False)
        media = _valid_reel_media(path)
        result = validate_media(media, self.profile)
        moov_check = next(c for c in result.checks if c.rule == "moov_atom")
        self.assertEqual(moov_check.severity, "warning")
        self.assertTrue(result.passed)

    def test_score_deducts_correctly_for_combination_of_failures(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(
            path,
            duration_seconds=200.0,  # duration_penalty 25
            video_streams=[_default_video_stream(codec_name="vp9")],  # video_codec_penalty 20
        )
        result = validate_media(media, self.profile, starting_score=100, minimum_score=0)
        self.assertEqual(result.score, 100 - self.profile.duration_penalty - self.profile.video_codec_penalty)

    def test_score_never_goes_below_minimum(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(
            path,
            duration_seconds=None,
            video_streams=[],
            audio_streams=[],
            has_video=False,
            has_audio=False,
            primary_video_stream_index=None,
            primary_audio_stream_index=None,
        )
        result = validate_media(media, self.profile, starting_score=100, minimum_score=0)
        self.assertGreaterEqual(result.score, 0)

    def test_summary_contains_ready_when_passed(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(path)
        result = validate_media(media, self.profile)
        self.assertIn("READY", result.summary)
        self.assertNotIn("NOT READY", result.summary)

    def test_summary_contains_not_ready_when_failed(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(path, duration_seconds=200.0)
        result = validate_media(media, self.profile)
        self.assertIn("NOT READY", result.summary)

    def test_format_human_report_contains_all_check_rules(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(path)
        result = validate_media(media, self.profile)
        report = format_human_report(result)
        for check in result.checks:
            self.assertIn(check.rule, report)
        self.assertIn("Score:", report)


# ---------------------------------------------------------------------------
# Individual rule unit tests (audio + moov_atom in isolation)
# ---------------------------------------------------------------------------


class RuleUnitTests(ValidatorTempTestCase):
    def test_audio_not_required_passes_without_audio(self) -> None:
        path = self._reel_path()
        media = _valid_reel_media(path, audio_streams=[], has_audio=False, primary_audio_stream_index=None)
        profile = dataclasses.replace(self.profile, audio_required=False)
        result = _rule_audio(media, profile)
        self.assertEqual(result.severity, "pass")

    def test_moov_atom_skipped_for_non_applicable_extension(self) -> None:
        path = self.temp_dir / "clip.webm"
        path.write_bytes(b"not-a-real-webm")
        media = _valid_reel_media(path)
        result = _rule_moov_atom(media, self.profile)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Moov atom detection
# ---------------------------------------------------------------------------


class MoovAtomDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)

    def test_moov_at_start_detected(self) -> None:
        path = self.temp_dir / "start.mp4"
        _write_mp4_moov_at_start(path)
        self.assertEqual(detect_moov_atom_position(path), "start")

    def test_moov_at_end_detected(self) -> None:
        path = self.temp_dir / "end.mp4"
        _write_mp4_moov_at_end(path)
        self.assertEqual(detect_moov_atom_position(path), "end")

    def test_malformed_file_returns_unknown(self) -> None:
        path = self.temp_dir / "garbage.mp4"
        path.write_bytes(b"\x00\x01\x02")  # shorter than one box header
        self.assertEqual(detect_moov_atom_position(path), "unknown")

    def test_missing_file_returns_unknown(self) -> None:
        path = self.temp_dir / "does_not_exist.mp4"
        self.assertEqual(detect_moov_atom_position(path), "unknown")

    def test_never_raises_on_zero_size_box(self) -> None:
        path = self.temp_dir / "zero_size.mp4"
        path.write_bytes((0).to_bytes(4, "big") + b"free" + b"padding")
        self.assertEqual(detect_moov_atom_position(path), "unknown")


# ---------------------------------------------------------------------------
# CLI / report
# ---------------------------------------------------------------------------


class FakeInspectorRunner:
    """Fakes the ffprobe layer (media_inspector's SubprocessRunner) so
    CLI-level tests never touch real ffprobe."""

    def __init__(self, probe_data: dict) -> None:
        self.probe_data = probe_data
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, timeout: int) -> ProcessResult:
        self.calls.append(command)
        return ProcessResult(returncode=0, stdout=json.dumps(self.probe_data))


def _probe_json_for_valid_reel() -> dict:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1080,
                "height": 1920,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "duration": "15.0",
                "bit_rate": "8000000",
                "pix_fmt": "yuv420p",
                "tags": {},
                "disposition": {"default": 1, "attached_pic": 0},
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "bit_rate": "128000",
                "duration": "15.0",
                "tags": {},
                "disposition": {"default": 1, "attached_pic": 0},
            },
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "format_long_name": "QuickTime / MOV",
            "duration": "15.0",
            "bit_rate": "8128000",
            "start_time": "0.0",
        },
        "chapters": [],
    }


class CliReportTests(ValidatorTempTestCase):
    def test_cli_json_flag_prints_valid_json(self) -> None:
        path = self._reel_path(moov_at_start=True)
        runner = FakeInspectorRunner(_probe_json_for_valid_reel())
        arguments = parse_arguments([str(path), "--json"])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = _run(arguments, runner=runner)
        self.assertEqual(exit_code, 0)
        parsed = json.loads(buffer.getvalue())
        self.assertIn("result", parsed)
        self.assertEqual(parsed["result"]["profile"], "instagram_reel")
        self.assertTrue(parsed["result"]["passed"])

    def test_cli_human_report_contains_expected_fields(self) -> None:
        path = self._reel_path(moov_at_start=True)
        runner = FakeInspectorRunner(_probe_json_for_valid_reel())
        arguments = parse_arguments([str(path)])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _run(arguments, runner=runner)
        output = buffer.getvalue()
        self.assertIn("File:", output)
        self.assertIn("Profile:", output)
        self.assertIn("Score:", output)
        self.assertIn("Checks:", output)

    def test_output_file_written(self) -> None:
        path = self._reel_path(moov_at_start=True)
        runner = FakeInspectorRunner(_probe_json_for_valid_reel())
        output_path = self.temp_dir / "report.json"
        arguments = parse_arguments([str(path), "--output", str(output_path)])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _run(arguments, runner=runner)
        self.assertTrue(output_path.is_file())
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)
        self.assertIn("schema_version", payload)

    def test_overwrite_refused_without_force(self) -> None:
        path = self._reel_path(moov_at_start=True)
        runner = FakeInspectorRunner(_probe_json_for_valid_reel())
        output_path = self.temp_dir / "report.json"
        output_path.write_text("existing", encoding="utf-8")
        arguments = parse_arguments([str(path), "--output", str(output_path)])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = _run(arguments, runner=runner)
        self.assertEqual(exit_code, 1)
        self.assertEqual(output_path.read_text(encoding="utf-8"), "existing")

    def test_force_flag_overwrites(self) -> None:
        path = self._reel_path(moov_at_start=True)
        runner = FakeInspectorRunner(_probe_json_for_valid_reel())
        output_path = self.temp_dir / "report.json"
        output_path.write_text("existing", encoding="utf-8")
        arguments = parse_arguments([str(path), "--output", str(output_path), "--force"])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = _run(arguments, runner=runner)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIn("result", payload)

    def test_profile_selection_defaults_to_instagram_reel(self) -> None:
        path = self._reel_path(moov_at_start=True)
        runner = FakeInspectorRunner(_probe_json_for_valid_reel())
        arguments = parse_arguments([str(path), "--json"])
        self.assertEqual(arguments.profile, "instagram_reel")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _run(arguments, runner=runner)
        parsed = json.loads(buffer.getvalue())
        self.assertEqual(parsed["result"]["profile"], "instagram_reel")

    def test_unknown_profile_produces_clear_error(self) -> None:
        path = self._reel_path(moov_at_start=True)
        runner = FakeInspectorRunner(_probe_json_for_valid_reel())
        arguments = parse_arguments([str(path), "--profile", "does_not_exist"])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = _run(arguments, runner=runner)
        self.assertEqual(exit_code, 1)
        self.assertIn("does_not_exist", buffer.getvalue())

    def test_validate_file_raises_unknown_profile_error_directly(self) -> None:
        path = self._reel_path(moov_at_start=True)
        runner = FakeInspectorRunner(_probe_json_for_valid_reel())
        with self.assertRaises(UnknownProfileError):
            validate_file(path, self.config, "does_not_exist", runner=runner)


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

    def test_no_publishing_module_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from src.publishing"))
            self.assertFalse(stripped.startswith("from .publishing"))

    def test_no_ffmpeg_render_invocation(self) -> None:
        self.assertNotIn('"ffmpeg"', MODULE_SOURCE)
        self.assertNotIn("'ffmpeg'", MODULE_SOURCE)

    def test_only_read_binary_mode_used_on_media_path(self) -> None:
        # detect_moov_atom_position must only ever open the inspected
        # file in read-binary mode ("rb"); no write mode against media.
        self.assertIn('"rb"', MODULE_SOURCE)
        self.assertNotIn('open("w"', MODULE_SOURCE)
        self.assertNotIn('open("wb"', MODULE_SOURCE)
        self.assertNotIn('open("a"', MODULE_SOURCE)

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("render", DISALLOWED_ACTIONS)
        self.assertIn("publish", DISALLOWED_ACTIONS)
        self.assertIn("transcode", DISALLOWED_ACTIONS)


if __name__ == "__main__":
    unittest.main()
