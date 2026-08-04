from __future__ import annotations

import copy
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from .media_inspector import (
    DISALLOWED_ACTIONS,
    FFprobeExecutionError,
    FFprobeNotFoundError,
    FFprobeOutputError,
    FFprobeTimeoutError,
    InspectorConfig,
    MediaFileEmptyError,
    MediaFileNotFoundError,
    MediaInspectorConfigError,
    MediaInspectorError,
    ProcessResult,
    ReportAlreadyExistsError,
    UnsafeReportPathError,
    UnsupportedMediaExtensionError,
    build_directory_envelope,
    build_media_info,
    build_single_file_envelope,
    default_inspector_config_path,
    default_runner,
    format_human_report,
    inspect_directory,
    inspect_file,
    load_inspector_config,
    parse_arguments,
    run_ffprobe,
    select_primary_stream_index,
    validate_media_path,
    write_report,
    _extract_rotation,
    _format_duration,
    _format_file_size,
    _run,
    _safe_float,
    _safe_int,
    _safe_rational,
)

MODULE_PATH = Path(__file__).resolve().parent / "media_inspector.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_probe_data() -> dict:
    return {
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                "profile": "High",
                "codec_type": "video",
                "codec_tag_string": "avc1",
                "codec_tag": "0x31637661",
                "width": 1080,
                "height": 1920,
                "coded_width": 1080,
                "coded_height": 1920,
                "display_aspect_ratio": "9:16",
                "sample_aspect_ratio": "1:1",
                "pix_fmt": "yuv420p",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
                "color_range": "tv",
                "field_order": "progressive",
                "r_frame_rate": "30000/1001",
                "avg_frame_rate": "30000/1001",
                "time_base": "1/30000",
                "duration": "12.540000",
                "bit_rate": "8000000",
                "nb_frames": "376",
                "tags": {"language": "und"},
                "disposition": {"default": 1, "attached_pic": 0},
            },
            {
                "index": 1,
                "codec_name": "aac",
                "codec_long_name": "AAC (Advanced Audio Coding)",
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "sample_fmt": "fltp",
                "bit_rate": "128000",
                "duration": "12.540000",
                "tags": {"language": "und"},
                "disposition": {"default": 1, "attached_pic": 0},
            },
        ],
        "format": {
            "filename": "input.mp4",
            "nb_streams": 2,
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "format_long_name": "QuickTime / MOV",
            "start_time": "0.000000",
            "duration": "12.540000",
            "size": "1234567",
            "bit_rate": "8128000",
            "tags": {},
        },
        "chapters": [],
    }


class FakeRunner:
    def __init__(
        self,
        *,
        probe_data: dict | None = None,
        stdout: str | None = None,
        returncode: int = 0,
        stderr: str = "",
    ) -> None:
        self.probe_data = probe_data
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[dict] = []

    def __call__(self, command: list[str], *, timeout: int) -> ProcessResult:
        self.calls.append({"command": command, "timeout": timeout})
        stdout = self.stdout if self.stdout is not None else json.dumps(self.probe_data or {})
        return ProcessResult(returncode=self.returncode, stdout=stdout, stderr=self.stderr)


class MediaTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = InspectorConfig()

    def _write_file(self, name: str, *, size: int = 100) -> Path:
        path = self.temp_dir / name
        path.write_bytes(b"x" * size)
        return path


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------


class FileValidationTests(MediaTempTestCase):
    def test_missing_file_fails(self) -> None:
        with self.assertRaises(MediaFileNotFoundError):
            validate_media_path(self.temp_dir / "does_not_exist.mp4", self.config)

    def test_directory_passed_as_file_fails(self) -> None:
        with self.assertRaises(MediaInspectorError):
            validate_media_path(self.temp_dir, self.config)

    def test_zero_byte_file_fails(self) -> None:
        path = self._write_file("empty.mp4", size=0)
        with self.assertRaises(MediaFileEmptyError):
            validate_media_path(path, self.config)

    def test_unsupported_extension_fails(self) -> None:
        path = self._write_file("notes.txt")
        with self.assertRaises(UnsupportedMediaExtensionError):
            validate_media_path(path, self.config)

    def test_valid_supported_extension_reaches_runner(self) -> None:
        path = self._write_file("clip.mp4")
        runner = FakeRunner(probe_data=_base_probe_data())
        inspect_file(path, self.config, runner=runner)
        self.assertEqual(len(runner.calls), 1)


# ---------------------------------------------------------------------------
# Runner / subprocess safety
# ---------------------------------------------------------------------------


class RunnerTests(MediaTempTestCase):
    def test_command_is_a_list(self) -> None:
        path = self._write_file("clip.mp4")
        runner = FakeRunner(probe_data=_base_probe_data())
        inspect_file(path, self.config, runner=runner)
        self.assertIsInstance(runner.calls[0]["command"], list)

    def test_shell_true_never_used(self) -> None:
        self.assertNotIn("shell=True", MODULE_SOURCE)

    def test_exact_file_path_passed_once(self) -> None:
        path = self._write_file("clip.mp4")
        runner = FakeRunner(probe_data=_base_probe_data())
        inspect_file(path, self.config, runner=runner)
        command = runner.calls[0]["command"]
        self.assertEqual(command.count(str(path.resolve())), 1)

    def test_timeout_is_used(self) -> None:
        path = self._write_file("clip.mp4")
        config = InspectorConfig(timeout_seconds=17)
        runner = FakeRunner(probe_data=_base_probe_data())
        inspect_file(path, config, runner=runner)
        self.assertEqual(runner.calls[0]["timeout"], 17)

    def test_missing_ffprobe_produces_not_found_error(self) -> None:
        with self.assertRaises(FFprobeNotFoundError):
            default_runner(["/definitely/not/a/real/binary/ffprobe_xyz"], timeout=5)

    def test_timeout_produces_ffprobe_timeout_error(self) -> None:
        with mock.patch(
            "src.media_inspector.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=1),
        ):
            with self.assertRaises(FFprobeTimeoutError):
                default_runner(["ffprobe"], timeout=1)

    def test_non_zero_exit_produces_execution_error(self) -> None:
        path = self._write_file("clip.mp4")
        runner = FakeRunner(returncode=1, stderr="boom")
        with self.assertRaises(FFprobeExecutionError):
            inspect_file(path, self.config, runner=runner)

    def test_invalid_json_produces_output_error(self) -> None:
        path = self._write_file("clip.mp4")
        runner = FakeRunner(stdout="not valid json {{{")
        with self.assertRaises(FFprobeOutputError):
            inspect_file(path, self.config, runner=runner)

    def test_missing_streams_key_produces_output_error(self) -> None:
        path = self._write_file("clip.mp4")
        runner = FakeRunner(stdout=json.dumps({"format": {}}))
        with self.assertRaises(FFprobeOutputError):
            inspect_file(path, self.config, runner=runner)


# ---------------------------------------------------------------------------
# Safe parsing helpers
# ---------------------------------------------------------------------------


class SafeParsingTests(unittest.TestCase):
    def test_safe_int_handles_none_and_na(self) -> None:
        self.assertIsNone(_safe_int(None))
        self.assertIsNone(_safe_int("N/A"))
        self.assertEqual(_safe_int("42"), 42)

    def test_safe_float_handles_none_and_na(self) -> None:
        self.assertIsNone(_safe_float(None))
        self.assertIsNone(_safe_float("N/A"))
        self.assertAlmostEqual(_safe_float("1.5"), 1.5)

    def test_rational_30000_1001_parses_accurately(self) -> None:
        result = _safe_rational("30000/1001")
        self.assertAlmostEqual(result, 30000 / 1001, places=6)

    def test_rational_zero_over_zero_returns_none(self) -> None:
        self.assertIsNone(_safe_rational("0/0"))

    def test_rational_na_returns_none(self) -> None:
        self.assertIsNone(_safe_rational("N/A"))

    def test_rational_missing_denominator_does_not_raise(self) -> None:
        self.assertIsNone(_safe_rational("30/"))

    def test_rational_plain_number(self) -> None:
        self.assertAlmostEqual(_safe_rational("25"), 25.0)


# ---------------------------------------------------------------------------
# Parsing full documents
# ---------------------------------------------------------------------------


class ParsingTests(MediaTempTestCase):
    def test_valid_video_audio_json_parses(self) -> None:
        media = build_media_info(self._write_file("clip.mp4"), _base_probe_data(), self.config)
        self.assertTrue(media.has_video)
        self.assertTrue(media.has_audio)
        self.assertEqual(len(media.video_streams), 1)
        self.assertEqual(len(media.audio_streams), 1)
        self.assertEqual(media.video_streams[0].codec_name, "h264")
        self.assertEqual(media.audio_streams[0].codec_name, "aac")

    def test_missing_optional_fields_do_not_crash(self) -> None:
        data = _base_probe_data()
        del data["streams"][0]["profile"]
        del data["streams"][0]["bit_rate"]
        del data["streams"][0]["nb_frames"]
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        video = media.video_streams[0]
        self.assertIsNone(video.profile)
        self.assertIsNone(video.bitrate)
        self.assertIsNone(video.frame_count)

    def test_rational_fps_parses_accurately_end_to_end(self) -> None:
        media = build_media_info(self._write_file("clip.mp4"), _base_probe_data(), self.config)
        self.assertAlmostEqual(media.video_streams[0].fps, 30000 / 1001, places=6)

    def test_zero_over_zero_frame_rate_returns_null_and_warning(self) -> None:
        data = _base_probe_data()
        data["streams"][0]["r_frame_rate"] = "0/0"
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertIsNone(media.video_streams[0].fps)
        self.assertTrue(
            any(w.code == "invalid_rational_value" for w in media.warnings), media.warnings
        )

    def test_na_bitrate_returns_null(self) -> None:
        data = _base_probe_data()
        data["streams"][0]["bit_rate"] = "N/A"
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertIsNone(media.video_streams[0].bitrate)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class RotationTests(unittest.TestCase):
    def test_rotation_from_tags_parses(self) -> None:
        stream = {"tags": {"rotate": "90"}}
        rotation, warnings = _extract_rotation(stream)
        self.assertEqual(rotation, 90)
        self.assertEqual(warnings, [])

    def test_rotation_from_side_data_parses(self) -> None:
        stream = {
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90.0}]
        }
        rotation, warnings = _extract_rotation(stream)
        self.assertEqual(rotation, 270)
        self.assertEqual(warnings, [])

    def test_contradictory_rotation_emits_warning_and_prefers_display_matrix(self) -> None:
        stream = {
            "tags": {"rotate": "90"},
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": 180.0}],
        }
        rotation, warnings = _extract_rotation(stream)
        self.assertEqual(rotation, 180)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "contradictory_rotation_metadata")

    def test_no_rotation_metadata_returns_none(self) -> None:
        rotation, warnings = _extract_rotation({})
        self.assertIsNone(rotation)
        self.assertEqual(warnings, [])


# ---------------------------------------------------------------------------
# Primary stream selection
# ---------------------------------------------------------------------------


class PrimaryStreamSelectionTests(unittest.TestCase):
    def test_attached_picture_is_not_selected_as_primary_when_real_video_exists(self) -> None:
        streams = [
            {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 1, "default": 0}},
            {"index": 1, "codec_type": "video", "disposition": {"attached_pic": 0, "default": 0}},
        ]
        self.assertEqual(select_primary_stream_index(streams, "video"), 1)

    def test_attached_picture_used_as_fallback_when_no_real_video_exists(self) -> None:
        streams = [
            {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 1, "default": 0}},
        ]
        self.assertEqual(select_primary_stream_index(streams, "video"), 0)

    def test_default_disposition_wins_primary_selection(self) -> None:
        streams = [
            {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 0, "default": 0}},
            {"index": 1, "codec_type": "video", "disposition": {"attached_pic": 0, "default": 1}},
        ]
        self.assertEqual(select_primary_stream_index(streams, "video"), 1)

    def test_lowest_index_is_deterministic_fallback(self) -> None:
        streams = [
            {"index": 2, "codec_type": "video", "disposition": {"attached_pic": 0, "default": 0}},
            {"index": 1, "codec_type": "video", "disposition": {"attached_pic": 0, "default": 0}},
        ]
        self.assertEqual(select_primary_stream_index(streams, "video"), 1)

    def test_no_matching_streams_returns_none(self) -> None:
        self.assertIsNone(select_primary_stream_index([], "video"))


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


class DerivedPropertyTests(MediaTempTestCase):
    def test_vertical_media_detected(self) -> None:
        data = _base_probe_data()  # 1080x1920
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(media.is_vertical)
        self.assertFalse(media.is_horizontal)

    def test_horizontal_media_detected(self) -> None:
        data = _base_probe_data()
        data["streams"][0]["width"] = 1920
        data["streams"][0]["height"] = 1080
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(media.is_horizontal)
        self.assertFalse(media.is_vertical)

    def test_square_media_detected(self) -> None:
        data = _base_probe_data()
        data["streams"][0]["width"] = 1000
        data["streams"][0]["height"] = 1000
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(media.is_square)

    def test_rotated_display_dimensions_swap_correctly(self) -> None:
        data = _base_probe_data()
        data["streams"][0]["width"] = 1920
        data["streams"][0]["height"] = 1080
        data["streams"][0]["tags"]["rotate"] = "90"
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertEqual(media.resolution, "1920x1080")
        self.assertEqual(media.display_resolution_after_rotation, "1080x1920")

    def test_duration_text_formatting(self) -> None:
        self.assertEqual(_format_duration(12.54), "00:00:12.540")
        self.assertEqual(_format_duration(3661.0), "01:01:01.000")

    def test_file_size_text_formatting(self) -> None:
        self.assertEqual(_format_file_size(500), "500 B")
        self.assertEqual(_format_file_size(2 * 1024 * 1024), "2.0 MB")
        self.assertEqual(_format_file_size(3 * 1024 * 1024 * 1024), "3.0 GB")


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class WarningTests(MediaTempTestCase):
    def test_no_audio_warning(self) -> None:
        data = _base_probe_data()
        data["streams"] = [s for s in data["streams"] if s["codec_type"] != "audio"]
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertFalse(media.has_audio)
        self.assertTrue(any(w.code == "no_audio_stream" for w in media.warnings))

    def test_no_video_warning(self) -> None:
        data = _base_probe_data()
        data["streams"] = [s for s in data["streams"] if s["codec_type"] != "video"]
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertFalse(media.has_video)
        self.assertTrue(any(w.code == "no_video_stream" for w in media.warnings))

    def test_multiple_video_streams_warning(self) -> None:
        data = _base_probe_data()
        second_video = copy.deepcopy(data["streams"][0])
        second_video["index"] = 2
        second_video["disposition"]["default"] = 0
        data["streams"].append(second_video)
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(any(w.code == "multiple_video_streams" for w in media.warnings))

    def test_multiple_audio_streams_warning(self) -> None:
        data = _base_probe_data()
        second_audio = copy.deepcopy(data["streams"][1])
        second_audio["index"] = 2
        second_audio["disposition"]["default"] = 0
        data["streams"].append(second_audio)
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(any(w.code == "multiple_audio_streams" for w in media.warnings))

    def test_container_stream_duration_mismatch_warning(self) -> None:
        data = _base_probe_data()
        data["format"]["duration"] = "20.0"  # stream duration is 12.54
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(
            any(w.code == "container_stream_duration_mismatch" for w in media.warnings)
        )

    def test_missing_bitrate_warning(self) -> None:
        data = _base_probe_data()
        del data["format"]["bit_rate"]
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(any(w.code == "missing_bitrate" for w in media.warnings))

    def test_missing_duration_warning(self) -> None:
        data = _base_probe_data()
        del data["format"]["duration"]
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(any(w.code == "missing_duration" for w in media.warnings))

    def test_suspected_variable_fps_warning(self) -> None:
        data = _base_probe_data()
        data["streams"][0]["r_frame_rate"] = "30/1"
        data["streams"][0]["avg_frame_rate"] = "24/1"
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(
            any(w.code == "variable_frame_rate_suspected" for w in media.warnings)
        )

    def test_attached_picture_present_warning(self) -> None:
        data = _base_probe_data()
        data["streams"][0]["disposition"]["attached_pic"] = 1
        media = build_media_info(self._write_file("clip.mp4"), data, self.config)
        self.assertTrue(any(w.code == "attached_picture_present" for w in media.warnings))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_path_exists(self) -> None:
        self.assertTrue(default_inspector_config_path().is_file())

    def test_real_config_loads_with_expected_defaults(self) -> None:
        config = load_inspector_config()
        self.assertEqual(config.ffprobe_binary, "ffprobe")
        self.assertIn(".mp4", config.video_extensions)
        self.assertIn(".mp3", config.audio_extensions)
        self.assertTrue(config.overwrite_requires_force)

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(MediaInspectorConfigError):
            load_inspector_config("/nonexistent/inspector.yaml")


# ---------------------------------------------------------------------------
# CLI / report
# ---------------------------------------------------------------------------


class CliReportTests(MediaTempTestCase):
    def test_human_report_contains_expected_metadata(self) -> None:
        media = build_media_info(self._write_file("clip.mp4"), _base_probe_data(), self.config)
        report = format_human_report(media)
        self.assertIn("File:", report)
        self.assertIn("Size:", report)
        self.assertIn("Container:", report)
        self.assertIn("Duration:", report)
        self.assertIn("Primary video:", report)
        self.assertIn("Primary audio:", report)
        self.assertIn("Streams:", report)
        self.assertIn("Warnings", report)
        self.assertIn("h264", report)

    def test_json_output_is_valid_json(self) -> None:
        media = build_media_info(self._write_file("clip.mp4"), _base_probe_data(), self.config)
        envelope = build_single_file_envelope(media, self.config)
        text = json.dumps(envelope)
        parsed = json.loads(text)
        self.assertEqual(parsed["schema_version"], self.config.report_schema_version)
        self.assertIn("inspected_at", parsed)
        self.assertIn("media", parsed)

    def test_cli_json_flag_prints_valid_json(self) -> None:
        path = self._write_file("clip.mp4")
        runner = FakeRunner(probe_data=_base_probe_data())
        arguments = parse_arguments([str(path), "--json"])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = _run(arguments, runner=runner)
        self.assertEqual(exit_code, 0)
        parsed = json.loads(buffer.getvalue())
        self.assertIn("media", parsed)

    def test_output_file_is_atomic_json(self) -> None:
        media = build_media_info(self._write_file("clip.mp4"), _base_probe_data(), self.config)
        envelope = build_single_file_envelope(media, self.config)
        output_path = self.temp_dir / "report.json"
        write_report(envelope, output_path, force=False)
        self.assertTrue(output_path.is_file())
        self.assertFalse(output_path.with_suffix(".json.tmp").exists())
        parsed = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["media"]["filename"], "clip.mp4")

    def test_overwrite_refused_without_force(self) -> None:
        media = build_media_info(self._write_file("clip.mp4"), _base_probe_data(), self.config)
        envelope = build_single_file_envelope(media, self.config)
        output_path = self.temp_dir / "report.json"
        output_path.write_text("existing", encoding="utf-8")
        with self.assertRaises(ReportAlreadyExistsError):
            write_report(envelope, output_path, force=False)
        self.assertEqual(output_path.read_text(encoding="utf-8"), "existing")

    def test_force_overwrites(self) -> None:
        media = build_media_info(self._write_file("clip.mp4"), _base_probe_data(), self.config)
        envelope = build_single_file_envelope(media, self.config)
        output_path = self.temp_dir / "report.json"
        output_path.write_text("existing", encoding="utf-8")
        write_report(envelope, output_path, force=True)
        parsed = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIn("media", parsed)

    def test_input_output_same_path_refused(self) -> None:
        input_path = self._write_file("clip.mp4")
        media = build_media_info(input_path, _base_probe_data(), self.config)
        envelope = build_single_file_envelope(media, self.config)
        with self.assertRaises(UnsafeReportPathError):
            write_report(envelope, input_path, force=True, input_path=input_path)

    def test_parse_arguments_requires_exactly_one_of_file_or_directory(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments([])
        with self.assertRaises(SystemExit):
            parse_arguments(["file.mp4", "--directory", "/tmp"])


# ---------------------------------------------------------------------------
# Directory inspection
# ---------------------------------------------------------------------------


class DirectoryInspectionTests(MediaTempTestCase):
    def test_directory_results_are_sorted(self) -> None:
        self._write_file("b.mp4")
        self._write_file("a.mp4")
        self._write_file("c.mp4")
        runner = FakeRunner(probe_data=_base_probe_data())
        result = inspect_directory(self.temp_dir, self.config, runner=runner)
        names = [Path(r.path).name for r in result.results]
        self.assertEqual(names, sorted(names))

    def test_recursive_directory_mode_works(self) -> None:
        subdir = self.temp_dir / "sub"
        subdir.mkdir()
        (subdir / "nested.mp4").write_bytes(b"x" * 10)
        self._write_file("top.mp4")
        runner = FakeRunner(probe_data=_base_probe_data())

        non_recursive = inspect_directory(self.temp_dir, self.config, recursive=False, runner=runner)
        self.assertEqual(non_recursive.total_files, 1)

        recursive = inspect_directory(self.temp_dir, self.config, recursive=True, runner=runner)
        self.assertEqual(recursive.total_files, 2)

    def test_unsupported_files_skipped(self) -> None:
        self._write_file("clip.mp4")
        self._write_file("notes.txt")
        runner = FakeRunner(probe_data=_base_probe_data())
        result = inspect_directory(self.temp_dir, self.config, runner=runner)
        self.assertEqual(result.total_files, 1)

    def test_one_corrupt_file_does_not_abort_directory_run(self) -> None:
        self._write_file("good.mp4")
        self._write_file("bad.mp4")

        class SelectiveRunner:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, command, *, timeout):
                self.calls += 1
                path = command[-1]
                if "bad.mp4" in path:
                    return ProcessResult(returncode=1, stderr="simulated failure")
                return ProcessResult(returncode=0, stdout=json.dumps(_base_probe_data()))

        runner = SelectiveRunner()
        result = inspect_directory(self.temp_dir, self.config, runner=runner)
        self.assertEqual(result.total_files, 2)
        self.assertEqual(result.successful, 1)
        self.assertEqual(result.failed, 1)

    def test_no_supported_files_exits_safely(self) -> None:
        self._write_file("notes.txt")
        runner = FakeRunner(probe_data=_base_probe_data())
        result = inspect_directory(self.temp_dir, self.config, runner=runner)
        self.assertEqual(result.total_files, 0)

        arguments = parse_arguments(["--directory", str(self.temp_dir)])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = _run(arguments, runner=runner)
        self.assertNotEqual(exit_code, 0)

    def test_directory_envelope_shape(self) -> None:
        self._write_file("clip.mp4")
        runner = FakeRunner(probe_data=_base_probe_data())
        result = inspect_directory(self.temp_dir, self.config, runner=runner)
        envelope = build_directory_envelope(result, self.config)
        for key in (
            "schema_version",
            "inspected_at",
            "directory",
            "recursive",
            "total_files",
            "successful",
            "failed",
            "results",
        ):
            self.assertIn(key, envelope)


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
            self.assertFalse(stripped.startswith("import src.social"))
        self.assertNotIn("InstagramSession(", MODULE_SOURCE)

    def test_no_publishing_module_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from src.publishing"))
            self.assertFalse(stripped.startswith("from .publishing"))
            self.assertFalse(stripped.startswith("import src.publishing"))

    def test_no_ffmpeg_render_invocation(self) -> None:
        # ffprobe is expected throughout; "ffmpeg" itself (the renderer)
        # must never appear as a command/binary reference.
        self.assertNotIn('"ffmpeg"', MODULE_SOURCE)
        self.assertNotIn("'ffmpeg'", MODULE_SOURCE)

    def test_no_file_mutation_of_inspected_media(self) -> None:
        for forbidden in (".write_bytes(", ".write_text(", "os.remove(", "unlink("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("render", DISALLOWED_ACTIONS)
        self.assertIn("publish", DISALLOWED_ACTIONS)
        self.assertIn("transcode", DISALLOWED_ACTIONS)


if __name__ == "__main__":
    unittest.main()
