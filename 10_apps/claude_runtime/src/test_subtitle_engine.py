from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import timeline_engine
from .subtitle_engine import (
    LineBreakMode,
    SubtitleAlignment,
    SubtitleConfig,
    SubtitleCue,
    SubtitleDocument,
    SubtitleInputError,
    SubtitleJSONError,
    SubtitleOutputExistsError,
    SubtitlePosition,
    SubtitleSRTError,
    SubtitleStyle,
    SubtitleStyleError,
    SubtitleTrack,
    SubtitleValidationError,
    SubtitleWarning,
    TimelineCompatibilityError,
    UnsafeSubtitleOutputError,
    build_subtitle_document,
    document_from_dict,
    document_to_dict,
    export_ass,
    export_srt,
    load_subtitle_config,
    load_subtitle_document,
    load_subtitle_source,
    load_timeline_for_subtitles,
    normalize_text,
    parse_arguments,
    parse_srt,
    plan_cue_line_wrap,
    plan_cue_position,
    save_subtitle_document,
    snap_to_frame_rate,
    validate_position_within_safe_area,
    validate_subtitle_document,
)

MODULE_PATH = Path(__file__).resolve().parent / "subtitle_engine.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


class SubtitleTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = load_subtitle_config()

    def _write_json(self, name: str, data: dict) -> Path:
        path = self.temp_dir / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _write_text(self, name: str, content: str) -> Path:
        path = self.temp_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def _entries_source(self, entries: list[dict], **extra) -> dict:
        payload = {"language": "en", "entries": entries}
        payload.update(extra)
        return payload


def _basic_valid_document(config: SubtitleConfig) -> SubtitleDocument:
    track = SubtitleTrack(
        track_id="track_subtitles",
        language="en",
        default_style_id="default",
        cues=[
            SubtitleCue(
                cue_id="cue_0001",
                track_id="track_subtitles",
                start_seconds=0.0,
                end_seconds=2.0,
                duration_seconds=2.0,
                text="hello there",
                style_id="default",
            ),
            SubtitleCue(
                cue_id="cue_0002",
                track_id="track_subtitles",
                start_seconds=2.0,
                end_seconds=4.0,
                duration_seconds=2.0,
                text="general kenobi",
                style_id="default",
            ),
        ],
    )
    return SubtitleDocument(
        schema_version="1.0",
        document_id="abc123",
        language="en",
        duration_seconds=4.0,
        tracks=[track],
        styles={"default": config.styles["default"]},
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_loads_from_repo(self) -> None:
        config = load_subtitle_config()
        self.assertEqual(config.default_language, "en")
        self.assertIn("default", config.styles)
        self.assertIn("luxury_editorial", config.styles)
        self.assertIn("traditional_chinese_caption", config.styles)

    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(Exception):
            load_subtitle_config("/nonexistent/subtitles.yaml")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class InputTests(SubtitleTempTestCase):
    def test_valid_entries_json_loads(self) -> None:
        source = self._write_json("source.json", self._entries_source([
            {"text": "hello", "start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        document = build_subtitle_document(source, self.config)
        self.assertEqual(len(document.tracks[0].cues), 1)

    def test_malformed_json_fails(self) -> None:
        source = self._write_text("bad.json", "{not valid json")
        with self.assertRaises(SubtitleJSONError):
            build_subtitle_document(source, self.config)

    def test_valid_srt_imports(self) -> None:
        srt = (
            "1\n00:00:00,000 --> 00:00:02,000\nHello there\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\nGeneral Kenobi\n"
        )
        source = self._write_text("source.srt", srt)
        document = build_subtitle_document(source, self.config)
        self.assertEqual(len(document.tracks[0].cues), 2)
        self.assertEqual(document.tracks[0].cues[0].text, "Hello there")

    def test_malformed_srt_fails(self) -> None:
        source = self._write_text("bad.srt", "1\nnot a timestamp\ntext\n")
        with self.assertRaises(SubtitleSRTError):
            build_subtitle_document(source, self.config)

    def test_unicode_cjk_emoji_preserved(self) -> None:
        text = "京都の静かな朝 🌸 早晨"
        source = self._write_json("source.json", self._entries_source([
            {"text": text, "start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        document = build_subtitle_document(source, self.config)
        self.assertEqual(document.tracks[0].cues[0].text, text)

    def test_empty_entries_list_fails(self) -> None:
        source = self._write_json("source.json", self._entries_source([]))
        with self.assertRaises(SubtitleInputError):
            build_subtitle_document(source, self.config)

    def test_missing_text_fails(self) -> None:
        source = self._write_json("source.json", self._entries_source([
            {"start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        with self.assertRaises(SubtitleInputError):
            build_subtitle_document(source, self.config)

    def test_unknown_metadata_preserved(self) -> None:
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0, "metadata": {"take": 3}},
        ]))
        document = build_subtitle_document(source, self.config)
        self.assertEqual(document.tracks[0].cues[0].metadata["take"], 3)

    def test_existing_plan_json_loads(self) -> None:
        plan = document_to_dict(_basic_valid_document(self.config))
        source = self._write_json("plan.json", plan)
        document = build_subtitle_document(source, self.config)
        self.assertEqual(len(document.tracks[0].cues), 2)

    def test_empty_input_file_fails(self) -> None:
        source = self._write_text("empty.json", "")
        with self.assertRaises(SubtitleInputError):
            build_subtitle_document(source, self.config)

    def test_missing_input_file_fails(self) -> None:
        with self.assertRaises(SubtitleInputError):
            build_subtitle_document(self.temp_dir / "does_not_exist.json", self.config)

    def test_unrecognized_json_shape_fails(self) -> None:
        source = self._write_json("source.json", {"foo": "bar"})
        with self.assertRaises(SubtitleInputError):
            build_subtitle_document(source, self.config)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class TimingValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_subtitle_config()

    def test_exact_timing_passes(self) -> None:
        result = validate_subtitle_document(_basic_valid_document(self.config), self.config)
        self.assertTrue(result.passed)

    def test_negative_start_fails(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].start_seconds = -1.0
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_zero_duration_fails(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].end_seconds = document.tracks[0].cues[0].start_seconds
        document.tracks[0].cues[0].duration_seconds = 0.0
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_end_before_start_fails(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].end_seconds = -1.0
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_overlap_fails_by_default(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[1].start_seconds = 1.0
        document.tracks[0].cues[1].end_seconds = 3.0
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_overlap_allowed_when_configured(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[1].start_seconds = 1.0
        document.tracks[0].cues[1].end_seconds = 3.0
        config = SubtitleConfig(allow_overlaps=True, styles=self.config.styles)
        result = validate_subtitle_document(document, config)
        self.assertTrue(result.passed)

    def test_sorted_cues_pass(self) -> None:
        result = validate_subtitle_document(_basic_valid_document(self.config), self.config)
        self.assertTrue(result.passed)

    def test_unsorted_cues_fail_at_validate(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues.reverse()
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_unsorted_cues_normalized_at_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.json"
            source.write_text(json.dumps({
                "language": "en",
                "entries": [
                    {"text": "second", "start_seconds": 2.0, "end_seconds": 4.0},
                    {"text": "first", "start_seconds": 0.0, "end_seconds": 2.0},
                ],
            }))
            document = build_subtitle_document(source, self.config)
            self.assertEqual([c.text for c in document.tracks[0].cues], ["first", "second"])

    def test_duration_beyond_timeline_fails(self) -> None:
        document = _basic_valid_document(self.config)
        fake_timeline = timeline_engine.Timeline(timeline_id="t1", duration_seconds=1.0, tracks=[])
        result = validate_subtitle_document(document, self.config, timeline=fake_timeline)
        self.assertFalse(result.passed)

    def test_minimum_duration_enforced(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].end_seconds = 0.1
        document.tracks[0].cues[0].duration_seconds = 0.1
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_maximum_duration_enforced(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].end_seconds = 100.0
        document.tracks[0].cues[0].duration_seconds = 100.0
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_frame_snapping_works_when_enabled(self) -> None:
        snapped = snap_to_frame_rate(1.012, 30)
        self.assertAlmostEqual(snapped, 1.0)

    def test_frame_snapping_applied_during_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.json"
            source.write_text(json.dumps({
                "language": "en",
                "entries": [{"text": "hi", "start_seconds": 0.012, "end_seconds": 2.012}],
            }))
            config = SubtitleConfig(
                frame_rate_snap_enabled=True, default_frame_rate=30, styles=self.config.styles
            )
            document = build_subtitle_document(source, config)
            self.assertAlmostEqual(document.tracks[0].cues[0].start_seconds, 0.0)

    def test_duplicate_cue_id_fails(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[1].cue_id = "cue_0001"
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_minimum_gap_enforced(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[1].start_seconds = 2.5
        document.tracks[0].cues[1].end_seconds = 4.5
        document.tracks[0].cues[1].duration_seconds = 2.0
        config = SubtitleConfig(minimum_gap_seconds=1.0, styles=self.config.styles)
        result = validate_subtitle_document(document, config)
        self.assertFalse(result.passed)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class TextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_subtitle_config()

    def test_repeated_spaces_normalized(self) -> None:
        self.assertEqual(normalize_text("hello    world", self.config), "hello world")

    def test_explicit_line_breaks_preserved(self) -> None:
        result = normalize_text("line one\nline two", self.config)
        self.assertEqual(result, "line one\nline two")

    def test_english_wrapping_works(self) -> None:
        text = "this is a long line of english text that should wrap"
        planned, _warnings = plan_cue_line_wrap(text, self.config)
        self.assertGreater(len(planned), 1)
        for line in planned:
            self.assertLessEqual(len(line), self.config.maximum_characters_per_line)

    def test_traditional_chinese_wrapping_works(self) -> None:
        text = "京都的清晨非常安靜街道上幾乎沒有行人只有微風輕輕吹過樹葉沙沙作響令人心曠神怡"
        self.assertGreater(len(text), self.config.maximum_characters_per_line)
        planned, _warnings = plan_cue_line_wrap(text, self.config)
        self.assertGreater(len(planned), 1)
        self.assertEqual("".join(planned), text)

    def test_japanese_wrapping_works(self) -> None:
        text = "きょうとのしずかなあさにひとりであるいていますとてもきもちがいいですかぜがふいています"
        self.assertGreater(len(text), self.config.maximum_characters_per_line)
        planned, _warnings = plan_cue_line_wrap(text, self.config)
        self.assertGreater(len(planned), 1)
        self.assertEqual("".join(planned), text)

    def test_maximum_line_count_warning(self) -> None:
        text = " ".join(["word"] * 40)
        planned, warnings = plan_cue_line_wrap(text, self.config)
        self.assertGreater(len(planned), self.config.maximum_lines_per_cue)
        self.assertTrue(any(w.code == "line_overflow" for w in warnings))

    def test_maximum_line_count_fails_when_configured(self) -> None:
        text = " ".join(["word"] * 40)
        config = SubtitleConfig(overflow_policy="fail", styles=self.config.styles,
                                 maximum_characters_per_line=self.config.maximum_characters_per_line,
                                 maximum_lines_per_cue=self.config.maximum_lines_per_cue,
                                 cjk_character_wrap=self.config.cjk_character_wrap,
                                 preserve_explicit_line_breaks=self.config.preserve_explicit_line_breaks)
        with self.assertRaises(SubtitleValidationError):
            plan_cue_line_wrap(text, config)

    def test_punctuation_preserved(self) -> None:
        text = "Wait... is that -- really -- true?!"
        self.assertEqual(normalize_text(text, self.config), text)

    def test_text_not_translated_or_rewritten(self) -> None:
        text = "京都の朝, hello world!  多余空格"
        normalized = normalize_text(text, self.config)
        # normalize_text only trims/collapses whitespace -- every
        # non-whitespace character must survive untouched, in order.
        original_non_space = [c for c in text if not c.isspace()]
        normalized_non_space = [c for c in normalized if not c.isspace()]
        self.assertEqual(original_non_space, normalized_non_space)

    def test_line_break_mode_preserve_disables_wrapping(self) -> None:
        text = "this is a long line of english text that should not wrap"
        planned, _warnings = plan_cue_line_wrap(text, self.config, line_break_mode=LineBreakMode.PRESERVE)
        self.assertEqual(planned, [text])


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


class StyleTests(SubtitleTempTestCase):
    def test_default_style_loads(self) -> None:
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        document = build_subtitle_document(source, self.config)
        self.assertIn("default", document.styles)

    def test_named_preset_loads(self) -> None:
        source = self._write_json("source.json", self._entries_source(
            [{"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0}], style="luxury_editorial"
        ))
        document = build_subtitle_document(source, self.config)
        self.assertIn("luxury_editorial", document.styles)
        self.assertEqual(document.tracks[0].default_style_id, "luxury_editorial")

    def test_unknown_style_preset_fails(self) -> None:
        source = self._write_json("source.json", self._entries_source(
            [{"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0}], style="does_not_exist"
        ))
        with self.assertRaises(SubtitleStyleError):
            build_subtitle_document(source, self.config)

    def test_per_entry_unknown_style_id_fails(self) -> None:
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0, "style_id": "nope"},
        ]))
        with self.assertRaises(SubtitleStyleError):
            build_subtitle_document(source, self.config)

    def test_color_validation_works(self) -> None:
        document = _basic_valid_document(self.config)
        document.styles["default"] = SubtitleStyle(style_id="default", primary_color="not-a-color")
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_font_size_validation_works(self) -> None:
        document = _basic_valid_document(self.config)
        document.styles["default"] = SubtitleStyle(style_id="default", font_size=0)
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_margins_validated(self) -> None:
        document = _basic_valid_document(self.config)
        document.styles["default"] = SubtitleStyle(style_id="default", margin_left=-5)
        result = validate_subtitle_document(document, self.config)
        self.assertFalse(result.passed)

    def test_style_metadata_round_trips(self) -> None:
        style = SubtitleStyle(style_id="default", metadata={"note": "hand-tuned"})
        document = _basic_valid_document(self.config)
        document.styles["default"] = style
        rebuilt = document_from_dict(document_to_dict(document))
        self.assertEqual(rebuilt.styles["default"].metadata, {"note": "hand-tuned"})


# ---------------------------------------------------------------------------
# Safe area
# ---------------------------------------------------------------------------


class SafeAreaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_subtitle_config()

    def test_default_position_inside_safe_area(self) -> None:
        style = self.config.styles["default"]
        position = plan_cue_position(style, self.config)
        self.assertEqual(validate_position_within_safe_area(position, self.config), [])

    def test_invalid_position_warns(self) -> None:
        position = SubtitlePosition(x=10, y=10, safe_area_enabled=True)
        warnings = validate_position_within_safe_area(position, self.config)
        self.assertTrue(warnings)

    def test_custom_position_retained(self) -> None:
        style = self.config.styles["default"]
        custom = SubtitlePosition(x=500, y=1400)
        position = plan_cue_position(style, self.config, cue_position=custom)
        self.assertEqual(position.x, 500)
        self.assertEqual(position.y, 1400)

    def test_top_obstruction_limit_respected(self) -> None:
        position = SubtitlePosition(x=540, y=self.config.safe_area_top - 1, safe_area_enabled=True)
        self.assertTrue(validate_position_within_safe_area(position, self.config))

    def test_bottom_obstruction_limit_respected(self) -> None:
        limit = self.config.canvas_height - self.config.safe_area_bottom
        position = SubtitlePosition(x=540, y=limit + 1, safe_area_enabled=True)
        self.assertTrue(validate_position_within_safe_area(position, self.config))

    def test_safe_area_disabled_skips_check(self) -> None:
        position = SubtitlePosition(x=0, y=0, safe_area_enabled=False)
        self.assertEqual(validate_position_within_safe_area(position, self.config), [])


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class JsonTests(SubtitleTempTestCase):
    def test_stable_document_id_across_builds(self) -> None:
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        first = build_subtitle_document(source, self.config)
        second = build_subtitle_document(source, self.config)
        self.assertEqual(first.document_id, second.document_id)

    def test_created_at_excluded_from_document_id(self) -> None:
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        first = build_subtitle_document(source, self.config)
        self.assertNotEqual(first.created_at, "")
        second = build_subtitle_document(source, self.config)
        self.assertNotEqual(first.created_at, second.created_at)
        self.assertEqual(first.document_id, second.document_id)

    def test_round_trip_preserved(self) -> None:
        document = _basic_valid_document(self.config)
        rebuilt = document_from_dict(document_to_dict(document))
        self.assertEqual(document_to_dict(rebuilt), document_to_dict(document))

    def test_atomic_write_no_tmp_left_behind(self) -> None:
        document = _basic_valid_document(self.config)
        out = self.temp_dir / "subtitles.json"
        save_subtitle_document(document, out)
        self.assertTrue(out.is_file())
        self.assertFalse((self.temp_dir / "subtitles.json.tmp").exists())

    def test_overwrite_refused_without_force(self) -> None:
        document = _basic_valid_document(self.config)
        out = self.temp_dir / "subtitles.json"
        save_subtitle_document(document, out)
        with self.assertRaises(SubtitleOutputExistsError):
            save_subtitle_document(document, out)

    def test_force_overwrites(self) -> None:
        document = _basic_valid_document(self.config)
        out = self.temp_dir / "subtitles.json"
        save_subtitle_document(document, out)
        save_subtitle_document(document, out, force=True)
        self.assertTrue(out.is_file())

    def test_output_path_directory_raises(self) -> None:
        document = _basic_valid_document(self.config)
        out_dir = self.temp_dir / "a_directory"
        out_dir.mkdir()
        with self.assertRaises(UnsafeSubtitleOutputError):
            save_subtitle_document(document, out_dir)

    def test_unknown_metadata_preserved_round_trip(self) -> None:
        document = _basic_valid_document(self.config)
        document.metadata = {"custom_field": {"nested": True}}
        rebuilt = document_from_dict(document_to_dict(document))
        self.assertEqual(rebuilt.metadata, {"custom_field": {"nested": True}})

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(SubtitleInputError):
            load_subtitle_document(self.temp_dir / "nope.json")

    def test_malformed_json_file_raises(self) -> None:
        bad = self._write_text("bad.json", "{not valid json")
        with self.assertRaises(SubtitleJSONError):
            load_subtitle_document(bad)


# ---------------------------------------------------------------------------
# SRT export
# ---------------------------------------------------------------------------


class SrtExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_subtitle_config()

    def test_timestamp_formatting_correct(self) -> None:
        document = _basic_valid_document(self.config)
        srt, _warnings = export_srt(document, self.config)
        self.assertIn("00:00:00,000 --> 00:00:02,000", srt)

    def test_sequence_numbering_correct(self) -> None:
        document = _basic_valid_document(self.config)
        srt, _warnings = export_srt(document, self.config)
        lines = srt.splitlines()
        self.assertEqual(lines[0], "1")

    def test_utf8_output(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].text = "京都の朝"
        srt, _warnings = export_srt(document, self.config)
        self.assertIn("京都の朝", srt)

    def test_cue_order_deterministic(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues.reverse()
        srt, _warnings = export_srt(document, self.config)
        first_text_line = srt.splitlines()[2]
        self.assertEqual(first_text_line, "hello there")

    def test_style_loss_warning_emitted(self) -> None:
        document = _basic_valid_document(self.config)
        _srt, warnings = export_srt(document, self.config)
        self.assertTrue(warnings)

    def test_multiline_text_preserved(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].text = "line one\nline two"
        srt, _warnings = export_srt(document, self.config)
        self.assertIn("line one\nline two", srt)

    def test_blank_line_between_entries(self) -> None:
        document = _basic_valid_document(self.config)
        srt, _warnings = export_srt(document, self.config)
        self.assertIn("\n\n", srt)


# ---------------------------------------------------------------------------
# ASS export
# ---------------------------------------------------------------------------


class AssExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_subtitle_config()

    def test_script_sections_valid(self) -> None:
        document = _basic_valid_document(self.config)
        ass = export_ass(document, self.config)
        self.assertIn("[Script Info]", ass)
        self.assertIn("[V4+ Styles]", ass)
        self.assertIn("[Events]", ass)

    def test_resolution_included(self) -> None:
        document = _basic_valid_document(self.config)
        ass = export_ass(document, self.config)
        self.assertIn(f"PlayResX: {self.config.canvas_width}", ass)
        self.assertIn(f"PlayResY: {self.config.canvas_height}", ass)

    def test_style_exported(self) -> None:
        document = _basic_valid_document(self.config)
        ass = export_ass(document, self.config)
        self.assertIn("Style: default,", ass)

    def test_event_timing_correct(self) -> None:
        document = _basic_valid_document(self.config)
        ass = export_ass(document, self.config)
        self.assertIn("Dialogue: 0,0:00:00.00,0:00:02.00,default", ass)

    def test_unicode_preserved(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].text = "京都の朝"
        ass = export_ass(document, self.config)
        self.assertIn("京都の朝", ass)

    def test_ass_escaping_correct(self) -> None:
        document = _basic_valid_document(self.config)
        document.tracks[0].cues[0].text = "look {here}\nand there"
        ass = export_ass(document, self.config)
        self.assertIn("look \\{here\\}\\Nand there", ass)

    def test_alignment_and_margins_mapped(self) -> None:
        document = _basic_valid_document(self.config)
        document.styles["default"] = SubtitleStyle(
            style_id="default", alignment=SubtitleAlignment.TOP_LEFT, margin_left=10, margin_right=20, margin_vertical=30
        )
        ass = export_ass(document, self.config)
        style_line = next(line for line in ass.splitlines() if line.startswith("Style: default,"))
        fields = style_line[len("Style: "):].split(",")
        self.assertEqual(fields[9], "7")  # top_left -> numpad 7
        self.assertEqual(fields[10:13], ["10", "20", "30"])

    def test_no_renderer_invocation(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import subprocess"))
            self.assertFalse(stripped.startswith("from subprocess"))
        for forbidden in ("subprocess.run(", "subprocess.Popen(", "subprocess.call("):
            self.assertNotIn(forbidden, MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Timeline compatibility
# ---------------------------------------------------------------------------


class TimelineCompatibilityTests(SubtitleTempTestCase):
    def _write_timeline(self, *, duration: float = 10.0, subtitle_track: bool = False) -> Path:
        video_track = timeline_engine.TimelineTrack(
            track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0,
            clips=[
                timeline_engine.VideoClip(
                    clip_id="clip_1", track_id="track_video", source_path="/scenes/scene_01.mp4",
                    start=0.0, end=duration, duration_seconds=duration, source_out=duration, scene_number=1,
                )
            ],
        )
        tracks = [video_track]
        if subtitle_track:
            tracks.append(
                timeline_engine.TimelineTrack(
                    track_id="track_subtitles_en", track_type=timeline_engine.TrackType.SUBTITLE, order=1,
                    metadata={"cues": [{"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0}], "language": "en"},
                )
            )
        timeline = timeline_engine.Timeline(timeline_id="tl1", duration_seconds=duration, tracks=tracks)
        path = self.temp_dir / "timeline.json"
        timeline_engine.save_timeline(timeline, path)
        return path

    def test_timeline_id_recorded(self) -> None:
        timeline_path = self._write_timeline()
        timeline = load_timeline_for_subtitles(timeline_path)
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        document = build_subtitle_document(source, self.config, timeline=timeline)
        self.assertEqual(document.timeline_id, "tl1")

    def test_duration_within_timeline_passes(self) -> None:
        timeline_path = self._write_timeline(duration=10.0)
        timeline = load_timeline_for_subtitles(timeline_path)
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 5.0},
        ]))
        document = build_subtitle_document(source, self.config, timeline=timeline)
        self.assertLessEqual(document.duration_seconds, timeline.duration_seconds)

    def test_cue_beyond_timeline_fails(self) -> None:
        timeline_path = self._write_timeline(duration=2.0)
        timeline = load_timeline_for_subtitles(timeline_path)
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 5.0},
        ]))
        with self.assertRaises(TimelineCompatibilityError):
            build_subtitle_document(source, self.config, timeline=timeline)

    def test_scene_boundary_spanning_cue_allowed(self) -> None:
        video_track = timeline_engine.TimelineTrack(
            track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0,
            clips=[
                timeline_engine.VideoClip(
                    clip_id="clip_1", track_id="track_video", source_path="/scenes/scene_01.mp4",
                    start=0.0, end=3.0, duration_seconds=3.0, source_out=3.0, scene_number=1,
                ),
                timeline_engine.VideoClip(
                    clip_id="clip_2", track_id="track_video", source_path="/scenes/scene_02.mp4",
                    start=3.0, end=6.0, duration_seconds=3.0, source_out=3.0, scene_number=2,
                ),
            ],
        )
        timeline_obj = timeline_engine.Timeline(timeline_id="tl2", duration_seconds=6.0, tracks=[video_track])
        timeline_path = self.temp_dir / "timeline.json"
        timeline_engine.save_timeline(timeline_obj, timeline_path)
        timeline = load_timeline_for_subtitles(timeline_path)

        source = self._write_json("source.json", self._entries_source([
            {"text": "spans both scenes", "start_seconds": 2.0, "end_seconds": 4.0},
        ]))
        document = build_subtitle_document(source, self.config, timeline=timeline)
        self.assertEqual(len(document.tracks[0].cues), 1)

    def test_disabled_timeline_clips_do_not_affect_subtitles(self) -> None:
        video_track = timeline_engine.TimelineTrack(
            track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0,
            clips=[
                timeline_engine.VideoClip(
                    clip_id="clip_1", track_id="track_video", source_path="/scenes/scene_01.mp4",
                    start=0.0, end=3.0, duration_seconds=3.0, source_out=3.0, scene_number=1, enabled=False,
                ),
                timeline_engine.VideoClip(
                    clip_id="clip_2", track_id="track_video", source_path="/scenes/scene_02.mp4",
                    start=3.0, end=6.0, duration_seconds=3.0, source_out=3.0, scene_number=2,
                ),
            ],
        )
        timeline_obj = timeline_engine.Timeline(timeline_id="tl3", duration_seconds=6.0, tracks=[video_track])
        timeline_path = self.temp_dir / "timeline.json"
        timeline_engine.save_timeline(timeline_obj, timeline_path)
        timeline = load_timeline_for_subtitles(timeline_path)

        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        document = build_subtitle_document(source, self.config, timeline=timeline)
        result = validate_subtitle_document(document, self.config, timeline=timeline)
        self.assertTrue(result.passed)

    def test_timeline_metadata_source_reads_subtitle_track(self) -> None:
        timeline_path = self._write_timeline(subtitle_track=True)
        timeline = load_timeline_for_subtitles(timeline_path)
        source = self._write_json("source.json", {"source_type": "timeline_metadata", "track_id": "track_subtitles_en"})
        document = build_subtitle_document(source, self.config, timeline=timeline)
        self.assertEqual(document.tracks[0].cues[0].text, "hi")
        self.assertEqual(document.tracks[0].track_id, "track_subtitles_en")

    def test_timeline_metadata_source_without_timeline_fails(self) -> None:
        source = self._write_json("source.json", {"source_type": "timeline_metadata", "track_id": "track_subtitles_en"})
        with self.assertRaises(SubtitleInputError):
            build_subtitle_document(source, self.config)

    def test_timeline_file_remains_byte_for_byte_unchanged(self) -> None:
        timeline_path = self._write_timeline(duration=10.0)
        before_hash = hashlib.sha256(timeline_path.read_bytes()).hexdigest()

        timeline = load_timeline_for_subtitles(timeline_path)
        source = self._write_json("source.json", self._entries_source([
            {"text": "hi", "start_seconds": 0.0, "end_seconds": 2.0},
        ]))
        document = build_subtitle_document(source, self.config, timeline=timeline)
        validate_subtitle_document(document, self.config, timeline=timeline)

        after_hash = hashlib.sha256(timeline_path.read_bytes()).hexdigest()
        self.assertEqual(before_hash, after_hash)

    def test_invalid_timeline_file_raises(self) -> None:
        with self.assertRaises(TimelineCompatibilityError):
            load_timeline_for_subtitles(self.temp_dir / "does_not_exist.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def test_input_and_validate_together_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--input", "x", "--validate", "y"])

    def test_one_mode_required(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_export_flags_accepted(self) -> None:
        args = parse_arguments(["--input", "x", "--export-srt", "s.srt", "--export-ass", "a.ass"])
        self.assertEqual(args.export_srt, "s.srt")
        self.assertEqual(args.export_ass, "a.ass")

    def test_timeline_and_language_accepted(self) -> None:
        args = parse_arguments(["--input", "x", "--timeline", "t.json", "--language", "ja"])
        self.assertEqual(args.timeline, "t.json")
        self.assertEqual(args.language, "ja")

    def test_output_is_optional(self) -> None:
        args = parse_arguments(["--input", "x"])
        self.assertIsNone(args.output)


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

    def test_no_ffmpeg_or_ffprobe_or_subprocess(self) -> None:
        # "ffmpeg"/"ffprobe" appear only in this module's own explanatory
        # comments/docstrings (describing what it never does) -- checked
        # precisely here rather than via a bare substring match, which
        # would false-fail on those comments.
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import subprocess"))
            self.assertFalse(stripped.startswith("from subprocess"))
        for forbidden in ("subprocess.run(", "subprocess.Popen(", "subprocess.call(", "ffmpeg_binary", "ffprobe_binary"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_media_file_writes(self) -> None:
        for forbidden in (".mp4", ".mov", ".m4v", ".wav"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_speech_recognition_or_whisper(self) -> None:
        # "transcribes" appears only in this module's own explanatory
        # comments ("never transcribes audio/speech") -- checked via
        # actual import lines, not a bare substring match.
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip().lower()
            self.assertFalse(stripped.startswith("import whisper"))
            self.assertFalse(stripped.startswith("from whisper"))
            self.assertFalse(stripped.startswith("import speech_recognition"))
            self.assertFalse(stripped.startswith("from speech_recognition"))

    def test_no_translation_library(self) -> None:
        for forbidden in ("googletrans", "translate(", "deepl"):
            self.assertNotIn(forbidden, MODULE_SOURCE.lower())

    def test_no_font_download_or_network_code(self) -> None:
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download_font"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_timeline_engine_mutation(self) -> None:
        self.assertNotIn("save_timeline(", MODULE_SOURCE)

    def test_only_reads_timeline_engine_public_api(self) -> None:
        self.assertIn("from . import timeline_engine", MODULE_SOURCE)
        self.assertNotIn("from .reel_builder", MODULE_SOURCE)
        self.assertNotIn("from .video_engine", MODULE_SOURCE)
        self.assertNotIn("from .music_mixer", MODULE_SOURCE)


if __name__ == "__main__":
    unittest.main()
