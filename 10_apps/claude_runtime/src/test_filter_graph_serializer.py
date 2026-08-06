from __future__ import annotations

import dataclasses
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from . import filter_graph_builder as fgb
from . import filter_graph_serializer as fgs

MODULE_PATH = Path(__file__).resolve().parent / "filter_graph_serializer.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _passed_validation() -> fgb.FilterGraphValidation:
    return fgb.FilterGraphValidation(passed=True, errors=[], warnings=[])


def _drawtext_spec(**overrides) -> fgb.FilterSpec:
    defaults = dict(
        filter_id="f_drawtext",
        filter_type=fgb.FilterType.DRAWTEXT,
        label_in=["0:v"],
        label_out=["outv"],
        parameters={"text": "Hello world", "x": 10, "y": 20, "font_file": "/fonts/body.ttf"},
        enable_expression="between(t,0,2)",
    )
    defaults.update(overrides)
    return fgb.FilterSpec(**defaults)


def _drawtext_graph(**spec_overrides) -> fgb.FilterGraph:
    spec = _drawtext_spec(**spec_overrides)
    graph_pass = fgb.FilterGraphPass(pass_id="pass_subtitle", pass_type="subtitle", hint_type="drawtext", filters=[spec])
    return fgb.FilterGraph(
        graph_id="graph123", renderer_plan_id="", overlay_plan_id=None,
        passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"], outputs=["outv"],
        dependencies={"pass_subtitle": []}, validation=_passed_validation(),
    )


def _ass_spec(**overrides) -> fgb.FilterSpec:
    defaults = dict(
        filter_id="f_ass", filter_type=fgb.FilterType.ASS, label_in=["0:v"], label_out=["outv"],
        parameters={"ass_path": "/subs/movie.ass"},
    )
    defaults.update(overrides)
    return fgb.FilterSpec(**defaults)


def _ass_graph(**spec_overrides) -> fgb.FilterGraph:
    spec = _ass_spec(**spec_overrides)
    graph_pass = fgb.FilterGraphPass(pass_id="pass_subtitle", pass_type="subtitle", hint_type="ass", filters=[spec])
    return fgb.FilterGraph(
        graph_id="graph_ass", renderer_plan_id="", overlay_plan_id=None,
        passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"], outputs=["outv"],
        dependencies={"pass_subtitle": []}, validation=_passed_validation(),
    )


def _overlay_spec(**overrides) -> fgb.FilterSpec:
    defaults = dict(
        filter_id="f_overlay",
        filter_type=fgb.FilterType.OVERLAY,
        label_in=["0:v", "1:v"],
        label_out=["outv"],
        parameters={"x": 10, "y": 20, "asset_id": "a1", "resolved_path": "/assets/logo.png", "has_alpha": True, "logical_role": "logo"},
    )
    defaults.update(overrides)
    return fgb.FilterSpec(**defaults)


def _overlay_graph(*, extra_inputs=None, **spec_overrides) -> fgb.FilterGraph:
    spec = _overlay_spec(**spec_overrides)
    if extra_inputs is None:
        extra_inputs = [fgb.ExtraInputSpec(label="1:v", resolved_path="/assets/logo.png", asset_id="a1", logical_role="logo")]
    graph_pass = fgb.FilterGraphPass(pass_id="pass_overlay", pass_type="overlay", filters=[spec])
    return fgb.FilterGraph(
        graph_id="graph_overlay", renderer_plan_id="", overlay_plan_id=None,
        passes=[graph_pass], filters=[spec], labels=["0:v", "1:v", "outv"], inputs=["0:v", "1:v"],
        outputs=["outv"], extra_inputs=extra_inputs, dependencies={"pass_overlay": []}, validation=_passed_validation(),
    )


class FilterGraphSerializerTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="filter_graph_serializer_test_"))
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.config = fgs.FilterGraphSerializerConfig()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationTests(unittest.TestCase):
    def test_default_config_loads(self):
        config = fgs.load_filter_graph_serializer_config()
        self.assertEqual(config.renderer, "ffmpeg")
        self.assertTrue(config.reject_unknown_filters)
        self.assertFalse(config.allow_custom_passthrough)
        self.assertTrue(config.prefer_simple_vf)
        self.assertFalse(config.force_filter_complex)

    def test_missing_config_raises(self):
        with self.assertRaises(fgs.FilterGraphSerializerConfigError):
            fgs.load_filter_graph_serializer_config("/nonexistent/filter_graph_serializer.yaml")

    def test_empty_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(fgs.FilterGraphSerializerConfigError):
                fgs.load_filter_graph_serializer_config(path)

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("serializer: [unclosed", encoding="utf-8")
            with self.assertRaises(fgs.FilterGraphSerializerConfigError):
                fgs.load_filter_graph_serializer_config(path)

    def test_unknown_filter_rejection_defaults_true(self):
        config = fgs.load_filter_graph_serializer_config()
        self.assertTrue(config.reject_unknown_filters)

    def test_simple_vf_preference_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("serializer:\n  prefer_simple_vf: false\n", encoding="utf-8")
            config = fgs.load_filter_graph_serializer_config(path)
            self.assertFalse(config.prefer_simple_vf)

    def test_forced_filter_complex_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("serializer:\n  force_filter_complex: true\n", encoding="utf-8")
            config = fgs.load_filter_graph_serializer_config(path)
            self.assertTrue(config.force_filter_complex)

    def test_label_pattern_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("labels:\n  generated_label_pattern: '^[a-z]+$'\n", encoding="utf-8")
            config = fgs.load_filter_graph_serializer_config(path)
            self.assertEqual(config.generated_label_pattern, "^[a-z]+$")

    def test_enable_function_allowlist_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("enable:\n  allowed_functions: [between, gte]\n", encoding="utf-8")
            config = fgs.load_filter_graph_serializer_config(path)
            self.assertEqual(config.enable_allowed_functions, ("between", "gte"))

    def test_default_config_path_resolves(self):
        path = fgs.default_filter_graph_serializer_config_path()
        self.assertTrue(str(path).endswith("config/video/filter_graph_serializer.yaml"))


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


class EscapingTests(FilterGraphSerializerTempTestCase):
    def test_escape_filter_value_wraps_in_quotes(self):
        self.assertEqual(fgs.escape_filter_value("hello", self.config), "'hello'")

    def test_escape_filter_value_escapes_apostrophe(self):
        result = fgs.escape_filter_value("it's", self.config)
        self.assertIn("'\\''", result)

    def test_escape_filter_value_rejects_nul_byte(self):
        with self.assertRaises(fgs.UnsafeFilterValueError):
            fgs.escape_filter_value("bad\x00value", self.config)

    def test_escape_drawtext_text_plain(self):
        self.assertEqual(fgs.escape_drawtext_text("hello", self.config), "hello")

    def test_escape_drawtext_text_apostrophe(self):
        self.assertEqual(fgs.escape_drawtext_text("it's", self.config), "it\\'s")

    def test_escape_drawtext_text_colon(self):
        self.assertEqual(fgs.escape_drawtext_text("a:b", self.config), "a\\:b")

    def test_escape_drawtext_text_backslash(self):
        self.assertEqual(fgs.escape_drawtext_text("a\\b", self.config), "a\\\\b")

    def test_escape_drawtext_text_comma(self):
        self.assertEqual(fgs.escape_drawtext_text("a,b", self.config), "a\\,b")

    def test_escape_drawtext_text_semicolon(self):
        self.assertEqual(fgs.escape_drawtext_text("a;b", self.config), "a\\;b")

    def test_escape_drawtext_text_brackets(self):
        self.assertEqual(fgs.escape_drawtext_text("[a]", self.config), "\\[a\\]")

    def test_escape_drawtext_text_percent(self):
        self.assertEqual(fgs.escape_drawtext_text("50%", self.config), "50%%")

    def test_escape_drawtext_text_backslash_escaped_before_others(self):
        # backslash escaping must happen first so inserted escape
        # backslashes are never themselves re-escaped.
        result = fgs.escape_drawtext_text("a:\\b", self.config)
        self.assertEqual(result, "a\\:\\\\b")

    def test_escape_drawtext_text_explicit_newline_passthrough(self):
        self.assertIn("\n", fgs.escape_drawtext_text("line1\nline2", self.config))

    def test_escape_drawtext_text_traditional_chinese_passthrough(self):
        text = "哈囉世界"
        self.assertEqual(fgs.escape_drawtext_text(text, self.config), text)

    def test_escape_drawtext_text_japanese_passthrough(self):
        text = "こんにちは"
        self.assertEqual(fgs.escape_drawtext_text(text, self.config), text)

    def test_escape_drawtext_text_emoji_passthrough(self):
        text = "hello 🎉🔥"
        self.assertEqual(fgs.escape_drawtext_text(text, self.config), text)

    def test_escape_drawtext_text_punctuation(self):
        text = "Wait... really?!"
        result = fgs.escape_drawtext_text(text, self.config)
        self.assertIn("Wait...", result)
        self.assertIn("really?!", result)

    def test_escape_drawtext_text_no_translation(self):
        text = "Bonjour le monde"
        self.assertEqual(fgs.escape_drawtext_text(text, self.config), text)

    def test_escape_drawtext_text_no_wrapping_added(self):
        text = "a" * 200
        result = fgs.escape_drawtext_text(text, self.config)
        self.assertNotIn("\n", result)

    def test_escape_drawtext_text_no_rewriting_of_case_or_spacing(self):
        text = "  Weird   Spacing  "
        self.assertEqual(fgs.escape_drawtext_text(text, self.config), text)

    def test_escape_drawtext_text_rejects_nul_byte(self):
        with self.assertRaises(fgs.UnsafeFilterValueError):
            fgs.escape_drawtext_text("bad\x00text", self.config)

    def test_escape_drawtext_text_toggles_respected(self):
        config = dataclasses.replace(self.config, drawtext_escape_colon=False)
        self.assertEqual(fgs.escape_drawtext_text("a:b", config), "a:b")


# ---------------------------------------------------------------------------
# Drawtext serialization
# ---------------------------------------------------------------------------


class DrawTextSerializationTests(FilterGraphSerializerTempTestCase):
    def test_basic_serialization_shape(self):
        spec = _drawtext_spec()
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertTrue(result.startswith("drawtext="))
        self.assertIn("text=Hello world", result)

    def test_missing_text_raises(self):
        spec = _drawtext_spec(parameters={"x": 1, "y": 2})
        with self.assertRaises(fgs.DrawTextSerializationError):
            fgs.serialize_drawtext_filter(spec, self.config)

    def test_font_file_param(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/fonts/a.ttf"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("fontfile='/fonts/a.ttf'", result)

    def test_font_path_fallback_key(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_path": "/fonts/b.ttf"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("fontfile='/fonts/b.ttf'", result)

    def test_missing_font_file_raises_when_required(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2})
        with self.assertRaises(fgs.DrawTextSerializationError):
            fgs.serialize_drawtext_filter(spec, self.config)

    def test_missing_font_file_ok_when_not_required(self):
        config = dataclasses.replace(self.config, drawtext_require_font_file=False)
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2})
        result = fgs.serialize_drawtext_filter(spec, config)
        self.assertNotIn("fontfile=", result)

    def test_font_size_param(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "font_size": 72})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("fontsize=72", result)

    def test_font_size_default(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("fontsize=48", result)

    def test_font_color_param(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "font_color": "#FF0000"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("fontcolor='#FF0000'", result)

    def test_border_color_param(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "border_color": "#00FF00"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("bordercolor='#00FF00'", result)

    def test_outline_color_fallback_key(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "outline_color": "#0000FF"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("bordercolor='#0000FF'", result)

    def test_border_width_param(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "border_width": 5})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("borderw=5", result)

    def test_outline_width_fallback_key(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "outline_width": 3})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("borderw=3", result)

    def test_x_y_explicit(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 42, "y": 84, "font_file": "/f.ttf"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("x=42", result)
        self.assertIn("y=84", result)

    def test_x_without_y_raises(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 42, "font_file": "/f.ttf"})
        with self.assertRaises(fgs.DrawTextSerializationError):
            fgs.serialize_drawtext_filter(spec, self.config)

    def test_non_numeric_x_raises(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": "bad", "y": 1, "font_file": "/f.ttf"})
        with self.assertRaises(fgs.DrawTextSerializationError):
            fgs.serialize_drawtext_filter(spec, self.config)

    def test_missing_position_and_anchor_raises(self):
        spec = _drawtext_spec(parameters={"text": "hi", "font_file": "/f.ttf"})
        with self.assertRaises(fgs.DrawTextSerializationError):
            fgs.serialize_drawtext_filter(spec, self.config)

    def test_all_nine_anchors_resolve(self):
        for anchor in ("top_left", "top_center", "top_right", "center_left", "center", "center_right",
                       "bottom_left", "bottom_center", "bottom_right"):
            spec = _drawtext_spec(parameters={"text": "hi", "anchor": anchor, "font_file": "/f.ttf"})
            result = fgs.serialize_drawtext_filter(spec, self.config)
            self.assertIn("x=", result)
            self.assertIn("y=", result)

    def test_bottom_center_uses_margin(self):
        spec = _drawtext_spec(parameters={"text": "hi", "anchor": "bottom_center", "font_file": "/f.ttf"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("x=(w-text_w)/2", result)
        self.assertIn(f"y=h-text_h-{self.config.drawtext_anchor_margin_pixels}", result)

    def test_unsupported_anchor_raises(self):
        spec = _drawtext_spec(parameters={"text": "hi", "anchor": "somewhere_weird", "font_file": "/f.ttf"})
        with self.assertRaises(fgs.DrawTextSerializationError):
            fgs.serialize_drawtext_filter(spec, self.config)

    def test_box_enabled_adds_box_options(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "box_enabled": True, "box_color": "#000000AA"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("box=1", result)
        self.assertIn("boxcolor=", result)

    def test_box_default_from_config(self):
        config = dataclasses.replace(self.config, drawtext_box_enabled_default=True)
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf"})
        result = fgs.serialize_drawtext_filter(spec, config)
        self.assertIn("box=1", result)

    def test_box_disabled_by_default(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf"})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertNotIn("box=1", result)

    def test_opacity_param_adds_alpha(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "opacity": 0.5})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("alpha=0.5", result)

    def test_full_opacity_omits_alpha(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "opacity": 1.0})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertNotIn("alpha=", result)

    def test_line_spacing_param(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf", "line_spacing": 10})
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("line_spacing=10", result)

    def test_enable_expression_present(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf"}, enable_expression="between(t,1.5,3.25)")
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertIn("between(t,1.5,3.25)", result)

    def test_no_enable_expression_when_none(self):
        spec = _drawtext_spec(parameters={"text": "hi", "x": 1, "y": 2, "font_file": "/f.ttf"}, enable_expression=None)
        result = fgs.serialize_drawtext_filter(spec, self.config)
        self.assertNotIn("enable=", result)

    def test_multiple_cues_via_filter_spec_dispatch(self):
        specs = [
            _drawtext_spec(filter_id="f1", parameters={"text": "First", "x": 1, "y": 2, "font_file": "/f.ttf"}),
            _drawtext_spec(filter_id="f2", parameters={"text": "Second", "x": 1, "y": 2, "font_file": "/f.ttf"}),
        ]
        results = [fgs.serialize_filter_spec(s, self.config) for s in specs]
        self.assertEqual(results[0].filter_type, fgb.FilterType.DRAWTEXT)
        self.assertIn("First", results[0].serialized)
        self.assertIn("Second", results[1].serialized)

    def test_deterministic_cue_order_preserved(self):
        specs = [_drawtext_spec(filter_id=f"f{i}", parameters={"text": f"cue{i}", "x": 1, "y": 2, "font_file": "/f.ttf"}) for i in range(5)]
        results = [fgs.serialize_filter_spec(s, self.config).serialized for s in specs]
        for i in range(5):
            self.assertIn(f"cue{i}", results[i])


# ---------------------------------------------------------------------------
# ASS serialization
# ---------------------------------------------------------------------------


class AssSerializationTests(FilterGraphSerializerTempTestCase):
    def test_basic_subtitle_path(self):
        spec = _ass_spec(parameters={"ass_path": "/subs/a.ass"})
        result = fgs.serialize_ass_filter(spec, self.config)
        self.assertTrue(result.startswith("subtitles=filename="))
        self.assertIn("/subs/a.ass", result)

    def test_missing_ass_path_raises(self):
        spec = _ass_spec(parameters={})
        with self.assertRaises(fgs.ASSSerializationError):
            fgs.serialize_ass_filter(spec, self.config)

    def test_spaces_in_path(self):
        spec = _ass_spec(parameters={"ass_path": "/subs/my movie.ass"})
        result = fgs.serialize_ass_filter(spec, self.config)
        self.assertIn("my movie.ass", result)

    def test_apostrophe_in_path_escaped(self):
        spec = _ass_spec(parameters={"ass_path": "/subs/it's a movie.ass"})
        result = fgs.serialize_ass_filter(spec, self.config)
        self.assertIn("'\\''", result)

    def test_colon_in_path(self):
        spec = _ass_spec(parameters={"ass_path": "C:/subs/a.ass"})
        result = fgs.serialize_ass_filter(spec, self.config)
        self.assertIn("C:/subs/a.ass", result)

    def test_backslash_in_path(self):
        spec = _ass_spec(parameters={"ass_path": "C:\\subs\\a.ass"})
        result = fgs.serialize_ass_filter(spec, self.config)
        self.assertIn("C:\\subs\\a.ass", result)

    def test_unicode_path(self):
        spec = _ass_spec(parameters={"ass_path": "/subs/電影字幕.ass"})
        result = fgs.serialize_ass_filter(spec, self.config)
        self.assertIn("電影字幕.ass", result)

    def test_fontsdir_included_when_allowed(self):
        spec = _ass_spec(parameters={"ass_path": "/a.ass", "fontsdir": "/fonts"})
        result = fgs.serialize_ass_filter(spec, self.config)
        self.assertIn("fontsdir='/fonts'", result)

    def test_fontsdir_rejected_when_not_allowed(self):
        config = dataclasses.replace(self.config, ass_allow_fontsdir=False)
        spec = _ass_spec(parameters={"ass_path": "/a.ass", "fontsdir": "/fonts"})
        with self.assertRaises(fgs.ASSSerializationError):
            fgs.serialize_ass_filter(spec, config)

    def test_no_fontsdir_ok(self):
        spec = _ass_spec(parameters={"ass_path": "/a.ass"})
        result = fgs.serialize_ass_filter(spec, self.config)
        self.assertNotIn("fontsdir=", result)

    def test_force_style_blocked_by_default(self):
        spec = _ass_spec(parameters={"ass_path": "/a.ass", "force_style": "FontSize=20"})
        with self.assertRaises(fgs.ASSSerializationError):
            fgs.serialize_ass_filter(spec, self.config)

    def test_force_style_allowed_when_configured(self):
        config = dataclasses.replace(self.config, ass_allow_force_style=True)
        spec = _ass_spec(parameters={"ass_path": "/a.ass", "force_style": "FontSize=20"})
        result = fgs.serialize_ass_filter(spec, config)
        self.assertIn("force_style=", result)

    def test_no_ass_file_read(self):
        # No path in this module ever opens/reads .ass content.
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(".ass" in stripped and ("read_text(" in stripped or "open(" in stripped))

    def test_no_mode_fallback_structural(self):
        # No code path in this module converts ass<->drawtext automatically.
        self.assertNotIn("SubtitleRenderMode.ASS, SubtitleRenderMode.DRAWTEXT =", MODULE_SOURCE)
        self.assertNotIn("fallback_mode", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Label serialization
# ---------------------------------------------------------------------------


class LabelSerializationTests(FilterGraphSerializerTempTestCase):
    def test_stream_label_accepted(self):
        self.assertEqual(fgs.serialize_label("0:v", self.config), "0:v")

    def test_generated_label_accepted(self):
        self.assertEqual(fgs.serialize_label("v0", self.config), "v0")

    def test_outv_accepted(self):
        self.assertEqual(fgs.serialize_label("outv", self.config), "outv")

    def test_overlay_asset_label_accepted(self):
        self.assertEqual(fgs.serialize_label("overlay_asset:logo-main", self.config), "overlay_asset:logo-main")

    def test_empty_label_rejected(self):
        with self.assertRaises(fgs.FilterGraphLabelSerializationError):
            fgs.serialize_label("", self.config)

    def test_invalid_characters_rejected(self):
        with self.assertRaises(fgs.FilterGraphLabelSerializationError):
            fgs.serialize_label("bad label!", self.config)

    def test_label_starting_with_digit_but_not_stream_form_rejected(self):
        with self.assertRaises(fgs.FilterGraphLabelSerializationError):
            fgs.serialize_label("9abc", self.config)

    def test_classify_label_stream_input(self):
        result = fgs.classify_label("0:v", self.config)
        self.assertEqual(result.kind, "stream_input")

    def test_classify_label_generated(self):
        result = fgs.classify_label("v0", self.config)
        self.assertEqual(result.kind, "generated")

    def test_classify_label_final_output(self):
        result = fgs.classify_label("outv", self.config, is_final_output=True)
        self.assertEqual(result.kind, "final_output")

    def test_missing_input_label_rejected_by_serialize_filter_graph(self):
        graph = _drawtext_graph()
        graph.filters[0].label_in = ["never_declared"]
        with self.assertRaises(fgs.FilterGraphLabelSerializationError):
            fgs.serialize_filter_graph(graph, self.config)

    def test_duplicate_output_label_configuration(self):
        graph = _drawtext_graph()
        graph.outputs = ["outv", "outv"]
        with self.assertRaises(fgs.MultipleFinalOutputsError):
            fgs.serialize_filter_graph(graph, self.config)

    def test_final_output_stable_across_calls(self):
        graph = _drawtext_graph()
        result_a = fgs.serialize_filter_graph(graph, self.config)
        result_b = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result_a.final_output_label, result_b.final_output_label)


# ---------------------------------------------------------------------------
# Enable expressions
# ---------------------------------------------------------------------------


class EnableExpressionTests(FilterGraphSerializerTempTestCase):
    def test_valid_between_expression(self):
        result = fgs.serialize_enable_expression("between(t,0,2)", self.config)
        self.assertEqual(result, "between(t,0.0,2.0)")

    def test_empty_expression_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("", self.config)

    def test_negative_start_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,-1,2)", self.config)

    def test_negative_end_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,0,-2)", self.config)

    def test_reversed_range_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,5,2)", self.config)

    def test_zero_length_range_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,2,2)", self.config)

    def test_unsupported_function_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("gte(t,2)", self.config)

    def test_semicolon_injection_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,0,2);rm -rf /", self.config)

    def test_shell_like_syntax_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("$(malicious)", self.config)

    def test_nan_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,nan,2)", self.config)

    def test_infinite_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,0,inf)", self.config)

    def test_non_numeric_timing_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,abc,2)", self.config)

    def test_wrong_time_variable_rejected(self):
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(x,0,2)", self.config)

    def test_function_not_in_allowlist_rejected(self):
        config = dataclasses.replace(self.config, enable_allowed_functions=())
        with self.assertRaises(fgs.FilterGraphExpressionSerializationError):
            fgs.serialize_enable_expression("between(t,0,2)", config)

    def test_never_evaluated_only_re_emitted(self):
        # Confirms this module never calls eval/exec on an expression.
        self.assertNotIn("eval(", MODULE_SOURCE)
        self.assertNotIn("exec(", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Output mode decision
# ---------------------------------------------------------------------------


class OutputModeTests(FilterGraphSerializerTempTestCase):
    def test_linear_drawtext_uses_simple_vf(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result.output_mode, "simple_vf")
        self.assertEqual(result.filter_argument_name, "-vf")

    def test_linear_ass_uses_simple_vf(self):
        graph = _ass_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result.output_mode, "simple_vf")

    def test_overlay_filter_forces_filter_complex(self):
        spec = fgb.FilterSpec(
            filter_id="f_overlay", filter_type=fgb.FilterType.OVERLAY,
            label_in=["0:v", "overlay_asset:logo"], label_out=["outv"], parameters={"x": 0, "y": 0},
        )
        graph_pass = fgb.FilterGraphPass(pass_id="pass_overlay", pass_type="overlay", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "overlay_asset:logo", "outv"],
            inputs=["0:v", "overlay_asset:logo"], outputs=["outv"], dependencies={"pass_overlay": []},
            validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result.output_mode, "filter_complex")
        self.assertEqual(result.filter_argument_name, "-filter_complex")

    def test_multiple_stream_inputs_forces_filter_complex(self):
        spec1 = fgb.FilterSpec(filter_id="s1", filter_type=fgb.FilterType.SCALE, label_in=["0:v"], label_out=["v0"], parameters={"width": 1080, "height": 1920})
        spec2 = fgb.FilterSpec(filter_id="s2", filter_type=fgb.FilterType.SCALE, label_in=["1:v"], label_out=["v1"], parameters={"width": 1080, "height": 1920})
        concat = fgb.FilterSpec(filter_id="c1", filter_type=fgb.FilterType.CONCAT, label_in=["v0", "v1"], label_out=["outv"], parameters={"n": 2, "v": 1, "a": 0})
        graph_pass = fgb.FilterGraphPass(pass_id="pass_video", pass_type="video", filters=[spec1, spec2, concat])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec1, spec2, concat],
            labels=["0:v", "1:v", "v0", "v1", "outv"], inputs=["0:v", "1:v"], outputs=["outv"],
            dependencies={"pass_video": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result.output_mode, "filter_complex")

    def test_branching_forces_filter_complex(self):
        spec1 = fgb.FilterSpec(filter_id="s1", filter_type=fgb.FilterType.FORMAT, label_in=["0:v"], label_out=["v0"], parameters={})
        spec2 = fgb.FilterSpec(filter_id="s2", filter_type=fgb.FilterType.FORMAT, label_in=["0:v"], label_out=["v1"], parameters={})
        graph_pass = fgb.FilterGraphPass(pass_id="pass_video", pass_type="video", filters=[spec1, spec2])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec1, spec2], labels=["0:v", "v0", "v1"],
            inputs=["0:v"], outputs=["v1"], dependencies={"pass_video": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result.output_mode, "filter_complex")

    def test_forced_filter_complex_config(self):
        config = dataclasses.replace(self.config, force_filter_complex=True)
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, config)
        self.assertEqual(result.output_mode, "filter_complex")

    def test_prefer_simple_vf_false_uses_filter_complex(self):
        config = dataclasses.replace(self.config, prefer_simple_vf=False)
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, config)
        self.assertEqual(result.output_mode, "filter_complex")

    def test_zero_final_outputs_rejected(self):
        graph = _drawtext_graph()
        graph.outputs = []
        with self.assertRaises(fgs.MultipleFinalOutputsError):
            fgs.serialize_filter_graph(graph, self.config)

    def test_multiple_final_outputs_rejected(self):
        graph = _drawtext_graph()
        graph.outputs = ["outv", "outv2"]
        with self.assertRaises(fgs.MultipleFinalOutputsError):
            fgs.serialize_filter_graph(graph, self.config)

    def test_require_single_output_can_be_disabled(self):
        config = dataclasses.replace(self.config, require_single_final_video_output=False)
        graph = _drawtext_graph()
        graph.outputs = ["outv", "outv2"]
        # Still succeeds structurally when the requirement is disabled
        # (label validation for outv2 would fail since it's undeclared
        # elsewhere, so use a graph where both outputs are valid labels).
        graph.labels.append("outv2")
        graph.filters[0].label_out = ["outv"]
        result = fgs.serialize_filter_graph(graph, config)
        self.assertEqual(result.output_labels, ["outv", "outv2"])

    def test_filter_complex_wraps_labels_in_brackets(self):
        config = dataclasses.replace(self.config, force_filter_complex=True)
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, config)
        self.assertIn("[0:v]", result.filter_expression)
        self.assertIn("[outv]", result.filter_expression)

    def test_simple_vf_has_no_brackets(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertNotIn("[", result.filter_expression)
        self.assertNotIn("]", result.filter_expression)


# ---------------------------------------------------------------------------
# Top-level graph serialization
# ---------------------------------------------------------------------------


class GraphSerializationTests(FilterGraphSerializerTempTestCase):
    def test_valid_graph_serializes(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertTrue(result.filter_expression)
        self.assertEqual(result.graph_id, graph.graph_id)

    def test_unvalidated_graph_rejected(self):
        graph = _drawtext_graph()
        graph.validation.passed = False
        graph.validation.errors = ["synthetic"]
        with self.assertRaises(fgs.FilterGraphNotValidatedError):
            fgs.serialize_filter_graph(graph, self.config)

    def test_default_validation_state_rejected(self):
        # A freshly-constructed FilterGraphValidation() defaults to
        # passed=False -- the serializer must not assume success.
        graph = _drawtext_graph()
        graph.validation = fgb.FilterGraphValidation()
        with self.assertRaises(fgs.FilterGraphNotValidatedError):
            fgs.serialize_filter_graph(graph, self.config)

    def test_pass_order_preserved(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        # Only one pass/filter in this fixture; assert the single
        # filter's identity is preserved end-to-end.
        self.assertEqual(result.filters[0].filter_id, graph.filters[0].filter_id)

    def test_filter_order_preserved_multi(self):
        specs = [
            _drawtext_spec(filter_id="f1", parameters={"text": "A", "x": 1, "y": 1, "font_file": "/f.ttf"}, label_out=["v0"]),
            _drawtext_spec(filter_id="f2", parameters={"text": "B", "x": 1, "y": 1, "font_file": "/f.ttf"}, label_in=["v0"], label_out=["outv"]),
        ]
        graph_pass = fgb.FilterGraphPass(pass_id="pass_subtitle", pass_type="subtitle", filters=specs)
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=specs, labels=["0:v", "v0", "outv"],
            inputs=["0:v"], outputs=["outv"], dependencies={"pass_subtitle": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual([f.filter_id for f in result.filters], ["f1", "f2"])
        self.assertLess(result.filter_expression.index("A"), result.filter_expression.index("B"))

    def test_filter_count_preserved(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result.filter_count, len(graph.filters))

    def test_final_output_label_preserved(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result.final_output_label, graph.outputs[0])

    def test_never_reorders_or_rebuilds_graph(self):
        graph = _drawtext_graph()
        before_filters = list(graph.filters)
        before_passes = list(graph.passes)
        fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(graph.filters, before_filters)
        self.assertEqual(graph.passes, before_passes)

    def test_unsupported_filter_type_rejected_by_default(self):
        spec = fgb.FilterSpec(filter_id="f_crop", filter_type="crop", label_in=["0:v"], label_out=["outv"], parameters={})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        with self.assertRaises(fgs.UnsupportedSerializedFilterError):
            fgs.serialize_filter_graph(graph, self.config)

    def test_custom_passthrough_when_enabled(self):
        config = dataclasses.replace(self.config, allow_custom_passthrough=True)
        spec = fgb.FilterSpec(filter_id="f_crop", filter_type="crop", label_in=["0:v"], label_out=["outv"], parameters={})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, config)
        self.assertIn("crop", result.filter_expression)

    def test_scale_filter_serialization(self):
        spec = fgb.FilterSpec(filter_id="s", filter_type=fgb.FilterType.SCALE, label_in=["0:v"], label_out=["outv"], parameters={"width": 1080, "height": 1920})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("scale=1080:1920", result.filter_expression)

    def test_format_filter_serialization(self):
        spec = fgb.FilterSpec(filter_id="f", filter_type=fgb.FilterType.FORMAT, label_in=["0:v"], label_out=["outv"], parameters={"pixel_format": "yuv420p"})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("format=yuv420p", result.filter_expression)

    def test_setpts_filter_serialization(self):
        spec = fgb.FilterSpec(filter_id="s", filter_type=fgb.FilterType.SETPTS, label_in=["0:v"], label_out=["outv"], parameters={})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("setpts=PTS-STARTPTS", result.filter_expression)

    def test_trim_filter_serialization(self):
        spec = fgb.FilterSpec(filter_id="t", filter_type=fgb.FilterType.TRIM, label_in=["0:v"], label_out=["outv"], parameters={"start_seconds": 1, "end_seconds": 5})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("trim=start=1:end=5", result.filter_expression)

    def test_fade_filter_serialization(self):
        spec = fgb.FilterSpec(filter_id="fd", filter_type=fgb.FilterType.FADE, label_in=["0:v"], label_out=["outv"], parameters={"type": "in", "start_seconds": 0, "duration_seconds": 2})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("fade=t=in:st=0:d=2", result.filter_expression)

    def test_alpha_filter_serialization(self):
        spec = fgb.FilterSpec(filter_id="a", filter_type=fgb.FilterType.ALPHA, label_in=["0:v"], label_out=["outv"], parameters={"opacity": 0.5})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "outv"], inputs=["0:v"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("colorchannelmixer=aa=0.5", result.filter_expression)

    def test_concat_filter_serialization_labels_only(self):
        spec = fgb.FilterSpec(filter_id="c", filter_type=fgb.FilterType.CONCAT, label_in=["v0", "v1"], label_out=["outv"], parameters={"n": 2, "v": 1, "a": 0})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="video", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["v0", "v1", "outv"], inputs=["v0", "v1"],
            outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("concat=n=2:v=1:a=0", result.filter_expression)

    def test_overlay_filter_structural_serialization(self):
        spec = fgb.FilterSpec(filter_id="o", filter_type=fgb.FilterType.OVERLAY, label_in=["0:v", "overlay_asset:logo"], label_out=["outv"], parameters={"x": 10, "y": 20})
        graph_pass = fgb.FilterGraphPass(pass_id="p", pass_type="overlay", filters=[spec])
        graph = fgb.FilterGraph(
            graph_id="g", passes=[graph_pass], filters=[spec], labels=["0:v", "overlay_asset:logo", "outv"],
            inputs=["0:v", "overlay_asset:logo"], outputs=["outv"], dependencies={"p": []}, validation=_passed_validation(),
        )
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("overlay=x=10:y=20", result.filter_expression)


# ---------------------------------------------------------------------------
# Real overlay filter serialization (Phase 11F.5)
# ---------------------------------------------------------------------------


class OverlayFilterSerializationTests(FilterGraphSerializerTempTestCase):
    def test_overlay_filter_references_real_stream_label(self):
        graph = _overlay_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertIn("[1:v]", result.filter_expression)

    def test_format_auto_emitted_when_alpha(self):
        spec = _overlay_spec(parameters={"x": 10, "y": 20, "has_alpha": True})
        body = fgs.serialize_overlay_filter(spec, self.config)
        self.assertIn(":format=auto", body)

    def test_format_auto_omitted_when_no_alpha(self):
        spec = _overlay_spec(parameters={"x": 10, "y": 20, "has_alpha": False})
        body = fgs.serialize_overlay_filter(spec, self.config)
        self.assertNotIn("format=auto", body)

    def test_format_auto_omitted_when_alpha_unknown(self):
        spec = _overlay_spec(parameters={"x": 10, "y": 20})
        body = fgs.serialize_overlay_filter(spec, self.config)
        self.assertNotIn("format=auto", body)

    def test_format_auto_suppressed_by_config(self):
        spec = _overlay_spec(parameters={"x": 10, "y": 20, "has_alpha": True})
        config = dataclasses.replace(self.config, overlay_format_auto_when_alpha=False)
        body = fgs.serialize_overlay_filter(spec, config)
        self.assertNotIn("format=auto", body)

    def test_custom_overlay_filter_name(self):
        spec = _overlay_spec(parameters={"x": 10, "y": 20})
        config = dataclasses.replace(self.config, overlay_filter_name="overlay_cuda")
        body = fgs.serialize_overlay_filter(spec, config)
        self.assertTrue(body.startswith("overlay_cuda="))

    def test_extra_inputs_surfaced_in_serialized_graph(self):
        graph = _overlay_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(len(result.extra_inputs), 1)
        self.assertEqual(result.extra_inputs[0].label, "1:v")
        self.assertEqual(result.extra_inputs[0].resolved_path, "/assets/logo.png")

    def test_extra_input_label_revalidated(self):
        bad_extra_inputs = [fgb.ExtraInputSpec(label="not valid!", resolved_path="/assets/logo.png", asset_id="a1", logical_role="logo")]
        graph = _overlay_graph(extra_inputs=bad_extra_inputs, label_in=["0:v", "not valid!"])
        with self.assertRaises(fgs.FilterGraphLabelSerializationError):
            fgs.serialize_filter_graph(graph, self.config)

    def test_unused_extra_input_warns(self):
        extra_inputs = [
            fgb.ExtraInputSpec(label="1:v", resolved_path="/assets/logo.png", asset_id="a1", logical_role="logo"),
            fgb.ExtraInputSpec(label="2:v", resolved_path="/assets/unused.png", asset_id="a2", logical_role="cta"),
        ]
        graph = _overlay_graph(extra_inputs=extra_inputs)
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertTrue(any("unused extra input" in w for w in result.warnings))

    def test_no_extra_inputs_no_warning(self):
        result = fgs.serialize_filter_graph(_drawtext_graph(), self.config)
        self.assertEqual(result.extra_inputs, [])
        self.assertFalse(any("unused extra input" in w for w in result.warnings))

    def test_extra_inputs_round_trip(self):
        graph = _overlay_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        data = fgs.serialized_filter_graph_to_dict(result)
        restored = fgs.serialized_filter_graph_from_dict(data)
        self.assertEqual(restored.extra_inputs[0].label, result.extra_inputs[0].label)
        self.assertEqual(restored.extra_inputs[0].resolved_path, result.extra_inputs[0].resolved_path)

    def test_serialization_id_changes_with_extra_input_path(self):
        graph_a = _overlay_graph()
        graph_b = _overlay_graph(
            extra_inputs=[fgb.ExtraInputSpec(label="1:v", resolved_path="/assets/other.png", asset_id="a1", logical_role="logo")],
        )
        result_a = fgs.serialize_filter_graph(graph_a, self.config)
        result_b = fgs.serialize_filter_graph(graph_b, self.config)
        self.assertNotEqual(result_a.serialization_id, result_b.serialization_id)

    def test_overlay_forces_filter_complex_and_real_syntax_together(self):
        graph = _overlay_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result.output_mode, fgs.FilterGraphOutputMode.FILTER_COMPLEX)
        self.assertIn("overlay=x=10:y=20:format=auto", result.filter_expression)


# ---------------------------------------------------------------------------
# Deterministic serialization_id
# ---------------------------------------------------------------------------


class SerializationIdTests(FilterGraphSerializerTempTestCase):
    def test_stable_across_repeated_serialization(self):
        graph = _drawtext_graph()
        result_a = fgs.serialize_filter_graph(graph, self.config)
        result_b = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(result_a.serialization_id, result_b.serialization_id)

    def test_excludes_timestamp(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        # No "created_at"-equivalent field feeds the hash -- SerializedFilterGraph
        # itself carries no timestamp field at all.
        self.assertFalse(hasattr(result, "created_at"))

    def test_changes_with_filter_expression(self):
        graph_a = _drawtext_graph()
        graph_b = _drawtext_graph(parameters={"text": "Different text", "x": 10, "y": 20, "font_file": "/fonts/body.ttf"})
        result_a = fgs.serialize_filter_graph(graph_a, self.config)
        result_b = fgs.serialize_filter_graph(graph_b, self.config)
        self.assertNotEqual(result_a.serialization_id, result_b.serialization_id)

    def test_changes_with_output_mode(self):
        graph = _drawtext_graph()
        forced_config = dataclasses.replace(self.config, force_filter_complex=True)
        result_a = fgs.serialize_filter_graph(graph, self.config)
        result_b = fgs.serialize_filter_graph(graph, forced_config)
        self.assertNotEqual(result_a.serialization_id, result_b.serialization_id)

    def test_is_16_hex_chars(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        self.assertEqual(len(result.serialization_id), 16)
        int(result.serialization_id, 16)


# ---------------------------------------------------------------------------
# JSON / diagnostics
# ---------------------------------------------------------------------------


class JSONDiagnosticsTests(FilterGraphSerializerTempTestCase):
    def test_round_trip(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        data = fgs.serialized_filter_graph_to_dict(result)
        restored = fgs.serialized_filter_graph_from_dict(data)
        self.assertEqual(restored.serialization_id, result.serialization_id)
        self.assertEqual(restored.filter_expression, result.filter_expression)

    def test_to_dict_json_serializable(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        json.dumps(fgs.serialized_filter_graph_to_dict(result))

    def test_atomic_write_no_tmp_left_behind(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        path = self.temp_dir / "serialization.json"
        fgs.save_filter_graph_serialization(result, path)
        self.assertTrue(path.exists())
        self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_overwrite_refused_without_force(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        path = self.temp_dir / "serialization.json"
        fgs.save_filter_graph_serialization(result, path)
        with self.assertRaises(fgs.FilterGraphSerializationJSONError):
            fgs.save_filter_graph_serialization(result, path)

    def test_overwrite_with_force(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        path = self.temp_dir / "serialization.json"
        fgs.save_filter_graph_serialization(result, path)
        fgs.save_filter_graph_serialization(result, path, force=True)

    def test_save_rejects_directory(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        with self.assertRaises(fgs.FilterGraphSerializationJSONError):
            fgs.save_filter_graph_serialization(result, self.temp_dir)

    def test_load_missing_file_raises(self):
        with self.assertRaises(fgs.FilterGraphSerializationJSONError):
            fgs.load_filter_graph_serialization(self.temp_dir / "missing.json")

    def test_load_malformed_json_raises(self):
        path = self.temp_dir / "bad.json"
        path.write_text("{not valid", encoding="utf-8")
        with self.assertRaises(fgs.FilterGraphSerializationJSONError):
            fgs.load_filter_graph_serialization(path)

    def test_load_non_object_root_raises(self):
        path = self.temp_dir / "list.json"
        path.write_text("[]", encoding="utf-8")
        with self.assertRaises(fgs.FilterGraphSerializationJSONError):
            fgs.load_filter_graph_serialization(path)

    def test_malformed_filters_entry_raises_cleanly(self):
        with self.assertRaises(fgs.FilterGraphSerializationJSONError):
            fgs.serialized_filter_graph_from_dict({"filters": ["not-an-object"]})

    def test_unknown_metadata_preserved(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        data = fgs.serialized_filter_graph_to_dict(result)
        data["metadata"]["custom_future_field"] = "kept"
        restored = fgs.serialized_filter_graph_from_dict(data)
        self.assertEqual(restored.metadata.get("custom_future_field"), "kept")

    def test_write_diagnostics_helper_returns_result(self):
        graph = _drawtext_graph()
        result = fgs.serialize_filter_graph(graph, self.config)
        path = self.temp_dir / "diag.json"
        wrapped = fgs.write_filter_graph_serialization_diagnostics(result, path)
        self.assertTrue(path.exists())
        self.assertEqual(wrapped.serialized.serialization_id, result.serialization_id)
        self.assertTrue(wrapped.written_at)


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

    def test_no_media_mutation(self):
        for forbidden in ("write_bytes(", "unlink(", "os.remove("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_png_logo_watermark_execution(self):
        for forbidden in ("PNG", "logo_path", "watermark", "sticker"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_automatic_ass_drawtext_fallback(self):
        self.assertNotIn("except ASSSerializationError", MODULE_SOURCE)
        self.assertNotIn("except DrawTextSerializationError", MODULE_SOURCE)

    def test_no_playwright_or_publish_reference(self):
        for forbidden in ("playwright", "Playwright", "instagram", "Instagram", "publish_reel", "upload_reel"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_network_or_download_code(self):
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_shell_true(self):
        self.assertNotIn("shell=True", MODULE_SOURCE)

    def test_repo_config_has_no_functional_ffmpeg_reference(self):
        content = (Path(__file__).resolve().parents[1] / "config" / "video" / "filter_graph_serializer.yaml").read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotIn("ffmpeg_binary", stripped)
            self.assertNotIn("ffprobe_binary", stripped)

    def test_no_import_of_execution_engines(self):
        for forbidden in ("video_engine", "renderer_execution_engine", "music_mixer", "media_inspector", "reel_builder", "subtitle_render_engine"):
            for line in MODULE_SOURCE.splitlines():
                stripped = line.strip()
                if stripped.startswith("from . import") or stripped.startswith("from ."):
                    self.assertNotIn(forbidden, stripped, f"{forbidden} should not be imported by filter_graph_serializer.py")

    def test_no_pillow_import_or_pixel_access(self):
        # Phase 11F.5: overlay filter serialization consumes only
        # already-resolved semantic parameters (has_alpha, resolved_path
        # as a string) from filter_graph_builder -- it never opens,
        # decodes, or reads image bytes itself.
        for forbidden in ("import PIL", "from PIL", "Image.open("):
            self.assertNotIn(forbidden, MODULE_SOURCE)


if __name__ == "__main__":
    unittest.main()
