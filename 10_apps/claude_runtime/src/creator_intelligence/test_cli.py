import json
import tempfile
import unittest
from pathlib import Path

from src.creator_intelligence.cli import main, parse_arguments


def _write_evidence_file(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def _valid_entry(**overrides):
    entry = {
        "evidence_type": "manual_note",
        "source_description": "operator note",
        "content_excerpt": "observed pattern",
        "collected_by": "operator_a",
        "tags": ["persona"],
    }
    entry.update(overrides)
    return entry


class ParseArgumentsTests(unittest.TestCase):
    def test_validate_requires_evidence_flag(self):
        with self.assertRaises(SystemExit):
            parse_arguments(["validate"])

    def test_build_requires_subject_label_and_output(self):
        with self.assertRaises(SystemExit):
            parse_arguments(["build", "--evidence", "x.json"])

    def test_show_requires_dna_flag(self):
        with self.assertRaises(SystemExit):
            parse_arguments(["show"])

    def test_valid_validate_args_parsed(self):
        args = parse_arguments(["validate", "--evidence", "x.json"])
        self.assertEqual(args.command, "validate")
        self.assertEqual(str(args.evidence), "x.json")


class CliTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class ValidateCommandTests(CliTempTestCase):
    def test_valid_evidence_file_exits_zero(self):
        evidence_path = self.tmp_path / "evidence.json"
        _write_evidence_file(evidence_path, [_valid_entry()])
        rc = main(["validate", "--evidence", str(evidence_path), "--json"])
        self.assertEqual(rc, 0)

    def test_invalid_evidence_type_exits_nonzero(self):
        evidence_path = self.tmp_path / "evidence.json"
        _write_evidence_file(evidence_path, [_valid_entry(evidence_type="scraped")])
        rc = main(["validate", "--evidence", str(evidence_path), "--json"])
        self.assertEqual(rc, 1)

    def test_missing_evidence_file_exits_nonzero(self):
        rc = main(["validate", "--evidence", str(self.tmp_path / "missing.json"), "--json"])
        self.assertEqual(rc, 1)

    def test_evidence_file_not_a_list_exits_nonzero(self):
        evidence_path = self.tmp_path / "evidence.json"
        evidence_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        rc = main(["validate", "--evidence", str(evidence_path), "--json"])
        self.assertEqual(rc, 1)

    def test_evidence_entry_missing_required_field_exits_nonzero(self):
        evidence_path = self.tmp_path / "evidence.json"
        _write_evidence_file(evidence_path, [{"source_description": "no type"}])
        rc = main(["validate", "--evidence", str(evidence_path), "--json"])
        self.assertEqual(rc, 1)


class BuildCommandTests(CliTempTestCase):
    def test_build_writes_output_file(self):
        evidence_path = self.tmp_path / "evidence.json"
        _write_evidence_file(evidence_path, [_valid_entry()])
        output_path = self.tmp_path / "dna.json"
        rc = main(
            ["build", "--evidence", str(evidence_path), "--subject-label", "Demo Creator X", "--output", str(output_path), "--json"]
        )
        self.assertEqual(rc, 0)
        self.assertTrue(output_path.is_file())

    def test_build_refuses_to_overwrite_without_force(self):
        evidence_path = self.tmp_path / "evidence.json"
        _write_evidence_file(evidence_path, [_valid_entry()])
        output_path = self.tmp_path / "dna.json"
        output_path.write_text("{}", encoding="utf-8")
        rc = main(
            ["build", "--evidence", str(evidence_path), "--subject-label", "Demo Creator X", "--output", str(output_path), "--json"]
        )
        self.assertEqual(rc, 1)

    def test_build_overwrites_with_force(self):
        evidence_path = self.tmp_path / "evidence.json"
        _write_evidence_file(evidence_path, [_valid_entry()])
        output_path = self.tmp_path / "dna.json"
        output_path.write_text("{}", encoding="utf-8")
        rc = main(
            [
                "build", "--evidence", str(evidence_path), "--subject-label", "Demo Creator X",
                "--output", str(output_path), "--force", "--json",
            ]
        )
        self.assertEqual(rc, 0)

    def test_build_output_round_trips_via_show(self):
        evidence_path = self.tmp_path / "evidence.json"
        _write_evidence_file(evidence_path, [_valid_entry()])
        output_path = self.tmp_path / "dna.json"
        main(["build", "--evidence", str(evidence_path), "--subject-label", "Demo Creator X", "--output", str(output_path), "--json"])
        rc = main(["show", "--dna", str(output_path), "--json"])
        self.assertEqual(rc, 0)


class ShowCommandTests(CliTempTestCase):
    def test_show_missing_file_exits_nonzero(self):
        rc = main(["show", "--dna", str(self.tmp_path / "missing.json"), "--json"])
        self.assertEqual(rc, 1)


class OutputFormatTests(CliTempTestCase):
    def test_json_flag_produces_parseable_json_on_stdout(self):
        import io
        import contextlib

        evidence_path = self.tmp_path / "evidence.json"
        _write_evidence_file(evidence_path, [_valid_entry()])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["validate", "--evidence", str(evidence_path), "--json"])
        json.loads(buffer.getvalue())

    def test_no_network_import_anywhere_in_cli_module(self):
        import src.creator_intelligence.cli as cli_module

        source = Path(cli_module.__file__).read_text(encoding="utf-8")
        for forbidden in ("import requests", "import httpx", "import playwright", "import selenium"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
