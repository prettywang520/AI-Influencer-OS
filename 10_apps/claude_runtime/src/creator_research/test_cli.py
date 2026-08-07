import json
import tempfile
import unittest
from pathlib import Path

from src.creator_research.cli import main, parse_arguments


class CliTempTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.queue_path = self.tmp_path / "queue.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _create_job(self, **overrides):
        args = [
            "--create-job",
            "--creator-id", overrides.get("creator_id", "c1"),
            "--platform", overrides.get("platform", "instagram"),
            "--username", overrides.get("username", "demo"),
            "--profile-url", overrides.get("profile_url", "https://x/demo"),
            "--queue", str(self.queue_path),
            "--json",
        ]
        main(args)
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        return raw[0]["job_id"]


class ParseArgumentsTests(unittest.TestCase):
    def test_requires_one_action(self):
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_create_job_requires_identity_fields(self):
        with self.assertRaises(SystemExit):
            parse_arguments(["--create-job"])

    def test_mutually_exclusive_actions_rejected_together(self):
        with self.assertRaises(SystemExit):
            parse_arguments(["--create-job", "--validate"])

    def test_no_connector_argument_exists_on_the_parser(self):
        # Structural guarantee: this CLI can never construct/accept a
        # Connector, so it has no way to actually run collection.
        args = parse_arguments(
            ["--create-job", "--creator-id", "c1", "--platform", "instagram", "--username", "u", "--profile-url", "https://x/u"]
        )
        arg_names = vars(args).keys()
        for forbidden in ("connector", "run", "execute", "playwright", "browser"):
            matches = [name for name in arg_names if forbidden in name.lower() and name != "connector_name"]
            self.assertEqual(matches, [], f"unexpected connector-execution-shaped flag: {matches}")


class CreateJobTests(CliTempTestCase):
    def test_create_job_exits_zero(self):
        rc = main(
            [
                "--create-job", "--creator-id", "c1", "--platform", "instagram", "--username", "demo",
                "--profile-url", "https://x/demo", "--queue", str(self.queue_path), "--json",
            ]
        )
        self.assertEqual(rc, 0)

    def test_create_job_is_idempotent(self):
        job_id_a = self._create_job()
        job_id_b = self._create_job()
        self.assertEqual(job_id_a, job_id_b)
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        self.assertEqual(len(raw), 1)

    def test_create_job_with_sections_filters(self):
        rc = main(
            [
                "--create-job", "--creator-id", "c1", "--platform", "instagram", "--username", "demo",
                "--profile-url", "https://x/demo", "--sections", "profile,grid",
                "--queue", str(self.queue_path), "--json",
            ]
        )
        self.assertEqual(rc, 0)
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        self.assertEqual(raw[0]["requested_sections"], ["profile", "grid"])

    def test_create_job_with_invalid_section_exits_nonzero(self):
        rc = main(
            [
                "--create-job", "--creator-id", "c1", "--platform", "instagram", "--username", "demo",
                "--profile-url", "https://x/demo", "--sections", "not_a_real_section",
                "--queue", str(self.queue_path), "--json",
            ]
        )
        self.assertEqual(rc, 1)

    def test_create_job_with_priority(self):
        rc = main(
            [
                "--create-job", "--creator-id", "c1", "--platform", "instagram", "--username", "demo",
                "--profile-url", "https://x/demo", "--priority", "5",
                "--queue", str(self.queue_path), "--json",
            ]
        )
        self.assertEqual(rc, 0)
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        self.assertEqual(raw[0]["priority"], 5)


class StatusTests(CliTempTestCase):
    def test_status_no_job_id_lists_all(self):
        self._create_job()
        rc = main(["--status", "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 0)

    def test_status_with_job_id_returns_that_job(self):
        job_id = self._create_job()
        rc = main(["--status", job_id, "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 0)

    def test_status_unknown_job_id_exits_nonzero(self):
        rc = main(["--status", "does_not_exist", "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 1)


class CancelTests(CliTempTestCase):
    def test_cancel_sets_job_failed(self):
        job_id = self._create_job()
        rc = main(["--cancel", job_id, "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 0)
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        self.assertEqual(raw[0]["status"], "failed")

    def test_cancel_unknown_job_id_exits_nonzero(self):
        rc = main(["--cancel", "does_not_exist", "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 1)


class ResumeTests(CliTempTestCase):
    def test_resume_failed_job_retries_it(self):
        job_id = self._create_job()
        main(["--cancel", job_id, "--queue", str(self.queue_path), "--json"])
        rc = main(["--resume", job_id, "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 0)
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        self.assertEqual(raw[0]["status"], "queued")

    def test_resume_queued_job_exits_nonzero(self):
        job_id = self._create_job()
        rc = main(["--resume", job_id, "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 1)

    def test_resume_unknown_job_exits_nonzero(self):
        rc = main(["--resume", "does_not_exist", "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 1)


class ValidateTests(CliTempTestCase):
    def test_validate_passes_on_clean_queue(self):
        self._create_job()
        rc = main(["--validate", "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 0)

    def test_validate_passes_on_missing_queue_file(self):
        rc = main(["--validate", "--queue", str(self.tmp_path / "does_not_exist.json"), "--json"])
        self.assertEqual(rc, 0)

    def test_validate_fails_on_tampered_job_id(self):
        self._create_job()
        raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
        raw[0]["job_id"] = "0" * 16
        self.queue_path.write_text(json.dumps(raw), encoding="utf-8")
        rc = main(["--validate", "--queue", str(self.queue_path), "--json"])
        self.assertEqual(rc, 1)

    def test_validate_never_writes_to_queue_file(self):
        self._create_job()
        before = self.queue_path.read_text(encoding="utf-8")
        main(["--validate", "--queue", str(self.queue_path), "--json"])
        after = self.queue_path.read_text(encoding="utf-8")
        self.assertEqual(before, after)


class OutputFormatTests(CliTempTestCase):
    def test_json_flag_produces_parseable_json_on_stdout(self):
        import contextlib
        import io

        self._create_job()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["--status", "--queue", str(self.queue_path), "--json"])
        json.loads(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
