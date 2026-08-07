import tempfile
import unittest
from pathlib import Path

from src.creator_research.instagram.config import (
    InstagramConfigError,
    default_config_path,
    load_instagram_connector_config,
)


class LoadInstagramConnectorConfigTests(unittest.TestCase):
    def test_loads_real_default_config(self):
        config = load_instagram_connector_config()
        self.assertEqual(config.connector_version, "12B.2")

    def test_default_path_exists_on_disk(self):
        self.assertTrue(default_config_path().is_file())

    def test_read_only_defaults_true(self):
        config = load_instagram_connector_config()
        self.assertTrue(config.read_only)

    def test_forbidden_actions_loaded(self):
        config = load_instagram_connector_config()
        for action in ("follow", "unfollow", "like", "unlike", "comment", "reply", "dm", "share", "save", "publish", "edit_profile"):
            self.assertIn(action, config.forbidden_actions)

    def test_allowed_actions_loaded(self):
        config = load_instagram_connector_config()
        for action in ("navigate", "read", "scroll"):
            self.assertIn(action, config.allowed_actions)

    def test_limits_loaded(self):
        config = load_instagram_connector_config()
        self.assertEqual(config.max_posts_per_job, 100)
        self.assertEqual(config.max_reels_per_job, 50)
        self.assertEqual(config.max_comments_per_post, 100)
        self.assertEqual(config.max_highlight_items, 100)
        self.assertEqual(config.max_scroll_rounds_per_section, 50)

    def test_rate_limit_defaults(self):
        config = load_instagram_connector_config()
        self.assertTrue(config.rate_limit_enabled)
        self.assertEqual(config.minimum_delay_seconds, 1.5)
        self.assertEqual(config.maximum_delay_seconds, 4.0)
        self.assertEqual(config.backoff_seconds, (10.0, 30.0, 60.0))
        self.assertEqual(config.max_retries_per_navigation, 2)

    def test_browser_defaults(self):
        config = load_instagram_connector_config()
        self.assertFalse(config.headless)
        self.assertTrue(config.reuse_existing_profile)
        self.assertEqual(config.profile_directory, "browser_profile/creator_research_instagram")

    def test_profile_directory_differs_from_reply_system(self):
        config = load_instagram_connector_config()
        self.assertNotEqual(config.profile_directory, "browser_profile/instagram")

    def test_resolved_profile_directory_is_absolute(self):
        config = load_instagram_connector_config()
        self.assertTrue(config.resolved_profile_directory().is_absolute())

    def test_resolved_checkpoint_directory_is_absolute(self):
        config = load_instagram_connector_config()
        self.assertTrue(config.resolved_checkpoint_directory().is_absolute())

    def test_resolved_diagnostics_directory_is_absolute(self):
        config = load_instagram_connector_config()
        self.assertTrue(config.resolved_diagnostics_directory().is_absolute())

    def test_evidence_defaults(self):
        config = load_instagram_connector_config()
        self.assertTrue(config.save_text)
        self.assertFalse(config.save_visual_annotations)

    def test_checkpoint_defaults(self):
        config = load_instagram_connector_config()
        self.assertTrue(config.checkpoint_enabled)
        self.assertEqual(config.checkpoint_every_items, 10)

    def test_diagnostics_defaults(self):
        config = load_instagram_connector_config()
        self.assertTrue(config.screenshots_on_error)
        self.assertTrue(config.redact_usernames_in_audience_comments)

    def test_access_limit_behavior_default(self):
        config = load_instagram_connector_config()
        self.assertEqual(config.access_limit_behavior, "skip")

    def test_missing_config_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "does_not_exist.yaml"
            with self.assertRaises(InstagramConfigError):
                load_instagram_connector_config(missing)

    def test_empty_config_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty = Path(tmp_dir) / "empty.yaml"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(InstagramConfigError):
                load_instagram_connector_config(empty)

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad = Path(tmp_dir) / "bad.yaml"
            bad.write_text("instagram: [unclosed", encoding="utf-8")
            with self.assertRaises(InstagramConfigError):
                load_instagram_connector_config(bad)

    def test_custom_config_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom = Path(tmp_dir) / "custom.yaml"
            custom.write_text(
                "instagram:\n  connector_version: '99.0'\nlimits:\n  max_posts_per_job: 5\n", encoding="utf-8"
            )
            config = load_instagram_connector_config(custom)
            self.assertEqual(config.connector_version, "99.0")
            self.assertEqual(config.max_posts_per_job, 5)

    def test_no_credential_field_exists_on_config(self):
        config = load_instagram_connector_config()
        field_names = {f for f in config.__dataclass_fields__}
        for forbidden in ("username", "password", "cookie", "token", "session_id", "auth"):
            self.assertNotIn(forbidden, field_names)


if __name__ == "__main__":
    unittest.main()
