# Instagram Live Research Runbook

**This document describes a smoke test that has NOT been run.** Every
step below requires the user's explicit, separate go-ahead before any
part of it executes -- Phase 12B.2 built and tested this connector
entirely against `FakeBrowserAdapter`; nothing in it has touched a
real browser or the real Instagram. Do not use `carlysuen112` or
`aitana_10_01` for the first smoke test, or any test, without a
separate explicit authorization even after live network access itself
is approved.

## Before running anything

1. Re-read [instagram_connector.md](instagram_connector.md) and
   [connector_contract.md](connector_contract.md) in full.
2. Confirm `config/creator_research/instagram.yaml`'s `instagram.read_only`
   is `true` and `access_limit_behavior` is `"skip"` (never modify
   these to attempt bypassing an access wall).
3. Confirm `browser.profile_directory` points at
   `browser_profile/creator_research_instagram` -- a profile separate
   from the reply-production system's own `browser_profile/instagram`.
4. Tune `selectors.py`'s candidate lists against a real, logged-in (or
   logged-out, for public content) Instagram session in a normal
   browser first -- the shipped defaults are untested placeholders
   (see instagram_connector.md's "Selector maintenance" section).

## Preparing a user-controlled browser session

1. A human operator runs Chromium via
   `navigator.PlaywrightInstagramBrowserAdapter`, pointed at
   `browser_profile/creator_research_instagram`, with `headless: false`.
2. If viewing content that requires being logged in, the operator logs
   in **manually**, in that visible browser window. This connector
   never reads, stores, or types a credential -- login is entirely the
   human's own action, exactly like `src/social/instagram_session.py`'s
   established Phase 8A pattern.
3. Close the browser normally (`adapter.close()`) when done -- the
   persistent profile retains the session for next time.

## Verifying read-only mode before any real run

Run the existing unit suite one more time immediately before a live
attempt, to confirm nothing has drifted:

```
python3 -m unittest discover -s src/creator_research/instagram -p "test_*.py" -t .
```

All tests must pass, including `test_structural_safety.py` (confirms
no credential handling, no forbidden action methods, no scraping
infrastructure exists anywhere in the package).

## Running the approved minimal smoke test

**Scope, exactly as approved -- nothing more:**
- One public profile.
- `collect_profile()` only -- no other section.
- Exactly one navigation (the profile page itself).
- No scrolling beyond what a normal profile-page load shows.
- No comments, no Reels, no Highlights.
- No media download.
- Writes confined to the configured `checkpoint.directory`/
  `diagnostics.directory` (both under `output/`, gitignored).

```python
from src.creator_research.instagram.config import load_instagram_connector_config
from src.creator_research.instagram.connector import InstagramResearchConnector
from src.creator_research.instagram.navigator import PlaywrightInstagramBrowserAdapter
from src.creator_research.jobs import ResearchJob
from src.creator_research.interfaces import ConnectorSection

config = load_instagram_connector_config()
adapter = PlaywrightInstagramBrowserAdapter(
    config.resolved_profile_directory(),
    headless=config.headless,
    viewport_width=config.viewport_width,
    viewport_height=config.viewport_height,
)
connector = InstagramResearchConnector(adapter, config)

job = ResearchJob(
    creator_id="<operator-chosen label, not a real handle unless separately authorized>",
    platform="instagram",
    username="<public profile username>",
    profile_url="https://www.instagram.com/<public profile username>/",
    connector_name="instagram",
    requested_sections=(ConnectorSection.PROFILE,),
)

bundle = connector.collect_profile(job)
print(bundle.items, bundle.warnings)
connector.save_diagnostics_for(job)
adapter.close()
```

**Stop immediately and report back if:**
- `AuthenticationRequiredError`/`AccessLimitedError`/`RateLimitedError`/
  `ChallengeDetectedError` is raised -- this is the connector working
  correctly (detecting and refusing to push past an obstacle), not a
  bug to route around.
- Any selector needs adjusting -- do that in a separate, reviewed
  change to `selectors.py`, not ad hoc.

## Inspecting diagnostics

```
cat output/creator_research/instagram/diagnostics/<connector_run_id>.json
```

Confirms `sections_attempted`, `sections_completed`,
`items_collected`, and any `access_limit_events`/`selector_failures`
-- review before deciding whether to expand scope.

## Resuming

If a later, separately-approved run needs to continue a prior job,
`checkpoint.load_checkpoint()` (used automatically by
`InstagramResearchConnector`) resumes from
`output/creator_research/instagram/checkpoints/<job_id>_<section>.json`,
revalidating `job_id`/`creator_id`/`connector_version` first.

## Handling access limits safely

If a section reports `access_limited`/`authentication_required`: stop,
do not adjust selectors to "find another way in," and do not enable
any bypass. If it's `rate_limited`: wait substantially longer than the
configured backoff before trying again, and consider whether the
request volume itself needs reducing (lower `max_posts_per_job`, etc.)
before retrying. If `challenge_detected`: stop entirely and have the
human operator resolve it manually in the visible browser (if they
choose to), never programmatically.
