# Instagram Research Connector

**Phase:** 12B.2
**Status:** Built and tested against a synthetic `FakeBrowserAdapter`
only. No live Instagram access, and no analysis of any real creator,
has occurred.

## Architecture

```
ResearchJob(platform="instagram", connector_name="instagram")
    |
    v
orchestrator.run_job()                          (creator_research, Phase 12B.1, unmodified)
    |
    v
InstagramResearchConnector.collect_*()           (this package)
    |                     \
    |                      -> rate_limit.RateLimiter (polite delay)
    |                      -> checkpoint.py (load/validate/save, per section)
    v
observer.observe_*()                             (adapter-driven page observation)
    |                     \
    |                      -> selectors.py (layered candidate lists)
    |                      -> navigator.InstagramBrowserAdapter (Protocol)
    v
parser.py                                        (pure text interpretation)
    |
    v
models.py records (ProfileRecord, PostRecord, ...)
    |
    v
evidence_mapper.py
    |
    v
creator_intelligence.evidence.Evidence           (reused directly, unmodified)
    |
    v
interfaces.EvidenceBundle -> orchestrator's EvidenceQueue -> ResearchReport
```

This connector never creates a second orchestration system -- it is
one `BaseConnector` subclass, run entirely through the existing
`orchestrator.run_job()`.

## Browser adapter

`navigator.InstagramBrowserAdapter` is a `Protocol` with 11 methods
(`open_profile`, `get_current_url`, `wait_for_page_ready`, `query`,
`query_all`, `click`, `scroll`, `get_text`, `get_attribute`,
`get_screenshot_reference`, `close`). Two implementations:

- **`PlaywrightInstagramBrowserAdapter`** -- the real adapter. Opens a
  persistent Chromium context (`playwright.sync_api`) against its own
  profile directory (`browser_profile/creator_research_instagram` by
  default -- **not** `browser_profile/instagram`, the reply-production
  system's own profile; two Playwright processes cannot safely share
  one `user_data_dir`). Has no method that types into a username/
  password field, reads a cookie, or solves a challenge -- login is
  always a human, manually, in a visible browser, exactly like
  `src/social/instagram_session.py`'s own established pattern (this
  package doesn't import that module -- see "Why not
  `src/social/instagram_session.py`" below -- but follows the same
  proven approach independently).
- **`FakeBrowserAdapter`** -- in-memory test/demo double. Tests
  register a small fake "DOM" (`FakePage.register(selector, elements,
  reveal_batch=...)`) and drive the connector against it with zero
  network access and zero Playwright dependency.

### Why not `src/social/instagram_session.py`?

`creator_research` is a deliberately separate top-level package from
the reply/publishing system. Importing `instagram_session.py` would
create a live dependency from the research agent onto the
reply-production package -- exactly the coupling the "do not modify
social reply production / Instagram publisher" boundary exists to
protect against, even for a read-only import. This connector instead
follows the *pattern* (persistent context, manual-login-only, pure
classification functions) without the *code*.

## Connector contract

`InstagramResearchConnector(BaseConnector)` implements all 10
`collect_*` methods. Each:

1. Applies `rate_limit.RateLimiter.polite_delay()` if
   `rate_limit.enabled`.
2. Navigates via the adapter and checks for an access-limit signal
   (`observer.detect_access_event()`) before observing anything.
3. Delegates to the matching `observer.observe_*()` function.
4. Converts the result to `Evidence` via `evidence_mapper.py`.
5. Updates that section's checkpoint (grid/reels) and diagnostics.
6. Returns an `EvidenceBundle`.

`collect_posts`/`collect_captions`/`collect_comments`/
`collect_creator_replies`/`collect_relationships`/
`collect_visual_examples` all depend on `collect_grid`'s discovery
step; `InstagramResearchConnector` caches discovered items/posts/
captions per `job_id` internally so calling multiple sections in one
job run only navigates each post once.

See [connector_contract.md](connector_contract.md) for the full
allowed/forbidden action list a connector implementation must follow.

## Evidence mapping

`Evidence` (from `creator_intelligence.evidence`) has 5 real fields
besides `tags`/`evidence_id` -- no metadata dict. Every
`evidence_mapper.py` function packs rich provenance into
`source_description` (a descriptive sentence: platform, section,
source URL, timestamps, metrics) and `tags` (free-form classification:
`["instagram", section, "reliability:<level>", ...]`), the same
convention `creator_intelligence.intake`'s own `to_evidence_from_*()`
functions already use. `content_excerpt` is always capped via the
shared `creator_intelligence.config.load_framework_config().caption_excerpt_max_chars`.

Relationship evidence reuses `creator_intelligence.models.RelationshipBasis`
directly (mapped from this package's own `caption_declared`/
`tagged_account`/`collaboration_label`/`visible_public_context`
vocabulary) -- never a duplicate schema.

## Rate limiting

`rate_limit.RateLimiter`: a randomized polite delay
(`minimum_delay_seconds`-`maximum_delay_seconds`) before each section's
navigation, an escalating `backoff_seconds` list on retry, and a hard
stop at `max_retries_per_navigation`. Never evasion -- on a detected
`rate_limited`/`challenge_detected` signal, the connector records the
event and stops/skips per `access_limit_behavior`, never retries past
it with a different strategy.

## Checkpoint / resume

One JSON file per `(job_id, section)` under
`output/creator_research/instagram/checkpoints/` (gitignored).
Resuming validates `job_id`/`creator_id`/`connector_version` before
reuse (`CheckpointMismatchError` on any identity mismatch -- never
silently adopted) and seeds `dedupe.Deduplicator` from the checkpoint's
`processed_source_ids` so already-collected items are never
re-emitted. Checkpointing here saves once per section, after that
section's observation completes -- not mid-scroll-loop.

## Selector maintenance

`selectors.py`'s candidate lists are **untested placeholder defaults**
(matching `src/social/instagram_models.py`'s own `CommentSelectors`
convention and caveat) -- they must be tuned against a real,
logged-in Instagram session before any live run produces meaningful
data. Nothing in the unit test suite depends on them being correct;
tests drive `FakeBrowserAdapter` with whatever selector strings they
choose to register.

## Access-limit behavior

Four signals, each with a matching exception:
`access_limited`/`AccessLimitedError`, `authentication_required`/
`AuthenticationRequiredError`, `rate_limited`/`RateLimitedError`,
`challenge_detected`/`ChallengeDetectedError`. `config.access_limit_behavior`
(`"skip"` by default) decides whether a section's failure aborts the
whole job (`"stop"`, re-raises through `orchestrator.run_job()`'s
existing `fail_fast`/`required_sections` handling) or is recorded as a
warning and the job continues to the next section.

## Diagnostics

`diagnostics.DiagnosticsRecorder` accumulates per-run structured
diagnostics (`connector_run_id`, sections attempted/completed, item/
duplicate counts, access/rate-limit events, selector failures,
timestamps) and saves atomic JSON under
`output/creator_research/instagram/diagnostics/`. No field for a
password, cookie, auth header, session token, or private message
exists on the dataclass.

## Privacy boundaries

See [connector_contract.md](connector_contract.md) and
[../creator_intelligence/ethics.md](../creator_intelligence/ethics.md)
-- both apply in full. In short: read-only, no login automation, no
access-control bypass, no bulk scraping, no credential handling, no
face recognition, no audio transcription, audience usernames redacted
by default.
