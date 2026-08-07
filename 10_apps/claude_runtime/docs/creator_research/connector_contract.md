# Connector Contract

This is the contract a future real connector (Instagram or otherwise)
must follow. Nothing in Phase 12B.1 implements it -- `connector.py`'s
`BaseConnector` only defines the shape.

## What a connector must implement

A subclass of `creator_research.connector.BaseConnector` overrides
whichever of the 10 `collect_*` methods it supports:

```python
def collect_profile(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_grid(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_posts(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_captions(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_comments(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_creator_replies(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_reels(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_highlights(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_relationships(self, job: ResearchJob) -> EvidenceBundle: ...
def collect_visual_examples(self, job: ResearchJob) -> EvidenceBundle: ...
```

Each method:
- Receives the `ResearchJob` being worked (has `creator_id`,
  `platform`, `username`, `profile_url`, `requested_sections`, etc.)
- Returns an `interfaces.EvidenceBundle`: `section`, `items:
  list[creator_intelligence.evidence.Evidence]`, `collected_at`,
  `warnings`, `metadata`.
- A method that isn't overridden raises
  `exceptions.ConnectorNotImplementedError` automatically -- a
  connector may legitimately implement only a subset of sections.
- Register the connector so `ResearchJob.connector_name` can resolve
  to it: `registry.register("my_connector", MyConnector)`.

## What a connector must NOT do

Everything in [ethics.md](../creator_intelligence/ethics.md) and
[evidence_guide.md](../creator_intelligence/evidence_guide.md) applies
to a real connector too. Specifically, a connector must never:

- **Log in** to any account, including a throwaway one, to reach
  otherwise-inaccessible content.
- **Automate a browser** (Playwright, Selenium, or any headless
  browser) to render or interact with a page.
- **Bypass an access control** (private account, follower-only
  content, rate limits) to reach content that wouldn't otherwise be
  visible to a logged-out, non-following visitor.
- **Scrape at scale** -- systematically fetch large volumes of content
  via automated HTTP requests designed to mimic or replace manual
  browsing.
- **Collect private messages, DMs, or non-public communication.**
- **Store credentials, cookies, or session tokens** anywhere this
  package or a connector persists state.
- **Construct `Evidence` with an `evidence_type` outside
  `creator_intelligence.evidence.EvidenceType`'s closed set**
  (`screenshot`, `manual_note`, `text_excerpt`, `exported_data`,
  `operator_observation`) -- there is no "scraped" or "api_fetch"
  member to construct evidence from in the first place, by design.

## What a connector may do

A connector that wraps an **operator-mediated** process is fully
within bounds -- e.g. one that reads already-collected
`creator_intelligence.intake` records from disk and reshapes them into
`EvidenceBundle`s, or one that prompts a human operator interactively
for each section and records their manual input. The boundary is not
"automated code touching data" -- it's "did a human, acting as any
member of the public could, actually observe this content."

## Registering a connector

```python
from src.creator_research.connector import BaseConnector, ConnectorRegistry
from src.creator_research.interfaces import EvidenceBundle, ConnectorSection

class MyConnector(BaseConnector):
    name = "my_connector"

    def collect_profile(self, job):
        ...
        return EvidenceBundle(section=ConnectorSection.PROFILE, items=[...])

registry = ConnectorRegistry()
registry.register("my_connector", MyConnector)

# ResearchJob(..., connector_name="my_connector")
```

`orchestrator.run_job()` never imports a concrete connector class
itself -- it only ever receives one as a parameter, or (in a future
phase) resolves one through a `ConnectorRegistry` by the name stored
on the job.
