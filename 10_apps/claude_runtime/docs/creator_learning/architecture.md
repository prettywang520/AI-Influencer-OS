# Creator Learning Engine -- Architecture (Phase 12C.0)

## What this phase is, and isn't

Phases 12A.0/12A.1 built **Creator Intelligence**: `Evidence`,
`CreatorDNA`, nine analyzers, confidence scoring, validation, and
serialization. Phases 12B.1/12B.2 built **Creator Research**: a
`ResearchJob` state machine, a queue, an orchestrator, and a
connector interface (with a read-only Instagram connector) that
produces `Evidence`. Neither phase persisted anything across runs --
run a research job twice and the system remembered nothing, and a
research job's evidence never reached the analyzers.

**Phase 12C.0 is the missing memory and synthesis layer.** It is not
a new scraper, not a new analyzer, and not a new confidence model. It
takes evidence collected by the systems that already exist, keeps it
durably per-creator, re-derives `CreatorDNA` as that store grows,
tracks how it changes over time, and renders the result as reports.

This package never opens a browser or makes a network request itself.
If a `Connector` is supplied to `CreatorLearningEngine.learn()`, it is
only ever run through `creator_research.orchestrator.run_job()` -- the
exact same gated path Phase 12B.1/12B.2 already established.
`connector=None` (the default) performs zero network activity.

## The learning-session flow

```
new evidence (Connector via run_job(), and/or an intake
evidence_bundle.json)
        |
        v
CreatorKnowledgeBase.merge_evidence()   -- dedup by Evidence.evidence_id,
        |                                  never drops/mutates an entry
        v
summary_builder.build_creator_dna()     -- the exact same, unmodified,
        |                                  pure function from 12A.0,
        |                                  called over the FULL store
        v
style_evolution.diff_creator_dna()      -- previous snapshot vs. new one
        |
        v
learning_history.append + learning_state.save   -- durable session log
        |
        v
reports/builder.py + reports/markdown.py   -- six Markdown reports,
                                               plus the CreatorDNA
                                               itself as a seventh,
                                               un-rebuilt export
```

## DNA facet -> existing `CreatorDNA` field mapping

The task's requested "12 DNA outputs" already exist as
`creator_intelligence.models.CreatorDNA`'s 13 trait fields. This phase
adds no new trait models:

| Requested facet | Backed by |
| --- | --- |
| Visual DNA | `visual_realism` + `photography` |
| Conversation DNA | `captions` + `replies` |
| Relationship Map | `relationships` + `relationship_claims` |
| Storytelling DNA | `storytelling` + `reels` |
| Posting DNA | `posting` |
| Growth DNA | `growth` |
| Brand DNA | `branding` |

The granular "learning targets" (coffee, travel, film grain, CCD
feeling, emoji/language mix, posting rhythm, ...) are **not** separate
models. `learning_targets.py` catalogs them as tags underneath the
trait they inform, and `compute_learning_target_coverage()` counts
how much evidence exists per tag -- a reporting view, not a new
analyzer, and never a gate: zero coverage of a topic is reported, not
treated as an error ("unknown remains unknown").

## Package layout

```
src/creator_learning/
    exceptions.py        LearningError hierarchy
    config.py             load_learning_config() -> config/creator_learning/engine.yaml
    creator_url.py           parse a profile URL -> (platform, username, profile_url, creator_id)
    knowledge_base.py           CreatorKnowledgeBase -- durable per-creator store
    evidence_bridge.py             merges evidence from a ResearchSession and/or an
                                    intake workspace's evidence_bundle.json into a KB
    learning_targets.py               granular tag catalog + coverage
    learning_history.py                  LearningHistoryEntry (one per session, append-only)
    style_evolution.py                      diff_creator_dna() -- pure, factual deltas only
    learning_session.py                        run_learning_session() -- the one orchestration step
    reports/
        models.py                                  six report dataclasses
        builder.py                                    build_*_report() -- pure functions
        markdown.py                                      render_*_report() -> Markdown text
    engine.py                                              CreatorLearningEngine facade
```

## Persistence layout

`output/creator_learning/<creator_id>/` (gitignored, same rule every
other real-creator data directory this project uses):

```
evidence_store.json           deduplicated union of every Evidence ever learned
learning_history.json         list of LearningHistoryEntry, one per session, append-only
learning_state.json           last_session_id, current gaps, updated_at
dna_snapshots/<session_id>.json   full CreatorDNA payload per session
reports/<session_id>/*.md     the six generated reports for that session
```

All writes are atomic (temp-file + `Path.replace()`), matching every
other persistence layer in this codebase. `creator_id` is the same
deterministic content-hash `creator_intelligence.intake.creator_id()`
already computes -- reused, not reinvented.

## Safety boundary

- No live-network capability is added by this package. A `Connector`
  passed to `learn()` runs through `creator_research`'s own
  already-gated `orchestrator.run_job()`; this package never imports
  Playwright, `requests`, or any HTTP client.
- No writes to `03_personas/` or anything under `src/social`/
  `src/publishing` -- this package doesn't import them.
- "Understand, not copy": confidence can only be `unknown` when
  evidence is absent (enforced by `creator_intelligence.confidence`,
  unmodified), relationship claims stay basis-qualified, excerpts stay
  short (validated by `creator_intelligence.validation`, unmodified),
  and the Reply Report explicitly never generates a reply.
- Structural safety tests (`test_structural_safety.py`) assert no
  network/browser imports, no credential handling, and no hardcoded
  reference to the two real target creator accounts.

See `workflow.md` for how to actually run a learning session.
