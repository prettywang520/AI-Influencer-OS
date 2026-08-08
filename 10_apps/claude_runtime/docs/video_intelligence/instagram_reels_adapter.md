# Instagram Reels Learning Adapter -- Architecture (Phase 12D.1)

## What this is, and isn't

`src/video_intelligence/instagram/` is a thin **adapter and
orchestration layer**. It normalizes Instagram Reel evidence into
`video_intelligence`'s existing `VideoEvidence` schema and drives it
through the existing, unmodified
`video_intelligence.workflow.VideoIntelligenceWorkflow.analyze_video()`
for many Reels at once. It builds **no new analyzer**, duplicates
**no** Video Intelligence analysis logic, and performs **no**
Instagram collection of its own — live acquisition remains owned by
`creator_research/instagram`. Nothing here opens a browser, calls an
external API, or transcribes audio.

Target flow:

```
Instagram Reel -> Instagram Research Connector -> Reel Evidence
    -> Instagram Reels Learning Adapter -> VideoEvidence
    -> 13 Video Intelligence analyzers -> VideoDNA
    -> Production DNA -> Video Knowledge Base
```

## The central integration finding

`InstagramResearchConnector.collect_reels()`
(`src/creator_research/instagram/connector.py`) returns
`EvidenceBundle.items: list[creator_intelligence.evidence.Evidence]`,
produced by `evidence_mapper.reel_to_evidence()`
(`src/creator_research/instagram/evidence_mapper.py`). `Evidence` has
exactly 5 real fields (no metadata dict, by long-standing design) —
every Reel's rich fields get flattened into one pipe-delimited
`source_description` string:

```
instagram reel | url=<url> | published_at=<ts> | views=<n> | likes=<n>
| comments=<n> | duration_seconds=<n>
```

with `content_excerpt` = the reel's caption, and
`tags=["instagram", "reels", "reliability:medium"]` — **no
domain-specific tags at all**. `reel.text_overlays` is **silently
dropped** by `reel_to_evidence()` — it is not recoverable from this
path. Every other `evidence_mapper.py` function uses the exact same
`"<kind> | key=value | ..."` convention and links back to a post/reel
via a `post=`/`url=` key.

This shapes the whole design:

1. **`mapper.reel_evidence_from_creator_research()` parses
   `source_description`** to recover structured fields
   (`_parse_source_description()`) — a fixed, machine-generated
   convention, not fragile guesswork, but a real integration gap
   (`text_overlays` is honestly unavailable via this path — recorded
   as a warning on every `InstagramReelLearningRecord`, never
   fabricated).
2. **A bare connector-sourced Reel `Evidence` object carries no
   domain tags** any `video_intelligence` analyzer scopes by. The
   adapter therefore accepts richer input than just that: its
   `InstagramReelEvidencePacket` (`models.py`) has an `annotations`
   list of already-domain-tagged `VideoEvidence` items — from an
   operator, or a future Phase 12D.2 annotator — which the adapter
   passes through **unchanged**, never classifying or reinterpreting
   caption/annotation text itself. `packet.annotations` items are only
   folded in from linked connector Evidence if they already carry a
   recognized domain tag (today's connector never sets one — this is
   forward-compatible, not dead code).

## Package layout

```
exceptions.py    AdapterError hierarchy
models.py          InstagramReelsConfig, InstagramReelEvidencePacket,
                    InstagramReelLearningRecord, BatchLearningReport,
                    compute_completeness(), TRAIT_TAGS
mapper.py            reel_evidence_from_creator_research() (Evidence ->
                      packets) + packet_to_video_evidence() (packet ->
                      VideoEvidence) -- the only file that imports
                      creator_intelligence.evidence.Evidence (a stable,
                      public type; never creator_research.instagram)
sampler.py             sample_reels() -- deterministic bucketed sampling
validator.py             validate_packet()/validate_packets() -- structural
                          checks only
report.py                  build_batch_report()/render_batch_report_markdown()
workflow.py                  learn_reel()/learn_reels() + CLI -- composes
                              the above with the existing, unmodified
                              VideoIntelligenceWorkflow.analyze_video()
```

**Deviation from the task's suggested file list**: no separate
`adapter.py`. `video_intelligence.workflow.VideoIntelligenceWorkflow`
already *is* the orchestration facade this package drives; a second
`adapter.py` would either duplicate `instagram/workflow.py`'s own
batch-facade role or become an empty pass-through.
`instagram/workflow.py` plays both roles: `learn_reel()`/`learn_reels()`
(the "adapter" entry points) plus the CLI.

## Zero modification to `video_intelligence` or `creator_research`

Confirmed unnecessary: `VideoPlatform.INSTAGRAM_REELS` already existed
(built in Phase 12D.0 precisely for this), and
`VideoIntelligenceWorkflow.analyze_video(record, evidence, *,
ingest_into_knowledge_base=True)` already does build-DNA → save →
report → knowledge-base-ingest in one call. `creator_research/instagram`
is untouched — `mapper.py` imports only
`creator_intelligence.evidence.Evidence` (a stable, public type), never
`creator_research.instagram` itself, so nothing in this package can
transitively import Playwright or a browser session (enforced by
`test_structural_safety.py`).

## A collision this design had to guard against (mapping-level)

Same class of bug found and fixed in Phase 12D.0 itself: `annotation`
tags are pass-through only, so the adapter relies on the caller having
already applied a real domain tag (`"hook"`, `"cta"`, ...) rather than
inferring one — this sidesteps the "does the adapter interpret
content" trap entirely, since the adapter never guesses which domain a
piece of text belongs to.

## Sampling, validation, completeness, de-identification

See `instagram_reels_learning_workflow.md` for practical usage;
`test_sampler.py`/`test_validator.py`/`test_knowledge.py`-equivalent
coverage in `test_workflow.py`'s `DeIdentificationTests` proves:
- `sample_reels()` is deterministic regardless of input order (content-
  based ordering + a fixed bucket-iteration order, never a seeded PRNG).
- `validate_packet()`/`validate_packets()` fail clearly on invalid
  data (bad URL, negative metrics, malformed timestamps, non-JSON
  metadata, duplicate Reels) and allow unknown data unconditionally.
- `compute_completeness()` reports coverage only (0.0/1.0 per trait,
  never a quality judgement), reusing each analyzer's own `TAGS`
  constant directly rather than a second vocabulary.
- No `ProductionPattern` the knowledge base produces from adapter-fed
  `VideoDNA` ever carries a `creator_label`, a real username, or a
  verbatim caption/script — enforced structurally by
  `knowledge_base.ProductionPattern`'s own field set (Phase 12D.0)
  and re-verified end-to-end here.

## Future Phase 12D.2 (Talking AI) handoff

`packet.annotations` is the explicit extension point: a future,
separate talking-AI analyzer (speech timing, lip sync, facial
micro-motion, blink, eye contact, head movement, gesture-speech
synchronization) would produce richer, already-domain-tagged
`VideoEvidence`-shaped annotations and attach them via this exact same
list — zero adapter change required, only a new evidence-producing
front door. This phase does not analyze `howdoyoulikedurian` (named
only as 12D.2's future production-technique study target) and does
not build WinWin or Yua persona data.
