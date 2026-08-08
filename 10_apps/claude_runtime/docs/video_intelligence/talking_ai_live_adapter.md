# Talking AI Live Learning Adapter -- Architecture (Phase 12D.3)

## What this is, and isn't

`src/video_intelligence/talking_ai/live/` is a thin **bridge and
orchestration layer**. It normalizes Instagram Reel *research*
evidence -- from `creator_research/instagram`'s connector, or an
operator who watched a public Reel by hand -- into `talking_ai`'s
existing `TalkingAIEvidence`/`SpeechSegment` schema and drives it
through the existing, unmodified
`talking_ai.workflow.TalkingAIWorkflow.analyze_talking_video()`. It
builds **no new analyzer** (the 9 domain analyzers, `naturalness.py`,
and `production_dna.py` from Phase 12D.2 are reused completely
unmodified) and performs **no** Instagram collection of its own --
live acquisition remains owned by `creator_research/instagram`.
Nothing here opens a browser, calls an external API, transcribes
audio, clones a voice, or performs facial recognition.

Target flow:

```
Instagram Reel research evidence -> Live Talking Reel Adapter
    -> TalkingAIEvidence/SpeechSegment -> existing Talking AI workflow
    -> TalkingAIProductionDNA -> Talking AI Knowledge Base
```

## The central integration findings

- **`TalkingAIEvidence`/`SpeechSegment` already are the target schema.**
  Every mouth/jaw/blink/gaze/head/facial/gesture/subtitle/camera/
  artifact/language/pause/emphasis/Cantonese-metadata field this task
  asks to bridge already exists on those Phase 12D.2 types. This
  package invents no new per-observation field -- the bridging problem
  is almost entirely about **identity** (stable Reel/record IDs),
  **coarse timing** (never fake frame precision), and **batch
  orchestration** (sampling/validation/completeness/reporting), not new
  evidence fields.
- **`creator_intelligence.evidence.Evidence`'s 5-field ceiling,
  confirmed again.** `creator_research/instagram/evidence_mapper.py`'s
  `reel_to_evidence()` flattens a rich `ReelRecord` (which *does* have
  `text_overlays: list[str]`) into an `Evidence` with only
  `source_description`/`content_excerpt`/`tags`/`evidence_type`/
  `collected_by` -- `text_overlays` is silently dropped, exactly the
  gap Phase 12D.1's own docs already flagged for Video Intelligence.
  `connector.collect_reels()` only ever returns the already-flattened
  `Evidence`. `observer_bridge.py` closes this gap: it imports only
  `creator_research.instagram.models.ReelRecord` (a stable, public,
  dependency-free dataclass -- never `.connector`/`.observer`/
  `.navigator`, so this file can never transitively import Playwright)
  and recovers `text_overlays` into `subtitle_observations` *before*
  that lossy flattening would otherwise happen.
- **`timestamp_seconds: float` is required, not optional, on
  `TalkingAIEvidence`.** Making it optional would ripple into all 9
  existing domain analyzers, which is not a narrow, additive change,
  and `talking_ai/evidence.py` is not in this phase's MODIFY
  allow-list. `timing_bridge.py` resolves this without touching
  `evidence.py` at all: purely-ordinal evidence ("first mouth motion,
  then blink, then gesture" -- no seconds at all) gets monotonically
  increasing *placeholder* seconds, capped at `confidence="low"`, with
  an explicit note (`"ordinal placeholder timing (not observed
  seconds)"`) and a batch-level warning -- the *order* is always real,
  the *seconds* are only ever real when the caller actually supplied
  real seconds.
- **Confirmed zero modification needed** to `talking_ai/evidence.py`,
  `talking_ai/workflow.py`, or `video_intelligence/instagram/models.py`
  (the three files this phase's task allows as a last resort).
  `talking_ai.workflow.TalkingAIWorkflow.analyze_talking_video()`
  already does build-DNA -> save -> report -> knowledge-base-ingest in
  one call -- `live/workflow.py` composes it, never re-implements it,
  exactly like Phase 12D.1 found for `VideoIntelligenceWorkflow.analyze_video()`.

## Pipeline

```
models.py            LiveTalkingReelRecord (one Reel's live-research
                      evidence collection), LiveTalkingLearningRecord
                      (one Reel's outcome), LiveBatchLearningReport,
                      LiveAdapterConfig, compute_live_completeness()
timing_bridge.py        coarse-timing bridge: exact/approximate/
                         ordinal-placeholder timestamp handling --
                         never fakes frame precision, never touches
                         evidence.py
observer_bridge.py        ReelRecord (creator_research.instagram.models,
                           stable type only) -> LiveTalkingReelRecord
                           identity + subtitle_observations (recovers
                           text_overlays)
mapper.py                    Evidence (creator_research, already-
                              flattened) -> LiveTalkingReelRecord
                              identity-only, via the same
                              "<kind> | key=value | ..." parsing
                              convention Phase 12D.1's own mapper.py
                              established
adapter.py                      LiveTalkingReelRecord ->
                                 (TalkingAIEvidence, SpeechSegment) --
                                 video_id rewriting + timing-bridge
                                 marker aggregation, never
                                 reinterpreting content
sampler.py                         deterministic bucketed sampling
                                    over LiveTalkingReelRecord
validator.py                          structural validation
diagnostics.py                          local batch-run diagnostics
                                         (no live session exists yet
                                         to record)
report.py                                 talking_ai_live_learning_report
                                           .json/.md
workflow.py                                 learn_live_talking_reel()/
                                             learn_live_talking_reels()
                                             + CLI (offline/local
                                             evidence mode only)
```

## Three ways a `LiveTalkingReelRecord` gets built

From richest/rarest to coarsest/most-available today:

1. **Hand-built / operator-authored (primary path today).** An
   operator who watched a public Reel constructs `TalkingAIEvidence`/
   `SpeechSegment` objects directly and assembles a
   `LiveTalkingReelRecord` from them, or via the CLI's JSON input. No
   browser, no connector -- available immediately.
2. **`observer_bridge.py` (richer, still fully offline).** Given one or
   more `ReelRecord` objects (built by hand today; produced by a
   future direct `observer.observe_reels()` call later -- this file
   doesn't care which), recovers identity plus `text_overlays` ->
   `subtitle_observations`.
3. **`mapper.py` (from already-collected, already-flattened
   `Evidence`).** Given `creator_research/instagram`'s actual connector
   output, recovers identity/metadata only (`text_overlays` is
   unavailable via this path, same honestly-documented gap as 12D.1).

All three converge on the same `LiveTalkingReelRecord` shape;
`adapter.py` does the one remaining step every path shares.

## What `adapter.py` deliberately does NOT do

`camera_observations`/`subtitle_observations`/`artifact_observations`
(the record's coarse, reel-level context lists) are **not**
auto-merged into individual `TalkingAIEvidence` items -- doing so would
attribute a whole-Reel characterization ("this Reel is handheld") to a
specific timestamp that never actually reported it, exactly the
fabrication "unknown remains unknown" forbids throughout this
codebase. They flow into `sampler.py`/`compute_live_completeness()`/
`report.py` only. This is a deliberate v1 scope boundary, not an
oversight.

## Sampling, validation, completeness, de-identification

`sampler.py` mirrors `video_intelligence/instagram/sampler.py`'s
`sample_reels()` shape exactly: pure, deterministic (content-based
ordering + fixed bucket-iteration round-robin). 14 buckets (task's own
§15 list); buckets needing data `LiveTalkingReelRecord`'s own required
fields don't carry (visible engagement, walking/indoor/outdoor
context) read an optional key from `record.metadata`/
`record.camera_observations` -- a record missing that signal simply
never joins that bucket, never fabricated.

`validator.py` treats a fully sparse record (only `source_url` known)
as valid, erroring only on genuinely invalid data (malformed URL,
negative/NaN/infinite timing, duplicate Reel, non-JSON metadata,
creator-label mismatch within a batch).

`compute_live_completeness()` reports coverage across 11 domains
(speech, mouth, blink, gaze, head, face, gesture, subtitle, camera,
artifact, timing) -- 0.0/1.0 per domain, never a quality/naturalness
judgement.

`LiveTalkingReelRecord`/`LiveTalkingLearningRecord` are allowed to
carry `source_url`/`source_evidence_ids`/`reel_id` for **local
traceability only** (task's own §19 exception, matching
`InstagramReelLearningRecord.source_url`'s existing precedent).
`TalkingAIProductionDNA`/`TalkingAIKnowledgeBase` are completely
unmodified from 12D.2 -- both already have no field that could hold a
username/creator ID/profile URL/verbatim script -- so nothing new is
needed there; `adapter.py` never copies a URL or identity string into
`TalkingAIEvidence.notes`.

## Safety boundary

- No live-network capability anywhere in this package -- no browser,
  no scraping, no Whisper/OpenAI/voice-cloning/face-recognition API.
  Every `TalkingAIEvidence`/`SpeechSegment`/`LiveTalkingReelRecord` is
  operator-supplied or synthetic.
- Structurally decoupled from `creator_intelligence`/`creator_learning`,
  from `creator_research.instagram`'s connector/observer/navigator
  (only the stable `ReelRecord` dataclass is imported), and from the
  production rendering pipeline.
- `output/video_intelligence/talking_ai/live/` is gitignored, same
  rule every other real-data directory in this project follows.
- `test_structural_safety.py` asserts `howdoyoulikedurian` (and every
  other real creator/persona name used in this codebase) is absent
  from source and config, and that `workflow.py`'s CLI has no live/
  browser/login flag of any kind.

See `talking_ai_live_runbook.md` for how a *future*, separately
authorized live run would work, and `talking_ai_workflow.md`/
`talking_ai_dna.md` for the unmodified downstream Talking AI pipeline
this adapter feeds.
