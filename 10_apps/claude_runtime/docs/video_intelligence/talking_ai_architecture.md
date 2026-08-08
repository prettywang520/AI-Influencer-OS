# Talking AI Intelligence -- Architecture (Phase 12D.2)

## What this is, and isn't

`video_intelligence.talking_ai` is a platform-neutral analysis package
scoped around one question: **why does a talking-AI video feel natural
or unnatural?** It does **not** generate video, does **not** perform
lip-sync generation, does **not** synthesize speech, and does **not**
clone a creator's voice. It analyzes operator-supplied (or, in a
future phase, richer observed) evidence and produces a
`TalkingAIProductionDNA` -- exactly the same "observe and explain,
never produce" boundary the parent `video_intelligence` package
already draws for ordinary video traits.

It is a **sibling** of `video_intelligence`'s own `VideoDNA`/knowledge
base, never an extension of it -- `TalkingAIProductionDNA` is its own
top-level record with its own `dna_id`, in its own file, and no field
was added to `VideoDNA`, `production_dna.TRAIT_FIELDS`, or any of the
parent package's four "last-resort" files. It is fully decoupled from
`creator_intelligence`/`creator_research`/`creator_learning`, from
`video_intelligence.instagram` (the Instagram Reels Learning Adapter --
except for one optional, read-only bridge function, see below), from
rendering/publishing/production/social, and from every persona
(Aiko/WinWin/Yua).

## Why a new evidence model was needed

The parent package's 13 analyzers (`hook.py`, `speech.py`, etc.) all
share one shape: `analyze_by_tag()` scores evidence by tag *count* --
there is no notion of time, ordering, or co-occurrence in that math.
Judging whether a talking-AI video looks natural requires exactly
that: is the mouth moving *at the same time* as speech; does gaze
return to camera *after* a pause; is head motion frozen for
suspiciously long. `TalkingAIEvidence` is therefore a **simultaneous
snapshot** at a timestamp (mouth shape, blink, gaze, head yaw/pitch/
roll, gesture, subtitle state, all co-occurring), fundamentally
different from `VideoEvidence`'s single tagged excerpt. `talking_ai/analyzer.py`
is a new, separate base with genuinely different utilities
(`intervals()`, `variability()`, `nearest_event_after()`/
`nearest_event_near()`) -- not a copy of the parent's own `analyzer.py`.

What **is** reused directly (imported, never redefined): `ConfidenceLevel`,
`ConfidenceThresholds`, `compute_confidence_level()`, `min_confidence()`
(`video_intelligence.models`) for the exact same "evidence quantity +
corroboration, zero evidence is always `unknown`" confidence
philosophy every package in this codebase already uses.

## Pipeline

```
evidence.py             -- TalkingAIEvidence (one timestamped snapshot)
                            + SpeechSegment (a coarser speech-level fact)
models.py                -- TalkingAIConfig, TalkingAIMetricResult,
                             TalkingNaturalnessResult, RecommendedRange,
                             TRAIT_FIELDS + the naturalness-dimension alias map
timing.py + analyzer.py    -- shared time-base + timeline statistics every
                               domain analyzer reuses
9 domain analyzer files       -- speech_sync (2 fields), lip_sync, blink,
                                  gaze, head_motion, facial_motion,
                                  gesture_sync, subtitle_sync, framing
naturalness.py                   -- aggregates the 10 domain results into
                                     one TalkingNaturalnessResult
production_dna.py                  -- build_talking_ai_dna(): runs all 10
                                       domain analyzers + naturalness,
                                       aggregates into one
                                       TalkingAIProductionDNA
                                       (content-hashed dna_id) +
                                       derive_recommended_ranges()
report.py / serialization.py         -- human-readable + durable JSON output
knowledge_base.py                      -- cross-video, de-identified
                                           pattern store
workflow.py                              -- analyze_talking_video()/
                                             analyze_talking_videos()
                                             facade + CLI + the optional
                                             InstagramReelLearningRecord bridge
```

## Naming reconciliation

The task's own vocabulary uses three slightly different spellings for
the same 10 domains: DNA field names (`blink`, `gaze`, `head_motion`,
`facial_motion`, `gesture_sync`, `framing`, ...), naturalness-dimension
names (`blink_variability`, `gaze_variability`, `head_micro_motion`,
`facial_micro_motion`, `gesture_speech_sync`, `camera_naturalness`,
...), and file names. `models.TRAIT_FIELDS` owns the canonical (DNA
field) spelling; `models.NATURALNESS_DIMENSION_ALIASES` maps each to
its naturalness-dimension spelling, defined once, used everywhere --
the same reconciliation Phase 12D.1 already had to make for its own
"lip_sync" vs "lipsync" priorities config.

## Honesty disciplines enforced structurally, not just by convention

- **Coarse-timing lip sync only**: `TalkingAIEvidence`/`SpeechSegment`
  define no phoneme-level field at all, so `lip_sync.py`'s rationale is
  unconditionally prefixed `"coarse-timing lip sync (frame/segment-level
  evidence only)"` whenever it produces a result -- no code path could
  ever claim phoneme-level validation.
- **"Breath-like pause behavior," never "breathing detected"**:
  `speech_sync.py`'s pause-behavior rationale always uses "breath-like
  pause behavior" language and explicitly states actual breathing is
  never inferred.
- **No human/AI classifier, anywhere**: there is no boolean/enum field
  in `TalkingNaturalnessResult` or `TalkingAIProductionDNA` that could
  hold a human/AI verdict. `naturalness.py`'s rationale is built from a
  fixed vocabulary of comparative phrases only ("more naturalistic" /
  "more mechanically regular" / "insufficient evidence").
- **No automatic AI-generation conclusion from one artifact**:
  `TalkingAIProductionDNA.artifact_patterns` is a raw tally of observed
  `ArtifactType` tags -- never an interpretive verdict.
- **"Not modeled in this schema version" over fabrication**: where a
  task-requested trait (eyebrow/cheek motion in `facial_motion.py`,
  subtitle highlight-timing in `subtitle_sync.py`, shot-distance/
  headroom/face-center stability in `framing.py`) has no corresponding
  field in the approved evidence schema, the metric key is present but
  its value stays structurally `None`, documented in-code -- never
  silently omitted, never invented.
- **Never fabricating an "absent" comparison as a "miss"**: every
  alignment/coverage ratio (`lip_sync.py`, `gesture_sync.py`,
  `subtitle_sync.py`, `speech_sync.py`'s pause-behavior half) only
  counts a reference window in its denominator when there is *some*
  evidence to compare against.

## Optional bridge from the Instagram Reels Learning Adapter

`workflow.talking_evidence_from_reel_record()` is a lossy, best-effort,
read-only bridge: it converts an already-analyzed
`InstagramReelLearningRecord`'s `VideoEvidence` tags (e.g.
`"blink_timing"`, `"eye_contact"`, `"handheld"`) into sparse
`TalkingAIEvidence`, populating only the fields a tag can honestly
imply. It reads `video_intelligence.instagram`'s public record type
only -- it does not modify that package, and no field it cannot infer
is ever fabricated. This is explicitly the extension point a future
Phase 12D.3 ("Talking AI Live Learning Adapter") would build on.

## Safety boundary

- No live-network capability anywhere in this package -- no browser,
  no scraping, no Whisper/OpenAI/voice-cloning/face-recognition API.
  Every `TalkingAIEvidence`/`SpeechSegment` item is operator-supplied
  or synthetic.
- Structurally decoupled from `creator_intelligence`/`creator_research`/
  `creator_learning`, from `video_intelligence.instagram`'s collection
  behavior, and from the production rendering pipeline.
- `TalkingAIProductionPattern` structurally cannot carry a
  `creator_label` or username -- de-identification is enforced by the
  data model, not just a docstring promise (see `talking_ai_dna.md`).
- `output/video_intelligence/talking_ai/` is gitignored, same rule
  every other real-data directory in this project follows.
- `test_structural_safety.py` asserts `howdoyoulikedurian` (and every
  other real creator/persona name used in this codebase) is absent
  from source and config.

See `talking_ai_workflow.md` for how to actually run an analysis, and
`talking_ai_dna.md` for the full `TalkingAIProductionDNA`/pattern data
model.
