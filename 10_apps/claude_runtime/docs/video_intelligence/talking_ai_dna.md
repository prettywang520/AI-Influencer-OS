# Talking AI Intelligence -- DNA & Pattern Model

## TalkingAIProductionDNA (`production_dna.py`)

One aggregate record per analyzed talking-AI video, content-hashed
`dna_id` (hash over every field except `generated_at` and the id
itself -- re-serializing an unchanged record at a different time never
mints a new id, the same convention every *DNA record in this codebase
already follows). A **sibling** of `video_intelligence.VideoDNA`,
never an extension of it -- its own dataclass, own file, no field ever
added to `VideoDNA` itself.

10 `TalkingAIMetricResult | None` fields, one per domain analyzer
(`speech_sync.py` alone produces two: `speech_rhythm` and
`pause_behavior`):

`speech_rhythm, lip_sync, pause_behavior, blink, gaze, head_motion,
facial_motion, gesture_sync, subtitle_sync, framing`

plus:

- `naturalness: TalkingNaturalnessResult | None` -- the §15
  multidimensional aggregate over the 10 domains above (see below).
- `artifact_patterns: dict[str, int]` -- a raw tally of observed
  `ArtifactType` tags (task §16) across all supplied evidence. Never
  an interpretive "AI-generated" conclusion -- one artifact tag never
  implies AI generation on its own, and this field cannot hold one.

## Each `TalkingAIMetricResult`

`trait_name`, `metrics: dict[str, float | None]` (a metric with no
evidence to compute it is present as a key with value `None` -- never
silently dropped, never fabricated as `0.0`), `score`, `confidence`,
`sample_count`, `evidence_ids`, `warnings`, `rationale`. This is the
one envelope shape every domain analyzer returns -- the analysis math
inside each one differs (timeline alignment for `lip_sync`/
`gesture_sync`/`subtitle_sync`, interval variability for `blink`/
`gaze`/`head_motion`, run-length tracking for `facial_motion`), but the
result shape does not.

### Metrics deliberately left `None`

Some task-requested traits have no corresponding field in the approved
`TalkingAIEvidence` schema. Rather than inventing a new evidence field
mid-implementation, the metric key stays present with value `None`,
documented in-code:

| File | Metric(s) left `None` | Why |
| --- | --- | --- |
| `facial_motion.py` | `eyebrow_movement`, `cheek_movement` | no eyebrow/cheek field on `TalkingAIEvidence` |
| `subtitle_sync.py` | `highlight_alignment` | no highlight-timing field |
| `framing.py` | `shot_distance_type`, `headroom`, `face_center_stability` | no distance/face-position field -- `camera_motion` (which *is* modeled) still drives `tripod_ratio`/`handheld_ratio`/`drift_ratio`/`punch_in_ratio`/`tracking_ratio` |

## Naturalness (§15) -- `TalkingNaturalnessResult`

`dimension_scores: dict[str, float]` and `dimension_confidence: dict[str, str]`,
keyed by the naturalness-dimension spelling (`models.NATURALNESS_DIMENSION_ALIASES`),
plus an overall `score`/`confidence`/`sample_count`/`evidence_ids`/`warnings`/`rationale`.

- Overall `score` is the mean of only the dimensions that had evidence
  (`confidence != unknown`) -- a dimension with no evidence is excluded
  entirely, never averaged in as `0.0`.
- Overall `confidence` is `min_confidence()` across only the evidenced
  dimensions -- the same weakest-link rule the parent framework uses.
- `rationale` is built from a fixed vocabulary of comparative phrases
  only (`"more naturalistic on ..."` / `"more mechanically regular on
  ..."` / `"insufficient evidence on ..."`), driven by which dimensions
  scored above/below the median observed score in this result -- never
  a claim about the video's actual human/AI origin.

**There is no boolean/enum field anywhere in `TalkingNaturalnessResult`
or `TalkingAIProductionDNA` that could hold a human/AI verdict.** This
is a structural guarantee (`test_naturalness.py::test_result_type_has_no_classification_field`,
`test_structural_safety.py::NoHumanAIClassificationTests`), not just a
docstring promise -- both dataclasses use `slots=True`, so attempting
to set an undeclared attribute like `is_ai_generated` raises
`AttributeError` at runtime.

## Recommended ranges (§18) -- `derive_recommended_ranges()`

Given a batch of `TalkingAIProductionDNA` records, derives a
`RecommendedRange(metric_name, low, high, sample_count, confidence)`
per metric -- but **only** once at least `minimum_pattern_corroboration`
independent videos supplied a value for that metric. A metric supplied
by fewer videos than the corroboration floor is silently skipped, not
fabricated with a wide or low-confidence range. A single source Reel
never becomes a universal rule.

## Knowledge base -- `TalkingAIProductionPattern` (`knowledge_base.py`)

`TalkingAIKnowledgeBase.ingest_talking_ai_dna(dna, evidence=..., speech_segments=...)`
folds one video's **structured, closed-vocabulary** observations into
a running, de-identified pattern library:

- Closed-vocabulary evidence fields: `gaze_direction`, `facial_expression`,
  `gesture_type`, `camera_motion`, `mouth_shape_category` (from
  `TalkingAIEvidence`), `language` (from `SpeechSegment`), and observed
  `ArtifactType` tags.
- Two boolean-presence patterns: `"breath-like pause behavior observed"`
  (when `pause_behavior.metrics["breath_like_pause_count"] > 0`) and
  `"subtitles visible for most of the video"` (when
  `subtitle_sync.metrics["subtitle_visible_ratio"] >= 0.5`).

Never from a numeric metric's raw value and never from an analyzer's
free-text rationale -- generalized pattern descriptions are built from
**counted categories only**, so unique per-video choreography or
phrasing is never preserved verbatim (task's own explicit requirement
for gesture/Cantonese/Mandarin patterns).

### De-identification, structurally enforced

`TalkingAIProductionPattern` has exactly these fields: `pattern_id`
(content hash of `category` + `description`), `category`, `description`,
`supporting_video_count`, `confidence` (corroboration-based, same
weakest-link math), `evidence_coverage` (task's own required §19
field -- the mean, across this pattern's supporting videos, of how many
of the 10 naturalness dimensions each source video actually had
evidence for), `example_video_ids` (capped at 5, traceability only),
`last_updated`. There is no field that could carry a `creator_label` or
username -- `test_knowledge_base.py::test_no_pattern_field_carries_a_creator_label_or_username`
verifies this structurally, not just by docstring promise.

Ingestion is idempotent: re-ingesting the same `video_id` for a pattern
it already contributed to never double-counts it (tracked via an
uncapped `membership.json`, which stores `video_id -> evidence_coverage`
per pattern, separate from the capped, persisted `example_video_ids`).
