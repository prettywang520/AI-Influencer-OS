# Video Intelligence OS -- VideoDNA & Pattern Model

## VideoDNA (`production_dna.py`)

One aggregate record per analyzed video, content-hashed `dna_id`
(hash over every field except `generated_at` and the id itself --
re-serializing an unchanged record at a different time never mints a
new id, exactly like `creator_intelligence.models.CreatorDNA`).

13 `TraitScore | None` fields, one per analyzer:

`hook, pacing, speech, lipsync, gesture, camera, framing, editing,
subtitles, audio, storytelling, cta, emotion`

plus two structured lists:

- `story_beats: list[StoryBeat]` -- `beat_type` (beginning/middle/end/
  problem/solution/payoff), `description`, `evidence_ids`.
- `cta_observations: list[CTAObservation]` -- `cta_type` (follow/
  comment/share/save/question), `timing_seconds`, `evidence_ids`.

## Mapping onto the task's named DNA outputs

The task's OUTPUT section names "Video DNA, Production DNA, Hook DNA,
Speech DNA, Camera DNA, Editing DNA, Gesture DNA, Subtitle DNA, CTA
DNA, Knowledge Base." These are **views onto the one `VideoDNA`
record**, not separate classes (the same lesson Phase 12C.0 learned
mapping "Visual DNA/Conversation DNA/..." onto `CreatorDNA`'s existing
trait fields):

| Named output | Backed by |
| --- | --- |
| Video DNA / Production DNA | the `VideoDNA` record itself |
| Hook DNA | `dna.hook` |
| Speech DNA | `dna.speech` |
| Camera DNA | `dna.camera` |
| Editing DNA | `dna.editing` |
| Gesture DNA | `dna.gesture` |
| Subtitle DNA | `dna.subtitles` |
| CTA DNA | `dna.cta` + `dna.cta_observations` |
| Knowledge Base | `knowledge_base.py` (below) |

`pacing`, `framing`, `lipsync`, `audio`, `storytelling`, `emotion` are
real trait fields -- every requested analysis domain from the task's
HOOK/SPEECH/LIP SYNC/GESTURE/CAMERA/EDITING/SUBTITLES/AUDIO/EMOTION/
CTA/STORY sections is covered -- they simply weren't given their own
named "X DNA" alias in the task text. No information is dropped, just
not aliased.

### Why `camera.py` and `framing.py` are two analyzers

The task's own CAMERA section lists "Composition" alongside distance/
angle/lens-feeling/handheld/tripod/tracking/selfie/friend-shot.
`camera.py` covers movement/equipment feel; `framing.py` covers
composition/subject placement specifically (rule of thirds, headroom,
symmetry) so it can be scored independently of camera movement. A
deliberate split, not an accidental duplicate.

### Why `pacing.py` and `editing.py` are two analyzers

`pacing.py` measures overall rhythm/tempo as a "meta" trait;
`editing.py` catalogs the concrete cut/zoom/transition/jump-cut
techniques that produce a given tempo. Complementary, not duplicated.

## Knowledge base -- `ProductionPattern` (`knowledge_base.py`)

`VideoKnowledgeBase.ingest_video_dna(dna, evidence)` folds one video's
**structured** observations into a running, de-identified pattern
library:

- From `story_beats`: one candidate pattern per `beat_type` present
  (`"story arc includes a 'problem' beat"`).
- From `cta_observations`: one candidate per `cta_type` present
  (`"uses a 'comment' call-to-action"`).
- From the 5 yaml-configured analyzers' own evidence tags (re-derived
  safely via `analyzer.observed_vocabulary()`, never by parsing
  free-text rationale): hook type, pacing tempo/rhythm, speech
  language/speed/energy, camera distance/equipment, subtitle style.

The remaining 6 analyzers (lipsync, gesture, framing, editing, audio,
emotion) have no yaml-driven controlled vocabulary in this phase, so
they contribute to a video's `VideoDNA` trait scores but not yet to
knowledge-base pattern extraction -- a deliberate v1 scope boundary
(safer to extract less than to risk deriving a "pattern" from
unstructured text), not an oversight. Adding a controlled vocabulary
yaml for any of them is a natural, additive future extension.

### De-identification, structurally enforced

`ProductionPattern` has exactly these fields: `pattern_id` (content
hash of `category` + `description`), `category`, `description` (a
short, controlled-vocabulary-derived generalization -- **never** a
video's verbatim excerpt), `supporting_video_count`, `confidence`
(derived from corroboration count via the same weakest-link math every
analyzer uses -- more independent supporting videos raises confidence,
never a single video pushing it past `low`), `example_video_ids`
(capped at 5, traceability only). There is no field that could carry a
`creator_label` or an unbounded excerpt -- "never copy creators" is a
data-model guarantee (`test_knowledge.py::test_no_creator_label_field_exists`),
not just a docstring promise.

Ingestion is idempotent: re-ingesting the same `video_id` for a
pattern it already contributed to never double-counts it (tracked via
an uncapped `membership.json`, separate from the capped, persisted
`example_video_ids`).
