# Video Intelligence OS -- Architecture (Phase 12D.0)

## What this is, and isn't

`video_intelligence` is a platform-neutral analysis package scoped
around **a video**, not a creator. Its job is not to produce or
render video -- it never touches the existing Timeline/Subtitle/
Overlay/Renderer engines, and it never opens a browser, calls an
external API, or runs Whisper/ffprobe/frame-decoding of any kind.
Every `VideoEvidence` item is operator-supplied or synthetic
(tests/demo), exactly like `creator_intelligence.evidence.Evidence`'s
own boundary. The goal is to explain **why a video works** --
hook, pacing, speech, lip sync, gesture, camera, framing, editing,
subtitles, audio, storytelling, CTA, emotion -- and to fold many
videos' analyses into a de-identified, reusable pattern library.

This package is fully decoupled from `creator_intelligence`/
`creator_research`/`creator_learning` and from the production video
pipeline (`timeline_engine.py`, `subtitle_engine.py`/
`subtitle_render_engine.py`, `overlay_plan_engine.py`/
`overlay_render_engine.py`, `renderer_plan_engine.py`/
`renderer_execution_engine.py`) -- nothing is imported in either
direction. Every one of those production modules already declares
itself "platform-neutral" and enumerates a "never renders / never
calls ffmpeg / never inspects real media" boundary in its own
docstring; `video_intelligence` is the mirror image: it only
*observes and explains* videos that already exist (typically someone
else's), never produces one.

## Pipeline

```
evidence.py          -- VideoRecord (one video's identity) + VideoEvidence
                         (one observation about it)
models.py             -- shared trait/confidence/story/CTA vocabulary
analyzer.py             -- AnalyzerContext + evidence_for_tags()/
                            analyze_by_tag()/observed_vocabulary() --
                            the base every analyzer shares
13 analyzer modules       -- hook, pacing, speech, lipsync, gesture,
                            camera, framing, editing, subtitles, audio,
                            storytelling, cta, emotion
production_dna.py          -- build_video_dna(): runs all 13, aggregates
                              into one VideoDNA (content-hashed dna_id)
report.py / serialization.py -- human-readable + durable JSON output
knowledge_base.py             -- cross-video, de-identified pattern store
workflow.py / cli.py            -- analyze_video() facade + its CLI
```

Every analyzer follows the exact same shape (fresh reimplementation of
`creator_intelligence`'s analyzer pattern, not imported): a
`TRAIT_NAME` constant, an `analyze(context) -> AnalyzerResult` method
built on the shared `analyze_by_tag()` helper, and confidence that can
only ever be `unknown` when no scoped evidence exists -- "unknown
remains unknown" carried over verbatim.

Five analyzers (hook, pacing, speech, camera, subtitles) additionally
read a yaml-configured controlled vocabulary
(`config/video_intelligence/{hooks,pacing,speech,camera,subtitles}.yaml`)
and surface which recognized terms were observed through
`TraitScore.rationale` via `analyzer.observed_vocabulary()` -- never by
inventing a bespoke typed field per analyzer, and never by parsing
free-text excerpts.

Two analyzers (storytelling, cta) additionally extract a second,
structured output beyond their `TraitScore` -- `story_beats: list[StoryBeat]`
and `cta_observations: list[CTAObservation]` -- mirroring
`creator_intelligence.human_authenticity_analyzer`'s
`extract_relationship_claims()` dual-output shape.

## A collision this design had to guard against

`CTAType.ALL` (`follow`/`comment`/`share`/`save`/`question`) and
`BeatType.ALL` (`beginning`/`middle`/`end`/`problem`/`solution`/`payoff`)
are common English words that can legitimately appear as tags for
*other* reasons -- e.g. `hooks.yaml`'s own `question` hook_type value.
Early in implementation, `cta.py`/`storytelling.py` scoped their
`analyze_by_tag()` calls using `("cta",) + CTAType.ALL` /
`("storytelling",) + BeatType.ALL` directly, which meant a hook
evidence item tagged `["hook", "question"]` was incorrectly swept into
CTA analysis too (caught by `test_cta.py`'s
`test_domain_tag_required_no_leakage_from_unrelated_question_tag`).
**Fixed**, and now a standing rule for this package: a domain analyzer's
top-level scoping `TAGS` must never include a shared/generic
sub-vocabulary value directly -- evidence must carry the analyzer's
own domain tag (`"cta"`, `"storytelling"`) to be considered at all;
type-specific vocabulary values are only matched *within* that
already-scoped subset.

## Knowledge base -- de-identified, cross-video patterns

This is the one place this package's design diverges from every prior
phase's per-creator knowledge base. `VideoKnowledgeBase` ingests many
`VideoDNA` records (from many videos, possibly many creators) and
extracts `ProductionPattern` records -- short, generalized
descriptions plus aggregate corroboration statistics, never a specific
video's verbatim content. See `dna.md` for the full design.

## Safety boundary

- No live-network capability anywhere in this package -- no browser,
  no scraping, no Whisper, no OpenAI/Gemini/external API. Every
  `VideoEvidence` item is operator-supplied or synthetic.
- Structurally decoupled from `creator_intelligence`/`creator_research`/
  `creator_learning` and from the production rendering pipeline --
  zero imports in either direction.
- `ProductionPattern` structurally cannot carry a `creator_label` or
  unbounded excerpt (see `dna.md`) -- "never copy creators" is enforced
  by the data model, not just a docstring promise.
- `output/video_intelligence/` is gitignored, same rule every other
  real-data directory in this project follows.

See `workflow.md` for how to actually run an analysis, and `dna.md`
for the full VideoDNA/pattern data model.
