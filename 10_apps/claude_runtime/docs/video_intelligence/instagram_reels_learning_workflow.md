# Instagram Reels Learning Workflow

See `instagram_reels_adapter.md` for the design. This is the practical
"how do I learn from a batch of Reels" guide.

## 1. From a real creator_research job

```python
from src.video_intelligence.instagram.mapper import reel_evidence_from_creator_research
from src.video_intelligence.instagram.workflow import learn_reels, load_instagram_reels_config

# `evidence` is whatever a real (or fake, in tests) InstagramResearchConnector
# job already produced -- e.g. session.evidence_queue.items_for_section("reels")
# plus any linked captions/creator_replies/relationships.
packets = reel_evidence_from_creator_research(evidence, creator_label="reference_creator")

config = load_instagram_reels_config()
report, records = learn_reels(packets, config=config)

print(report.reels_analyzed, report.video_dna_ids)
```

`text_overlays` will not be present via this path (see
`instagram_reels_adapter.md`'s integration finding) -- everything else
structural (url, timing, views/likes/comments, caption) is.

## 2. Hand-building richer packets (operator or future annotator)

```python
from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType
from src.video_intelligence.instagram.models import InstagramReelEvidencePacket

packet = InstagramReelEvidencePacket(
    reel_url="https://www.instagram.com/reel/abc123/",
    creator_label="reference_creator",
    duration_seconds=14.5, views=15000, likes=1200, comments=80,
    caption="a coffee shop morning routine, come with me!",
    annotations=[
        VideoEvidence(
            video_id="",  # rewritten automatically to the packet's own reel_id
            evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
            source_description="opening line",
            content_excerpt="opens with a direct question",
            timestamp_seconds=0.5,
            tags=["hook", "question"],
        ),
    ],
)
```

Any of the 13 analysis domains (`hook`, `pacing`, `speech`, `lipsync`,
`gesture`, `camera`, `framing`, `editing`, `subtitles`, `audio`,
`storytelling`, `cta`, `emotion`) can be supplied this way -- the
adapter never classifies or reinterprets the content, it only carries
forward what's already tagged.

## 3. Running a batch

```python
from src.video_intelligence.instagram.workflow import learn_reels

report, records = learn_reels(packets, config=config)  # sample=True by default
```

`sample_reels()` runs first (deterministic, bucketed -- see
`instagram_reels_adapter.md`'s §16 design), then each sampled packet
is validated and run through
`video_intelligence.workflow.VideoIntelligenceWorkflow.analyze_video()`
(unmodified) -- build DNA, save, report, and fold structured patterns
into the shared knowledge base, per the same package's own established
behavior.

Pass `sample=False` to learn from every valid packet without
sampling (useful for a deliberately curated, already-small batch).

## 4. CLI (local evidence mode only)

```
python3 -m src.video_intelligence.instagram.workflow \
    --evidence path/to/reels_evidence.json \
    --output output/instagram_reels_batch/

python3 -m src.video_intelligence.instagram.workflow \
    --evidence path/to/reels_evidence.json --output ... --validate-only

python3 -m src.video_intelligence.instagram.workflow \
    --evidence path/to/reels_evidence.json --output ... --force --json
```

`reels_evidence.json` is a JSON list of packet-shaped dicts (`reel_url`
required; `published_at`/`duration_seconds`/`views`/`likes`/`comments`/
`caption`/`creator_label`/`annotations` all optional). No subcommand
can construct a browser, call an external API, or transcribe audio --
every input is a local JSON file. Writes
`instagram_reels_learning_report.json` and `.md` to `--output`.

## 5. Reading back what's been learned

Every `learn_reel()`/`learn_reels()` call runs through the unmodified
`VideoIntelligenceWorkflow`, so the same read APIs from Phase 12D.0
apply directly:

```python
from src.video_intelligence.workflow import VideoIntelligenceWorkflow

workflow = VideoIntelligenceWorkflow()
workflow.load_dna(reel_id)                      # VideoDNA | None
workflow.load_report_markdown(reel_id)            # str | None
workflow.knowledge_base().list_patterns()           # de-identified, cross-Reel patterns
workflow.knowledge_base().list_patterns("hook")      # scoped to one category
```

## 6. What gets written to disk

Unchanged from Phase 12D.0 (this adapter writes nothing new outside
that layout):

```
output/video_intelligence/<reel_id>/
    video_dna.json
    report.md
output/video_intelligence/knowledge_base/
    patterns.json       de-identified ProductionPattern records
    membership.json
```

`output/` is gitignored; nothing here is ever committed by this
package.
