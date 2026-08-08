# Video Intelligence OS -- Workflow

See `architecture.md` for the design and `dna.md` for the data model.
This is the practical "how do I run an analysis" guide.

## 1. Recording evidence about a reference video

`VideoEvidence` is always operator-supplied -- no scraping, no
Whisper, no browser. Tag each observation with its analysis domain
(`hook`, `pacing`, `speech`, `lipsync`, `gesture`, `camera`, `framing`,
`editing`, `subtitles`, `audio`, `storytelling`, `cta`, `emotion`) plus,
where relevant, a controlled-vocabulary value from
`config/video_intelligence/{hooks,pacing,speech,camera,subtitles}.yaml`:

```python
from src.video_intelligence.evidence import VideoRecord, VideoEvidence, VideoEvidenceType, VideoPlatform

record = VideoRecord(
    platform=VideoPlatform.INSTAGRAM_REELS,
    creator_label="reference_creator",   # free text, never a real handle requirement
    reference="illustrative example",
)

evidence = [
    VideoEvidence(
        video_id=record.video_id,
        evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
        source_description="opening line",
        content_excerpt="opens with a direct question",
        timestamp_seconds=0.5,
        tags=["hook", "question"],
    ),
    VideoEvidence(
        video_id=record.video_id,
        evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION,
        source_description="closing CTA",
        content_excerpt="asks viewers to comment",
        timestamp_seconds=12.0,
        tags=["cta", "comment"],
    ),
]
```

## 2. Running an analysis

```python
from src.video_intelligence.workflow import VideoIntelligenceWorkflow

workflow = VideoIntelligenceWorkflow()  # loads config/video_intelligence/engine.yaml
result = workflow.analyze_video(record, evidence)

print(result.dna.dna_id, result.dna.overall_confidence)
print(result.report_markdown)
print([p.description for p in result.patterns])
```

This single call: builds a `VideoDNA` (all 13 analyzers), saves it to
`output/video_intelligence/<video_id>/video_dna.json`, renders and
saves a report to `.../report.md`, and -- unless
`ingest_into_knowledge_base=False` -- folds the video's structured
patterns into the shared, de-identified pattern library.

## 3. Reading back what's been analyzed

```python
workflow.load_dna(video_id)                 # VideoDNA | None
workflow.load_report_markdown(video_id)      # str | None
workflow.knowledge_base().list_patterns()     # list[ProductionPattern], ranked by support
workflow.knowledge_base().list_patterns("hook")  # scoped to one category
```

## 4. CLI

```
python3 -m src.video_intelligence.cli analyze \
    --platform instagram_reels --creator-label reference_creator --reference "illustrative example" \
    --evidence path/to/evidence_bundle.json

python3 -m src.video_intelligence.cli report --video-id <video_id>
python3 -m src.video_intelligence.cli patterns [--category hook]
python3 -m src.video_intelligence.cli validate --evidence path/to/evidence_bundle.json --video-id <video_id>
```

`evidence_bundle.json` is a JSON list of evidence dicts (same shape as
`VideoEvidence`'s fields; `video_id` is optional per-entry and defaults
to the one computed from `--platform`/`--creator-label`/`--reference`).
No subcommand can construct a browser, call an external API, or
transcribe audio -- every input is a local JSON file.

## 5. What gets written to disk

```
output/video_intelligence/<video_id>/
    video_dna.json
    report.md
output/video_intelligence/knowledge_base/
    patterns.json       de-identified ProductionPattern records
    membership.json      internal, uncapped video_id-per-pattern tracking
                          (never exposed directly -- see dna.md)
```

`output/` is gitignored; nothing under `output/video_intelligence/` is
ever committed by this package.
