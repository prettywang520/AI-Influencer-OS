# Phase 12A.1 -- Creator Evidence Intake Workflow

**Status:** Tooling only. No evidence about either real target creator
(`carlysuen112`, `aitana_10_01`) has been collected. Read
[ethics.md](ethics.md) and [evidence_guide.md](evidence_guide.md)
before collecting anything -- they are binding.

## What this phase adds

`src/creator_intelligence/intake.py`, `intake_manifest.py`, and
`intake_validator.py` turn manually-gathered observations into
well-formed `Evidence` records (Phase 12A.0's own model, unmodified),
organized per-creator into an **intake workspace**, with a
completeness report showing how much evidence has been gathered
against the configured targets in `config/creator_intelligence/intake.yaml`.
This phase does **not** run any analyzer -- see [§ Phase 12A.2](#phase-12a2-recommendation)
below.

## 1. Initialize a workspace

```
python3 -m src.creator_intelligence.intake \
  --init \
  --username carlysuen112 \
  --platform instagram \
  --profile-url "https://www.instagram.com/carlysuen112/" \
  --output output/creator_intelligence/intake/carlysuen112

python3 -m src.creator_intelligence.intake \
  --init \
  --username aitana_10_01 \
  --platform instagram \
  --profile-url "https://www.instagram.com/aitana_10_01/" \
  --output output/creator_intelligence/intake/aitana_10_01
```

`--output` points into `output/`, which is gitignored -- real intake
data for either creator is never auto-tracked by git. `--init` only
creates directories/template files/an ownership marker; it never
accesses the network.

## 2. Add evidence

Every `--add-*` flag takes a path to a small JSON file. Copy the
matching template from `<output>/_templates/` (or
`data/creator_intelligence/intake_templates/`) and fill it in with
observations from content you have viewed publicly.

```
python3 -m src.creator_intelligence.intake --add-caption my_caption.json --output <dir>
python3 -m src.creator_intelligence.intake --add-reply-pair my_reply.json --output <dir>
python3 -m src.creator_intelligence.intake --add-screenshot my_screenshot.json --output <dir>
python3 -m src.creator_intelligence.intake --add-reel-note my_reel.json --output <dir>
python3 -m src.creator_intelligence.intake --add-highlight-note my_highlight.json --output <dir>
python3 -m src.creator_intelligence.intake --add-manual-note my_note.json --output <dir>
```

Each ingested item is stored twice: as its full-fidelity typed record
(under `<output>/{category}/`) and as a Phase-12A.0-compatible
`Evidence` entry in `<output>/evidence_bundle.json`. `<output>/intake_manifest.json`
is rebuilt after every add.

## 3. Validate

```
python3 -m src.creator_intelligence.intake_validator --intake <dir> [--strict] [--json]
```

Read-only. Checks ids, links, path safety, record shape, and manifest
consistency; reports a completeness breakdown. `--strict` also fails
on any category still below its `target` (not just below `minimum`).
Completeness is a coverage measure only -- it is never a quality,
confidence, or creator score.

## 4. Real creator handoff

Evidence for `carlysuen112` and `aitana_10_01` must be supplied by a
human operator. Two options:

**Option A -- Manual screenshot intake.** An operator manually browses
the creator's public profile (as any member of the public could),
takes screenshots, and copies caption/reply text by hand into the
`--add-*` JSON templates. This is the default, always-available path.

**Option B -- User-owned lawful export.** If the operator has export
files they are personally permitted to use (e.g. a creator-provided
media kit, or their own saved research notes), those can be summarized
into the same `--add-*` templates via `exported_data`/`manual_note`
evidence types.

Neither option involves scraping, login automation, or bypassing any
access control -- both remain fully within the boundaries in
[evidence_guide.md](evidence_guide.md).

## Phase 12A.2 recommendation

Not implemented by this phase. The recommended flow for a future
phase:

```
validated intake -> analyzer suite -> Creator DNA -> evidence-backed report -> confidence report
```

i.e., once `intake_validator` reports a passing, sufficiently-complete
workspace, a future phase loads `evidence_bundle.json`, passes it to
`summary_builder.build_creator_dna()` (Phase 12A.0, unmodified), and
produces a `CreatorDNA` plus a human-readable report citing back to
the original evidence. This phase does not perform that step.
