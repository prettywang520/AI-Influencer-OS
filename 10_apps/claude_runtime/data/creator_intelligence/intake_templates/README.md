# Intake Templates

Fill-in-the-blank starting points for `src/creator_intelligence/intake.py`'s
`--add-*` flags. These are **never real creator data** -- every value is
a placeholder or `null`. `--init` copies this directory into every new
intake workspace's `_templates/` folder for convenience.

| Template | Used with |
| --- | --- |
| `creator_profile_template.json` | Reference only -- `--init` builds `creator_profile.json` from CLI flags, not this file |
| `screenshot_template.json` | `--add-screenshot FILE.json` |
| `visual_realism_annotation_template.json` | Reference shape for `screenshot_template.json`'s `visual_annotations` field |
| `caption_template.json` | `--add-caption FILE.json` |
| `reply_pair_template.json` | `--add-reply-pair FILE.json` |
| `reel_note_template.json` | `--add-reel-note FILE.json` |
| `highlight_note_template.json` | `--add-highlight-note FILE.json` |
| `manual_note_template.json` | `--add-manual-note FILE.json` |

Every `"_comment"` key is documentation only and is ignored by the
loader -- do not remove the field it documents, just replace the
placeholder value next to it.
