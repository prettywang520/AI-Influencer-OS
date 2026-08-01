# AIKO OS — System Architecture Review

**Prepared by:** Lead Engineer review (automated read-only audit)
**Date:** 2026-07-31
**Scope:** Full repository at `/Users/wadewong/Documents/AI-Influencer-OS`
**Method:** Static reading of the full directory tree and the contents of every non-empty source, config, and spec file. No code was modified.

---

## 1. System Overview

AIKO OS is a system for running a single AI-generated Instagram persona ("Aiko Sato," a 24-year-old Japanese luxury travel creator) as a semi-automated content and community-management operation. The repository has **two distinct layers** that must be read differently:

1. **Specification / knowledge layer** (repo root: `01_core/` … `09_production/`, `brain/`, `business/`, `organization/`, `docs/`, `runtime/`, `templates/`, `CLAUDE.md/docs/`) — a large Markdown/YAML "agent operating system" describing roles, SOPs, brand rules, and agent personas in prose. This layer is mostly a **scaffold**: of the 1,568 tracked `.md/.py/.yaml/.json` files in the repo, **1,078 are empty placeholders** and only 490 contain real content. Most of the numbered folders (organizational structure, brain engines, business docs, runtime docs, architecture docs) are titled files with no body text yet — they represent intended structure, not implemented behavior.
2. **Runtime layer** (`10_apps/claude_runtime/`) — a working Python application that actually plans, prompts, validates, and (for replies) processes content. This is where the real logic lives, and it is the subject of most of this report.

The two layers are connected through `03_personas/aiko/`, which holds the **live data** the Python runtime reads at execution time (planning databases, motion/wardrobe libraries, reply-brain libraries) — distinct from the `02_agents/*` Markdown specs, which appear to be the design/spec mirror of that same domain (e.g. `02_agents/instagram_reply_agent/*.yaml` mirrors `03_personas/aiko/reply_brain/*.yaml` but is not the file the code actually loads).

Persona identity: **Aiko Sato**, defined by a fixed character description embedded in `prompt_engine.py` (hair, build, jewellery, "premium cream cardigan… beige travel dress," camera). Content premise: daily luxury-travel documentary-style Instagram content (1 feed post + 4 stories + 5 Reel scenes) generated from a deterministic planning pipeline, with images produced manually via ChatGPT and reels via Runway (no image/video generation API is currently wired in).

---

## 2. Services & Dependencies

### Runtime location
`10_apps/claude_runtime/` — a self-contained Python 3.14 app with its own `.venv`, `requirements.txt`, `main.py`, `src/`, `config/`, `output/`, `tests/`, `logs/`, and a Chrome `browser_profile/` (used for browser automation / Instagram session control).

### Python dependencies (`requirements.txt`)
| Package | Purpose |
|---|---|
| `openai` | OpenAI API client (present but not yet called from any read file — no `client.chat...` / `client.images...` usage found in the modules reviewed) |
| `pydantic`, `pydantic_core` | Data validation (available; most of the codebase actually uses plain `dataclasses`, not Pydantic) |
| `PyYAML` | Loads every planning/motion/wardrobe/reply YAML database |
| `pyperclip` | Clipboard copy for the manual image-production workflow (macOS `pbcopy` is also called directly via `subprocess`) |
| `python-dotenv` | `.env` loading (root `.env` file is currently empty) |
| `httpx`, `httpcore`, `anyio`, `h11`, `sniffio`, `distro`, `jiter`, `certifi`, `idna`, `typing_extensions` | Transitive dependencies of `openai`/`httpx` |
| `tqdm` | Progress bars |

### External/system dependencies
- **macOS `pbcopy`** — hard dependency of `production_worker.py` for the manual image workflow; the code explicitly states "this worker currently expects macOS."
- **Pillow (`PIL`)** — used by `image_pipeline.py` for image validation, dimension/aspect checks, and thumbnail generation. Imported but **not listed in `requirements.txt`** (a real dependency gap).
- **Chrome/Playwright-style browser profile** (`10_apps/claude_runtime/browser_profile/`) — supports `src/browser.py`, `src/providers/chatgpt_computer_use.py`, and the `src/social/instagram_session.py` / `instagram_reader.py` / `instagram_publisher.py` controllers for logged-in browser automation against ChatGPT and Instagram.
- **No database server** — all persistence is flat JSON/YAML files under `output/`, `03_personas/aiko/*/`, with atomic writes via `*.tmp` + `Path.replace()`.

### Internal "services" (each is a standalone CLI module with a `build_*()` factory and `main()`)
| Module | Role |
|---|---|
| `content_service.py` | Builds one day's content plan (prompts, captions, hashtags) |
| `production_service.py` | Wraps content generation into a manual production package (image/reel task queues) |
| `production_worker.py` | Human-in-the-loop worker that steps through the image queue |
| `image_pipeline.py` | Validates/organizes images once a human has generated and saved them |
| `reply_service.py` / `reply_worker.py` / `reply_dispatcher.py` | The Instagram auto-reply pipeline |
| `director.py` / `executor.py` | An older, calendar-driven runner (see §9, Gaps) |

---

## 3. Production Pipeline

Entry point: `ProductionService.run()` in `src/production_service.py`, invoked via `python -m src.production_service --date YYYY-MM-DD` (factory: `build_production_service()`).

```
target_date
  → ContentService.generate()            (see §4)
  → prepare day_root directory tree      (prompts/ captions/ reels/ images/ videos/ queues/ json/ logs/ archive/)
  → build ImageProductionTask list        (1 feed + 4 story tasks, provider="chatgpt_manual")
  → build ReelProductionTask list         (5 scene tasks, provider="runway_manual")
  → assemble ProductionManifest           (status starts at "ready_for_images")
  → write production_manifest.json, image_queue.json, reel_queue.json, production_summary.md
```

Output root: `10_apps/claude_runtime/output/<date>/`. Status progresses through the literal states `planned → ready_for_images → images_in_progress → images_complete → ready_to_publish → published` (or `failed`), though nothing in the reviewed code automates the transition past `images_complete` — publishing automation is explicitly deferred ("later publishing automation" per the module docstring).

`ProductionWorker` (`production_worker.py`) is the manual labor loop:
- `--next` copies the next pending image prompt to the clipboard and prints the expected save path.
- `--complete` verifies the saved file exists, is a supported image type, and is non-empty, then marks the task `completed`.
- `--sync` bulk-marks tasks complete if files already exist at the expected paths.
- `--retry` / `--fail` / `--status` manage failure handling.

`ImagePipeline` (`image_pipeline.py`) is the post-production QA step once images exist: computes SHA-256 checksum, verifies dimensions/aspect ratio (feed target 4:5, story target 9:16, minimum resolution 720×900), generates a 360×450 JPEG thumbnail via Pillow, writes per-image metadata, and rolls results up into `image_manifest.json` and back into `production_manifest.json`.

**No step in this pipeline calls an image-generation API.** Every image is produced by a human pasting the copied prompt into ChatGPT and saving the file at the exact path the pipeline expects.

---

## 4. Content Generation

Entry point: `ContentService.generate()` in `src/content_service.py` (factory `build_service()`, also runnable directly via `main.py` or `python -m src.content_service --date ...`).

```
Planning YAML (03_personas/aiko/planning/*.yaml)
  → PlanningLoader        loads/validates the planning databases
  → PlanningHistory        anti-repeat state (output/planning_history.json)
  → PlannerSelector        picks date-appropriate country/city/venue/theme
  → StoryBuilder           resolves stage → behavior/emotion/interaction/camera/motion (see §7)
  → DynamicContentPlanner  assembles the full DailyContentPlan (1 feed + 4 stories + 5 reel scenes)
  → PromptEngine           renders each moment into an image-generation prompt (see §5)
  → CaptionEngine          generates feed/story/reel captions
  → HashtagEngine          generates hashtag set
  → ReelsEngine            renders reel-scene prompts
  → ContentValidator       quality gate — raises RuntimeError if it fails
  → ContentService.save()  writes prompts/, captions/, reels/, hashtags.txt, content_plan.json, dashboard.md
  → PlanningHistory.remember_plan()  records the plan to prevent repeats
```

Two content planners exist side by side:
- **`content_planner.py` (`ContentPlanner`)** — a fully **hardcoded** example plan (one fixed "Rainy Bookstore Afternoon at Daikanyama T-Site" story). Appears to be a reference/demo implementation, not used by `build_service()`.
- **`dynamic_content_planner.py` (`DynamicContentPlanner`)** — the actual planner wired into `ContentService`, which generates a *different* plan each day by combining `PlannerSelector` + `StoryBuilder` output against the planning YAML databases (`behavior_database.yaml`, `camera_database.yaml`, `emotion_database.yaml`, `interaction_database.yaml`, `location_rotation.yaml`, `scene_database.yaml`, `seasonal_rules.yaml`, `story_templates.yaml`, `weather_rules.yaml`).

Output per day: `10_apps/claude_runtime/output/<date>/{prompts,captions,reels}/*.txt`, `hashtags.txt`, `content_plan.json`, `dashboard.md`.

---

## 5. Image Generation

There is **no automated image-generation call** anywhere in the reviewed codebase — `src/image_generator.py` exists as a file but is **completely empty** (a stub). Image generation today is a **manual, human-in-the-loop process**:

1. `PromptEngine.build_moment_prompt()` (`prompt_engine.py`) renders a structured prompt per `ContentMoment`: a fixed **CHARACTER** block (Aiko's appearance, hardcoded), the moment's location/behavior/emotion/interaction/body-motion/camera-story/daily-details, a fixed **STYLE** block ("luxury travel documentary photography… 8K"), and a fixed **NEGATIVE_RULES** block (no posing, no catalogue composition, no duplicated limbs, etc.).
2. `ContentService.save()` writes each rendered prompt to `output/<date>/prompts/feed_prompt.txt` / `story_N_prompt.txt`.
3. `ProductionService` wraps each prompt into an `ImageProductionTask` in `image_queue.json` with `provider="chatgpt_manual"`.
4. `ProductionWorker.prepare_next()` copies the prompt text to the clipboard (`pbcopy`) so a human can paste it into ChatGPT and generate the image, then save it to the exact `expected_output_file` path.
5. `ProductionWorker.complete_current()` / `ImagePipeline.run()` validate the resulting file (exists, correct type, non-empty, correct aspect ratio) and mark it complete.

Video/Reel generation follows the identical pattern via `ReelsEngine` + `PromptEngine.build_reel_scene_prompt()`, with `provider="runway_manual"` in `reel_queue.json` — also entirely human-driven (no Runway API call found).

`src/providers/chatgpt_computer_use.py` and `src/browser.py` exist and suggest an intended future path toward *browser-automated* (not API-automated) ChatGPT image generation, but no code path in the modules reviewed currently invokes them from the production pipeline.

---

## 6. Reply System

Entry point: `ReplyService` in `src/reply_service.py` (factory `build_service()`), which composes a `ReplyWorker` and a `ReplyDispatcher` around a shared `ReplyQueue`.

```
reply_queue.json (pending comment)
  → ReplyWorker.process_item()
      → parse CommentInput + ContentContext from the queue payload
      → ReplyMemory.get_excluded_reply_ids()   (anti-repeat per follower)
      → InstagramReplyAgent.process()           ← core decision engine
      → ReplyQueue.complete() / .fail()
      → ReplyMemory.remember_reply()             (if approved)
      → ReplyLogger.log_completed() / .log_failed()
  → ReplyDispatcher.dispatch_once()
      → PublisherAdapter.dispatch()  →  DryRunInstagramConnector   (no live publish yet)
      → ReplyLogger.write("reply_dispatch_result")
      → dispatch_state.json          (prevents duplicate dispatch)
```

`InstagramReplyAgent` (`reply_agent.py`) is a **rule-based, non-LLM** decision engine — no model call is made to draft a reply:
1. `ReplyClassifier` classifies intent and emotion from the raw comment text using `intent_engine.yaml` / `emotion_engine.yaml` (loaded from `03_personas/aiko/reply_brain/` via `ReplyLoader`).
2. If the classified action is `ignore`/`escalate`/`block_recommended`/`hold_for_review`, the agent returns immediately without generating text.
3. Otherwise, a **library** is selected (`location`, `hotel`, `food`, `shopping`, `camera`, `outfit`, `engagement`, `flirting`, `negative`, …) and a **reply group** inside that library is chosen via keyword-matching helper methods (e.g. `_resolve_hotel_group`, `_resolve_camera_group`, `_resolve_flirting_group`) or a deterministic "verified venue" reply when city+venue are both known.
4. `ReplySelector` performs weighted random selection of a candidate line from the YAML library, honoring `excluded_reply_ids` (recent replies to this follower) and retrying up to `maximum_attempts` (3) if a candidate fails validation.
5. `ReplyValidator` enforces hard limits (≤25 words, ≤3 emojis) before a candidate is accepted.
6. Result is one of: `publish` (approved, with reply text), `hold_for_review`, `escalate`, `ignore`, `block_recommended`.

`PublisherAdapter.dispatch()` maps the agent's decision onto Instagram connector operations (`publish_comment_reply`, `hide_comment`) — but the only connector currently wired in is `DryRunInstagramConnector`, so **no reply is actually posted to Instagram today**; the pipeline runs end-to-end in dry-run mode with full logging.

Supporting `src/social/` package (`instagram_dm_controller.py`, `instagram_reply_controller.py`, `instagram_reader.py`, `instagram_publisher.py`, `instagram_session.py`, `threads_controller.py`, `community_service.py`, `approval_service.py`) contains the browser-driven live-Instagram integration surface, but `social_router.py` — the file that would presumably route between these controllers — is currently **empty**, so this layer is not yet centrally wired together.

Data note: the runtime reply brain lives at **`03_personas/aiko/reply_brain/`** (the YAMLs `ReplyLoader` actually reads), not at `02_agents/instagram_reply_agent/` (which holds a parallel Markdown/YAML *specification* of the same agent — useful as design documentation but not the executing config).

---

## 7. Motion Engine — Where It's Connected

**`MotionEngine` (`src/motion_engine.py`) is fully implemented and already wired into the live content pipeline** — it is not a stub.

Connection path:
```
ContentService.generate()
  → DynamicContentPlanner
      → StoryBuilder(loader=PlanningLoader)      [src/story_builder.py]
          self.motion_engine = MotionEngine(
              library_path = 03_personas/aiko/motion/motion_library.yaml,
              rules_path   = 03_personas/aiko/motion/motion_rules.yaml,
              history_path = 10_apps/claude_runtime/output/history/motion.json,
          )
          → StoryBuilder.resolve_scene() calls motion_engine.choose(
                production_date, stage, activity=theme,
                used_today=used_behaviors, previous_category=...)
          → StoryBuilder.build_theme_scenes() calls motion_engine.remember_day(...)
              after all stages for the day are resolved
```

`MotionEngine.choose()` selects one behavior/motion candidate for a given story stage (e.g. `arrival`, `hotel` stages, `coffee_shop`, `leaving`) by:
- Mapping stage/activity to a motion category via alias tables (`stage_categories`, `activity_aliases`, `stage_aliases`).
- Filtering out motions used earlier that day (`used_today`), motions used in the last N days (`no_repeat_previous_days` rule, read from `motion_rules.yaml`), and motions in the same category as the immediately preceding scene (`no_same_category_consecutively`).
- Falling back progressively (day-repeat filter → any unused-today → any candidate) so a valid motion is always returned.

The chosen motion's `behavior_id`/`behavior_text` becomes `ResolvedScene.behavior_text`, which flows into `ContentMoment.behavior` and `ContentMoment.body_motion` (via `StoryBuilder._build_body_motion`), and ultimately into the rendered image/reel prompt text (`PromptEngine`). **This is the mechanism that makes each day's body motion/behavior different and non-repeating** — it is the most "alive" subsystem in the content pipeline.

---

## 8. Wardrobe Engine — Where It Should Be Connected (Currently Is Not)

**`WardrobeEngine` (`src/wardrobe_engine.py`) exists but is an unwired stub**, unlike Motion Engine:

- `WardrobeEngine.choose(city, weather, season, venue, activity)` **ignores every argument** and always returns the same hardcoded outfit dict (`"fitted rib knit top"`, `"high waist denim skirt"`, `"white sneakers"`, etc.).
- A `grep` across `src/` confirms `WardrobeEngine`/`wardrobe_engine` is referenced **only inside its own file** — no other module imports or calls it.
- Supporting data already exists at `03_personas/aiko/wardrobe/` (`tops.yaml`, `bottoms.yaml`, `dress.yaml`, `outerwear.yaml`, `season.yaml`, `rules.yaml`) — structurally parallel to `03_personas/aiko/motion/`, but nothing reads these files today.
- Confirming the gap at the data-model level: `ContentMoment` (`content_models.py`) has **no `outfit`/`wardrobe` field at all**. Clothing is currently a **fixed sentence** baked into `PromptEngine.CHARACTER` ("a premium cream cardigan over a soft beige travel dress…") applied identically to every single generated image, regardless of city, weather, season, venue, or activity.

**Where it should connect** (mirroring the Motion Engine pattern in §7):

1. **Instantiate** `WardrobeEngine` inside `StoryBuilder.__init__`, pointed at `03_personas/aiko/wardrobe/*.yaml` and an `output/history/wardrobe.json` anti-repeat file — same shape as the existing `MotionEngine` wiring.
2. **Implement** `WardrobeEngine.choose()` to actually read `tops.yaml`/`bottoms.yaml`/`dress.yaml`/`outerwear.yaml`, filter by the `season.yaml`/`rules.yaml` constraints, and select based on the `city`/`weather`/`season`/`venue`/`activity` parameters it already declares but ignores.
3. **Call it once per day** (not per scene — outfit should be consistent within a day, unlike behavior/motion which varies per scene) — most naturally from `DynamicContentPlanner` or `StoryBuilder.build_theme_scenes()`, before the per-scene loop, then pass the resulting outfit description into every `ContentMoment`/`ReelScene` built that day.
4. **Add an `outfit` field** to `ContentMoment` (and probably `ReelScene`) in `content_models.py` so the chosen wardrobe is structured data, not just baked prompt text.
5. **Update `PromptEngine`** to interpolate `moment.outfit` into the CHARACTER block instead of the current hardcoded clothing sentence, and to call `WardrobeEngine.save()` (already implemented) to persist the day's chosen outfit to history for anti-repeat purposes going forward.

This is a direct analogue of the Motion Engine integration already proven to work in `StoryBuilder` — the same call shape (`choose()` → apply to `ContentMoment` → `save()`/`remember_day()` for history) can be reused with minimal new plumbing.

---

## 9. Gaps & Observations (for awareness, not requested actions)

- **Two parallel "run the pipeline" entry points exist**: `main.py` + `content_service.py`/`production_service.py` (the actively-used dynamic pipeline), versus `director.py` + `executor.py`, which reads a separate `config/calendar.yaml` (weekday → theme/feed/stories) and shells out to `src/main.py <theme>/<feed>` via `subprocess`. These two paths don't obviously share state and may be legacy/experimental vs. current.
- **Two `calendar.yaml` files exist** — one at repo-root `config/calendar.yaml` (read by `director.py`) and one at `10_apps/claude_runtime/config/calendar.yaml` (app-level) — worth confirming which is authoritative.
- **Pillow (`PIL`) is used but not declared** in `requirements.txt`.
- **`social_router.py` is empty** — the social-platform routing layer referenced by folder structure isn't implemented yet.
- **Reply/publish is dry-run only** — `DryRunInstagramConnector` is the only connector wired to `PublisherAdapter`; going live requires swapping in a real Instagram connector (the `src/social/instagram_publisher.py` scaffold appears to be the intended target).
- **The specification layer (01–09, brain/, business/, organization/, docs/, runtime/, CLAUDE.md/docs/) is ~70% empty placeholder files.** Treat filenames there as a roadmap of intended documentation/behavior, not as current truth — always verify against `10_apps/claude_runtime/src/` before relying on anything from that layer.

---

## 10. Directory Reference (top level)

| Path | Contents |
|---|---|
| `01_core/`, `02_agents/` | Agent role/spec Markdown + YAML (mostly design docs; `02_agents/instagram_reply_agent/*` mirrors the real reply-brain data) |
| `03_personas/aiko/` | **Live runtime data**: `planning/`, `motion/`, `wardrobe/`, `reply_brain/`, plus large spec trees (`06_motion/`, `00_identity/`, etc.) |
| `04_workflows/`, `05_plugins/` | Workflow/plugin specs (mostly placeholders) |
| `06_memory/` | Memory-system specs (placeholders) |
| `07_database/` | Structured knowledge YAML (cities, camera, fashion, poses, hashtags, etc.) — reference/lookup data |
| `08_content/`, `09_production/` | Sample content requests and production folders |
| `10_apps/claude_runtime/` | **The actual Python runtime** — see §2–§8 |
| `brain/`, `business/`, `organization/`, `docs/`, `runtime/`, `templates/`, `CLAUDE.md/docs/` | High-level org/product/architecture documentation (mostly empty placeholders as of this review) |
