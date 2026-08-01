from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from .caption_engine import _clip_to_words, _short_subject
from .content_models import DailyContentPlan

# ---------------------------------------------------------------------------
# Phase 9 — Auto Daily Director data models, wardrobe selection, Threads
# generation, and supplementary quality-gate checks.
# ---------------------------------------------------------------------------
#
# This module never publishes anything, never generates an image
# through an API, and never touches src/social/ or the Instagram
# comment/DM systems.


class DailyDirectorConfigError(RuntimeError):
    """Raised when config/daily_director.yaml is missing or invalid."""


class WardrobeSelectionError(RuntimeError):
    """Raised when a wardrobe outfit cannot be selected safely."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OutputPathsConfig:
    root: str
    history_dir: str
    logs_dir: str
    wardrobe_history_file: str
    log_filename_template: str
    content_plan_file: str
    production_manifest_file: str
    dashboard_file: str
    prompts_dir: str
    reel_prompts_dir: str
    captions_dir: str
    hashtags_dir: str
    queues_dir: str
    images_dir: str
    feed_prompt_file: str
    story_prompt_template: str
    reel_prompt_template: str
    feed_caption_file: str
    story_caption_template: str
    reel_caption_file: str
    threads_post_file: str
    feed_hashtags_file: str
    reel_hashtags_file: str
    threads_hashtags_file: str
    image_queue_file: str
    reel_queue_file: str


@dataclass(slots=True)
class CaptionStyleConfig:
    maximum_words: int = 80
    maximum_emojis: int = 3


@dataclass(slots=True)
class ThreadsStyleConfig:
    maximum_characters: int = 280
    maximum_emojis: int = 2


@dataclass(slots=True)
class HashtagCountsConfig:
    feed_count: int = 12
    reel_count: int = 8
    threads_count: int = 3


@dataclass(slots=True)
class WardrobeRulesConfig:
    no_same_outfit_two_days: bool = True
    no_same_top_three_days: bool = True
    fallback_top: str = "fitted long sleeve top"
    fallback_bottom: str = "high waist trousers"
    fallback_dress: str = "fitted knit dress"
    fallback_shoes: str = "white sneakers"
    fallback_bag: str = "cream leather shoulder bag"
    fallback_jewellery: str = "minimal gold jewellery"
    fallback_hair: str = "long silky loose hair"


@dataclass(slots=True)
class DailyDirectorConfig:
    output: OutputPathsConfig
    caption_style: CaptionStyleConfig
    threads_style: ThreadsStyleConfig
    hashtags: HashtagCountsConfig
    wardrobe: WardrobeRulesConfig
    minimum_daily_details: int = 3


DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "daily_director.yaml"


def _runtime_root() -> Path:
    """
    daily_run_models.py location:

    10_apps/claude_runtime/src/daily_run_models.py

    parents[1] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[1]


def load_daily_director_config(
    *,
    config_path: str | Path | None = None,
) -> DailyDirectorConfig:
    runtime_root = _runtime_root()

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (runtime_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve()
    )

    if not resolved_config_path.exists():
        raise DailyDirectorConfigError(
            f"daily_director config not found: {resolved_config_path}"
        )

    try:
        with resolved_config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise DailyDirectorConfigError(
            f"Invalid YAML in {resolved_config_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict) or not raw:
        raise DailyDirectorConfigError(
            f"daily_director config is empty or invalid: {resolved_config_path}"
        )

    output_section = raw.get("output") or {}
    caption_section = raw.get("caption_style") or {}
    threads_section = raw.get("threads_style") or {}
    hashtags_section = raw.get("hashtags") or {}
    wardrobe_section = raw.get("wardrobe") or {}
    quality_section = raw.get("quality_gate") or {}

    output = OutputPathsConfig(
        root=str(output_section.get("root", "output")),
        history_dir=str(output_section.get("history_dir", "output/history")),
        logs_dir=str(output_section.get("logs_dir", "output/logs")),
        wardrobe_history_file=str(
            output_section.get("wardrobe_history_file", "wardrobe.json")
        ),
        log_filename_template=str(
            output_section.get(
                "log_filename_template", "auto_daily_director_{date}.json"
            )
        ),
        content_plan_file=str(
            output_section.get("content_plan_file", "content_plan.json")
        ),
        production_manifest_file=str(
            output_section.get(
                "production_manifest_file", "production_manifest.json"
            )
        ),
        dashboard_file=str(output_section.get("dashboard_file", "dashboard.md")),
        prompts_dir=str(output_section.get("prompts_dir", "prompts")),
        reel_prompts_dir=str(
            output_section.get("reel_prompts_dir", "prompts/reels")
        ),
        captions_dir=str(output_section.get("captions_dir", "captions")),
        hashtags_dir=str(output_section.get("hashtags_dir", "hashtags")),
        queues_dir=str(output_section.get("queues_dir", "queues")),
        images_dir=str(output_section.get("images_dir", "images")),
        feed_prompt_file=str(
            output_section.get("feed_prompt_file", "feed_prompt.txt")
        ),
        story_prompt_template=str(
            output_section.get(
                "story_prompt_template", "story_{index}_prompt.txt"
            )
        ),
        reel_prompt_template=str(
            output_section.get(
                "reel_prompt_template", "shot_{index:02d}_prompt.txt"
            )
        ),
        feed_caption_file=str(
            output_section.get("feed_caption_file", "feed_caption.txt")
        ),
        story_caption_template=str(
            output_section.get(
                "story_caption_template", "story_{index}_caption.txt"
            )
        ),
        reel_caption_file=str(
            output_section.get("reel_caption_file", "reel_caption.txt")
        ),
        threads_post_file=str(
            output_section.get("threads_post_file", "threads_post.txt")
        ),
        feed_hashtags_file=str(
            output_section.get("feed_hashtags_file", "feed_hashtags.txt")
        ),
        reel_hashtags_file=str(
            output_section.get("reel_hashtags_file", "reel_hashtags.txt")
        ),
        threads_hashtags_file=str(
            output_section.get(
                "threads_hashtags_file", "threads_hashtags.txt"
            )
        ),
        image_queue_file=str(
            output_section.get("image_queue_file", "image_queue.json")
        ),
        reel_queue_file=str(
            output_section.get("reel_queue_file", "reel_queue.json")
        ),
    )

    caption_style = CaptionStyleConfig(
        maximum_words=int(caption_section.get("maximum_words", 80)),
        maximum_emojis=int(caption_section.get("maximum_emojis", 3)),
    )

    threads_style = ThreadsStyleConfig(
        maximum_characters=int(threads_section.get("maximum_characters", 280)),
        maximum_emojis=int(threads_section.get("maximum_emojis", 2)),
    )

    hashtags = HashtagCountsConfig(
        feed_count=int(hashtags_section.get("feed_count", 12)),
        reel_count=int(hashtags_section.get("reel_count", 8)),
        threads_count=int(hashtags_section.get("threads_count", 3)),
    )

    wardrobe = WardrobeRulesConfig(
        no_same_outfit_two_days=bool(
            wardrobe_section.get("no_same_outfit_two_days", True)
        ),
        no_same_top_three_days=bool(
            wardrobe_section.get("no_same_top_three_days", True)
        ),
        fallback_top=str(
            wardrobe_section.get("fallback_top", "fitted long sleeve top")
        ),
        fallback_bottom=str(
            wardrobe_section.get("fallback_bottom", "high waist trousers")
        ),
        fallback_dress=str(
            wardrobe_section.get("fallback_dress", "fitted knit dress")
        ),
        fallback_shoes=str(
            wardrobe_section.get("fallback_shoes", "white sneakers")
        ),
        fallback_bag=str(
            wardrobe_section.get(
                "fallback_bag", "cream leather shoulder bag"
            )
        ),
        fallback_jewellery=str(
            wardrobe_section.get(
                "fallback_jewellery", "minimal gold jewellery"
            )
        ),
        fallback_hair=str(
            wardrobe_section.get("fallback_hair", "long silky loose hair")
        ),
    )

    return DailyDirectorConfig(
        output=output,
        caption_style=caption_style,
        threads_style=threads_style,
        hashtags=hashtags,
        wardrobe=wardrobe,
        minimum_daily_details=int(
            quality_section.get("minimum_daily_details", 3)
        ),
    )


# ---------------------------------------------------------------------------
# Atomic file writes — shared by the wardrobe selector and the director.
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


# ---------------------------------------------------------------------------
# Wardrobe (requirement 8, 10)
# ---------------------------------------------------------------------------
#
# src/wardrobe_engine.py's WardrobeEngine.choose() is a non-functional
# stub: it ignores every argument (city, weather, season, venue,
# activity) and always returns the identical hardcoded outfit, so it
# cannot satisfy requirement 10's season-aware, anti-repeat selection.
# wardrobe_engine.py and content_models.py are not in Phase 9's
# editable file list, so this is a standalone implementation reading
# 03_personas/aiko/wardrobe/*.yaml directly, rather than a wrapper
# around the stub. Selected outfits are attached to every visual
# content item's OUTPUT record by auto_daily_director.py rather than
# added as a field on the shared ContentMoment/ReelScene dataclasses.


@dataclass(slots=True)
class WardrobeSelection:
    top: str | None
    bottom: str | None
    dress: str | None
    shoes: str
    bag: str
    jewellery: str
    outerwear: str | None
    hair: str
    city: str
    weather: str
    season: str
    venue: str
    activity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SEASON_FALLBACK_ORDER = {
    "spring": ["spring", "summer", "autumn", "winter"],
    "summer": ["summer", "spring", "autumn", "winter"],
    "autumn": ["autumn", "summer", "spring", "winter"],
    "winter": ["winter", "autumn", "summer", "spring"],
}


class WardrobeSelector:
    """
    Season-aware wardrobe selection with anti-repeat history, matching
    the rules already declared (but never implemented) in
    03_personas/aiko/wardrobe/rules.yaml: no identical full outfit on
    two consecutive days, no identical top on three consecutive days.

    "Appropriate to city/weather/venue/activity" is honest about a
    real data-granularity limit: 03_personas/aiko/wardrobe/tops.yaml
    and bottoms.yaml are only differentiated by season today (no
    city/weather/venue/activity dimension exists in that data at
    all), so those parameters are accepted, recorded on the
    selection, and available for a future data expansion, but do not
    yet influence the choice beyond season.
    """

    def __init__(
        self,
        *,
        wardrobe_root: Path,
        history_path: Path,
        rules: WardrobeRulesConfig,
        random_seed: int | None = None,
    ) -> None:
        self.wardrobe_root = wardrobe_root
        self.history_path = history_path
        self.rules = rules
        self.random = random.Random(random_seed)

        self._tops = self._load_season_pool("tops.yaml")
        self._bottoms = self._load_season_pool("bottoms.yaml")
        self._dresses = self._load_flat_pool("dress.yaml")
        self._outerwear = self._load_flat_pool("outerwear.yaml")

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        path = self.wardrobe_root / filename

        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return data if isinstance(data, dict) else {}

    @staticmethod
    def _split_items(value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        # tops.yaml / bottoms.yaml store each season as one newline
        # -separated string rather than a YAML list.
        return [line.strip() for line in str(value).splitlines() if line.strip()]

    def _load_season_pool(self, filename: str) -> dict[str, list[str]]:
        data = self._load_yaml(filename)

        return {
            str(season).strip().lower(): self._split_items(value)
            for season, value in data.items()
        }

    def _load_flat_pool(self, filename: str) -> list[str]:
        data = self._load_yaml(filename)
        items: list[str] = []

        for value in data.values():
            items.extend(self._split_items(value))

        return items

    def _pool_for_season(
        self,
        pool: dict[str, list[str]],
        season: str,
    ) -> list[str]:
        season_key = season.strip().lower()
        order = _SEASON_FALLBACK_ORDER.get(season_key, [season_key])

        for candidate_season in order:
            candidates = pool.get(candidate_season)

            if candidates:
                return candidates

        for candidates in pool.values():
            if candidates:
                return candidates

        return []

    def _read_history(self) -> dict[str, Any]:
        if not self.history_path.exists():
            return {}

        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

        return data if isinstance(data, dict) else {}

    def _recent_outfits(
        self,
        *,
        production_date: str,
        days: int,
    ) -> list[dict[str, Any]]:
        history = self._read_history()
        target = date.fromisoformat(production_date)
        recent: list[dict[str, Any]] = []

        for offset in range(1, days + 1):
            day = (target - timedelta(days=offset)).isoformat()
            entry = history.get(day)

            if isinstance(entry, dict):
                recent.append(entry)

        return recent

    def _choose_avoiding_repeat(
        self,
        pool: list[str],
        *,
        avoid_value: str | None,
    ) -> str:
        if not pool:
            raise WardrobeSelectionError(
                "No wardrobe candidates available for this season"
            )

        candidates = (
            [item for item in pool if item != avoid_value]
            if avoid_value
            else list(pool)
        )

        if not candidates:
            candidates = list(pool)

        return self.random.choice(candidates)

    def choose(
        self,
        *,
        production_date: str,
        city: str,
        weather: str,
        season: str,
        venue: str,
        activity: str,
    ) -> WardrobeSelection:
        top_pool = self._pool_for_season(self._tops, season) or [
            self.rules.fallback_top
        ]
        bottom_pool = self._pool_for_season(self._bottoms, season) or [
            self.rules.fallback_bottom
        ]
        dress_pool = self._dresses or [self.rules.fallback_dress]

        yesterday_outfits = self._recent_outfits(
            production_date=production_date, days=1
        )
        yesterday_outfit = yesterday_outfits[0] if yesterday_outfits else None

        last_two_days = self._recent_outfits(
            production_date=production_date, days=2
        )
        recent_tops = {
            entry.get("top") for entry in last_two_days if entry.get("top")
        }

        use_dress = bool(dress_pool) and self.random.random() < 0.35

        top: str | None = None
        bottom: str | None = None
        dress: str | None = None

        if use_dress:
            dress = self._choose_avoiding_repeat(
                dress_pool,
                avoid_value=(
                    (yesterday_outfit or {}).get("dress")
                    if self.rules.no_same_outfit_two_days
                    else None
                ),
            )
        else:
            available_tops = (
                [item for item in top_pool if item not in recent_tops]
                if self.rules.no_same_top_three_days
                else list(top_pool)
            )

            if not available_tops:
                available_tops = list(top_pool)

            top = self.random.choice(available_tops)

            bottom = self._choose_avoiding_repeat(
                bottom_pool,
                avoid_value=(
                    (yesterday_outfit or {}).get("bottom")
                    if self.rules.no_same_outfit_two_days
                    else None
                ),
            )

            if (
                self.rules.no_same_outfit_two_days
                and yesterday_outfit
                and top == yesterday_outfit.get("top")
                and bottom == yesterday_outfit.get("bottom")
                and len(available_tops) > 1
            ):
                remaining_tops = [
                    item for item in available_tops if item != top
                ]

                if remaining_tops:
                    top = self.random.choice(remaining_tops)

        return WardrobeSelection(
            top=top,
            bottom=bottom,
            dress=dress,
            shoes=self.rules.fallback_shoes,
            bag=self.rules.fallback_bag,
            jewellery=self.rules.fallback_jewellery,
            outerwear=(
                self.random.choice(self._outerwear)
                if self._outerwear
                else None
            ),
            hair=self.rules.fallback_hair,
            city=city,
            weather=weather,
            season=season,
            venue=venue,
            activity=activity,
        )

    def save(
        self,
        *,
        production_date: str,
        outfit: WardrobeSelection,
    ) -> None:
        history = self._read_history()
        history[production_date] = outfit.to_dict()
        write_json_atomic(self.history_path, history)


def build_wardrobe_selector(
    *,
    config: DailyDirectorConfig,
    random_seed: int | None = None,
) -> WardrobeSelector:
    project_root = Path(__file__).resolve().parents[3]
    runtime_root = _runtime_root()

    return WardrobeSelector(
        wardrobe_root=project_root / "03_personas" / "aiko" / "wardrobe",
        history_path=(
            runtime_root
            / config.output.history_dir
            / config.output.wardrobe_history_file
        ),
        rules=config.wardrobe,
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# Reel scene stage labels (requirement 7, 8)
# ---------------------------------------------------------------------------
#
# ReelScene (content_models.py) has no stage-label field at all —
# only scene_number. content_models.py is not editable, so the
# beginning/development/interaction/emotional_beat/ending progression
# is attached positionally to auto_daily_director.py's own OUTPUT
# records instead, mirroring how dynamic_content_planner.py already
# positionally remaps each theme's own story stage names onto the
# fixed arrival/discovery/quiet_moment/leaving Story progression.

_REEL_STAGE_LABELS = {
    1: "beginning",
    2: "development",
    3: "interaction",
    4: "emotional_beat",
    5: "ending",
}


def reel_stage_label(scene_number: int) -> str:
    return _REEL_STAGE_LABELS.get(scene_number, f"scene_{scene_number}")


def reel_real_life(*, scene, location: str) -> str:
    """
    ReelScene (content_models.py) has no real_life field at all —
    only ContentMoment does. Requirement 8 requires every visual
    content item, reel scenes included, to contain one. Synthesised
    the same way StoryBuilder._build_real_life() already builds it
    for Feed/Story moments, attached to auto_daily_director.py's own
    OUTPUT record rather than the shared dataclass.
    """
    stage_label = reel_stage_label(scene.scene_number).replace("_", " ")

    return (
        f"At {location}, Aiko is actively "
        f"{scene.behavior.rstrip('.').lower()}. "
        f"This shot advances the {stage_label} beat of the Reel."
    )


# ---------------------------------------------------------------------------
# Threads post (requirement 12)
# ---------------------------------------------------------------------------

_THREADS_OPENERS = [
    "okay so",
    "not me",
    "today was",
    "little update —",
    "random thought:",
    "quick one —",
]

_THREADS_HOOKS = [
    "anyone else do this?",
    "tell me i'm not the only one",
    "what would you have done?",
    "should i make this a regular thing?",
    "okay your turn, what's your version of this?",
    "no notes, honestly",
]

_THREADS_JAPANESE = ["ね", "なんか", "ちょっと", "本当に", "やっぱり"]


def generate_threads_post(
    *,
    plan: DailyContentPlan,
    feed_caption: str,
    style: ThreadsStyleConfig,
    rng: random.Random,
) -> str:
    """
    Casual, personal Threads-voice post: distinct tone and wording
    from the more polished Feed caption, under maximum_characters,
    may mix English and short Japanese, ends with a natural
    conversation hook. Guaranteed not identical to feed_caption.
    """
    behavior = _short_subject(plan.feed.behavior)

    def _build() -> str:
        opener = rng.choice(_THREADS_OPENERS)
        hook = rng.choice(_THREADS_HOOKS)
        japanese = (
            f" {rng.choice(_THREADS_JAPANESE)}" if rng.random() < 0.4 else ""
        )

        text = f"{opener} {behavior} at {plan.venue}{japanese}. {hook}"

        return " ".join(text.split()).strip()

    text = _build()
    attempts = 0

    while text.strip().lower() == feed_caption.strip().lower() and attempts < 5:
        text = _build()
        attempts += 1

    if len(text) > style.maximum_characters:
        text = text[: style.maximum_characters - 1].rstrip() + "…"

    return text


# ---------------------------------------------------------------------------
# Supplementary quality gate (requirement 9)
# ---------------------------------------------------------------------------
#
# ContentValidator (content_validator.py, reused via .validate(),
# called first and unmodified) already checks: exactly 4 stories,
# exactly 5 reel scenes, missing real_life/behavior/emotion/
# interaction/camera_story per Feed+Story moment, minimum 3 daily
# details per Feed+Story+reel-scene, duplicate Feed/Story behavior,
# duplicate Feed/Story camera, missing interaction per reel scene,
# the arrival/discovery/quiet_moment/leaving Story progression,
# missing captions. content_validator.py is not in Phase 9's editable
# file list, so this function supplies exactly the checks it does
# NOT already cover: missing body_motion (any item), missing wardrobe
# (day-level), and reel-scene-level missing_emotion / duplicate
# behavior / duplicate camera (ContentValidator only checks these
# for Feed+Stories, not reel scenes).


def run_supplementary_quality_gate(
    *,
    plan: DailyContentPlan,
    wardrobe: WardrobeSelection,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    moments = [plan.feed, *plan.stories]

    for moment in moments:
        if not moment.body_motion.strip():
            errors.append(f"{moment.content_id}:missing_body_motion")

    if not (wardrobe.top or wardrobe.dress):
        errors.append("missing_wardrobe")

    reel_behaviors: list[str] = []
    reel_cameras: list[str] = []

    for scene in plan.reel_scenes:
        prefix = f"reel_scene_{scene.scene_number}"

        if not scene.body_motion.strip():
            errors.append(f"{prefix}:missing_body_motion")

        if not scene.emotion.strip():
            errors.append(f"{prefix}:missing_emotion")

        reel_behaviors.append(scene.behavior.strip().lower())
        reel_cameras.append(scene.camera_story.strip().lower())

    if len(reel_behaviors) != len(set(reel_behaviors)):
        errors.append("duplicate_reel_behavior_detected")

    if len(reel_cameras) != len(set(reel_cameras)):
        errors.append("duplicate_reel_camera_detected")

    return errors, warnings


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DailyDirectorImageTask:
    task_id: str
    production_date: str
    content_id: str
    content_type: str
    title: str
    prompt_file: str
    expected_output_file: str
    status: str = "pending"
    provider: str = "chatgpt_manual"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DailyDirectorReelTask:
    task_id: str
    production_date: str
    scene_number: int
    title: str
    duration_seconds: float
    prompt_file: str
    expected_output_file: str
    status: str = "pending"
    provider: str = "runway_manual"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DailyDirectorManifest:
    production_id: str
    production_date: str
    created_at: str
    country: str
    city: str
    venue: str
    theme: str
    story_summary: str
    status: str
    feed_count: int
    story_count: int
    reel_scene_count: int
    image_tasks: list[DailyDirectorImageTask]
    reel_tasks: list[DailyDirectorReelTask]
    hashtags: dict[str, list[str]]
    quality_gate: dict[str, Any]
    wardrobe: dict[str, Any]
    output_directory: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_tasks"] = [task.to_dict() for task in self.image_tasks]
        data["reel_tasks"] = [task.to_dict() for task in self.reel_tasks]
        return data


@dataclass(slots=True)
class StatusReport:
    production_date: str
    exists: bool
    production_status: str
    quality_gate_status: str
    image_task_count: int
    reel_task_count: int
    files_present: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)
    next_recommended_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunLogEntry:
    started_at: str
    date: str
    finished_at: str | None = None
    selection: dict[str, Any] = field(default_factory=dict)
    services_called: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    quality_gate: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    result: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
