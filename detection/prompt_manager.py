"""Prompt (object category) parsing, per-model preparation and preset storage."""
from __future__ import annotations

import json
import re
from pathlib import Path

from utils.config_loader import load_config, resolve_path


class PromptError(ValueError):
    """Raised when no valid prompts remain after cleaning."""


def normalize_prompt(text: str) -> str:
    """Collapse whitespace and strip surrounding spaces/periods."""
    return re.sub(r"\s+", " ", str(text)).strip().strip(".").strip()


def parse_prompts(raw: str | list[str]) -> list[str]:
    """Split comma/newline separated text (or a list) into a clean, de-duplicated list.

    Order is preserved; duplicates are detected case-insensitively; empties are removed.
    """
    if isinstance(raw, str):
        parts = re.split(r"[,\n\r;]+", raw)
    else:
        parts = [piece for item in raw for piece in re.split(r"[\n\r;]+", str(item))]
    seen: set[str] = set()
    cleaned: list[str] = []
    for part in parts:
        norm = normalize_prompt(part)
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            cleaned.append(norm)
    return cleaned


def require_prompts(prompts: list[str]) -> list[str]:
    cleaned = parse_prompts(prompts)
    if not cleaned:
        raise PromptError("Please enter at least one object category.")
    return cleaned


def prepare_for_grounding_dino(prompts: list[str]) -> str:
    """Grounding DINO expects ONE lower-case string with classes separated by periods."""
    cleaned = require_prompts(prompts)
    return " . ".join(p.lower() for p in cleaned) + " ."


def prepare_for_yolo_world(prompts: list[str]) -> list[str]:
    """YOLO-World's set_classes() expects a list of class names."""
    return require_prompts(prompts)


def match_label_to_prompt(label: str, prompts: list[str]) -> str:
    """Map a (possibly partial/composite) Grounding DINO label back to one of the prompts."""
    norm = normalize_prompt(label).lower()
    lookup = {p.lower(): p for p in prompts}
    if norm in lookup:
        return lookup[norm]
    candidates = [orig for low, orig in lookup.items() if low in norm or norm in low]
    if candidates:
        return max(candidates, key=len)
    return norm


# ----------------------------------------------------------------- presets
def _prompts_dir() -> Path:
    path = resolve_path(load_config()["paths"]["prompts_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_test_prompts() -> dict:
    path = _prompts_dir() / "test_prompts.json"
    if not path.exists():
        return {"categories": [], "common": [], "uncommon": []}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_saved_presets() -> dict[str, list[str]]:
    path = _prompts_dir() / "saved_prompts.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    return {str(k): parse_prompts(v) for k, v in data.items() if isinstance(v, list)}


def save_prompt_preset(name: str, prompts: list[str]) -> None:
    name = normalize_prompt(name)
    if not name:
        raise PromptError("Preset name cannot be empty.")
    prompts = require_prompts(prompts)
    presets = load_saved_presets()
    presets[name] = prompts
    with (_prompts_dir() / "saved_prompts.json").open("w", encoding="utf-8") as fh:
        json.dump(presets, fh, indent=2, ensure_ascii=False)


def get_all_presets() -> dict[str, list[str]]:
    """Built-in test sets followed by the user's saved presets."""
    test = load_test_prompts()
    presets: dict[str, list[str]] = {}
    if test.get("categories"):
        presets[f"Test set - all {len(test['categories'])} categories"] = test["categories"]
    if test.get("common"):
        presets["Test set - common"] = test["common"]
    if test.get("uncommon"):
        presets["Test set - uncommon industrial"] = test["uncommon"]
    presets.update(load_saved_presets())
    return presets
