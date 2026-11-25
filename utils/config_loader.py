"""Centralised YAML configuration loading and project paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
_CONFIG_FILES = ("config.yaml", "model_config.yaml")


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and merge all YAML config files. The result is cached; do not mutate it."""
    cfg: dict[str, Any] = {}
    for name in _CONFIG_FILES:
        path = CONFIG_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing configuration file: {path}")
        with path.open("r", encoding="utf-8") as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_project_dirs() -> None:
    """Create every directory the application writes to."""
    cfg = load_config()
    for rel in cfg["paths"].values():
        resolve_path(rel).mkdir(parents=True, exist_ok=True)
    for model_cfg in cfg["models"].values():
        for key in ("cache_dir", "weights_dir"):
            if key in model_cfg:
                resolve_path(model_cfg[key]).mkdir(parents=True, exist_ok=True)
