"""
Pipeline configuration: config/rift.yaml plus an optional overlay.

The default file is resolved relative to the repo root, not the working directory, so
every script finds it wherever it is launched from. An overlay YAML (RIFT_CONFIG env var
or --config on any experiment script) is deep-merged on top: use it for workspace-specific
judges, catalog slugs or negotiated prices without editing the shared file.
"""

from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "rift.yaml"
OVERLAY_ENV = "RIFT_CONFIG"

_overlay_path: Path | None = None


def set_overlay(path: str | Path | None) -> None:
    """Select an overlay file for this process (what --config does); clears the cache."""
    global _overlay_path
    _overlay_path = Path(path) if path else None
    load.cache_clear()


def _deep_merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else deepcopy(v)
    return out


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


@lru_cache(maxsize=1)
def load() -> dict:
    cfg = _read(DEFAULT_CONFIG_PATH)
    overlay = _overlay_path or (Path(os.environ[OVERLAY_ENV]) if os.getenv(OVERLAY_ENV) else None)
    if overlay:
        cfg = _deep_merge(cfg, _read(overlay))
        cfg["_overlay"] = str(overlay)
    return cfg


def section(name: str) -> dict:
    return load().get(name) or {}
