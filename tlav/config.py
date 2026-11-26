"""Paths and persisted state under ~/.tlav (override with TLAV_HOME)."""

import json
import os
from pathlib import Path


def home() -> Path:
    p = Path(os.environ.get("TLAV_HOME", Path.home() / ".tlav"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def runtime_dir() -> Path:
    p = home() / "runtime"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_file() -> Path:
    return home() / "state.json"


def load_state() -> dict:
    try:
        return json.loads(_state_file().read_text())
    except (OSError, ValueError):
        return {}


def save_state(**updates) -> dict:
    state = load_state()
    state.update(updates)
    _state_file().write_text(json.dumps(state, indent=2))
    return state
