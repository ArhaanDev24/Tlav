"""Paths and persisted state under ~/.tlav (override with TLAV_HOME)."""

import json
import os
import time
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
    # write a new file and swap it in: a crash or power cut never leaves a half-written state.json
    path = _state_file()
    tmp = path.with_name(f".state.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:          # Windows: another Tlav process has it open for a moment
            if attempt == 19:
                raise
            time.sleep(0.05)
    return state
