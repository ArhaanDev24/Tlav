"""Tlav's own web chat, served by llama-server (`--path`) from ~/.tlav/web.

The page lives in the package (tlav/web/index.html). Before each start Tlav
copies it next to a small tlav.json describing the model, so the browser chat
shows the same name, speed and old-computer status lines as the terminal.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from . import __version__, config

PAGE = Path(__file__).parent / "web" / "index.html"


def prepare(info: Optional[dict] = None) -> Optional[Path]:
    """Write ~/.tlav/web (page + tlav.json). Returns the folder, or None if the page is missing."""
    if not PAGE.exists():
        return None
    from .chat import VERBS, default_think
    out = config.home() / "web"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAGE, out / "index.html")
    info = dict(info or {})
    data = {"model": info.get("model", ""), "quant": info.get("quant", ""), "ctx": int(info.get("ctx") or 4096),
            "est_tps": float(info.get("est_tps") or 0), "reasons": bool(info.get("reasons", False)),
            "where": info.get("where_long") or ("your computers" if info.get("where", "").endswith("computers")
                                                else "this computer"),
            "think_default": default_think(info), "verbs": VERBS, "version": __version__}
    (out / "tlav.json").write_text(json.dumps(data, indent=1))
    return out
