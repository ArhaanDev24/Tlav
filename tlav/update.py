"""Keeping Tlav up to date: a quick, cached check for a newer version, and `tlav update`.

Tlav never updates itself silently. Plain `tlav` asks once per new version; `tlav update`
does it on request. The check reads one tiny file (the version line) from the repository,
at most once a day, and gives up after a second and a half so a slow or missing connection
never delays the chat.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from . import __version__, config

VERSION_URL = os.environ.get("TLAV_VERSION_URL",
                             "https://raw.githubusercontent.com/ArhaanDev24/Tlav/main/tlav/__init__.py")
SOURCE = os.environ.get("TLAV_SOURCE", "https://github.com/ArhaanDev24/Tlav/archive/refs/heads/main.zip")
DAY = 24 * 3600


def parse(v: str) -> Tuple[int, ...]:
    """'0.10.2' -> (0, 10, 2), so 0.10 counts as newer than 0.9."""
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:4]) or (0,)


def newer(latest: Optional[str], current: str = __version__) -> bool:
    return bool(latest) and parse(latest) > parse(current)


def latest_version(timeout: float = 5.0) -> Optional[str]:
    """The newest published version, or None if it can't be read right now."""
    try:
        req = urllib.request.Request(VERSION_URL, headers={"User-Agent": "tlav/" + __version__})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read(4096).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - offline, blocked, slow: just don't know
        return None
    m = re.search(r"""__version__\s*=\s*["']([0-9][0-9A-Za-z.\-]*)["']""", text)
    return m.group(1) if m else None


def check(max_age: float = DAY, timeout: float = 1.5) -> Optional[str]:
    """A newer version than this one, or None. Uses yesterday's answer when it can."""
    if os.environ.get("TLAV_NO_UPDATE_CHECK"):
        return None
    st = config.load_state().get("update_check") or {}
    if time.time() - st.get("at", 0) < max_age:
        latest = st.get("latest")
    else:
        latest = latest_version(timeout) or st.get("latest")
        config.save_state(update_check={"at": time.time(), "latest": latest})
    return latest if newer(latest) else None


def cached_newer() -> Optional[str]:
    """A newer version we already know about (no network)."""
    latest = (config.load_state().get("update_check") or {}).get("latest")
    return latest if newer(latest) else None


def should_ask(latest: str) -> bool:
    """Ask about each new version once; after a 'no', only remind quietly."""
    return config.load_state().get("update_declined") != latest


def declined(latest: str) -> None:
    config.save_state(update_declined=latest)


def installed_by_installer() -> bool:
    """Is this the copy the one-line installer put in Tlav's folder (and can therefore replace)?"""
    from .remove import _installed_here
    return _installed_here()


def _pip_cmd() -> list:
    return [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
            "--upgrade", "--force-reinstall", SOURCE]


def _engine_cmd() -> list:
    return [sys.executable, "-m", "tlav", "install-runtime", "--if-changed"]


def windows_script(python: str, source: str) -> str:
    """The .cmd that finishes the update on Windows once this Tlav has exited (tlav.exe is locked
    while it runs)."""
    return "\r\n".join([
        "@echo off",
        "title Updating Tlav",
        "echo Updating Tlav...",
        "ping 127.0.0.1 -n 3 >nul",
        f'"{python}" -m pip install --quiet --disable-pip-version-check --upgrade --force-reinstall "{source}"',
        "if errorlevel 1 (",
        "  echo.",
        "  echo The update didn't finish. Check your internet connection and run: tlav update",
        ") else (",
        f'  "{python}" -m tlav install-runtime --if-changed',
        "  echo.",
        "  echo Tlav is up to date. Close this window and type: tlav",
        ")",
        "pause",
        "",
    ])


def run_update() -> bool:
    """Install the newest Tlav over this one. True when it finished here (False: finishing
    in its own window on Windows, or failed)."""
    if os.name == "nt":
        script = Path(tempfile.gettempdir()) / "tlav-update.cmd"
        script.write_text(windows_script(sys.executable, SOURCE), encoding="ascii", errors="replace")
        subprocess.Popen(["cmd", "/c", str(script)], creationflags=0x00000010,   # CREATE_NEW_CONSOLE
                         close_fds=True)
        print("  A small window is finishing the update. When it says it's done, type `tlav` again.")
        return False
    print("  Getting the newest Tlav...")
    if subprocess.run(_pip_cmd()).returncode != 0:
        print("  The update didn't finish. Check your internet connection and run `tlav update` again.")
        return False
    # the new version may pin a different AI engine: helpers and main PC must use the same one
    subprocess.run(_engine_cmd())
    config.save_state(update_declined=None, update_check={"at": 0, "latest": None})
    return True
