"""Deleting models (`tlav remove`) and removing Tlav completely (`tlav uninstall`).

Only Tlav's own files are touched: the models it downloaded or built, the AI engine it
compiled, its chats and settings, its startup services, the `tlav` command the installer
created and the PATH entry the Windows installer added. Python is left alone.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from . import config, server, ui

# Everything Tlav itself creates inside its home folder.
OWN_ENTRIES = ("models", "runtime", "venv", "web", "chats", "chat_history", "state.json",
               "discovered.json", "catalog.json", "server.log", "compress.log")


def _home() -> Path:
    """Tlav's folder, without creating it (config.home() would make it again)."""
    return Path(os.environ.get("TLAV_HOME", Path.home() / ".tlav")).expanduser()


def _size(p: Path) -> int:
    try:
        if p.is_symlink() or p.is_file():
            return p.lstat().st_size
        return sum(f.lstat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
    except OSError:
        return 0


def gb(n: int) -> str:
    if n < 1024 ** 2:
        return "under 1 MB"
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 * 0.1 else f"{n / 1024 ** 2:.0f} MB"


# ------------------------------------------------------------------ models

class ModelFile:
    """One downloaded or built model, with the helper files that belong to it."""

    def __init__(self, path: Path):
        self.path = path
        stem = path.name[:-len(".gguf")] if path.name.endswith(".gguf") else path.name.split(".gguf")[0]
        # importance matrices Tlav built for it, and an unfinished download of it
        self.extras = [p for pat in (".imatrix*", ".logits*") for p in path.parent.glob(stem + pat) if p != path]
        part = path.with_name(path.name + ".part")
        if part.exists() and part != path:
            self.extras.append(part)
        self.unfinished = path.name.endswith(".part")
        self.size = sum(_size(p) for p in [path] + self.extras)

    @property
    def name(self) -> str:
        return self.path.name[:-len(".part")] if self.unfinished else self.path.name


def model_files() -> List[ModelFile]:
    d = _home() / "models"
    if not d.is_dir():
        return []
    finished = sorted(p for p in d.glob("*.gguf") if ".imatrix" not in p.name)
    names = {p.name for p in finished}
    parts = sorted(p for p in d.glob("*.gguf.part") if p.name[:-len(".part")] not in names)
    return [ModelFile(p) for p in finished + parts]


def _active_paths() -> Tuple[Optional[str], Optional[str]]:
    act = config.load_state().get("active") or {}
    return act.get("path"), act.get("draft_path")


def match(files: List[ModelFile], wanted: str) -> List[ModelFile]:
    """A number from the list, a file name, or part of one (case and -/_/space don't matter)."""
    if wanted.isdigit() and 1 <= int(wanted) <= len(files):
        return [files[int(wanted) - 1]]
    norm = lambda s: s.lower().replace("_", "-").replace(" ", "-")
    exact = [f for f in files if norm(f.name) == norm(wanted)]
    return exact or [f for f in files if norm(wanted) in norm(f.name)]


def describe(files: List[ModelFile]) -> None:
    active, draft = _active_paths()
    for i, f in enumerate(files, 1):
        tag = ("in use" if str(f.path) == active else "speed helper" if str(f.path) == draft
               else "unfinished download" if f.unfinished else "")
        num = ui.c(ui.DIM, f"{i:>2}") if ui.ON else f"{i:>2}"
        print(f"  {num}  {f.name:<52} {gb(f.size):>9}  {ui.c(ui.ORANGE, tag) if (tag and ui.ON) else tag}")


def delete_models(chosen: List[ModelFile]) -> int:
    """Delete these models. If the one in use goes, stop it and forget it. Returns bytes freed."""
    active, draft = _active_paths()
    paths = {str(f.path) for f in chosen}
    if active in paths and server.status().get("running"):
        server.stop()
    freed, busy = 0, []
    for f in chosen:
        for p in [f.path] + f.extras:
            n = _size(p)
            try:
                p.unlink()
                freed += n
            except FileNotFoundError:
                pass
            except OSError:  # Windows: a program still has it open
                busy.append(p.name)
    if busy:
        raise RuntimeError("Some files are in use and were not deleted: " + ", ".join(busy)
                           + ". Run `tlav stop`, close any chat using them, then try again.")
    state = config.load_state()
    act = state.get("active")
    if act and act.get("path") in paths:
        config.save_state(active=None)
    elif act and act.get("draft_path") in paths:
        act["draft_path"] = None
        config.save_state(active=act)
    return freed


# ------------------------------------------------------------------ uninstall

def _installed_here() -> bool:
    """Is the running `tlav` the installer's copy inside Tlav's folder?"""
    try:
        return Path(sys.prefix).resolve().is_relative_to(_home().resolve())  # type: ignore[attr-defined]
    except AttributeError:  # Python < 3.9
        try:
            Path(sys.prefix).resolve().relative_to(_home().resolve())
            return True
        except ValueError:
            return False
    except OSError:
        return False


def _launcher() -> Optional[Path]:
    """~/.local/bin/tlav, if it's the installer's link into Tlav's folder."""
    link = Path.home() / ".local" / "bin" / "tlav"
    if not link.is_symlink():
        return None
    target = Path(os.readlink(str(link)))
    try:
        target.relative_to(_home())
        return link
    except ValueError:
        return link if not target.exists() and "tlav" in str(target) else None


def plan() -> List[Tuple[str, List[Path], int]]:
    """What `tlav uninstall` would remove: (label, paths, bytes)."""
    home = _home()
    out = []
    models = home / "models"
    if models.exists():
        out.append(("Models", [models], _size(models)))
    if (home / "runtime").exists():
        out.append(("AI engine (llama.cpp built for this PC)", [home / "runtime"], _size(home / "runtime")))
    data = [home / n for n in ("chats", "chat_history", "web", "state.json", "discovered.json", "catalog.json",
                                "server.log", "compress.log") if (home / n).exists()]
    if data:
        out.append(("Chats, settings and logs", data, sum(_size(p) for p in data)))
    prog = [home / "venv"] if (home / "venv").exists() else []
    link = _launcher()
    if link:
        prog.append(link)
    if prog:
        out.append(("The tlav program", prog, sum(_size(p) for p in prog)))
    return out


def _remove_services() -> List[str]:
    from . import service
    done = []
    for role in ("server", "worker"):
        try:
            if os.name == "nt":
                q = subprocess.run(["schtasks", "/Query", "/TN", f"Tlav-{role}"], capture_output=True)
                if q.returncode != 0:
                    continue
            else:
                paths = service._paths(role)
                if not any(Path(p).exists() for p in paths.values()):
                    continue
            done.append(service.uninstall(role))
        except Exception:  # noqa: BLE001 - never let a service problem stop the uninstall
            pass
    return done


def _windows_path_cleanup(scripts: Path) -> bool:
    """Take Tlav's folder off the user's PATH (the Windows installer put it there)."""
    try:
        import winreg  # type: ignore
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE) as k:
            value, kind = winreg.QueryValueEx(k, "Path")
            parts = [p for p in value.split(";") if p and Path(p.rstrip("\\/")) != scripts]
            if len(parts) == len([p for p in value.split(";") if p]):
                return False
            winreg.SetValueEx(k, "Path", 0, kind, ";".join(parts))
        try:  # tell open programs the PATH changed
            import ctypes
            ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x1A, 0, "Environment", 2, 1000, None)
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception:  # noqa: BLE001
        return False


def uninstall(keep_models: bool = False) -> List[str]:
    """Remove Tlav. Returns notes for the user."""
    home = _home()
    notes = []
    if server.status().get("running"):
        server.stop()
        notes.append("Stopped the running model.")
    notes += _remove_services()

    keep = {"models"} if keep_models else set()
    running_inside = _installed_here()
    later = []  # Windows can't delete the program that is running right now
    for name in OWN_ENTRIES:
        p = home / name
        if name in keep or not (p.exists() or p.is_symlink()):
            continue
        if name == "venv" and running_inside and os.name == "nt":
            later.append(p)
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except OSError:
                pass

    link = _launcher()
    if link:
        try:
            link.unlink()
        except OSError:
            pass
    if os.name == "nt" and _windows_path_cleanup(home / "venv" / "Scripts"):
        notes.append("Removed Tlav from your PATH.")

    # The default folder is entirely Tlav's; a custom TLAV_HOME only loses Tlav's own files.
    default = home == (Path.home() / ".tlav")
    if not keep_models and not later and home.exists():
        if default:
            shutil.rmtree(home, ignore_errors=True)
        else:
            try:
                home.rmdir()
            except OSError:
                pass
    if later:
        # finish once this program has exited (a few seconds), in the background
        target = home if (default and not keep_models) else later[0]
        subprocess.Popen(f'cmd /c ping 127.0.0.1 -n 4 >nul & rmdir /s /q "{target}"', shell=True,
                         creationflags=0x00000008 | 0x00000200, close_fds=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        notes.append("The last files are deleted a few seconds after this window returns to the prompt.")
    if keep_models:
        notes.append(f"Your models were kept in {home / 'models'}.")
    if not running_inside:
        notes.append("This tlav was installed with pip, not the installer. Finish with:  pip uninstall tlav")
    return notes
