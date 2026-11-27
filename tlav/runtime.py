"""Locate or install the llama.cpp runtime.

Search order: $TLAV_LLAMA_BIN, ~/.tlav/runtime (installed by Tlav), $PATH
(e.g. `brew install llama.cpp`). Installing prefers building from source with
flags tuned for *this* CPU (GGML_NATIVE picks up AVX/AVX2/FMA/F16C), which is
also the only reliable option on old macOS versions that prebuilt binaries no
longer support. Linux/Windows can use prebuilt releases instead.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .hardware import Hardware

TOOLS = ["llama-server", "llama-cli", "llama-quantize", "llama-perplexity", "llama-bench", "llama-imatrix",
         "rpc-server"]
# Upstream renamed some binaries over time; accept every known name.
ALIASES = {"rpc-server": ["rpc-server", "ggml-rpc-server"]}
REPO = "ggml-org/llama.cpp"
EXE = ".exe" if os.name == "nt" else ""


class RuntimeMissing(RuntimeError):
    pass


def _bin_dirs() -> List[Path]:
    dirs = []
    if os.environ.get("TLAV_LLAMA_BIN"):
        dirs.append(Path(os.environ["TLAV_LLAMA_BIN"]))
    saved = config.load_state().get("runtime_bin")
    if saved:
        dirs.append(Path(saved))
    dirs.append(config.runtime_dir() / "bin")
    return dirs


def find(tool: str) -> Optional[str]:
    names = ALIASES.get(tool, [tool])
    for d in _bin_dirs():
        for n in names:
            p = d / (n + EXE)
            if p.exists():
                return str(p)
    for n in names:
        w = shutil.which(n)
        if w:
            return w
    return None


def require(tool: str) -> str:
    p = find(tool)
    if not p:
        raise RuntimeMissing(f"'{tool}' not found. Run `tlav install-runtime` first.")
    return p


_HELP_CACHE: Dict[str, str] = {}


def help_text(tool: str) -> str:
    """`tool --help` output (cached), used to adapt to the installed llama.cpp version."""
    if tool not in _HELP_CACHE:
        exe = find(tool)
        text = ""
        if exe:
            try:
                res = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=30)
                text = res.stdout + res.stderr
            except (OSError, subprocess.SubprocessError):
                pass
        _HELP_CACHE[tool] = text
    return _HELP_CACHE[tool]


_ENGINE: Optional[str] = None


def engine_version() -> str:
    """The installed llama.cpp build (its git commit), e.g. '8c64180'; '' if unknown.
    Computers in a cluster must run the same build: the network protocol changes between them."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ""
        exe = find("llama-server") or find("llama-cli")
        if exe:
            try:
                out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
                m = re.search(r"commit\s+([0-9a-f]{6,})", out.stdout + out.stderr)
                _ENGINE = m.group(1) if m else ""
            except (OSError, subprocess.SubprocessError):
                pass
    return _ENGINE


def status() -> Dict[str, Optional[str]]:
    return {t: find(t) for t in TOOLS}


def cmake_flags(hw: Hardware) -> List[str]:
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DBUILD_SHARED_LIBS=OFF",
             "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DGGML_NATIVE=ON",
             "-DGGML_RPC=ON"]  # RPC lets several machines share one model (tlav worker / cluster)
    if hw.os_name == "macOS":
        flags.append("-DGGML_METAL=" + ("ON" if hw.arch == "arm64" else "OFF"))
        if hw.arch != "arm64":
            flags.append("-DGGML_BLAS=OFF")  # Accelerate on old macOS is slower than ggml's AVX2 kernels
    if hw.gpu == "cuda" and shutil.which("nvcc"):
        flags.append("-DGGML_CUDA=ON")
    return flags


def _ensure_cmake() -> str:
    c = shutil.which("cmake")
    if c:
        return c
    print("cmake not found; installing it with pip (works on old macOS too)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "cmake"])
    c = shutil.which("cmake")
    if c:
        return c
    return str(Path(sys.executable).parent / "cmake")


def _fetch_source(dest: Path) -> Path:
    src = dest / "llama.cpp"
    if src.exists():
        return src
    if shutil.which("git"):
        subprocess.check_call(["git", "clone", "--depth", "1", f"https://github.com/{REPO}", str(src)])
        return src
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/master"
    tgz = dest / "llama.tar.gz"
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(dest)  # trusted upstream archive
    tgz.unlink()
    (dest / "llama.cpp-master").rename(src)
    return src


def _target_name(src: Path, tool: str) -> str:
    """CMake target for a tool in this llama.cpp checkout (handles upstream renames)."""
    if tool == "rpc-server":
        cm = src / "tools" / "rpc" / "CMakeLists.txt"
        if cm.exists():
            m = re.search(r"set\(TARGET\s+([\w-]+)\)", cm.read_text())
            if m:
                return m.group(1)
    return tool


def build_from_source(hw: Hardware, jobs: Optional[int] = None) -> Path:
    rt = config.runtime_dir()
    src = _fetch_source(rt)
    cmake = _ensure_cmake()
    build = src / "build"
    print("Configuring llama.cpp with:", " ".join(cmake_flags(hw)))
    subprocess.check_call([cmake, "-S", str(src), "-B", str(build)] + cmake_flags(hw))
    jobs = jobs or max(1, hw.logical_cores)
    targets = [_target_name(src, t) for t in TOOLS]
    subprocess.check_call([cmake, "--build", str(build), "--config", "Release", "-j", str(jobs),
                           "--target"] + targets)
    out = rt / "bin"
    out.mkdir(exist_ok=True)
    for t in targets:
        for cand in (build / "bin" / (t + EXE), build / "bin" / "Release" / (t + EXE)):
            if cand.exists():
                shutil.copy2(cand, out / cand.name)
    config.save_state(runtime_bin=str(out), runtime_source="source")
    return out


def _asset_keywords(hw: Hardware) -> Optional[List[str]]:
    if hw.os_name == "Linux" and hw.arch == "x86_64":
        return ["bin-ubuntu-x64"]
    if hw.os_name == "Windows" and hw.arch == "x86_64":
        return ["bin-win-cpu-x64", "bin-win-avx2-x64"]
    if hw.os_name == "macOS" and hw.arch == "arm64":
        return ["bin-macos-arm64"]
    return None  # Intel macOS and others: build from source


def install_prebuilt(hw: Hardware) -> Optional[Path]:
    keys = _asset_keywords(hw)
    if not keys:
        return None
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/latest",
                                 headers={"User-Agent": "tlav", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        release = json.load(r)
    asset = None
    for k in keys:
        asset = next((a for a in release.get("assets", []) if k in a["name"]
                      and a["name"].endswith((".zip", ".tar.gz"))), None)
        if asset:
            break
    if not asset:
        return None
    rt = config.runtime_dir()
    archive = rt / asset["name"]
    print(f"Downloading {asset['name']} ...")
    urllib.request.urlretrieve(asset["browser_download_url"], archive)
    target = rt / "prebuilt"
    shutil.rmtree(target, ignore_errors=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target)
    archive.unlink()
    server = next(target.rglob("llama-server" + EXE), None)
    if not server:
        return None
    for p in server.parent.iterdir():
        if p.is_file() and os.name != "nt":
            p.chmod(p.stat().st_mode | 0o111)
    config.save_state(runtime_bin=str(server.parent), runtime_source="prebuilt:" + release.get("tag_name", ""))
    return server.parent


def install(hw: Hardware, prefer: str = "auto") -> Path:
    if prefer in ("auto", "prebuilt"):
        try:
            p = install_prebuilt(hw)
            if p:
                return p
        except Exception as e:  # network/API problems: fall back to building
            print(f"Prebuilt download unavailable ({e}); building from source instead.")
        if prefer == "prebuilt":
            raise RuntimeMissing("No prebuilt llama.cpp for this platform.")
    return build_from_source(hw)


def platform_note(hw: Hardware) -> str:
    if hw.os_name == "macOS" and hw.arch == "x86_64":
        return ("Intel Mac: Tlav builds llama.cpp from source with AVX2 and Metal disabled. "
                "You need Apple's command line tools: run `xcode-select --install` once.")
    return ""
