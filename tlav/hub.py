"""Minimal Hugging Face downloader (stdlib only, resumable)."""

import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from . import config

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def _request(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "tlav"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def list_files(repo: str) -> List[str]:
    with urllib.request.urlopen(_request(f"{HF}/api/models/{repo}"), timeout=30) as r:
        data = json.load(r)
    return [s["rfilename"] for s in data.get("siblings", [])]


def find_quant_file(files: List[str], quant: str) -> Optional[str]:
    """Pick the single-file GGUF for a quant, e.g. 'X-Q4_K_M.gguf' (not 'X-Q4_K_M-fp16.gguf')."""
    pat = re.compile(r"^(.*)[-.]" + re.escape(quant) + r"\.gguf$", re.I)
    # reject mixed files such as 'X-Q8_0-f16.gguf' (Q8_0 weights, f16 embeddings)
    other_quant = re.compile(r"[-.](I?Q\d\w*|b?f16|f32)$", re.I)
    hits = []
    for f in files:
        m = pat.match(f)
        if m and "/" not in f and not other_quant.search(m.group(1)):
            hits.append(f)
    if not hits:
        return None
    return sorted(hits, key=len)[0]


def find_source_file(files: List[str]) -> Optional[tuple]:
    """Highest-precision file available to compress from: (filename, quant)."""
    for q in ("F16", "BF16", "f16", "bf16", "Q8_0"):
        f = find_quant_file(files, q)
        if f:
            return f, q.upper()
    return None


def _fmt(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB" if n > 1024 ** 3 else f"{n / 1024 ** 2:.0f} MB"


def download(repo: str, filename: str, dest_dir: Optional[Path] = None, quiet: bool = False,
             retries: int = 4) -> Path:
    """Download with resume. Flaky connections are retried automatically (continuing where they
    stopped); a file that isn't a valid model is fetched again."""
    dest_dir = dest_dir or config.models_dir()
    dest = dest_dir / filename
    if dest.exists():
        if filename.endswith(".gguf") and not _looks_like_gguf(dest):
            dest.unlink()   # damaged or not a model (e.g. an error page saved by mistake): fetch again
        else:
            return dest
    import socket
    import urllib.error
    for attempt in range(retries + 1):
        try:
            return _download_once(repo, filename, dest, quiet)
        except urllib.error.HTTPError:
            raise                      # missing/private file: retrying won't help
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, IOError) as e:
            if attempt == retries or getattr(e, "errno", None) == 28:   # 28 = disk full
                raise
            wait = 2 ** (attempt + 1)
            if not quiet:
                sys.stderr.write(f"\n  connection dropped ({str(e)[:60]}); continuing in {wait}s...\n")
            time.sleep(wait)
    return dest


# Set by the setup screen to draw its own progress bar: PROGRESS(done_bytes, total_bytes, bytes_per_s)
PROGRESS = None


def _looks_like_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF" and path.stat().st_size > 1024 * 1024
    except OSError:
        return False


def _download_once(repo: str, filename: str, dest: Path, quiet: bool) -> Path:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = _request(f"{HF}/{repo}/resolve/main/{filename}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored the range: start over
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        need = total - have
        free = shutil.disk_usage(dest.parent).free
        if need and free < need + 200 * 1024 ** 2:
            raise OSError(28, f"Not enough disk space for {filename}: it needs {_fmt(need)} more, "
                              f"only {_fmt(free)} is free. Free some space, or pick a smaller model.")
        mode = "ab" if have else "wb"
        t0, done, last = time.time(), have, 0.0
        tty = sys.stderr.isatty()
        with open(part, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if PROGRESS and total and (now - last > 0.2 or done == total):
                    last = now
                    PROGRESS(done, total, (done - have) / max(now - t0, 1e-6))
                elif not quiet and not PROGRESS and total and (now - last > (0.25 if tty else 10) or done == total):
                    last = now
                    speed = (done - have) / max(time.time() - t0, 1e-6)
                    sys.stderr.write(("\r" if tty else "") + f"  {filename}: {_fmt(done)} / {_fmt(total)} "
                                     f"({100 * done / total:.0f}%, {_fmt(speed)}/s)   " + ("" if tty else "\n"))
                    sys.stderr.flush()
    if not quiet and not PROGRESS:
        sys.stderr.write("\n")
    if total and part.stat().st_size != total:
        raise IOError(f"Download incomplete ({part.stat().st_size} of {total} bytes); rerun to resume.")
    part.rename(dest)
    return dest
