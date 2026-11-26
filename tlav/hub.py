"""Minimal Hugging Face downloader (stdlib only, resumable)."""

import json
import os
import re
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


def download(repo: str, filename: str, dest_dir: Optional[Path] = None, quiet: bool = False) -> Path:
    dest_dir = dest_dir or config.models_dir()
    dest = dest_dir / filename
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = _request(f"{HF}/{repo}/resolve/main/{filename}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored the range: start over
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        t0, done = time.time(), have
        with open(part, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if not quiet and total:
                    speed = (done - have) / max(time.time() - t0, 1e-6)
                    sys.stderr.write(f"\r  {filename}: {_fmt(done)} / {_fmt(total)} "
                                     f"({100 * done / total:.0f}%, {_fmt(speed)}/s)   ")
                    sys.stderr.flush()
    if not quiet:
        sys.stderr.write("\n")
    if total and part.stat().st_size != total:
        raise IOError(f"Download incomplete ({part.stat().st_size} of {total} bytes); rerun to resume.")
    part.rename(dest)
    return dest
