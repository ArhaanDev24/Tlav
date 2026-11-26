"""Discover new models on Hugging Face, so Tlav knows about releases without a code update.

Only *original* models from known AI labs are considered (no merges, role-play
fine-tunes or "uncensored" edits), quantized by publishers whose GGUF files are
consistently correct. Each one is measured from its real file header
(parameters, layers, context cache size, reasoning mode) and given a quality
score estimated from the hand-scored catalog (see catalog.estimate_quality).
"""

import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from . import catalog, config, finder, gguf
from .hub import HF, _request

# GGUF publishers, best first (when two publish the same model, the first wins).
QUANTIZERS = ["bartowski", "unsloth", "ggml-org", "lmstudio-community"]

# Organisations that train and release original models.
LABS = {
    "qwen", "google", "meta-llama", "microsoft", "mistralai", "huggingfacetb", "liquidai", "ibm-granite",
    "deepseek-ai", "openai", "nvidia", "allenai", "tiiuae", "zai-org", "thudm", "moonshotai",
    "minimaxai", "baidu", "tencent", "internlm", "cohereforai", "coherelabs", "ai21labs",
    "stepfun-ai", "xiaomimimo", "inclusionai", "swiss-ai", "arcee-ai", "servicenow-ai", "kwaipilot",
}

# Words that mark edited or niche models even when the base model is official.
SKIP = re.compile(r"abliterat|uncensor|heretic|nsfw|roleplay|-rp\b|merge|distill|preview|"
                  r"-base\b|-pt\b|guard|embed|rerank|reward|-vl\b|vlm|omni|video|vision|\dv\b|-v-|"
                  r"ocr|tts|asr|audio|speech|-mtp\b|math|qat|unquantized|-1m\b", re.I)
CHAT_SUFFIX = re.compile(r"[-_](it|instruct|chat)([-_]\d+)?$", re.I)

TEXT_PIPELINES = {None, "text-generation", "image-text-to-text", "conversational"}
MAX_PARAMS_B = 130          # bigger than any cluster of old PCs can hold
REFRESH_DAYS = 7


def _get(url: str):
    with urllib.request.urlopen(_request(url), timeout=30) as r:
        return json.load(r)


def base_of(tags: List[str]) -> Optional[str]:
    for t in tags:
        if t.startswith("base_model:quantized:"):
            return t.split(":", 2)[2]
    for t in tags:
        if t.startswith("base_model:") and t.count(":") == 1:
            return t.split(":", 1)[1]
    return None


def acceptable(repo: dict) -> Optional[str]:
    """The base model id if this GGUF repo is an official, general-purpose model."""
    if repo.get("pipeline_tag") not in TEXT_PIPELINES:
        return None
    base = base_of(repo.get("tags", []))
    if not base or "/" not in base or base.split("/")[0].lower() not in LABS:
        return None
    if SKIP.search(base) or SKIP.search(repo["id"]):
        return None
    return base


def list_candidates(per_publisher: int = 100, lister=None) -> Dict[str, dict]:
    """base model -> best GGUF repo listing, across the trusted publishers."""
    lister = lister or _get
    found: Dict[str, dict] = {}
    for author in QUANTIZERS:
        q = urllib.parse.urlencode({"author": author, "filter": "gguf", "sort": "downloads", "direction": "-1",
                                    "limit": str(per_publisher)})
        try:
            repos = lister(f"{HF}/api/models?{q}&expand[]=tags&expand[]=createdAt&expand[]=downloads"
                           f"&expand[]=pipeline_tag")
        except Exception:
            continue
        for r in repos:
            base = acceptable(r)
            if base and base.lower() not in found:
                found[base.lower()] = {"repo": r["id"], "base": base, "created": (r.get("createdAt") or "")[:7],
                                       "downloads": r.get("downloads", 0)}
    # A base (pre-chat) model whose chat version is also listed: keep only the chat version.
    names = {b for b in found}
    for b in list(found):
        if not CHAT_SUFFIX.search(b) and any(n != b and n.startswith(b + "-") and CHAT_SUFFIX.search(n)
                                             for n in names):
            found.pop(b)
    return found


def _clean(name: str) -> str:
    return re.sub(r"[-_](bf16|f16|fp8|gguf)$", "", name, flags=re.I)


def _slug(base: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", _clean(base.split("/")[-1]).lower()).strip("-")


def _pretty(base: str) -> str:
    name = _clean(base.split("/")[-1])
    name = re.sub(r"[-_](it|instruct|chat)$", "", name, flags=re.I)
    return name.replace("-", " ").replace("_", " ")


def measure(entry: dict) -> Optional[dict]:
    """Read the real file header of one candidate and build a catalog entry."""
    try:
        files = finder.repo_files(entry["repo"])
        if not any(q in files for q in ("Q4_K_M", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q8_0")):
            return None
        fn = min(files.values(), key=lambda x: x[1])[0]
        info = gguf.read_remote(entry["repo"], fn)
    except Exception:
        return None
    pb = info.n_params / 1e9
    if pb > MAX_PARAMS_B or not info.layers:
        return None
    moe = bool(info.expert_count)
    ab = info.active_params / 1e9 if moe else None
    q = catalog.estimate_quality(ab if moe else pb, moe_total_b=pb if moe else None, released=entry["created"])
    tags = ["chat"] + (["reasoning"] if info.reasons else [])
    if re.search(r"coder|code", entry["base"], re.I):
        tags = ["code"] + tags[1:]
    return {"id": _slug(entry["base"]), "name": _pretty(entry["base"]), "repo": entry["repo"],
            "params_b": round(pb, 3), "layers": info.layers, "kv_heads": info.kv_heads, "head_dim": info.head_dim,
            "max_ctx": info.context_length,
            "quality": {"chat": q - (6 if "code" in tags else 0), "code": q if "code" in tags else q - 3},
            "tags": tags, "active_params_b": round(ab, 3) if ab else None,
            "stream_params_b": round(info.stream_params / 1e9, 4), "released": entry["created"],
            "base": entry["base"], "estimated": True, "downloads": entry["downloads"]}


def update(progress=print) -> List[dict]:
    """Find, measure and save new models. Returns the discovered entries."""
    known = {(m.base or "").lower() for m in catalog.load(include_discovered=False)}
    cands = [c for b, c in list_candidates().items() if b not in known]
    progress(f"  {len(cands)} new original models from trusted publishers; reading their file headers...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        found = [m for m in ex.map(measure, cands) if m]
    ids = set()
    out = []
    for m in sorted(found, key=lambda m: -m["downloads"]):
        if m["id"] not in ids:
            ids.add(m["id"])
            out.append(m)
    (config.home() / "discovered.json").write_text(json.dumps({"updated": time.time(), "models": out}, indent=1))
    return out


def age_days() -> Optional[float]:
    path = config.home() / "discovered.json"
    if not path.exists():
        return None
    try:
        return (time.time() - json.loads(path.read_text()).get("updated", 0)) / 86400
    except (OSError, ValueError):
        return None


def refresh_if_stale(progress=print) -> None:
    """Quietly keep the discovered list fresh (weekly); never fails setup."""
    age = age_days()
    if age is not None and age < REFRESH_DAYS:
        return
    progress("  Looking for newly released models on Hugging Face...")
    try:
        n = len(update(progress))
        progress(f"  {n} new models added to the search.")
    except Exception as e:
        progress(f"  (could not refresh the model list: {str(e)[:60]})")
