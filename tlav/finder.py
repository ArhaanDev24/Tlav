"""Find GGUF models on Hugging Face and judge how well they fit this machine."""

import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import catalog, gguf, planner
from .hub import HF, _request

QUANT_RE = re.compile(r"[-.](I?Q\d[A-Z0-9_]*|MXFP4[A-Z0-9_]*|NVFP4[A-Z0-9_]*|F16|BF16)\.gguf$", re.I)
TEXT_PIPELINES = {None, "text-generation", "image-text-to-text", "conversational"}


@dataclass
class Found:
    repo: str
    downloads: int
    likes: int
    files: Dict[str, int]                 # quant -> size in bytes (single-file GGUFs only)
    info: Optional[gguf.GGUFInfo] = None
    plan: Optional[planner.Plan] = None
    error: str = ""
    created: str = ""                     # YYYY-MM, sharpens the quality estimate

    @property
    def model(self) -> Optional[catalog.Model]:
        if not self.info:
            return None
        import dataclasses
        m = _named(self.info.to_model(self.repo.split("/")[-1].lower(), self.repo), self.repo)
        if self.created:
            moe = self.info.expert_count
            q = catalog.estimate_quality(m.active_params_b if moe else m.params_b,
                                         moe_total_b=m.params_b if moe else None, released=self.created)
            m = dataclasses.replace(m, quality={"chat": q, "code": q - 3}, released=self.created, estimated=True)
        return m


def _named(model: catalog.Model, repo: str) -> catalog.Model:
    """Use the repository name ('Qwen3-8B') instead of odd internal GGUF names."""
    import dataclasses
    name = re.sub(r"[-_]GGUF$", "", repo.split("/")[-1], flags=re.I).replace("_", " ")
    return dataclasses.replace(model, name=name)


def search_repos(query: str, limit: int = 20) -> List[dict]:
    q = urllib.parse.urlencode({"search": query, "filter": "gguf", "sort": "downloads",
                                "direction": "-1", "limit": str(limit)})
    with urllib.request.urlopen(_request(f"{HF}/api/models?{q}"), timeout=30) as r:
        return json.load(r)


def repo_files(repo: str) -> Dict[str, tuple]:
    """quant name -> (filename, size bytes) for single-file GGUFs in the repo root."""
    with urllib.request.urlopen(_request(f"{HF}/api/models/{repo}/tree/main"), timeout=30) as r:
        return parse_tree(json.load(r))


def parse_tree(tree: list) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for item in tree:
        path = item.get("path", "")
        if (item.get("type") != "file" or "/" in path or re.search(r"-\d{5}-of-\d{5}", path)
                or "mmproj" in path.lower()):  # skip split files and vision projectors
            continue
        m = QUANT_RE.search(path)
        if not m:
            continue
        base = path[: m.start()]
        if re.search(r"[-.](I?Q\d\w*|b?f16)$", base, re.I):
            continue  # mixed files like X-Q8_0-f16.gguf
        size = item.get("lfs", {}).get("size") or item.get("size", 0)
        quant = m.group(1).upper()
        if quant not in out or len(path) < len(out[quant][0]):
            out[quant] = (path, size)
    return out


def inspect(repo: str, nodes: List[planner.Node], task: str = "chat", min_tps: float = 2.5,
            min_quant: str = catalog.DEFAULT_MIN_QUANT, created: str = "") -> Found:
    f = Found(repo, 0, 0, {}, created=created)
    try:
        files = repo_files(repo)
        f.files = {q: s for q, (_, s) in files.items()}
        if not files:
            f.error = "no single-file GGUF"
            return f
        smallest = min(files.values(), key=lambda x: x[1])[0]
        f.info = gguf.read_remote(repo, smallest)
        model = f.model
        quants = [catalog.quant_from_file(size, model.params_b, q) for q, (_, size) in files.items()]
        f.plan = planner.plan_for_model(model, nodes, task, min_tps, min_quant, quants=quants)
    except Exception as e:  # network errors, odd repos: report and move on
        f.error = str(e)[:80]
    return f


def search(query: str, nodes: List[planner.Node], limit: int = 10, **kw) -> List[Found]:
    repos = [r for r in search_repos(query, limit=limit * 2)
             if r.get("pipeline_tag") in TEXT_PIPELINES][:limit]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda r: inspect(r["id"], nodes, created=(r.get("createdAt") or "")[:7], **kw),
                              repos))
    for r, meta in zip(results, repos):
        r.downloads, r.likes = meta.get("downloads", 0), meta.get("likes", 0)
    # fitting models first, then by quality estimate, then popularity
    results.sort(key=lambda x: (x.plan is not None, round(x.plan.quality) if x.plan else 0, x.downloads),
                 reverse=True)
    return results


def model_from_repo(repo: str) -> tuple:
    """(Model, {quant: filename}) for `tlav setup --hf user/repo`."""
    files = repo_files(repo)
    if not files:
        raise ValueError(f"{repo} has no single-file GGUF models")
    smallest = min(files.values(), key=lambda x: x[1])[0]
    info = gguf.read_remote(repo, smallest)
    model = _named(info.to_model(repo.split("/")[-1].lower(), repo), repo)
    return model, {q: fn for q, (fn, _) in files.items()}


# ------------------------------------------------------------------ refine catalog picks with real files

@dataclass
class Checked:
    plan: Optional[planner.Plan]          # re-planned with the real file sizes (None: dropped)
    estimate: planner.Plan                # the plan made from Tlav's size estimates
    note: str = ""                        # why it was dropped, if it was


def refine(plans: List[planner.Plan], nodes: List[planner.Node], task: str = "chat", min_tps: float = 2.5,
           min_quant: str = catalog.DEFAULT_MIN_QUANT, kv: str = planner.DEFAULT_KV,
           lister=None) -> List[Checked]:
    """Re-check shortlisted catalog models against the files that really exist on Hugging Face.

    Only the repository's file list is fetched (a few KB per model), so the
    pick uses exact download sizes without downloading any weights.
    """
    lister = lister or repo_files

    def one(p: planner.Plan) -> Checked:
        try:
            files = lister(p.model.repo)
        except Exception as e:  # offline, renamed repo, rate limit
            return Checked(None, p, f"could not check ({str(e)[:40]})")
        quants = [catalog.quant_from_file(size, p.model.params_b, q)
                  for q, (_, size) in files.items() if catalog.retention_for(q) is not None]
        if not quants:
            return Checked(None, p, "no ready-made files")
        real = planner.plan_for_model(p.model, nodes, task, min_tps, min_quant, kv, quants)
        return Checked(real, p, "" if real else "real files are too big or too slow here")

    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(one, plans))


def why(plan: planner.Plan, runner_up: Optional[planner.Plan], min_tps: float) -> str:
    """One plain sentence on why this model won."""
    where = f"{len(plan.nodes)} computers" if plan.is_cluster else "this PC"
    s = (f"highest quality that fits {where} at ≥{min_tps:g} tokens/s; "
         f"{plan.quant.name} keeps ~{plan.quant.retention * 100:.1f}% of the original")
    if runner_up and round(runner_up.quality) == round(plan.quality):
        s += f"; as good as {runner_up.model.name} but faster"
    return s
