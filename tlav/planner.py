"""Pick the best model + compression level + settings for one machine or a cluster.

For every model and every quantization level we estimate:
  memory = weights + KV cache(context) + runtime overhead
  speed  = usable memory bandwidth / bytes read per token
and keep the combos that fit and are fast enough to be usable. The winner is
the highest quality after compression, so a big model compressed to Q4_K_M
beats a small model kept at Q8_0 when that is better.

Clusters: llama.cpp's RPC backend splits a model's layers across machines, so
their RAM adds up. Each token still flows through every machine in turn, so
the time per token is the *sum* of each machine's share plus network delay.
Clustering lets you run a bigger model; it does not make a model faster.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from . import catalog
from .catalog import KV_BYTES, Model, Quant
from .hardware import Hardware

RUNTIME_OVERHEAD_GB = 0.4      # llama.cpp buffers, compute graph (per machine, cluster plans)
CONTEXT_CHOICES = (8192, 4096, 2048, 1024)
MIN_CTX = 1024                 # 1024 tokens is still a usable chat on a 2 GB machine


def overhead_gb(model: Model) -> float:
    """Compute buffers scale with model size: tiny models need far less than 0.4 GB."""
    return min(RUNTIME_OVERHEAD_GB, 0.15 + 0.05 * model.params_b)
RPC_ROUNDTRIPS_PER_TOKEN = 4   # graph compute + tensor get/set per remote machine per token
RPC_FIXED_MS_PER_TOKEN = 2.0   # serialization/scheduling cost per remote machine per token
DEFAULT_KV = "q8_0"            # halves context memory; quality impact is negligible (verified by --kv-check)


def ram_budget_gb(hw: Hardware) -> float:
    """RAM we allow the model to use, leaving room for the OS and a browser."""
    if hw.gpu == "cuda" and hw.gpu_vram_gb:
        # with CUDA we offload to VRAM and spill the rest into RAM
        return hw.gpu_vram_gb * 0.9 + max(0.0, hw.ram_total_gb * 0.6 - 2.0)
    if hw.ram_total_gb >= 6:
        reserve = max(2.0, hw.ram_total_gb * 0.25)
    else:
        # Small machines (2-4 GB netbooks, old laptops) usually run a light OS:
        # keep ~30% for it instead of a flat 2 GB, or nothing would ever fit.
        reserve = max(0.7, hw.ram_total_gb * 0.3)
    if hw.os_name == "macOS":
        reserve += 1.0 if hw.ram_total_gb >= 6 else 0.5  # macOS is heavier than a minimal Linux
    return max(0.25, hw.ram_total_gb - reserve)


@dataclass
class Node:
    """One machine: the local one (endpoint None) or a `tlav worker` on the network."""
    hw: Hardware
    gbps: float                      # usable memory bandwidth
    endpoint: Optional[str] = None   # "ip:port" of its rpc-server
    latency_ms: float = 0.0          # round trip from the main machine
    name: str = "this PC"

    @property
    def budget(self) -> float:
        return ram_budget_gb(self.hw)


@dataclass
class Plan:
    model: Model
    quant: Quant
    ctx: int
    memory_gb: float
    est_tps: float
    quality: float                  # task score after compression
    kv: str = DEFAULT_KV
    nodes: List[Node] = field(default_factory=list)
    split: List[int] = field(default_factory=list)   # layers per node, same order as nodes
    draft: Optional[Model] = None

    @property
    def is_cluster(self) -> bool:
        return len(self.nodes) > 1

    @property
    def label(self) -> str:
        s = f"{self.model.name} @ {self.quant.name}"
        return s + (f" across {len(self.nodes)} computers" if self.is_cluster else "")

    def to_dict(self) -> dict:
        d = {"model": self.model.id, "quant": self.quant.name, "ctx": self.ctx, "kv": self.kv,
             "memory_gb": round(self.memory_gb, 2), "est_tps": round(self.est_tps, 1),
             "quality": round(self.quality, 1)}
        if self.is_cluster:
            d["nodes"] = [n.endpoint or "local" for n in self.nodes]
            d["split"] = self.split
        if self.draft:
            d["draft"] = self.draft.id
        return d


def _node_gbps(node: Node, model: Model, quant: Quant, weights_here_gb: float) -> float:
    gbps = node.gbps
    if node.hw.gpu == "cuda" and node.hw.gpu_vram_gb >= weights_here_gb:
        gbps = max(gbps, 150.0)  # conservative for old GTX-class cards
    elif node.hw.gpu == "metal":
        gbps = max(gbps, 60.0)
    return gbps


def estimate_tps(model: Model, quant: Quant, effective_gbps: float, hw: Hardware) -> float:
    node = Node(hw, effective_gbps)
    return _cluster_tps(model, quant, [node], [1.0])


def _cluster_tps(model: Model, quant: Quant, nodes: List[Node], shares: List[float]) -> float:
    per_token_gb = model.active_gb(quant) * 1.074  # GiB -> GB
    t = 0.0
    for node, share in zip(nodes, shares):
        if share <= 0:
            continue
        gbps = _node_gbps(node, model, quant, model.weights_gb(quant) * share)
        t += per_token_gb * share / gbps
        if node.endpoint:
            t += (RPC_ROUNDTRIPS_PER_TOKEN * max(node.latency_ms, 0.5) + RPC_FIXED_MS_PER_TOKEN) / 1000
    return 1.0 / t if t else 0.0


def _split_layers(layers: int, capacities: List[float]) -> List[int]:
    total = sum(capacities)
    raw = [layers * c / total for c in capacities]
    out = [int(r) for r in raw]
    # hand out remaining layers to the largest remainders
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: layers - sum(out)]:
        out[i] += 1
    return out


def evaluate_nodes(model: Model, quant: Quant, nodes: List[Node], task: str = "chat",
                   kv: str = DEFAULT_KV) -> Optional[Plan]:
    """Fit a model on one or more machines. Returns None if it does not fit."""
    weights = model.weights_gb(quant)
    main = nodes[0]
    if weights * 1.1 > main.hw.disk_free_gb > 0:
        return None  # the main machine stores the file; workers receive weights over the network
    for ctx in CONTEXT_CHOICES:
        ctx = min(ctx, model.max_ctx)
        need = weights + model.kv_cache_gb(ctx, KV_BYTES[kv])
        if len(nodes) == 1:
            mem = need + overhead_gb(model)
            if mem <= main.budget:
                tps = _cluster_tps(model, quant, nodes, [1.0])
                return Plan(model, quant, ctx, mem, tps, model.score(task) * quant.retention, kv,
                            nodes=list(nodes), split=[model.layers])
            continue
        caps = [max(0.0, n.budget - RUNTIME_OVERHEAD_GB) for n in nodes]
        if sum(caps) < need:
            continue
        split = _split_layers(model.layers, caps)
        shares = [s / model.layers for s in split]
        if any(need * sh > cap + 1e-9 for sh, cap in zip(shares, caps)):
            continue  # rounding pushed one machine over its budget
        tps = _cluster_tps(model, quant, nodes, shares)
        mem = need + RUNTIME_OVERHEAD_GB * len(nodes)
        return Plan(model, quant, ctx, mem, tps, model.score(task) * quant.retention, kv,
                    nodes=list(nodes), split=split)
    return None


def evaluate(model: Model, quant: Quant, hw: Hardware, effective_gbps: float,
             task: str = "chat", kv: str = DEFAULT_KV) -> Optional[Plan]:
    return evaluate_nodes(model, quant, [Node(hw, effective_gbps)], task, kv)


COMFORTABLE_TPS = 5.0  # roughly reading speed


def _quants_for(min_quant: str, quants: Optional[List[Quant]] = None) -> List[Quant]:
    """Compression levels to try, best quality first.

    `quants` describes the files that actually exist (e.g. a Hugging Face repo),
    using their real bits per weight; otherwise Tlav's standard levels are used.
    """
    floor = catalog.QUANT_BY_NAME[min_quant].bpw - 0.3
    if quants is not None:
        return sorted((q for q in quants if q.bpw >= floor and q.bpw < 12), key=lambda q: -q.bpw)
    return [q for q in catalog.quants_down_to(min_quant) if q.name != "F16"]  # F16 never pays off on CPU


def plan_for_model(model: Model, nodes: List[Node], task: str = "chat", min_tps: float = 2.5,
                   min_quant: str = catalog.DEFAULT_MIN_QUANT, kv: str = DEFAULT_KV,
                   quants: Optional[List[Quant]] = None) -> Optional[Plan]:
    """Pick the compression level (and machines) for one model.

    Use the least compression that still runs comfortably fast; if none is
    comfortable, use the most compressed allowed level (fastest), since the
    quality gap between Q6_K and Q4_K_M is far smaller than the speed gap.
    A model that fits on this computer alone is never split across the network.
    """
    def best_on(ns):
        usable = []
        for quant in _quants_for(min_quant, quants):
            plan = evaluate_nodes(model, quant, ns, task, kv)
            if plan and plan.est_tps >= min_tps:
                usable.append(plan)
        if not usable:
            return None
        comfy = [p for p in usable if p.est_tps >= COMFORTABLE_TPS]
        return comfy[0] if comfy else usable[-1]

    local = best_on(nodes[:1])
    if local or len(nodes) == 1:
        return local
    return best_on(nodes)


def attach_draft(plan: Plan, min_params_b: float = 3.0) -> Plan:
    """Add a small draft model for speculative decoding when RAM allows.

    The big model verifies every drafted token, so the output is exactly what
    the big model alone would produce; only speed changes.
    """
    if plan.is_cluster or plan.model.is_moe or not plan.model.draft or plan.model.params_b < min_params_b:
        return plan
    try:
        d = catalog.get(plan.model.draft)
    except KeyError:
        return plan
    dq = catalog.QUANT_BY_NAME["Q8_0"]
    extra = d.weights_gb(dq) + d.kv_cache_gb(plan.ctx, KV_BYTES[plan.kv]) + 0.2
    if plan.memory_gb + extra <= plan.nodes[0].budget and d.weights_gb(dq) < plan.model.weights_gb(plan.quant) / 3:
        plan.draft = d
        plan.memory_gb += extra
    return plan


def candidates(hw: Hardware, effective_gbps: float, task: str = "chat", min_tps: float = 2.5,
               min_quant: str = catalog.DEFAULT_MIN_QUANT, kv: str = DEFAULT_KV,
               workers: Optional[List[Node]] = None, models: Optional[List[Model]] = None) -> List[Plan]:
    """Best plan per model, sorted best first. `workers` enables cluster plans."""
    nodes = [Node(hw, effective_gbps)] + list(workers or [])
    plans = []
    for model in models if models is not None else catalog.load():
        if task == "code" and "code" not in model.tags and model.score("code") < 60:
            continue
        plan = plan_for_model(model, nodes, task, min_tps, min_quant, kv)
        if plan:
            plans.append(plan)
    # quality first; prefer faster when quality is within 1 point
    plans.sort(key=lambda p: (round(p.quality), p.est_tps), reverse=True)
    return plans


def best(hw: Hardware, effective_gbps: float, **kw) -> Optional[Plan]:
    c = candidates(hw, effective_gbps, **kw)
    return c[0] if c else None


def best_quant_for(model: Model, hw: Hardware, effective_gbps: float,
                   min_quant: str = catalog.AGGRESSIVE_MIN_QUANT, kv: str = DEFAULT_KV,
                   workers: Optional[List[Node]] = None) -> Optional[Plan]:
    """Least compression that still fits (used by `tlav compress --fit` and `setup --model`)."""
    nodes = [Node(hw, effective_gbps)] + list(workers or [])
    for ns in ([nodes[0]], nodes) if workers else ([nodes[0]],):
        for quant in _quants_for(min_quant):
            plan = evaluate_nodes(model, quant, ns, kv=kv)
            if plan:
                return plan
    return None
