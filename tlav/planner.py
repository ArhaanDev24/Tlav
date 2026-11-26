"""Pick the best model + compression level + settings for a machine.

For every catalog model and every quantization level we estimate:
  memory = weights + KV cache(context) + runtime overhead
  speed  = effective bandwidth / bytes read per token
and keep the combos that fit in the RAM budget and are fast enough to be
usable. The winner is the highest quality after compression, so a big model
compressed to Q4_K_M beats a small model kept at Q8_0 when that is better.
"""

from dataclasses import dataclass
from typing import List, Optional

from . import catalog
from .catalog import Model, Quant
from .hardware import Hardware

RUNTIME_OVERHEAD_GB = 0.4      # llama.cpp buffers, compute graph
CONTEXT_CHOICES = (8192, 4096, 2048)
MIN_CTX = 2048


def ram_budget_gb(hw: Hardware) -> float:
    """RAM we allow the model to use, leaving room for the OS and a browser."""
    if hw.gpu == "cuda" and hw.gpu_vram_gb:
        # with CUDA we offload to VRAM and spill the rest into RAM
        return hw.gpu_vram_gb * 0.9 + max(0.0, hw.ram_total_gb * 0.6 - 2.0)
    reserve = max(2.0, hw.ram_total_gb * 0.25)
    if hw.os_name == "macOS":
        reserve += 1.0  # macOS (WindowServer, Spotlight...) is heavier than a minimal Linux
    return max(0.5, hw.ram_total_gb - reserve)


@dataclass
class Plan:
    model: Model
    quant: Quant
    ctx: int
    memory_gb: float
    est_tps: float
    quality: float          # task score after compression
    fits_prebuilt: bool = True

    @property
    def label(self) -> str:
        return f"{self.model.name} @ {self.quant.name}"

    def to_dict(self) -> dict:
        return {"model": self.model.id, "quant": self.quant.name, "ctx": self.ctx,
                "memory_gb": round(self.memory_gb, 2), "est_tps": round(self.est_tps, 1),
                "quality": round(self.quality, 1)}


def estimate_tps(model: Model, quant: Quant, effective_gbps: float, hw: Hardware) -> float:
    gbps = effective_gbps
    if hw.gpu == "cuda" and hw.gpu_vram_gb >= model.weights_gb(quant):
        gbps = max(gbps, 150.0)  # conservative for old GTX-class cards
    elif hw.gpu == "metal":
        gbps = max(gbps, 60.0)
    per_token_gb = model.active_gb(quant) * 1.074  # GiB -> GB
    return gbps / per_token_gb if per_token_gb else 0.0


def evaluate(model: Model, quant: Quant, hw: Hardware, effective_gbps: float,
             task: str = "chat") -> Optional[Plan]:
    budget = ram_budget_gb(hw)
    weights = model.weights_gb(quant)
    if weights * 1.1 > hw.disk_free_gb > 0:
        return None
    for ctx in CONTEXT_CHOICES:
        ctx = min(ctx, model.max_ctx)
        mem = weights + model.kv_cache_gb(ctx) + RUNTIME_OVERHEAD_GB
        if mem <= budget:
            tps = estimate_tps(model, quant, effective_gbps, hw)
            return Plan(model, quant, ctx, mem, tps, model.score(task) * quant.retention)
    return None


COMFORTABLE_TPS = 5.0  # roughly reading speed


def plan_for_model(model: Model, hw: Hardware, effective_gbps: float, task: str = "chat",
                   min_tps: float = 2.5, min_quant: str = catalog.DEFAULT_MIN_QUANT) -> Optional[Plan]:
    """Pick the compression level for one model.

    Use the least compression that still runs comfortably fast; if none is
    comfortable, use the most compressed allowed level (fastest), since the
    quality gap between Q6_K and Q4_K_M is far smaller than the speed gap.
    """
    usable = []
    for quant in catalog.quants_down_to(min_quant):
        if quant.name == "F16":
            continue  # never worth it on CPU-only old machines
        plan = evaluate(model, quant, hw, effective_gbps, task)
        if plan and plan.est_tps >= min_tps:
            usable.append(plan)
    if not usable:
        return None
    comfy = [p for p in usable if p.est_tps >= COMFORTABLE_TPS]
    return comfy[0] if comfy else usable[-1]


def candidates(hw: Hardware, effective_gbps: float, task: str = "chat", min_tps: float = 2.5,
               min_quant: str = catalog.DEFAULT_MIN_QUANT) -> List[Plan]:
    """Best plan per model, sorted best first."""
    plans = []
    for model in catalog.load():
        if task == "code" and "code" not in model.tags and model.score("code") < 60:
            continue
        plan = plan_for_model(model, hw, effective_gbps, task, min_tps, min_quant)
        if plan:
            plans.append(plan)
    # quality first; prefer faster when quality is within 1 point
    plans.sort(key=lambda p: (round(p.quality), p.est_tps), reverse=True)
    return plans


def best(hw: Hardware, effective_gbps: float, **kw) -> Optional[Plan]:
    c = candidates(hw, effective_gbps, **kw)
    return c[0] if c else None


def best_quant_for(model: Model, hw: Hardware, effective_gbps: float,
                   min_quant: str = catalog.AGGRESSIVE_MIN_QUANT) -> Optional[Plan]:
    """Least compression that still fits this machine (used by `tlav compress --fit`)."""
    for quant in catalog.quants_down_to(min_quant):
        if quant.name == "F16":
            continue
        plan = evaluate(model, quant, hw, effective_gbps)
        if plan:
            return plan
    return None
