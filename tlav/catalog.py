"""Model catalog plus quantization (compression) levels."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Quant:
    name: str
    bpw: float          # average bits per weight in the GGUF file
    retention: float    # share of original quality kept (from published perplexity/KL studies)
    note: str


# Ordered from largest/best to smallest. Retention numbers are deliberately
# cautious; Q8_0 and Q6_K are effectively indistinguishable from the original.
QUANTS: List[Quant] = [
    Quant("F16", 16.0, 1.000, "original precision"),
    Quant("Q8_0", 8.5, 0.999, "practically identical to original"),
    Quant("Q6_K", 6.56, 0.996, "practically identical to original"),
    Quant("Q5_K_M", 5.69, 0.990, "near-lossless"),
    Quant("Q4_K_M", 4.85, 0.975, "very small loss, best size/quality balance"),
    Quant("Q3_K_M", 3.91, 0.940, "noticeable loss, use only to fit a much better model"),
    Quant("Q2_K", 3.0, 0.850, "large loss, last resort"),
]
QUANT_BY_NAME: Dict[str, Quant] = {q.name: q for q in QUANTS}

# Quants Tlav considers "quality preserved" by default.
DEFAULT_MIN_QUANT = "Q4_K_M"
AGGRESSIVE_MIN_QUANT = "Q3_K_M"


def quants_down_to(min_name: str) -> List[Quant]:
    out = []
    for q in QUANTS:
        out.append(q)
        if q.name == min_name:
            break
    return out


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    repo: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    max_ctx: int
    quality: Dict[str, int] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    active_params_b: Optional[float] = None

    @property
    def is_moe(self) -> bool:
        return self.active_params_b is not None and self.active_params_b < self.params_b

    def weights_gb(self, quant: Quant) -> float:
        return self.params_b * 1e9 * quant.bpw / 8 / 1024 ** 3

    def active_gb(self, quant: Quant) -> float:
        """Bytes streamed per generated token (all weights, or active experts for MoE)."""
        p = self.active_params_b if self.is_moe else self.params_b
        return p * 1e9 * quant.bpw / 8 / 1024 ** 3

    def kv_cache_gb(self, ctx: int, bytes_per_elem: int = 2) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * ctx * bytes_per_elem / 1024 ** 3

    def score(self, task: str) -> int:
        return self.quality.get(task, self.quality.get("chat", 50))


def _catalog_path() -> Path:
    return Path(__file__).parent / "data" / "catalog.json"


def load() -> List[Model]:
    data = json.loads(_catalog_path().read_text())
    return [Model(**m) for m in data["models"]]


def get(model_id: str) -> Model:
    for m in load():
        if m.id == model_id:
            return m
    raise KeyError(f"Unknown model '{model_id}'. Run `tlav models` to see the catalog.")
