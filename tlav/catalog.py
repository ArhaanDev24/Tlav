"""Model catalog plus quantization (compression) levels."""

import json
import math
import os
import urllib.request
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
    draft: Optional[str] = None   # small same-tokenizer model for speculative decoding
    stream_params_b: Optional[float] = None  # weights read per token (no embedding lookups, active experts)
    released: Optional[str] = None           # YYYY-MM, used to estimate quality of new models
    base: Optional[str] = None               # original model on Hugging Face (org/name)
    estimated: bool = False                  # quality score estimated, not curated

    @property
    def is_moe(self) -> bool:
        return self.active_params_b is not None and self.active_params_b < self.params_b

    def weights_gb(self, quant: Quant) -> float:
        return self.params_b * 1e9 * quant.bpw / 8 / 1024 ** 3

    def active_gb(self, quant: Quant) -> float:
        """Bytes streamed per generated token (all weights, or active experts for MoE)."""
        p = self.stream_params_b or (self.active_params_b if self.is_moe else self.params_b)
        return p * 1e9 * quant.bpw / 8 / 1024 ** 3

    def kv_cache_gb(self, ctx: int, bytes_per_elem: float = 2) -> float:
        """K+V cache size. 2 bytes = f16, ~1.06 bytes = q8_0 (Tlav's default)."""
        return 2 * self.layers * self.kv_heads * self.head_dim * ctx * bytes_per_elem / 1024 ** 3

    def score(self, task: str) -> int:
        return self.quality.get(task, self.quality.get("chat", 50))


# Bytes per KV-cache element: f16 vs q8_0 (34 bytes per 32 values).
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32}

CATALOG_URL = os.environ.get(
    "TLAV_CATALOG_URL", "https://raw.githubusercontent.com/ArhaanDev24/Tlav/main/tlav/data/catalog.json")


def _years(released: Optional[str]) -> Optional[float]:
    try:
        y, m = released.split("-")[:2]
        return int(y) - 2024 + (int(m) - 1) / 12
    except (AttributeError, ValueError):
        return None


def _effective_b(params_b: float, active_b: Optional[float]) -> float:
    """Mixture-of-experts: geometric mean of total and active size ('dense-equivalent')."""
    return math.sqrt(params_b * active_b) if active_b and active_b < params_b else params_b


_FIT: Optional[List[float]] = None


def quality_fit() -> List[float]:
    """Least-squares fit of the hand-scored catalog: score = a + b*log2(size) + c*years since 2024.

    Newer models of the same size are clearly better (training data and methods
    improve every year), so release date matters as much as size. The fit is
    learned from the curated scores, so it improves when they do.
    """
    global _FIT
    if _FIT is not None:
        return _FIT
    rows = []
    for m in load(include_discovered=False):
        yrs = _years(m.released)
        if yrs is None or ("code" in m.tags and "chat" not in m.tags):
            continue
        rows.append(((1.0, math.log2(_effective_b(m.params_b, m.active_params_b)), yrs), m.quality.get("chat", 50)))
    if len(rows) < 6:
        _FIT = [45.0, 7.2, 0.0]
        return _FIT
    a = [[sum(x[i] * x[j] for x, _ in rows) for j in range(3)] + [sum(x[i] * y for x, y in rows)] for i in range(3)]
    for i in range(3):  # Gauss-Jordan on the 3x3 normal equations
        piv = a[i][i] or 1e-9
        a[i] = [v / piv for v in a[i]]
        for k in range(3):
            if k != i:
                a[k] = [vk - a[k][i] * vi for vk, vi in zip(a[k], a[i])]
    _FIT = [a[i][3] for i in range(3)]
    return _FIT


def estimate_quality(params_b: float, moe_total_b: Optional[float] = None, released: Optional[str] = None) -> int:
    """Capability score for models outside the curated catalog.

    `params_b` is the active size for mixture-of-experts (with `moe_total_b` the total).
    Estimated scores get a small penalty, so on a tie a hand-scored model wins.
    """
    eff = _effective_b(moe_total_b, params_b) if moe_total_b else params_b
    a, b, c = quality_fit()
    yrs = _years(released)
    if yrs is None:
        yrs = 1.0  # unknown date: assume an average 2025 model
    return int(max(25, min(92, a + b * math.log2(max(eff, 0.05)) + c * yrs - 1.5)))


# Every other level publishers ship. "_L" files keep embeddings/output at Q8_0, "IQ" levels
# use a lattice codebook: both keep more quality per bit. Numbers are cautious readings of
# llama.cpp's published KL-divergence tables, on the same scale as QUANTS.
EXTRA_RETENTION: Dict[str, float] = {
    "BF16": 1.0, "Q6_K_L": 0.997, "Q5_K_L": 0.992, "Q5_K_S": 0.987, "Q4_K_L": 0.978,
    "IQ4_NL": 0.971, "IQ4_XS": 0.970, "Q4_K_S": 0.968, "Q4_1": 0.960, "Q4_0": 0.955,
    "Q3_K_XL": 0.955, "Q3_K_L": 0.950, "IQ3_M": 0.940, "IQ3_S": 0.932, "IQ3_XS": 0.925,
    "Q3_K_S": 0.920, "IQ3_XXS": 0.905, "Q2_K_L": 0.870, "IQ2_M": 0.860, "IQ2_S": 0.820,
    "IQ2_XS": 0.800, "IQ2_XXS": 0.760,
    # unsloth "dynamic" (UD) files: sensitive layers kept at higher precision
    "Q8_K_XL": 0.9995, "Q6_K_XL": 0.997, "Q5_K_XL": 0.993, "Q4_K_XL": 0.980, "Q2_K_XL": 0.880,
    "IQ3_XXS_XL": 0.910,
}


def retention_for(name: str, bpw: Optional[float] = None) -> Optional[float]:
    """Share of quality a compression level keeps (None if unknown and no size to judge by)."""
    n = name.upper()
    if n in QUANT_BY_NAME:
        return QUANT_BY_NAME[n].retention
    if n in EXTRA_RETENTION:
        return EXTRA_RETENTION[n]
    if bpw is None:
        return None
    return next((q.retention for q in reversed(QUANTS) if q.bpw >= bpw - 0.3), 1.0)


def quant_from_file(size_bytes: int, params_b: float, name: str = "file") -> Quant:
    """Describe an existing GGUF file as a Quant using its real bits/weight."""
    bpw = size_bytes * 8 / (params_b * 1e9) if params_b else 4.85
    known = QUANT_BY_NAME.get(name.upper())
    retention = retention_for(name, bpw)
    note = known.note if known else (f"keeps ~{retention * 100:.1f}% quality" if name.upper() in EXTRA_RETENTION
                                     else "from file")
    return Quant(name.upper(), round(bpw, 2), retention, note)


def _catalog_path() -> Path:
    try:
        from . import config
        override = config.home() / "catalog.json"
        if override.exists():
            return override
    except OSError:
        pass
    return Path(__file__).parent / "data" / "catalog.json"


def _discovered_path() -> Optional[Path]:
    try:
        from . import config
        p = config.home() / "discovered.json"
        return p if p.exists() else None
    except OSError:
        return None


def load(include_discovered: bool = True) -> List[Model]:
    """Curated catalog plus models found on Hugging Face by `tlav models --update`."""
    fields = set(Model.__dataclass_fields__)
    data = json.loads(_catalog_path().read_text())
    models = [Model(**{k: v for k, v in m.items() if k in fields}) for m in data["models"]]
    path = _discovered_path() if include_discovered else None
    if path:
        try:
            extra = json.loads(path.read_text()).get("models", [])
        except (OSError, ValueError):
            extra = []
        ids = {m.id for m in models}
        bases = {(m.base or "").lower() for m in models}
        for m in extra:
            if m.get("id") not in ids and (m.get("base") or "").lower() not in bases:
                models.append(Model(**{k: v for k, v in m.items() if k in fields}))
                ids.add(m["id"])
    return models


def get(model_id: str) -> Model:
    for m in load():
        if m.id == model_id:
            return m
    raise KeyError(f"Unknown model '{model_id}'. Run `tlav models` to see the catalog.")


def update(url: str = CATALOG_URL) -> int:
    """Download the latest catalog into ~/.tlav/catalog.json. Returns the model count."""
    from . import config
    req = urllib.request.Request(url, headers={"User-Agent": "tlav"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok and "githubusercontent" in url:
        req.add_header("Authorization", f"token {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    if not isinstance(data.get("models"), list) or not data["models"]:
        raise ValueError("downloaded catalog is empty or malformed")
    (config.home() / "catalog.json").write_text(json.dumps(data, indent=1))
    return len(data["models"])
