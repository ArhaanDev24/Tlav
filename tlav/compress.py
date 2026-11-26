"""Compression = quantization with a quality check.

Quantization stores weights with fewer bits (16 -> ~5 bits for Q4_K_M), making
a model 3-4x smaller and faster on memory-bound old machines. It is not
mathematically lossless, so Tlav *measures* the result: it computes perplexity
(how surprised the model is by reference text; lower is better) for the
original and the compressed model, and rejects the compressed one if it is
worse than the tolerance (default 3%). Q8_0/Q6_K typically land under 0.5%,
Q5_K_M around 1%, Q4_K_M around 2-3%.
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

from . import catalog, config, hub, runtime

EVAL_TEXT = Path(__file__).parent / "data" / "eval.txt"


def quantize(src: Path, quant: str, threads: int, out: Optional[Path] = None) -> Path:
    q = runtime.require("llama-quantize")
    base = re.sub(r"[-.](f16|bf16|f32|q8_0)$", "", src.stem, flags=re.I)
    out = out or src.parent / f"{base}-{quant}.gguf"
    if out.exists():
        return out
    cmd = [q]
    if not re.search(r"[-.](f16|bf16|f32)$", src.stem, re.I):
        cmd.append("--allow-requantize")  # source already quantized (e.g. Q8_0)
    cmd += [str(src), str(out), quant, str(threads)]
    log = config.home() / "compress.log"
    print(f"Compressing {src.name} -> {out.name} (details in {log}) ...")
    with open(log, "a") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise RuntimeError(f"llama-quantize failed; see {log}")
    return out


def parse_perplexity(output: str) -> Optional[float]:
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", output)
    if m:
        return float(m.group(1))
    vals = re.findall(r"\[\d+\]([0-9.]+)", output)
    return float(vals[-1]) if vals else None


def perplexity(model: Path, threads: int, ctx: int = 256, chunks: int = 4) -> Optional[float]:
    exe = runtime.require("llama-perplexity")
    cmd = [exe, "-m", str(model), "-f", str(EVAL_TEXT), "-c", str(ctx), "--chunks", str(chunks),
           "-t", str(threads), "-b", str(ctx)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return parse_perplexity(res.stdout + res.stderr)


def quality_delta(ref_ppl: float, new_ppl: float) -> float:
    """Percent perplexity increase (0 = identical quality)."""
    return 100.0 * (new_ppl - ref_ppl) / ref_ppl


def compress(source: Path, quant: str, threads: int, verify: bool = True,
             tolerance_pct: float = 3.0) -> dict:
    """Quantize `source` to `quant`; if verification fails, step up to the next larger quant."""
    order = [q.name for q in catalog.QUANTS if q.name != "F16"]
    start = order.index(quant)
    ref = perplexity(source, threads) if verify else None
    if verify and ref is None:
        print("Warning: could not measure reference perplexity; skipping quality check.")
    result = {}
    for name in reversed(order[: start + 1]):  # target first, then progressively larger
        out = quantize(source, name, threads)
        result = {"path": str(out), "quant": name, "size_gb": round(out.stat().st_size / 1024 ** 3, 2),
                  "ref_ppl": ref}
        if not verify or ref is None:
            return result
        ppl = perplexity(out, threads)
        result["ppl"] = ppl
        if ppl is None:
            return result
        delta = quality_delta(ref, ppl)
        result["delta_pct"] = round(delta, 2)
        print(f"  {name}: perplexity {ppl:.3f} vs original {ref:.3f} ({delta:+.2f}%)")
        if delta <= tolerance_pct:
            result["verified"] = True
            return result
        print(f"  {name} loses more than {tolerance_pct}% quality; trying a larger level.")
    result["verified"] = False
    return result


def fetch_source(model: catalog.Model) -> Path:
    files = hub.list_files(model.repo)
    src = hub.find_source_file(files)
    if not src:
        raise RuntimeError(f"No F16/BF16/Q8_0 file in {model.repo} to compress from.")
    print(f"Downloading source weights {src[0]} ({src[1]}) ...")
    return hub.download(model.repo, src[0], config.models_dir())
