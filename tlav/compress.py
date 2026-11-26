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

EVAL_TEXT = Path(__file__).parent / "data" / "eval.txt"      # used only to *measure* quality
CALIB_TEXT = Path(__file__).parent / "data" / "calib.txt"    # used only to *tune* compression


def importance_matrix(src: Path, threads: int, text: Path = CALIB_TEXT) -> Optional[Path]:
    """Measure which weights matter most (llama-imatrix) so compression protects them.

    Calibration uses a different text from the quality test, so the quality
    check stays honest.
    """
    exe = runtime.find("llama-imatrix")
    if not exe:
        return None
    out = src.parent / (src.stem + ".imatrix.gguf")
    if out.exists():
        return out
    log = config.home() / "compress.log"
    print(f"Measuring weight importance on calibration text (details in {log}) ...")
    cmd = [exe, "-m", str(src), "-f", str(text), "-o", str(out), "-c", "256", "-t", str(threads),
           "-b", "256"]
    with open(log, "a") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return out if res.returncode == 0 and out.exists() else None


def quantize(src: Path, quant: str, threads: int, out: Optional[Path] = None,
             imatrix: Optional[Path] = None) -> Path:
    q = runtime.require("llama-quantize")
    base = re.sub(r"[-.](f16|bf16|f32|q8_0)$", "", src.stem, flags=re.I)
    out = out or src.parent / f"{base}-{quant}{'-imat' if imatrix else ''}.gguf"
    if out.exists():
        return out
    cmd = [q]
    if not re.search(r"[-.](f16|bf16|f32)$", src.stem, re.I):
        cmd.append("--allow-requantize")  # source already quantized (e.g. Q8_0)
    if imatrix:
        cmd += ["--imatrix", str(imatrix)]
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


def perplexity(model: Path, threads: int, ctx: int = 256, chunks: int = 4,
               kv: str = "f16") -> Optional[float]:
    exe = runtime.require("llama-perplexity")
    cmd = [exe, "-m", str(model), "-f", str(EVAL_TEXT), "-c", str(ctx), "--chunks", str(chunks),
           "-t", str(threads), "-b", str(ctx)]
    if kv != "f16":
        h = runtime.help_text("llama-perplexity")
        cmd += (["-fa", "on"] if (not h or "on|off" in h) else ["-fa"]) + ["-ctk", kv, "-ctv", kv]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return parse_perplexity(res.stdout + res.stderr)


def parse_kld(output: str) -> Optional[dict]:
    """Mean KL divergence, 'same top token' %, and perplexity ratio from llama-perplexity."""
    kld = re.search(r"Mean\s+KLD:\s*([0-9.]+)", output)
    top = re.search(r"Same top p:\s*([0-9.]+)", output)
    ratio = re.search(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)", output)
    if not kld:
        return None
    return {"kld": float(kld.group(1)), "same_top_pct": float(top.group(1)) if top else None,
            "ppl_ratio": float(ratio.group(1)) if ratio else None}


def kld_base(source: Path, threads: int, ctx: int = 256, chunks: int = 4) -> Optional[Path]:
    """Save the original model's predictions once, to compare every compressed version against."""
    exe = runtime.find("llama-perplexity")
    if not exe:
        return None
    out = source.parent / (source.stem + ".logits")
    if out.exists():
        return out
    cmd = [exe, "-m", str(source), "-f", str(EVAL_TEXT), "-c", str(ctx), "--chunks", str(chunks),
           "-t", str(threads), "-b", str(ctx), "--kl-divergence-base", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return out if res.returncode == 0 and out.exists() else None


def kld(model: Path, base: Path, threads: int) -> Optional[dict]:
    """How closely the compressed model reproduces the original's word-by-word predictions.

    KL divergence is the standard measure of quantization damage (0 = identical);
    'same top' is how often both models would pick the same next word.
    """
    exe = runtime.require("llama-perplexity")
    cmd = [exe, "-m", str(model), "--kl-divergence-base", str(base), "--kl-divergence",
           "-t", str(threads), "-b", "256"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return parse_kld(res.stdout + res.stderr)


def describe(m: dict) -> str:
    top = f"same next word as the original {m['same_top_pct']:.1f}% of the time, " if m.get("same_top_pct") else ""
    ppl = f", perplexity {100 * (m['ppl_ratio'] - 1):+.2f}%" if m.get("ppl_ratio") else ""
    return f"{top}KL divergence {m['kld']:.4f}{ppl}"


def compress_to_size(source: Path, target_gb: float, threads: int, use_imatrix: bool = True,
                     verify: bool = True) -> dict:
    """Build a file of exactly `target_gb`: llama.cpp picks a precision per tensor (using the
    importance matrix) so the most sensitive weights get the most bits. Fills a PC's RAM
    budget precisely instead of rounding down to the nearest ready-made level."""
    q = runtime.require("llama-quantize")
    if "--target-size" not in runtime.help_text("llama-quantize"):
        raise RuntimeError("This llama.cpp is too old for exact-size compression; run `tlav install-runtime`.")
    imat = importance_matrix(source, threads) if use_imatrix else None
    base = re.sub(r"[-.](f16|bf16|f32|q8_0)$", "", source.stem, flags=re.I)
    mib = int(target_gb * 1024)
    out = source.parent / f"{base}-fit{target_gb:.2f}G{'-imat' if imat else ''}.gguf"
    if not out.exists():
        cmd = [q] + (["--imatrix", str(imat)] if imat else []) + ["--target-size", f"{mib}m", str(source),
                                                                   str(out), "Q4_K_M", str(threads)]
        if not re.search(r"[-.](f16|bf16|f32)$", source.stem, re.I):
            cmd.insert(1, "--allow-requantize")
        log = config.home() / "compress.log"
        print(f"Compressing {source.name} to exactly {target_gb:.2f} GB, choosing the precision of every "
              f"tensor (details in {log}) ...")
        with open(log, "a") as f:
            res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"llama-quantize failed; see {log}")
    result = {"path": str(out), "quant": "FIT", "size_gb": round(out.stat().st_size / 1024 ** 3, 2),
              "imatrix": bool(imat)}
    if verify:
        b = kld_base(source, threads)
        m = kld(out, b, threads) if b else None
        if m:
            result.update(m)
            print(f"  exact fit ({result['size_gb']} GB): {describe(m)}")
    return result


def quality_delta(ref_ppl: float, new_ppl: float) -> float:
    """Percent perplexity increase (0 = identical quality)."""
    return 100.0 * (new_ppl - ref_ppl) / ref_ppl


def kv_check(model: Path, threads: int, kv: str = "q8_0", tolerance_pct: float = 0.5) -> dict:
    """Does a compressed context cache (KV) change quality for this model? Measured, not assumed."""
    ref = perplexity(model, threads)
    new = perplexity(model, threads, kv=kv)
    if ref is None or new is None:
        return {"kv": "f16", "ok": False, "reason": "could not measure"}
    delta = quality_delta(ref, new)
    ok = delta <= tolerance_pct
    return {"kv": kv if ok else "f16", "ok": ok, "ref_ppl": ref, "ppl": new, "delta_pct": round(delta, 3)}


def compress(source: Path, quant: str, threads: int, verify: bool = True,
             tolerance_pct: float = 3.0, use_imatrix: bool = True) -> dict:
    """Quantize `source` to `quant`; if verification fails, step up to the next larger quant."""
    order = [q.name for q in catalog.QUANTS if q.name != "F16"]
    start = order.index(quant)
    ref = perplexity(source, threads) if verify else None
    base_logits = kld_base(source, threads) if verify else None
    if verify and ref is None:
        print("Warning: could not measure reference perplexity; skipping quality check.")
    imat = importance_matrix(source, threads) if use_imatrix else None
    result = {}
    for name in reversed(order[: start + 1]):  # target first, then progressively larger
        variants = [quantize(source, name, threads)]
        if imat and name != "Q8_0":
            variants.append(quantize(source, name, threads, imatrix=imat))
        if not verify or ref is None:
            out = variants[-1]
            return {"path": str(out), "quant": name, "size_gb": round(out.stat().st_size / 1024 ** 3, 2),
                    "ref_ppl": ref, "imatrix": out != variants[0]}
        # Measure every variant and keep the one that really is closest to the original:
        # by KL divergence when available (the more sensitive measure), else by perplexity.
        klds = {v: kld(v, base_logits, threads) for v in variants} if base_logits else {}
        if klds and all(klds.values()):
            best = min(variants, key=lambda v: klds[v]["kld"])
            variants = [best] + [v for v in variants if v != best]
            for v in variants[1:]:
                v.unlink(missing_ok=True)
            variants = variants[:1]
        scored = [(perplexity(v, threads), v) for v in variants]
        scored = [(s, v) for s, v in scored if s is not None] or [(None, variants[-1])]
        ppl, out = min(scored, key=lambda x: x[0] if x[0] is not None else float("inf"))
        for _, v in scored:
            if v != out:
                v.unlink(missing_ok=True)
        result = {"path": str(out), "quant": name, "size_gb": round(out.stat().st_size / 1024 ** 3, 2),
                  "ref_ppl": ref, "ppl": ppl, "imatrix": "-imat" in out.name}
        if klds.get(out):
            result.update(klds[out])
        if ppl is None:
            return result
        delta = quality_delta(ref, ppl)
        result["delta_pct"] = round(delta, 2)
        how = "importance-matrix" if result["imatrix"] else "standard"
        print(f"  {name} ({how}): perplexity {ppl:.3f} vs original {ref:.3f} ({delta:+.2f}%)")
        if klds.get(out):
            print(f"    {describe(klds[out])}")
        if delta <= tolerance_pct:
            result["verified"] = True
            return result
        print(f"  {name} loses more than {tolerance_pct}% quality; trying a larger level.")
        if name != order[0]:  # keep the last attempt so a result always exists
            out.unlink(missing_ok=True)  # don't leave rejected files behind
    result["verified"] = False
    return result


def fetch_source(model: catalog.Model) -> Path:
    files = hub.list_files(model.repo)
    src = hub.find_source_file(files)
    if not src:
        raise RuntimeError(f"No F16/BF16/Q8_0 file in {model.repo} to compress from.")
    have = (config.models_dir() / src[0]).exists()
    print(f"{'Using' if have else 'Downloading'} the original weights {src[0]} ({src[1]}) ...")
    return hub.download(model.repo, src[0], config.models_dir())
