"""Benchmarks.

Token generation on CPU is limited by memory bandwidth: every generated token
streams the full set of weights through the CPU once. So the quick benchmark
measures bandwidth (pure Python, memcpy-speed via bytearray slicing) and the
planner turns that into a tokens/sec prediction. When llama.cpp is installed,
`real_benchmark` measures actual tokens/sec on a model with llama-bench.
"""

import json
import re
import subprocess
import time
from typing import Optional

from . import config

# Ratio between llama.cpp's multi-threaded decode bandwidth and our
# single-threaded memcpy figure. Single-thread copies under-use the memory
# channels, while decoding only reads; on DDR3/DDR4 desktops the two end up
# roughly equal, so 1.0 (calibrate with `tlav bench --real`).
DECODE_EFFICIENCY = 1.0


def memory_bandwidth_gbps(size_mb: int = 256, rounds: int = 6) -> float:
    size = size_mb * 1024 * 1024
    src = bytearray(size)
    dst = bytearray(size)
    for i in range(0, size, 4096):  # touch pages so the OS really allocates them
        src[i] = 1
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        dst[:] = src
        best = min(best, time.perf_counter() - t0)
    # a copy reads `size` bytes and writes `size` bytes
    return (2 * size / best) / 1e9


def quick_benchmark(save: bool = True) -> dict:
    bw = memory_bandwidth_gbps()
    result = {"bandwidth_gbps": round(bw, 2), "effective_gbps": round(bw * DECODE_EFFICIENCY, 2),
              "timestamp": time.time()}
    if save:
        config.save_state(benchmark=result)
    return result


def cached_or_run() -> dict:
    b = config.load_state().get("benchmark")
    return b if b else quick_benchmark()


def parse_llama_bench_json(text: str) -> dict:
    """Extract prompt-processing and generation tok/s from `llama-bench -o json`."""
    data = json.loads(text)
    out = {}
    for row in data:
        if row.get("n_gen", 0) > 0:
            out["gen_tps"] = round(float(row["avg_ts"]), 2)
        elif row.get("n_prompt", 0) > 0:
            out["prompt_tps"] = round(float(row["avg_ts"]), 2)
    return out


def real_benchmark(model_path: str, threads: int, bench_bin: str) -> Optional[dict]:
    cmd = [bench_bin, "-m", model_path, "-t", str(threads), "-p", "64", "-n", "32", "-r", "2", "-o", "json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"\[.*\]", out.stdout, re.S)
    if not m:
        return None
    try:
        res = parse_llama_bench_json(m.group(0))
    except (ValueError, KeyError):
        return None
    res["model"] = model_path
    state = config.load_state()
    results = state.get("real_benchmarks", {})
    results[model_path] = res
    config.save_state(real_benchmarks=results)
    return res
