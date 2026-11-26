"""Turn a plan + hardware into concrete llama.cpp launch settings."""

from typing import List

from .hardware import Hardware
from .planner import Plan, ram_budget_gb


def settings(plan: Plan, hw: Hardware) -> dict:
    # Decoding is memory-bound: more threads than physical cores only adds
    # contention, and hyper-threads hurt on old Intel CPUs.
    threads = max(1, hw.physical_cores)
    headroom = ram_budget_gb(hw) - plan.memory_gb
    s = {
        "threads": threads,
        "threads_batch": max(threads, min(hw.logical_cores, threads * 2)),
        "ctx": plan.ctx,
        "batch": 512 if hw.ram_total_gb >= 12 else 256,
        "ubatch": 256 if hw.ram_total_gb >= 12 else 128,
        # mlock keeps weights from being swapped out, but only when there is spare RAM
        "mlock": headroom > 2.0 and hw.os_name != "Windows",
        "gpu_layers": 0,
    }
    if hw.gpu == "metal":
        s["gpu_layers"] = 999
    elif hw.gpu == "cuda" and hw.gpu_vram_gb:
        frac = min(1.0, (hw.gpu_vram_gb * 0.9) / max(plan.model.weights_gb(plan.quant), 0.1))
        s["gpu_layers"] = 999 if frac >= 1.0 else int(plan.model.layers * frac)
    return s


def server_args(model_path: str, s: dict, host: str = "127.0.0.1", port: int = 8080) -> List[str]:
    args = ["-m", model_path, "-c", str(s["ctx"]), "-t", str(s["threads"]),
            "-tb", str(s["threads_batch"]), "-b", str(s["batch"]), "-ub", str(s["ubatch"]),
            "--host", host, "--port", str(port)]
    if s.get("mlock"):
        args.append("--mlock")
    if s.get("gpu_layers"):
        args += ["-ngl", str(s["gpu_layers"])]
    return args
