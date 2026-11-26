"""Turn a plan + hardware into concrete llama.cpp launch settings."""

from typing import List, Optional

from . import runtime
from .hardware import Hardware
from .planner import Plan


def settings(plan: Plan, hw: Hardware, low_ram: bool = False) -> dict:
    # Decoding is memory-bound: more threads than physical cores only adds
    # contention, and hyper-threads hurt on old Intel CPUs.
    threads = max(1, hw.physical_cores)
    budget = plan.nodes[0].budget if plan.nodes else 0
    local_share = plan.split[0] / plan.model.layers if plan.is_cluster and plan.model.layers else 1.0
    headroom = budget - plan.memory_gb * local_share
    small = low_ram or hw.ram_total_gb < 12
    s = {
        "threads": threads,
        "threads_batch": max(threads, min(hw.logical_cores, threads * 2)),
        "ctx": plan.ctx,
        "batch": 256 if small else 512,
        "ubatch": 128 if small else 256,
        # mlock keeps weights from being swapped out, but only when there is spare RAM
        "mlock": headroom > 2.0 and hw.os_name != "Windows" and not low_ram,
        "gpu_layers": 0,
        "kv": plan.kv,
        "rpc": [],
        "tensor_split": [],
        "draft": None,
        # One person uses an old PC: one slot avoids per-slot buffers. The prompt
        # cache (llama-server defaults to 8 GB!) only gets RAM that is truly spare.
        "parallel": 1,
        "cache_ram_mb": int(max(0, min(1024, (headroom - 1.0) * 1024 * 0.5))),
    }
    if plan.is_cluster:
        # llama.cpp keeps the first (layers - ngl) layers on this CPU and
        # offloads the rest to the RPC devices, divided by --tensor-split.
        remote = plan.split[1:]
        s["rpc"] = [n.endpoint for n in plan.nodes[1:]]
        s["gpu_layers"] = sum(remote)
        s["tensor_split"] = remote
    elif hw.gpu == "metal":
        s["gpu_layers"] = 999
    elif hw.gpu == "cuda" and hw.gpu_vram_gb:
        frac = min(1.0, (hw.gpu_vram_gb * 0.9) / max(plan.model.weights_gb(plan.quant), 0.1))
        s["gpu_layers"] = 999 if frac >= 1.0 else int(plan.model.layers * frac)
    return s


def server_args(model_path: str, s: dict, host: str = "127.0.0.1", port: int = 8080,
                draft_path: Optional[str] = None, help_text: Optional[str] = None) -> List[str]:
    """llama-server arguments, adapted to the flags the installed llama.cpp version has."""
    h = runtime.help_text("llama-server") if help_text is None else help_text
    known = bool(h)

    def has(flag: str) -> bool:
        return (not known) or (flag in h)

    args = ["-m", model_path, "-c", str(s["ctx"]), "-t", str(s["threads"]),
            "-tb", str(s["threads_batch"]), "-b", str(s["batch"]), "-ub", str(s["ubatch"]),
            "--host", host, "--port", str(port)]
    kv = s.get("kv", "f16")
    if kv != "f16":
        # quantized V cache needs flash attention; older builds take a bare -fa
        args += (["-fa", "on"] if (not known or "on|off" in h) else ["-fa"]) + ["-ctk", kv, "-ctv", kv]
    if s.get("parallel") and has("--parallel"):
        args += ["-np", str(s["parallel"])]
    if "cache_ram_mb" in s and known and "--cache-ram" in h:
        args += ["-cram", str(s["cache_ram_mb"])]
    if s.get("mlock"):
        if known and "--load-mode" in h:
            args += ["--load-mode", "mmap+mlock"]
        elif known and "--mlock" in h:
            args.append("--mlock")
    if s.get("rpc"):
        args += ["--rpc", ",".join(s["rpc"])]
        if s.get("tensor_split"):
            args += ["-ts", ",".join(str(x) for x in s["tensor_split"])]
    if s.get("gpu_layers"):
        args += ["-ngl", str(s["gpu_layers"])]
    if draft_path:
        args += ["-md", draft_path]
        if known and "--spec-type" in h and "draft-simple" in h:
            args += ["--spec-type", "draft-simple"]   # newer builds default to no speculation
        if known and "--spec-draft-n-max" in h:      # newer llama.cpp naming
            args += ["--spec-draft-n-max", "8", "--spec-draft-n-min", "1"]
        else:
            args += ["--draft-max", "8", "--draft-min", "1"]
        if kv != "f16" and has("-ctkd"):
            args += ["-ctkd", kv, "-ctvd", kv]
    return args
