"""Tlav command line interface."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__, benchmark, catalog, compress, config, hardware, hub, planner, runtime, server, tuner


def _hw(args) -> hardware.Hardware:
    if getattr(args, "ram", None):
        return hardware.simulated(args.ram, cores=args.cores, simd=args.simd, gpu=args.gpu,
                                  vram_gb=args.vram)
    return hardware.detect()


def _gbps(args, hw) -> float:
    if getattr(args, "ram", None):
        # simulated profile: dual-channel DDR3-1600 (e.g. 2013 iMac) decodes at ~16 GB/s
        return args.bandwidth or 16.0
    return benchmark.cached_or_run()["effective_gbps"]


def _print_plans(plans, top):
    print(f"{'#':<3}{'Model':<28}{'Compression':<12}{'RAM use':>9}{'Speed':>12}{'Quality':>9}  Context")
    for i, p in enumerate(plans[:top], 1):
        print(f"{i:<3}{p.model.name:<28}{p.quant.name:<12}{p.memory_gb:>7.1f}GB"
              f"{p.est_tps:>8.1f} t/s{p.quality:>9.1f}  {p.ctx}")


# ------------------------------------------------------------------ commands

def cmd_doctor(args):
    hw = hardware.detect()
    if args.json:
        print(json.dumps(hw.to_dict(), indent=2))
        return
    print(hw.summary())
    print(f"\nModel RAM budget: {planner.ram_budget_gb(hw):.1f} GB")
    note = runtime.platform_note(hw)
    if note:
        print("Note:", note)
    rt = runtime.status()
    print("\nRuntime (llama.cpp):", "installed" if rt["llama-server"] else "not installed (`tlav install-runtime`)")
    config.save_state(hardware=hw.to_dict())


def cmd_bench(args):
    print("Measuring memory bandwidth (this decides how fast models can run)...")
    b = benchmark.quick_benchmark()
    print(f"  raw memory bandwidth:  {b['bandwidth_gbps']:.1f} GB/s")
    print(f"  usable for inference:  {b['effective_gbps']:.1f} GB/s")
    if args.real:
        model = args.model or config.load_state().get("active", {}).get("path")
        if not model:
            sys.exit("No model to benchmark; pass --model PATH or run `tlav setup` first.")
        hw = hardware.detect()
        print(f"Running llama-bench on {model} ...")
        r = benchmark.real_benchmark(model, hw.physical_cores, runtime.require("llama-bench"))
        if r:
            print(f"  prompt processing: {r.get('prompt_tps', 0):.1f} tok/s")
            print(f"  generation:        {r.get('gen_tps', 0):.1f} tok/s")
        else:
            print("  llama-bench failed.")


def cmd_models(args):
    print(f"{'ID':<22}{'Name':<28}{'Params':>8}  Q4_K_M size  Tags")
    q4 = catalog.QUANT_BY_NAME["Q4_K_M"]
    for m in catalog.load():
        size = f"{m.params_b:.1f}B" + ("*" if m.is_moe else "")
        print(f"{m.id:<22}{m.name:<28}{size:>8}  {m.weights_gb(q4):>8.1f} GB  {', '.join(m.tags)}")
    print("\n* mixture-of-experts: reads only a few billion parameters per token, so it runs fast.")
    print("\nCompression levels:")
    for q in catalog.QUANTS:
        print(f"  {q.name:<8} {q.bpw:>5.2f} bits/weight  keeps ~{q.retention * 100:.1f}% quality  ({q.note})")


def cmd_recommend(args):
    hw = _hw(args)
    gbps = _gbps(args, hw)
    min_q = catalog.AGGRESSIVE_MIN_QUANT if args.aggressive else catalog.DEFAULT_MIN_QUANT
    plans = planner.candidates(hw, gbps, task=args.task, min_tps=args.min_tps, min_quant=min_q)
    print(f"Machine: {hw.ram_total_gb:.0f} GB RAM, {hw.physical_cores} cores, {hw.simd.upper()}, "
          f"{hw.gpu}; budget {planner.ram_budget_gb(hw):.1f} GB, ~{gbps:.1f} GB/s usable bandwidth\n")
    if not plans:
        print("No model fits comfortably. Try --min-tps 1 or --aggressive.")
        return
    if args.json:
        print(json.dumps([p.to_dict() for p in plans[: args.top]], indent=2))
        return
    _print_plans(plans, args.top)
    print(f"\nBest pick: {plans[0].label}  ->  `tlav setup` installs and runs it.")


def cmd_install_runtime(args):
    hw = hardware.detect()
    note = runtime.platform_note(hw)
    if note:
        print(note)
    prefer = "source" if args.source else ("prebuilt" if args.prebuilt else "auto")
    path = runtime.install(hw, prefer=prefer)
    print(f"llama.cpp ready in {path}")


def _resolve_plan(args, hw, gbps):
    min_q = catalog.AGGRESSIVE_MIN_QUANT if args.aggressive else catalog.DEFAULT_MIN_QUANT
    if args.model:
        model = catalog.get(args.model)
        if args.quant:
            plan = planner.evaluate(model, catalog.QUANT_BY_NAME[args.quant], hw, gbps, args.task)
            if not plan:
                sys.exit(f"{model.name} at {args.quant} does not fit this machine.")
            return plan
        plan = planner.best_quant_for(model, hw, gbps, min_quant=min_q)
        if not plan:
            sys.exit(f"{model.name} does not fit this machine even when compressed.")
        return plan
    plan = planner.best(hw, gbps, task=args.task, min_tps=args.min_tps, min_quant=min_q)
    if not plan:
        sys.exit("No model fits comfortably. Try `tlav setup --aggressive` or `--min-tps 1`.")
    return plan


def _obtain_model(plan, hw, local_compress: bool) -> Path:
    files = hub.list_files(plan.model.repo)
    ready = hub.find_quant_file(files, plan.quant.name)
    if ready and not local_compress:
        print(f"Downloading pre-compressed {ready} ...")
        return hub.download(plan.model.repo, ready)
    src = compress.fetch_source(plan.model)
    res = compress.compress(src, plan.quant.name, hw.physical_cores, verify=True)
    return Path(res["path"])


def cmd_setup(args):
    print("Step 1/5  Detecting hardware")
    hw = hardware.detect()
    print("  " + hw.summary().replace("\n", "\n  "))
    print("\nStep 2/5  Benchmarking")
    gbps = benchmark.quick_benchmark()["effective_gbps"] if not args.skip_bench else benchmark.cached_or_run()["effective_gbps"]
    print(f"  usable memory bandwidth: {gbps:.1f} GB/s")
    print("\nStep 3/5  Choosing the best model for this machine")
    plan = _resolve_plan(args, hw, gbps)
    if not args.model:
        _print_plans(planner.candidates(hw, gbps, task=args.task, min_tps=args.min_tps), 3)
    print(f"\n  -> {plan.label}: ~{plan.memory_gb:.1f} GB RAM, ~{plan.est_tps:.1f} tokens/s, "
          f"{plan.quant.note}")
    if not args.yes:
        if input("\nProceed? [Y/n] ").strip().lower() in ("n", "no"):
            return
    print("\nStep 4/5  Installing runtime and model")
    if not runtime.find("llama-server"):
        note = runtime.platform_note(hw)
        if note:
            print("  " + note)
        runtime.install(hw)
    path = _obtain_model(plan, hw, args.local_compress)
    s = tuner.settings(plan, hw)
    config.save_state(active={"path": str(path), "plan": plan.to_dict(), "settings": s})
    print(f"  model: {path}")
    print(f"  settings: {s}")
    if args.no_run:
        print("\nDone. Start it any time with `tlav run`.")
        return
    print("\nStep 5/5  Starting your local AI")
    _run_active(args.host, args.port, not args.no_browser)


def _run_active(host, port, browser, model_path=None):
    state = config.load_state()
    active = state.get("active")
    if model_path:
        hw = hardware.detect()
        size = Path(model_path).stat().st_size / 1024 ** 3
        s = {"threads": hw.physical_cores, "threads_batch": hw.logical_cores, "ctx": 4096,
             "batch": 512 if hw.ram_total_gb >= 12 else 256, "ubatch": 256, "mlock": False, "gpu_layers": 0}
        if hw.ram_total_gb - size < 4:
            s["ctx"] = 2048
    elif active:
        model_path, s = active["path"], active["settings"]
    else:
        sys.exit("No model set up yet. Run `tlav setup` first (or `tlav run --model file.gguf`).")
    args = tuner.server_args(model_path, s, host=host, port=port)
    print(f"Loading {Path(model_path).name} (first load can take a minute on old disks)...")
    proc = server.start(args)
    url = config.load_state()["server_url"]
    print(f"\nReady!  Web chat: {url}   OpenAI-compatible API: {url}/v1")
    print("Press Ctrl+C to stop.")
    if browser:
        server.open_browser(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nStopped.")


def cmd_run(args):
    _run_active(args.host, args.port, not args.no_browser, args.model)


def cmd_chat(args):
    url = args.url or config.load_state().get("server_url", "http://127.0.0.1:8080")
    if not server.healthy(url):
        sys.exit(f"No server at {url}. Start one with `tlav run` in another terminal.")
    server.chat_loop(url, args.system)


def cmd_compress(args):
    hw = hardware.detect()
    if args.file:
        src = Path(args.file).expanduser()
        if not src.exists():
            sys.exit(f"{src} not found")
        target = args.to or "Q4_K_M"
    else:
        model = catalog.get(args.model)
        if args.fit or not args.to:
            gbps = benchmark.cached_or_run()["effective_gbps"]
            plan = planner.best_quant_for(model, hw, gbps)
            if not plan:
                sys.exit(f"{model.name} does not fit this machine even at Q3_K_M.")
            target = plan.quant.name
            print(f"Least compression that fits this machine: {target} (~{plan.memory_gb:.1f} GB)")
        else:
            target = args.to
        src = compress.fetch_source(model)
    res = compress.compress(src, target, hw.physical_cores, verify=not args.no_verify,
                            tolerance_pct=args.tolerance)
    print(json.dumps(res, indent=2))


def cmd_list(args):
    files = sorted(config.models_dir().glob("*.gguf"))
    active = config.load_state().get("active", {}).get("path")
    if not files:
        print("No models downloaded yet.")
    for f in files:
        mark = "*" if str(f) == active else " "
        print(f"{mark} {f.name:<55} {f.stat().st_size / 1024 ** 3:6.2f} GB")


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tlav", description="Turn old PCs and laptops into local AI workstations.")
    p.add_argument("--version", action="version", version=f"tlav {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="detect hardware and show what Tlav sees")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("bench", help="benchmark memory bandwidth (and real tokens/s with --real)")
    s.add_argument("--real", action="store_true", help="run llama-bench on a model")
    s.add_argument("--model", help="GGUF file for --real")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("models", help="show the model catalog and compression levels")
    s.set_defaults(func=cmd_models)

    def plan_opts(s):
        s.add_argument("--task", choices=["chat", "code"], default="chat")
        s.add_argument("--min-tps", type=float, default=2.5, help="slowest acceptable tokens/second")
        s.add_argument("--aggressive", action="store_true", help="allow Q3_K_M compression to fit bigger models")

    s = sub.add_parser("recommend", help="rank models for this (or a simulated) machine")
    plan_opts(s)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--json", action="store_true")
    g = s.add_argument_group("simulate another machine")
    g.add_argument("--ram", type=float, help="RAM in GB")
    g.add_argument("--cores", type=int, default=4)
    g.add_argument("--simd", default="avx2", choices=["baseline", "avx", "avx2", "avx512f", "neon"])
    g.add_argument("--gpu", default="none", choices=["none", "cuda", "metal"])
    g.add_argument("--vram", type=float, default=0.0)
    g.add_argument("--bandwidth", type=float, help="usable GB/s (default 16, dual-channel DDR3)")
    s.set_defaults(func=cmd_recommend)

    s = sub.add_parser("install-runtime", help="install llama.cpp tuned for this CPU")
    s.add_argument("--source", action="store_true", help="always build from source")
    s.add_argument("--prebuilt", action="store_true", help="only use prebuilt binaries")
    s.set_defaults(func=cmd_install_runtime)

    s = sub.add_parser("setup", help="do everything: detect, benchmark, pick, compress, run")
    plan_opts(s)
    s.add_argument("--model", help="catalog id to force (see `tlav models`)")
    s.add_argument("--quant", choices=[q.name for q in catalog.QUANTS[1:]])
    s.add_argument("--local-compress", action="store_true",
                   help="compress on this machine instead of downloading a pre-compressed file")
    s.add_argument("--skip-bench", action="store_true")
    s.add_argument("--yes", "-y", action="store_true")
    s.add_argument("--no-run", action="store_true")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to use it from other devices on your network")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("compress", help="compress (quantize) a model and verify its quality")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="catalog id")
    src.add_argument("--file", help="your own .gguf file (F16/BF16/Q8_0)")
    s.add_argument("--to", choices=[q.name for q in catalog.QUANTS[1:]])
    s.add_argument("--fit", action="store_true", help="pick the least compression that fits this machine")
    s.add_argument("--no-verify", action="store_true")
    s.add_argument("--tolerance", type=float, default=3.0, help="max %% perplexity increase allowed")
    s.set_defaults(func=cmd_compress)

    s = sub.add_parser("run", help="start the local AI server + web chat")
    s.add_argument("--model", help="run a specific .gguf instead of the set-up one")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("chat", help="chat in the terminal with a running server")
    s.add_argument("--url")
    s.add_argument("--system", help="system prompt")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("list", help="list downloaded models")
    s.set_defaults(func=cmd_list)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except runtime.RuntimeMissing as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
