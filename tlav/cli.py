"""Tlav command line interface."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import (__version__, benchmark, catalog, chat, cluster, compress, config, discover, finder, gguf, hardware, hub,
               planner, runtime, server, service, tuner)


# ------------------------------------------------------------------ helpers

def _min_quant(args) -> str:
    if getattr(args, "max_quality", False):
        return "Q6_K"
    return catalog.AGGRESSIVE_MIN_QUANT if getattr(args, "aggressive", False) else catalog.DEFAULT_MIN_QUANT


def _nodes(args) -> List[planner.Node]:
    """This computer (real or simulated) plus cluster workers if requested."""
    rams = getattr(args, "ram", None)
    if rams:
        sims = [hardware.simulated(r, cores=args.cores, simd=args.simd, gpu=args.gpu, vram_gb=args.vram)
                for r in rams]
        bw = args.bandwidth or 16.0  # dual-channel DDR3-1600 (e.g. 2013 iMac) decodes at ~16 GB/s
        nodes = [planner.Node(sims[0], bw)]
        nodes += [planner.Node(h, bw, endpoint=f"simulated-{i}:50052", latency_ms=0.5, name=f"simulated PC {i}")
                  for i, h in enumerate(sims[1:], 2)]
        return nodes
    hw = hardware.detect()
    nodes = [planner.Node(hw, benchmark.cached_or_run()["effective_gbps"])]
    if getattr(args, "cluster", False):
        workers = cluster.saved_workers()
        if not workers:
            sys.exit("No cluster workers saved. Run `tlav worker` on the other PCs, then `tlav cluster scan`.")
        nodes += cluster.to_nodes(workers)
        if len(nodes) == 1:
            sys.exit("None of the saved workers are reachable.")
    return nodes


def _where(p: planner.Plan) -> str:
    if not p.is_cluster:
        return "this PC"
    return " + ".join(f"{n.name.split('.')[0]}:{s}L" for n, s in zip(p.nodes, p.split))


def _print_plans(plans, top):
    print(f"{'#':<3}{'Model':<30}{'Compression':<13}{'RAM use':>8}{'Speed':>12}{'Quality':>9}  {'Context':<8}Where")
    for i, p in enumerate(plans[:top], 1):
        extra = " +draft" if p.draft else ""
        print(f"{i:<3}{p.model.name[:29]:<30}{p.quant.name:<13}{p.memory_gb:>6.1f}GB"
              f"{p.est_tps:>8.1f} t/s{('~' if p.model.estimated else '') + f'{p.quality:.1f}':>9}  "
              f"{p.ctx:<8}{_where(p)}{extra}")
    if any(p.model.estimated for p in plans[:top]):
        print("~ quality estimated from size and release date (model found on Hugging Face)")


def _cluster_notes(plan: planner.Plan) -> None:
    if not plan.is_cluster:
        return
    print("\n  Cluster: layers are split across machines, so their RAM adds up. Every token passes through")
    print("  each machine in turn: this runs a BIGGER model, not a faster one. Use Ethernet if you can.")
    for n, s in zip(plan.nodes, plan.split):
        lat = f", {n.latency_ms:.1f} ms ({cluster.network_advice(n.latency_ms)})" if n.endpoint else ""
        print(f"    {n.name:<22} {s:>3} layers  budget {n.budget:.1f} GB{lat}")


def _choose(args, nodes, show: int = 5):
    """Rank the catalog for these machines, then check the top models against the files that
    really exist on Hugging Face (exact sizes, no weights downloaded). Returns ranked plans."""
    if not getattr(args, "offline", False):
        discover.refresh_if_stale()
    plans = planner.candidates(nodes[0].hw, nodes[0].gbps, task=args.task, min_tps=args.min_tps,
                               min_quant=_min_quant(args), kv=args.kv, workers=nodes[1:])
    if not plans:
        return []
    short = planner.shortlist(plans, show)
    if getattr(args, "offline", False):
        print(f"  Ranked {len(plans)} models by their specs (offline: sizes are estimates).")
        return plans
    checked = finder.refine(short, nodes, args.task, args.min_tps, _min_quant(args), args.kv)
    if all(c.plan is None and c.note.startswith("could not check") for c in checked):
        print(f"  Ranked {len(plans)} models by their specs (Hugging Face unreachable: sizes are estimates).")
        return plans
    good = planner.rank([c.plan for c in checked if c.plan])
    dropped = [c for c in checked if c.plan is None]
    print(f"  Ranked {len(plans)} models by your PC's specs, then checked the top {len(short)} "
          f"against their real files:")
    for c in dropped:
        print(f"    - {c.estimate.model.name}: {c.note}")
    if not good:  # every shortlisted model failed the real check; fall back to the rest of the ranking
        return plans[len(short):] or plans
    return good + [p for p in plans[len(short):]]


# ------------------------------------------------------------------ info commands

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
    print("Cluster support:    ", "yes" if rt["rpc-server"] else "no (reinstall runtime: `tlav install-runtime`)")
    workers = cluster.saved_workers()
    if workers:
        print(f"Cluster workers:     {len(workers)} saved ({', '.join(w['endpoint'] for w in workers)})")
    config.save_state(hardware=hw.to_dict())


def cmd_bench(args):
    print("Measuring memory bandwidth (this decides how fast models can run)...")
    b = benchmark.quick_benchmark()
    print(f"  raw memory bandwidth:  {b['bandwidth_gbps']:.1f} GB/s")
    print(f"  usable for inference:  {b['effective_gbps']:.1f} GB/s")
    if not args.real:
        return
    active = config.load_state().get("active", {})
    model = args.model or active.get("path")
    if not model:
        sys.exit("No model to benchmark; pass --model PATH or run `tlav setup` first.")
    hw = hardware.detect()
    print(f"Running llama-bench on {Path(model).name} ...")
    r = benchmark.real_benchmark(model, hw.physical_cores, runtime.require("llama-bench"))
    if not r:
        print("  llama-bench failed.")
        return
    print(f"  prompt processing: {r.get('prompt_tps', 0):.1f} tok/s")
    print(f"  generation:        {r.get('gen_tps', 0):.1f} tok/s")
    info = gguf.read(model)
    m = info.to_model("bench")
    q = catalog.quant_from_file(Path(model).stat().st_size, m.params_b)
    predicted = planner.estimate_tps(m, q, b["effective_gbps"], hw)
    cal = benchmark.calibrate(r.get("gen_tps", 0), predicted)
    print(f"  predicted {predicted:.1f} tok/s -> calibration factor now {cal:.2f} (used for future estimates)")


def cmd_models(args):
    if args.update:
        try:
            print(f"Hand-scored catalog updated: {catalog.update()} models.")
        except Exception as e:
            print(f"Hand-scored catalog not updated ({str(e)[:60]}); using the built-in one.")
        print("Searching Hugging Face for new original models from trusted publishers...")
        try:
            found = discover.update()
        except Exception as e:
            sys.exit(f"Could not reach Hugging Face ({e}).")
        print(f"{len(found)} more models are now part of every search. `tlav models` lists them all.")
        return
    print(f"{'ID':<22}{'Name':<28}{'Params':>8}  Q4_K_M size  Tags")
    q4 = catalog.QUANT_BY_NAME["Q4_K_M"]
    models = catalog.load()
    for m in sorted(models, key=lambda m: (m.estimated, m.params_b)):
        size = f"{m.params_b:.1f}B" + ("*" if m.is_moe else "")
        src = "  (found on Hugging Face)" if m.estimated else ""
        print(f"{m.id[:21]:<22}{m.name[:27]:<28}{size:>8}  {m.weights_gb(q4):>8.1f} GB  {', '.join(m.tags)}{src}")
    print("\n* mixture-of-experts: reads only a few billion parameters per token, so it runs fast.")
    age = discover.age_days()
    print(f"{sum(m.estimated for m in models)} models found on Hugging Face "
          + (f"{age:.0f} days ago" if age is not None else "(none yet)") + "; `tlav models --update` refreshes.")
    print("Not listed? `tlav search <name>` finds any GGUF model on Hugging Face.")
    print("\nCompression levels:")
    for q in catalog.QUANTS:
        print(f"  {q.name:<8} {q.bpw:>5.2f} bits/weight  keeps ~{q.retention * 100:.1f}% quality  ({q.note})")


def cmd_recommend(args):
    nodes = _nodes(args)
    total = sum(n.budget for n in nodes)
    hw = nodes[0].hw
    who = f"{len(nodes)} computers" if len(nodes) > 1 else f"{hw.ram_total_gb:.0f} GB RAM, {hw.physical_cores} cores"
    if not args.json:
        print(f"Machine: {who}, {hw.simd.upper()}, {hw.gpu}; model budget {total:.1f} GB, "
              f"~{nodes[0].gbps:.1f} GB/s usable bandwidth, context cache {args.kv}")
    plans = _choose(args, nodes, max(5, args.top)) if not args.json else planner.candidates(
        nodes[0].hw, nodes[0].gbps, task=args.task, min_tps=args.min_tps, min_quant=_min_quant(args),
        kv=args.kv, workers=nodes[1:])
    plans = [planner.attach_draft(p) for p in plans] if not args.no_draft else plans
    if not args.json:
        print()
    if not plans:
        print("No model fits comfortably. Try --min-tps 1 or --aggressive.")
        return
    if args.json:
        print(json.dumps([p.to_dict() for p in plans[: args.top]], indent=2))
        return
    _print_plans(plans, args.top)
    best = plans[0]
    print(f"\nBest pick: {best.label}  ->  `tlav setup{' --cluster' if best.is_cluster else ''}`")
    print(f"  why: {finder.why(best, plans[1] if len(plans) > 1 else None, args.min_tps)}")
    _cluster_notes(best)


def cmd_search(args):
    nodes = _nodes(args)
    print(f"Searching Hugging Face for '{args.query}' and checking each model against your "
          f"{'cluster' if len(nodes) > 1 else 'machine'}...\n")
    results = finder.search(args.query, nodes, limit=args.top, task=args.task, min_tps=args.min_tps,
                            min_quant=_min_quant(args))
    print(f"{'Repository':<52}{'Params':>8}  {'Best fit':<12}{'RAM':>7}{'Speed':>11}  Downloads")
    for r in results:
        params = f"{r.model.params_b:.1f}B" if r.info else "?"
        if r.plan:
            p = r.plan
            fit = f"{p.quant.name:<12}{p.memory_gb:>5.1f}GB{p.est_tps:>7.1f} t/s"
        else:
            fit = f"{'-':<12}{(r.error or 'too big/slow'):<23}"[:35]
        print(f"{r.repo[:51]:<52}{params:>8}  {fit}  {r.downloads:,}")
    print("\nInstall one with:  tlav setup --hf <repository>")


# ------------------------------------------------------------------ runtime

def cmd_install_runtime(args):
    hw = hardware.detect()
    note = runtime.platform_note(hw)
    if note:
        print(note)
    prefer = "source" if args.source else ("prebuilt" if args.prebuilt else "auto")
    path = runtime.install(hw, prefer=prefer)
    print(f"llama.cpp ready in {path}")


# ------------------------------------------------------------------ setup

def _resolve_plan(args, nodes):
    """Returns (plan, files) where files maps quant -> filename when using --hf."""
    if args.hf:
        model, files = finder.model_from_repo(args.hf)
        sizes = {q: s for q, (_, s) in finder.repo_files(args.hf).items()}
        quants = [catalog.quant_from_file(sizes[q], model.params_b, q) for q in files]
        if args.quant:
            quants = [q for q in quants if q.name == args.quant.upper()]
        plan = planner.plan_for_model(model, nodes, args.task, args.min_tps, _min_quant(args), args.kv, quants)
        if not plan:
            plan = planner.plan_for_model(model, nodes, args.task, 0.5, catalog.AGGRESSIVE_MIN_QUANT, args.kv, quants)
            if plan:
                print(f"  note: {args.hf} will be slow here (~{plan.est_tps:.1f} tok/s)")
        if not plan:
            sys.exit(f"{args.hf} does not fit this {'cluster' if len(nodes) > 1 else 'machine'}.")
        return plan, files
    if args.model:
        model = catalog.get(args.model)
        if args.quant:
            q = catalog.QUANT_BY_NAME[args.quant]
            plan = planner.evaluate_nodes(model, q, nodes[:1], args.task, args.kv)
            if not plan and len(nodes) > 1:
                plan = planner.evaluate_nodes(model, q, nodes, args.task, args.kv)
            if not plan:
                sys.exit(f"{model.name} at {args.quant} does not fit.")
            return plan, None
        plan = planner.plan_for_model(model, nodes, args.task, 0.5, _min_quant(args), args.kv)
        if not plan:
            sys.exit(f"{model.name} does not fit even when compressed. Try --cluster with another PC.")
        return plan, None
    plans = _choose(args, nodes)
    if not plans:
        sys.exit("No model fits comfortably. Try `tlav setup --aggressive`, `--min-tps 1`, or a cluster.")
    print()
    _print_plans(plans, 3)
    print(f"\n  Best for this {'cluster' if plans[0].is_cluster else 'PC'}: {plans[0].label}")
    print(f"  why: {finder.why(plans[0], plans[1] if len(plans) > 1 else None, args.min_tps)}")
    _custom_tip(plans[0], args)
    return plans[0], None


def _obtain_model(plan, hw, local_compress: bool, files: Optional[dict] = None, custom: bool = False,
                  min_tps: float = 0.0) -> Path:
    if custom and not plan.is_cluster:
        fit = planner.fit_target(plan.model, plan.nodes[0], plan.kv, min_tps=min_tps)
        if fit:
            src = compress.fetch_source(plan.model)
            res = compress.compress_to_size(src, fit[0], hw.physical_cores)
            plan.quant = catalog.quant_from_file(int(res["size_gb"] * 1024 ** 3), plan.model.params_b, "FIT")
            plan.ctx = fit[1]
            return Path(res["path"])
        print("  (no room for an exact-fit build; using a ready-made file)")
    if files:  # --hf repo
        fn = files[plan.quant.name]
        print(f"Downloading {fn} ...")
        return hub.download(plan.model.repo, fn)
    names = hub.list_files(plan.model.repo)
    ready = hub.find_quant_file(names, plan.quant.name)
    if ready and not local_compress:
        have = (config.models_dir() / ready).exists()
        print(f"{'Using the downloaded' if have else 'Downloading pre-compressed'} {ready} ...")
        return hub.download(plan.model.repo, ready)
    src = compress.fetch_source(plan.model)
    res = compress.compress(src, plan.quant.name, hw.physical_cores, verify=True)
    return Path(res["path"])


def _custom_tip(plan, args):
    """Tell the user when an exact-fit build would use clearly more of their RAM."""
    if getattr(args, "custom", False) or plan.is_cluster:
        return
    fit = planner.fit_target(plan.model, plan.nodes[0], plan.kv, min_tps=args.min_tps)
    weights = plan.model.weights_gb(plan.quant)
    if fit and fit[0] > weights * 1.12:
        bpw = fit[0] * 8 * 1024 ** 3 / (plan.model.params_b * 1e9)
        q = catalog.Quant("FIT", bpw, catalog.retention_for("", bpw), "")
        tps = planner.estimate_tps(plan.model, q, plan.nodes[0].gbps, plan.nodes[0].hw)
        print(f"  tip: `tlav setup --custom` builds a {fit[0]:.1f} GB version of {plan.model.name} sized exactly for "
              f"this PC ({bpw:.1f} bits/weight vs {plan.quant.bpw:.1f}, ~{tps:.1f} tokens/s); it downloads the "
              f"original weights first.")


def _obtain_draft(plan) -> Optional[Path]:
    if not plan.draft:
        return None
    names = hub.list_files(plan.draft.repo)
    fn = hub.find_quant_file(names, "Q8_0")
    if not fn:
        plan.draft = None
        return None
    print(f"Downloading draft model {fn} (for faster, identical answers) ...")
    return hub.download(plan.draft.repo, fn)


def cmd_setup(args):
    print("Step 1/5  Detecting hardware")
    hw = hardware.detect()
    print("  " + hw.summary().replace("\n", "\n  "))
    print("\nStep 2/5  Benchmarking")
    if not args.skip_bench:
        benchmark.quick_benchmark()
    nodes = _nodes(args)
    print(f"  usable memory bandwidth: {nodes[0].gbps:.1f} GB/s")
    if len(nodes) > 1:
        print(f"  cluster: {len(nodes) - 1} worker(s): " + ", ".join(f"{n.name} ({n.budget:.1f} GB)" for n in nodes[1:]))
    print("\nStep 3/5  Choosing the best model")
    plan, files = _resolve_plan(args, nodes)
    if not args.no_draft and not files:
        plan = planner.attach_draft(plan)
    print(f"\n  -> {plan.label}: ~{plan.memory_gb:.1f} GB RAM, ~{plan.est_tps:.1f} tokens/s, {plan.quant.note}")
    print(f"     context cache: {plan.kv} ({'half the memory of f16, quality checked' if plan.kv != 'f16' else 'full precision'})")
    if plan.draft:
        print(f"     speculative decoding with {plan.draft.name}: same answers, usually faster")
    _cluster_notes(plan)
    if not args.yes and input("\nProceed? [Y/n] ").strip().lower() in ("n", "no"):
        return
    print("\nStep 4/5  Installing runtime and model")
    if not runtime.find("llama-server") or (plan.is_cluster and not runtime.find("rpc-server")):
        note = runtime.platform_note(hw)
        if note:
            print("  " + note)
        runtime.install(hw)
    path = _obtain_model(plan, hw, args.local_compress, files, custom=args.custom, min_tps=args.min_tps)
    draft = _obtain_draft(plan)
    if args.kv_check and plan.kv != "f16":
        print("  Checking that the compressed context cache keeps quality...")
        res = compress.kv_check(path, hw.physical_cores, plan.kv)
        print(f"  context cache {res['kv']}: {res.get('delta_pct', '?')}% perplexity change")
        plan.kv = res["kv"]
    print(f"  model: {path}")
    if not plan.is_cluster and not args.skip_bench and runtime.find("llama-bench"):
        measured = _speed_check(plan, path, hw, nodes)
        if (measured and measured < args.min_tps * 0.6 and not getattr(args, "_rechecked", False)
                and not (args.model or args.hf)):
            print(f"  That is slower than this PC needs ({args.min_tps:g} tokens/s).")
            if args.yes or input("  Pick again using the measured speed? [Y/n] ").strip().lower() not in ("n", "no"):
                args._rechecked = True
                args.skip_bench = True
                print()
                return cmd_setup(args)
    s = tuner.settings(plan, hw, low_ram=args.low_ram)
    config.save_state(active={"path": str(path), "draft_path": str(draft) if draft else None,
                              "plan": plan.to_dict(), "settings": s})
    if args.no_run:
        print("\nDone. Type `tlav` any time to chat, in the terminal or your browser"
              + (" - keep the workers running." if plan.is_cluster else "."))
        return
    print("\nStep 5/5  Starting your local AI")
    if args.detach:
        _launch(path, s, args.host, args.port, not args.no_browser, draft=draft, detach=True)
        return
    _launch(path, s, args.host, args.port, False, draft=draft, detach=True, quiet=True)
    print("  Ready.")
    _open_where(config.load_state()["server_url"], "web" if args.web else _ask_where())


def _ask_where(default: str = "terminal") -> str:
    """Terminal or web browser? Asked after setup and every time `tlav` opens."""
    if not sys.stdin.isatty():
        return default
    print("\n  Where do you want to chat?")
    print("    1  Here in the terminal")
    print("    2  In your web browser")
    try:
        ans = input("  Choose 1 or 2 [1]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return "web" if ans in ("2", "w", "web", "b", "browser") else "terminal"


def _open_where(url: str, where: str) -> None:
    if where == "web":
        server.open_browser(url)
        print(f"\n  Opened {url} in your browser.")
        print("  Tlav keeps running in the background: `tlav` brings you back, `tlav stop` frees its memory.")
        return
    _open_chat(url)


def cmd_open(args):
    """Plain `tlav`: start the set-up model if needed, then ask terminal or web."""
    st = server.status()
    url = st["url"] if st["running"] else None
    if not url:
        print(chat.dim("  Starting your model (first start can take a minute on old disks)..."))
        url = _start_active(args.port)
    _open_where(url, args.where or _ask_where())


def _speed_check(plan, path, hw, nodes):
    """Measure the installed model once (no extra download) and learn this PC's real speed."""
    print("  Measuring its real speed on this PC (about a minute)...")
    r = benchmark.real_benchmark(str(path), hw.physical_cores, runtime.require("llama-bench"))
    if not r or not r.get("gen_tps"):
        print("  (speed test skipped: llama-bench did not run)")
        return
    measured, predicted = r["gen_tps"], plan.est_tps
    cal = benchmark.calibrate(measured, predicted)
    print(f"  measured {measured:.1f} tokens/s (estimated {predicted:.1f}); future picks use this PC's real speed")
    plan.est_tps = round(measured, 1)
    return measured


# ------------------------------------------------------------------ run / chat / status

def _plan_for_file(path: Path, nodes, kv: str, force_split: bool = False) -> Optional[planner.Plan]:
    try:
        info = gguf.read(str(path))
    except (ValueError, OSError, EOFError) as e:
        print(f"  could not read model header ({e}); using safe defaults")
        return None
    model = info.to_model(path.stem)
    q = catalog.quant_from_file(path.stat().st_size, model.params_b)
    plan = None if force_split else planner.evaluate_nodes(model, q, nodes[:1], kv=kv)
    if not plan and len(nodes) > 1:
        plan = planner.evaluate_nodes(model, q, nodes, kv=kv)
    return plan


def _launch(model_path, s, host, port, browser, draft=None, detach=False, quiet=False):
    args = tuner.server_args(str(model_path), s, host=host, port=port,
                             draft_path=str(draft) if draft else None)
    print(f"Loading {Path(model_path).name}"
          + (f" across {len(s['rpc']) + 1} computers" if s.get("rpc") else "")
          + " (first load can take a few minutes on old disks/networks)...")
    proc = server.start(args, detach=detach)
    url = config.load_state()["server_url"]
    if quiet:
        return proc
    print(f"\nReady!  Web chat: {url}   OpenAI-compatible API: {url}/v1")
    if host == "0.0.0.0":
        print(f"From phones/other PCs on your network: http://{cluster.lan_ip()}:{port}")
    if browser:
        server.open_browser(url)
    if detach:
        print("Running in the background. `tlav status` to check, `tlav stop` to stop.")
        return
    print("Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nStopped.")


def cmd_run(args):
    if server.status()["running"]:
        sys.exit("Tlav is already running (`tlav status`). Stop it first with `tlav stop`.")
    nodes = _nodes(args) if args.cluster else None
    if args.model:
        path = Path(args.model).expanduser()
        hw = hardware.detect()
        nodes = nodes or [planner.Node(hw, benchmark.cached_or_run()["effective_gbps"])]
        plan = _plan_for_file(path, nodes, args.kv, force_split=args.force_split)
        if plan:
            s = tuner.settings(plan, hw, low_ram=args.low_ram)
            if plan.is_cluster:
                _cluster_notes(plan)
        else:
            s = {"threads": hw.physical_cores, "threads_batch": hw.logical_cores, "ctx": 2048, "batch": 256,
                 "ubatch": 128, "mlock": False, "gpu_layers": 0, "kv": args.kv, "rpc": [], "tensor_split": []}
        return _launch(path, s, args.host, args.port, not args.no_browser, detach=args.detach)
    active = config.load_state().get("active")
    if not active:
        sys.exit("No model set up yet. Run `tlav setup` first (or `tlav run --model file.gguf`).")
    s = active["settings"]
    if args.cluster and not s.get("rpc"):
        sys.exit("The set-up model runs on this PC alone. Re-run `tlav setup --cluster` to plan a cluster.")
    if s.get("rpc"):
        down = [e for e in s["rpc"] if cluster.ping(e, tries=1) is None]
        if down:
            sys.exit(f"Cluster worker(s) not reachable: {', '.join(down)}. Start `tlav worker` there first.")
    draft = active.get("draft_path") if not args.no_draft else None
    _launch(active["path"], s, args.host, args.port, not args.no_browser, draft=draft, detach=args.detach)


def cmd_status(args):
    st = server.status()
    if not st["running"]:
        print("Tlav is not running. Start it with `tlav run` (add --detach to run in the background).")
    else:
        print(f"Running (pid {st['pid']}) at {st['url']}")
        mem = st.get("memory") or {}
        if mem.get("used") is not None:
            extra = f" (+{mem['cache']:.2f} GB file cache the OS can reclaim)" if mem.get("cache") else ""
            print(f"  memory in use: {mem['used']:.2f} GB{extra}")
        if st.get("model_path"):
            print(f"  model: {Path(st['model_path']).name}   context: {st.get('ctx')}")
    act = st.get("active")
    if act:
        p = act["plan"]
        print(f"Set-up model: {p['model']} @ {p['quant']}, ~{p['memory_gb']} GB, ~{p['est_tps']} tok/s"
              + (f", cluster {p['nodes']}" if p.get("nodes") else ""))
    if args.json:
        print(json.dumps(st, indent=2, default=str))


def cmd_stop(args):
    print("Stopped." if server.stop() else "Tlav was not running.")


def _start_active(port: int = 8080, host: str = "127.0.0.1") -> str:
    """Start the set-up model in the background (used by the chat). Returns its URL."""
    active = config.load_state().get("active")
    if not active:
        sys.exit("No model set up yet. Run `tlav` (or `tlav setup`) and Tlav picks and installs one.")
    if not Path(active["path"]).exists():
        sys.exit(f"The model file is missing ({active['path']}). Run `tlav setup` again.")
    s = active["settings"]
    if s.get("rpc"):
        down = [e for e in s["rpc"] if cluster.ping(e, tries=1) is None]
        if down:
            sys.exit(f"Cluster worker(s) not reachable: {', '.join(down)}. Start `tlav worker` there first.")
    _launch(active["path"], s, host, port, False, draft=active.get("draft_path"), detach=True, quiet=True)
    return config.load_state()["server_url"]


def _open_chat(url: str, system=None, info=None, think=None):
    info = info or chat.model_info(config.load_state().get("active"))
    chat.Chat(url, info, system=system, think=think).loop()


def cmd_chat(args):
    think = {"on": True, "off": False}.get(args.think) if args.think else None
    if args.url:
        if not server.healthy(args.url):
            sys.exit(f"No model answers at {args.url}.")
        return _open_chat(args.url, args.system, {"model": args.url}, think)
    if args.model:  # a specific file: run it for this chat
        if server.status()["running"]:
            sys.exit("A model is already running; `tlav stop` first to chat with another file.")
        hw = hardware.detect()
        path = Path(args.model).expanduser()
        plan = _plan_for_file(path, [planner.Node(hw, benchmark.cached_or_run()["effective_gbps"])], planner.DEFAULT_KV)
        s = tuner.settings(plan, hw) if plan else {"threads": hw.physical_cores, "threads_batch": hw.logical_cores,
                                                   "ctx": 2048, "batch": 256, "ubatch": 128, "mlock": False,
                                                   "gpu_layers": 0, "kv": planner.DEFAULT_KV, "rpc": [],
                                                   "tensor_split": []}
        _launch(path, s, "127.0.0.1", args.port, False, detach=True, quiet=True)
        info = {"model": path.stem, "ctx": s["ctx"], "est_tps": plan.est_tps if plan else None}
        return _open_chat(config.load_state()["server_url"], args.system, info, think)
    st = server.status()
    url = st["url"] if st["running"] else None
    if not url:
        print(chat.dim("  Starting your model (first start can take a minute on old disks)..."))
        url = _start_active(args.port)
    _open_chat(url, args.system, think=think)


# ------------------------------------------------------------------ cluster

def cmd_worker(args):
    if not runtime.find("rpc-server"):
        hw = hardware.detect()
        print("This computer needs the AI engine with cluster support; installing it now.")
        runtime.install(hw)
    cluster.run_worker(host=args.host, port=args.port, token=args.token, discovery_port=args.discovery_port,
                       cache=not args.no_cache, threads=args.threads, mem_mb=args.mem)


def cmd_cluster(args):
    if args.action == "scan":
        print("Looking for Tlav workers on your network...")
        found = cluster.scan(timeout=args.timeout, token=args.token, port=args.discovery_port)
        if not found:
            print("None found. Make sure `tlav worker` runs on the other PCs, or add one by IP:\n"
                  "  tlav cluster add 192.168.1.23")
            return
        saved = []
        for w in found:
            lat = cluster.reachable(w)
            w.pop("candidates", None)
            if lat is None:
                print(f"  {w.get('hostname', '?'):<20} answered, but its engine at {w['endpoint']} is not reachable "
                      "(firewall? TCP 50052)")
                continue
            w["latency_ms"] = lat
            hw = w.get("hardware") or {}
            print(f"  {w['hostname']:<20} {w['endpoint']:<22} {hw.get('ram_total_gb', 0):>5.0f} GB RAM  "
                  f"{hw.get('physical_cores', '?')} cores  {w['latency_ms']:.1f} ms "
                  f"({cluster.network_advice(w['latency_ms'])})")
            saved.append(w)
        others = [x for x in cluster.saved_workers() if x["endpoint"] not in {w["endpoint"] for w in saved}]
        cluster.save_workers(others + saved)
        if saved:
            print(f"\nSaved. Now: `tlav recommend --cluster` or `tlav setup --cluster`.")
    elif args.action == "add":
        if not args.address:
            sys.exit("Usage: tlav cluster add IP[:PORT]")
        w = cluster.add_worker(args.address, token=args.token, port=args.discovery_port)
        print(f"Added {w['hostname']} at {w['endpoint']} ({w['latency_ms']:.1f} ms, "
              f"{cluster.network_advice(w['latency_ms'])})")
        if not w.get("hardware"):
            print("  (worker did not answer discovery; assuming 8 GB RAM. Open UDP 50053 for exact info.)")
    elif args.action == "remove":
        print("Removed." if cluster.remove_worker(args.address or "") else "No such worker.")
    elif args.action == "clear":
        cluster.save_workers([])
        print("Cluster cleared.")
    else:
        workers = cluster.saved_workers()
        if not workers:
            print("No workers saved. Run `tlav worker` on other PCs, then `tlav cluster scan`.")
        for w in workers:
            lat = cluster.ping(w["endpoint"], tries=1)
            state = (f"online, {lat:.1f} ms" if lat is not None
                     else "not answering (offline, or busy serving a running model)")
            ram = (w.get("hardware") or {}).get("ram_total_gb", "?")
            print(f"  {w.get('hostname', '?'):<20} {w['endpoint']:<22} {ram} GB RAM  {state}")


def cmd_service(args):
    role = "worker" if args.worker else "server"
    if args.action == "install":
        if role == "server" and not config.load_state().get("active"):
            sys.exit("Run `tlav setup --no-run` first so the service knows which model to start.")
        print(service.install(role, host=args.host, port=args.port))
    else:
        print(service.uninstall(role))


# ------------------------------------------------------------------ compress / list

def cmd_compress(args):
    hw = hardware.detect()
    if args.file:
        src = Path(args.file).expanduser()
        if not src.exists():
            sys.exit(f"{src} not found")
        target = args.to or "Q4_K_M"
        if args.fit:
            info = gguf.read(str(src))
            model = info.to_model(src.stem)
            plan = planner.best_quant_for(model, hw, benchmark.cached_or_run()["effective_gbps"])
            if not plan:
                sys.exit("This model does not fit this machine even at Q3_K_M. Try a cluster.")
            target = plan.quant.name
            print(f"Least compression that fits this machine: {target} (~{plan.memory_gb:.1f} GB)")
    else:
        model = catalog.get(args.model)
        if args.exact or args.size:
            target = None
        elif args.fit or not args.to:
            plan = planner.best_quant_for(model, hw, benchmark.cached_or_run()["effective_gbps"])
            if not plan:
                sys.exit(f"{model.name} does not fit this machine even at Q3_K_M.")
            target = plan.quant.name
            print(f"Least compression that fits this machine: {target} (~{plan.memory_gb:.1f} GB)")
        else:
            target = args.to
        src = compress.fetch_source(model)
    if args.exact or args.size:
        node = planner.Node(hw, benchmark.cached_or_run()["effective_gbps"])
        m = gguf.read(str(src)).to_model(src.stem) if args.file else model
        fit = (args.size, None) if args.size else planner.fit_target(m, node, min_tps=2.5)
        if not fit:
            sys.exit("Not enough RAM for this model even at ~4 bits per weight. Try a cluster.")
        res = compress.compress_to_size(src, fit[0], hw.physical_cores, use_imatrix=not args.no_imatrix,
                                        verify=not args.no_verify)
        print(json.dumps(res, indent=2))
        return
    res = compress.compress(src, target, hw.physical_cores, verify=not args.no_verify,
                            tolerance_pct=args.tolerance, use_imatrix=not args.no_imatrix)
    if args.kv_check:
        res["kv_check"] = compress.kv_check(Path(res["path"]), hw.physical_cores)
    print(json.dumps(res, indent=2))


def cmd_list(args):
    files = sorted(p for p in config.models_dir().glob("*.gguf") if ".imatrix." not in p.name)
    active = config.load_state().get("active", {}).get("path")
    if not files:
        print("No models downloaded yet.")
    for f in files:
        mark = "*" if str(f) == active else " "
        print(f"{mark} {f.name:<60} {f.stat().st_size / 1024 ** 3:6.2f} GB")


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tlav", description="Turn old PCs and laptops into local AI workstations.")
    p.add_argument("--version", action="version", version=f"tlav {__version__}")
    sub = p.add_subparsers(dest="cmd")

    def plan_opts(s, simulate=False):
        s.add_argument("--task", choices=["chat", "code"], default="chat")
        s.add_argument("--min-tps", type=float, default=2.5, help="slowest acceptable tokens/second")
        s.add_argument("--aggressive", action="store_true", help="allow Q3_K_M compression to fit bigger models")
        s.add_argument("--max-quality", action="store_true",
                       help="only near-lossless compression (Q6_K/Q8_0, <0.5%% measured change)")
        s.add_argument("--kv", choices=["q8_0", "f16"], default=planner.DEFAULT_KV,
                       help="context cache precision (q8_0 halves its memory)")
        s.add_argument("--cluster", action="store_true", help="also use saved `tlav worker` computers")
        s.add_argument("--no-draft", action="store_true", help="disable speculative decoding")
        if simulate:
            g = s.add_argument_group("simulate machines (repeat --ram for a cluster: --ram 16 --ram 16)")
            g.add_argument("--ram", type=float, action="append", help="RAM in GB")
            g.add_argument("--cores", type=int, default=4)
            g.add_argument("--simd", default="avx2", choices=["baseline", "avx", "avx2", "avx512f", "neon"])
            g.add_argument("--gpu", default="none", choices=["none", "cuda", "metal"])
            g.add_argument("--vram", type=float, default=0.0)
            g.add_argument("--bandwidth", type=float, help="usable GB/s (default 16, dual-channel DDR3)")

    def run_opts(s):
        s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to use it from other devices on your network")
        s.add_argument("--port", type=int, default=8080)
        s.add_argument("--no-browser", action="store_true")
        s.add_argument("--detach", action="store_true", help="run in the background")
        s.add_argument("--low-ram", action="store_true", help="smaller buffers, never lock memory")

    s = sub.add_parser("doctor", help="detect hardware and show what Tlav sees")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("bench", help="benchmark memory bandwidth (and real tokens/s with --real)")
    s.add_argument("--real", action="store_true", help="run llama-bench on a model and calibrate estimates")
    s.add_argument("--model", help="GGUF file for --real")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("models", help="show the model catalog and compression levels")
    s.add_argument("--update", action="store_true", help="download the latest catalog")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("recommend", help="rank models for this machine, a cluster, or a simulated setup")
    plan_opts(s, simulate=True)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--json", action="store_true")
    s.add_argument("--offline", action="store_true", help="don't check real file sizes on Hugging Face")
    s.set_defaults(func=cmd_recommend)

    s = sub.add_parser("search", help="find GGUF models on Hugging Face that fit this machine")
    s.add_argument("query")
    plan_opts(s, simulate=True)
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("install-runtime", help="install llama.cpp tuned for this CPU")
    s.add_argument("--source", action="store_true", help="always build from source")
    s.add_argument("--prebuilt", action="store_true", help="only use prebuilt binaries")
    s.set_defaults(func=cmd_install_runtime)

    s = sub.add_parser("setup", help="do everything: detect, benchmark, pick, compress, run")
    plan_opts(s)
    run_opts(s)
    src = s.add_mutually_exclusive_group()
    src.add_argument("--model", help="catalog id to force (see `tlav models`)")
    src.add_argument("--hf", help="any Hugging Face GGUF repo, e.g. bartowski/Qwen_Qwen3-8B-GGUF")
    s.add_argument("--quant", help="force a compression level, e.g. Q5_K_M")
    s.add_argument("--local-compress", action="store_true",
                   help="compress on this machine instead of downloading a pre-compressed file")
    s.add_argument("--kv-check", action="store_true", help="measure the context-cache quality impact first")
    s.add_argument("--custom", action="store_true",
                   help="build a file sized exactly for this PC from the original weights (best quality, big download)")
    s.add_argument("--skip-bench", action="store_true")
    s.add_argument("--yes", "-y", action="store_true")
    s.add_argument("--no-run", action="store_true")
    s.add_argument("--web", action="store_true", help="open the browser chat without asking")
    s.add_argument("--offline", action="store_true", help="pick from specs only, without checking file sizes online")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("compress", help="compress (quantize) a model and verify its quality")
    srcg = s.add_mutually_exclusive_group(required=True)
    srcg.add_argument("--model", help="catalog id")
    srcg.add_argument("--file", help="your own .gguf file (F16/BF16/Q8_0)")
    s.add_argument("--to", choices=[q.name for q in catalog.QUANTS[1:]])
    s.add_argument("--fit", action="store_true", help="pick the least compression that fits this machine")
    s.add_argument("--exact", action="store_true",
                   help="build a file sized exactly for this PC's RAM (precision chosen per tensor)")
    s.add_argument("--size", type=float, help="build a file of exactly this many GB")
    s.add_argument("--no-verify", action="store_true")
    s.add_argument("--no-imatrix", action="store_true", help="skip importance-matrix compression")
    s.add_argument("--kv-check", action="store_true", help="also test a compressed context cache")
    s.add_argument("--tolerance", type=float, default=3.0, help="max %% perplexity increase allowed")
    s.set_defaults(func=cmd_compress)

    s = sub.add_parser("run", help="start the local AI server + web chat")
    s.add_argument("--model", help="run a specific .gguf instead of the set-up one")
    s.add_argument("--cluster", action="store_true", help="use saved workers (with --model)")
    s.add_argument("--force-split", action="store_true",
                   help="with --cluster: split across machines even if the model fits here")
    s.add_argument("--kv", choices=["q8_0", "f16"], default=planner.DEFAULT_KV)
    s.add_argument("--no-draft", action="store_true")
    s.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)  # default; used by services
    run_opts(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="is Tlav running, with what, using how much memory")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("stop", help="stop a background server")
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("open", help="start your model and choose terminal or web chat (what plain `tlav` does)")
    s.add_argument("--where", choices=["terminal", "web"], help="skip the question")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(func=cmd_open)

    s = sub.add_parser("chat", help="the Tlav chat terminal (starts your model if needed)")
    s.add_argument("--url", help="chat with a model server somewhere else")
    s.add_argument("--model", help="chat with a specific .gguf file")
    s.add_argument("--system", help="system prompt")
    s.add_argument("--think", choices=["on", "off"], help="model reasoning (default: on if this PC is fast)")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("worker", help="share this PC's RAM/CPU with a main Tlav computer (cluster)")
    s.add_argument("--host", help="interface to listen on (default: this PC's LAN address)")
    s.add_argument("--port", type=int, default=cluster.RPC_PORT)
    s.add_argument("--discovery-port", type=int, default=cluster.DISCOVERY_PORT)
    s.add_argument("--token", help="only answer scans that present this shared word")
    s.add_argument("--no-cache", action="store_true", help="don't keep received weights on disk")
    s.add_argument("--threads", type=int)
    s.add_argument("--mem", type=int, help="limit shared memory, in MB")
    s.set_defaults(func=cmd_worker)

    s = sub.add_parser("cluster", help="manage the computers that work together")
    s.add_argument("action", choices=["scan", "add", "remove", "list", "clear"], nargs="?", default="list")
    s.add_argument("address", nargs="?", help="IP[:PORT] for add/remove")
    s.add_argument("--token")
    s.add_argument("--timeout", type=float, default=2.0)
    s.add_argument("--discovery-port", type=int, default=cluster.DISCOVERY_PORT)
    s.set_defaults(func=cmd_cluster)

    s = sub.add_parser("service", help="start Tlav automatically at login (always-on AI box)")
    s.add_argument("action", choices=["install", "uninstall"])
    s.add_argument("--worker", action="store_true", help="run as a cluster worker instead of the server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(func=cmd_service)

    s = sub.add_parser("list", help="list downloaded models")
    s.set_defaults(func=cmd_list)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:  # just `tlav`: chat if a model is set up, otherwise set one up
        argv = ["open"] if config.load_state().get("active") else ["setup"]
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (runtime.RuntimeMissing, ConnectionError, KeyError) as e:
        sys.exit(str(e).strip("'\""))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
