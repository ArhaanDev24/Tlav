from tlav import benchmark, catalog, finder, hardware, planner, tuner


def test_q8_kv_cache_halves_context_memory():
    m = catalog.get("qwen3-8b")
    f16 = m.kv_cache_gb(8192, catalog.KV_BYTES["f16"])
    q8 = m.kv_cache_gb(8192, catalog.KV_BYTES["q8_0"])
    assert 0.5 < q8 / f16 < 0.56


def test_q8_kv_lets_more_fit():
    # on an 8 GB machine, Qwen2.5 7B gets twice the context with the compressed cache
    hw, m, q = hardware.simulated(8), catalog.get("qwen2.5-7b"), catalog.QUANT_BY_NAME["Q4_K_M"]
    a = planner.evaluate(m, q, hw, 16.0, kv="f16")
    b = planner.evaluate(m, q, hw, 16.0, kv="q8_0")
    assert a is not None and b is not None and b.ctx == 2 * a.ctx


def test_max_quality_mode_only_near_lossless():
    for p in planner.candidates(hardware.simulated(32), 16.0, min_quant="Q6_K"):
        assert p.quant.name in ("Q6_K", "Q8_0")


def test_draft_attached_only_when_it_fits():
    hw = hardware.simulated(16)
    plan = planner.attach_draft(planner.plan_for_model(catalog.get("qwen3-8b"), [planner.Node(hw, 16.0)]))
    assert plan.draft is not None and plan.draft.id == "qwen3-0.6b"
    small = planner.attach_draft(planner.plan_for_model(catalog.get("qwen3-1.7b"), [planner.Node(hw, 16.0)]))
    assert small.draft is None


def test_server_args_adapt_to_llama_cpp_version():
    s = {"ctx": 4096, "threads": 4, "threads_batch": 4, "batch": 512, "ubatch": 256, "kv": "q8_0",
         "mlock": True, "rpc": [], "tensor_split": [], "gpu_layers": 0}
    new = tuner.server_args("m", s, draft_path="d",
                            help_text="--load-mode -fa [on|off|auto] --spec-type draft-simple --spec-draft-n-max -ctkd")
    assert "--load-mode" in new and "--spec-type" in new and "--spec-draft-n-max" in new
    old = tuner.server_args("m", s, draft_path="d", help_text="--mlock -fa, --flash-attn --draft-max")
    assert "--mlock" in old and "--draft-max" in old and old[old.index("-fa") + 1] == "-ctk"


def test_quant_from_file_uses_real_size():
    q = catalog.quant_from_file(4_680_000_000, 7.62, "Q4_K_M")
    assert 4.8 < q.bpw < 5.0 and q.retention == catalog.QUANT_BY_NAME["Q4_K_M"].retention


def test_estimate_quality_grows_with_size():
    assert catalog.estimate_quality(1) < catalog.estimate_quality(7) < catalog.estimate_quality(30)


def test_calibration_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("TLAV_HOME", str(tmp_path))
    assert benchmark.calibrate(100.0, 1.0) <= 3.0
    assert benchmark.calibrate(0.01, 10.0) >= 0.3


def test_finder_parses_repo_tree():
    tree = [{"type": "file", "path": "M-Q4_K_M.gguf", "lfs": {"size": 100}},
            {"type": "file", "path": "M-Q8_0.gguf", "size": 200},
            {"type": "file", "path": "mmproj-M-F16.gguf", "size": 5},
            {"type": "file", "path": "M-Q8_0-00001-of-00002.gguf", "size": 5},
            {"type": "file", "path": "M-Q8_0-f16.gguf", "size": 5},
            {"type": "file", "path": "README.md", "size": 1}]
    files = finder.parse_tree(tree)
    assert files == {"Q4_K_M": ("M-Q4_K_M.gguf", 100), "Q8_0": ("M-Q8_0.gguf", 200)}
