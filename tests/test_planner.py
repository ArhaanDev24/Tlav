from tlav import catalog, hardware, planner


def test_catalog_is_valid():
    models = catalog.load()
    ids = [m.id for m in models]
    assert len(ids) == len(set(ids))
    for m in models:
        assert m.repo.count("/") == 1
        assert m.params_b > 0 and m.layers > 0 and m.kv_heads > 0 and m.head_dim > 0
        assert "chat" in m.quality


def test_compression_shrinks_weights():
    m = catalog.get("qwen3-8b")
    f16 = m.weights_gb(catalog.QUANT_BY_NAME["F16"])
    q4 = m.weights_gb(catalog.QUANT_BY_NAME["Q4_K_M"])
    assert 14 < f16 < 16
    assert 4 < q4 < 5
    assert f16 / q4 > 3


def test_moe_reads_only_active_params():
    m = catalog.get("qwen3-30b-a3b")
    q = catalog.QUANT_BY_NAME["Q4_K_M"]
    assert m.is_moe
    assert m.active_gb(q) < m.weights_gb(q) / 5


def test_plans_fit_budget():
    for ram in (4, 8, 16, 32, 64):
        hw = hardware.simulated(ram)
        for p in planner.candidates(hw, 16.0):
            assert p.memory_gb <= planner.ram_budget_gb(hw)
            assert p.est_tps >= 2.5
            assert p.ctx >= planner.MIN_CTX


def test_imac_2013_16gb_gets_a_7_to_8b_model():
    best = planner.best(hardware.simulated(16, cores=4, simd="avx2"), 16.0)
    assert best is not None
    assert 7 <= best.model.params_b <= 9.5
    assert best.quant.name in ("Q4_K_M", "Q5_K_M", "Q6_K")


def test_more_ram_never_gives_worse_model():
    prev = 0.0
    for ram in (4, 8, 16, 32, 64):
        best = planner.best(hardware.simulated(ram), 16.0)
        q = best.quality if best else 0.0
        assert q >= prev
        prev = q


def test_default_never_compresses_below_q4():
    for ram in (4, 8, 16, 32):
        for p in planner.candidates(hardware.simulated(ram), 16.0):
            assert p.quant.bpw >= catalog.QUANT_BY_NAME["Q4_K_M"].bpw


def test_best_quant_for_prefers_least_compression():
    m = catalog.get("qwen3-4b")
    plan = planner.best_quant_for(m, hardware.simulated(64), 16.0)
    assert plan.quant.name == "Q8_0"
    small = planner.best_quant_for(m, hardware.simulated(6), 16.0)
    assert small is not None and small.quant.bpw < 8.5


def test_low_ram_machines_get_a_tiny_model():
    # 2 GB netbook with a light Linux: a ~1B model, compressed, short context
    two = planner.best(hardware.simulated(2, os_name="Linux"), 6.0)
    assert two is not None and two.model.params_b <= 1.6 and two.memory_gb <= planner.ram_budget_gb(hardware.simulated(2, os_name="Linux"))
    three = planner.best(hardware.simulated(3, os_name="Linux"), 6.0)
    assert three is not None and three.model.params_b > two.model.params_b


def test_one_gb_is_honestly_too_small():
    assert planner.best(hardware.simulated(1, os_name="Linux"), 6.0) is None
