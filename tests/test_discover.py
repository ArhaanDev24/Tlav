from tlav import catalog, compress, discover, hardware, planner


def _repo(id_, base, pipeline="text-generation"):
    return {"id": id_, "pipeline_tag": pipeline, "createdAt": "2026-08-01T00:00:00Z", "downloads": 10,
            "tags": ["gguf", f"base_model:{base}", f"base_model:quantized:{base}"]}


def test_only_official_general_models_pass():
    assert discover.acceptable(_repo("bartowski/Qwen_Qwen3.8-27B-GGUF", "Qwen/Qwen3.8-27B")) == "Qwen/Qwen3.8-27B"
    assert discover.acceptable(_repo("x/Qwen3-8B-abliterated-GGUF", "Qwen/Qwen3-8B")) is None
    assert discover.acceptable(_repo("x/Cydonia-GGUF", "TheDrummer/Cydonia-24B")) is None   # not a lab
    assert discover.acceptable(_repo("x/Qwen3-VL-GGUF", "Qwen/Qwen3-VL-8B-Instruct")) is None
    assert discover.acceptable(_repo("x/whisper-GGUF", "openai/whisper", pipeline="automatic-speech-recognition")) is None


def test_publisher_priority_and_base_dedup():
    listing = {
        "bartowski": [_repo("bartowski/google_gemma-4-31B-it-GGUF", "google/gemma-4-31B-it")],
        "unsloth": [_repo("unsloth/gemma-4-31B-it-GGUF", "google/gemma-4-31B-it"),
                    _repo("unsloth/Qwen2.5-Coder-32B-GGUF", "Qwen/Qwen2.5-Coder-32B"),
                    _repo("unsloth/Qwen2.5-Coder-32B-Instruct-GGUF", "Qwen/Qwen2.5-Coder-32B-Instruct")],
    }

    def lister(url):
        return next((v for k, v in listing.items() if f"author={k}" in url), [])

    found = discover.list_candidates(lister=lister)
    assert found["google/gemma-4-31b-it"]["repo"].startswith("bartowski/")
    assert "qwen/qwen2.5-coder-32b" not in found               # base model dropped, chat version kept
    assert "qwen/qwen2.5-coder-32b-instruct" in found


def test_quality_estimate_rewards_size_and_recency():
    a, b, c = catalog.quality_fit()
    assert b > 3 and c > 1                                       # bigger and newer both help
    old = catalog.estimate_quality(8, released="2024-06")
    new = catalog.estimate_quality(8, released="2026-06")
    big = catalog.estimate_quality(32, released="2024-06")
    assert new > old and big > old


def test_more_quant_levels_have_retention():
    assert catalog.retention_for("IQ4_XS") < catalog.retention_for("Q4_K_M") < catalog.retention_for("Q4_K_L")
    assert catalog.retention_for("WEIRD") is None
    assert 0.9 < catalog.retention_for("WEIRD", bpw=4.5) < 1


def test_floor_keeps_default_quality():
    names = ["Q4_K_S", "IQ4_XS", "Q4_K_M", "Q4_K_L", "Q5_K_M", "Q3_K_M"]
    qs = [catalog.quant_from_file(int(8.95e9 * 5 / 8), 8.95, n) for n in names]
    kept = [q.name for q in planner._quants_for("Q4_K_M", qs)]
    assert kept == ["Q5_K_M", "Q4_K_L", "Q4_K_M"]


def test_fit_target_respects_ram_and_speed():
    m = catalog.get("qwen3.5-9b")
    node = planner.Node(hardware.simulated(16), 16.0)
    size, ctx = planner.fit_target(m, node)
    assert size <= node.budget and ctx == 8192
    slow_size, _ = planner.fit_target(m, node, min_tps=2.5)
    assert slow_size < size
    assert planner.fit_target(m, planner.Node(hardware.simulated(4), 16.0)) is None


def test_parse_kld():
    out = """Mean PPL(Q)/PPL(base)         :   1.012071 ±   0.012450
Mean    KLD:   0.028425 ±   0.000984
Same top p: 96.031 ± 0.617 %"""
    r = compress.parse_kld(out)
    assert r == {"kld": 0.028425, "same_top_pct": 96.031, "ppl_ratio": 1.012071}
    assert "96.0%" in compress.describe(r)
    assert compress.parse_kld("nothing") is None


def test_lookup_tables_do_not_count_as_streamed():
    m = catalog.get("qwen3.5-9b")
    q = catalog.QUANT_BY_NAME["Q4_K_M"]
    assert m.active_gb(q) < m.weights_gb(q)
