import json

from tlav import benchmark, compress, hub, tuner, catalog, hardware, planner


def test_find_quant_file_picks_exact_match():
    files = ["gemma-2-9b-it-Q4_K_M-fp16.gguf", "gemma-2-9b-it-Q4_K_M.gguf", "gemma-2-9b-it-Q4_K_S.gguf",
             "README.md", "gemma-2-9b-it-f16.gguf", "gemma-2-9b-it-Q8_0.gguf"]
    assert hub.find_quant_file(files, "Q4_K_M") == "gemma-2-9b-it-Q4_K_M.gguf"
    assert hub.find_quant_file(files, "Q6_K") is None
    assert hub.find_source_file(files) == ("gemma-2-9b-it-f16.gguf", "F16")
    assert hub.find_source_file(["x-Q8_0.gguf"]) == ("x-Q8_0.gguf", "Q8_0")
    mixed = ["gemma-2-9b-it-Q8_0-f16.gguf", "gemma-2-9b-it-Q8_0.gguf"]
    assert hub.find_source_file(mixed) == ("gemma-2-9b-it-Q8_0.gguf", "Q8_0")


def test_parse_perplexity():
    out = "[1]9.8123,[2]10.0121,[3]9.9\nFinal estimate: PPL = 9.9312 +/- 0.4"
    assert compress.parse_perplexity(out) == 9.9312
    assert compress.parse_perplexity("[1]5.5,[2]6.25") == 6.25
    assert compress.parse_perplexity("nothing") is None
    assert round(compress.quality_delta(10.0, 10.25), 2) == 2.5


def test_parse_llama_bench_json():
    data = [{"n_prompt": 64, "n_gen": 0, "avg_ts": 41.2}, {"n_prompt": 0, "n_gen": 32, "avg_ts": 4.37}]
    assert benchmark.parse_llama_bench_json(json.dumps(data)) == {"prompt_tps": 41.2, "gen_tps": 4.37}


def test_bandwidth_is_positive():
    assert benchmark.memory_bandwidth_gbps(size_mb=16, rounds=2) > 0.1


def test_tuner_uses_physical_cores_and_builds_args():
    hw = hardware.simulated(16, cores=4)
    plan = planner.evaluate(catalog.get("qwen3-8b"), catalog.QUANT_BY_NAME["Q4_K_M"], hw, 16.0)
    s = tuner.settings(plan, hw)
    assert s["threads"] == 4 and s["gpu_layers"] == 0
    args = tuner.server_args("/m.gguf", s, port=9000)
    assert args[:2] == ["-m", "/m.gguf"] and "9000" in args
