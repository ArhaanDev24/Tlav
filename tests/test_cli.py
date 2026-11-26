import json

import pytest

from tlav import cli


def test_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "setup" in capsys.readouterr().out


def test_models(capsys):
    cli.main(["models"])
    out = capsys.readouterr().out
    assert "qwen3-8b" in out and "Q4_K_M" in out


def test_recommend_simulated_json(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("TLAV_HOME", str(tmp_path))
    cli.main(["recommend", "--ram", "16", "--json"])
    out = capsys.readouterr().out
    plans = json.loads(out[out.index("["):])
    assert plans and plans[0]["memory_gb"] < 16


def test_runtime_cmake_flags_intel_mac():
    from tlav import hardware, runtime
    hw = hardware.simulated(16, simd="avx2", os_name="macOS")
    flags = runtime.cmake_flags(hw)
    assert "-DGGML_METAL=OFF" in flags and "-DGGML_NATIVE=ON" in flags
