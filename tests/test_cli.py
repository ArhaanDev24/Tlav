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


def test_ask_where(monkeypatch):
    import io
    from tlav import cli

    class TTY(io.StringIO):
        def isatty(self):
            return True

    for answer, where in (("", "terminal"), ("1", "terminal"), ("2", "web"), ("web", "web")):
        monkeypatch.setattr("sys.stdin", TTY(answer + "\n"))
        assert cli._ask_where() == where
    monkeypatch.setattr("sys.stdin", io.StringIO("2\n"))   # not a terminal: never asks
    assert cli._ask_where() == "terminal"


def test_plain_tlav_opens_or_sets_up(monkeypatch):
    from tlav import cli, config
    seen = []
    real = cli.build_parser

    def parser():
        p = real()
        for name, sub in p._subparsers._group_actions[0].choices.items():
            sub.set_defaults(func=lambda a, n=name: seen.append(n))
        return p

    monkeypatch.setattr(cli, "build_parser", parser)
    monkeypatch.setattr(config, "load_state", lambda: {"active": {"path": "x"}})
    cli.main([])
    monkeypatch.setattr(config, "load_state", lambda: {})
    cli.main([])
    assert seen == ["open", "setup"]
