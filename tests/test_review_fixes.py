"""Bugs found in the full code review."""
import json
import socket
import subprocess
import sys
import urllib.error

import pytest

from tlav import hub, server


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TLAV_HOME", str(tmp_path))
    return tmp_path


def test_stop_never_kills_a_program_that_reused_the_pid(home):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (home / "state.json").write_text(json.dumps({"server_pid": other.pid}))
        assert server.stop() is False                   # not llama-server: left alone
        assert other.poll() is None
        assert json.loads((home / "state.json").read_text())["server_pid"] is None
        assert server.status()["running"] is False
    finally:
        other.kill()


def test_port_free_sees_a_busy_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        assert server.port_free("127.0.0.1", s.getsockname()[1]) is False
    finally:
        s.close()


def _gguf(path, size=2 * 1024 * 1024):
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))


def test_download_finishes_when_only_the_rename_was_missed(home, monkeypatch):
    models = home / "models"
    models.mkdir()
    _gguf(models / "M-Q4_K_M.gguf.part")
    def fail_416(*a, **k):
        raise urllib.error.HTTPError("u", 416, "Range Not Satisfiable", {}, None)
    monkeypatch.setattr(hub, "_download_once", fail_416)
    out = hub.download("org/repo", "M-Q4_K_M.gguf", models, quiet=True)
    assert out.exists() and not (models / "M-Q4_K_M.gguf.part").exists()


def test_download_retries_a_busy_server(home, monkeypatch):
    calls = []
    def flaky(repo, filename, dest, quiet):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
        _gguf(dest)
        return dest
    monkeypatch.setattr(hub, "_download_once", flaky)
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    out = hub.download("org/repo", "M.gguf", home, quiet=True)
    assert out.exists() and len(calls) == 3


def test_download_gives_up_at_once_on_a_missing_file(home, monkeypatch):
    def missing(*a, **k):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    monkeypatch.setattr(hub, "_download_once", missing)
    with pytest.raises(urllib.error.HTTPError):
        hub.download("org/repo", "M.gguf", home, quiet=True)


def test_every_install_uses_the_same_pinned_engine(monkeypatch):
    from tlav import runtime
    monkeypatch.delenv("TLAV_LLAMA_TAG", raising=False)
    names = [n for n, _ in runtime._release_assets(runtime.PINNED_TAG, ["bin-ubuntu-x64", "bin-win-cpu-x64"])]
    assert f"llama-{runtime.PINNED_TAG}-bin-ubuntu-x64.tar.gz" in names
    assert f"llama-{runtime.PINNED_TAG}-bin-win-cpu-x64.zip" in names
    url = runtime._release_assets("b1", ["bin-macos-arm64"])[0][1]
    assert url == "https://github.com/ggml-org/llama.cpp/releases/download/b1/llama-b1-bin-macos-arm64.tar.gz"


def test_source_checkout_of_another_version_is_replaced(tmp_path, monkeypatch):
    from tlav import runtime
    monkeypatch.setenv("TLAV_LLAMA_TAG", "b100")
    old = tmp_path / "llama.cpp"
    old.mkdir()
    (old / ".tlav-tag").write_text("b99")
    cloned = []
    def fake_clone(cmd):
        cloned.append(cmd)
        (tmp_path / "llama.cpp").mkdir()
    monkeypatch.setattr(runtime.shutil, "which", lambda n: "/usr/bin/git")
    monkeypatch.setattr(runtime.subprocess, "check_call", fake_clone)
    src = runtime._fetch_source(tmp_path)
    assert cloned and "--branch" in cloned[0] and "b100" in cloned[0]
    assert (src / ".tlav-tag").read_text() == "b100"
    cloned.clear()
    runtime._fetch_source(tmp_path)          # same version: reused, not downloaded again
    assert not cloned


def test_state_survives_a_crash_mid_write(tmp_path, monkeypatch):
    from tlav import config
    monkeypatch.setenv("TLAV_HOME", str(tmp_path))
    config.save_state(active={"path": "/m.gguf"}, cluster=[{"endpoint": "10.0.0.5:50052"}])
    real = config.os.replace
    def crash(*a):
        raise KeyboardInterrupt          # power cut / Ctrl+C at the worst moment
    monkeypatch.setattr(config.os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        config.save_state(server_pid=123)
    monkeypatch.setattr(config.os, "replace", real)
    state = config.load_state()
    assert state["active"]["path"] == "/m.gguf" and state["cluster"]   # nothing lost


def test_worker_from_a_newer_tlav_doesnt_crash_the_main_computer():
    from tlav import hardware
    hw = hardware.from_dict({"ram_total_gb": 16, "physical_cores": 4, "os_name": "Windows",
                             "a_field_from_the_future": True})
    assert hw.ram_total_gb == 16 and hw.os_name == "Windows"


def test_bandwidth_test_fits_a_low_memory_pc(monkeypatch):
    from tlav import benchmark, hardware
    monkeypatch.setattr(hardware, "_ram", lambda: (2.0, 0.4))     # 2 GB PC, 400 MB free
    assert benchmark._test_size_mb() <= 64


def test_login_services_survive_unusual_paths():
    import xml.dom.minidom
    from pathlib import Path
    from tlav import service
    unit = service.systemd_unit("worker", ["/home/me/My Files/python", "-m", "tlav", "worker"])
    assert 'ExecStart="/home/me/My Files/python" -m tlav worker' in unit
    plist = service.launchd_plist("worker", ["/Users/R&D <team>/python", "-m", "tlav"], Path("/tmp"))
    xml.dom.minidom.parseString(plist)          # valid XML: macOS can load it


def test_interrupted_compression_never_leaves_a_file_that_looks_finished(tmp_path, monkeypatch):
    from tlav import compress, runtime
    monkeypatch.setenv("TLAV_HOME", str(tmp_path))
    src = tmp_path / "M-f16.gguf"
    src.write_bytes(b"GGUF" + b"\0" * 100)
    monkeypatch.setattr(runtime, "require", lambda tool: "/bin/quantize")
    def killed(cmd, **kw):
        open(cmd[-3], "wb").write(b"GGUF half-written")   # the tool got halfway...
        raise KeyboardInterrupt                             # ...then Ctrl+C / crash
    monkeypatch.setattr(compress.subprocess, "run", killed)
    with pytest.raises(KeyboardInterrupt):
        compress.quantize(src, "Q4_K_M", 2)
    assert sorted(p.name for p in tmp_path.glob("M-*")) == ["M-f16.gguf"]     # nothing left behind
    def fails(cmd, **kw):
        open(cmd[-3], "wb").write(b"GGUF broken")
        return compress.subprocess.CompletedProcess(cmd, 1)
    monkeypatch.setattr(compress.subprocess, "run", fails)
    with pytest.raises(RuntimeError):
        compress.quantize(src, "Q4_K_M", 2)
    assert not (tmp_path / "M-Q4_K_M.gguf").exists()                         # so it isn't reused next time
