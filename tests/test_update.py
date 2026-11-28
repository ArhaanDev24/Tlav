"""Updates: a cached daily check, one question per version, and `tlav update`."""
import argparse
import re
import time
from pathlib import Path

import pytest

from tlav import __version__, cli, config, update


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TLAV_HOME", str(tmp_path))
    monkeypatch.delenv("TLAV_NO_UPDATE_CHECK", raising=False)
    return tmp_path


def test_versions_compare_as_numbers():
    assert update.newer("0.10.0", "0.9.9")
    assert update.newer("1.0", "0.99.3")
    assert not update.newer("0.4.0", "0.4.0")
    assert not update.newer(None, "0.4.0") and not update.newer("", "0.4.0")


def test_check_asks_the_internet_at_most_once_a_day(home, monkeypatch):
    calls = []
    monkeypatch.setattr(update, "latest_version", lambda timeout: calls.append(timeout) or "9.0.0")
    assert update.check() == "9.0.0"
    assert update.check() == "9.0.0"           # cached: no second request
    assert len(calls) == 1 and calls[0] <= 1.5   # never slows the start by more than 1.5 s


def test_offline_check_is_quiet_and_keeps_the_last_answer(home, monkeypatch):
    config.save_state(update_check={"at": 0, "latest": "9.0.0"})
    monkeypatch.setattr(update, "latest_version", lambda timeout: None)   # no internet
    assert update.check() == "9.0.0"
    assert config.load_state()["update_check"]["at"] > time.time() - 5     # don't retry every start


def test_no_check_when_turned_off(home, monkeypatch):
    monkeypatch.setenv("TLAV_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(update, "latest_version", lambda timeout: pytest.fail("went online"))
    assert update.check() is None


def test_each_version_is_asked_about_once(home):
    assert update.should_ask("9.0.0")
    update.declined("9.0.0")
    assert not update.should_ask("9.0.0")
    assert update.should_ask("9.1.0")          # the next one is asked about again


def test_plain_tlav_offers_the_update(home, monkeypatch):
    monkeypatch.setattr(update, "installed_by_installer", lambda: True)
    monkeypatch.setattr(update, "check", lambda: "9.0.0")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    ran = []
    monkeypatch.setattr(update, "run_update", lambda: ran.append(1) or True)
    monkeypatch.setattr(cli.ui, "ask", lambda q, default="y": True)
    assert cli._offer_update() is True and ran
    monkeypatch.setattr(cli.ui, "ask", lambda q, default="y": False)
    assert cli._offer_update() is False
    assert cli._offer_update() is False        # declined: not asked again for 9.0.0


def test_engine_is_replaced_only_when_the_pin_changed(home, monkeypatch, capsys):
    from tlav import runtime
    installed = []
    monkeypatch.setattr(runtime, "install", lambda hw, prefer="auto": installed.append(1) or Path("/x"))
    monkeypatch.setattr(runtime, "find", lambda tool: "/x/llama-server")
    monkeypatch.setattr(cli.hardware, "detect", lambda: cli.hardware.simulated(8))
    args = argparse.Namespace(if_changed=True, source=False, prebuilt=False)
    config.save_state(runtime_source="prebuilt:" + runtime.llama_tag())
    cli.cmd_install_runtime(args)
    assert not installed                       # same build: nothing to do
    config.save_state(runtime_source="prebuilt:b1")
    cli.cmd_install_runtime(args)
    assert installed                           # the new version pins another build


def test_windows_update_finishes_after_tlav_exits():
    script = update.windows_script(r"C:\Users\me\.tlav\venv\Scripts\python.exe", update.SOURCE)
    assert "ping 127.0.0.1" in script                     # waits for tlav.exe to be released
    assert "--force-reinstall" in script and update.SOURCE in script
    assert "install-runtime --if-changed" in script and "pause" in script
    script.encode("ascii")                                # cmd reads it without surprises


def test_both_version_numbers_match():
    toml = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', toml, re.M).group(1) == __version__
