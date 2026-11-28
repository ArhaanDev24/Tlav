"""Connecting computers: the worker's activity line, plain-words help and the `tlav cluster` screen."""
import argparse
import types

from tlav import anim, cli, cluster, ui


def test_quick_checks_dont_count_as_helping():
    a = cluster.Activity()
    a.feed("Accepted client connection", now=0.0)
    a.feed("Client connection closed", now=0.05)          # a reachability check
    assert a.busy_for(now=10.0) == 0 and a.helped == 0


def test_a_main_computer_using_this_pc():
    a = cluster.Activity()
    a.feed("Accepted client connection", now=0.0)         # the model load stays connected
    assert a.busy_for(now=1.0) == 0                         # too early to call it "in use"
    assert a.busy_for(now=30.0) == 30.0
    a.feed("Accepted client connection", now=31.0)        # a quick check on top of the session
    a.feed("Client connection closed", now=31.1)
    assert a.busy_for(now=40.0) == 40.0
    a.feed("Client connection closed", now=60.0)          # the main computer let go
    assert a.busy_for(now=61.0) == 0 and a.helped == 1


def test_firewall_help_names_the_fix_for_each_system():
    assert "New-NetFirewallRule" in cluster.firewall_help("Windows", 50052, 50053)
    assert "Allow" in cluster.firewall_help("macOS")
    linux = cluster.firewall_help("Linux", 50061, 50071)
    assert "ufw allow 50061/tcp" in linux and "50071/udp" in linux


def _plan(name, quality, is_cluster):
    return types.SimpleNamespace(model=types.SimpleNamespace(name=name), quality=quality,
                                 is_cluster=is_cluster, est_tps=5.0)


def test_helpers_offered_only_for_a_clearly_better_model():
    alone = _plan("Qwen3.5 9B", 78.0, False)
    assert cli._clearly_better(_plan("Qwen3.6 35B", 88.0, True), alone)
    assert not cli._clearly_better(_plan("Qwen3 14B", 80.0, True), alone)    # barely better: don't bother
    assert not cli._clearly_better(_plan("Qwen3.5 9B", 78.0, False), alone)  # helpers not even used
    assert cli._clearly_better(_plan("Qwen3 4B", 60.0, True), None)          # nothing fits alone


def test_links_grow_and_never_wrap():
    links = anim.Links([("this PC", "your network")], "listening", enabled=False)
    i = links.add("this PC", "study-imac", "connecting")
    assert i == 1 and links.rw == len("your network")      # the widest name sets the column
    links.width = 60
    links.done(i, "connected · " + "x" * 200)
    line = links._render(links.lines[i], 0.0)
    assert len(line) <= 60 and line.endswith("…")


def test_banner_box_stays_square_with_coloured_lines(monkeypatch, capsys):
    monkeypatch.setattr(ui, "ON", True)
    monkeypatch.setattr(ui, "width", lambda: 80)
    ui.banner("sub", ["plain line", "type " + ui.c(ui.BOLD, "tlav cluster")])
    import re
    rows = [re.sub(r"\033\[[0-9;]*m", "", r) for r in capsys.readouterr().out.splitlines() if "│" in r]
    assert len({len(r) for r in rows}) == 1


def test_connect_finds_saves_and_offers(monkeypatch):
    found = [{"hostname": "study-imac", "endpoint": "10.0.0.5:50052", "candidates": ["10.0.0.5:50052"],
              "hardware": {"ram_total_gb": 16, "physical_cores": 4, "os_name": "macOS"}, "gbps": 12},
             {"hostname": "garage-pc", "endpoint": "10.0.0.9:50052", "candidates": ["10.0.0.9:50052"],
              "hardware": {"ram_total_gb": 8, "physical_cores": 2, "os_name": "Windows"}, "gbps": 9}]
    saved, offered = [], []
    monkeypatch.setattr(cluster, "saved_workers", lambda: [])
    monkeypatch.setattr(cluster, "save_workers", lambda ws: saved.extend(ws))
    monkeypatch.setattr(cluster, "scan", lambda **k: [dict(w) for w in found])
    monkeypatch.setattr(cluster, "reachable", lambda w: 0.4 if w["hostname"] == "study-imac" else None)
    monkeypatch.setattr(cluster, "engine_mismatch", lambda w: None)
    monkeypatch.setattr(cli, "_offer_cluster_setup", lambda ws: offered.extend(ws))
    monkeypatch.setattr(ui, "ON", False)
    args = argparse.Namespace(token=None, discovery_port=50053, timeout=0.1, wait=0.1)
    cli.cmd_cluster_connect(args)
    assert [w["hostname"] for w in saved] == ["study-imac"]           # the blocked one isn't saved
    assert [w["hostname"] for w in offered] == ["study-imac"]
