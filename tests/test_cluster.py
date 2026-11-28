import socket

from tlav import catalog, cluster, hardware, planner, tuner


def _free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker_node(ram=16, latency=0.5):
    return planner.Node(hardware.simulated(ram), 16.0, endpoint="10.0.0.2:50052", latency_ms=latency, name="pc2")


def test_messages_roundtrip():
    q = cluster.parse(cluster.make_query("secret"))
    assert q["type"] == "query" and q["token"] == "secret"
    assert cluster.parse(b"garbage") is None
    assert cluster.parse(b'{"magic": "other"}') is None


def test_discovery_over_loopback_with_token():
    port = _free_udp_port()
    info = {"hostname": "pc2", "rpc_host": "127.0.0.1", "rpc_port": 50099, "hardware": None, "gbps": 9.0}
    r = cluster.DiscoveryResponder(info, token="s3cret", port=port, bind="127.0.0.1")
    r.start()
    try:
        assert cluster.scan(timeout=0.8, port=port, targets=["127.0.0.1"], token="wrong") == []
        found = cluster.scan(timeout=0.8, port=port, targets=["127.0.0.1"], token="s3cret")
        assert len(found) == 1 and found[0]["endpoint"] == "127.0.0.1:50099"
    finally:
        r.stop()


def test_two_computers_fit_what_one_cannot():
    hw = hardware.simulated(16)
    moe = catalog.get("qwen3-30b-a3b")
    assert planner.plan_for_model(moe, [planner.Node(hw, 16.0)]) is None
    plan = planner.plan_for_model(moe, [planner.Node(hw, 16.0), _worker_node()])
    assert plan is not None and plan.is_cluster
    assert sum(plan.split) == moe.layers
    for node, layers in zip(plan.nodes, plan.split):
        need = (plan.memory_gb - planner.RUNTIME_OVERHEAD_GB * len(plan.nodes)) * layers / moe.layers
        assert need <= node.budget


def test_model_that_fits_locally_is_not_split():
    plan = planner.plan_for_model(catalog.get("qwen3-4b"), [planner.Node(hardware.simulated(16), 16.0), _worker_node()])
    assert plan and not plan.is_cluster


def test_cluster_is_never_faster_than_slowest_share():
    m, q = catalog.get("qwen2.5-7b"), catalog.QUANT_BY_NAME["Q4_K_M"]
    nodes = [planner.Node(hardware.simulated(16), 16.0), _worker_node(latency=5.0)]
    local = planner.evaluate_nodes(m, q, nodes[:1])
    split = planner.evaluate_nodes(m, q, nodes)
    assert split.est_tps < local.est_tps


def test_split_layers_sums_and_favours_capacity():
    s = planner._split_layers(48, [10.0, 5.0, 5.0])
    assert sum(s) == 48 and s[0] == 24


def test_tuner_cluster_args():
    hw = hardware.simulated(16)
    plan = planner.plan_for_model(catalog.get("qwen3-30b-a3b"), [planner.Node(hw, 16.0), _worker_node()])
    s = tuner.settings(plan, hw)
    assert s["rpc"] == ["10.0.0.2:50052"] and s["gpu_layers"] == plan.split[1]
    args = tuner.server_args("/m.gguf", s, help_text="--rpc -ngl -ts --load-mode -fa [on|off|auto]")
    assert "--rpc" in args and args[args.index("-ngl") + 1] == str(plan.split[1])


def test_network_advice():
    assert "wired" in cluster.network_advice(0.4)
    assert "Ethernet" in cluster.network_advice(20)


def test_link_animation_draws_and_settles(capsys):
    import re
    import time
    from tlav import anim
    links = anim.Links([("this PC", "study-imac"), ("this PC", "attic-pc")], "connecting", enabled=True).start()
    time.sleep(0.25)
    links.done(0, "connected · 0.4 ms")
    links.done(1, "no answer", ok=False)
    links.finish()
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert "connecting…" in out and "■" in out            # a packet was drawn while connecting
    assert "━" in out and "╌" in out                       # settled: solid when ok, dashed when not
    assert "connected · 0.4 ms" in out and "no answer" in out


def test_link_plain_output_without_terminal(capsys):
    from tlav import anim
    links = anim.Links([("this PC", "pc2")], "connecting", enabled=False).start()
    links.finish("connected")
    assert capsys.readouterr().out == "  this PC -> pc2: connected\n"


def test_engine_mismatch(monkeypatch):
    from tlav import runtime
    monkeypatch.setattr(runtime, "engine_version", lambda: "aaa111")
    assert cluster.engine_mismatch({"endpoint": "x:1", "engine": "aaa111"}) is None
    assert cluster.engine_mismatch({"endpoint": "x:1"}) is None             # older worker: unknown
    assert "tlav update" in cluster.engine_mismatch({"endpoint": "x:1", "hostname": "pc2", "engine": "bbb222"})


def test_relocate_follows_worker_by_name(monkeypatch):
    saved = [{"hostname": "attic-pc", "endpoint": "192.168.1.20:50052"}]
    store = {"w": saved}
    monkeypatch.setattr(cluster, "saved_workers", lambda: store["w"])
    monkeypatch.setattr(cluster, "save_workers", lambda w: store.update(w=w))
    monkeypatch.setattr(cluster, "scan", lambda **kw: [{"hostname": "attic-pc", "endpoint": "192.168.1.31:50052"}])
    monkeypatch.setattr(cluster, "reachable", lambda w: 0.4)
    moved = cluster.relocate(saved)
    assert moved == {"192.168.1.20:50052": "192.168.1.31:50052"}
    assert store["w"][0]["endpoint"] == "192.168.1.31:50052"
