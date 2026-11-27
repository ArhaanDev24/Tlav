"""Combine several computers into one AI machine.

How it works
------------
* Each helper computer runs `tlav worker`. That starts llama.cpp's
  `rpc-server`, which offers this machine's RAM and CPU to a main computer,
  plus a tiny UDP responder so the main computer can find it automatically.
* The main computer runs `tlav cluster scan` (or `tlav cluster add IP`) and
  then `tlav setup --cluster` / `tlav run --cluster`. llama-server loads the
  model and sends each worker its share of the layers over the network.

Security: rpc-server has no authentication. Anyone who can reach its port can
run computations on it and read the tensors it holds. Only use it on a home
network you trust, never on public Wi-Fi, and never forward the port on your
router. The optional discovery token only prevents accidental joins.
"""

import hmac
import json
import socket
import subprocess
import threading
import time
from typing import Dict, List, Optional

from . import __version__, benchmark, config, hardware, runtime
from .planner import Node

RPC_PORT = 50052
DISCOVERY_PORT = 50053
MAGIC = "tlav-cluster-v1"


def lan_ip() -> str:
    """IP of the interface used to reach the local network (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _token_ok(expected: Optional[str], given: Optional[str]) -> bool:
    if not expected:
        return True
    return bool(given) and hmac.compare_digest(expected.encode(), str(given).encode())


# ------------------------------------------------------------------ messages

def make_query(token: Optional[str] = None) -> bytes:
    return json.dumps({"magic": MAGIC, "type": "query", "token": token}).encode()


def make_reply(info: dict) -> bytes:
    return json.dumps(dict(info, magic=MAGIC, type="worker")).encode()


def parse(data: bytes) -> Optional[dict]:
    try:
        msg = json.loads(data.decode())
    except (ValueError, UnicodeDecodeError):
        return None
    return msg if isinstance(msg, dict) and msg.get("magic") == MAGIC else None


def worker_info(rpc_host: str, rpc_port: int, hw: hardware.Hardware, gbps: float) -> dict:
    return {"hostname": socket.gethostname(), "rpc_host": rpc_host, "rpc_port": rpc_port,
            "version": __version__, "engine": runtime.engine_version(), "hardware": hw.to_dict(),
            "gbps": round(gbps, 2)}


def engine_mismatch(worker: dict) -> Optional[str]:
    """A warning if the worker runs a different llama.cpp build (they can't work together)."""
    theirs, ours = worker.get("engine"), runtime.engine_version()
    if theirs and ours and theirs != ours:
        return (f"{worker.get('hostname', worker['endpoint'])} runs AI engine build {theirs}, this PC runs {ours}. "
                "Different builds can't share a model: run `tlav install-runtime` on both so they match.")
    return None


# ------------------------------------------------------------------ worker side

class DiscoveryResponder(threading.Thread):
    def __init__(self, info: dict, token: Optional[str] = None, port: int = DISCOVERY_PORT,
                 bind: str = "0.0.0.0"):
        super().__init__(daemon=True)
        self.info, self.token = info, token
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.sock.settimeout(0.5)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            msg = parse(data)
            if msg and msg.get("type") == "query" and _token_ok(self.token, msg.get("token")):
                self.sock.sendto(make_reply(self.info), addr)

    def stop(self):
        self._stop.set()
        self.sock.close()


def run_worker(host: Optional[str] = None, port: int = RPC_PORT, token: Optional[str] = None,
               discovery_port: int = DISCOVERY_PORT, cache: bool = True, threads: Optional[int] = None,
               mem_mb: Optional[int] = None) -> None:
    exe = runtime.require("rpc-server")
    hw = hardware.detect()
    gbps = benchmark.cached_or_run()["effective_gbps"]
    host = host or lan_ip()
    cmd = [exe, "-H", host, "-p", str(port), "-t", str(threads or hw.physical_cores)]
    if cache:
        cmd.append("-c")  # keep received weights on disk so the next load is fast
    if mem_mb:
        if "--mem" in runtime.help_text("rpc-server") or " -m," in runtime.help_text("rpc-server"):
            cmd += ["-m", str(mem_mb)]
        else:
            print("  (this AI engine build can't limit shared memory; --mem is ignored)")
    info = worker_info(host, port, hw, gbps)
    responder = DiscoveryResponder(info, token, discovery_port)
    responder.start()
    print(f"Tlav worker '{info['hostname']}' sharing {hw.ram_total_gb:.0f} GB RAM, {hw.physical_cores} cores")
    print(f"  RPC endpoint: {host}:{port}    discovery: UDP {discovery_port}")
    print("  On the main computer run:  tlav cluster scan   (or: tlav cluster add "
          f"{host}{'' if port == RPC_PORT else ':' + str(port)})")
    print("  Only use this on a trusted home network. Ctrl+C to stop.\n")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        responder.stop()


# ------------------------------------------------------------------ main side

def scan(timeout: float = 2.0, token: Optional[str] = None, port: int = DISCOVERY_PORT,
         targets: Optional[List[str]] = None) -> List[dict]:
    """Broadcast (or unicast to `targets`) and collect worker replies."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.3)
    query = make_query(token)
    dests = targets or ["255.255.255.255", "<broadcast>"]
    for d in dests:
        try:
            sock.sendto(query, (d, port))
        except OSError:
            pass
    found: Dict[str, dict] = {}
    end = time.time() + timeout
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(65536)
        except socket.timeout:
            continue
        msg = parse(data)
        if msg and msg.get("type") == "worker":
            port_ = msg.get("rpc_port", RPC_PORT)
            # The address the reply came from and the one the worker advertises can differ
            # (several network cards, VPNs, a worker bound to localhost). Keep both; the
            # caller uses whichever one actually answers.
            hosts = [h for h in dict.fromkeys([msg.get("rpc_host"), addr[0]]) if h and h != "0.0.0.0"]
            msg["candidates"] = [f"{h}:{port_}" for h in hosts]
            msg["endpoint"] = msg["candidates"][0] if msg["candidates"] else f"{addr[0]}:{port_}"
            if not any(e in found for e in msg["candidates"]):
                found[msg["endpoint"]] = msg
    sock.close()
    return list(found.values())


def reachable(worker: dict) -> Optional[float]:
    """Ping the worker's candidate addresses; keep the first that answers as its endpoint."""
    for e in worker.get("candidates") or [worker["endpoint"]]:
        lat = ping(e)
        if lat is not None:
            worker["endpoint"] = e
            return lat
    return None


def ping(endpoint: str, tries: int = 3, timeout: float = 2.0) -> Optional[float]:
    """TCP connect round-trip to a worker's rpc-server, in ms (None if unreachable)."""
    host, _, port = endpoint.rpartition(":")
    best = None
    for _ in range(tries):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                pass
        except OSError:
            continue
        ms = (time.perf_counter() - t0) * 1000
        best = ms if best is None else min(best, ms)
    return best


def saved_workers() -> List[dict]:
    return config.load_state().get("cluster", [])


def save_workers(workers: List[dict]) -> None:
    config.save_state(cluster=workers)


def add_worker(address: str, token: Optional[str] = None, port: int = DISCOVERY_PORT) -> dict:
    """Add a worker by IP (asks it for its info) or by explicit ip:rpcport."""
    host, _, rpc_port = address.partition(":")
    replies = scan(timeout=2.0, token=token, port=port, targets=[host])
    if replies:
        w = replies[0]
        w["candidates"] = [f"{host}:{rpc_port or w.get('rpc_port', RPC_PORT)}"] + w.get("candidates", [])
    else:
        # discovery blocked by a firewall: accept a bare rpc endpoint with unknown hardware
        w = {"hostname": host, "endpoint": f"{host}:{rpc_port or RPC_PORT}", "hardware": None, "gbps": None}
    lat = reachable(w)
    w.pop("candidates", None)
    if lat is None:
        raise ConnectionError(f"Cannot reach rpc-server at {w['endpoint']}. Is `tlav worker` running there?")
    w["latency_ms"] = round(lat, 2)
    workers = [x for x in saved_workers() if x["endpoint"] != w["endpoint"]] + [w]
    save_workers(workers)
    return w


def relocate(workers: List[dict], timeout: float = 2.5, port: int = DISCOVERY_PORT) -> Dict[str, str]:
    """Find saved workers whose address changed (routers reassign them) by their computer name.
    Returns {old endpoint: new endpoint} and updates the saved list."""
    found = scan(timeout=timeout, port=port)
    by_name = {}
    for w in found:
        if reachable(w) is not None:
            by_name.setdefault(w.get("hostname"), w)
    moved = {}
    saved = saved_workers()
    for w in workers:
        new = by_name.get(w.get("hostname"))
        if new and new["endpoint"] != w["endpoint"]:
            moved[w["endpoint"]] = new["endpoint"]
    if moved:
        for w in saved:
            if w["endpoint"] in moved:
                w["endpoint"] = moved[w["endpoint"]]
        save_workers(saved)
    return moved


def display_name(worker: dict, workers: List[dict]) -> str:
    """The computer's name; with its port added when two share a name (two workers on one PC)."""
    name = worker.get("hostname") or worker["endpoint"]
    if sum(1 for w in workers if (w.get("hostname") or w["endpoint"]) == name) > 1:
        return f"{name} :{worker['endpoint'].rsplit(':', 1)[-1]}"
    return name


def remove_worker(endpoint: str) -> bool:
    before = saved_workers()
    after = [w for w in before if w["endpoint"] != endpoint and w.get("hostname") != endpoint]
    save_workers(after)
    return len(after) != len(before)


def to_nodes(workers: List[dict], refresh_latency: bool = True) -> List[Node]:
    nodes = []
    for w in workers:
        hw = hardware.from_dict(w.get("hardware")) or hardware.simulated(8, os_name="Linux")
        lat = ping(w["endpoint"], tries=2) if refresh_latency else w.get("latency_ms")
        if refresh_latency and lat is None:
            print(f"  warning: worker {w['endpoint']} is not reachable; leaving it out")
            continue
        nodes.append(Node(hw, w.get("gbps") or 10.0, endpoint=w["endpoint"],
                          latency_ms=lat or w.get("latency_ms") or 1.0, name=w.get("hostname", w["endpoint"])))
    return nodes


def network_advice(latency_ms: float) -> str:
    if latency_ms < 1.0:
        return "excellent (wired)"
    if latency_ms < 3.0:
        return "good"
    if latency_ms < 8.0:
        return "ok (Wi-Fi?) - Ethernet will be noticeably faster"
    return "slow - use an Ethernet cable between the computers"
