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
import signal
import socket
import subprocess
import sys
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
                "Different builds can't share a model: run `tlav update` on both computers so they match.")
    return None


# ------------------------------------------------------------------ plain-words help

def firewall_help(os_name: Optional[str], rpc_port: int = RPC_PORT, discovery_port: int = DISCOVERY_PORT) -> str:
    """How to let the main computer in, for the helper's operating system."""
    o = (os_name or "").lower()
    if o.startswith("win"):
        return ("On that Windows PC, click \"Allow access\" when Windows asks about the firewall, or run this once\n"
                "    in PowerShell opened as administrator:\n"
                f"      New-NetFirewallRule -DisplayName 'Tlav worker' -Direction Inbound -Protocol TCP -LocalPort {rpc_port} "
                "-Action Allow\n"
                f"      New-NetFirewallRule -DisplayName 'Tlav discovery' -Direction Inbound -Protocol UDP -LocalPort "
                f"{discovery_port} -Action Allow")
    if o.startswith(("darwin", "mac")):
        return ("On that Mac, click \"Allow\" when macOS asks about incoming connections, or open System Settings >\n"
                "    Network > Firewall > Options and allow rpc-server.")
    return ("On that Linux PC, if a firewall is on, run:\n"
            f"      sudo ufw allow {rpc_port}/tcp && sudo ufw allow {discovery_port}/udp")


NOT_FOUND_HELP = """No helper computers answered. Check that:
  1. Each helper has Tlav installed and is showing the `tlav worker` screen.
  2. They're on the same network as this computer (same Wi-Fi or router).
  3. Their firewall lets Tlav in (Windows asks the first time: click "Allow access").
Still nothing? Add a helper by the address shown on its screen:  tlav cluster add 192.168.1.23"""


class Activity:
    """Reads rpc-server's output to tell whether a main computer is using this PC right now.

    rpc-server prints one line per connection. Quick checks from Tlav (is it reachable?) open
    and close at once; a main computer running a model keeps its connection open.
    """

    BUSY_AFTER = 2.0   # seconds a connection must stay open to count as "in use"

    def __init__(self):
        self.starts: List[float] = []
        self.helped = 0          # how many times a main computer used this PC
        self.lock = threading.Lock()

    def feed(self, line: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self.lock:
            if "Accepted client connection" in line:
                self.starts.append(now)
            elif "Client connection closed" in line and self.starts:
                began = self.starts.pop()          # quick checks come and go on top of a long session
                if now - began >= self.BUSY_AFTER and not self.starts:
                    self.helped += 1

    def busy_for(self, now: Optional[float] = None) -> float:
        """Seconds the main computer has been using this PC (0 if it isn't)."""
        now = time.time() if now is None else now
        with self.lock:
            if not self.starts:
                return 0.0
            age = now - self.starts[0]
            return age if age >= self.BUSY_AFTER else 0.0


def _ago(secs: float) -> str:
    secs = int(secs)
    return f"{secs}s" if secs < 60 else f"{secs // 60}m" if secs < 3600 else f"{secs // 3600}h {secs % 3600 // 60}m"


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
    from . import ui
    exe = runtime.require("rpc-server")
    hw = hardware.detect()
    gbps = benchmark.cached_or_run()["effective_gbps"]
    chosen = host
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

    addr = host + ("" if port == RPC_PORT else f":{port}")
    ui.banner("This computer is ready to help.", [
        f"{info['hostname']} · {hw.ram_total_gb:.0f} GB RAM · {hw.physical_cores} cores · address {addr}",
        "",
        "On your main computer, type:   " + ui.c(ui.BOLD, "tlav cluster"),
        ui.c(ui.DIM, "Not found by itself? Use:  tlav cluster add " + addr),
    ] if ui.ON else [
        f"{info['hostname']} · {hw.ram_total_gb:.0f} GB RAM · {hw.physical_cores} cores · address {addr}",
        "On your main computer, type:   tlav cluster",
        "If it isn't found:   tlav cluster add " + addr])
    from . import update
    latest = update.check()
    if latest:
        print(ui.c(ui.ORANGE, f"  ! Tlav {latest} is out. Helpers should match the main computer:")
              + "\n    press Ctrl+C, run `tlav update`, then `tlav worker` again.\n")
    if not chosen and host.startswith("127."):
        print(ui.c(ui.ORANGE, "  ! This computer isn't connected to a network, so no other computer can reach it.")
              + "\n    Connect it to Wi-Fi or Ethernet, then run `tlav worker` again.\n")
    print(ui.c(ui.DIM, "  Home networks you trust only. Ctrl+C stops sharing."))
    print(ui.c(ui.DIM, "  Share at every start:  tlav service install --worker"))
    print()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            bufsize=1)
    act, tail = Activity(), []

    def _read():
        for line in proc.stdout:
            act.feed(line)
            tail.append(line.rstrip())
            del tail[:-20]
    reader = threading.Thread(target=_read, daemon=True)
    reader.start()

    # Stop the engine however the worker is stopped (Ctrl+C, closing the window, `kill`), so it
    # never keeps sharing this computer in the background on its own.
    def _quit(signum, frame):
        raise KeyboardInterrupt
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _quit)
    last, frame = None, 0
    try:
        while proc.poll() is None:
            busy = act.busy_for()
            if busy:
                text = f"helping your main computer · holding its share of the model · {_ago(busy)}"
            else:
                text = "waiting for your main computer" + (f" · helped {act.helped}x so far" if act.helped else "")
            if ui.ON:
                sq = ui.c(ui.ORANGE, ui.FRAMES[frame % 4]) if not busy else ui.c(ui.GREEN, "■")
                sys.stdout.write(f"\r\033[K  {sq} {text}{'' if busy else '…'}")
                sys.stdout.flush()
                frame += 1
            elif text != last:
                print("  " + text)
            last = text
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if ui.ON:
            sys.stdout.write("\r\033[K")
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        responder.stop()
    code = proc.returncode
    reader.join(timeout=1)
    text = "\n".join(tail)
    if code not in (0, None, -15, -2, 1) or ("bind" in text.lower() and "fail" in text.lower()):
        if "address already in use" in text.lower() or "bind" in text.lower():
            raise RuntimeError(f"Port {port} is already in use: is another `tlav worker` running on this computer? "
                               f"Stop it, or start this one with `tlav worker --port {port + 10}`.")
        raise RuntimeError("The AI engine stopped unexpectedly:\n  " + "\n  ".join(tail[-6:]))
    print("  Stopped sharing this computer.")


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
        if replies:   # it answered the search, so it's the firewall blocking the engine port
            raise ConnectionError(f"Found {w.get('hostname', host)}, but its firewall blocks Tlav.\n  "
                                  + firewall_help((w.get("hardware") or {}).get("os_name"),
                                                  int(w["endpoint"].rsplit(":", 1)[1]), port))
        raise ConnectionError(f"Nothing answered at {w['endpoint']}. Check that `tlav worker` is running on that "
                              "computer and that the address matches the one on its screen.\n  If it is, its "
                              "firewall may be blocking Tlav (Windows asks the first time: click \"Allow access\").")
    w["latency_ms"] = round(lat, 2)
    workers = [x for x in saved_workers() if x["endpoint"] != w["endpoint"]] + [w]
    save_workers(workers)
    return w


def relocate(workers: List[dict], timeout: float = 2.5, port: Optional[int] = None) -> Dict[str, str]:
    """Find saved workers whose address changed (routers reassign them) by their computer name.
    Returns {old endpoint: new endpoint} and updates the saved list."""
    found = scan(timeout=timeout, port=port or config.load_state().get("discovery_port") or DISCOVERY_PORT)
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
