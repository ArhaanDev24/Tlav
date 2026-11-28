"""Run llama-server (OpenAI-compatible API + built-in web chat) and a terminal chat client."""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from typing import List, Optional

from . import config, runtime


def port_free(host: str, port: int) -> bool:
    """Can a server listen here? (Binding is the honest test: another program may be listening.)"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            # On Windows SO_REUSEADDR lets two programs share a port; ask for it exclusively instead.
            s.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", -5), 1)
        else:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0" if host == "0.0.0.0" else host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def pick_port(host: str, port: int, tries: int = 20) -> int:
    for p in range(port, port + tries):
        if port_free(host, p):
            return p
    raise RuntimeError(f"Ports {port}-{port + tries - 1} are all in use; pass --port with a free one.")


def start(args: List[str], wait: bool = True, timeout: int = 900, detach: bool = False) -> subprocess.Popen:
    exe = runtime.require("llama-server")
    args = list(args)
    if "--port" in args:
        i = args.index("--port") + 1
        host = args[args.index("--host") + 1] if "--host" in args else "127.0.0.1"
        want = int(args[i])
        got = pick_port(host, want)
        if got != want:
            # Something else (another AI app, an old Tlav) already uses the port. Never talk to it
            # by mistake: start on the next free port instead.
            print(f"  Port {want} is busy (another program uses it); Tlav will use {got}.")
            args[i] = str(got)
    log = open(config.home() / "server.log", "w")
    kw = {}
    if detach:
        if os.name == "nt":
            kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
    proc = subprocess.Popen([exe] + args, stdout=log, stderr=subprocess.STDOUT, **kw)
    port = args[args.index("--port") + 1] if "--port" in args else "8080"
    host = args[args.index("--host") + 1] if "--host" in args else "127.0.0.1"
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    config.save_state(server_url=url, server_pid=proc.pid)
    if wait:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                raise RuntimeError(why_failed(config.home() / "server.log"))
            if healthy(url) and proc.poll() is None:
                break
            time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become ready in time.")
    return proc


def why_failed(log_path) -> str:
    """Turn the engine's log into a plain reason for a failed start."""
    try:
        text = open(log_path, errors="ignore").read()[-20000:]
    except OSError:
        text = ""
    low = text.lower()
    if "failed to load model" in low or "invalid magic" in low or "failed to read" in low \
            or "tensor data is not within" in low or "corrupted" in low:
        reason = ("the model file looks damaged or incomplete. Delete it and run `tlav` (or `tlav setup`) "
                  "again to download a fresh copy")
    elif "unknown model architecture" in low:
        reason = "this AI engine is too old for this model. Run `tlav install-runtime` to update it"
    elif "failed to allocate" in low or "out of memory" in low or "cannot allocate" in low:
        reason = "there wasn't enough free memory. Close other programs, or run `tlav setup` to pick a smaller model"
    elif "rpc" in low and ("failed" in low or "error" in low):
        reason = ("a cluster worker didn't accept the model. Check `tlav worker` runs on each computer and that they "
                  "all run the same AI engine build (`tlav install-runtime` on each)")
    else:
        errors = [l.split(" E ", 1)[-1].strip() for l in text.splitlines() if " E " in l]
        reason = errors[-1] if errors else "it stopped while starting"
    return f"The model couldn't start: {reason}.\n  Details: {log_path}"


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def stream_chat(url: str, messages: list, max_tokens: int = 1024) -> str:
    body = json.dumps({"messages": messages, "stream": True, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
            except (ValueError, KeyError, IndexError):
                continue
            sys.stdout.write(delta)
            sys.stdout.flush()
            text.append(delta)
    print()
    return "".join(text)


def chat_loop(url: str, system: Optional[str] = None) -> None:
    messages = [{"role": "system", "content": system}] if system else []
    print("Chatting with your local model. Type /exit to quit, /reset to clear history.\n")
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return
        if user == "/reset":
            messages = messages[:1] if system else []
            continue
        messages.append({"role": "user", "content": user})
        sys.stdout.write("ai  > ")
        reply = stream_chat(url, messages)
        messages.append({"role": "assistant", "content": reply})


# ------------------------------------------------------------------ status / stop

def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        # reap it if we started it ourselves (a finished child stays a "zombie" until reaped)
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except (ChildProcessError, OSError):
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def memory_gb(pid: int) -> dict:
    """{'used': private memory, 'cache': reclaimable file-mapped pages} in GB, best effort.

    Plain RSS double-counts on CPUs where llama.cpp repacks weights into a
    faster layout: the original file stays mapped but the OS can drop it.
    """
    path = f"/proc/{pid}/status"
    if os.path.exists(path):
        text = open(path).read()
        def kb(key):
            m = re.search(key + r":\s+(\d+) kB", text)
            return int(m.group(1)) / 1024 ** 2 if m else 0.0
        return {"used": kb("RssAnon") + kb("RssShmem"), "cache": kb("RssFile")}
    rss = rss_gb(pid)
    return {"used": rss, "cache": None}


def rss_gb(pid: int) -> Optional[float]:
    """Resident memory of a process in GB (best effort, no dependencies)."""
    try:
        if os.path.exists(f"/proc/{pid}/status"):
            m = re.search(r"VmRSS:\s+(\d+) kB", open(f"/proc/{pid}/status").read())
            return int(m.group(1)) / 1024 ** 2 if m else None
        if os.name != "nt":
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
            return int(out.strip()) / 1024 ** 2 if out.strip() else None
    except (OSError, ValueError):
        pass
    return None


def get_json(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def status() -> dict:
    st = config.load_state()
    pid, url = st.get("server_pid"), st.get("server_url")
    alive = pid_alive(pid) and is_engine(pid)
    info = {"running": bool(alive and url and healthy(url)), "pid": pid if alive else None, "url": url,
            "active": st.get("active")}
    if alive:
        info["memory"] = memory_gb(pid)
        props = get_json(url + "/props") if url else None
        if props:
            info["model_path"] = props.get("model_path")
            info["ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
    return info


def is_engine(pid: Optional[int]) -> bool:
    """Is this process really Tlav's llama-server? (A saved PID can be reused by another program.)"""
    if not pid:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True).stdout.lower()
            return "llama-server" in out
        cmdline = f"/proc/{pid}/cmdline"
        if os.path.exists(cmdline):
            return b"llama-server" in open(cmdline, "rb").read()
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
        return "llama-server" in out
    except OSError:
        return True   # can't tell: fall back to trusting the saved PID


def stop() -> bool:
    st = config.load_state()
    pid = st.get("server_pid")
    if not pid_alive(pid) or not is_engine(pid):
        if pid:
            config.save_state(server_pid=None)   # it's gone (or the number now belongs to another program)
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGTERM)
        # wait until it has really exited and freed its memory and port
        for _ in range(100):
            if not pid_alive(pid):
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
    config.save_state(server_pid=None)
    return True
