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


def start(args: List[str], wait: bool = True, timeout: int = 900, detach: bool = False) -> subprocess.Popen:
    exe = runtime.require("llama-server")
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
                raise RuntimeError(f"llama-server exited early; see {config.home() / 'server.log'}")
            if healthy(url):
                break
            time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become ready in time.")
    return proc


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
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
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
    alive = pid_alive(pid)
    info = {"running": bool(alive and url and healthy(url)), "pid": pid if alive else None, "url": url,
            "active": st.get("active")}
    if alive:
        info["memory"] = memory_gb(pid)
        props = get_json(url + "/props") if url else None
        if props:
            info["model_path"] = props.get("model_path")
            info["ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
    return info


def stop() -> bool:
    st = config.load_state()
    pid = st.get("server_pid")
    if not pid_alive(pid):
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGTERM)
    config.save_state(server_pid=None)
    return True
