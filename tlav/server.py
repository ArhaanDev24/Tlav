"""Run llama-server (OpenAI-compatible API + built-in web chat) and a terminal chat client."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from typing import List, Optional

from . import config, runtime


def start(args: List[str], wait: bool = True, timeout: int = 600) -> subprocess.Popen:
    exe = runtime.require("llama-server")
    log = open(config.home() / "server.log", "w")
    proc = subprocess.Popen([exe] + args, stdout=log, stderr=subprocess.STDOUT)
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
