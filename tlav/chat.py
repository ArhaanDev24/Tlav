"""Tlav chat: a terminal chat for the model Tlav set up.

Streams answers, shows speed after each reply, keeps conversations on disk,
trims old turns to fit the model's context and lets Ctrl+C stop an answer
without leaving the chat. Standard library only.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import config, server

# ------------------------------------------------------------------ look

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


dim = lambda s: _c("2", s)           # noqa: E731
bold = lambda s: _c("1", s)          # noqa: E731
accent = lambda s: _c("38;5;209", s)  # noqa: E731  Tlav orange
green = lambda s: _c("38;5;108", s)  # noqa: E731

HELP = """\
  /new              start a fresh conversation
  /system TEXT      set how the assistant should behave (empty: clear)
  /think [on|off]   let the model reason before answering (slower, smarter)
  /temp X           creativity, 0 = precise, 1 = creative (now {temp})
  /save [NAME]      save this conversation
  /load NAME        open a saved conversation (/load last = the previous one)
  /list             list saved conversations
  /copy             copy the last answer
  /stats            speed, memory and model details
  /web              open the browser chat
  /stop             stop the model and quit
  /exit             quit (the model keeps running in the background)
  \"\"\"               start/end a message with several lines
  Ctrl+C stops an answer; Ctrl+D quits."""

CHARS_PER_TOKEN = 3.5


# What Tlav says while the model gets going: a nod to the old machines it runs on.
VERBS = [
    "Defragmenting the C: drive", "Blowing dust out of the fan", "Waiting for the dial-up handshake",
    "Rewinding the cassette with a pencil", "Inserting disk 2 of 7", "Warming up the CRT",
    "Pressing any key", "Turning it off and on again", "Blowing on the cartridge",
    "Degaussing the monitor", "Spinning up the 5,400 RPM drive", "Feeding the punch cards",
    "Consulting the BIOS", "Counting to 640K", "Checking the turbo button",
    "Calibrating the ball mouse", "Untangling the printer cable", "Paging to swap, sorry",
    "Waiting for the screensaver to finish", "Adjusting the rabbit ears", "Rebooting the hamster wheel",
    "Asking the paperclip for help", "Compiling, go make some tea", "Tlaving",
]


class Mark:
    """The Tlav square, animated while the model answers.

    The logo is a dark square with an orange top-right quarter; here that quarter moves
    corner to corner (top-right, bottom-right, bottom-left, top-left).

    1. Waiting: a header line at the top left shows the moving square, an old-computer
       phrase that changes every few seconds, and the seconds elapsed.
    2. Writing: the header settles into "■ Tlav" and the square follows the last
       character like a pen until the answer is done.

    Only on real terminals; plain output stays plain.
    """
    FRAMES = "▝▗▖▘"

    def __init__(self, enabled: bool = _COLOR, interval: float = 0.14, verb_every: float = 2.8,
                 verbs: Optional[List[str]] = None):
        import random
        import threading
        self.on = enabled
        self.interval = interval
        self.verb_every = verb_every
        self.verbs = list(verbs or VERBS)
        random.shuffle(self.verbs)
        self.lock = threading.Lock()
        self.i = 0
        self.writing = False
        self.t0 = time.time()
        self.col = 0          # cursor column, so Tlav wraps lines itself (keeps the indent,
        self.word = ""        # never splits a word, leaves a column for the moving square)
        self.style = None
        self.width = max(20, shutil.get_terminal_size((80, 24)).columns)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True) if enabled else None

    def _frame(self) -> str:
        return accent(self.FRAMES[self.i % 4])

    def verb(self) -> str:
        return self.verbs[int((time.time() - self.t0) / self.verb_every) % len(self.verbs)]

    def _header(self) -> str:
        secs = int(time.time() - self.t0)
        return f"\r\033[K  {self._frame()} {self.verb()}…" + dim(f"  {secs}s" if secs else "")

    def _run(self):
        while not self._stop.wait(self.interval):
            with self.lock:
                self.i += 1
                sys.stdout.write("\b" + self._frame() if self.writing else self._header())
                sys.stdout.flush()

    def start(self, prefix: str = "  ") -> None:
        self.t0 = time.time()
        if not self.on:
            sys.stdout.write(prefix)
            sys.stdout.flush()
            return
        sys.stdout.write(self._header())
        sys.stdout.flush()
        if self._thread:
            self._thread.start()

    def _word_out(self) -> str:
        """Print the pending word, moving to a new indented line first if it doesn't fit."""
        w, self.word = self.word, ""
        if not w:
            return ""
        limit = self.limit
        out = []
        if self.col + len(w) > limit and self.col > 2:
            out.append("\n  ")
            self.col = 2
        while len(w) > limit - self.col:   # a word longer than a whole line
            cut = limit - self.col
            out.append(self._paint(w[:cut]) + "\n  ")
            w, self.col = w[cut:], 2
        out.append(self._paint(w))
        self.col += len(w)
        return "".join(out)

    @property
    def limit(self) -> int:
        # Text stops two columns before the edge: the square takes the next one, and it must
        # never sit in the very last column, where a backspace lands one column short.
        return self.width - 2

    def _paint(self, s: str) -> str:
        return self.style(s) if self.style else s

    def _flow(self, text: str) -> str:
        """Word-wrap streamed text to the terminal width, keeping the 2-space indent.
        Letters wait in `word` until the word ends, so a word is never split."""
        out = []
        for ch in text:
            if ch == "\n":
                out.append(self._word_out() + "\n")
                self.col = 0
            elif ch == " ":
                out.append(self._word_out())
                if self.col < self.limit:
                    out.append(" ")
                    self.col += 1
            else:
                self.word += ch
        return "".join(out)

    def write(self, text: str, style=None) -> None:
        if not self.on:
            sys.stdout.write(style(text) if style else text)
            sys.stdout.flush()
            return
        with self.lock:
            out = ""
            if style is not self.style:     # finish the pending word in its own style
                out = self._word_out()
                self.style = style
            if not self.writing:            # first words: the header settles into the logo
                self.writing = True
                self.col = 2
                sys.stdout.write(f"\r\033[K  {accent('■')} {bold('Tlav')}\n  ")
            else:
                sys.stdout.write("\b \b")
            sys.stdout.write(out + self._flow(text) + self._frame())
            sys.stdout.flush()

    def stop(self) -> None:
        if not self.on or self._stop.is_set():
            return
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        with self.lock:
            if self.writing:
                sys.stdout.write("\b \b" + self._word_out())
            else:
                sys.stdout.write("\r\033[K")
            sys.stdout.flush()


# ------------------------------------------------------------------ pure helpers (unit tested)

def estimate_tokens(messages: List[dict]) -> int:
    return sum(int(len(m.get("content") or "") / CHARS_PER_TOKEN) + 4 for m in messages)


def trim_history(messages: List[dict], ctx: int, reserve: int) -> Tuple[List[dict], int]:
    """Drop the oldest turns until the prompt fits in ctx - reserve tokens.

    The system prompt and the newest message are always kept. Returns the
    trimmed list and how many messages were dropped.
    """
    system = [m for m in messages[:1] if m["role"] == "system"]
    rest = messages[len(system):]
    budget = max(256, ctx - reserve)
    dropped = 0
    while len(rest) > 1 and estimate_tokens(system + rest) > budget:
        # drop a whole user/assistant pair when possible
        n = 2 if len(rest) > 2 and rest[0]["role"] == "user" and rest[1]["role"] == "assistant" else 1
        rest = rest[n:]
        dropped += n
    return system + rest, dropped


class ThinkSplitter:
    """Separates <think>...</think> reasoning from the answer in streamed text."""

    def __init__(self):
        self.buf = ""
        self.inside = False

    def feed(self, text: str) -> List[Tuple[str, str]]:
        """Returns [(kind, text)] with kind 'think' or 'answer'."""
        self.buf += text
        out = []
        while self.buf:
            tag = "</think>" if self.inside else "<think>"
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    out.append(("think" if self.inside else "answer", self.buf[:i]))
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            # keep a possible partial tag at the end for the next chunk
            keep = next((k for k in range(min(len(tag) - 1, len(self.buf)), 0, -1)
                         if tag.startswith(self.buf[-k:])), 0)
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                out.append(("think" if self.inside else "answer", emit))
            self.buf = self.buf[len(emit):]
            break
        return out


def parse_command(line: str) -> Tuple[str, str]:
    """'/temp 0.3' -> ('temp', '0.3'); plain text -> ('', text)."""
    if not line.startswith("/") or line.startswith("//"):
        return "", line[1:] if line.startswith("//") else line
    name, _, arg = line[1:].partition(" ")
    return name.lower().strip(), arg.strip()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")[:60] or "chat"


# ------------------------------------------------------------------ streaming

class Answer:
    def __init__(self):
        self.text = ""
        self.reasoning = ""
        self.tokens = 0
        self.tps = 0.0
        self.seconds = 0.0
        self.stopped = False


def stream(url: str, messages: List[dict], temperature: float, max_tokens: int, think: Optional[bool],
           on_text: Callable[[str, str], None]) -> Answer:
    """POST a streaming chat request; calls on_text(kind, text) as tokens arrive."""
    body = {"messages": messages, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
            "timings_per_token": False}
    if think is not None:
        body["chat_template_kwargs"] = {"enable_thinking": think}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ans, split, t0, first = Answer(), ThinkSplitter(), time.time(), None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                t = chunk.get("timings")
                if t:
                    ans.tokens = int(t.get("predicted_n") or ans.tokens)
                    ans.tps = float(t.get("predicted_per_second") or ans.tps)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                parts = []
                if delta.get("reasoning_content"):
                    parts.append(("think", delta["reasoning_content"]))
                if delta.get("content"):
                    parts += split.feed(delta["content"])
                for kind, text in parts:
                    if first is None:
                        first = time.time()
                    if kind == "think":
                        ans.reasoning += text
                    else:
                        ans.text += text
                    if not ans.tps:
                        ans.tokens += 1   # rough count until the server reports timings
                    on_text(kind, text)
    except KeyboardInterrupt:
        ans.stopped = True
    ans.seconds = time.time() - t0
    if not ans.tps and first and ans.tokens:
        ans.tps = ans.tokens / max(0.01, time.time() - first)
    ans.text = ans.text.strip()
    return ans


# ------------------------------------------------------------------ the chat session

class Chat:
    def __init__(self, url: str, info: Dict, system: Optional[str] = None, think: Optional[bool] = None):
        self.url = url
        self.info = info                      # model, quant, ctx, where, est_tps
        self.messages: List[dict] = [{"role": "system", "content": system}] if system else []
        self.temperature = 0.7
        self.ctx = int(info.get("ctx") or 4096)
        self.max_tokens = max(256, min(2048, self.ctx // 2))
        # Reasoning makes answers better but on a slow PC it can take minutes: default it on
        # only when the machine is quick enough.
        self.reasons = info.get("reasons", True)   # model can think before answering (Qwen3/3.5...)
        self.think = think if think is not None else (self.reasons and float(info.get("est_tps") or 0) >= 8)
        self.speeds: List[float] = []
        self.stopped = False
        self.last = ""

    # -- files
    @staticmethod
    def chats_dir() -> Path:
        p = config.home() / "chats"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, name: str) -> Path:
        path = self.chats_dir() / (safe_name(name) + ".json")
        path.write_text(json.dumps({"model": self.info.get("model"), "saved": time.time(),
                                    "messages": self.messages}, indent=1))
        return path

    def load(self, name: str) -> bool:
        path = self.chats_dir() / (safe_name(name) + ".json")
        if not path.exists():
            return False
        self.messages = json.loads(path.read_text()).get("messages", [])
        return True

    # -- output
    def header(self) -> None:
        i = self.info
        line = f"{i.get('model', 'local model')}  {dim(i.get('quant', ''))}"
        print()
        print(f"  {accent('■')} {bold('Tlav')}  {line}")
        extra = f" · reasoning {'on' if self.think else 'off'}" if self.reasons else ""
        print(dim(f"    {i.get('where', 'this PC')} · {self.ctx} tokens of memory · offline{extra}"))
        print(dim("    Type a message. /help for commands, Ctrl+C stops an answer.\n"))

    def ask(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        prompt, dropped = trim_history(self.messages, self.ctx, self.max_tokens)
        if dropped:
            print(dim(f"  (oldest {dropped} messages left out so the model has room to answer)"))
        state = {"kind": None}
        mark = Mark()

        def on_text(kind, chunk):
            if kind == "think" and not self.think:
                return
            if kind != state["kind"]:
                if state["kind"] is not None:
                    mark.write("\n\n  " if state["kind"] == "think" else "\n  ")
                if kind == "think":
                    mark.write("thinking · ", dim)
                state["kind"] = kind
            out = chunk.replace("\n", "\n  ")
            mark.write(out, dim if kind == "think" else None)

        mark.start("  ")   # the square starts moving the moment you press Enter
        try:
            ans = stream(self.url, prompt, self.temperature, self.max_tokens,
                         self.think if self.reasons else None, on_text)
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            mark.stop()
            self.messages.pop()
            print(f"\n  The model stopped answering ({e}). Is it still running? Try `tlav status`.")
            return
        finally:
            mark.stop()
        print()
        if ans.text:
            self.messages.append({"role": "assistant", "content": ans.text})
            self.last = ans.text
        else:
            self.messages.pop()
        if ans.tps:
            self.speeds.append(ans.tps)
        stop = " · stopped" if ans.stopped else ""
        print(dim(f"  {ans.tokens} tokens · {ans.tps:.1f} tokens/s · {ans.seconds:.1f}s{stop}") + "\n")

    # -- commands
    def command(self, name: str, arg: str) -> bool:
        """Run a /command. Returns False to leave the chat."""
        if name in ("exit", "quit", "q", "bye"):
            return False
        if name in ("help", "h", "?"):
            print(HELP.format(temp=self.temperature))
        elif name in ("new", "reset", "clear"):
            self.autosave()
            self.messages = [m for m in self.messages[:1] if m["role"] == "system"]
            print(dim("  New conversation."))
        elif name == "system":
            self.messages = [m for m in self.messages if m["role"] != "system"]
            if arg:
                self.messages.insert(0, {"role": "system", "content": arg})
            print(dim("  System prompt " + ("set." if arg else "cleared.")))
        elif name == "think" and not self.reasons:
            print(dim("  This model answers directly; it has no reasoning mode."))
        elif name == "think":
            self.think = {"on": True, "off": False}.get(arg.lower(), not self.think)
            print(dim(f"  Reasoning {'on: better answers, slower' if self.think else 'off: faster answers'}."))
        elif name in ("temp", "temperature"):
            try:
                self.temperature = max(0.0, min(2.0, float(arg)))
                print(dim(f"  Temperature {self.temperature}."))
            except ValueError:
                print(dim(f"  Usage: /temp 0.7 (now {self.temperature})"))
        elif name == "save":
            name_ = arg or time.strftime("chat-%Y%m%d-%H%M")
            print(dim(f"  Saved to {self.save(name_)}"))
        elif name in ("load", "open"):
            if not arg:
                print(dim("  Usage: /load NAME (see /list)"))
            elif self.load(arg):
                n = sum(1 for m in self.messages if m["role"] != "system")
                print(dim(f"  Loaded '{arg}' ({n} messages). Carry on where you left off."))
            else:
                print(dim(f"  No saved chat '{arg}'. See /list."))
        elif name in ("list", "ls"):
            files = sorted(self.chats_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for f in files[:20]:
                print(dim(f"  {f.stem:<30} {time.strftime('%d %b %H:%M', time.localtime(f.stat().st_mtime))}"))
            if not files:
                print(dim("  No saved chats yet. /save NAME keeps this one."))
        elif name == "copy":
            print(dim("  Copied." if self.last and copy_text(self.last) else
                      "  Nothing to copy." if not self.last else "  No clipboard tool found."))
        elif name == "stats":
            st = server.status()
            mem = (st.get("memory") or {}).get("used")
            avg = sum(self.speeds) / len(self.speeds) if self.speeds else 0
            print(dim(f"  model    {self.info.get('model')} {self.info.get('quant', '')}"))
            print(dim(f"  runs on  {self.info.get('where', 'this PC')}"))
            print(dim(f"  memory   {mem:.2f} GB in use" if mem else "  memory   unknown"))
            print(dim(f"  speed    {avg:.1f} tokens/s average" + (f" (estimated {self.info['est_tps']})"
                                                                     if self.info.get("est_tps") else "")))
            print(dim(f"  context  ~{estimate_tokens(self.messages)} of {self.ctx} tokens used"))
        elif name == "web":
            server.open_browser(self.url)
            print(dim(f"  Opened {self.url}"))
        elif name == "stop":
            self.autosave()
            print(dim("  Model stopped, memory freed." if server.stop() else "  The model was not started by Tlav."))
            self.stopped = True
            return False
        else:
            print(dim(f"  Unknown command /{name}. /help lists them."))
        return True

    def autosave(self) -> None:
        if any(m["role"] == "user" for m in self.messages):
            self.save("last")

    # -- loop
    def read(self) -> Optional[str]:
        # \001/\002 tell readline the colour codes take no space, or the prompt shows them raw
        line = input("\001\033[38;5;209m\002› \001\033[0m\002" if _COLOR else "› ")
        if line.strip() != '"""':
            return line.strip()
        lines = []
        while True:
            more = input(dim("· "))
            if more.strip() == '"""':
                return "\n".join(lines).strip()
            lines.append(more)

    def loop(self) -> None:
        self.header()
        _setup_readline()
        while True:
            try:
                line = self.read()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print(dim("\n  (Ctrl+D or /exit to quit)"))
                continue
            if not line:
                continue
            name, arg = parse_command(line)
            if name:
                if not self.command(name, arg):
                    break
                continue
            self.ask(arg)
        self.autosave()
        if not self.stopped:
            print(dim("  Bye. The model keeps running in the background: `tlav` brings you back, `tlav stop` frees its memory."))


# ------------------------------------------------------------------ small system helpers

def copy_text(text: str) -> bool:
    for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"], ["clip"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=5)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


def _setup_readline() -> None:
    try:
        import atexit
        import readline
    except ImportError:  # Windows without pyreadline
        return
    hist = config.home() / "chat_history"
    try:
        readline.read_history_file(hist)
    except OSError:
        pass
    readline.set_history_length(1000)
    atexit.register(lambda: _write_history(readline, hist))


def _write_history(readline, hist: Path) -> None:
    try:
        readline.write_history_file(hist)
    except OSError:
        pass


def model_info(active: Optional[dict]) -> Dict:
    """Header details from the set-up model's plan."""
    if not active:
        return {}
    p = active.get("plan", {})
    from . import catalog
    try:
        name = catalog.get(p.get("model", "")).name
    except KeyError:
        name = Path(active.get("path", "model")).stem
    nodes = p.get("nodes")
    try:
        reasons = "reasoning" in catalog.get(p.get("model", "")).tags
    except KeyError:
        reasons = True
    return {"model": name, "quant": p.get("quant", ""), "reasons": reasons, "ctx": active.get("settings", {}).get("ctx") or p.get("ctx"),
            "where": f"{len(nodes)} computers" if nodes else "this PC", "est_tps": p.get("est_tps")}
