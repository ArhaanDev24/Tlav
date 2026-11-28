"""Tlav chat: a terminal chat for the model Tlav set up.

Streams answers, shows speed after each reply, keeps conversations on disk,
trims old turns to fit the model's context and lets Ctrl+C stop an answer
without leaving the chat. Standard library only.
"""

import json
import os
import re
import shutil
import signal
import threading
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
  /thoughts         show or hide the reasoning text itself
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
        self.thought_chunks = 0
        self.rows = shutil.get_terminal_size((80, 24)).lines
        self.live = True          # live tokens/s on the bottom row while writing
        self.status_on = False
        self.resized = False
        self.tokens = 0
        self.write_t0 = 0.0
        self.markdown = True     # show **bold**, `code` and headings as styles, not symbols
        self.bold = self.code = self.heading = self.star = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True) if enabled else None

    def _frame(self) -> str:
        return accent(self.FRAMES[self.i % 4])

    def verb(self) -> str:
        return self.verbs[int((time.time() - self.t0) / self.verb_every) % len(self.verbs)]

    def _header(self) -> str:
        secs = int(time.time() - self.t0)
        extra = (f"  {secs}s" if secs else "") + (f" · thinking, {self.thought_chunks} tokens"
                                                    if self.thought_chunks else "")
        line = f"  {self.FRAMES[self.i % 4]} {self.verb()}…{extra}"
        if len(line) > self.width - 1:   # never wrap the status line, or \r can't redraw it
            line = line[: self.width - 2] + "…"
        head, rest = line[:3], line[3:]
        verb_end = rest.find("…") + 1
        return "\r\033[K" + head[:2] + accent(head[2]) + rest[:verb_end] + dim(rest[verb_end:])

    def thought(self, text: str) -> None:
        """Reasoning text arrived: count it on the status line instead of printing it."""
        if not self.on:
            return
        with self.lock:
            self.thought_chunks += 1
            if not self.writing:
                sys.stdout.write(self._header())
                sys.stdout.flush()

    def _run(self):
        last_status = 0.0
        while not self._stop.wait(self.interval):
            with self.lock:
                self.i += 1
                if self.resized:
                    self._on_resize()
                sys.stdout.write("\b" + self._frame() if self.writing else self._header())
                if self.status_on and time.time() - last_status >= 1.0:
                    last_status = time.time()
                    self._draw_status()
                sys.stdout.flush()

    # -- live speed on a status line pinned to the bottom row while the answer is written
    def count(self, n: int = 1) -> None:
        """One more token of the answer arrived."""
        self.tokens += n

    def _status_text(self) -> str:
        secs = max(0.0, time.time() - (self.write_t0 or time.time()))
        tps = self.tokens / secs if secs >= 0.5 else 0.0
        return f"  {self.tokens} tokens · {tps:.1f} tokens/s · {int(secs)}s" + "   Ctrl+C stops"

    def _draw_status(self) -> None:
        # save the cursor, write on the reserved last row, come back: the answer is not disturbed
        sys.stdout.write(f"\0337\033[{self.rows};1H\033[2K" + dim(self._status_text()[: self.width - 1]) + "\0338")

    def _status_begin(self) -> None:
        if self.rows < 6:
            return
        # make sure the cursor is not on the last row, then keep that row out of the scrolling area
        sys.stdout.write("\n\033[1A" + f"\0337\033[1;{self.rows - 1}r\0338")
        self.status_on = True
        self._draw_status()

    def _status_end(self) -> None:
        if not self.status_on:
            return
        self.status_on = False
        sys.stdout.write(f"\0337\033[r\033[{self.rows};1H\033[2K\0338")

    def start(self, prefix: str = "  ") -> None:
        self.t0 = time.time()
        if not self.on:
            sys.stdout.write(prefix)
            sys.stdout.flush()
            return
        sys.stdout.write(self._header())
        sys.stdout.flush()
        # follow window resizes: the reserved status row and the wrap width depend on the size
        self._old_winch = None
        if hasattr(signal, "SIGWINCH") and threading.current_thread() is threading.main_thread():
            self._old_winch = signal.signal(signal.SIGWINCH, lambda *a: setattr(self, "resized", True))
        if self._thread:
            self._thread.start()

    def _on_resize(self) -> None:
        """The window changed size: move the status row and re-wrap at the new width."""
        self.resized = False
        size = shutil.get_terminal_size((self.width, self.rows))
        old = self.rows
        self.width, self.rows = max(20, size.columns), size.lines
        if self.status_on:
            # wipe the status where it was, release the old region, reserve the new last row
            sys.stdout.write(f"\0337\033[{old};1H\033[2K\033[r\0338")
            if self.rows >= 6:
                sys.stdout.write(f"\0337\033[1;{self.rows - 1}r\0338")
                self._draw_status()
            else:
                self.status_on = False
    # Markdown inside a word is kept as invisible markers until the word is printed, so the
    # wrapping math only counts letters people actually see.
    MARKS = {"\x01": "\033[1m", "\x02": "\033[22m", "\x03": "\033[38;5;180m", "\x04": "\033[39m"}

    @classmethod
    def _visible(cls, s: str) -> str:
        return "".join(ch for ch in s if ch not in cls.MARKS)

    def _ansi(self, s: str) -> str:
        return "".join(self.MARKS.get(ch, ch) for ch in s)

    def _word_out(self) -> str:
        """Print the pending word, moving to a new indented line first if it doesn't fit."""
        w, self.word = self.word, ""
        if not w:
            return ""
        limit = self.limit
        out = []
        n = len(self._visible(w))
        if self.col + n > limit and self.col > 2:
            out.append("\n  ")
            self.col = 2
        if n > limit - self.col:            # a word longer than a whole line: cut it (plain)
            w = self._visible(w)
            while len(w) > limit - self.col:
                cut = limit - self.col
                out.append(self._paint(w[:cut]) + "\n  ")
                w, self.col = w[cut:], 2
            n = len(w)
        out.append(self._paint(self._ansi(w)))
        self.col += n
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
            if self.star:                   # a lone '*' before this character
                self.star = False
                if ch == "*":               # '**' toggles bold
                    self.bold = not self.bold
                    self.word += "\x01" if self.bold else "\x02"
                    continue
                at_line_start = not self.word and self.col <= 2
                self.word += "•" if (ch == " " and at_line_start) else "*"
            if ch == "*" and self.markdown:
                self.star = True
            elif ch == "`" and self.markdown:
                self.code = not self.code
                self.word += "\x03" if self.code else "\x04"
            elif ch == "\n":
                if self.bold or self.code or self.heading:     # markdown never spans lines
                    self.word += ("\x02" if self.bold or self.heading else "") + ("\x04" if self.code else "")
                    self.bold = self.code = self.heading = False
                out.append(self._word_out() + "\n")
                self.col = 0
            elif ch == " ":
                if (self.markdown and self.word and self.col <= 2 and len(self.word) <= 6
                        and set(self.word) == {"#"}):           # '## Heading' -> a bold line
                    self.word, self.heading = "\x01", True
                    continue
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
                self.write_t0 = time.time()
                sys.stdout.write(f"\r\033[K  {accent('■')} {bold('Tlav')}\n")
                if self.live:
                    self._status_begin()
                sys.stdout.write("  ")
            else:
                sys.stdout.write("\b \b")
            sys.stdout.write(out + self._flow(text) + self._frame())
            sys.stdout.flush()

    def stop(self) -> None:
        if not self.on or self._stop.is_set():
            return
        self._stop.set()
        if getattr(self, "_old_winch", None) is not None and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGWINCH, self._old_winch)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        with self.lock:
            if self.writing:
                if self.star:
                    self.word += "*"
                    self.star = False
                sys.stdout.write("\b \b" + self._word_out() + ("\033[0m" if self.bold or self.code or self.heading else ""))
                self._status_end()
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


def shorten(text: str, max_tokens: int, chars_per_token: float = 3.0) -> str:
    """Keep the start and the end of a message that is too long for the model's memory."""
    keep = max(200, int(max_tokens * chars_per_token))
    if len(text) <= keep:
        return text
    head = int(keep * 0.7)
    return text[:head] + "\n\n[… the middle of this message was left out to fit the model's memory …]\n\n" \
        + text[-(keep - head):]


_WIN_RESERVED = {"con", "prn", "aux", "nul"} | {f"{d}{i}" for d in ("com", "lpt") for i in range(1, 10)}


def safe_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")[:60] or "chat"
    # Windows can't create files called CON, NUL, AUX... (even with an extension)
    return "chat-" + out if out.split(".")[0].lower() in _WIN_RESERVED else out


# ------------------------------------------------------------------ streaming

class Answer:
    def __init__(self):
        self.text = ""
        self.reasoning = ""
        self.tokens = 0
        self.tps = 0.0
        self.seconds = 0.0
        self.stopped = False
        self.finished = False   # the server said the answer is complete
        self.hit_limit = False  # ...because it reached the length limit

    @property
    def cut(self) -> bool:
        """The stream ended without the server finishing the answer (it crashed, or a cluster
        worker dropped out) and not because the user pressed Ctrl+C."""
        return not self.finished and not self.stopped


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
                if choices[0].get("finish_reason"):
                    ans.finished = True
                    ans.hit_limit = choices[0]["finish_reason"] == "length"
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
        self.think = think if think is not None else default_think(info)
        self.show_thoughts = False   # reasoning runs quietly behind the status line unless asked
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
        """False if there's no such chat; ValueError if the file is damaged."""
        path = self.chats_dir() / (safe_name(name) + ".json")
        if not path.exists():
            return False
        try:
            msgs = json.loads(path.read_text()).get("messages", [])
        except (OSError, ValueError, AttributeError):
            raise ValueError("damaged") from None
        if not isinstance(msgs, list) or not all(isinstance(m, dict) and isinstance(m.get("role"), str)
                                                 and isinstance(m.get("content"), str) for m in msgs):
            raise ValueError("damaged")
        self.messages = msgs
        return True

    # -- output
    def header(self) -> None:
        i = self.info
        line = f"{i.get('model', 'local model')}  {dim(i.get('quant', ''))}"
        print()
        print(f"  {accent('■')} {bold('Tlav')}  {line}")
        extra = f" · reasoning {'on' if self.think else 'off'}" if self.reasons else ""
        print(dim(f"    {i.get('where', 'this PC')} · {self.ctx} tokens of memory · offline{extra}"))
        from .update import cached_newer
        latest = cached_newer()
        if latest:
            print(dim(f"    Tlav {latest} is out: after this chat, type  tlav update"))
        print(dim("    Type a message. /help for commands, Ctrl+C stops an answer.\n"))

    def ask(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        # thinking needs room for the reasoning *and* the answer
        budget = max(self.max_tokens, self.ctx // 2) if (self.think and self.reasons) else self.max_tokens
        prompt, dropped = trim_history(self.messages, self.ctx, budget)
        if dropped:
            print(dim(f"  (oldest {dropped} messages left out so the model has room to answer)"))
        room = self.ctx - budget - estimate_tokens(prompt[:-1]) - 16
        if estimate_tokens(prompt[-1:]) > room:
            prompt[-1] = dict(prompt[-1], content=shorten(prompt[-1]["content"], room))
            print(dim("  (your message is longer than this model can hold at once; Tlav kept its start and end)"))
        state = {"kind": None}
        mark = Mark()

        def on_text(kind, chunk):
            if kind == "think" and (not self.think or not self.show_thoughts):
                mark.thought(chunk)     # the status line counts it; the text stays out of the way
                return
            if kind != state["kind"]:
                if state["kind"] is not None:
                    mark.write("\n\n  " if state["kind"] == "think" else "\n  ")
                if kind == "think":
                    mark.write("thinking · ", dim)
                state["kind"] = kind
            out = chunk.replace("\n", "\n  ")
            if kind == "answer":
                mark.count()
            mark.write(out, dim if kind == "think" else None)

        mark.start("  ")   # the square starts moving the moment you press Enter
        try:
            try:
                ans = stream(self.url, prompt, self.temperature, budget,
                             self.think if self.reasons else None, on_text)
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
                # still too long for the model (our token estimate was low): try once with half
                prompt[-1] = dict(prompt[-1], content=shorten(prompt[-1]["content"], max(64, room // 2), 2.0))
                ans = stream(self.url, prompt, self.temperature, budget,
                             self.think if self.reasons else None, on_text)
        except urllib.error.HTTPError as e:
            mark.stop()
            self.messages.pop()
            print("\n  " + ("That message is too long for this model's memory. Shorten it or send it in parts."
                            if e.code == 400 else f"The model returned an error ({e.code})."))
            return
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            mark.stop()
            self.messages.pop()
            gone = not server.healthy(self.url)
            print("\n  " + ("The model isn't running any more. " if gone else f"The model isn't answering ({e}). ")
                  + self._restart_hint())
            return
        finally:
            mark.stop()
        print()
        if ans.cut and (ans.text or ans.reasoning or ans.tokens):
            print("  " + accent("!") + " The answer was cut off: the model stopped responding. " + self._restart_hint())
        elif ans.hit_limit and ans.text:
            print(dim("  (the answer reached its length limit; ask it to continue)"))
        if ans.text:
            self.messages.append({"role": "assistant", "content": ans.text})
            self.last = ans.text
        else:
            self.messages.pop()
            if ans.reasoning and not ans.stopped:
                print(dim("  It used all its room thinking and never got to the answer. /think off answers "
                          "directly; /thoughts shows the reasoning."))
        if ans.tps:
            self.speeds.append(ans.tps)
        stop = " · stopped" if ans.stopped else ""
        print(dim(f"  {ans.tokens} tokens · {ans.tps:.1f} tokens/s · {ans.seconds:.1f}s{stop}") + "\n")

    def _restart_hint(self) -> str:
        if str(self.info.get("where", "")).endswith("computers"):
            return ("In a cluster this usually means a worker computer went offline: check `tlav cluster list`, "
                    "start `tlav worker` there again, then type `tlav`.")
        return "Type `tlav` to start it again (or `tlav status` to check it)."

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
        elif name in ("thoughts", "thought") and self.reasons:
            self.show_thoughts = {"on": True, "off": False}.get(arg.lower(), not self.show_thoughts)
            print(dim(f"  Reasoning text {'shown as it is written' if self.show_thoughts else 'hidden (the status line shows progress)'}."))
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
            try:
                print(dim(f"  Saved to {self.save(name_)}"))
            except OSError as e:
                print(dim(f"  Couldn't save the chat ({e.strerror or e}). Is the disk full?"))
        elif name in ("load", "open"):
            try:
                found = bool(arg) and self.load(arg)
            except ValueError:
                print(dim(f"  The saved chat '{arg}' is damaged and can't be opened."))
                return True
            if not arg:
                print(dim("  Usage: /load NAME (see /list)"))
            elif found:
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
            try:
                self.save("last")
            except OSError:
                pass       # a full disk must not stop the chat; /save reports it when asked

    # -- loop
    def read(self) -> Optional[str]:
        # \001/\002 mark the colour codes as invisible for readline, and the reset must come before
        # the final space: readline drops a code at the very end, which left typed text orange.
        line = input("\001\033[38;5;209m\002›\001\033[0m\002 " if _COLOR else "› ")
        if line.strip() != '"""':
            return line.strip()
        lines = []
        while True:
            more = input("\001\033[2m\002·\001\033[0m\002 " if _COLOR else "· ")
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
            if server.healthy(self.url):
                print(dim("  Bye. The model keeps running in the background: `tlav` brings you back, "
                          "`tlav stop` frees its memory."))
            else:
                print(dim("  Bye. The model has stopped; type `tlav` to start it again."))


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


def default_think(info: Dict) -> bool:
    """Reason before answering only where it pays: a fast PC and a model big enough to reason
    well (tiny models can spend minutes second-guessing a greeting)."""
    if not info.get("reasons", True):
        return False
    params = info.get("params_b")
    return float(info.get("est_tps") or 0) >= 8 and (params is None or params >= 3)


def file_info(path: str, ctx: Optional[int] = None, est_tps: Optional[float] = None) -> Dict:
    """Header details for any GGUF file (used when the running model isn't the set-up one)."""
    from . import gguf
    stem = Path(path).stem
    m = re.match(r"(.+?)[-.](I?Q\d[A-Z0-9_]*|BF16|F16|FIT[0-9.]*G?)(?:[-.].*)?$", stem, re.I)
    name, quant = (m.group(1), m.group(2).upper()) if m else (stem, "")
    if "_" in name:                      # publisher prefix: Qwen_Qwen3.5-0.8B -> Qwen3.5-0.8B
        head, rest = name.split("_", 1)
        if rest.lower().startswith(head.lower()) or (head.islower() and head.isalpha()):
            name = rest                  # ...and google_gemma-3-1b -> gemma-3-1b
    name = name.replace("_", " ").replace("-", " ")
    info = {"model": name, "quant": quant, "ctx": ctx, "est_tps": est_tps, "where": "this PC"}
    try:
        g = gguf.read(path)
        info.update(reasons=g.reasons, params_b=g.n_params / 1e9)
    except (ValueError, OSError, EOFError):
        info.update(reasons=True)
    return info


def running_info(status: Dict) -> Dict:
    """Details of the model the server is actually running."""
    active = status.get("active") or {}
    path = status.get("model_path")
    if not path or (active.get("path") and Path(active["path"]).resolve() == Path(path).resolve()):
        info = model_info(active)
    else:
        info = file_info(path, ctx=status.get("ctx"))
    rpc = config.load_state().get("server_rpc")
    if rpc:
        info["where"] = f"{len(rpc) + 1} computers"
    return info


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
        m = catalog.get(p.get("model", ""))
        reasons, params = "reasoning" in m.tags, m.params_b
    except KeyError:
        reasons, params = True, None
    return {"model": name, "quant": p.get("quant", ""), "reasons": reasons, "params_b": params, "ctx": active.get("settings", {}).get("ctx") or p.get("ctx"),
            "where": f"{len(nodes)} computers" if nodes else "this PC", "est_tps": p.get("est_tps")}
