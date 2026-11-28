"""Tlav's terminal look: the wordmark, animated step lines, a download bar and styled questions.

Everything degrades to plain text when the output isn't a real terminal (pipes, logs, CI).
"""

import os
import re
import shutil
import sys
import threading
import time
from typing import Optional

if os.name == "nt":
    os.system("")   # switches Windows 10+ consoles into ANSI mode so colours and boxes draw correctly
ON = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
ORANGE, GREEN, RED = "\033[38;5;209m", "\033[38;5;108m", "\033[38;5;167m"
LIGHT, DIM, BOLD, RESET = "\033[38;5;255m", "\033[2m", "\033[1m", "\033[0m"
FRAMES = "▝▗▖▘"     # the logo's orange quarter moving corner to corner


def c(code: str, s: str) -> str:
    return f"{code}{s}{RESET}" if ON else s


def width() -> int:
    return max(40, shutil.get_terminal_size((80, 24)).columns)


# ------------------------------------------------------------------ wordmark

# 5-row pixel letters; '#' is a filled cell. The logo sits on the left: a light square with an
# orange top-right quarter, the same mark as the website and the chat.
_LETTERS = {
    "T": ["#####", "..#..", "..#..", "..#..", "..#.."],
    "L": ["#....", "#....", "#....", "#....", "#####"],
    "A": [".###.", "#...#", "#####", "#...#", "#...#"],
    "V": ["#...#", "#...#", "#...#", ".#.#.", "..#.."],
}


def wordmark() -> list:
    """The TLAV wordmark as terminal lines (each pixel is two characters wide)."""
    rows = []
    for r in range(5):
        line = []
        for ch in "TLAV":
            line.append("".join("██" if px == "#" else "  " for px in _LETTERS[ch][r]))
        rows.append("  ".join(line))
    logo = [c(LIGHT, "████") + c(ORANGE, "████"), c(LIGHT, "████") + c(ORANGE, "████"),
            c(LIGHT, "████████"), c(LIGHT, "████████"), "        "]
    return [logo[r] + "   " + c(LIGHT, rows[r]) for r in range(5)]


def banner(subtitle: str = "", lines: Optional[list] = None) -> None:
    """A welcome box in the spirit of Claude Code's: the mark, a tagline and what happens next."""
    from . import __version__
    w = min(width() - 2, 74)
    if not ON or w < 60:
        print(f"Tlav {__version__} · Old PCs. New minds.")
        if subtitle:
            print(subtitle)
        for ln in lines or []:
            print("  " + ln)
        print()
        return
    inner = w - 4
    top = c(DIM, "╭" + "─" * (w - 2) + "╮")
    bot = c(DIM, "╰" + "─" * (w - 2) + "╯")
    side = c(DIM, "│")

    def row(text: str = "", visible: Optional[int] = None) -> str:
        n = len(re.sub(r"\033\[[0-9;]*m", "", text)) if visible is None else visible
        return f"{side} {text}{' ' * max(0, inner - n)} {side}"

    print(top)
    print(row())
    for ln in wordmark():
        print(row(ln, 8 + 3 + 4 * 10 + 3 * 2))     # logo, gap, four 5-pixel letters, letter gaps
    print(row())
    tag = f"Old PCs. New minds.  v{__version__}"
    print(row(c(BOLD, "Old PCs. New minds.") + "  " + c(DIM, "v" + __version__), len(tag)))
    if subtitle:
        print(row(c(DIM, subtitle), len(subtitle)))
    if lines:
        print(row())
        for ln in lines:
            print(row(ln))
    print(row())
    print(bot)
    print()


# ------------------------------------------------------------------ steps

class Step:
    """One line per stage: a moving square while it works, then a settled result.

        with ui.Step("Measuring your computer") as s:
            ...
            s.done("16 GB RAM · 4 cores")
    """
    LABEL = 20

    def __init__(self, label: str, enabled: Optional[bool] = None):
        self.label = label
        self.on = ON if enabled is None else enabled
        self.note = ""
        self.result: Optional[str] = None
        self.ok = True
        self.t0 = time.time()
        self.i = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.out = sys.stdout           # the real terminal, even while quiet() captures prints

    def _line(self, frame: str, tail: str) -> str:
        return f"\r\033[K  {frame} {self.label:<{self.LABEL}} {tail}"

    def _draw(self):
        secs = int(time.time() - self.t0)
        note = self.note if self.note.startswith("\033") or not self.note else c(DIM, self.note)
        tail = (note + "  " if note else "") + c(DIM, f"{secs}s" if secs and not self.note.startswith("\033") else "")
        self.out.write(self._line(c(ORANGE, FRAMES[self.i % 4]), tail))
        self.out.flush()

    def _run(self):
        while not self._stop.wait(0.14):
            with self._lock:
                self.i += 1
                self._draw()

    def __enter__(self):
        if self.on:
            self._draw()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def status(self, note: str) -> None:
        """Change the small grey text after the label while the step runs."""
        self.note = note

    def done(self, result: str = "") -> None:
        self.result = result

    def fail(self, result: str) -> None:
        self.result, self.ok = result, False

    def pause(self):
        """Stop drawing (before printing other lines or asking a question)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self.on:
            self.out.write("\r\033[K")
            self.out.flush()

    def __exit__(self, exc_type, exc, tb):
        self.pause()
        if exc_type is not None and self.result is None:
            self.fail("stopped" if exc_type is KeyboardInterrupt else "failed")
        mark = c(GREEN if self.ok else RED, "■" if self.ok else "✗")
        text = self.result if self.result is not None else ""
        if self.on:
            self.out.write(f"  {mark} {self.label:<{self.LABEL}} {text}\n")
        else:
            self.out.write(f"  {'done' if self.ok else 'failed'}: {self.label}" + (f" - {text}" if text else "") + "\n")
        self.out.flush()
        return False


class quiet:
    """Capture what the code inside prints, to show it later (with --details, or if it fails)."""

    def __init__(self, show: bool = False):
        self.show = show
        self.text = ""

    def __enter__(self):
        import contextlib
        import io
        self._buf = io.StringIO()
        self._cm = contextlib.redirect_stdout(self._buf) if not self.show else contextlib.nullcontext()
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._cm.__exit__(exc_type, exc, tb)
        self.text = self._buf.getvalue()
        if exc_type is not None and self.text.strip():
            sys.stdout.write(self.text)          # something went wrong: show everything it said
        return False


def detail(text: str) -> None:
    """Dim lines under a step, wrapped to the terminal and indented under the step's result."""
    import textwrap
    indent = " " * (2 + 2 + Step.LABEL + 1)
    if not ON:
        print(f"      {text}")
        return
    for line in textwrap.wrap(text, width=max(30, width() - len(indent) - 1)):
        print(c(DIM, indent + line))


def ask(question: str, default: str = "y") -> bool:
    """A yes/no question styled like the steps; Enter takes the default."""
    hint = "[Y/n]" if default == "y" else "[y/N]"
    prompt = f"  {c(ORANGE, '?')} {question} {c(DIM, hint)} " if ON else f"  ? {question} {hint} "
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        ans = ""
    if not ans:
        return default == "y"
    return ans in ("y", "yes")


def enter_pressed() -> bool:
    """True once the user has pressed Enter (never blocks). Works in real terminals on every OS."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        if os.name == "nt":
            import msvcrt
            hit = False
            while msvcrt.kbhit():
                if msvcrt.getwch() in ("\r", "\n"):
                    hit = True
            return hit
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True
    except (OSError, ValueError):
        pass
    return False


def heading(text: str) -> None:
    print(c(BOLD, text) if ON else text)


# ------------------------------------------------------------------ download bar

def _fmt(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} MB"


def bar_line(name: str, done: int, total: int, speed: float) -> str:
    """'  ▕██████████░░░░░░▏ 62%  3.55 / 5.75 GB  19 MB/s  about 2 min left'"""
    frac = done / total if total else 0
    n = 22
    full = int(frac * n)
    bar = "█" * full + "░" * (n - full)
    left = (total - done) / speed if speed > 0 else 0
    eta = ("less than a minute left" if left < 60 else f"about {left / 60:.0f} min left") if speed > 0 and done < total else ""
    head = f"  {c(ORANGE, '▕' + bar + '▏')} {frac * 100:3.0f}%  {_fmt(done)} / {_fmt(total)}  {_fmt(speed)}/s"
    return head + (f"  {c(DIM, eta)}" if eta else "")
