"""Small terminal animations for cluster connections.

A link line shows this PC and another computer as two Tlav squares, with a
packet gliding between them while they talk:

    ■ this PC  ·  ·  ·  ▪ ■ ·  ·  ·  ■ study-imac   connecting…  2s

When the work is done the line settles into a solid connection:

    ■ this PC ━━━━━━━━━━━━━━━━━━━━━ ■ study-imac   connected · 0.4 ms, wired

Several lines can animate at once (one per worker while the model's layers
are sent). On anything that isn't a real terminal, plain lines are printed.
"""

import math
import os
import shutil
import sys
import threading
import time
from typing import List, Optional

_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
ORANGE, GREEN, DIM, BOLD, RESET = "\033[38;5;209m", "\033[38;5;108m", "\033[2m", "\033[1m", "\033[0m"
TRACK = 22          # cells between the two squares
FPS = 14


def _c(code: str, s: str, on: bool) -> str:
    return f"{code}{s}{RESET}" if on else s


class _Line:
    def __init__(self, left: str, right: str, label: str, offset: float):
        self.left, self.right, self.label = left, right, label
        self.offset = offset            # so several links don't pulse in lockstep
        self.final: Optional[str] = None      # the settled note (plain text)
        self.ok = True


class Links:
    """One or more animated links. Use as a context manager, or start()/finish()."""

    def __init__(self, pairs: List[tuple], label: str = "connecting", enabled: Optional[bool] = None,
                 direction: int = 1):
        self.on = _TTY if enabled is None else enabled
        self.lines = [_Line(a, b, label, i * 0.37) for i, (a, b) in enumerate(pairs)]
        self.direction = direction      # 1: this PC -> worker (sending), -1: replies coming back
        self.t0 = time.time()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._drawn = 0
        self.width = shutil.get_terminal_size((80, 24)).columns
        lw = max(len(ln.left) for ln in self.lines) if self.lines else 7
        rw = max(len(ln.right) for ln in self.lines) if self.lines else 7
        self.lw, self.rw = lw, rw
        # keep every line on one row (a wrapped line would break the redraw): ~34 columns of text
        self.track = max(6, min(TRACK, self.width - lw - rw - 46))

    # -- drawing
    def _track(self, ln: _Line, t: float) -> str:
        if ln.final is not None:
            bar = "━" * self.track if ln.ok else "╌" * self.track
            return _c(GREEN if ln.ok else DIM, bar, self.on)
        # a packet glides across with a short fading trail; eased so it slows near each end
        phase = ((t / 1.6) + ln.offset) % 1.0
        ease = 0.5 - 0.5 * math.cos(phase * math.pi)
        n = self.track
        pos = ease * (n - 1)
        if self.direction < 0:
            pos = (n - 1) - pos
        cells = []
        for i in range(n):
            d = (pos - i) * self.direction
            if abs(i - pos) < 0.5:
                cells.append(_c(ORANGE, "■", self.on))
            elif 0 < d < 2.5:
                cells.append(_c(ORANGE + DIM, "▪", self.on))
            elif i % 3 == 1:
                cells.append(_c(DIM, "·", self.on))
            else:
                cells.append(" ")
        return "".join(cells)

    def _render(self, ln: _Line, t: float) -> str:
        sq_l = _c(ORANGE, "■", self.on)
        sq_r = _c(ORANGE if ln.final is None else (GREEN if ln.ok else DIM), "■", self.on)
        if ln.final is not None:
            tail, color = ln.final, (GREEN if ln.ok else DIM)
        else:
            secs = int(time.time() - self.t0)
            tail, color = f"{ln.label}…" + (f"  {secs}s" if secs else ""), DIM
        # never wrap: a wrapped line would throw the redraw off by a row
        room = self.width - 1 - (self.lw + self.rw + self.track + 11)
        if len(tail) > room:
            tail = tail[:max(0, room - 1)] + "…"
        return (f"  {sq_l} {ln.left:<{self.lw}} {self._track(ln, t)} {sq_r} {_c(BOLD, f'{ln.right:<{self.rw}}', self.on)}"
                f"   {_c(color, tail, self.on)}")

    def _draw(self):
        t = time.time() - self.t0
        out = []
        if self._drawn:
            out.append(f"\033[{self._drawn}A")
        for ln in self.lines:
            out.append("\r\033[K" + self._render(ln, t) + "\n")
        out.append("\033[J")          # wipe anything left below (e.g. after the terminal echoed Enter)
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._drawn = len(self.lines)

    def _run(self):
        while not self._stop.wait(1 / FPS):
            with self.lock:
                self._draw()

    # -- control
    def start(self) -> "Links":
        self.t0 = time.time()
        if self.on and self.lines:
            with self.lock:
                self._draw()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def add(self, left: str, right: str, label: str = "connecting") -> int:
        """Add a link while the others keep animating (a worker that just appeared). Returns its index."""
        with self.lock:
            self.lines.append(_Line(left, right, label, len(self.lines) * 0.37))
            self.lw = max(self.lw, len(left))
            self.rw = max(self.rw, len(right))
            self.track = max(6, min(TRACK, self.width - self.lw - self.rw - 46))
            if self.on and self._thread is None and not self._stop.is_set():
                self._draw()
            return len(self.lines) - 1

    def cursor_moved(self, rows: int = 1) -> None:
        """The terminal moved the cursor down (it echoed the Enter key); keep the redraw aligned."""
        with self.lock:
            if self._drawn:
                self._drawn += rows

    def done(self, index: int, note: str, ok: bool = True) -> None:
        """Settle one link: solid green when ok, dashed grey when not."""
        ln = self.lines[index]
        with self.lock:
            ln.final = note
            ln.ok = ok

    def finish(self, note: Optional[str] = None, ok: bool = True) -> None:
        """Stop animating; any link not settled yet gets `note`."""
        for i, ln in enumerate(self.lines):
            if ln.final is None:
                self.done(i, note or ("connected" if ok else "no answer"), ok)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self.on:
            with self.lock:
                self._draw()
        else:
            for ln in self.lines:
                print(f"  {ln.left} -> {ln.right}: {ln.final}")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.finish("stopped" if exc_type else None, ok=exc_type is None)
        return False
