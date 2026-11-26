#!/usr/bin/env bash
# Tlav installer for macOS and Linux. Creates ~/.tlav/venv and links `tlav` into ~/.local/bin.
set -euo pipefail

REPO_URL="${TLAV_REPO:-https://github.com/ArhaanDev24/Tlavv}"
TLAV_HOME="${TLAV_HOME:-$HOME/.tlav}"
BIN_DIR="$HOME/.local/bin"

PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
  echo "Python 3.8+ is required. On macOS: https://www.python.org/downloads/macos/" >&2
  exit 1
fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' || {
  echo "Python 3.8+ is required (found $("$PY" --version))." >&2; exit 1; }

if [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "x86_64" ] && ! xcode-select -p >/dev/null 2>&1; then
  echo "Intel Mac detected: Apple's command line tools are needed to build the AI engine."
  echo "Starting their installer; re-run this script when it finishes."
  xcode-select --install || true
  exit 0
fi

mkdir -p "$TLAV_HOME" "$BIN_DIR"
"$PY" -m venv "$TLAV_HOME/venv"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$SRC_DIR" ] && [ -f "$SRC_DIR/pyproject.toml" ]; then
  "$TLAV_HOME/venv/bin/pip" install --quiet --upgrade "$SRC_DIR"
else
  "$TLAV_HOME/venv/bin/pip" install --quiet --upgrade "git+$REPO_URL"
fi
ln -sf "$TLAV_HOME/venv/bin/tlav" "$BIN_DIR/tlav"

echo
echo "Tlav installed. Next:"
echo "  tlav doctor   # see what your machine can do"
echo "  tlav setup    # pick, install and run the best model"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) echo; echo "Add $BIN_DIR to your PATH, e.g.: echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc";; esac
