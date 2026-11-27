#!/usr/bin/env bash
# Tlav installer for macOS and Linux.
#
#   curl -fsSL https://tlav.online/install.sh | bash
#
# Needs only Python 3.8+ (no git). Creates ~/.tlav/venv and links `tlav` into ~/.local/bin.
set -euo pipefail

SOURCE="${TLAV_SOURCE:-https://github.com/ArhaanDev24/Tlav/archive/refs/heads/main.zip}"
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
"$PY" -m venv "$TLAV_HOME/venv" || {
  echo "Could not create a Python environment. On Ubuntu/Debian run: sudo apt install python3-venv" >&2; exit 1; }

# From a checkout (./install.sh), install that folder; piped from the website, download the latest Tlav.
SRC_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
TARGET="$SOURCE"
if [ -n "$SRC_DIR" ] && [ -f "$SRC_DIR/pyproject.toml" ]; then TARGET="$SRC_DIR"; fi
echo "Getting Tlav..."
"$TLAV_HOME/venv/bin/python" -m pip install --quiet --disable-pip-version-check --upgrade --force-reinstall "$TARGET"
ln -sf "$TLAV_HOME/venv/bin/tlav" "$BIN_DIR/tlav"

echo
"$TLAV_HOME/venv/bin/tlav" welcome 2>/dev/null || {
  echo "Tlav installed. Next, just type:"
  echo "  tlav          # picks, installs and runs the best model for this computer, then opens the chat"
}
# Make sure `tlav` is found in new terminals.
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    case "$(basename "${SHELL:-sh}")" in
      zsh) RC="$HOME/.zshrc" ;;
      bash) if [ "$(uname)" = "Darwin" ]; then RC="$HOME/.bash_profile"; else RC="$HOME/.bashrc"; fi ;;
      *) RC="$HOME/.profile" ;;
    esac
    LINE="export PATH=\"$BIN_DIR:\$PATH\""
    if ! grep -qsF "$LINE" "$RC"; then printf '\n# Added by the Tlav installer\n%s\n' "$LINE" >> "$RC"; fi
    echo
    echo "Added $BIN_DIR to your PATH in $RC. Open a new terminal, or run this once now:"
    echo "  $LINE"
    ;;
esac
