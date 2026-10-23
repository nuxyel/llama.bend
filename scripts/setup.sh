#!/usr/bin/env bash
# One-shot, idempotent setup for the llama.bend checks. No sudo.
#
#   scripts/setup.sh      # Bend 2.0.38, Lean 4.34.0, the Python venv of reference/
#
# Pins: Bend 2.0.38 (checked by SHA256) and Lean v4.34.0 (the kernel behind `bend --verdict`). The same steps as
# bend-ml's scripts/setup.sh (MIT). The model is downloaded afterwards by `make prep`.
set -euo pipefail

BEND_VERSION=2.0.38
LEAN_VERSION=4.34.0

cd "$(dirname "$0")/.."
export BEND_NO_TELEMETRY=1
export PATH="$HOME/.bend/bin:$HOME/.elan/bin:$PATH"

say() { printf '\n== %s\n' "$*"; }
have_bend() { command -v bend >/dev/null 2>&1 && [ "$(bend version 2>/dev/null | awk '{print $NF}')" = "$BEND_VERSION" ]; }
have_lean() { command -v lean >/dev/null 2>&1 && lean --version 2>/dev/null | grep -q "version $LEAN_VERSION"; }

# --- 1. Bend, pinned and verified -------------------------------------------------------
say "Bend $BEND_VERSION"
if have_bend; then
  echo "already installed"
else
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  OS=linux; ARCH=x64;   SHA=f19d088b079c9e1a8834ecd153b7d58c11a5f8f9ab1bdb78dbc96a6bf49aaa81 ;;
    Linux-aarch64) OS=linux; ARCH=arm64; SHA=1f7c6e74a6ef7412cf08eee0e4d45fe5960f89d66fdaca2466f6a4856f0d400a ;;
    *) echo "unsupported platform $(uname -s)-$(uname -m): Bend runs on Linux (use WSL on Windows)"; exit 1 ;;
  esac
  HOME_BEND=${BEND_HOME:-$HOME/.bend}
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  NAME="bend-$BEND_VERSION-$OS-$ARCH.tar.gz"
  curl --proto '=https' --tlsv1.2 -fsSL -o "$TMP/$NAME" "https://github.com/bendlang/bend/releases/download/v$BEND_VERSION/$NAME"
  echo "$SHA  $TMP/$NAME" | sha256sum -c - >/dev/null || { echo "SHA256 mismatch for $NAME"; exit 1; }
  tar -xzf "$TMP/$NAME" -C "$TMP" 2>/dev/null
  mkdir -p "$HOME_BEND/bin"
  rm -rf "$HOME_BEND/bend2" "$HOME_BEND/guide"
  mv "$TMP/bend/bend2" "$TMP/bend/guide" "$HOME_BEND/"
  mv -f "$TMP/bend/bin/bend" "$HOME_BEND/bin/bend"
fi
have_bend || { echo "ERROR: bend $BEND_VERSION not found after install"; exit 1; }
bend version

# --- 2. Lean (kernel for --verdict) -----------------------------------------------------
say "Lean $LEAN_VERSION"
if have_lean; then
  echo "already installed"
else
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --no-modify-path --default-toolchain "leanprover/lean4:v$LEAN_VERSION"
fi
have_lean || { echo "ERROR: lean $LEAN_VERSION not found after install"; exit 1; }
lean --version

# --- 3. Python environment --------------------------------------------------------------
say "Python environment"
[ -d reference/.venv ] || python3 -m venv reference/.venv
reference/.venv/bin/pip install -q --upgrade pip
reference/.venv/bin/pip install -q -r reference/requirements.txt
reference/.venv/bin/python -c "import torch, transformers, tokenizers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| tokenizers', tokenizers.__version__)"

say "setup finished: run 'make prep' (downloads SmolLM2-135M), then 'make check'"
