#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FJ="${FJ:-/Users/shanesweetman/Downloads/fastjet}"
PYTHIA="${PYTHIA:-/Users/shanesweetman/Downloads/pythia/pythia8315}"
EVENTS="${EVENTS:-10000}"
SEED="${SEED:-12345}"
SRC="${SRC:-src/pythia/Pythia1.cc}"
BIN="${BIN:-TMD}"
OUT="${OUT:-output.root}"
PROGRESS="${PROGRESS:-progress.txt}"

cd "$REPO_DIR"

if ! command -v root-config >/dev/null 2>&1; then
  echo "root-config was not found on PATH. Please load ROOT first." >&2
  exit 1
fi

if [ ! -x "$FJ/bin/fastjet-config" ]; then
  echo "FastJet config not found or not executable: $FJ/bin/fastjet-config" >&2
  exit 1
fi

if [ ! -d "$PYTHIA/include" ] || [ ! -d "$PYTHIA/lib" ]; then
  echo "PYTHIA path does not look valid: $PYTHIA" >&2
  exit 1
fi

rm -f "$BIN" "$OUT" "$PROGRESS"

# Avoid inherited macOS dynamic-library settings interfering with compilation.
unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES

ROOT_CFLAGS=($(root-config --cflags))
ROOT_LIBS=($(root-config --libs))
FASTJET_CXXFLAGS=($("$FJ/bin/fastjet-config" --cxxflags))
FASTJET_LIBS=($("$FJ/bin/fastjet-config" --libs))

g++ -O2 -std=c++17 \
  -I"$PYTHIA/include" \
  "${ROOT_CFLAGS[@]}" \
  "${FASTJET_CXXFLAGS[@]}" \
  "$SRC" -o "$BIN" \
  -L"$PYTHIA/lib" -lpythia8 \
  "${ROOT_LIBS[@]}" \
  "${FASTJET_LIBS[@]}" \
  -Wl,-rpath,"$PYTHIA/lib" -Wl,-rpath,"$FJ/lib"

# The default is a small reproducibility check. Thesis-scale production used
# much larger event samples, up to 100M events, stored outside GitHub.
env DYLD_LIBRARY_PATH="$PYTHIA/lib:$FJ/lib" \
    DYLD_FALLBACK_LIBRARY_PATH="$PYTHIA/lib:$FJ/lib" \
    "./$BIN" "$EVENTS" "$SEED" "$OUT" "$PROGRESS"

echo "Wrote $OUT"
