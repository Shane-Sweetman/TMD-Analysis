#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$REPO_DIR"

echo "This compatibility script delegates to the root Makefile."
exec make run \
  FJ="${FJ:-/Users/shanesweetman/Downloads/fastjet}" \
  PYTHIA="${PYTHIA:-/Users/shanesweetman/Downloads/pythia/pythia8315}" \
  EVENTS="${EVENTS:-10000}" \
  SEED="${SEED:-12345}" \
  SRC="${SRC:-src/pythia/Pythia1.cc}" \
  BIN="${BIN:-TMD}" \
  OUT="${OUT:-output.root}" \
  PROGRESS="${PROGRESS:-progress.txt}"
