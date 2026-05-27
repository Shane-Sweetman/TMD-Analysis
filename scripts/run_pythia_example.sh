#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$REPO_DIR"

echo "This compatibility script delegates to the root Makefile."
exec make run \
  EVENTS="${EVENTS:-10000}" \
  SEED="${SEED:-12345}" \
  OUT="${OUT:-output.root}" \
  PROGRESS="${PROGRESS:-progress.txt}"
