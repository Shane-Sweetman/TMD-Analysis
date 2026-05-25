#!/usr/bin/env bash
set -euo pipefail

ROOT_FILE="${1:-output.root}"

if ! command -v root >/dev/null 2>&1; then
  echo "ROOT executable not found on PATH." >&2
  exit 1
fi

if [ ! -f "$ROOT_FILE" ]; then
  echo "ROOT file not found: $ROOT_FILE" >&2
  exit 1
fi

root -l -e "TFile::Open(\"$ROOT_FILE\"); new TBrowser();"
