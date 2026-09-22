#!/usr/bin/env bash
# Fetch the two checkpoints into ./models/.
#
#   canada-quant/GLM-5.3-Flash-W4A16-MTP    ~178 GB, 21 files   (target)
#   incoai/GLM-5.3-Flash-DFlash2            ~2.2 GB,  5 files   (drafter)
#
# Budget ~180 GB of free disk and, on a normal home line, a couple of hours
# for the target. The drafter takes about a minute.
#
# Usage:  ./download.sh [target|drafter]      (no argument = both)
#
# Env overrides:
#   VENV          venv from install.sh   (default ./venv, for its `hf` CLI)
#   MODELS_DIR    destination            (default ./models)
#   TARGET_REPO   default canada-quant/GLM-5.3-Flash-W4A16-MTP
#   DRAFTER_REPO  default incoai/GLM-5.3-Flash-DFlash2
#   HF_TOKEN      optional; unauthenticated downloads are rate limited
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
TARGET_REPO="${TARGET_REPO:-canada-quant/GLM-5.3-Flash-W4A16-MTP}"
DRAFTER_REPO="${DRAFTER_REPO:-incoai/GLM-5.3-Flash-DFlash2}"
WHAT="${1:-both}"

HF="$VENV/bin/hf"
[ -x "$HF" ] || HF="$(command -v hf || true)"
[ -n "$HF" ] || { echo "download.sh: no 'hf' CLI. Run ./install.sh first, or pip install 'huggingface_hub[hf_xet]'." >&2; exit 1; }

mkdir -p "$MODELS_DIR"

fetch() {
  local repo="$1" dest="$MODELS_DIR/${1##*/}"
  echo
  echo "==> $repo  ->  $dest"
  "$HF" download "$repo" --local-dir "$dest"
}

case "$WHAT" in
  target)  fetch "$TARGET_REPO" ;;
  drafter) fetch "$DRAFTER_REPO" ;;
  both)    fetch "$TARGET_REPO"; fetch "$DRAFTER_REPO" ;;
  *) echo "download.sh: expected 'target', 'drafter' or nothing." >&2; exit 2 ;;
esac

echo
du -sh "$MODELS_DIR"/* 2>/dev/null || true
echo
echo "Done. Next: ./serve.sh"
