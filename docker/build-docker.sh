#!/usr/bin/env bash
# Build and check all final filesystem layers, including the base image.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${1:?Usage: build-docker.sh IMAGE [docker build arguments]}"
shift
case "${IMAGE_LAYER_COUNT_CHECK:-1}" in
  0|1) ;;
  *) echo '[image-layers] FAIL: IMAGE_LAYER_COUNT_CHECK must be 0 or 1' >&2; exit 2 ;;
esac
docker build -t "$IMAGE" "$@" "$HERE"
docker image inspect "$IMAGE" | python3 "$HERE/layer_count.py"
