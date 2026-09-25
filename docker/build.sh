#!/usr/bin/env bash
# Assemble the image the Dockerfile in this directory describes, without a
# container runtime. This is how the release image was built. The /opt tree
# (Python, venv, engine source) is built on the host under a staging path,
# rewritten to /opt, packed as one layer and appended to the pinned CUDA base
# with crane, in a throwaway registry on 127.0.0.1. The result is saved as an
# OCI layout in out/, from which `crane push` publishes the same manifest
# (same digest).
#
# Needs: x86_64 Linux, uv, git, curl, python3, GNU tar, crane
# (github.com/google/go-containerregistry), and a CUDA 13.x toolkit at
# CUDA_HOME (default /usr/local/cuda-13.3).
# Nothing here touches a GPU: the import check runs with CUDA_VISIBLE_DEVICES
# empty.
#
#   ./build.sh                  full build
#   STEP=assemble ./build.sh    re-run only the layer + image assembly
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRANE="${CRANE:-crane}"
UV="${UV:-uv}"
BUILD_CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
STAGE="${STAGE:-/var/tmp/cmp170hx-image/stage}"
OUT="$HERE/out"
BASE="nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
VLLM_REPO="https://github.com/Morrowmake/vllm-cmp170hx.git"
VLLM_BRANCH="ampere-glm53"
VLLM_COMMIT="0ed7d3e7f3f855646a139701598a9b40d5745688"
PYTHON_VERSION="3.12"
REG="127.0.0.1:5055"
NAME="vllm-cmp170hx"
TAG="1.3.0-0ed7d3e7f3"
PIP=(--extra-index-url https://flashinfer.ai/whl/)
export UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1 UV_NO_CONFIG=1
# git records an identity in reflogs; keep the image free of the builder's
export GIT_AUTHOR_NAME=Morrowmake GIT_AUTHOR_EMAIL=noreply@github.com
export GIT_COMMITTER_NAME=Morrowmake GIT_COMMITTER_EMAIL=noreply@github.com
export GIT_CONFIG_GLOBAL=/dev/null
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

build_tree() {
    rm -rf "$STAGE"; mkdir -p "$STAGE/opt"
    log "python $PYTHON_VERSION"
    UV_PYTHON_INSTALL_DIR="$STAGE/python-dl" "$UV" python install "$PYTHON_VERSION"
    mv "$STAGE"/python-dl/cpython-${PYTHON_VERSION}.*-linux-x86_64-gnu "$STAGE/opt/python"
    rm -rf "$STAGE/python-dl"
    "$UV" venv --relocatable --python "$STAGE/opt/python/bin/python$PYTHON_VERSION" "$STAGE/opt/venv"

    log "engine source @ $VLLM_COMMIT"
    git clone --filter=blob:none --branch "$VLLM_BRANCH" "$VLLM_REPO" "$STAGE/opt/vllm-src"
    git -C "$STAGE/opt/vllm-src" checkout -B "$VLLM_BRANCH" "$VLLM_COMMIT"
    git -C "$STAGE/opt/vllm-src" submodule update --init --recursive --depth 1

    export VIRTUAL_ENV="$STAGE/opt/venv"
    log "torch"
    "$UV" pip install "${PIP[@]}" "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"
    log "vllm (upstream precompiled extensions)"
    VLLM_USE_PRECOMPILED=1 CUDA_HOME="$BUILD_CUDA_HOME" "$UV" pip install "${PIP[@]}" -e "$STAGE/opt/vllm-src"
    log "runtime extras"
    "$UV" pip install "${PIP[@]}" \
        "flashinfer-python==0.6.18.post1" "flashinfer-cubin==0.6.18.post1" \
        "tilelang==0.1.12" "apache-tvm-ffi==0.1.11" "ninja" "huggingface_hub[hf_xet]>=1.0"
    "$UV" pip freeze > "$STAGE/opt/venv/requirements.lock.txt"

    log "verify imports (no GPU)"
    CUDA_VISIBLE_DEVICES= "$STAGE/opt/venv/bin/python" - <<'PY' | tee "$OUT/verify.txt"
import importlib, sys, torch
print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__} (cuda {torch.version.cuda})")
import vllm
print(f"vllm        {vllm.__version__}")
import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops  # noqa: F401
for mod in ("triton", "tilelang", "flashinfer"):
    m = importlib.import_module(mod)
    print(f"{mod:<11} {getattr(m, '__version__', 'ok')}")
from vllm.v1.attention.backends.mla import triton_mla_sparse            # noqa: F401
from vllm.v1.attention.ops import triton_mqa_logits, triton_e4m3        # noqa: F401
from vllm.v1.worker.gpu import prologue_fuse                            # noqa: F401
from vllm.distributed.device_communicators import host_shm_all_reduce   # noqa: F401
print("sm_80 patch modules ok")
PY
    unset VIRTUAL_ENV
}

relocate() {
    log "relocate $STAGE/opt -> /opt"
    find "$STAGE/opt" -name __pycache__ -type d -prune -exec rm -rf {} +
    rm -rf "$STAGE/opt/vllm-src/.git/logs"
    find "$STAGE/opt/vllm-src" -path '*/.git/modules/*/logs' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    # text files that name the staging path (uv writes the interpreter's
    # install location into sysconfig, so that one maps to /opt/python)
    grep -rlI --null -F "$STAGE/python-dl/" "$STAGE/opt" \
        | xargs -0 -r sed -i -E "s#$STAGE/python-dl/cpython-[^/\"' ]+#/opt/python#g" || true
    grep -rlI --null -F "$STAGE" "$STAGE/opt" | xargs -0 -r sed -i "s#$STAGE/opt#/opt#g" || true
    # symlinks that point into the staging path
    find "$STAGE/opt" -type l -print0 | while IFS= read -r -d '' l; do
        t="$(readlink "$l")"
        case "$t" in "$STAGE"/*) ln -sfn "${t#"$STAGE"}" "$l" ;; esac
    done
    # nothing may still name the staging path, a home directory or this host
    local bad=0
    if grep -rl -F -a "$STAGE" "$STAGE/opt" | head -5 | grep .; then bad=1; fi
    if grep -rl -a -F "$HOME/" "$STAGE/opt" | head -5 | grep .; then bad=1; fi
    if find "$STAGE/opt" -type l -lname '/home/*' | grep .; then bad=1; fi
    # the host name, reported for review (it can occur inside unrelated binaries)
    grep -rlI -w -F "$(hostname -s)" "$STAGE/opt" --exclude-dir=.git | head -20 > "$OUT/hostname-hits.txt" || true
    [ "$bad" = 0 ] || { echo "relocate: leftover host paths (above)" >&2; exit 1; }
}

layer() {
    log "layer"
    tar --sort=name --owner=0 --group=0 --numeric-owner \
        --mtime='2026-09-25 00:00:00Z' --format=posix \
        --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime \
        -C "$STAGE" -cf "$OUT/layer.tar" opt
    ls -l "$OUT/layer.tar"
}

assemble() {
    log "assemble in a registry at $REG"
    mkdir -p "$OUT/registry-data"
    "$CRANE" registry serve --address "$REG" --disk "$OUT/registry-data" >"$OUT/registry.log" 2>&1 &
    REG_PID=$!
    trap 'kill "${REG_PID:-}" 2>/dev/null || true' EXIT
    for _ in $(seq 50); do curl -fsS "http://$REG/v2/" >/dev/null 2>&1 && break; sleep 0.2; done
    "$CRANE" copy --platform linux/amd64 "$BASE" "$REG/nvidia-cuda:13.3.1-devel-ubuntu24.04"
    "$CRANE" append -b "$REG/nvidia-cuda:13.3.1-devel-ubuntu24.04" -f "$OUT/layer.tar" -t "$REG/$NAME:layered"
    "$CRANE" mutate "$REG/$NAME:layered" -t "$REG/$NAME:$TAG" \
        --entrypoint /opt/venv/bin/vllm \
        --exposed-ports 8000/tcp \
        -e PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        -e VIRTUAL_ENV=/opt/venv \
        -e CUDA_HOME=/usr/local/cuda \
        -e 'NVIDIA_REQUIRE_CUDA=cuda>=13.0' \
        -l org.opencontainers.image.title=vllm-cmp170hx \
        -l "org.opencontainers.image.description=vLLM fork for GLM-5.3-Flash W4A16 on 4x NVIDIA CMP 170HX (sm_80) as pinned by glm53-flash-cmp170hx-recipe 1.3.0; weights not included" \
        -l org.opencontainers.image.source=https://github.com/Morrowmake/vllm-cmp170hx \
        -l org.opencontainers.image.revision=$VLLM_COMMIT \
        -l org.opencontainers.image.url=https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/tree/v1.3.0 \
        -l org.opencontainers.image.version=$TAG \
        -l org.opencontainers.image.licenses=Apache-2.0 \
        -l org.opencontainers.image.base.name=docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04 \
        -l org.opencontainers.image.base.digest=sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87
    local digest; digest="$("$CRANE" digest "$REG/$NAME:$TAG")"
    rm -rf "$OUT/oci"
    "$CRANE" pull --format oci "$REG/$NAME@$digest" "$OUT/oci"
    "$CRANE" manifest "$REG/$NAME@$digest" > "$OUT/manifest.json"
    "$CRANE" config "$REG/$NAME@$digest" > "$OUT/config.json"
    # the saved layout must push back to the same digest
    "$CRANE" push "$OUT/oci" "$REG/$NAME-roundtrip:check" >/dev/null
    local rt; rt="$("$CRANE" digest "$REG/$NAME-roundtrip:check")"
    [ "$rt" = "$digest" ] || { echo "round trip digest $rt != $digest" >&2; exit 1; }
    python3 - "$OUT/manifest.json" <<'PY' | tee "$OUT/size.txt"
import json, sys
m = json.load(open(sys.argv[1]))
total = sum(l["size"] for l in m["layers"]) + m["config"]["size"]
print(f"compressed size {total} bytes ({total/1e9:.2f} GB), {len(m['layers'])} layers, top layer {m['layers'][-1]['size']/1e9:.2f} GB")
PY
    echo "$digest" > "$OUT/digest.txt"
    log "image ghcr.io/morrowmake/$NAME:$TAG  digest $digest (not pushed)"
}

case "${STEP:-all}" in
    all) build_tree; relocate; layer; assemble ;;
    relocate) relocate; layer; assemble ;;
    assemble) layer; assemble ;;
    *) echo "unknown STEP" >&2; exit 2 ;;
esac
