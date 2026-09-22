#!/usr/bin/env bash
# Build the Python environment for GLM-5.3-Flash on 4x CMP 170HX.
#
# Creates a uv venv (Python 3.12), clones the patched vLLM fork at the pinned
# commit, and installs it editable together with the runtime extras that the
# fork's own requirements do not pull in.
#
# Usage:  ./install.sh
#
# Env overrides:
#   VENV                venv directory                  (default ./venv)
#   VENV_CLEAR          1 = rebuild the venv from scratch instead of reusing it
#   VLLM_SRC            fork checkout directory         (default ./vllm-src)
#   VLLM_REPO           git remote                      (default the Morrowmake fork)
#   VLLM_BRANCH         branch to fetch                 (default ampere-glm53)
#   VLLM_COMMIT         commit to pin                   (default cf80da1839)
#   PYTHON_VERSION      interpreter uv provisions       (default 3.12)
#   CUDA_HOME           CUDA toolkit root               (default /usr/local/cuda-13.3)
#   BUILD_FROM_SOURCE   1 = compile the CUDA/C++ extensions instead of using
#                       the matching precompiled vLLM wheel (default 0, see below)
#   MAX_JOBS            parallel compile jobs when BUILD_FROM_SOURCE=1 (default 16)
#   TORCH_INDEX_URL     extra index for torch, if you need a non-PyPI build
#
# --- On BUILD_FROM_SOURCE -----------------------------------------------------
# Every patch on the ampere-glm53 branch is Python, Triton or TileLang: the diff
# against upstream touches no .cu, .cpp or CMakeLists file, so the compiled
# extensions (vllm/_C*, _moe_C*, _flashmla_C*, ...) are bit-identical to
# upstream's. Our production box therefore installs with VLLM_USE_PRECOMPILED=1,
# which downloads upstream's wheel for the same base commit and takes minutes
# instead of hours. Those wheels already carry sm_80 cubins, which is why four
# GA100 cards run them.
#
# Set BUILD_FROM_SOURCE=1 if you would rather compile everything yourself; it
# needs the full CUDA 13.3 toolkit, ~60 GB of scratch and 1-2 hours. It pins
# TORCH_CUDA_ARCH_LIST=8.0 so nvcc only emits sm_80, which is all these cards
# need and keeps the build as short as it can be.
#
# --- Prerequisites ------------------------------------------------------------
#   * Ubuntu 26.04 (or similar), NVIDIA driver 610.x with the CMP unlock in
#     place -- see hardware/README.md.
#   * uv            https://docs.astral.sh/uv/   (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   * git, git-lfs
#   * CUDA 13.3 toolkit at /usr/local/cuda-13.3. NVIDIA had no working
#     ubuntu2604 index when we built this, so we used the ubuntu2404 repo:
#
#       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
#       sudo dpkg -i cuda-keyring_1.1-1_all.deb
#       sudo apt-get update
#       sudo apt-get install -y cuda-toolkit-13-3 git-lfs
#
#     The toolkit is only strictly required for BUILD_FROM_SOURCE=1 and for
#     TileLang/Triton JIT at runtime; nvcc from 13.3 is what we run with.
# ------------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
VLLM_SRC="${VLLM_SRC:-$REPO_ROOT/vllm-src}"
VLLM_REPO="${VLLM_REPO:-https://github.com/Morrowmake/vllm.git}"
VLLM_BRANCH="${VLLM_BRANCH:-ampere-glm53}"
VLLM_COMMIT="${VLLM_COMMIT:-cf80da1839}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
BUILD_FROM_SOURCE="${BUILD_FROM_SOURCE:-0}"
MAX_JOBS="${MAX_JOBS:-16}"

log() { printf '\n==> %s\n' "$*"; }

command -v uv >/dev/null || { echo "install.sh: uv not found. https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
command -v git >/dev/null || { echo "install.sh: git not found." >&2; exit 1; }
if [ ! -d "$CUDA_HOME" ]; then
  echo "install.sh: no CUDA toolkit at $CUDA_HOME (set CUDA_HOME, or install cuda-toolkit-13-3)." >&2
  [ "$BUILD_FROM_SOURCE" = "1" ] && exit 1
  echo "install.sh: continuing anyway -- the precompiled path does not need nvcc to install," >&2
  echo "            but TileLang and Triton will want a toolkit at runtime." >&2
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

log "Cloning $VLLM_REPO ($VLLM_BRANCH @ $VLLM_COMMIT) into $VLLM_SRC"
if [ ! -d "$VLLM_SRC/.git" ]; then
  # Blobless, but NOT shallow: vLLM's precompiled-wheel resolution runs
  # `git merge-base` against upstream main, which needs real history.
  git clone --filter=blob:none --branch "$VLLM_BRANCH" "$VLLM_REPO" "$VLLM_SRC"
else
  git -C "$VLLM_SRC" fetch origin "$VLLM_BRANCH"
fi
# Stay on a named branch rather than a detached HEAD, for the same reason
# (`git branch --show-current` has to return something).
git -C "$VLLM_SRC" checkout -B "$VLLM_BRANCH" "$VLLM_COMMIT"
git -C "$VLLM_SRC" submodule update --init --recursive --depth 1
echo "    HEAD: $(git -C "$VLLM_SRC" log --oneline -1)"

if [ -x "$VENV/bin/python" ] && [ "${VENV_CLEAR:-0}" != "1" ]; then
  log "Reusing venv at $VENV (set VENV_CLEAR=1 to rebuild it from scratch)"
else
  log "Creating venv at $VENV (Python $PYTHON_VERSION)"
  uv venv --clear --python "$PYTHON_VERSION" "$VENV"
fi
export VIRTUAL_ENV="$VENV"

PIP_ARGS=(--extra-index-url https://flashinfer.ai/whl/)
[ -n "${TORCH_INDEX_URL:-}" ] && PIP_ARGS+=(--extra-index-url "$TORCH_INDEX_URL")

# torch 2.13.0 from PyPI is the cu130 build on linux-x86_64 (torch.version.cuda
# == '13.0', __version__ == '2.13.0+cu130'). That is exactly what our production
# environment runs, so no pytorch.org index is needed. Set TORCH_INDEX_URL to
# https://download.pytorch.org/whl/cu130 if your platform resolves differently.
log "Installing torch 2.13.0 (cu130)"
uv pip install "${PIP_ARGS[@]}" \
  "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"

log "Installing the patched vLLM (editable)"
if [ "$BUILD_FROM_SOURCE" = "1" ]; then
  echo "    compiling CUDA extensions for sm_80 with MAX_JOBS=$MAX_JOBS"
  unset VLLM_USE_PRECOMPILED || true
  env -u VLLM_USE_PRECOMPILED \
    TORCH_CUDA_ARCH_LIST="8.0" \
    CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" \
    MAX_JOBS="$MAX_JOBS" \
    NVCC_THREADS=2 \
    VIRTUAL_ENV="$VENV" CUDA_HOME="$CUDA_HOME" \
    uv pip install "${PIP_ARGS[@]}" --no-build-isolation -e "$VLLM_SRC"
else
  echo "    using upstream's precompiled extensions (see the header)"
  VLLM_USE_PRECOMPILED=1 VIRTUAL_ENV="$VENV" CUDA_HOME="$CUDA_HOME" \
    uv pip install "${PIP_ARGS[@]}" -e "$VLLM_SRC"
fi

# Extras the fork's requirements do not pull in on their own but that the
# sm_80 paths import at runtime, pinned to the versions production runs.
log "Installing runtime extras"
uv pip install "${PIP_ARGS[@]}" \
  "flashinfer-python==0.6.18.post1" \
  "flashinfer-cubin==0.6.18.post1" \
  "tilelang==0.1.12" \
  "apache-tvm-ffi==0.1.11" \
  "ninja" \
  "huggingface_hub[hf_xet]>=1.0"

log "Verifying"
"$VENV/bin/python" - <<'PY'
import importlib, sys
import torch
print(f"    python      {sys.version.split()[0]}")
print(f"    torch       {torch.__version__} (cuda {torch.version.cuda})")
import vllm
print(f"    vllm        {vllm.__version__}")
# The compiled extensions must load. On this build they are the stable-ABI
# modules; _custom_ops is the import that actually binds the torch ops.
import vllm._C_stable_libtorch      # noqa: F401
import vllm._moe_C_stable_libtorch  # noqa: F401
import vllm._custom_ops             # noqa: F401
print("    vllm C ext  ok (_C_stable_libtorch, _moe_C_stable_libtorch, _custom_ops)")
for mod in ("triton", "tilelang", "flashinfer"):
    m = importlib.import_module(mod)
    print(f"    {mod:<11} {getattr(m, '__version__', 'ok')}")
from vllm.v1.attention.backends.mla import triton_mla_sparse  # noqa: F401
from vllm.v1.attention.ops import triton_mqa_logits, triton_e4m3  # noqa: F401
from vllm.v1.worker.gpu import prologue_fuse  # noqa: F401
from vllm.distributed.device_communicators import host_shm_all_reduce  # noqa: F401
print("    sm_80 patch modules ok")
PY

cat <<EOF

Done.
  venv   $VENV
  vllm   $VLLM_SRC  ($(git -C "$VLLM_SRC" rev-parse --short HEAD))

Next:  ./download.sh   then   ./serve.sh
EOF
