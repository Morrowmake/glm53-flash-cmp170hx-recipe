#!/usr/bin/env bash
# Launch GLM-5.3-Flash W4A16 on 4x CMP 170HX from the patched vLLM checkout.
#
# Usage: serve.sh [dflash|mtp|none|--help]
#   (no argument)  same as "dflash" -- the default.
#   dflash         DFlash2 drafter, num_speculative_tokens = SPEC_N (default 3).
#   mtp            MTP drafter,     num_speculative_tokens = SPEC_N (default 3).
#   none           no speculative decoding.
# A --speculative-config inside EXTRA_ARGS wins over all of the above and no
# second one is added, so callers can pass their own JSON.
# Set DRY=1 to print the command that would run instead of running it.
#
# Paths (all relative to this repo unless overridden):
#   VENV            venv from install.sh            (default ./venv)
#   MODEL           target model directory          (default ./models/GLM-5.3-Flash-W4A16-MTP)
#   DFLASH_MODEL    drafter directory               (default ./models/GLM-5.3-Flash-DFlash2)
#   CUDA_HOME       CUDA toolkit root               (default /usr/local/cuda-13.3)
# Env overrides: MAX_LEN, MAX_SEQS, MAX_BATCHED, PREFILL_CAP (upstream long-prefill chunk cap,
#   UNCONDITIONAL; default 0 = off -- LEAVE IT 0, see below), GPU_UTIL, PORT, SPEC_N,
#   SERVED_NAME, REASONING_PARSER, TOOL_PARSER, MM_CAP, PP, TP,
#   VLLM_PP_LAYER_PARTITION, EXTRA_ARGS.
#
# ===== sm_80 feature flags ==================================================
# Seven features, validated individually and then together.
# All are OFF by default IN THE CODE; the block below is the only thing that
# turns them on, so each one is a one-variable kill switch -- set it to 0 in
# the environment and restart, no rebuild and no revert:
#
#   VLLM_GLM5_PREFILL_OVERLAP=0   TP prefill comm/compute overlap  [tp-prefill-overlap]
#   VLLM_GLM5_PREFILL_KERNELS=0   sm_80 prefill kernels            [prefill-kernels]
#   VLLM_GLM5_PROLOGUE_FUSE=0     fused eager decode prologue      [decode-prologue]
#   VLLM_GLM5_LOCAL_LOGITS=0      batch-sharded logits + sampling  [decode-prologue]
#   VLLM_GLM5_DECODE_KERNELS=0    sm_80 decode kernels             [decode-small-kernels]
#   VLLM_GLM5_THIN_GEMM=0         sm_80 thin-M BF16 GEMM           [thin-gemm]
#   FAIR_PREFILL=0                decode-aware prefill chunking    [fair-prefill]
#   VLLM_GLM5_HOST_ALLREDUCE=0    host-staged no-P2P all-reduce    [pcie-allreduce]
#
# Measured together vs the pre-merge default: prefill +13.4%, TTFT@23K -14.9%,
# ms/step c1 -1.22 (paired, drift 0.58), decode retention during someone
# else's prefill 7% -> 18%, KV unchanged.
# Exactness moved slightly outside the noise floor (0.4119 vs floors 0.2007 /
# 0.2984) and gsm8k went 1.000 -> 0.980 at n=50; decode-small-kernels owns
# that, so VLLM_GLM5_DECODE_KERNELS=0 is the first switch to try if output
# quality is ever in question.
#
# PREFILL_CAP MUST STAY 0. It is upstream's UNCONDITIONAL chunk cap: it cost
# -15.3% prefill / +16.5% TTFT@23K in production, and as a side effect it
# drops chunks below the two prefill gates and silently disables the overlap
# and the prefill kernels. The decode-aware cap (FAIR_PREFILL) is the one to
# use -- it only applies while something is actually decoding.
# ============================================================================
set -euo pipefail
MODE=${1:-dflash}
case "$MODE" in
  -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
  dflash|mtp|none) ;;
  *) echo "serve.sh: unknown argument '$MODE' (expected dflash, mtp, none or --help)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
MODEL="${MODEL:-$REPO_ROOT/models/GLM-5.3-Flash-W4A16-MTP}"
DFLASH_MODEL="${DFLASH_MODEL:-$REPO_ROOT/models/GLM-5.3-Flash-DFlash2}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"

# Avoid caching-allocator fragmentation during MoE weight loading (middle stages filled 64 GB on first run).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Layout: default TENSOR-PARALLEL 4 (c1 135-145 tok/s vs PP4 80-85; PP4 keeps faster long-prompt TTFT,
# 5.0 s vs 11.8 s at 23K, and more KV). Switch with PP=4 TP=1.
# TP=4 assumes wide links between the cards: it moves ~9.4 MB per layer during
# prefill and ~100 small collectives per decode step. On narrow links (x4) use
# PP=4 TP=1, which only passes activations between stages. The PP path works
# but is untuned here -- see the README section on link width.
# Under PP the balanced layer split is 3 dense + 42 MoE (~3.8 GiB each). MTP keeps a 13.8 GiB BF16
# draft layer on the last stage, so the balanced split differs by mode. DFlash's drafter KV rides the
# MLA tensors, so it uses the non-MTP split.
PP=${PP:-1}; TP=${TP:-4}   # PP*TP must be 4
if [ "$PP" = "4" ]; then
  if [ "$MODE" = "mtp" ]; then DEFAULT_PART=14,12,12,7; else DEFAULT_PART=13,11,11,10; fi
  export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-$DEFAULT_PART}"
elif [ -n "${VLLM_PP_LAYER_PARTITION:-}" ]; then export VLLM_PP_LAYER_PARTITION; else unset VLLM_PP_LAYER_PARTITION; fi
# Replicated input-embedding table under TP (VLLM_GLM5_REPLICATED_EMBED): skips the 2 hidden-size
# all-reduces per prefill chunk / decode step (target + MTP drafter) for +0.74 GiB per rank
# (-6% KV tokens). Off by default -- the KV is worth more here. Sharded table under PP-only.
if [ "$TP" -gt 1 ]; then export VLLM_GLM5_REPLICATED_EMBED=${VLLM_GLM5_REPLICATED_EMBED:-0}; fi

# --- sm_80 feature flags (see the header for the kill switches) -------------
# [tp-prefill-overlap] split each mHC layer's post-attention part into S token
# micro-batches and fly the two per-layer all-reduces on a side stream. TP>1
# only; S=4 measured worse than S=2. _MIN_TOKENS stays at its 512 default and
# _CROSS_LAYER stays off (in tree, never validated on a GPU).
if [ "$TP" -gt 1 ]; then export VLLM_GLM5_PREFILL_OVERLAP=${VLLM_GLM5_PREFILL_OVERLAP:-1}; export VLLM_GLM5_PREFILL_OVERLAP_SPLITS=${VLLM_GLM5_PREFILL_OVERLAP_SPLITS:-2}; fi
# [prefill-kernels] sm_80 mHC pre-norm projection + sparse-MLA DSA attention.
export VLLM_GLM5_PREFILL_KERNELS=${VLLM_GLM5_PREFILL_KERNELS:-1}
# 384, not the code default 512: FAIR_PREFILL caps the chunk at 384 while
# something decodes, so a 512 gate switches these kernels off exactly then.
# Measured: contended 20K TTFT 12.98 -> 11.88 s, lone prompt flat.
export VLLM_GLM5_PREFILL_MIN_TOKENS=${VLLM_GLM5_PREFILL_MIN_TOKENS:-384}
# [decode-prologue] fuse the eager prologue, and sample a 1/TP slice of the
# batch per rank instead of all-gathering full-vocab logits everywhere.
export VLLM_GLM5_PROLOGUE_FUSE=${VLLM_GLM5_PROLOGUE_FUSE:-1}
# Batch-sharded sampling only means anything when there is more than one TP
# rank to shard across, so it defaults off under PP-only.
if [ "$TP" -gt 1 ]; then
  export VLLM_GLM5_LOCAL_LOGITS=${VLLM_GLM5_LOCAL_LOGITS:-1}
else
  export VLLM_GLM5_LOCAL_LOGITS=${VLLM_GLM5_LOCAL_LOGITS:-0}
fi
# [decode-small-kernels] sm_80 mHC fused post+pre, MoE routing/align, KDA decode.
# The per-family switches and token bounds keep their committed defaults --
# in particular VLLM_GLM5_DECODE_MOE_MAX_TOKENS is 8. Do not set them here.
export VLLM_GLM5_DECODE_KERNELS=${VLLM_GLM5_DECODE_KERNELS:-1}
# [thin-gemm] sm_80 thin-M BF16 GEMM for the layers W4A16 leaves unquantized.
# Bounded at M <= 32 (VLLM_GLM5_THIN_GEMM_MAX_TOKENS), which is a cudagraph
# capture size -- re-measure before changing it.
export VLLM_GLM5_THIN_GEMM=${VLLM_GLM5_THIN_GEMM:-1}
# [pcie-allreduce] host-staged all-reduce for the TP decode collectives. These
# cards have no GPU peer access, so CustomAllreduce disables itself and NCCL's
# SHM ring pays 2(N-1) sequential host hops per message. The host-staged path
# does it in one round trip through a shared /dev/shm segment: measured 2.09x
# per all-reduce call in a decode trace (78.6 -> 37.6 us mean), ms/step c1
# 19.10 -> 17.63 and c4 37.6 -> 31.8 (paired OFF/ON/OFF/ON).
# Messages above 512 KiB stay on NCCL (VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE) --
# prefill already runs near PCIe wire speed on the ring. Also gains 4,033 KV
# tokens. Kill switch: set it to 0 and restart.
# There are no TP all-reduces to replace under PP-only, so it defaults off there.
if [ "$TP" -gt 1 ]; then
  export VLLM_GLM5_HOST_ALLREDUCE=${VLLM_GLM5_HOST_ALLREDUCE:-1}
else
  export VLLM_GLM5_HOST_ALLREDUCE=${VLLM_GLM5_HOST_ALLREDUCE:-0}
fi
# [fair-prefill] decode-aware chunking, passed as real serve args below rather
# than through EXTRA_ARGS so EXTRA_ARGS stays free for callers.
FAIR_ARGS=()
if [ "${FAIR_PREFILL:-1}" = "1" ]; then
  FAIR_ARGS=(--prefill-chunk-with-decodes "${FAIR_CHUNK:-384}" --max-num-partial-prefills "${FAIR_PARTIAL:-2}")
fi

# Multimodal: vision + video stay on. Default is UNCAPPED (the profiler reserves for a
# context-filling video on stage 0, ~150k fewer KV tokens). Set MM_CAP=1 to bound inputs instead:
# profiler dummy = MM_IMAGES images + 1 video of MM_FRAMES frames, and the same bound is enforced
# on real inputs (media-io-kwargs frames, mm-processor-kwargs pixels).
MM_IMAGES=${MM_IMAGES:-4}; MM_FRAMES=${MM_FRAMES:-32}; MM_MAX_PIXELS=${MM_MAX_PIXELS:-1003520}
MM_ARGS=()
if [ "${MM_CAP:-0}" = "1" ]; then MM_ARGS=(
  --limit-mm-per-prompt "{\"image\":$MM_IMAGES,\"video\":{\"count\":1,\"num_frames\":$MM_FRAMES,\"width\":1280,\"height\":784}}"
  --mm-processor-kwargs "{\"max_pixels\":$MM_MAX_PIXELS}"
  --media-io-kwargs "{\"video\":{\"num_frames\":$MM_FRAMES}}"
); fi

SPEC=()
if [[ "${EXTRA_ARGS:-}" == *--speculative-config* ]]; then
  : # the caller supplied its own speculative-config; never add a second one
elif [ "$MODE" = "mtp" ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_N:-3}}")
elif [ "$MODE" = "dflash" ]; then
  SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":${SPEC_N:-3}}")
fi

CMD=("$VENV/bin/vllm" serve "$MODEL"
  --served-model-name "${SERVED_NAME:-glm-5.3-flash}"
  --pipeline-parallel-size "$PP"
  --tensor-parallel-size "$TP"
  --max-model-len "${MAX_LEN:-262144}"
  --max-num-seqs "${MAX_SEQS:-8}"
  --max-num-batched-tokens "${MAX_BATCHED:-2048}"
  --long-prefill-token-threshold "${PREFILL_CAP:-0}"
  --gpu-memory-utilization "${GPU_UTIL:-0.95}"
  --trust-remote-code
  --reasoning-parser "${REASONING_PARSER:-glm47}"
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-glm47}"
  --port "${PORT:-8000}"
  "${MM_ARGS[@]}" "${SPEC[@]}" "${FAIR_ARGS[@]}" ${EXTRA_ARGS:-})

if [ "${DRY:-0}" = "1" ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
exec "${CMD[@]}"
