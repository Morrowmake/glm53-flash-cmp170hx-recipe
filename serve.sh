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
# Set DRY=1 to print the environment and the command that would run instead
# of running them.
#
# Paths (all relative to this repo unless overridden):
#   VENV            venv from install.sh            (default ./venv)
#   MODEL           target model directory          (default ./models/GLM-5.3-Flash-W4A16-MTP)
#   DFLASH_MODEL    drafter directory               (default ./models/GLM-5.3-Flash-DFlash2)
#   CUDA_HOME       CUDA toolkit root               (default /usr/local/cuda-13.3)
# Env overrides: MAX_LEN, MAX_SEQS, MAX_BATCHED (default 3460 = 3,456-token
#   prefill chunks), PREFILL_CAP (upstream long-prefill chunk cap, UNCONDITIONAL;
#   default 0 = off -- LEAVE IT 0, see below), GPU_UTIL, PORT, SPEC_N,
#   SERVED_NAME, REASONING_PARSER, TOOL_PARSER, MM_CAP, PP, TP,
#   VLLM_PP_LAYER_PARTITION, EXTRA_ARGS.
#
# Layout: tensor-parallel 4 (PP=1, TP=4), the one layout this release
#   supports. It assumes PCIe Gen2 x16 links between the cards. A
#   pipeline-parallel layout for narrower links is planned for a later release.
#
# ===== sm_80 feature flags ==================================================
# The performance features, the determinism fixes and the KV levers, plus an
# optional PCIe peer-to-peer gate, validated individually and then together.
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
#   VLLM_GLM5_SHARED_EXPERT_REORDER=0  MoE shared experts after routed dispatch  [shared-expert-stream]
#   VLLM_GLM5_DECODE_IDX_GLUE=0 / VLLM_GLM5_DECODE_KDA_V2=0 /
#   VLLM_GLM5_DECODE_MOE_ROUTE_V2=0 / VLLM_GLM5_DECODE_MHC_V2=0 /
#   VLLM_GLM5_DRAFTER_ROPE_FIT=0  second-generation decode, TP only  [tp4-decode-v2]
#   VLLM_GLM5_DETERMINISTIC_MOE_ALIGN=0 / VLLM_GLM5_MOE_MASK_PADDING=0 /
#   VLLM_GLM5_TOPK_TIEFIX=0 / VLLM_GLM5_TOPK_SORTED=0 /
#   VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS=0  repeatable output          [determinism]
#   VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512 / VLLM_GLM5_DRAFTER_SELECTOR_SHARD=0 /
#   VLLM_GLM5_INDEXER_DECODE_ROWS=0 / VLLM_GLM5_INDEXER_GATHER_CLAMP=0
#                                 KV headroom, TP only          [kv-levers]
#   VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1  device-memory custom all-reduce over
#                                 PCIe peer-to-peer. DEFAULT 0 HERE, because it
#                                 needs peer-to-peer enabled at the driver level
#                                 -- see the optional section in the README.
#                                 On this release: ms/step -3.7% at c1, -6.9%
#                                 at c4, -8.3% at c6, -6.2% at c8, cold prefill
#                                 unchanged, KV +13,209 tokens.
#                                 0 keeps the host-staged path.  [pcie-p2p-gate]
#   VLLM_CUSTOM_ALLREDUCE_ALGO=   which CustomAllreduce kernel; 2stage here
#                                 because the built-in crossover is NVLink-tuned
#                                 and 1stage measured worse on Gen2 x16. Inert
#                                 unless the gate above is 1. Unset = upstream.
#   VLLM_GLM5_PREFILL_OVERLAP_BACKEND=  which communicator carries the prefill
#                                 overlap's split collectives. NOT set here: the
#                                 code default is nccl, which measured fastest.
#   MAX_BATCHED=2048              1,152-token prefill chunks, as 1.2.0 shipped
#                                 (the default 3460 gives 3,456-token
#                                 chunks).                 [prefill-chunk-3456]
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
  -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
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

# The caching allocator is chosen further down, together with the PCIe-P2P gate:
# CustomAllreduce needs legacy CUDA IPC handles, which the expandable_segments
# (VMM) allocator cannot provide. Record whether the caller set it explicitly so
# that choice can win.
ALLOC_CONF_EXPLICIT=${PYTORCH_CUDA_ALLOC_CONF+1}
# Layout: TENSOR-PARALLEL 4. Every layer is split across all four cards, so it
# is the fastest per request, but it assumes PCIe Gen2 x16 links: it moves
# ~9.4 MB per layer during prefill and ~100 small collectives per decode step,
# so on stock x4 links it is bus-bound. PP is left as a knob because the engine
# supports it, but PP=4 is UNSUPPORTED IN THIS RELEASE -- nothing below is
# tuned for it. A pipeline-parallel layout is planned for a later release.
PP=${PP:-1}; TP=${TP:-4}   # PP*TP must be 4
# Under PP the balanced layer split is 3 dense + 42 MoE (~3.8 GiB each). MTP keeps a 13.8 GiB BF16
# draft layer on the last stage, so the balanced split differs by mode. DFlash's drafter KV rides the
# MLA tensors, so it uses the non-MTP split.
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
# The overlap now runs beside a live CustomAllreduce and carries its split
# collectives on NCCL (VLLM_GLM5_PREFILL_OVERLAP_BACKEND, code default nccl),
# which is why the PCIe-P2P gate no longer costs prefill. Left unset here.
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
# [pcie-p2p-gate] device-memory CustomAllreduce over PCIe peer-to-peer, instead
# of staging every collective through the host.
#   0 (default here) -- host-staged path serves. This is what the recipe ships,
#                       because peer-to-peer has to be enabled at the driver
#                       level first and most cards do not have it.
#   1                -- CustomAllreduce owns the TP all-reduce over PCIe P2P.
#                       Only set this if peer-to-peer is actually available:
#                       `nvidia-smi topo -p2p r` must report OK, not GNS.
# Measured on this release, four cards with peer-to-peer available, one boot
# each, release defaults otherwise:
#   gate 0: ms/step c1 15.841, c4 30.092, c6 38.778, c8 44.705,
#           cold prefill 2,484 tok/s, KV 1,174,567
#   gate 1: ms/step c1 15.250, c4 28.015, c6 35.549, c8 41.921,
#           cold prefill 2,490 tok/s, KV 1,187,776
# The gate ties both decisions together, so 0 is a complete fallback rather
# than a drop to NCCL: VLLM_GLM5_HOST_ALLREDUCE stays at 1 and serves.
export VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=${VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE:-0}
# 2stage, not the built-in crossover: that crossover takes one-shot below
# 512 KiB, which is tuned for NVLink and wrong on Gen2 x16. Forcing 1stage
# everywhere measured worse than either (c1 16.83 / c4 33.00). Unset it to get
# upstream's crossover back. Read only while a CustomAllreduce is actually
# serving, so it is inert when the gate above is 0.
export VLLM_CUSTOM_ALLREDUCE_ALGO=${VLLM_CUSTOM_ALLREDUCE_ALGO:-2stage}
# CustomAllreduce registers its captured graph buffers through legacy CUDA IPC
# handles, which the expandable_segments (VMM) allocator cannot provide, so the
# gate also picks the allocator. Not a safety net -- the engine detects the VMM
# allocator itself and stands the gate down with a warning rather than crashing
# -- but with expandable_segments on, setting the gate would simply do nothing.
# An explicit PYTORCH_CUDA_ALLOC_CONF in the environment wins.
if [ -n "${ALLOC_CONF_EXPLICIT:-}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF
elif [ "$VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE" = "1" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
else
  # Avoids caching-allocator fragmentation during MoE weight loading.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

# [shared-expert-stream] enqueue the routed experts first, then submit the MoE
# shared experts to the aux stream, so the two actually run at the same time.
# With the upstream order the shared experts were submitted before the gate and
# had retired before the routed Marlin kernels were queued behind them.
# Measured alongside the sparse-MLA decode retune: shared-expert GEMM time
# overlapping the routed Marlin kernels 0.01% -> 73.3% on a rank-0 decode
# trace; ms/step c1 17.155 -> 17.010 and c4 32.27 -> 32.10 against a
# base-to-base drift of 0.03 and 0.04; cold prefill, KV cache size, TTFT and
# gsm8k unchanged, logprob drift inside the same-server floor.
# Decode only: the 256-token shared-experts stream threshold keeps the
# 1152-token prefill chunks off this path entirely.
# Kill switch: set it to 0 and restart, no rebuild.
export VLLM_GLM5_SHARED_EXPERT_REORDER=${VLLM_GLM5_SHARED_EXPERT_REORDER:-1}

# [tp4-decode-v2] five decode changes, each OFF in the engine code and turned
# on here under TP only. Kill switch for any of them: set it to 0 and restart.
#   VLLM_GLM5_DECODE_IDX_GLUE      sparse-attention indexer decode glue folded
#                                  into fewer kernels, plus the MoE shared add
#   VLLM_GLM5_DECODE_KDA_V2        KDA decode with its gate projections fused
#   VLLM_GLM5_DECODE_MOE_ROUTE_V2  MoE gate GEMV + top-k + align in one kernel
#                                  (up to 32 tokens)
#   VLLM_GLM5_DECODE_MHC_V2        mHC decode v2 (up to 32 tokens)
#   VLLM_GLM5_DRAFTER_ROPE_FIT     size the DFlash2 drafter's RoPE cache to the
#                                  context instead of 1M positions: output-
#                                  identical, more KV
# The v2 decode kernels need VLLM_GLM5_DECODE_KERNELS=1 (set above). New
# thin-GEMM rows for 24-row batches ride VLLM_GLM5_THIN_GEMM and need no switch.
# Under PP (unsupported here) they default to 0.
# Measured together, TP=4, PCIe P2P gate on, ms/step before -> after:
#   c1 16.26 -> 15.05, c4 29.98 -> 27.76, c6 42.00 -> 34.62, c8 43.92 -> 41.55;
#   cold prefill flat; KV +2,048 tokens (RoPE fit +12,288, kernels -10,240).
if [ "$TP" -gt 1 ]; then TP4V2_DEFAULT=1; else TP4V2_DEFAULT=0; fi
for _v in VLLM_GLM5_DECODE_IDX_GLUE VLLM_GLM5_DECODE_KDA_V2 VLLM_GLM5_DECODE_MOE_ROUTE_V2 VLLM_GLM5_DECODE_MHC_V2 VLLM_GLM5_DRAFTER_ROPE_FIT; do
  export "$_v=${!_v:-$TP4V2_DEFAULT}"
done
echo "serve.sh: [tp4-decode-v2] IDX_GLUE=$VLLM_GLM5_DECODE_IDX_GLUE KDA_V2=$VLLM_GLM5_DECODE_KDA_V2 MOE_ROUTE_V2=$VLLM_GLM5_DECODE_MOE_ROUTE_V2 MHC_V2=$VLLM_GLM5_DECODE_MHC_V2 DRAFTER_ROPE_FIT=$VLLM_GLM5_DRAFTER_ROPE_FIT"

# [determinism] a request on its own returns the same output every time: the
# same tokens and the same log-probabilities on every repeat and across
# restarts. Each fix is OFF in the engine code and turned on here, in every
# layout. Kill switch for any of them: set it to 0 and restart.
#   VLLM_GLM5_DETERMINISTIC_MOE_ALIGN  MoE block alignment by counting sort, in
#                                      a fixed order (no atomics)
#   VLLM_GLM5_MOE_MASK_PADDING         CUDA-graph padding rows routed to no
#                                      expert (inside the fused router when it
#                                      is on)
#   VLLM_GLM5_TOPK_TIEFIX              sparse-attention indexer top-k keeps the
#                                      lowest index among exact ties
#   VLLM_GLM5_TOPK_SORTED              indexer top-k rows in ascending order
#   VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS   batches of up to this many rows run the
#                                      tie fix + sort split across more
#                                      programs (8; 0 = one program per row)
# Measured on vs off, four alternating boots: ms/step c1 +0.59%, c4 -1.75%,
# c8 -3.74% (inside restart noise), prefill -0.32%.
# Never set VLLM_MOE_SKIP_PADDING=0 while VLLM_GLM5_MOE_MASK_PADDING=1: the
# padding mask relies on the buffer that SKIP_PADDING=1 (the engine default)
# fills, so serve.sh refuses that combination.
export VLLM_GLM5_DETERMINISTIC_MOE_ALIGN=${VLLM_GLM5_DETERMINISTIC_MOE_ALIGN:-1}
export VLLM_GLM5_MOE_MASK_PADDING=${VLLM_GLM5_MOE_MASK_PADDING:-1}
export VLLM_GLM5_TOPK_TIEFIX=${VLLM_GLM5_TOPK_TIEFIX:-1}
export VLLM_GLM5_TOPK_SORTED=${VLLM_GLM5_TOPK_SORTED:-1}
export VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS=${VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS:-8}
if [ "${VLLM_MOE_SKIP_PADDING:-1}" = "0" ] && [ "$VLLM_GLM5_MOE_MASK_PADDING" = "1" ]; then
  echo "serve.sh: VLLM_MOE_SKIP_PADDING=0 with VLLM_GLM5_MOE_MASK_PADDING=1 would mask real rows; refusing" >&2
  exit 2
fi
echo "serve.sh: [determinism] MOE_ALIGN=$VLLM_GLM5_DETERMINISTIC_MOE_ALIGN MASK_PADDING=$VLLM_GLM5_MOE_MASK_PADDING TOPK_TIEFIX=$VLLM_GLM5_TOPK_TIEFIX TOPK_SORTED=$VLLM_GLM5_TOPK_SORTED SPLIT_ROWS=$VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS"

# [kv-levers] return memory the engine reserves but a step cannot use to the KV
# pool. TP only. Outputs bit-identical, no measurable step-time cost; together
# +39,626 KV tokens.
#   VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128  prefill indexer logits budget; set
#                                  512 (upstream's default) to switch it off
#   VLLM_GLM5_DRAFTER_SELECTOR_SHARD  DFlash2 selector tables split across the
#                                  cards (one extra all-reduce per draft step)
#   VLLM_GLM5_INDEXER_DECODE_ROWS  indexer decode block tables sized by the
#                                  decode rows a step can hold
#   VLLM_GLM5_INDEXER_GATHER_CLAMP indexer gather workspace clamp; ON in the
#                                  engine code already, 0 is its kill switch
if [ "$TP" -gt 1 ]; then
  export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-128}
  export VLLM_GLM5_DRAFTER_SELECTOR_SHARD=${VLLM_GLM5_DRAFTER_SELECTOR_SHARD:-1}
  export VLLM_GLM5_INDEXER_DECODE_ROWS=${VLLM_GLM5_INDEXER_DECODE_ROWS:-1}
fi
echo "serve.sh: [kv-levers] MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-512} SELECTOR_SHARD=${VLLM_GLM5_DRAFTER_SELECTOR_SHARD:-0} DECODE_ROWS=${VLLM_GLM5_INDEXER_DECODE_ROWS:-0} GATHER_CLAMP=${VLLM_GLM5_INDEXER_GATHER_CLAMP:-1}"

# [prefill-chunk-3456] batched-token budget, which sets the prefill chunk.
# The KDA state page forces chunk ends onto 1152-token blocks, and DFlash
# reserves SPEC_N draft slots out of the budget, so the chunk is
# floor((MAX_BATCHED - SPEC_N) / 1152) x 1152:
#   2048 -> 1152 (1.2.0), 2312 -> 2304, 3460 -> 3456 (the default here).
# Larger chunks let each prefill-overlap half carry more tokens per pass over
# the experts: 3,456-token chunks measured +2.4% to +3.6% cold prefill against
# 2,304, for about 35,600 fewer KV tokens (-3%, a larger chunk raises the
# activation peak the memory profiler reserves for). Decode ms/step unchanged.
# Contended prefill is unaffected (FAIR_PREFILL caps it at 384 while anything
# decodes). If SPEC_N goes above 4, raise MAX_BATCHED to 3456 + SPEC_N to
# keep 3,456-token chunks. Under PP (unsupported here) the default stays 2312.
# Kill switch: MAX_BATCHED=2048 restores 1.2.0's chunking.
if [ "$TP" -gt 1 ]; then MAX_BATCHED_DEFAULT=3460; else MAX_BATCHED_DEFAULT=2312; fi
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
  --max-num-batched-tokens "${MAX_BATCHED:-$MAX_BATCHED_DEFAULT}"
  --long-prefill-token-threshold "${PREFILL_CAP:-0}"
  --gpu-memory-utilization "${GPU_UTIL:-0.95}"
  --trust-remote-code
  --reasoning-parser "${REASONING_PARSER:-glm47}"
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-glm47}"
  --port "${PORT:-8000}"
  "${MM_ARGS[@]}" "${SPEC[@]}" "${FAIR_ARGS[@]}" ${EXTRA_ARGS:-})

if [ "${DRY:-0}" = "1" ]; then
  echo "serve.sh: DRY=1, environment the server would get:"
  env | LC_ALL=C sort | grep -E '^(VLLM_GLM5_|VLLM_SPARSE_|VLLM_ALLOW_PCIE_|VLLM_CUSTOM_ALLREDUCE_|VLLM_PP_|VLLM_MOE_|PYTORCH_CUDA_ALLOC_CONF=|NCCL_)' | sed 's/^/  /' || true
  echo "serve.sh: DRY=1, command:"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
exec "${CMD[@]}"
