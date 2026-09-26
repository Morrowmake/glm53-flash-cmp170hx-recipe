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
# API:
#   HOST            address to listen on (default 127.0.0.1: this machine only).
#                   0.0.0.0 listens on every interface -- set API_KEY first.
#   PORT            default 8000
#   API_KEY         if set, every /v1 request must send "Authorization: Bearer
#                   <key>" (passed to vLLM as VLLM_API_KEY, so it never appears
#                   on the command line). Unset = no key: anyone who can reach
#                   HOST:PORT can use the model.
#   SERVED_MODEL_NAME  model id clients send (default glm-5.3-flash; the older
#                   name SERVED_NAME is still read if this one is unset)
# Env overrides: LAYOUT, MAX_LEN, MAX_SEQS, MAX_BATCHED (default 3460 =
#   3,456-token prefill chunks under TP4, 2312 = 2,304-token chunks under PP4),
#   PREFILL_CAP (upstream long-prefill chunk cap, UNCONDITIONAL; default 0 = off
#   -- LEAVE IT 0, see below), GPU_UTIL, SPEC_N, REASONING_PARSER, TOOL_PARSER,
#   MM_CAP, PP, TP, VLLM_PP_LAYER_PARTITION, BLOCK_SIZE, EXTRA_ARGS.
#
# Layout (LAYOUT):
#   tp4 (default)  tensor-parallel 4 (PP=1, TP=4). Fastest per request; one or
#                  two interactive users. Assumes PCIe Gen2 x16 links.
#   pp4            pipeline-parallel 4 (PP=4, TP=1). Each card holds a quarter
#                  of the layers and passes activations on: much faster
#                  prefill, about twice the KV, many parallel users, and far
#                  less traffic between the cards. Measured on x16 links.
#   An explicit PP/TP still wins over LAYOUT.
#
# ===== sm_80 feature flags ==================================================
# The performance features, the repeatable-output fixes and the KV headroom
# changes, plus an optional PCIe peer-to-peer switch, measured one by one and
# then together. All are OFF by default IN THE CODE; the block below is the
# only thing that turns them on, so each one is a one-variable kill switch --
# set it to 0 in the environment and restart, no rebuild and no revert. (Two
# exceptions: VLLM_SPARSE_INDEXER_MAX_LOGITS_MB is off at 512, and
# VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY works the other way round, see below.)
#
#   VLLM_GLM5_PREFILL_OVERLAP=0   TP prefill comm/compute overlap
#   VLLM_GLM5_PREFILL_KERNELS=0   sm_80 prefill kernels
#   VLLM_GLM5_PROLOGUE_FUSE=0     fused eager decode prologue
#   VLLM_GLM5_LOCAL_LOGITS=0      batch-sharded logits + sampling
#   VLLM_GLM5_DECODE_KERNELS=0    sm_80 decode kernels
#   VLLM_GLM5_THIN_GEMM=0         sm_80 thin-M BF16 GEMM
#   FAIR_PREFILL=0                decode-aware prefill chunking
#   VLLM_GLM5_HOST_ALLREDUCE=0    host-staged no-P2P all-reduce
#   VLLM_GLM5_SHARED_EXPERT_REORDER=0  MoE shared experts after routed dispatch
#   VLLM_GLM5_DECODE_IDX_GLUE=0 / VLLM_GLM5_DECODE_KDA_V2=0 /
#   VLLM_GLM5_DECODE_MOE_ROUTE_V2=0 / VLLM_GLM5_DECODE_MHC_V2=0 /
#   VLLM_GLM5_DRAFTER_ROPE_FIT=0  second-generation decode, both layouts
#   VLLM_GLM5_DETERMINISTIC_MOE_ALIGN=0 / VLLM_GLM5_MOE_MASK_PADDING=0 /
#   VLLM_GLM5_TOPK_TIEFIX=0 / VLLM_GLM5_TOPK_SORTED=0 /
#   VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS=0 / VLLM_GLM5_FLA_PIN_AUTOTUNE=0
#                                 repeatable output
#   VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512 / VLLM_GLM5_DRAFTER_SELECTOR_SHARD=0 /
#   VLLM_GLM5_INDEXER_DECODE_ROWS=0 / VLLM_GLM5_INDEXER_GATHER_CLAMP=0
#                                 KV headroom (the selector shard is TP only)
#   VLLM_KV_MAMBA_INFLIGHT_STATES=0 / VLLM_KV_SWA_INFLIGHT_SCRATCH=0
#                                 KV accounting for prefill chunks in flight,
#                                 both layouts (0 reports the old, larger pool)
#   VLLM_GLM5_SMLA_PREFILL_PRED_LOAD=0  sparse-attention prefill gather that
#                                 skips empty slots, both layouts
#   VLLM_GLM5_TP4_KDA_PREFILL=0 / VLLM_GLM5_TP4_MARLIN_PREFILL=0
#                                 TP4 prefill kernels, LAYOUT=tp4 only
#   VLLM_GLM5_PP_KDA_PREFILL=0 / VLLM_GLM5_PP_SPARSE_MLA_PREFILL=0 /
#   VLLM_GLM5_PP_MARLIN_PREFILL=0 PP4 prefill kernels, LAYOUT=pp4 only
#   VLLM_PP_SPREAD_DECODES=0 / VLLM_PP_PACKED_HOP=0 / VLLM_PP_HOP_NO_METADATA=0 /
#   VLLM_PP_SPLIT_DRAFT_EVENT=0 / VLLM_GLM5_PP_FOLD_DRAFT_FC=0
#                                 PP4 pipeline and drafter changes, PP only
#   VLLM_PP_DRAFT_TAIL_STAGE=-1   PP4 drafter tail on stage 3 instead of 2
#   VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY=1  the sparse-attention decode schedule
#                                 from before 1.2.0 (the retuned one is on in
#                                 the engine; not set here)
#   VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1  device-memory custom all-reduce over
#                                 PCIe peer-to-peer. DEFAULT 0 HERE, because it
#                                 needs peer-to-peer enabled at the driver level
#                                 -- see the optional section in the README.
#                                 Measured on 1.3.0: decode step -3.7% at 1 user,
#                                 -6.9% at 4, -8.3% at 6, -6.2% at 8, cold
#                                 prefill unchanged, KV +13,209 tokens.
#                                 0 keeps the host-staged path.
#   VLLM_CUSTOM_ALLREDUCE_ALGO=   which CustomAllreduce kernel; 2stage here
#                                 because the built-in crossover is NVLink-tuned
#                                 and 1stage measured worse on Gen2 x16. Inert
#                                 unless the switch above is 1. Unset = upstream.
#   GLM5_NCCL_P2P_SYS=1           NCCL over peer-to-peer too (NCCL_P2P_LEVEL=SYS),
#                                 TP4 with the switch above at 1 only.
#                                 Default PENDING (0 for now), see below.
#   VLLM_GLM5_PREFILL_OVERLAP_BACKEND=  which communicator carries the prefill
#                                 overlap's split collectives. NOT set here: the
#                                 code default is nccl, which measured fastest.
#   MAX_BATCHED=2048              1,152-token prefill chunks, as 1.2.0 shipped
#                                 (the default 3460 gives 3,456-token chunks).
#
# If output quality is ever in question, try VLLM_GLM5_DECODE_KERNELS=0 first.
#
# PREFILL_CAP MUST STAY 0. It is upstream's UNCONDITIONAL chunk cap: it cost
# -15.3% prefill / +16.5% time to first token on a 23K prompt here, and as a
# side effect it drops chunks below the two prefill gates and silently
# disables the overlap and the prefill kernels. The decode-aware cap
# (FAIR_PREFILL) is the one to use -- it only applies while something is
# actually decoding.
# ============================================================================
set -euo pipefail
MODE=${1:-dflash}
case "$MODE" in
  -h|--help) sed -n '2,43p' "$0"; exit 0 ;;
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

# The caching allocator is chosen further down, together with the PCIe peer-to-peer switch:
# CustomAllreduce needs legacy CUDA IPC handles, which the expandable_segments
# (VMM) allocator cannot provide. Record whether the caller set it explicitly so
# that choice can win.
ALLOC_CONF_EXPLICIT=${PYTORCH_CUDA_ALLOC_CONF+1}
# Layout. LAYOUT=tp4 (default): TENSOR-PARALLEL 4. Every layer is split across
# all four cards, so it is the fastest per request, but it assumes PCIe Gen2 x16
# links: it moves ~9.4 MB per layer during prefill and ~100 small collectives
# per decode step, so on stock x4 links it is bus-bound.
# LAYOUT=pp4: PIPELINE-PARALLEL 4 (PP=4, TP=1). Each card holds a quarter of the
# layers and hands only activations to the next one, so it needs far less link
# bandwidth; it prefills much faster and holds about twice the KV, and it suits
# many parallel users rather than one fast one. It runs with the DFlash2
# drafter like TP4. Its numbers were measured on x16 links.
# An explicit PP/TP in the environment still wins over LAYOUT.
LAYOUT=${LAYOUT:-tp4}
case "$LAYOUT" in
  tp4) ;;
  pp4) PP=${PP:-4}; TP=${TP:-1} ;;
  *) echo "serve.sh: unknown LAYOUT '$LAYOUT' (expected tp4 or pp4)" >&2; exit 2 ;;
esac
PP=${PP:-1}; TP=${TP:-4}   # PP*TP must be 4
# Under PP the balanced layer split is 3 dense + 42 MoE (~3.8 GiB each). MTP keeps a 13.8 GiB BF16
# draft layer on the last stage, so the balanced split differs by mode. DFlash's drafter KV rides the
# MLA tensors, so it uses the non-MTP split.
if [ "$PP" = "4" ]; then
  if [ "$MODE" = "mtp" ]; then DEFAULT_PART=14,12,12,7; else DEFAULT_PART=13,11,11,10; fi
  export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-$DEFAULT_PART}"
elif [ -n "${VLLM_PP_LAYER_PARTITION:-}" ]; then export VLLM_PP_LAYER_PARTITION; else unset VLLM_PP_LAYER_PARTITION; fi
# KV block size under PP with DFlash2. Each stage holds the full 64-head
# linear-attention (KDA) state, which sets the attention block to 4,480 tokens;
# the drafter's matching block would then be 1,120 tokens, not a multiple of 64,
# which pushes the drafter out of the shared KV layout and makes the last stage
# pay about 6x per block. 4608 is the next multiple of 256: drafter block 1,152,
# state page padded 5%, valid for SPEC_N up to 8.
BLOCK_ARGS=()
if [ "$PP" -gt 1 ] && [ "$MODE" = "dflash" ]; then BLOCK_ARGS=(--block-size "${BLOCK_SIZE:-4608}"); fi
# Replicated input-embedding table under TP (VLLM_GLM5_REPLICATED_EMBED): skips the 2 hidden-size
# all-reduces per prefill chunk / decode step (target + MTP drafter) for +0.74 GiB per rank
# (-6% KV tokens). Off by default -- the KV is worth more here. Sharded table under PP-only.
if [ "$TP" -gt 1 ]; then export VLLM_GLM5_REPLICATED_EMBED=${VLLM_GLM5_REPLICATED_EMBED:-0}; fi

# --- pipeline-parallel (PP4) --------------------------------------------------
# Pipeline and drafter changes, PP only, each a kill switch at 0:
#   VLLM_PP_SPREAD_DECODES     spread decoding requests over all in-flight
#                              micro-batches, so no stage idles while one
#                              micro-batch carries every decode
#   VLLM_PP_PACKED_HOP         stage-to-stage hand-off packed into one transfer
#   VLLM_PP_HOP_NO_METADATA    hand-off without the per-step metadata exchange
#   VLLM_PP_SPLIT_DRAFT_EVENT  finer synchronisation of the drafter's inputs
#   VLLM_GLM5_PP_FOLD_DRAFT_FC drafter input projection folded
# Outputs identical with and without the first four; the fold is checked
# against a 64-bit reference. Measured together against PP4 without them:
# 4-user aggregate +7.3%, cold prefill +2.0%, 1-user step -0.8%, 8,366 fewer KV
# tokens.
if [ "$PP" -gt 1 ]; then
  export VLLM_PP_SPREAD_DECODES=${VLLM_PP_SPREAD_DECODES:-1}
  export VLLM_PP_PACKED_HOP=${VLLM_PP_PACKED_HOP:-1}
  export VLLM_PP_HOP_NO_METADATA=${VLLM_PP_HOP_NO_METADATA:-1}
  export VLLM_PP_SPLIT_DRAFT_EVENT=${VLLM_PP_SPLIT_DRAFT_EVENT:-1}
  export VLLM_GLM5_PP_FOLD_DRAFT_FC=${VLLM_GLM5_PP_FOLD_DRAFT_FC:-1}
  echo "serve.sh: pipeline: SPREAD=$VLLM_PP_SPREAD_DECODES PACKED_HOP=$VLLM_PP_PACKED_HOP HOP_NO_METADATA=$VLLM_PP_HOP_NO_METADATA SPLIT_DRAFT_EVENT=$VLLM_PP_SPLIT_DRAFT_EVENT FOLD_DRAFT_FC=$VLLM_GLM5_PP_FOLD_DRAFT_FC"
fi
# Drafter tail on stage 2 (PP4 only): moves the drafter's vocabulary pass and
# token selection from the last stage, which also runs the sampler, to stage 2.
# Outputs are meant to be bit-identical either way. -1 keeps it on the last
# stage (the engine default), 2 moves it.
# PENDING: the default is decided by this release's validation (on if
# bit-identical, 4-user aggregate at least +3% and 1 user not slower); until
# then it stays -1.
PP_DRAFT_TAIL_DEFAULT=-1
if [ "$PP" -gt 1 ] && [ "$MODE" = "dflash" ]; then
  export VLLM_PP_DRAFT_TAIL_STAGE=${VLLM_PP_DRAFT_TAIL_STAGE:-$PP_DRAFT_TAIL_DEFAULT}
  echo "serve.sh: drafter tail stage: $VLLM_PP_DRAFT_TAIL_STAGE (-1 = last stage)"
fi

# --- prefill kernels, per layout ----------------------------------------------
# Sparse-attention prefill gather, both layouts: empty top-k slots fetch
# nothing instead of all reading the same cache row. Bitwise-identical output.
# PP4 (64 heads): 450 -> 356 ms over the sparse-attention layers of an
# 8.2K-token prompt; TP4 (16 heads): the first chunk's kernel 1.58x faster.
# Kill switch: 0.
export VLLM_GLM5_SMLA_PREFILL_PRED_LOAD=${VLLM_GLM5_SMLA_PREFILL_PRED_LOAD:-1}
# PP4 prefill kernels (LAYOUT=pp4 only; each gated in the engine to 64 heads /
# whole experts on sm_80): linear-attention (KDA) chunked prefill, a new
# sparse-attention prefill kernel and a split-block Marlin MoE prefill. Each is
# at least as accurate as the code it replaces against a 64-bit reference on
# real inputs. With the gather above: cold prefill +19.5%. Kill switches: 0.
if [ "$LAYOUT" = pp4 ]; then
  export VLLM_GLM5_PP_KDA_PREFILL=${VLLM_GLM5_PP_KDA_PREFILL:-1}
  export VLLM_GLM5_PP_SPARSE_MLA_PREFILL=${VLLM_GLM5_PP_SPARSE_MLA_PREFILL:-1}
  export VLLM_GLM5_PP_MARLIN_PREFILL=${VLLM_GLM5_PP_MARLIN_PREFILL:-1}
  echo "serve.sh: PP4 prefill kernels: KDA=$VLLM_GLM5_PP_KDA_PREFILL SPARSE_MLA=$VLLM_GLM5_PP_SPARSE_MLA_PREFILL MARLIN=$VLLM_GLM5_PP_MARLIN_PREFILL PRED_LOAD=$VLLM_GLM5_SMLA_PREFILL_PRED_LOAD"
fi
# TP4 prefill kernels (LAYOUT=tp4 only; gated in the engine to sm_80): the
# KDA chunked prefill at 16 heads (1.42x per prompt per card) and the
# split-block Marlin MoE prefill at TP4 shards (1.24x; from 384 tokens,
# VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS). Accuracy against a 64-bit reference
# at least equal to the code they replace. Kill switches: 0.
if [ "$LAYOUT" = tp4 ]; then
  export VLLM_GLM5_TP4_KDA_PREFILL=${VLLM_GLM5_TP4_KDA_PREFILL:-1}
  export VLLM_GLM5_TP4_MARLIN_PREFILL=${VLLM_GLM5_TP4_MARLIN_PREFILL:-1}
  echo "serve.sh: TP4 prefill kernels: KDA=$VLLM_GLM5_TP4_KDA_PREFILL MARLIN=$VLLM_GLM5_TP4_MARLIN_PREFILL PRED_LOAD=$VLLM_GLM5_SMLA_PREFILL_PRED_LOAD"
fi

# --- KV accounting --------------------------------------------------------------
# With prefill chunks in flight, a request can hold more linear-attention state
# blocks and drafter window blocks than the engine used to reserve for it, so
# the KV pool it reported was larger than it can actually hold. These two
# settings make the reserve match, in both layouts. They change the reported
# pool and the start-up fit check only; outputs cannot change.
#   VLLM_KV_MAMBA_INFLIGHT_STATES  linear-attention states held by chunks in
#                                  flight (TP4: 17,932 fewer tokens reported,
#                                  -1.5%)
#   VLLM_KV_SWA_INFLIGHT_SCRATCH   drafter window blocks charged once per
#                                  running request (PP4: +2.15% over the first
#                                  one alone; TP4 unchanged)
# 0 brings back the old, larger figure. Kill switches: 0.
export VLLM_KV_MAMBA_INFLIGHT_STATES=${VLLM_KV_MAMBA_INFLIGHT_STATES:-1}
export VLLM_KV_SWA_INFLIGHT_SCRATCH=${VLLM_KV_SWA_INFLIGHT_SCRATCH:-1}
echo "serve.sh: KV accounting: MAMBA_INFLIGHT_STATES=$VLLM_KV_MAMBA_INFLIGHT_STATES SWA_INFLIGHT_SCRATCH=$VLLM_KV_SWA_INFLIGHT_SCRATCH"

# --- sm_80 feature flags (see the header for the kill switches) -------------
# Prefill overlap: split each mHC layer's post-attention part into S token
# micro-batches and fly the two per-layer all-reduces on a side stream. TP>1
# only; S=4 measured worse than S=2. _MIN_TOKENS stays at its 512 default and
# _CROSS_LAYER stays off (in tree, never validated on a GPU).
if [ "$TP" -gt 1 ]; then export VLLM_GLM5_PREFILL_OVERLAP=${VLLM_GLM5_PREFILL_OVERLAP:-1}; export VLLM_GLM5_PREFILL_OVERLAP_SPLITS=${VLLM_GLM5_PREFILL_OVERLAP_SPLITS:-2}; fi
# The overlap now runs beside a live CustomAllreduce and carries its split
# collectives on NCCL (VLLM_GLM5_PREFILL_OVERLAP_BACKEND, code default nccl),
# which is why the peer-to-peer switch no longer costs prefill. Left unset here.
# Prefill kernels: sm_80 mHC pre-norm projection + sparse-MLA DSA attention.
export VLLM_GLM5_PREFILL_KERNELS=${VLLM_GLM5_PREFILL_KERNELS:-1}
# 384, not the code default 512: FAIR_PREFILL caps the chunk at 384 while
# something decodes, so a 512 gate switches these kernels off exactly then.
# Measured: contended 20K TTFT 12.98 -> 11.88 s, lone prompt flat.
export VLLM_GLM5_PREFILL_MIN_TOKENS=${VLLM_GLM5_PREFILL_MIN_TOKENS:-384}
# Decode prologue: fuse the eager prologue, and sample a 1/TP slice of the
# batch per rank instead of all-gathering full-vocab logits everywhere.
export VLLM_GLM5_PROLOGUE_FUSE=${VLLM_GLM5_PROLOGUE_FUSE:-1}
# Batch-sharded sampling only means anything when there is more than one TP
# rank to shard across, so it defaults off under PP-only.
if [ "$TP" -gt 1 ]; then
  export VLLM_GLM5_LOCAL_LOGITS=${VLLM_GLM5_LOCAL_LOGITS:-1}
else
  export VLLM_GLM5_LOCAL_LOGITS=${VLLM_GLM5_LOCAL_LOGITS:-0}
fi
# Decode kernels: sm_80 mHC fused post+pre, MoE routing/align, KDA decode.
# The per-family switches and token bounds keep their committed defaults --
# in particular VLLM_GLM5_DECODE_MOE_MAX_TOKENS is 8. Do not set them here.
export VLLM_GLM5_DECODE_KERNELS=${VLLM_GLM5_DECODE_KERNELS:-1}
# Thin GEMM: sm_80 thin-M BF16 GEMM for the layers W4A16 leaves unquantized.
# Bounded at M <= 32 (VLLM_GLM5_THIN_GEMM_MAX_TOKENS), which is a cudagraph
# capture size -- re-measure before changing it.
export VLLM_GLM5_THIN_GEMM=${VLLM_GLM5_THIN_GEMM:-1}
# Host-staged all-reduce for the TP decode collectives. Stock cards have no GPU
# peer access, so CustomAllreduce disables itself and NCCL's SHM ring pays
# 2(N-1) sequential host hops per message. The host-staged path does it in one
# round trip through a shared /dev/shm segment: measured 2.09x per all-reduce
# call in a decode trace (78.6 -> 37.6 us mean), decode step 19.10 -> 17.63 ms
# at 1 user and 37.6 -> 31.8 ms at 4 (alternating off/on restarts).
# Messages above 512 KiB stay on NCCL (VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE) --
# prefill already runs near PCIe wire speed on the ring. Also gains 4,033 KV
# tokens. Kill switch: set it to 0 and restart.
# Set in both layouts, as validated: under PP4 there is no tensor-parallel
# all-reduce for it to replace, so it has nothing to do there.
export VLLM_GLM5_HOST_ALLREDUCE=${VLLM_GLM5_HOST_ALLREDUCE:-1}
# PCIe peer-to-peer switch: device-memory CustomAllreduce over PCIe peer-to-peer, instead
# of staging every collective through the host.
#   0 (default here) -- host-staged path serves. This is what the recipe ships,
#                       because peer-to-peer has to be enabled at the driver
#                       level first and most cards do not have it.
#   1                -- CustomAllreduce owns the TP all-reduce over PCIe P2P.
#                       Only set this if peer-to-peer is actually available:
#                       `nvidia-smi topo -p2p r` must report OK, not GNS.
# Under PP4 there is no tensor-parallel all-reduce, so the switch does nothing
# there; the stage-to-stage hand-off uses peer-to-peer by itself where the
# driver offers it.
# Measured on 1.3.0 (TP4), four cards with peer-to-peer available, one boot
# each, release defaults otherwise:
#   switch 0: decode step at 1 / 4 / 6 / 8 users 15.841 / 30.092 / 38.778 /
#             44.705 ms, cold prefill 2,484 tok/s, KV 1,174,567
#   switch 1: decode step at 1 / 4 / 6 / 8 users 15.250 / 28.015 / 35.549 /
#             41.921 ms,
#             cold prefill 2,490 tok/s, KV 1,187,776
# The switch ties both decisions together, so 0 is a complete fallback rather
# than a drop to NCCL: VLLM_GLM5_HOST_ALLREDUCE stays at 1 and serves.
export VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=${VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE:-0}
# 2stage, not the built-in crossover: that crossover takes one-shot below
# 512 KiB, which is tuned for NVLink and wrong on Gen2 x16. Forcing 1stage
# everywhere measured worse than either (16.83 ms at 1 user, 33.00 at 4). Unset it to get
# upstream's crossover back. Read only while a CustomAllreduce is actually
# serving, so it is inert when the switch above is 0.
export VLLM_CUSTOM_ALLREDUCE_ALGO=${VLLM_CUSTOM_ALLREDUCE_ALGO:-2stage}
# CustomAllreduce registers its captured graph buffers through legacy CUDA IPC
# handles, which the expandable_segments (VMM) allocator cannot provide, so the
# switch also picks the allocator. Not a safety net -- the engine detects the VMM
# allocator itself and stands the switch down with a warning rather than
# crashing -- but with expandable_segments on, setting it would do nothing.
# An explicit PYTORCH_CUDA_ALLOC_CONF in the environment wins.
if [ -n "${ALLOC_CONF_EXPLICIT:-}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF
elif [ "$TP" -gt 1 ] && [ "$VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE" = "1" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
else
  # Avoids caching-allocator fragmentation during MoE weight loading.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
# NCCL over peer-to-peer (TP4 with the switch above at 1). NCCL treats these
# cards, each on its own root port, as not peer-capable and runs its ring
# through host memory; NCCL_P2P_LEVEL=SYS lets it use peer-to-peer instead,
# which carries the large prefill collectives. GLM5_NCCL_P2P_SYS=1 turns it on,
# 0 off; an explicit NCCL_P2P_LEVEL always wins. Nothing changes with the
# switch above at 0 (the default) or under PP4.
# PENDING: whether it is on by default when peer-to-peer is on is decided by
# this release's validation (on if quality is unchanged and prefill gains);
# until then the default is 0.
NCCL_P2P_SYS_DEFAULT=0
if [ "$TP" -gt 1 ] && [ "$VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE" = "1" ] \
   && [ "${GLM5_NCCL_P2P_SYS:-$NCCL_P2P_SYS_DEFAULT}" = "1" ]; then
  export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
fi

# Shared-expert overlap: enqueue the routed experts first, then submit the MoE
# shared experts to the aux stream, so the two actually run at the same time.
# With the upstream order the shared experts were submitted before the gate and
# had retired before the routed Marlin kernels were queued behind them.
# Measured alongside the sparse-MLA decode retune: shared-expert GEMM time
# overlapping the routed Marlin kernels 0.01% -> 73.3% on a rank-0 decode
# trace; decode step 17.155 -> 17.010 ms at 1 user and 32.27 -> 32.10 at 4,
# against 0.03 and 0.04 ms between two unchanged restarts; cold prefill, KV
# cache size, time to first token and GSM8K unchanged, and log-probabilities
# within what two restarts of the same server show.
# Decode only: the 256-token shared-experts stream threshold keeps the
# 1152-token prefill chunks off this path entirely.
# Kill switch: set it to 0 and restart, no rebuild.
export VLLM_GLM5_SHARED_EXPERT_REORDER=${VLLM_GLM5_SHARED_EXPERT_REORDER:-1}

# Second-generation decode: five changes, each OFF in the engine code and turned
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
# On in both layouts: under PP4 the shapes are the same replicated ones, and
# the KDA v2 decode is gated to one sequence at 64 heads.
# Measured together, TP=4, PCIe peer-to-peer on, decode step before -> after:
#   1 user 16.26 -> 15.05 ms, 4 users 29.98 -> 27.76, 6 users 42.00 -> 34.62,
#   8 users 43.92 -> 41.55;
#   cold prefill flat; KV +2,048 tokens (RoPE fit +12,288, kernels -10,240).
TP4V2_DEFAULT=1
for _v in VLLM_GLM5_DECODE_IDX_GLUE VLLM_GLM5_DECODE_KDA_V2 VLLM_GLM5_DECODE_MOE_ROUTE_V2 VLLM_GLM5_DECODE_MHC_V2 VLLM_GLM5_DRAFTER_ROPE_FIT; do
  export "$_v=${!_v:-$TP4V2_DEFAULT}"
done
echo "serve.sh: second-generation decode: IDX_GLUE=$VLLM_GLM5_DECODE_IDX_GLUE KDA_V2=$VLLM_GLM5_DECODE_KDA_V2 MOE_ROUTE_V2=$VLLM_GLM5_DECODE_MOE_ROUTE_V2 MHC_V2=$VLLM_GLM5_DECODE_MHC_V2 DRAFTER_ROPE_FIT=$VLLM_GLM5_DRAFTER_ROPE_FIT"

# Repeatable output: a request on its own returns the same output every time: the
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
#   VLLM_GLM5_FLA_PIN_AUTOTUNE         the linear-attention (KDA) prefill
#                                      kernels use one pinned configuration
#                                      per shape instead of tuning themselves
#                                      in each new process, so a fresh install
#                                      and a cleared cache give the same
#                                      output as every other start. The pinned
#                                      configurations are exactly as accurate
#                                      as the tuned ones against a 64-bit
#                                      reference, at no measurable cost.
# Measured on vs off, four alternating restarts: decode step +0.59% at 1 user,
# -1.75% at 4, -3.74% at 8 (within restart-to-restart variation), prefill
# -0.32%.
# Never set VLLM_MOE_SKIP_PADDING=0 while VLLM_GLM5_MOE_MASK_PADDING=1: the
# padding mask relies on the buffer that SKIP_PADDING=1 (the engine default)
# fills, so serve.sh refuses that combination.
export VLLM_GLM5_DETERMINISTIC_MOE_ALIGN=${VLLM_GLM5_DETERMINISTIC_MOE_ALIGN:-1}
export VLLM_GLM5_MOE_MASK_PADDING=${VLLM_GLM5_MOE_MASK_PADDING:-1}
export VLLM_GLM5_TOPK_TIEFIX=${VLLM_GLM5_TOPK_TIEFIX:-1}
export VLLM_GLM5_TOPK_SORTED=${VLLM_GLM5_TOPK_SORTED:-1}
export VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS=${VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS:-8}
export VLLM_GLM5_FLA_PIN_AUTOTUNE=${VLLM_GLM5_FLA_PIN_AUTOTUNE:-1}
if [ "${VLLM_MOE_SKIP_PADDING:-1}" = "0" ] && [ "$VLLM_GLM5_MOE_MASK_PADDING" = "1" ]; then
  echo "serve.sh: VLLM_MOE_SKIP_PADDING=0 with VLLM_GLM5_MOE_MASK_PADDING=1 would mask real rows; refusing" >&2
  exit 2
fi
echo "serve.sh: repeatable output: MOE_ALIGN=$VLLM_GLM5_DETERMINISTIC_MOE_ALIGN MASK_PADDING=$VLLM_GLM5_MOE_MASK_PADDING TOPK_TIEFIX=$VLLM_GLM5_TOPK_TIEFIX TOPK_SORTED=$VLLM_GLM5_TOPK_SORTED SPLIT_ROWS=$VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS FLA_PIN=$VLLM_GLM5_FLA_PIN_AUTOTUNE"

# KV headroom: return memory the engine reserves but a step cannot use to the
# KV pool. Outputs bit-identical, no measurable step-time cost; under TP4
# together +39,626 KV tokens (+3.45%, measured with peer-to-peer on). The logits
# budget and the decode rows apply in both layouts; the selector shard splits
# across tensor-parallel cards, so it is TP only.
#   VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128  prefill indexer logits budget; set
#                                  512 (upstream's default) to switch it off
#   VLLM_GLM5_DRAFTER_SELECTOR_SHARD  DFlash2 selector tables split across the
#                                  cards (one extra all-reduce per draft step)
#   VLLM_GLM5_INDEXER_DECODE_ROWS  indexer decode block tables sized by the
#                                  decode rows a step can hold
#   VLLM_GLM5_INDEXER_GATHER_CLAMP indexer gather workspace clamp; ON in the
#                                  engine code already, 0 is its kill switch
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-128}
export VLLM_GLM5_INDEXER_DECODE_ROWS=${VLLM_GLM5_INDEXER_DECODE_ROWS:-1}
if [ "$TP" -gt 1 ]; then
  export VLLM_GLM5_DRAFTER_SELECTOR_SHARD=${VLLM_GLM5_DRAFTER_SELECTOR_SHARD:-1}
fi
echo "serve.sh: KV headroom: MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-512} SELECTOR_SHARD=${VLLM_GLM5_DRAFTER_SELECTOR_SHARD:-0} DECODE_ROWS=${VLLM_GLM5_INDEXER_DECODE_ROWS:-0} GATHER_CLAMP=${VLLM_GLM5_INDEXER_GATHER_CLAMP:-1}"

# Prefill chunk: batched-token budget, which sets the prefill chunk.
# The KDA state page forces chunk ends onto 1152-token blocks, and DFlash
# reserves SPEC_N draft slots out of the budget, so the chunk is
# floor((MAX_BATCHED - SPEC_N) / 1152) x 1152:
#   2048 -> 1152 (1.2.0), 2312 -> 2304, 3460 -> 3456 (the default here).
# Larger chunks let each prefill-overlap half carry more tokens per pass over
# the experts: 3,456-token chunks measured +2.4% cold prefill against 2,304 at
# 180 W per card, for about 35,600 fewer KV tokens (-3%, a larger chunk raises the
# activation peak the memory profiler reserves for). Decode step unchanged.
# Contended prefill is unaffected (FAIR_PREFILL caps it at 384 while anything
# decodes). If SPEC_N goes above 4, raise MAX_BATCHED to 3456 + SPEC_N to
# keep 3,456-token chunks. Under PP4 the default is 2312 (2,304-token chunks),
# the chunk it was validated with.
# Kill switch: MAX_BATCHED=2048 restores 1.2.0's chunking.
if [ "$TP" -gt 1 ]; then MAX_BATCHED_DEFAULT=3460; else MAX_BATCHED_DEFAULT=2312; fi
# Fair prefill: decode-aware chunking, passed as real serve args below rather
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

# The key goes to vLLM through the environment rather than --api-key, so it
# does not show up in the process list.
if [ -n "${API_KEY:-}" ]; then export VLLM_API_KEY="$API_KEY"; fi

CMD=("$VENV/bin/vllm" serve "$MODEL"
  --served-model-name "${SERVED_MODEL_NAME:-${SERVED_NAME:-glm-5.3-flash}}"
  --host "${HOST:-127.0.0.1}"
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
  ${MM_ARGS[@]+"${MM_ARGS[@]}"} ${SPEC[@]+"${SPEC[@]}"} ${FAIR_ARGS[@]+"${FAIR_ARGS[@]}"}
  ${BLOCK_ARGS[@]+"${BLOCK_ARGS[@]}"} ${EXTRA_ARGS:-})

if [ "${DRY:-0}" = "1" ]; then
  echo "serve.sh: DRY=1, layout $LAYOUT (PP=$PP TP=$TP), environment the server would get:"
  env | LC_ALL=C sort | grep -E '^(VLLM_GLM5_|VLLM_SPARSE_|VLLM_ALLOW_PCIE_|VLLM_CUSTOM_ALLREDUCE_|VLLM_PP_|VLLM_KV_|VLLM_MOE_|PYTORCH_CUDA_ALLOC_CONF=|NCCL_)' | sed 's/^/  /' || true
  if [ -n "${VLLM_API_KEY:-}" ]; then echo "  VLLM_API_KEY=(set, not shown)"; else echo "  (no API key: /v1 is open to anyone who can reach ${HOST:-127.0.0.1}:${PORT:-8000})"; fi
  echo "serve.sh: DRY=1, command:"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
exec "${CMD[@]}"
