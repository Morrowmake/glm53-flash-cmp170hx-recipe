# Engine switches

The engine features behind the published numbers are controlled by about
forty internal variables (`VLLM_GLM5_*`, `VLLM_PP_*`, `VLLM_KV_*`,
`VLLM_CUSTOM_ALLREDUCE_ALGO`, `PYTORCH_CUDA_ALLOC_CONF` and a few others).
You do not need any of them to run the model: `serve.sh` sets every one to its
release default. They exist so that each feature can be switched off on its
own, which makes them the first tool for troubleshooting.

Each is listed, with its default, in
[`.env.advanced.example`](../.env.advanced.example). To change one, put the line
in your `.env` (`./start.sh` reads only `.env`), or set it for one run on the
command line. For a container started by hand or with Docker Compose, set it in
[`docker/container.env`](../docker/container.env).

## Troubleshooting with kill switches

If output quality is ever in question, try `VLLM_GLM5_DECODE_KERNELS=0` first.
If a problem goes away with one feature switched off, that feature is the
place to look; include the switch and the result when you report it.


Each feature is one variable. Set it to `0` and restart; no rebuild, no
revert. The PP4 draft tail is switched off with
`VLLM_PP_DRAFT_TAIL_STAGE=-1` (default `2`). Two others work differently:
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` is switched off with `512`, and `VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY` is switched *on* (`1`)
to go back to the old schedule.

```bash
VLLM_GLM5_DECODE_KERNELS=0 ./start.sh restart
```

| Variable | Feature | Layout |
|---|---|---|
| `VLLM_PP_DRAFT_TAIL_STAGE` | stage `2` runs the drafter's final step; `-1` switches it off | PP4 |
| `VLLM_GLM5_PREFILL_OVERLAP` | prefill overlap | TP4 |
| `VLLM_GLM5_PREFILL_KERNELS` | Ampere prefill kernels (mHC projection, sparse attention) | both |
| `VLLM_GLM5_SMLA_PREFILL_PRED_LOAD` | sparse-attention prefill gather that skips empty slots | both |
| `VLLM_GLM5_TP4_KDA_PREFILL`, `VLLM_GLM5_TP4_MARLIN_PREFILL` | linear-attention and MoE prefill kernels at TP4 shapes | TP4 |
| `VLLM_GLM5_PP_KDA_PREFILL`, `VLLM_GLM5_PP_SPARSE_MLA_PREFILL`, `VLLM_GLM5_PP_MARLIN_PREFILL` | linear-attention, sparse-attention and MoE prefill kernels at PP4 shapes | PP4 |
| `VLLM_PP_SPREAD_DECODES` | decodes spread over every in-flight micro-batch | PP4 |
| `VLLM_PP_PACKED_HOP`, `VLLM_PP_HOP_NO_METADATA` | packed, metadata-free hand-off between stages | PP4 |
| `VLLM_PP_SPLIT_DRAFT_EVENT`, `VLLM_GLM5_PP_FOLD_DRAFT_FC` | drafter synchronisation and input projection | PP4 |
| `VLLM_GLM5_PROLOGUE_FUSE` | fused decode prologue | both |
| `VLLM_GLM5_LOCAL_LOGITS` | batch-sharded logits and sampling | TP4 |
| `VLLM_GLM5_DECODE_KERNELS` | fused decode kernels | both |
| `VLLM_GLM5_DECODE_IDX_GLUE`, `VLLM_GLM5_DECODE_KDA_V2`, `VLLM_GLM5_DECODE_MOE_ROUTE_V2`, `VLLM_GLM5_DECODE_MHC_V2` | second-generation decode kernels | both |
| `VLLM_GLM5_DRAFTER_ROPE_FIT` | drafter position table sized to the context (more KV) | both |
| `VLLM_GLM5_THIN_GEMM` | small-batch GEMMs | both |
| `VLLM_GLM5_HOST_ALLREDUCE` | host-staged all-reduce | TP4 |
| `VLLM_GLM5_SHARED_EXPERT_REORDER` | shared experts overlapped with the routed experts | both |
| `FAIR_PREFILL` | fair prefill | both |
| `VLLM_GLM5_DETERMINISTIC_MOE_ALIGN` | same output every time: MoE block alignment in a fixed order | both |
| `VLLM_GLM5_MOE_MASK_PADDING` | same output every time: CUDA-graph padding rows kept out of the MoE | both |
| `VLLM_GLM5_TOPK_TIEFIX`, `VLLM_GLM5_TOPK_SORTED` | same output every time: indexer top-k consistent on ties, in a fixed order | both |
| `VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS` | the two above spread over more programs for batches up to `8` rows; `0` = one per row | both |
| `VLLM_GLM5_FLA_PIN_AUTOTUNE` | same output on every install: pinned linear-attention prefill configurations | both |
| `VLLM_KV_MAMBA_INFLIGHT_STATES`, `VLLM_KV_SWA_INFLIGHT_SCRATCH` | KV reserve for prefill chunks in flight; `0` reports the old, larger pool | both |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | KV headroom: prefill indexer logits budget, `128` here; `512` (upstream's) switches it off | both |
| `VLLM_GLM5_DRAFTER_SELECTOR_SHARD` | KV headroom: drafter selector tables split across the cards | TP4 |
| `VLLM_GLM5_INDEXER_DECODE_ROWS` | KV headroom: indexer decode tables sized by the decode rows | both |
| `VLLM_GLM5_INDEXER_GATHER_CLAMP` | KV headroom: indexer gather workspace clamp (on in the engine itself) | both |
| `VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY` | set to `1` for the sparse-attention decode schedule from before the retune (default `0`, set in the engine) | both |

`DRY=1 ./start.sh` prints the container command and the environment the server
would get, so you can check what is on. It runs the preflight and says whether
a real start would pull or download, but pulls, downloads and launches nothing,
so it also works before the first install.

## Other engine settings

| Variable | Default | What it does |
|---|---|---|
| `VLLM_GLM5_PREFILL_OVERLAP_SPLITS` | `2` | splits used by the TP prefill overlap |
| `VLLM_GLM5_PREFILL_MIN_TOKENS` | `384` | chunk-size gate for the Ampere prefill kernels; 384, not the code default 512, because `FAIR_CHUNK` caps the chunk at 384 while something decodes |
| `VLLM_GLM5_MARLIN_DECODE_CUDA`, `VLLM_GLM5_MARLIN_PREFILL_CUDA` | decode on if installed, prefill off | optional compiled Marlin; see [Optional compiled Marlin](compiled-marlin.md) |
| `VLLM_GLM5_MARLIN_DECODE_VARIANT` | `orig` | compiled decode reduction order: `orig` (faster) or `exact` (released order) |
| `VLLM_GLM5_DFLASH_ADAPTIVE_K` | `1` | load-adaptive draft depth (`VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS`, default `5,4`: depth at 1 and 2 requests; `SPEC_N` beyond); `0` restores the fixed depth and the 3460 TP4 token budget |
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | `2stage` | which CustomAllreduce kernel; the built-in crossover is NVLink-tuned. Inert unless [peer-to-peer](how-to-use.md#pcie-peer-to-peer-optional) is on |
| `GLM5_NCCL_P2P_SYS` | `1` | with peer-to-peer on under TP4, NCCL goes card to card as well (`NCCL_P2P_LEVEL=SYS`); `0` turns that part off |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | allocator compatibility default, independent of layout and peer-to-peer; explicit values are preserved |
| `VLLM_GLM5_REPLICATED_EMBED` | `0` | replicated input-embedding table under TP: saves 2 all-reduces per step for +0.74 GiB per rank (-6% KV tokens); off, because the KV is worth more here |
