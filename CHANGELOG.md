# Changelog

## 1.0.0 — 2026-09-22

First public cut of the recipe.

**Engine.** Pinned to [Morrowmake/vllm](https://github.com/Morrowmake/vllm)
`ampere-glm53` at commit `cf80da1839`, "[GLM-5.3-Flash] Host-staged all-reduce
for PCIe-only nodes without P2P". Seven sm_80 features, each off by default in
the engine and turned on by `serve.sh`, each with a one-variable kill switch:
Ampere sparse-MLA / indexer / kpool backends, TP prefill comm/compute overlap,
host-staged all-reduce for nodes without peer access, thin-M BF16 GEMM, sm_80
decode kernels, sm_80 prefill kernels, fused decode prologue with
batch-sharded logits, and decode-aware fair chunked prefill. Pipeline
parallelism is enabled but unoptimised.

**Serving.** GLM-5.3-Flash at W4A16 (group size 128) across four GPUs, TP=4,
262,144-token context, DFlash2 speculation at k=3, glm47 reasoning and
tool-call parsers, vision and video on. No FP8, no KV-cache quantisation, no
offload. KV pool 1,158,144 tokens at `--gpu-memory-utilization 0.95`.

**Interface.** One entry point, `./start.sh`, with `install`, `download`,
`stop`, `restart`, `status`, `logs`, `update`, `smoke` and `help`. Every step
is idempotent and skipped when already done. Configuration in `.env`, copied
from `.env.example` on first run; a prefix env assignment overrides it for
every key. `install.sh`, `download.sh` and `stop.sh` are thin wrappers.

**Results.** Decode and cold-prefill measurements taken 2026-09-18 against
MiaAI-Lab's published 2x DGX Spark figures, using their benchmark prompts.

**Known limits.** The PP=4 layout works but is unoptimised: the seven feature
flags were developed and validated under TP=4, and PP=4 was last measured on an
early first-day build with MTP, before any of this. **To be re-measured** on
this pinned tree, with both MTP and DFlash, and the numbers in the README
updated. DFlash under pipeline parallelism has never been tried.
