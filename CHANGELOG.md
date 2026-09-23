# Changelog

## 1.3.0 — 2026-09-24

**Engine pin moves to `ff4750db5d`** (54 commits on upstream `496c6472cb`).
Installed version `0.29.1rc1.dev573+gff4750db5.precompiled`.

**Optional PCIe peer-to-peer.** Where the driver advertises peer access, the TP
all-reduce can run in device memory rather than staging through the host:
`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE`, with `VLLM_CUSTOM_ALLREDUCE_ALGO` and
the caching-allocator choice tied to the same variable. Worth -5.8% ms/step at
one stream and -10.8% at four, cold prefill unchanged, about 21,500 more KV
tokens. **It ships at 0 here**, because it needs peer-to-peer enabled at the
driver level; at 0 the recipe behaves exactly as 1.2.0 with no driver change.
See "Optional: PCIe peer-to-peer" in the README.

**Prefill overlap beside a live CustomAllreduce.** The overlap no longer stands
down when a CustomAllreduce is present; its split collectives ride NCCL
(`VLLM_GLM5_PREFILL_OVERLAP_BACKEND`, code default `nccl`). That is what
removes the prefill cost the peer-to-peer path used to carry.

**Determinism instruments**, both default 0 and documented as available:
`VLLM_GLM5_TOPK_CANONICAL` and `VLLM_GLM5_DETERMINISTIC_MOE_ALIGN` (0/1/2).

The seven launcher variables are now declared in the engine, so starting the
server no longer prints unknown-variable warnings.

Measured in the shipped configuration, peer-to-peer gate off: 17.03 ms/step at
one stream, 32.26 at four, cold prefill 2,222 tok/s, KV pool 1,160,192 tokens
(4.43x of 262,144), TTFT about 2.6 s at 6.2K and 9.8 s at 23.3K. With the gate
on those become 15.88 and 28.18 ms/step (176.9 tok/s at one stream, 1.83
accepted tokens per step), 2,260 tok/s and 1,181,696 tokens (4.51x).

Long context: needle retrieval 30/30 out to 262K, a verbatim copy at 262K
returned byte-exact, and no content crossing between concurrent requests.

<!-- PENDING 1.3.0: quality rows (perplexity, HumanEval, full GSM8K) -->


## 1.2.0 — 2026-09-22

**Engine pin moves to `434dea1a1b`** (42 commits on upstream `496c6472cb`).
Installed version `0.29.1rc1.dev561+g434dea1a1.precompiled`.

**Correctness.** 64-bit KV row offsets in the sparse-attention kernels, closing
a silent-corruption risk above roughly 4.2M KV tokens; a vocabulary clamp in
three sampler kernels; a 512 MiB transient freed in the indexer's chunk loop.

**Performance.** The sparse-MLA decode schedule is retuned — wider KV tile, two
pipeline stages, head tile sized to the rank — with the old schedule available
as `VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY=1`. A new flag,
`VLLM_GLM5_SHARED_EXPERT_REORDER` (default 1, kill switch 0), submits the MoE
shared experts after the routed dispatch so the two overlap: shared-expert GEMM
time overlapping the routed kernels 0.01% -> 73.3% on a rank-0 decode trace.

Measured against the previous pin with alternating restarts: ms/step at one
stream 17.16 -> 17.01 (-0.85%), at four streams 32.27 -> 32.10 (-0.53%), cold
prefill flat at ~2,238 tok/s, TTFT 2.59 s at 6.2K and 9.74 s at 23K, GSM8K
0.98-1.00, KV pool 1,160,192 tokens (4.43x). A 262,143-token prompt — the
largest a 262,144-token context accepts — is served in 126 s, and a
200,043-token prompt in 94 s. Single-stream decode on this commit's parent
tree averaged 169.5, 167.5 and 172.0 tok/s over three runs; the decode rows in
the README come from a different protocol and are unchanged.


## 1.1.1 — 2026-09-22

Fork repository renamed to `vllm-cmp170hx`; no code change. Every clone URL
and link now names it directly rather than leaning on GitHub's redirect from
the old path. The commit pin is unchanged at `69c33802d0`, so an existing
install is already correct and does not need rebuilding — only a checkout's
`origin` remote is worth repointing.


## 1.1.0 — 2026-09-22

**Engine pin moves to `69c33802d0`.** The fork branch was rebased onto upstream
main (`496c6472cb`), carrying 32 commits, and picked up two drafter
cache-handling fixes along the way. Installed version string is now
`0.29.1rc1.dev551+g69c33802d.precompiled` — taken from a clean install of the
pin, not from a running server, because a server can be ahead of what the fork
publishes.

Validated on four sm_80 cards against the previous pin: ms/step at one stream
17.11 vs 17.61, at four streams 32.04 vs 32.86, cold prefill 2,243 tok/s,
TTFT on a 23,255-token prompt 9.77 s, GSM8K 0.980 at n=50, KV pool 1,160,192
tokens at a 262,144-token context.

`./start.sh install` needs no change for the new pin, but **a pin bump is not
a `git checkout`**. The fork adds no C++ of its own, so the compiled
extensions always come from upstream — but upstream's own ABI moved across
this rebase (`_moe_C::moe_align_block_size` gained a `scatter_idx` argument),
so extensions built for the old base will not run the new Python. `start.sh`
handles this: it records the installed commit in `venv/.recipe-stamp`, so a
changed pin re-runs the install and downloads the extensions for the new base
commit. Updating the checkout without reinstalling fails at engine start with
`expected at most 7 argument(s) but received 8`.


## 1.0.4 — 2026-09-22

Tensor-parallel only. The pipeline-parallel material is replaced by one
statement about link width and a pointer to a recipe that covers it.


## 1.0.3 — 2026-09-22

Renamed the closing README heading to "Source".


## 1.0.2 — 2026-09-22

Title names the model and the cards so the repo can be found by either; the
hook moves to a subtitle under the byline.


## 1.0.1 — 2026-09-22

README leads with what the machine does: the model, the rate and what the cards
cost, with a byline and badges. Documentation trimmed to what the thing is and
what it is worth, with nothing about how it was made.


## 1.0.0 — 2026-09-22

First public cut of the recipe.

**Engine.** Pinned to [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
`ampere-glm53` at commit `cf80da1839`, "[GLM-5.3-Flash] Host-staged all-reduce
for PCIe-only nodes without P2P". Seven sm_80 features, each off by default in
the engine and turned on by `serve.sh`, each with a one-variable kill switch:
Ampere sparse-MLA / indexer / kpool backends, TP prefill comm/compute overlap,
host-staged all-reduce for nodes without peer access, thin-M BF16 GEMM, sm_80
decode kernels, sm_80 prefill kernels, fused decode prologue with
batch-sharded logits, and decode-aware fair chunked prefill. Pipeline
parallelism is enabled in the engine but not supported by this recipe.

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

**Known limits.** Tensor-parallel only. The seven features target TP=4 shapes
and the recipe assumes PCIe Gen2 x16 links between the cards; for a
pipeline-parallel recipe on the same hardware see
[JJ48/glm53-flash-170hx-serving](https://github.com/JJ48/glm53-flash-170hx-serving).