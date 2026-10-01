# How it works

## What makes it fast, and correct

All of this is in the fork. The Ampere backends and the correctness fixes are
always on. Every performance feature can be turned off with a single variable
(see [Engine switches](engine-switches.md)): nearly all ship off in the engine code
and are switched on by this repository's `serve.sh`; the retuned
sparse-attention decode schedule is on in the engine itself.

- **Ampere sparse attention.** GLM-5.3-Flash uses DeepSeek-style sparse
  attention, whose upstream kernels need Hopper. The fork adds an Ampere
  sparse-MLA backend, an Ampere path for the attention indexer and its FP8
  stores, and Ampere versions of the key-pool compression. Without these the
  model does not run on these cards at all.
- **Two layouts, each tuned.** Tensor-parallel 4 splits every layer across the
  cards for the fastest single answer. Pipeline-parallel 4 gives each card a
  quarter of the layers: decoding requests are spread over every in-flight
  micro-batch so no card idles, the hand-off between stages is packed into one
  transfer without a metadata round trip, and each card has the room for about
  twice the KV.
- **Prefill kernels for each layout.** The linear-attention (KDA) prefill, the
  sparse-attention prefill and the MoE prefill each have Ampere kernels for the
  shapes their layout actually runs — 16 heads and quarter-width experts under
  TP4, 64 heads and whole experts under PP4 — and the sparse-attention gather
  no longer reads the same cache row for every empty slot.
- **Fused decode kernels.** The mHC mixing (the model's hyper-connection
  residual streams), MoE routing and block alignment, and the linear-attention
  decode each run as fused kernels built for the small batches decode actually
  sees, in a second generation: the MoE gate, top-k and alignment in one
  launch, a faster mHC decode, KDA decode with its gate projections fused, and
  the indexer's decode glue folded into fewer kernels. They now run in both
  layouts.
- **Tuned small-batch GEMMs.** W4A16 leaves some layers in BF16, where cuBLAS is
  slow with very few rows. A thin-batch kernel with per-shape tuning covers
  batches up to 32 rows.
- **Retuned sparse-attention decode schedule**, and MoE shared experts that
  genuinely overlap the routed experts instead of finishing before them.
- **Host-staged all-reduce.** Stock CMP 170HX cards refuse GPU peer access, so
  every tensor-parallel collective would take NCCL's slow multi-hop path
  through the host. The fork does each small all-reduce in one round trip
  through shared host memory: −7.7% step time at one user and −15% at four when
  it landed.
- **PCIe peer-to-peer all-reduce, optional.** Where the driver does allow peer
  access, the all-reduce runs in device memory instead — see
  [PCIe peer-to-peer](how-to-use.md#pcie-peer-to-peer-optional).
- **Prefill overlap.** During tensor-parallel prefill each layer's all-reduces
  run on a side stream while the MoE computes.
- **Fair prefill.** A long prompt no longer starves users who are mid-answer:
  while anyone is decoding, prefill is taken in small slices. Decode speed
  during someone else's long prompt went from 7% to 18% of normal.
- **Same request, same output — on every install.** A request on its own
  returns the same tokens and the same log-probabilities every time, across
  restarts and now across fresh installs. MoE block alignment runs in a fixed
  order, CUDA-graph padding rows stay out of the MoE, the indexer's top-k is
  consistent on ties and returned in a fixed order, and the linear-attention
  prefill kernels use pinned configurations instead of timing themselves in
  each process. On by default; they cost less than restart-to-restart
  variation.
- **Every custom kernel checked against a high-precision reference.** Each
  kernel the fork adds on the default path is replayed on real inputs captured
  from a running server and compared with a 64-bit reference, side by side
  with the upstream or PyTorch code it replaces, and must be at least as
  accurate.
- **KV figures you can use.** The engine now reserves what prefill chunks in
  flight really hold, so the reported pool is one the server can fill. Right-
  sized workspaces and the drafter's selector tables split across the cards
  return memory to the pool without changing a single output bit.
- **A real correctness fix in the key pool** (since 1.3.0): a rejected draft can
  no longer overwrite the tail of the sparse-attention key pool.
- **64-bit KV row offsets** in the sparse-attention kernels, closing a silent
  corruption risk in very large KV pools, and a vocabulary clamp in the sampler
  kernels.

## What runs

| | |
|---|---|
| API | OpenAI-compatible, `http://127.0.0.1:8000/v1`; this machine only, no API key, unless you [change that](how-to-use.md#serving-other-machines) |
| Model id | `glm-5.3-flash` |
| Weights | [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) — INT4 weights, FP16 activations, group size 128 |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), 320B MoE |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), adaptive depth: up to 7 draft tokens per step at one request, up to 5 at two, 3 under load, following each request's acceptance |
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) `ampere-glm53` @ [`3a2bf16dae`](https://github.com/Morrowmake/vllm-cmp170hx/commit/3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340), on upstream vLLM `e55d076f89` |
| Container image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:6320381b3d0f80ee8a0a36b92013228cc1a7b01ec202030749fab2aa1ad25663` — the engine at that pin, no weights ([docker/](../docker/README.md)) |
| Layout | tensor-parallel 4 (`LAYOUT=tp4`, default; assumes PCIe Gen2 x16) or pipeline-parallel 4 (`LAYOUT=pp4`) — see [Choosing a layout](results.md#choosing-a-layout) |
| Context | 262,144 tokens |
| KV cache | full precision, **not quantised**; measured at 262,144 context in native 1.6.0: TP4/peer-to-peer off 1,072,150 tokens, TP4/peer-to-peer on 1,073,093, PP4 1,914,216 |
| Prefill | TP4 3,456-token chunks, PP4 2,304-token chunks; long prompts yield to running requests ([fair prefill](#what-makes-it-fast-and-correct)) |
| Prefix caching | on |
| Tools and reasoning | `--enable-auto-tool-choice`, glm47 tool-call and reasoning parsers |
| Vision | images and video on |
| Memory target | `--gpu-memory-utilization 0.95` |
