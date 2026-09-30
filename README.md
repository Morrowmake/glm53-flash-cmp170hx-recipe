<h1 align="center">GLM-5.3-Flash on 4× NVIDIA CMP 170HX</h1>

<p align="center">
  <strong>by <a href="https://x.com/Morrowmake">Morrowmake</a></strong>
  <br><br>
  <a href="https://x.com/Morrowmake"><img alt="Follow on X" src="https://img.shields.io/badge/Follow-%40Morrowmake-000000?style=flat&logo=x&logoColor=white"></a>
  &nbsp;
  <a href="https://github.com/Morrowmake/vllm-cmp170hx/tree/e77f89da2016c3949dba8550c6455b9421ed7365"><img alt="engine" src="https://img.shields.io/badge/engine-vLLM%20fork%20%40%20e77f89da20-4b32c3?style=flat"></a>
  &nbsp;
  <img alt="release" src="https://img.shields.io/badge/release-1.5.0-2ea44f?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/recipe-MIT-blue?style=flat">
</p>

**A 320B-parameter MoE with a 262,144-token context, served on four CMP 170HX
cards in two layouts. Native 1.5.0 results: tensor-parallel at 267.3 tok/s for one
user and 758.9 tok/s across eight, or pipeline-parallel with
6,575 tok/s cold prefill and a 2,334,498-token KV pool. The weights
are W4A16 and nothing else is cut: the KV cache is full precision, there is no
FP8 anywhere, and nothing is offloaded to CPU or disk. Single-request
outputs were identical across the repeated checks reported below.**

This repository installs and runs **GLM-5.3-Flash** on **four NVIDIA CMP 170HX
cards** behind an OpenAI-compatible API, with tool calls, reasoning, images and
video, and a **DFlash2** speculative drafter. Upstream vLLM's sparse-attention
path needs a Hopper GPU; the CMP 170HX is Ampere (sm_80). Our
[vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere-glm53) adds
the Ampere kernels that make the model run at all, then spends the rest of its
patches on making it fast and making it repeatable. It ships as a container
image, so the engine and its Python environment stay out of your system.

> **Which setups this release is for.** Two layouts, one switch
> (`LAYOUT=tp4` or `LAYOUT=pp4`):
>
> - **Tensor-parallel 4** (the default) is optimised for **PCIe x16 links** —
>   cards with the x16 capacitor modification. It gives each request the
>   fastest answer. On cards limited to x4 links it is bus-bound and much
>   slower.
> - **Pipeline-parallel 4** passes only activations between the cards, so it
>   needs far less link bandwidth; it is the layout **for cards limited to x4
>   links**, and for many parallel users and long prompts. Its numbers on this
>   page were **measured on our x16 cards**; we have not yet measured it on x4
>   links.
>
> `./start.sh` checks your link width and suggests `LAYOUT=pp4` if a card is
> narrower than x16.

One command sets it up and starts it (needs Docker with the NVIDIA Container
Toolkit, see [What you need](#what-you-need)):

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe
./start.sh                    # tensor-parallel 4
LAYOUT=pp4 ./start.sh restart # or pipeline-parallel 4
```

What changed in this release is in [CHANGELOG.md](CHANGELOG.md).

---

## Results

### Optional Marlin and image packaging (1.5.0)

Release 1.5.0 adds optional compiled Marlin and a Git-free runtime image.
The throughput tables below were remeasured on the native 1.5.0 engine with
the optional library installed. Quality results retain their explicitly named
release scope; throughput measurements are not quality or universal exactness claims.
See [Optional compiled Marlin](#optional-compiled-marlin) for defaults and limits.

### Allocator compatibility default (1.4.3)

With DFlash2 k=3 and a 262,144-token context, the new
`expandable_segments:False` default measured **1,176,646 KV tokens under
TP4/peer-to-peer off** and **2,334,498 under PP4**: respectively +20,011 tokens
(+20 physical blocks) and +14,170 (+4) against True on the same engine.
These are not peer-to-peer gains. Explicit allocator overrides remain intact.

Four alternating True/False starts per layout passed startup checks, short/long
decode-drift and KL-divergence comparisons, and single-request repeatability,
with no new GPU Xid errors. TP4 paired decode-step cost was +0.38–0.83% at
1/4/6/8 users; PP4 paired means at 1/4/8 differed by at most 0.21%. PP4 at six
users was scheduling-confounded; no speed conclusion is drawn there.
Fixed-batch repeatability also passed under TP4 and PP4 in both eager and
CUDA-graph modes. Native update checks passed with `.env` preserved and no
engine reinstall. Validation for this change is DFlash-only: MTP/no-drafter
and conflicting allocator aliases were not GPU-validated.
It does not establish a root-cause fix for driver or virtualisation failures.

### Quality (1.4.3)

Full native fixed-batch results with DFlash2 k=3, repeated twice in each configuration:

| Layout | HumanEval, pass@1 (164 tasks) | GSM8K (1,319 problems) |
|---|---:|---:|
| TP4, peer-to-peer off | **162/164 — 98.78%** | **1,281/1,319 — 97.12%** |
| PP4, peer-to-peer off | **163/164 — 99.39%** | **1,284/1,319 — 97.35%** |

The 1.4.0 baseline and 1.4.3 configuration matched exactly on generated token
IDs, answers and scores in both full repeats of each same-layout comparison.
The PP4 comparisons also isolated the engine update, draft-tail enablement
and allocator default: all were exact, with no per-task gains or losses.
This is evidence for those measured paths, not universal equivalence across
layouts, batches or workloads.

These results use fixed-order batches of up to eight, identical prompts and
generation settings within each comparison, and a corrected HumanEval scorer
that executes the extracted completion with its supplied task context.
HumanEval allows 4,096 tokens per reply; GSM8K allows 3,072. Length-capped
outputs remain in the scores. These controlled results supersede the earlier
unmatched quality figures; those separate runs are not release-change controls.

**Rolling concurrency, separately measured:** on the unchanged 1.4.0
TP4/peer-to-peer-on server, HumanEval scored **162/164 twice serially** and
**163/164, 162/164 and 162/164** with up to eight continuously replenished
requests in original, shuffled and reversed order. There was no demonstrated
aggregate accuracy loss, but individual tasks both improved and worsened.
This is not a 1.4.3 or PP4 comparison, nor proof of batch invariance.

### Release 1.5.0 throughput

Measured on 2026-09-30 using the original throughput workloads and the released
engine and launcher. These are native measurements, not a new container-speed
comparison. The optional library was installed: compiled decode uses its release
defaults (off for TP4, on for PP4), compiled prefill is off, and
`expandable_segments:False` applies to all three columns. Native installations
without the optional library do not use the measured PP4 compiled-decode path.

DFlash2 at k=3, 262,144-token context, the defaults of release 1.5.0. Three
columns: tensor-parallel 4 with the cards talking through the host (the
default), tensor-parallel 4 with the optional
[PCIe peer-to-peer](#pcie-peer-to-peer-optional) path, and pipeline-parallel 4
(`LAYOUT=pp4`, peer-to-peer off: the setup of a stock driver on x4 cards,
which is what this layout is for).

| | TP4, peer-to-peer off (default) | TP4, peer-to-peer on (optional) | PP4, peer-to-peer off (`LAYOUT=pp4`) |
|---|---:|---:|---:|
| Streaming decode, 1 user, structured / code / prose | **267.3 / 260.9 / 186.0 tok/s** | **282.0 / 274.0 / 198.5 tok/s** | **141.8 / 139.8 / 103.5 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **758.9 / 674.1 / 531.1 tok/s** | **816.8 / 741.1 / 563.7 tok/s** | **603.0 / 577.2 / 445.1 tok/s** |
| Cold prefill | **2,657 tok/s** | **3,053 tok/s** | **6,575 tok/s** |
| One-token response time, 6,217 / 23,255-token prompt | 2.39 / 8.73 s | 2.05 / 7.52 s | 1.46 / 3.64 s |
| KV pool at 262,144 context | 1,176,646 tokens (4.49 full-length requests) | 1,177,646 tokens (4.49) | 2,334,498 tokens (8.91) |

All at 180 W per card (a power limit we set on our cards; the scripts never
change power, clock or fan settings), PCIe x16 links, one server start per
column. Decode uses three fixed prompt types—structured, code and prose—with
a 400-token cap, temperature 0 and median of five runs. Single-user tok/s
measures streaming decode after the first token. Eight-user aggregate is actual
completion tokens divided by concurrent batch wall time, including prefill and
client overhead; it is not the per-stream rate multiplied by eight.

Two of the 855 measured decode requests stopped at 298 tokens rather than the
400-token cap: one structured eight-user request under TP4/P2P-on and one under
PP4. Both began with instruction-analysis prose, misinterpreting the original
`(stream 4/8)` prompt suffix as splitting the counting task. They remain included
using their actual token counts, with no selective rerun or exclusion. These are
throughput observations, not evidence of output correctness or a quality comparison.

Cold prefill is the median of nine rates on the same three real-text prompts
of 23,945, 34,299 and 37,905 tokens, with zero observed prefix-cache hits.
The one-token response row retains the original nonstreaming elapsed-time
measurement (median of three); it includes response handling and is distinct
from streamed time to first token. The KV pool accounts for prefill chunks in
flight (see [Honest KV figures](#measured-while-building-this-release)).

For historical reference, PP4 with peer-to-peer on was measured in 1.4.0 at
141.0 / 139.4 / 105.1 tok/s for one user, 532.7 / 493.9 / 377.8 across eight,
6,606 tok/s cold prefill. That fourth configuration was not remeasured here.
Older release comparisons below retain their original scope.

Historical perplexity on the fixed 60-document set was **3.2781 under TP4**
and **3.2735 under PP4** (1.4.0); it was not remeasured for the allocator change.
Current HumanEval and GSM8K results are in [Quality (1.4.3)](#quality-143).

### Choosing a layout

Prefill and KV figures compare the native 1.5.0 peer-to-peer-off columns above.

| | Tensor-parallel 4 (`tp4`, default) | Pipeline-parallel 4 (`pp4`) |
|---|---|---|
| Each card holds | a quarter of every layer | a quarter of the layers |
| Best for | one or two interactive users: the fastest answer per request | many parallel users or clients, long prompts, large shared contexts |
| Prefill | 2,657 tok/s | 6,575 tok/s (2.47×) |
| KV pool | 1,176,646 tokens | 2,334,498 tokens (1.98×) |
| Traffic between cards | ~9.4 MB per layer during prefill, ~100 small collectives per decode step | activations only, once per stage |
| Links | PCIe x16 | built for x4; measured on x16 |

Both run the same model, the same drafter and the same repeatable-output fixes;
switching is `LAYOUT=pp4 ./start.sh restart` and back with `LAYOUT=tp4`.

### Measured while building this release

These are historical 1.4.0/1.4.1 measurements on the same four cards, not
remeasurements of the new allocator default. They are not the release table
above; each line says what it was measured against.

- **Pipeline-parallel prefill +19.5%.** New prefill kernels for 64 heads and
  whole experts — the linear-attention (KDA) prefill, the sparse-attention
  prefill and the MoE prefill — took PP4 cold prefill from 5,520 to 6,586–6,613
  tok/s, and time to first token on a 23,255-token prompt from 4.27 to 3.64 s,
  with decode unchanged. Each kernel is at least as accurate as the code it
  replaces against a 64-bit reference on real inputs.
- **A faster pipeline.** Decodes spread over every in-flight micro-batch and a
  leaner hand-off between stages: +7.3% aggregate decode at four users and
  +2.0% cold prefill under PP4, outputs identical apart from the drafter's
  folded input projection, which is checked against a 64-bit reference.
- **Tensor-parallel prefill kernels.** The linear-attention prefill runs 1.42×
  faster per prompt per card and the MoE prefill 1.24×, as accurate as before
  against a 64-bit reference.
- **Against release 1.3.0**, tensor-parallel 4: cold prefill 2,484 → 2,670
  tok/s (+7.5%) with peer-to-peer off and 2,490 → 3,062 (+23%) with it on,
  where NCCL now also runs card to card (worth +13.7% on its own, outputs
  identical).
  Eight-user aggregate 745.4 → 763.1 tok/s structured (+2.4%) off and
  808.1 → 848.3 (+5.0%) on; one user unchanged (264.6 → 264.4 structured).
- **Honest KV figures.** With prefill chunks in flight, a request can hold more
  linear-attention state than the engine used to reserve for it, so the KV pool
  it reported was larger than the server could actually fill: under TP4,
  release 1.3.x's figure was 17,932 tokens (1.5%) too high. Near a full pool
  that could only mean a request being paused and resumed, never wrong output,
  but the published number should be one you can use. The reserve now
  matches, and every KV figure on this page is the corrected one.
- **The same output on every install.** The linear-attention prefill kernels
  used to pick their tile configuration by timing the options in each new
  process, so a fresh install or a cleared cache could compute in a different
  order from ours. They now use pinned configurations, exactly as accurate as
  the tuned ones against a 64-bit reference (1.000×) and within 0.03% of their
  speed on one card.
- **Container at native speed.** Release 1.3.x's image ran within 0.4% of the
  native install on decode at 1 / 4 / 8 users and on cold prefill.
  On the same four cards at 180 W with peer-to-peer off, the image runs at the native install's speed: 15.60 / 29.16 / 43.99 ms per decode step at 1 / 4 / 8 users (native 15.87 / 29.51 / 44.45) and 2,672 tokens/s cold prefill (native 2,670).
- **A newer upstream.** The fork now sits on upstream vLLM's 0.30.1 development
  line (FlashInfer 0.7.0).

### Decode and prefill in detail

Release 1.5.0, native tensor-parallel 4 with peer-to-peer off and the
configuration described above: temperature 0, median of five, original
structured/code/prose prompts and cache-busting nonces.

**Decode**, 400 max tokens. `Stream` is per request,
`(completion_tokens − 1) / (end − first token)`; concurrent aggregate is
`sum(completion_tokens) / batch wall`, including prefill and client overhead.
At one user the original client's aggregate field duplicates its streaming
rate, so it is omitted here rather than presented as end-to-end throughput.
Each request carries a unique nonce to prevent prefix-cache reuse.

| Prompt type | Users | Stream tok/s | Aggregate tok/s | TTFT (cold) |
|---|---:|---:|---:|---:|
| Structured (count 1→200) | ×1 | **267.3** | — | 0.063 s |
|  | ×2 | **209.7** | **393.1** | 0.109 s |
|  | ×4 | **149.5** | **533.2** | 0.247 s |
|  | ×8 | **105.7** | **758.9** | 0.312 s |
| Code (clamp_00…clamp_49) | ×1 | **260.9** | — | 0.158 s |
|  | ×2 | **183.4** | **324.8** | 0.248 s |
|  | ×4 | **140.9** | **489.9** | 0.392 s |
|  | ×8 | **98.6** | **674.1** | 0.617 s |
| Prose (hash map) | ×1 | **186.0** | — | 0.067 s |
|  | ×2 | **136.8** | **260.5** | 0.158 s |
|  | ×4 | **106.1** | **392.0** | 0.253 s |
|  | ×8 | **73.1** | **531.1** | 0.311 s |

Prose decodes slower than structured or code text because the drafter's guesses
are accepted less often.

**Cold prefill by prompt length**, native release 1.5.0 pipeline-parallel 4
(peer-to-peer off), original real-text corpus and unique uncached windows,
`max_tokens=1`, median of two, `prompt tokens / streamed TTFT` measured client
side. All twelve requests recorded zero prefix-cache hits. (Tensor-parallel 4
runs at 2,657 tok/s on the separate 24K–38K prompts of the results table.)

| Prompt | TTFT | tok/s |
|---:|---:|---:|
| ~8k | 1.91 s | **4,171** |
| ~16k | 2.78 s | **5,738** |
| ~32k | 4.83 s | **6,616** |
| ~64k | 9.04 s | **7,082** |
| ~128k | 17.78 s | **7,177** |
| ~250k | 35.38 s | **7,074** |

---

## Optional compiled Marlin

The launcher selects TP4 compiled decode off, PP4 compiled decode on
when the optional library is installed, and compiled prefill off in both layouts.
The effective `PP=4 TP=1` configuration selects the PP4 default even when those
dimensions override `LAYOUT`. One `vllm._ampere_marlin_C` library serves both
layouts; changing layout never rebuilds it. The independent switches override
these defaults explicitly:

```bash
VLLM_GLM5_MARLIN_DECODE_CUDA=1 ./start.sh restart
VLLM_GLM5_MARLIN_PREFILL_CUDA=1 ./start.sh restart
```

An explicit `0` or `1` in the environment or `.env` remains authoritative across
updates. Leave the flags commented out to follow the defaults. Unset PP4 decode
uses CPU-only module discovery, without importing the extension or probing CUDA.
If the library is absent it defaults off with a banner; if present it defaults on
and startup must validate compatibility. Explicit `1` with an absent or incompatible
library, or default-on with an incompatible library, fails before serving, never
silently disabling the requested feature. Both flags off means no extension load.
`DRY=1` prints the resolved flags and command but skips compatibility validation.

The compiled decode path is limited to eligible small batches in TP4 and PP4.
The compiled prefill path is PP4-only, within the engine's validated shape and
token bounds; TP4 prefill retains the released implementation. Unsupported
shapes use the released paths. These are not universal kernel replacements,
and no whole-server speedup is claimed here.

The pinned container image includes the optional library. To build it natively,
use `RUNTIME=native VLLM_BUILD_AMPERE_MARLIN=1 ./start.sh install`.
Native compilation is opt-in; it is independent of runtime enablement.

Normal native installs use the precompiled base engine without compiling this
library. Opting in requires a CUDA toolkit (`CUDA_HOME`) and C++ compiler.
`install` and `update` add the library even at an unchanged engine pin. A separate
stamp includes source, requirements, Python/torch ABI, compiler/toolkit versions
and binary digest; matching builds are reused. Engine reinstalls invalidate the
optional binary, without demanding its toolkit when build is off. Failed rebuilds
cannot leave a stale library in service. Enabling a runtime switch without the
library fails before serving, never silently falling back. Both switches off
means no optional-extension load. Existing `.env` and explicit pin overrides are preserved.

The [container build](docker/README.md#build-it) prebuilds the same library with
no visible GPUs. Marlin needs no startup compiler or layout-specific rebuild;
other engine kernels still need the image's existing toolkit.

## What runs

| | |
|---|---|
| API | OpenAI-compatible, `http://127.0.0.1:8000/v1`; this machine only, no API key, unless you [change that](#serving-other-machines) |
| Model id | `glm-5.3-flash` |
| Weights | [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) — INT4 weights, FP16 activations, group size 128 |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), 320B MoE |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), 3 draft tokens per step |
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) `ampere-glm53` @ [`e77f89da20`](https://github.com/Morrowmake/vllm-cmp170hx/commit/e77f89da2016c3949dba8550c6455b9421ed7365), on upstream vLLM `e55d076f89` |
| Container image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:6320381b3d0f80ee8a0a36b92013228cc1a7b01ec202030749fab2aa1ad25663` — the engine at that pin, no weights ([docker/](docker/README.md)) |
| Layout | tensor-parallel 4 (`LAYOUT=tp4`, default; assumes PCIe Gen2 x16) or pipeline-parallel 4 (`LAYOUT=pp4`) — see [Choosing a layout](#choosing-a-layout) |
| Context | 262,144 tokens |
| KV cache | full precision, **not quantised**; measured with the False allocator default at 262,144 context in native 1.5.0: TP4/peer-to-peer off 1,176,646 tokens, TP4/peer-to-peer on 1,177,646, PP4 2,334,498 |
| Prefill | TP4 3,456-token chunks, PP4 2,304-token chunks; long prompts yield to running requests ([fair prefill](#what-makes-it-fast-and-correct)) |
| Prefix caching | on |
| Tools and reasoning | `--enable-auto-tool-choice`, glm47 tool-call and reasoning parsers |
| Vision | images and video on |
| Memory target | `--gpu-memory-utilization 0.95` |

---

## What makes it fast, and correct

All of this is in the fork. The Ampere backends and the correctness fixes are
always on. Every performance feature can be turned off with a single variable
(see [Kill switches](#kill-switches)): nearly all ship off in the engine code
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
  [PCIe peer-to-peer](#pcie-peer-to-peer-optional).
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

---

## How to use this repo

### What you need

| | |
|---|---|
| GPUs | 4× NVIDIA CMP 170HX, **each exposing 64 GiB** (`nvidia-smi` shows 65,536 MiB). For tensor-parallel 4, **PCIe Gen2 x16 links**; pipeline-parallel 4 is built for narrower links. Stock cards expose less memory and run at x4; getting to 64 GiB and x16 is outside this repository. The preflight stops before any download if a card reports under 60 GiB, and warns below x16 ([Link width](#link-width)) |
| OS and driver | Linux (the commands below are for Ubuntu) with NVIDIA driver **580 or newer**; `nvidia-smi` must list all four cards |
| Container runtime | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), usable by your user, so that `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi` lists all four cards. Not needed for the [native install](#native-install-for-developers) |
| Tools | `git`, `curl`, `flock` and `setsid` (util-linux, on most systems already), `jq` for the smoke test |
| Disk | about 180 GiB (193 GB) for the two checkpoints, plus the engine image (about 10.6 GB compressed) and the kernel compile caches |

### Step by step

`./start.sh` on its own does steps 1–3 in order and skips anything already
done, so running it twice is safe. The same steps one at a time:

```bash
./install.sh       # 1. pull the engine image, pinned by digest (about 10.6 GB compressed)
./download.sh      # 2. fetch the model (~178 GiB) and the drafter (~2.2 GiB) into ./models
./start.sh         # 3. start the container, wait for /health, print the KV pool size
./start.sh smoke   # 4. one chat request and one tool call against the running server
./start.sh stop    # stop it; the image, weights and caches stay, so the next start is quick
```

The container runs the engine image with this repository's `serve.sh` as its
entry point, as your user rather than root, with the checkpoints mounted
read-only and the kernel compile caches in `./cache` (owned by you). It
publishes the API on `127.0.0.1:8000` only. Its output goes to
`logs/serve.log`, as a native start's does, and `./start.sh stop` stops only the
container this checkout started. The first boot builds the compile caches and
takes about 4–5 minutes longer than later ones (below); later boots take about
3 minutes under TP4 and 2 under PP4 (weight loading and CUDA-graph capture).

**The first boot after an install or an update is slower.** FlashInfer 0.7.0
compiles two of its kernel modules (top-k, about 160 s, and sampling, about
60 s) into the empty cache, which adds about 4–5 minutes. Later boots reuse
them. While that build runs, the log can show lines like
`No available shared memory broadcast block found in 60 seconds`; they are
harmless and stop once the build finishes.

**Historical smoke test output** from the published release (tensor-parallel 4,
peer-to-peer off, earlier True allocator default; current KV differs):

```
==> waiting for http://127.0.0.1:8000/health (up to 900s)
    healthy
    served models: glm-5.3-flash

==> chat request
    reply: The user is asking about PCIe peer-to-peer (P2P) and its relevance to tensor parallelism. Let me think about what I know here.  **Tensor parallelism basics:** T ...
    usage: prompt=28 completion=200 wall=1.27s  ->  156.9 tok/s

==> tool-call request
    tool_call: get_weather({"city": "Reykjavik", "unit": "celsius"})
    usage: prompt=199 completion=66 wall=0.45s  ->  147.8 tok/s

==> KV cache
    GPU KV cache size: 1,156,635 tokens, Maximum concurrency for 262,144 tokens per request: 4.41x

smoke: ok
```

The rates in the smoke test include the time to first token of a short
request, so they read lower than the decode table.

### Choose the layout

```bash
LAYOUT=pp4 ./start.sh restart     # pipeline-parallel 4, this run
# or set LAYOUT=pp4 in .env and ./start.sh restart to keep it
```

Everything else — the drafter, the context, the kill switches, the API — works
the same in both layouts. `./start.sh status` shows which one is running.

### Talk to it

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-flash",
       "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
       "max_tokens": 400}'
```

Any OpenAI-compatible client works with base URL `http://127.0.0.1:8000/v1`
and model `glm-5.3-flash`. The model's reasoning comes back in the `reasoning`
field of the message and the answer in `content`. `reasoning_effort` of `low`,
`high` or `max` sets how much it thinks. Do not send `reasoning_effort: "none"`
or `"chat_template_kwargs": {"enable_thinking": false}`: the model still thinks,
and its reasoning then lands inside `content`.

### Serving other machines

By default the API listens on `127.0.0.1` only, so nothing outside this
machine can reach it, and it has **no API key**. To serve other machines, set
both in `.env` and restart:

```bash
HOST=0.0.0.0          # every interface; or one of this machine's addresses
API_KEY=<a long random string>
```

With `API_KEY` set, every `/v1` request must send
`Authorization: Bearer <key>` (OpenAI clients: pass it as the API key);
`./start.sh smoke` and `status` send it for you. Without `API_KEY`, anyone who
can reach the port can use the model. The key is handed to vLLM through its
`VLLM_API_KEY` variable, so it does not appear in the process list.

### Settings

Everything lives in `.env`, copied from [`.env.example`](.env.example) on first
run. A value on the command line beats `.env` for that run:

```bash
MAX_LEN=131072 ./start.sh restart
```

| Key | Default | What it does |
|---|---|---|
| `LAYOUT` | `tp4` | `tp4` tensor-parallel 4, `pp4` pipeline-parallel 4 ([Choosing a layout](#choosing-a-layout)) |
| `RUNTIME` | `container` | `native` runs the engine from a venv instead ([Native install](#native-install-for-developers)); a checkout that already has a native install keeps it |
| `MAX_LEN` | `262144` | context ceiling; lower it for more concurrent full-length requests |
| `MAX_SEQS` | `8` | concurrent requests |
| `GPU_UTIL` | `0.95` | share of each card's memory the engine may use |
| `SPEC_MODE` / `SPEC_N` | `dflash` / `3` | speculative drafter and draft depth; `mtp` uses the MTP head in the model checkpoint, `none` turns speculation off |
| `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE` | `0` | the [peer-to-peer](#pcie-peer-to-peer-optional) switch (TP4) |
| `MAX_BATCHED` | `3460` (TP4), `2312` (PP4) | sets the prefill chunk: 3,456 / 2,304 tokens; `2048` gives 1,152 |
| `FAIR_PREFILL` / `FAIR_CHUNK` | `1` / `384` | fair prefill and its slice size while others decode |
| `PREFILL_CAP` | `0` | upstream's unconditional chunk cap. **Leave it 0**: it cost 15% prefill here and turns the prefill features off |
| `MM_CAP` | `0` | `1` bounds image and video inputs, which returns roughly 150k KV tokens |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | where it listens; see [Serving other machines](#serving-other-machines) |
| `API_KEY` | unset | bearer key required on `/v1`; unset means no key |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | the model id clients send |
| `MODELS_DIR` | `models/` in this repo | where the checkpoints go; give an absolute path |
| `IMAGE` | this release's image, by digest | the engine image (container) |
| `CONTAINER_NAME` | `glm53-flash` | the container's name |
| `READY_TIMEOUT` | `1800` | seconds `./start.sh` waits for `/health` |
| `EXTRA_ARGS` | unset | appended to the `vllm serve` command line |
| `HF_TOKEN` | unset | a Hugging Face token makes the download faster |

Every setting in `.env.example` is commented out and shows its default;
uncommenting one overrides that default. A `.env` you have not edited
therefore picks up a later release's new defaults without changes. To move
to a later release, run `./start.sh update`.

Write each setting as `KEY=value` on its own line. In an unquoted value,
anything after a space and `#` is a note and is ignored; quote the value
(`"..."`) to keep it exactly as written.

### PCIe peer-to-peer (optional)

**What it is.** Under tensor-parallel 4 the four cards exchange data after
every layer. By default that goes through host memory, because a stock CMP
170HX refuses GPU peer access: `nvidia-smi topo -p2p r` answers `GNS` on every
pair. Where peer access *is* available, the all-reduce can run card to card in
device memory instead, which is faster.

**The default is off.** With `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0` the recipe
needs no driver change of any kind and uses the host-staged path described
above. If you do nothing, this is what you run. Under pipeline-parallel 4 the
switch does nothing: the hand-off between stages uses peer-to-peer by itself
where the driver offers it.

**Who should turn it on.** Only people who already have peer-to-peer working on
their cards. It has to be advertised by the driver. We use a build of the
[cmpunlocker](https://github.com/asm64-hooligan/cmpunlocker) project, set up
with that project's own installer (its `install.sh --p2p`, not this
repository's `install.sh`, which has no such option). How to install it is
covered by that project's documentation, not here; none of it is ours.

It is **topology-dependent**. We verified it on our machine: four cards on
EPYC root ports, all pairs.
[bayley/cmpunlocker](https://github.com/bayley/cmpunlocker) reports the mailbox
path dead behind PLX switches on a Xeon, so a different board may simply not
have it.

**Historical gains.** With the earlier allocator defaults, the first two columns
of the published 1.4.1 [results table](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/blob/v1.4.1/README.md#results): step time −3.7% at one user, −4.3% at four,
−7.6% at six and −9.1% at eight; single-user decode +4.3% to +7.5%; eight-user
aggregate +6.2% to +11.3%; cold prefill +14.7% (2,670 → 3,062 tok/s); and
21,011 more KV tokens. That comparison changed both P2P and allocator mode;
the KV delta is not a P2P gain with the new independent False default.

**Turn it on.**

```bash
nvidia-smi topo -p2p r              # every pair must say OK, not GNS
# then set VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1 in .env, or for one run:
VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1 ./start.sh restart
```

`serve.sh` selects the `2stage` all-reduce kernel (upstream's default
crossover is tuned for NVLink). With peer-to-peer on under TP4, it also sets
NCCL's peer-to-peer level (`NCCL_P2P_LEVEL=SYS`), so the large prefill
collectives go card to card: historically +13.7% cold prefill with identical
outputs, measured before this allocator-default change.
`GLM5_NCCL_P2P_SYS=0` turns that part off.

The allocator compatibility default is `expandable_segments:False`, independent
of layout and peer-to-peer. Explicit `PYTORCH_CUDA_ALLOC_CONF` values, including
empty or composed settings, are preserved; the startup allocator banner reports
both allocator variable names, not inferred precedence. `PYTORCH_ALLOC_CONF`
is inherited for native launches but is not forwarded into the container;
conflicting aliases are not covered by the legacy-variable default. This is a
compatibility default, not a fix for an underlying driver or virtualisation bug.

**Check it works.**

```bash
grep "all-reduce backends" logs/serve.log | grep "tp:0" | tail -1
```

`logs/serve.log` keeps every start, so read the last line. With peer-to-peer
on, the list starts with `CUSTOM`: `['CUSTOM', 'PYNCCL']`.
With it off, or if the driver does not actually grant peer access, it reads
`['HOSTSHM', 'PYNCCL']` — the engine falls back to the host-staged path on its
own rather than failing. Then run `./start.sh smoke`.

**Turn it off.** Set `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0` in `.env` (or
delete the line) and `./start.sh restart`. That restores the host-staged path;
it does not change the allocator configuration.

### Kill switches

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

### Day to day

| Command | |
|---|---|
| `./start.sh` | preflight → install → download → launch → wait for `/health` |
| `./start.sh restart` | stop, then start (picks up `.env` changes) |
| `./start.sh status` | runtime and layout, container, `/health`, KV line, install and checkpoint state |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh smoke` | one chat request and one tool call |
| `./start.sh stop` | stop the server this checkout started |
| `./start.sh update` | `git pull`, pull the new engine if the pin moved, restart |
| `IMAGE=<earlier image> ./start.sh update` | roll back to an earlier engine, for that run only |
| `DRY=1 ./start.sh` | print what would be pulled, downloaded and launched, without doing it |

`stop` only stops the container recorded in `logs/container.id`, after
confirming it carries this checkout's label; it never matches by name alone,
so another container or vLLM on the same machine is never touched.

The engine pin lives in `start.sh`, so a `git pull` can move it, and `start.sh`
pulls the matching image whenever it changes. Uncomment `IMAGE` in `.env` to
freeze it.

**Updating from 1.4.x.** Run `./start.sh update`. Release 1.5.0 moves the
engine and image pins; your `.env`, including explicit `IMAGE`, `VLLM_COMMIT`
and runtime-switch overrides, is preserved. Remove an old explicit pin only
if you want to follow this release. The image includes optional Marlin:
unset PP4 decode enables it, TP4 decode and both prefill defaults remain off.
Native users must opt in to compilation as described above; without an installed
library, unset PP4 decode remains off. The allocator default remains
`expandable_segments:False`, with explicit allocator overrides preserved.

**Updating from 1.3.x.** Run `./start.sh update`, nothing else. What happens:

1. It pulls this release and restarts into its `start.sh`.
2. Your checkout already has a native install, so it **stays native** (the
   container is the default only for fresh checkouts).
3. The engine pin moved to a new upstream base, so `start.sh` **rebuilds the
   venv** from scratch for the new engine and its own dependency pins (a few
   minutes; the checkpoints are not touched).
4. It boots the server. **The first boot is about 4–5 minutes slower** than
   later ones while FlashInfer compiles two kernel modules; the log may show
   `No available shared memory broadcast block found in 60 seconds` lines
   meanwhile, which are harmless.
5. The layout stays tensor-parallel 4 and peer-to-peer stays as you had it.
   With the new False allocator default, TP4/peer-to-peer off measured
   1,176,646 KV tokens. Explicit allocator overrides are preserved.

**Moving a native install to the container.** Install Docker and the NVIDIA
Container Toolkit ([What you need](#what-you-need)), then set
`RUNTIME=container` in `.env` and run `./start.sh restart`: it stops the native
server, pulls the image and starts the container on the same checkpoints.
`RUNTIME=native` and another restart go back. The venv stays on disk until you
delete `venv/` and `vllm-src/` yourself.

**Rolling back.** `IMAGE=<image> ./start.sh update` rolls the engine back for
that run only; the next plain `./start.sh` or `restart` goes back to this
release's image. To stay rolled back, set `IMAGE=<image>` in `.env`. Release
1.3.x's image is
`ghcr.io/morrowmake/vllm-cmp170hx@sha256:ee978fb3e3d11cf8577a014539a7ad4e2a8dff95163fc5d96c0a06fcf9c64640`.
That changes the engine only; to go back to an earlier release's scripts and
defaults as well, check out its tag (`git checkout v1.3.1`, then
`./start.sh restart`; back again with `git checkout main` and
`./start.sh update`). Releases before 1.4.0 install natively.

**Older releases keep installing.** Each release pins the fork by commit, and
the fork's branch moves to a newer upstream from time to time. Every commit a
release has pinned is kept on the fork under a tag, `glm53-recipe-<version>`,
so an older release — or a rollback to its engine — still finds its commit
after the branch has moved on. Engine pins from 1.1.0 on can be installed;
1.0.x's pin predates the first rebase of the fork branch and cannot.

### Native install (for developers)

`RUNTIME=native` builds the engine on this machine instead: a venv with the
fork checked out at the pin, installed editable, so you can change the engine's
Python and restart. A checkout that already has a native install from an
earlier release keeps using it. It needs, in addition to the above:

| | |
|---|---|
| CUDA | a 13.x toolkit at `/usr/local/cuda-13.3`, or set `CUDA_HOME` |
| Tools | `uv`, Python 3.12, `wget` for the CUDA commands |
| Disk | about 20 GiB for the venv, the engine source and compile caches |

```bash
RUNTIME=native ./install.sh    # build ./venv and the pinned vLLM fork (about six minutes)
RUNTIME=native ./start.sh      # or set RUNTIME=native in .env
```

`./serve.sh` runs a native server in the foreground instead if you prefer; it
reads its settings from the environment, not from `.env`. A server started
that way is invisible to `./start.sh status`, `stop` and `restart`: stop it with
Ctrl-C before using `./start.sh` again.

Rolling back a native install works the same way with the engine commit:
`VLLM_COMMIT=<sha> ./start.sh update` for one run, or `VLLM_COMMIT=<sha>` in
`.env` to stay there. When the pin moves to a new upstream base, `start.sh`
rebuilds the venv, because the engine's own dependency pins change with it.

**CUDA.** You need a 13.x toolkit. These commands are for Ubuntu. NVIDIA had no
working `ubuntu2604` repository index when we built this, so we used the
`ubuntu2404` one:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3
```

The install uses upstream's **precompiled** base CUDA extensions, which already
carry sm_80 code. Those base extensions remain usable: the optional Marlin
CUDA/C++ extension is distinct and is compiled only with
`VLLM_BUILD_AMPERE_MARLIN=1`. To compile the base extensions yourself:
`BUILD_FROM_SOURCE=1 MAX_JOBS=16 ./install.sh` (full toolkit, ~60 GB of
scratch, one to two hours).

### Link width

Tensor parallelism moves about 9.4 MB per layer between the cards during
prefill and roughly 100 small collectives per decode step. It assumes PCIe Gen2
x16 links; on x4 links it is bus-bound and far slower than the numbers above.
Pipeline parallelism passes only each stage's activations to the next card,
which is why it is the layout for narrower links. `./start.sh` prints each
card's link width in its preflight and warns if any is narrower than x16 while
the layout is tensor-parallel.

### Run the container by hand

[`docker/`](docker/README.md) has the Dockerfile and the build script for the
image, and a `docker run` command with [`container.env`](docker/container.env)
for running it without `./start.sh`.

---

## Status and roadmap

- **Two layouts in 1.4.0.** Tensor-parallel 4 for the fastest single answer on
  x16 links; pipeline-parallel 4 for many parallel users, long prompts and
  cards limited to x4 links.
- **Pipeline-parallel on x4 links** is next to be measured: every PP4 number on
  this page comes from x16 links.
- **Optimisation continues** on both layouts; every change ships with a kill
  switch and measured numbers.

---

## Known limits

- **Ampere only.** Every kernel in the fork is written for sm_80. On newer GPUs
  upstream vLLM's own kernels are better, and forcing these features on
  elsewhere is untested.
- **Wide links for tensor-parallel.** See [Link width](#link-width).
- **Pipeline-parallel is measured on x16 links.** It needs far less link
  bandwidth by design, but its numbers on x4 links are not measured yet.
- **Repeatable per request, not per batch.** A request on its own gives the
  same result every time. When requests share a batch, the result can differ
  slightly from the same request alone, by about as much as with every
  optimisation switched off.
- **Context and KV are a trade.** Raising `MAX_LEN` lowers how many full-length
  requests fit at once. With `MM_CAP=0` (the default) the memory profiler also
  reserves room for a context-filling video, roughly 150k KV tokens.
- **The DFlash2 checkpoint is needed for the default mode.** `SPEC_MODE=mtp`
  uses the MTP head inside the model checkpoint, and `none` turns speculation
  off; both are slower.
- **Host-staged all-reduce is only for cards without peer access.** Where
  peer-to-peer works, it stands aside for the device-memory path.

---

## Licences

**This recipe** — the scripts and the documentation — is MIT, © 2026 Morrowmake.
See [LICENSE](LICENSE).

**The vLLM fork** is [Apache-2.0](https://github.com/Morrowmake/vllm-cmp170hx/blob/e77f89da2016c3949dba8550c6455b9421ed7365/LICENSE).

The downloaded models have separate licences:

- **Target:** [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP/blob/main/README.md)
  declares `license: mit` (MIT), inherited from the base model
  [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE).
  Its card says: "Follow the base model's usage terms."
- **Default drafter:** [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/main/README.md)
  declares `license: cc-by-nc-nd-4.0` (CC BY-NC-ND 4.0). Its card says
  "for research and evaluation. For commercial licensing, contact contact@inco.ai."
  The [licence](https://creativecommons.org/licenses/by-nc-nd/4.0/) requires
  attribution, prohibits commercial use, and prohibits distributing modified material.

To run without the external drafter, set `SPEC_MODE=none` in `.env` (no
speculation), or `SPEC_MODE=mtp` (the target checkpoint's built-in MTP head),
then run `./start.sh`. Both modes skip downloading and loading the external
drafter. For direct use of `serve.sh`, pass `none` or `mtp` as its argument.
The performance figures above use DFlash2, not these alternatives.

**The benchmark prompts** come from MiaAI-Lab's repository (AGPL-3.0) and
their sparkDash prompt constants (MIT), used as data with attribution. No code from their repositories is included here.

## Credits

- **[MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)**
  for publishing the benchmark prompts used in the decode tables.
- **[incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)** for the
  DFlash2 drafter checkpoint.
- **[canada-quant](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP)**
  for the W4A16 quantisation.
- **[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash)** for GLM-5.3-Flash,
  and the **[vLLM](https://github.com/vllm-project/vllm)** project for the
  engine these patches sit on top of.

## Source

The patches are ours; the fork branch is the code —
[Morrowmake/vllm-cmp170hx @ `ampere-glm53`](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere-glm53).
