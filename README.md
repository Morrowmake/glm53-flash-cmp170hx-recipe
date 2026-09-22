# GLM-5.3-Flash on 4x CMP 170HX

Everything needed to serve **GLM-5.3-Flash** — a 320B-parameter MoE — at
**W4A16** across four **NVIDIA CMP 170HX** cards, using upstream vLLM with our
Ampere patches.

There is no FP8 anywhere, no KV-cache quantisation, and no CPU or disk offload.
The weights are INT4 with FP16 activations (group size 128), the KV cache is
full precision, and the whole model lives in HBM2e across the four cards. The
patches are what make an sm_80 GPU run GLM-5.3-Flash's DeepSeek-style sparse
attention at all, and what make tensor parallelism survive a PCIe Gen 2 bus with
no peer-to-peer.

Four cards, 256 GB of HBM2e, a 262,144-token context with a 1.15M-token KV pool,
**165–240 tok/s** of decode for one user and **~2,200 tok/s** of cold prefill.

```
./install.sh      # venv + the patched vLLM fork, pinned
./download.sh     # ~180 GB of checkpoints
./serve.sh        # OpenAI-compatible server on :8000
./smoke.sh        # one chat request, one tool call
```

---

## Tested on

| | |
|---|---|
| GPUs | 4x NVIDIA CMP 170HX — GA100, sm_80; our cards expose 64 GB HBM2e each |
| Interconnect | PCIe Gen 2 x16, no peer-to-peer |
| CPU | AMD EPYC 7663 |
| RAM | 247 GB |
| OS | Ubuntu 26.04 |
| Driver | NVIDIA 610.57 |
| CUDA | 13.3 |

Results were measured at a 180 W power limit.

Peer-to-peer was unavailable on this machine — `nvidia-smi topo -p2p r` answers
`GPU not supported` on every pair — so vLLM's `CustomAllreduce` switches itself
off and NCCL falls back to a shared-memory ring costing `2(N-1)` sequential host
hops per message. Much of the serve configuration below exists to work around
that, and `VLLM_GLM5_HOST_ALLREDUCE` is the patch that replaces it.

## Requirements

- **4 CUDA GPUs with roughly 60 GB or more each.** The W4A16 weights take about
  45 GB per card at TP=4, and the KV cache takes what is left.
- **sm_80 or newer.** The Ampere patches are exercised on sm_80; newer parts run
  fine, and the Ampere-specific backends are only selected where they are
  needed.
- **CUDA 13.x toolkit.**
- **Python 3.12.**
- **About 185 GB of disk** for the two checkpoints.

---

## Results

### Against 2x DGX Spark

![GLM-5.3-Flash on 4x CMP 170HX versus 2x DGX Spark](assets/glm53-cmp170hx-vs-dgx-spark-full-2026-09-18.jpg)

![Three-bar summary](assets/glm53-vs-dgx-spark-simple-2026-09-18.jpg)

The published figures and the benchmark prompts on the DGX Spark side are
[MiaAI-Lab's](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks),
quoted from their README — thank you to them for publishing both.

- **Theirs:** 2x DGX Spark, EXL3 quant, DFlash2 k=7, 850K–1M declared context, as published.
- **Ours:** 4x CMP 170HX, vLLM W4A16, DFlash2 k=3, 262,144 context, measured 2026-09-18, thinking off, temperature 0, median of 5.

#### Decode

400 max tokens, median of 5. `Stream tok/s` is per request,
`(completion_tokens - 1) / (end - first token)`; `Agg tok/s` is
`sum(completion_tokens) / wall` across all streams. Our runs prepend a unique
nonce so nothing hits the prefix cache, which makes our TTFT column pessimistic
against theirs; the warm column reruns their exact prompt with no nonce.

| Prompt type | Conc | Theirs stream | Ours stream | Theirs agg | Ours agg | Theirs TTFT | Ours TTFT (cold) | Ours TTFT (warm) | Ours accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Structured (count 1→200) | ×1 | 62.9 | **238.7** | 62.9 | **238.7** | 719 ms | 65 ms | 60 ms | 0.977 |
|  | ×2 | 51.7 | **185.7** | 103.3 | **348.8** | 6620 ms | 116 ms | — | 0.972 |
|  | ×3 | not published | **133.6** | not published | **368.0** | not published | 250 ms | — | 0.957 |
|  | ×4 | 37.1 | **137.5** | 146.5 | **497.9** | 6300 ms | 255 ms | — | 0.942 |
|  | ×8 | not published | **94.5** | not published | **669.8** | not published | 341 ms | — | 0.934 |
| Code (clamp_00…clamp_49) | ×1 | 62.9 | **232.4** | 62.9 | **232.4** | 719 ms | 159 ms | 158 ms | 0.968 |
|  | ×2 | 51.7 | **167.0** | 103.3 | **301.5** | 6620 ms | 249 ms | — | 0.908 |
|  | ×3 | not published | **124.5** | not published | **335.2** | not published | 354 ms | — | 0.921 |
|  | ×4 | 37.1 | **127.3** | 146.5 | **437.7** | 6300 ms | 399 ms | — | 0.917 |
|  | ×8 | not published | **87.4** | not published | **603.4** | not published | 624 ms | — | 0.928 |
| Prose (hash map) | ×1 | 36.1 | **165.1** | 37.1 | **165.1** | 333 ms | 69 ms | 60 ms | 0.584 |
|  | ×2 | 25.0 | **130.5** | 51.1 | **239.6** | 365 ms | 164 ms | — | 0.611 |
|  | ×3 | 22.3 | **95.5** | 65.8 | **265.5** | 405 ms | 254 ms | — | 0.628 |
|  | ×4 | 19.4 | **98.3** | 75.3 | **360.4** | 401 ms | 259 ms | — | 0.627 |
|  | ×8 | not published | **65.5** | not published | **469.7** | not published | 343 ms | — | 0.605 |

Their 3x Spark (TP=3) prose table, for reference — 512 tokens, not 400, so it is
a reference block rather than a row above:

| Conc | Theirs TP3 stream | Theirs TP3 agg | Theirs TP3 TTFT | Ours prose stream | Ours prose agg |
|---:|---:|---:|---:|---:|---:|
| ×1 | 40.1 | 40.1 | 255 ms | 165.1 | 165.1 |
| ×2 | 28.7 | 56.6 | 411 ms | 130.5 | 239.6 |
| ×3 | 25.5 | 75.5 | 323 ms | 95.5 | 265.5 |
| ×4 | 22.8 | 88.4 | 351 ms | 98.3 | 360.4 |

Their lab `tests/bench_decode.py` medians against our warm ×1 on their exact
prompt:

| Prompt | Theirs lab tok/s | Theirs accept | Ours warm ×1 tok/s | Ours accept |
|---|---:|---:|---:|---:|
| structured | 65.1 | 0.959 | 238.4 | 0.980 |
| prose | 27.1 | 0.341 | 172.2 | 0.617 |

#### Cold prefill

Unique uncached text, `max_tokens=1`, median of 2. `Prefill tok/s` is
`prompt tokens / TTFT`, measured client side. Prefix-cache hit deltas from
`/metrics` were 0 on every rung, confirming nothing was warm.

| Rung | Theirs prompt tok | Theirs TTFT | Theirs tok/s | Ours prompt tok | Ours TTFT | Ours tok/s |
|---|---:|---:|---:|---:|---:|---:|
| ~8k | 8,221 | 5.51 s | 1492.1 | 7,977 | 3.61 s | **2208.7** |
| ~16k | 16,411 | 10.56 s | 1553.7 | 15,979 | 7.19 s | **2223.8** |
| ~32k | 32,797 | 22.96 s | 1428.2 | 31,925 | 14.63 s | **2182.9** |
| ~64k | 65,566 | 41.31 s | 1587.0 | 64,056 | 29.23 s | **2191.4** |
| ~128k | 131,101 | 83.95 s | 1561.7 | 127,587 | 59.36 s | **2149.4** |
| ~250k | 262,173 | 172.84 s | 1516.8 | 250,281 | 120.54 s | **2076.3** |

Their top rung is 262,173 tokens, which does not fit under our 262,144 ceiling,
so our top rung is ~250k against their longer prompt.

### Our production numbers

The configuration in this repo, as it runs day to day:

| | |
|---|---:|
| Decode, 1 user, structured | 238.7 tok/s |
| Decode, 1 user, code | 232.4 tok/s |
| Decode, 1 user, prose | 165.1 tok/s |
| Decode, 4 users, aggregate | 360–498 tok/s by prompt type |
| Decode, 8 users, aggregate | 470–670 tok/s by prompt type |
| Cold prefill, 8K–128K | 2,149–2,224 tok/s |
| Cold prefill, 250K | 2,076 tok/s |
| TTFT, 23,255-token prompt | 9.79 s (2,375 prompt tok/s) |
| KV pool at `--max-model-len 262144` | 1,158,144 tokens (4.42x concurrency) |
| GSM8K, n=50 at concurrency 8 | 0.980 |
| Independent run | [localmaxxing.com](https://www.localmaxxing.com/en/runs/cmu6l4y49081alq01svunzaqb) |

Decode speculation is DFlash2 at k=3. Accept ratios run 0.91–0.98 on structured
and code prompts and 0.58–0.63 on prose, which is why prose decodes slower
despite being the same model on the same cards.

---

## What is in the patches

The fork is [Morrowmake/vllm](https://github.com/Morrowmake/vllm), branch
`ampere-glm53`, pinned in `install.sh` to commit `cf80da1839`. Every patch is
Python, Triton or TileLang — nothing touches vLLM's CUDA or C++ sources. Each
feature is **off by default in the code** and turned on only by `serve.sh`, so
every one of them is a single-variable kill switch.

**Ampere sparse-MLA, indexer and kpool backends.** GLM-5.3-Flash uses
DeepSeek-style sparse attention, whose upstream kernels want Hopper. We added a
`TRITON_MLA_SPARSE` backend, a non-DeepGEMM logits path for the DSA indexer with
sm_80-safe FP8 stores, a Triton e4m3 dequant that does the conversion in PTX
where sm_80 has no instruction for it, and sm_80 variants of the kpool
compression. Without these the model does not run on these cards at all; the
rest of the list is performance.

**TP prefill comm/compute overlap** (`VLLM_GLM5_PREFILL_OVERLAP`). Each mHC
layer's post-attention section is split into token micro-batches so its two
all-reduces fly on a side stream while the MoE computes. Two splits measured
best; four were worse.

**Host-staged all-reduce for PCIe without P2P** (`VLLM_GLM5_HOST_ALLREDUCE`).
With peer access refused, NCCL's shared-memory ring pays `2(N-1)` sequential
host hops per message. This path does one round trip through a shared `/dev/shm`
segment instead: 2.09x faster per all-reduce in a decode trace (78.6 → 37.6 µs
mean), ms/step 19.10 → 17.63 at one stream and 37.6 → 31.8 at four. Messages
over 512 KiB stay on NCCL, where prefill already runs near wire speed.

**Thin-batch GEMM** (`VLLM_GLM5_THIN_GEMM`). W4A16 leaves some linears
unquantized, and cuBLAS is poor at BF16 GEMMs with very few rows. A thin-M sm_80
kernel covers `M <= 32`, which is also a CUDA-graph capture size.

**Decode kernels** (`VLLM_GLM5_DECODE_KERNELS`). sm_80 implementations of the
fused mHC post+pre norm, MoE routing and block alignment, and KDA decode, bounded
to the small token counts decode actually sees.

**Prefill kernels** (`VLLM_GLM5_PREFILL_KERNELS`). sm_80 mHC pre-norm projection
and sparse-MLA DSA attention for the prefill shapes, gated at a minimum chunk
size so they do not fire on the tiny chunks fair prefill produces.

**Fused prologue and batch-sharded logits** (`VLLM_GLM5_PROLOGUE_FUSE`,
`VLLM_GLM5_LOCAL_LOGITS`). The eager decode prologue collapses into one fused
kernel, and each rank samples its own `1/TP` slice of the batch instead of every
rank all-gathering full-vocab logits — a meaningful saving when the all-gather
crosses PCIe Gen 2.

**Fair chunked prefill** (`FAIR_PREFILL`). A decode-aware prefill budget: while
requests are decoding, prefill chunks are capped so a long prompt cannot starve
them. Decode retention during someone else's prefill went from 7% to 18% of
baseline. Unlike upstream's unconditional cap it only applies when something is
actually decoding.

**Pipeline parallelism enablement.** Deferred mHC post state is materialised at
stage boundaries and the MTP drafter loads the target embedding under PP, so
`PP=4 TP=1` works. It is not the default — see *Known limits*.

Measured all together against the pre-merge default: prefill +13.4%,
TTFT@23K −14.9%, ms/step at one stream −1.22 (paired, drift 0.58), decode
retention during prefill 7% → 18%, KV pool unchanged.

---

## Install

You need a working NVIDIA driver, a CUDA 13.x toolkit and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

**CUDA 13.3.** NVIDIA had no working `ubuntu2604` repository index when we built
this, so we used the `ubuntu2404` one, which installs cleanly:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3 git-lfs
```

Then:

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe
./install.sh
```

`install.sh` creates `./venv` on Python 3.12, clones the fork into `./vllm-src`
at the pinned commit, installs torch 2.13.0+cu130, installs the fork editable,
adds flashinfer 0.6.18.post1, TileLang 0.1.12 and ninja, and then verifies that
`torch`, `vllm`, the compiled extensions, Triton, TileLang, flashinfer and the
sm_80 patch modules all import.

It takes about six minutes, because by default it uses upstream's **precompiled**
extensions rather than compiling them. That is not a shortcut: the diff between
`ampere-glm53` and upstream touches no `.cu`, `.cpp` or `CMakeLists` file, so the
compiled objects are identical to upstream's, and upstream's wheels already carry
sm_80 cubins. It is exactly how our production environment was built. If you
would rather compile everything yourself:

```bash
BUILD_FROM_SOURCE=1 MAX_JOBS=16 ./install.sh
```

which pins `TORCH_CUDA_ARCH_LIST=8.0`, needs the full toolkit and ~60 GB of
scratch, and takes one to two hours.

Then fetch the checkpoints — about 180 GB, so give it a while:

```bash
./download.sh
```

| Repository | Size | Role |
|---|---:|---|
| [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) | ~178 GB, 21 files | target model |
| [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | ~2.2 GB, 5 files | DFlash2 drafter |

---

## Run

```bash
./serve.sh              # DFlash2 drafter, k=3 — the default
./serve.sh mtp          # MTP drafter instead
./serve.sh none         # no speculative decoding
DRY=1 ./serve.sh        # print the command instead of running it
```

Model load plus CUDA-graph capture takes several minutes. When it is up you have
an OpenAI-compatible server on `:8000` serving `glm-5.3-flash`, with the glm47
reasoning and tool-call parsers enabled.

```bash
./smoke.sh
```

sends one chat request and one tool call, prints tok/s from each usage block,
and echoes the KV line from the log:

```
==> waiting for http://127.0.0.1:8000/health (up to 900s)
    healthy
    served models: glm-5.3-flash
==> chat request
    usage: prompt=28 completion=200 wall=1.64s  ->  121.7 tok/s
==> tool-call request
    tool_call: get_weather({"city": "Reykjavik", "unit": "celsius"})
    usage: prompt=199 completion=66 wall=0.48s  ->  136.3 tok/s
==> KV cache
    GPU KV cache size: 1,158,144 tokens, Maximum concurrency for 262,144 tokens per request: 4.42x
```

`smoke.sh` reads `./logs/serve.log` for that last line; point `SERVE_LOG` at
wherever you actually sent the server's output.

---

## Configuration

Every knob is an environment variable read by `serve.sh`. The defaults are what
produced the numbers above; the header comment in `serve.sh` is the full
reference.

### Paths

| Variable | Default | |
|---|---|---|
| `VENV` | `./venv` | environment built by `install.sh` |
| `MODEL` | `./models/GLM-5.3-Flash-W4A16-MTP` | target model |
| `DFLASH_MODEL` | `./models/GLM-5.3-Flash-DFlash2` | drafter |
| `CUDA_HOME` | `/usr/local/cuda-13.3` | toolkit root |

### Engine

| Variable | Default | |
|---|---|---|
| `TP` | `4` | tensor-parallel size |
| `PP` | `1` | pipeline-parallel size; `PP*TP` must be 4 |
| `MAX_LEN` | `262144` | context ceiling |
| `MAX_SEQS` | `8` | concurrent sequences |
| `MAX_BATCHED` | `2048` | batched tokens per step |
| `GPU_UTIL` | `0.95` | memory target. 0.97 was judged too risky here |
| `PORT` | `8000` | |
| `SERVED_NAME` | `glm-5.3-flash` | |
| `SPEC_N` | `3` | speculative tokens (DFlash2 k) |
| `REASONING_PARSER` | `glm47` | |
| `TOOL_PARSER` | `glm47` | |
| `EXTRA_ARGS` | *(empty)* | appended verbatim; a `--speculative-config` here wins |
| `VLLM_PP_LAYER_PARTITION` | mode-dependent | only under `PP=4` |

### Prefill

| Variable | Default | |
|---|---|---|
| `PREFILL_CAP` | `0` | upstream's **unconditional** long-prefill chunk cap. **Leave it 0** — see below |
| `FAIR_PREFILL` | `1` | decode-aware chunking |
| `FAIR_CHUNK` | `384` | chunk size while something is decoding |
| `FAIR_PARTIAL` | `2` | concurrent partial prefills |
| `VLLM_GLM5_PREFILL_MIN_TOKENS` | `384` | gate for the prefill kernels |

`PREFILL_CAP` is not a gentler version of `FAIR_PREFILL`. It applies
unconditionally, cost −15.3% prefill and +16.5% TTFT@23K in production, and as a
side effect drops chunks below the two prefill gates, silently disabling the
overlap and the prefill kernels. `FAIR_PREFILL` is the one you want: it only
bites while requests are actually decoding.

### Multimodal

| Variable | Default | |
|---|---|---|
| `MM_CAP` | `0` | `0` leaves vision and video uncapped, which makes the memory profiler reserve for a context-filling video and costs ~150k KV tokens. `1` bounds both the profiler dummy and real inputs |
| `MM_IMAGES` | `4` | images per prompt when `MM_CAP=1` |
| `MM_FRAMES` | `32` | video frames when `MM_CAP=1` |
| `MM_MAX_PIXELS` | `1003520` | pixel budget when `MM_CAP=1` |

### Kill switches

All seven features are on by default in `serve.sh` and off by default in the
code. Set any of these to `0` and restart — no rebuild, no revert, one variable
changed:

| Variable | Feature |
|---|---|
| `VLLM_GLM5_PREFILL_OVERLAP` | TP prefill comm/compute overlap |
| `VLLM_GLM5_PREFILL_KERNELS` | sm_80 prefill kernels |
| `VLLM_GLM5_PROLOGUE_FUSE` | fused eager decode prologue |
| `VLLM_GLM5_LOCAL_LOGITS` | batch-sharded logits and sampling |
| `VLLM_GLM5_DECODE_KERNELS` | sm_80 decode kernels |
| `VLLM_GLM5_THIN_GEMM` | sm_80 thin-M BF16 GEMM |
| `VLLM_GLM5_HOST_ALLREDUCE` | host-staged no-P2P all-reduce |
| `FAIR_PREFILL` | decode-aware prefill chunking |

If output quality is ever in question, turn off `VLLM_GLM5_DECODE_KERNELS`
first. Exactness moved slightly outside its noise floor when the seven were
merged (0.4119 against floors of 0.2007 and 0.2984) and GSM8K went 1.000 → 0.980
at n=50; the decode kernels own that movement.

```bash
VLLM_GLM5_DECODE_KERNELS=0 ./serve.sh
```

---

## Known limits

**The backends here target sm_80.** Every kernel in the patches is written for
Ampere. On newer architectures upstream vLLM's own kernels are better, and the
Ampere paths are only selected where they are needed. The flags gate the
features, not the architecture, so forcing them on elsewhere is untested.

**The no-P2P path is only worth it without P2P.** `VLLM_GLM5_HOST_ALLREDUCE`
replaces NCCL's shared-memory ring for small collectives, which is a large win
when peer access is unavailable and pointless when it is not. Where P2P works,
leave it off. It is also why per-stream decode here does not scale with device
count the way it would over a fast fabric.

**The prefill chunk observer validated at 1152.** The overlap was measured and
tuned at a 1152-token chunk; other chunk sizes work but were not characterised,
and the prefill kernels will not engage below `VLLM_GLM5_PREFILL_MIN_TOKENS`.

**The DFlash2 checkpoint is required for the default mode.** `./serve.sh` with
no argument wants `./models/GLM-5.3-Flash-DFlash2`. Use `./serve.sh mtp` for the
MTP head that ships inside the target checkpoint, or `./serve.sh none` for no
speculation — both are slower.

**Context and KV are a trade.** At `--max-model-len 262144` and
`--gpu-memory-utilization 0.95` the KV pool is 1,158,144 tokens, which is 4.42x
concurrency at full context. Raising `MAX_LEN` lowers that multiplier; with
`MM_CAP=0` the memory profiler also reserves for a context-filling video, which
costs roughly 150k KV tokens.

**TP=4 is the default for a reason.** `PP=4 TP=1` gives faster long-prompt TTFT
(about 5.0 s against 11.8 s at 23K) and a larger KV pool, but roughly half the
single-stream decode rate (80–85 against 135–145 tok/s at the time it was
measured). Pick by workload.

---

## License

This recipe — the scripts, the documentation and the figures — is MIT, © 2026
Morrowmake. See [LICENSE](LICENSE).

The vLLM fork it installs is Apache-2.0, like upstream vLLM. The model
checkpoints carry their own licenses from their respective publishers.

## How we got here

These patches came out of a long optimisation campaign against a single
question: what does it take to serve a modern sparse-attention MoE on GA100
silicon behind a PCIe Gen 2 bus with no peer-to-peer? Each feature was built on
its own branch, validated in isolation against a measured noise floor, then
merged and re-validated together; the ones that did not survive that were
dropped. The full history — branch by branch, with the commit messages that
record what each one measured — is on
[Morrowmake/vllm @ `ampere-glm53`](https://github.com/Morrowmake/vllm/commits/ampere-glm53).
