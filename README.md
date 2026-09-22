# GLM-5.3-Flash on 4× NVIDIA CMP 170HX

<p>
  <strong>by <a href="https://x.com/Morrowmake">Morrowmake</a></strong>
  &nbsp;·&nbsp;
  <a href="https://x.com/Morrowmake"><img alt="Follow on X" src="https://img.shields.io/badge/Follow-%40Morrowmake-000000?style=flat&logo=x&logoColor=white"></a>
  &nbsp;
  <a href="https://github.com/Morrowmake/vllm/tree/ampere-glm53"><img alt="engine" src="https://img.shields.io/badge/engine-vLLM%20fork%20%40%2069c33802d0-4b32c3?style=flat"></a>
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/recipe-MIT-blue?style=flat">
</p>

**320B MoE at 165–240 tok/s, 256K context, OpenAI-compatible — on four ~$1,100
mining cards.**

This is everything needed to run **GLM-5.3-Flash** on **four NVIDIA CMP 170HX
cards** with our **[vLLM fork](https://github.com/Morrowmake/vllm/tree/ampere-glm53)**.
You get a 320B-parameter MoE on your own machine behind an OpenAI-compatible
API: a **262,144-token context**, tool calls and reasoning, images and video,
and **165–240 tok/s** for one user. The weights are W4A16 and nothing else is
reduced — the KV cache is full precision, there is no FP8 anywhere, and nothing
is offloaded to CPU or disk.

## Quick start

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe
cp .env.example .env          # optional: ./start.sh does this on first run
./start.sh                    # preflight, install, download, launch, wait for /health
```

`./start.sh` runs every step and skips the ones already done, so running it
twice is safe and the second run just launches. First time through it builds the
venv (about six minutes), fetches ~180 GB of checkpoints, then starts the
server; weight load and CUDA-graph capture take a few more minutes.

A prefix env assignment beats `.env` for every key:

```bash
MAX_LEN=131072 ./start.sh restart
VLLM_GLM5_DECODE_KERNELS=0 ./start.sh restart
SPEC_MODE=mtp ./start.sh restart
```

This recipe is tensor-parallel only — see [Link width](#link-width).

---

## Results

### Against 2x DGX Spark

![GLM-5.3-Flash on 4x CMP 170HX versus 2x DGX Spark](assets/glm53-cmp170hx-vs-dgx-spark-full-2026-09-18.jpg)

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
| Cold prefill | 2,243 tok/s |
| Cold prefill, 250K prompt | 2,076 tok/s |
| TTFT, 23,255-token prompt | 9.77 s (2,380 prompt tok/s) |
| KV pool at `--max-model-len 262144` | 1,160,192 tokens (4.43x concurrency) |
| GSM8K, n=50 at concurrency 8 | 0.980 |
| Independent run | [localmaxxing.com](https://www.localmaxxing.com/en/runs/cmu6l4y49081alq01svunzaqb) |

Prefill, TTFT, KV and GSM8K are from the pinned engine; the decode rows and
the 250K prefill rung are the 2026-09-18 sweep reproduced in the comparison
above.

Decode speculation is DFlash2 at k=3. Accept ratios run 0.91–0.98 on structured
and code prompts and 0.58–0.63 on prose, which is why prose decodes slower
despite being the same model on the same cards.

---

## What runs

| | |
|---|---|
| API | OpenAI-compatible, `http://127.0.0.1:8000/v1` |
| Model id | `glm-5.3-flash` |
| Weights | [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) — INT4 weights, FP16 activations, group size 128 |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), 320B MoE |
| Engine | [Morrowmake/vllm](https://github.com/Morrowmake/vllm) `ampere-glm53` @ `69c33802d0` |
| Layout | TP=4, PP=1. **Assumes PCIe Gen2 x16 between the cards** — see [Link width](#link-width) |
| Attention | Triton sparse-MLA (DSA) on sm_80, with the sm_80 indexer and kpool paths |
| Context | 262,144 tokens |
| KV cache | 1,160,192 tokens at `--gpu-memory-utilization 0.95`; 4.43x concurrency at full context; **not quantised** |
| Prefix caching | on |
| Speculation | DFlash2 ([`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)) at k=3; `SPEC_MODE=mtp` or `none` to change |
| Tools + reasoning | `--enable-auto-tool-choice`, glm47 tool-call and reasoning parsers |
| CUDA graphs | captured at the default power-of-two decode batch shapes |
| Vision | image and video on, uncapped by default (`MM_CAP=1` to bound them) |

---

## Link width

This recipe assumes four cards on **PCIe Gen2 x16** links, which is what ours
run. Tensor parallelism splits every layer across all four cards, so it leans
on those links hard: about 9.4 MB per layer during prefill and roughly 100
small collectives per decode step. On stock x4 links — a quarter of the
bandwidth — TP=4 is bus-bound and will be far slower than the numbers above.

For x4 cards, a pipeline-parallel recipe for the same hardware lives at
[JJ48/glm53-flash-170hx-serving](https://github.com/JJ48/glm53-flash-170hx-serving).

---

## Configuration

Everything lives in `.env`, copied from `.env.example` on first run and
gitignored. Read [`.env.example`](.env.example) for the full annotated set; the
headlines are:

| Key | Default | |
|---|---|---|
| `VLLM_COMMIT` | *(commented out)* | engine pin. Left to `start.sh`'s default so a `git pull` can move it |
| `PP` / `TP` | `1` / `4` | layout; `PP*TP` must equal your GPU count |
| `MAX_LEN` | `262144` | context ceiling; lowering it raises KV concurrency |
| `MAX_SEQS` | `8` | concurrent sequences |
| `MAX_BATCHED` | `2048` | batched tokens per scheduler step |
| `GPU_UTIL` | `0.95` | memory target. 0.97 was too tight here |
| `SPEC_MODE` / `SPEC_N` | `dflash` / `3` | speculator and draft depth; `mtp` or `none` |
| `PORT` / `SERVED_MODEL_NAME` | `8000` / `glm-5.3-flash` | |
| `REASONING_PARSER` / `TOOL_PARSER` | `glm47` / `glm47` | |
| `PREFILL_CAP` | `0` | upstream's **unconditional** chunk cap. Leave it 0 |
| `FAIR_PREFILL` / `FAIR_CHUNK` | `1` / `384` | decode-aware chunking — the cap you actually want |
| `MM_CAP` | `0` | `1` bounds vision and video, buying back ~150k KV tokens |
| `READY_TIMEOUT` | `1800` | seconds to wait for `/health` |
| `BUILD_FROM_SOURCE` | `0` | `1` compiles the CUDA extensions instead of using upstream's |

`PREFILL_CAP` is not a gentler `FAIR_PREFILL`. It applies unconditionally, cost
−15.3% prefill and +16.5% TTFT@23K here, and as a side effect drops chunks below
the two prefill gates, silently disabling the overlap and the prefill kernels.
`FAIR_PREFILL` only bites while requests are actually decoding.

### Kill switches

All seven features are off by default in the engine and turned on only by
`serve.sh`, so each one is a single variable you can set to `0` and restart — no
rebuild, no revert:

| Key | Feature |
|---|---|
| `VLLM_GLM5_PREFILL_OVERLAP` | TP prefill comm/compute overlap |
| `VLLM_GLM5_PREFILL_KERNELS` | sm_80 prefill kernels |
| `VLLM_GLM5_PROLOGUE_FUSE` | fused eager decode prologue |
| `VLLM_GLM5_LOCAL_LOGITS` | batch-sharded logits and sampling |
| `VLLM_GLM5_DECODE_KERNELS` | sm_80 decode kernels |
| `VLLM_GLM5_THIN_GEMM` | sm_80 thin-M BF16 GEMM |
| `VLLM_GLM5_HOST_ALLREDUCE` | host-staged all-reduce for nodes without peer access |
| `FAIR_PREFILL` | decode-aware prefill chunking |

If output quality is ever in question, turn `VLLM_GLM5_DECODE_KERNELS` off
first. Exactness moved slightly outside its noise floor when the seven were
merged (0.4119 against floors of 0.2007 and 0.2984) and GSM8K went 1.000 → 0.980
at n=50; the decode kernels own that movement.

```bash
VLLM_GLM5_DECODE_KERNELS=0 ./start.sh restart
```

---

## Operating

| Command | |
|---|---|
| `./start.sh` | preflight → install → download → launch → wait for `/health` |
| `./start.sh install` | build the venv and the pinned fork only |
| `./start.sh download` | fetch the two checkpoints only |
| `./start.sh stop` | stop the server this checkout started |
| `./start.sh restart` | stop, then start |
| `./start.sh status` | process, `/health`, KV line, install and checkpoint state |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh update` | `git pull`, reinstall if the pin moved, restart |
| `./start.sh smoke` | one chat request and one tool call |
| `./start.sh help` | the header of `start.sh` |

`install.sh`, `download.sh` and `stop.sh` are one-line wrappers around the
matching subcommand. `serve.sh` is the internal launcher `start.sh` execs; you
can run it in the foreground yourself, and `DRY=1 ./start.sh` prints the command
it would run.

Lifecycle commands on a checkout are serialised by a `flock` on
`logs/lifecycle.lock`. `stop` only ever signals the PID in `logs/vllm.pid`, and
only after confirming that process is the server this checkout launched — it
never searches by process name, so another vLLM on the same machine is never
touched.

### Updating and rolling back

```bash
./start.sh update                          # pull, reinstall if the pin moved, restart
VLLM_COMMIT=<older sha> ./start.sh update  # roll back to a known-good engine
```

`update` re-executes itself after a pull that changed this repo, so it acts on
the new pin rather than the one it started with. The pin lives in `start.sh`,
not in your `.env`, precisely so a pull can move it; uncomment `VLLM_COMMIT` in
`.env` if you would rather freeze it.

---

## What is in the patches

The fork is [Morrowmake/vllm](https://github.com/Morrowmake/vllm), branch
`ampere-glm53`, pinned in `start.sh` to commit `69c33802d0`. Every patch is
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
stage boundaries and the MTP drafter loads the target embedding under PP, so the
engine supports a pipeline layout. This recipe does not — it is tuned for TP=4
throughout. For a pipeline-parallel recipe on these cards, see
[JJ48/glm53-flash-170hx-serving](https://github.com/JJ48/glm53-flash-170hx-serving).

All seven together, against the same engine with all seven off: prefill
+13.4%, TTFT@23K −14.9%, ms/step at one stream −1.22, decode retention during
someone else's prefill 7% → 18%, KV pool unchanged.

---

## Installing by hand

`./start.sh` does all of this for you. If you want the pieces:

```bash
./install.sh     # venv + the pinned fork, verified imports
./download.sh    # the two checkpoints
./serve.sh       # launch in the foreground
./smoke.sh       # one chat request, one tool call
```

**CUDA.** You need a 13.x toolkit. NVIDIA had no working `ubuntu2604`
repository index when we built this, so we used the `ubuntu2404` one:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3 git-lfs
```

**Changing the pin means reinstalling.** `start.sh` records the installed
commit in `venv/.recipe-stamp` and re-runs the install when `VLLM_COMMIT`
changes, which is what you want: the compiled extensions have to match the
upstream commit the branch sits on, and upstream's ABI does move. Updating the
checkout by hand without reinstalling gives you a mismatched extension and an
engine that dies on startup.

**Why the install is quick.** By default it uses upstream's **precompiled**
extensions rather than compiling them, which takes minutes instead of hours.
That is not a shortcut: the diff between `ampere-glm53` and upstream touches no
`.cu`, `.cpp` or `CMakeLists` file, so the compiled objects are identical to
upstream's, and upstream's wheels already carry sm_80 cubins. To compile them
yourself, `BUILD_FROM_SOURCE=1 MAX_JOBS=16 ./install.sh` — it pins
`TORCH_CUDA_ARCH_LIST=8.0`, needs the full toolkit and ~60 GB of scratch, and
takes one to two hours.

**Checkpoints.**

| Repository | Size | Role |
|---|---:|---|
| [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) | ~178 GB, 21 files | target model |
| [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | ~2.2 GB, 5 files | DFlash2 drafter |

**Smoke test output**, against a running server:

```
==> waiting for http://127.0.0.1:8000/health (up to 900s)
    healthy
    served models: glm-5.3-flash
==> chat request
    usage: prompt=28 completion=200 wall=1.44s  ->  138.5 tok/s
==> tool-call request
    tool_call: get_weather({"city": "Reykjavik", "unit": "celsius"})
    usage: prompt=199 completion=66 wall=0.48s  ->  138.3 tok/s
==> KV cache
    GPU KV cache size: 1,160,192 tokens, Maximum concurrency for 262,144 tokens per request: 4.43x
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
`--gpu-memory-utilization 0.95` the KV pool is 1,160,192 tokens, which is 4.43x
concurrency at full context. Raising `MAX_LEN` lowers that multiplier; with
`MM_CAP=0` the memory profiler also reserves for a context-filling video, which
costs roughly 150k KV tokens.

**TP=4 only, and it assumes wide links.** Tensor parallelism moves about
9.4 MB per layer between cards during prefill and ~100 small collectives per
decode step, so on x4 links it is bus-bound. This recipe does not support a
pipeline layout — see [Link width](#link-width).

---

## License

**This recipe** — the scripts and the documentation — is MIT, © 2026 Morrowmake.
See [LICENSE](LICENSE).

**The vLLM fork** it installs is Apache-2.0, like upstream vLLM; our patches are
contributed under that licence.

**The weights** are MIT: the quantisation
[`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP)
and the base model
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).

**The DFlash2 drafter**
([`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2))
is **CC BY-NC-ND 4.0** — research and evaluation only, non-commercial, no
derivatives. It is the default speculator here, so read that before you deploy
this anywhere commercial. `SPEC_MODE=mtp` serves the MTP head inside the MIT
target checkpoint instead and does not use it at all.

**The benchmark prompts and the published DGX Spark figures** are quoted from
MiaAI-Lab's repository (AGPL-3.0) and their sparkDash prompt constants (MIT),
used as data with attribution. No code from their repositories is included here.

## Credits

- **[MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)**
  for publishing their 2x DGX Spark figures and their benchmark prompts, which
  are the entire comparison column above. Thank you for publishing both.
- **[incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)** for the
  DFlash2 drafter checkpoint.
- **[canada-quant](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP)**
  for the W4A16 quantisation.
- **[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash)** for GLM-5.3-Flash,
  and the **[vLLM](https://github.com/vllm-project/vllm)** project for the
  engine these patches sit on top of.

## Source

The patches are ours; the fork branch is the code —
[Morrowmake/vllm @ `ampere-glm53`](https://github.com/Morrowmake/vllm/tree/ampere-glm53).
