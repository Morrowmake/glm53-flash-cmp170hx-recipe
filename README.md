<h1 align="center">GLM-5.3-Flash on 4× NVIDIA CMP 170HX</h1>

<p align="center">
  <strong>by <a href="https://x.com/Morrowmake">Morrowmake</a></strong>
  <br><br>
  <a href="https://x.com/Morrowmake"><img alt="Follow on X" src="https://img.shields.io/badge/Follow-%40Morrowmake-000000?style=flat&logo=x&logoColor=white"></a>
  &nbsp;
  <a href="https://github.com/Morrowmake/vllm-cmp170hx/tree/3bbb99a5344c3a9cc7e0a1c8ff0d602263520ef5"><img alt="engine" src="https://img.shields.io/badge/engine-vLLM%20fork%20%40%203bbb99a534-4b32c3?style=flat"></a>
  &nbsp;
  <img alt="release" src="https://img.shields.io/badge/release-1.3.0-2ea44f?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/recipe-MIT-blue?style=flat">
</p>

**A 320B-parameter MoE with a 262,144-token context, served on four CMP 170HX
cards at 264.6 tok/s for one user and 745.4 tok/s across eight. The weights are
W4A16 and nothing else is cut: the KV cache is full precision, there is no FP8
anywhere, and nothing is offloaded to CPU or disk. A request sent on its own
gives the same output every time.**

This repository installs and runs **GLM-5.3-Flash** on **four NVIDIA CMP 170HX
cards** behind an OpenAI-compatible API, with tool calls, reasoning, images and
video, and a **DFlash2** speculative drafter. Upstream vLLM's sparse-attention
path needs a Hopper GPU; the CMP 170HX is Ampere (sm_80). Our
[vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere-glm53) adds
the Ampere kernels that make the model run at all, then spends the rest of its
patches on making it fast and making it repeatable.

> **Which setups this release is for.** Release 1.3.0 is optimised for
> **tensor-parallel 4 on PCIe x16 links** — cards with the x16 capacitor
> modification. That is where every number on this page was measured. On cards
> limited to x4 links, tensor-parallel is bus-bound and much slower; a
> **pipeline-parallel 4 layout optimised for x4 cards is in active development**
> and will ship in a later release (see [Status and roadmap](#status-and-roadmap)).
> `./start.sh` checks your link width and warns if a card is narrower than x16.

One command sets it up and starts it:

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe
./start.sh
```

What changed in this release is in [CHANGELOG.md](CHANGELOG.md).

---

## Results

### Release 1.3.0

Tensor-parallel 4, DFlash2 at k=3, 262,144-token context, the defaults in this
repository. Two columns: the default, with the cards talking through the host,
and the optional [PCIe peer-to-peer](#pcie-peer-to-peer-optional) path.

| | Peer-to-peer off (default) | Peer-to-peer on (optional) |
|---|---:|---:|
| Decode, 1 user, structured / code / prose | **264.6 / 260.4 / 188.4 tok/s** | **274.8 / 273.0 / 197.9 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **745.4 / 664.5 / 519.8 tok/s** | **808.1 / 720.5 / 556.3 tok/s** |
| Decode step, 1 / 4 / 6 / 8 users | 15.84 / 30.09 / 38.78 / 44.71 ms | 15.25 / 28.02 / 35.55 / 41.92 ms |
| Cold prefill | **2,484 tok/s** | **2,490 tok/s** |
| Time to first token, 6,217 / 23,255-token prompt | 2.50 s / 8.97 s | 2.49 s / 8.94 s |
| KV pool at 262,144 context | 1,174,567 tokens (4.48 full-length requests) | 1,187,776 tokens (4.53) |

All at 180 W per card (a power limit we set on our cards; the scripts never
change power, clock or fan settings), one server start per column. Decode tok/s is the
per-request streaming rate on three fixed prompt types — structured,
code and prose (400 tokens, temperature 0, median of 5); prose is slower because the
drafter's guesses are accepted less often. Cold prefill is the median over
real-text prompts of 23.9K to 37.9K tokens.

**Quality**, measured on the default (peer-to-peer off):

| | |
|---|---:|
| Perplexity, fixed 60-document set | **3.2858** (bit-identical with peer-to-peer on) |
| GSM8K, all 1,319 problems | **0.975** (none cut off at 3,072 tokens) |
| HumanEval, all 164, pass@1 | **0.9573** (157/164) |

HumanEval allows 4,096 tokens per reply and scores the last complete fenced code
block of the reply, reasoning included; 12 replies hit the 4,096-token limit.
Scoring the first code block instead gives 0.8537.

### Measured while building this release

These come from development runs on the same four cards. They are not the
release table above; each line says what it was measured against.

- **−23% on each decode step (−25% with peer-to-peer).** One user: 20.5 ms
  per decode step with every optional optimisation in the fork switched off
  (measured on release 1.0.0's engine), 15.8 ms with this release's defaults
  and 15.3 ms with peer-to-peer on.
- **Second-generation decode kernels**, measured together with peer-to-peer
  on: step time −7.4% at one and four users, −17.6% at six, −5.4% at eight,
  cold prefill flat.
- **The same answer every time.** A request on its own now returns the same
  first token and the same log-probabilities on every repeat, across restarts
  (16 of 16 test prompts, 8 repeats over 2 boots). The fixes cost under 1% of
  step time.
- **More KV for free.** +39,626 KV tokens (+3.45%, measured with
  peer-to-peer on) from right-sized workspaces and a drafter table split across the cards, with outputs
  bit-identical and no step-time cost.
- **Fast does not mean different.** On the previous engine, perplexity with
  every optimisation on and every optimisation off differed by less than
  run-to-run variation.
- **Long context holds up.** On the previous engine: needle retrieval 30/30
  from 8K to 262K tokens, a verbatim copy of a passage at 262,000 tokens
  returned byte-exact, and nothing leaked between concurrent requests.

### Decode and prefill in detail

Release 1.3.0 with the defaults (peer-to-peer off), temperature 0, median of 5.
Against release 1.0.0, one user now decodes at 264.6 tok/s instead of 238.7 on
structured text, 260.4 instead of 232.4 on code and 188.4 instead of 165.1 on
prose, and cold prefill at ~128k tokens runs at 2,425 tok/s instead of 2,149.

**Decode**, 400 max tokens. `Stream` is per request,
`(completion_tokens − 1) / (end − first token)`; `Agg` is
`sum(completion_tokens) / wall` across all streams. Each request carries a
unique nonce so nothing hits the prefix cache.

| Prompt type | Users | Stream tok/s | Aggregate tok/s | TTFT (cold) |
|---|---:|---:|---:|---:|
| Structured (count 1→200) | ×1 | **264.6** | **264.6** | 66 ms |
|  | ×2 | **206.2** | **386.0** | 113 ms |
|  | ×4 | **146.7** | **531.3** | 262 ms |
|  | ×8 | **104.5** | **745.4** | 343 ms |
| Code (clamp_00…clamp_49) | ×1 | **260.4** | **260.4** | 168 ms |
|  | ×2 | **177.4** | **308.5** | 264 ms |
|  | ×4 | **137.6** | **470.3** | 403 ms |
|  | ×8 | **97.3** | **664.5** | 635 ms |
| Prose (hash map) | ×1 | **188.4** | **188.4** | 69 ms |
|  | ×2 | **140.0** | **261.7** | 165 ms |
|  | ×4 | **104.4** | **384.3** | 265 ms |
|  | ×8 | **71.5** | **519.8** | 344 ms |

Prose decodes slower than structured or code text because the drafter's guesses
are accepted less often (0.58–0.60 against 0.84–0.98).

**Cold prefill**, unique uncached text, `max_tokens=1`, median of 2,
`prompt tokens / TTFT` measured client side.

| Prompt | TTFT | tok/s |
|---:|---:|---:|
| 7,978 | 3.27 s | **2,442.8** |
| 15,974 | 6.38 s | **2,502.2** |
| 31,931 | 12.78 s | **2,497.9** |
| 64,057 | 25.84 s | **2,478.6** |
| 127,586 | 52.62 s | **2,424.7** |
| 250,280 | 107.61 s | **2,325.7** |

---

## What runs

| | |
|---|---|
| API | OpenAI-compatible, `http://127.0.0.1:8000/v1`; this machine only, no API key, unless you [change that](#serving-other-machines) |
| Model id | `glm-5.3-flash` |
| Weights | [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) — INT4 weights, FP16 activations, group size 128 |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), 320B MoE |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), 3 draft tokens per step |
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) `ampere-glm53` @ [`3bbb99a534`](https://github.com/Morrowmake/vllm-cmp170hx/commit/3bbb99a5344c3a9cc7e0a1c8ff0d602263520ef5) |
| Layout | tensor-parallel 4 (`TP=4`, `PP=1`); assumes PCIe Gen2 x16 between the cards — see [Link width](#link-width) |
| Context | 262,144 tokens |
| KV cache | full precision, **not quantised**; 1,174,567 tokens at 262,144 context (1,187,776 with peer-to-peer on) |
| Prefill | 3,456-token chunks; long prompts yield to running requests ([fair prefill](#what-makes-it-fast-and-correct)) |
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
- **Fused decode kernels.** The mHC mixing (the model's hyper-connection
  residual streams), MoE routing and block alignment,
  and the linear-attention (KDA) decode each run as fused kernels built for the
  small batches decode actually sees. This release adds a second generation:
  the MoE gate, top-k and alignment in one launch, a faster mHC decode, KDA
  decode with its gate projections fused, and the indexer's decode glue folded
  into fewer kernels.
- **Tuned small-batch GEMMs.** W4A16 leaves some layers in BF16, where cuBLAS is
  slow with very few rows. A thin-batch kernel with per-shape tuning covers
  batches up to 32 rows, now including 24-row batches.
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
- **Prefill overlap.** During prefill each layer's all-reduces run on a side
  stream while the MoE computes, and it keeps working alongside the
  peer-to-peer path.
- **Fair prefill.** A long prompt no longer starves users who are mid-answer:
  while anyone is decoding, prefill is taken in small slices. Decode speed
  during someone else's long prompt went from 7% to 18% of normal.
- **Bigger prefill chunks.** 3,456-token chunks instead of 1,152: +8.0% cold
  prefill going to 2,304, and +2.4% more going to 3,456 (at 180 W per card).
- **Same request, same output.** A request on its own returns the same tokens
  and the same log-probabilities every time, across restarts. Four sources of
  run-to-run variation are fixed: MoE block alignment in a fixed order,
  CUDA-graph padding rows kept out of the MoE, and the indexer's top-k made
  consistent on ties and returned in a fixed order. On by default; they cost
  less than restart-to-restart variation.
- **Every custom kernel checked against a high-precision reference.** Each
  kernel the fork adds on the default path is replayed on real inputs captured
  from a running server and compared with a 64-bit reference, side by side
  with the upstream or PyTorch code it replaces. The worst mean error in this
  release is 1.08× that of the replaced code, and several kernels are more
  accurate than it. Three kernels that fell short were fixed for this release,
  including one prefill kernel shipped since 1.0.0.
- **A real correctness fix in the key pool.** With speculative decoding, a
  rejected draft could overwrite the tail of the sparse-attention key pool,
  leaving a wrong key in the indexer cache once a conversation passed 2,048
  tokens. The tail is now sized for the draft depth, at no memory cost.
- **KV headroom.** Workspaces sized to what a step can actually use, and the
  drafter's selector tables split across the cards, return memory to the KV
  pool (+39,626 tokens) without changing a single output bit.
- **64-bit KV row offsets** in the sparse-attention kernels, closing a silent
  corruption risk in very large KV pools, and a vocabulary clamp in the sampler
  kernels.

---

## How to use this repo

### What you need

| | |
|---|---|
| GPUs | 4× NVIDIA CMP 170HX, **each exposing 64 GiB** (`nvidia-smi` shows 65,536 MiB) **on a PCIe Gen2 x16 link**. Stock cards expose less memory and run at x4; getting to 64 GiB and x16 is outside this repository. The preflight stops before any download if a card reports under 60 GiB, and warns below x16 ([Link width](#link-width)) |
| OS and driver | Linux (the commands below are for Ubuntu) with NVIDIA driver **580 or newer**, which the CUDA 13 build of PyTorch needs; `nvidia-smi` must list all four cards |
| CUDA | a 13.x toolkit at `/usr/local/cuda-13.3`, or set `CUDA_HOME` ([CUDA](#cuda)) |
| Tools | `uv`, `git`, `curl`, Python 3.12, `flock` and `setsid` (util-linux, on most systems already), `jq` for the smoke test, `wget` for the CUDA commands |
| Disk | about 180 GiB (193 GB) for the two checkpoints, plus about 20 GiB for the venv, the engine source and compile caches |

### Step by step

`./start.sh` on its own does steps 1–3 in order and skips anything already
done, so running it twice is safe. The same steps one at a time:

```bash
./install.sh       # 1. build ./venv and the pinned vLLM fork (about six minutes)
./download.sh      # 2. fetch the model (~178 GiB) and the drafter (~2.2 GiB) into ./models
./start.sh         # 3. launch in the background, wait for /health, print the KV pool size
./start.sh smoke   # 4. one chat request and one tool call against the running server
./start.sh stop    # stop it; weights and venv stay, so the next start is quick
```

Weight loading and CUDA-graph capture take a few minutes after the download.
`./serve.sh` runs the server in the foreground instead of step 3 if you prefer;
it reads its settings from the environment, not from `.env`. A server started
that way is invisible to `./start.sh status`, `stop` and `restart`: stop it with
Ctrl-C before using `./start.sh` again.

**Smoke test output** on this release looks like this (this sample has
peer-to-peer on; with the default, the KV line reads 1,174,567 tokens and
4.48x):

```
==> waiting for http://127.0.0.1:8000/health (up to 900s)
    healthy
    served models: glm-5.3-flash

==> chat request
    reply: PCIe peer-to-peer (P2P) allows GPUs to transfer data directly to each other's memory over the PCIe bus without routing through host memory, avoiding costly stag ...
    usage: prompt=28 completion=121 wall=0.87s  ->  139.8 tok/s

==> tool-call request
    tool_call: get_weather({"city": "Reykjavik", "unit": "celsius"})
    usage: prompt=199 completion=66 wall=0.44s  ->  148.7 tok/s

==> KV cache
    GPU KV cache size: 1,187,776 tokens, Maximum concurrency for 262,144 tokens per request: 4.53x

smoke: ok
```

The rates in the smoke test include the time to first token of a short
request, so they read lower than the decode table.

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
| `MAX_LEN` | `262144` | context ceiling; lower it for more concurrent full-length requests |
| `MAX_SEQS` | `8` | concurrent requests |
| `GPU_UTIL` | `0.95` | share of each card's memory the engine may use |
| `SPEC_MODE` / `SPEC_N` | `dflash` / `3` | speculative drafter and draft depth; `mtp` uses the MTP head in the model checkpoint, `none` turns speculation off |
| `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE` | `0` | the [peer-to-peer](#pcie-peer-to-peer-optional) switch |
| `MAX_BATCHED` | `3460` | sets the prefill chunk: 3,456 tokens; `2048` gives 1,152 |
| `FAIR_PREFILL` / `FAIR_CHUNK` | `1` / `384` | fair prefill and its slice size while others decode |
| `PREFILL_CAP` | `0` | upstream's unconditional chunk cap. **Leave it 0**: it cost 15% prefill here and turns the prefill features off |
| `MM_CAP` | `0` | `1` bounds image and video inputs, which returns roughly 150k KV tokens |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | where it listens; see [Serving other machines](#serving-other-machines) |
| `API_KEY` | unset | bearer key required on `/v1`; unset means no key |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | the model id clients send |
| `MODELS_DIR` | `models/` in this repo | where the checkpoints go; give an absolute path |
| `CUDA_HOME` | `/usr/local/cuda-13.3` | CUDA toolkit root |
| `READY_TIMEOUT` | `1800` | seconds `./start.sh` waits for `/health` |
| `EXTRA_ARGS` | unset | appended to the `vllm serve` command line |
| `HF_TOKEN` | unset | a Hugging Face token makes the download faster |

Every setting in `.env.example` is commented out and shows its default;
uncommenting one overrides that default. A `.env` you have not edited
therefore picks up a later release's new defaults without changes. To move
to a later release, run `./start.sh update`.

### PCIe peer-to-peer (optional)

**What it is.** The four cards exchange data after every layer. By default that
goes through host memory, because a stock CMP 170HX refuses GPU peer access:
`nvidia-smi topo -p2p r` answers `GNS` on every pair. Where peer access *is*
available, the all-reduce can run card to card in device memory instead, which
is faster.

**The default is off.** With `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0` the recipe
needs no driver change of any kind and uses the host-staged path described
above. If you do nothing, this is what you run.

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

**What you gain.** On this release, the two columns of the
[results table](#release-130): step time −3.7% at one user, −6.9% at four,
−8.3% at six and −6.2% at eight; single-user decode +3.9% to +5.0%; eight-user
aggregate +7.0% to +8.4%; cold prefill unchanged (2,484 → 2,490 tok/s); and
13,209 more KV tokens.

**Turn it on.**

```bash
nvidia-smi topo -p2p r              # every pair must say OK, not GNS
# then set VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1 in .env, or for one run:
VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1 ./start.sh restart
```

`serve.sh` ties the rest to that one variable: it selects the `2stage`
all-reduce kernel (upstream's default crossover is tuned for NVLink) and the
caching-allocator mode the peer-to-peer path needs.

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
delete the line) and `./start.sh restart`. That restores the host-staged path
and the default allocator in one step.

### Kill switches

Each feature is one variable. Set it to `0` and restart; no rebuild, no
revert. Two work differently: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` is switched
off with `512`, and `VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY` is switched *on*
(`1`) to go back to the old schedule.

```bash
VLLM_GLM5_DECODE_KERNELS=0 ./start.sh restart
```

| Variable | Feature |
|---|---|
| `VLLM_GLM5_PREFILL_OVERLAP` | prefill overlap |
| `VLLM_GLM5_PREFILL_KERNELS` | Ampere prefill kernels |
| `VLLM_GLM5_PROLOGUE_FUSE` | fused decode prologue |
| `VLLM_GLM5_LOCAL_LOGITS` | batch-sharded logits and sampling |
| `VLLM_GLM5_DECODE_KERNELS` | fused decode kernels |
| `VLLM_GLM5_DECODE_IDX_GLUE`, `VLLM_GLM5_DECODE_KDA_V2`, `VLLM_GLM5_DECODE_MOE_ROUTE_V2`, `VLLM_GLM5_DECODE_MHC_V2` | second-generation decode kernels |
| `VLLM_GLM5_DRAFTER_ROPE_FIT` | drafter position table sized to the context (more KV) |
| `VLLM_GLM5_THIN_GEMM` | small-batch GEMMs |
| `VLLM_GLM5_HOST_ALLREDUCE` | host-staged all-reduce |
| `VLLM_GLM5_SHARED_EXPERT_REORDER` | shared experts overlapped with the routed experts |
| `FAIR_PREFILL` | fair prefill |
| `VLLM_GLM5_DETERMINISTIC_MOE_ALIGN` | same output every time: MoE block alignment in a fixed order |
| `VLLM_GLM5_MOE_MASK_PADDING` | same output every time: CUDA-graph padding rows kept out of the MoE |
| `VLLM_GLM5_TOPK_TIEFIX`, `VLLM_GLM5_TOPK_SORTED` | same output every time: indexer top-k consistent on ties, in a fixed order |
| `VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS` | the two above spread over more programs for batches up to `8` rows; `0` = one per row |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | KV headroom: prefill indexer logits budget, `128` here; `512` (upstream's) switches it off |
| `VLLM_GLM5_DRAFTER_SELECTOR_SHARD` | KV headroom: drafter selector tables split across the cards |
| `VLLM_GLM5_INDEXER_DECODE_ROWS` | KV headroom: indexer decode tables sized by the decode rows |
| `VLLM_GLM5_INDEXER_GATHER_CLAMP` | KV headroom: indexer gather workspace clamp (on in the engine itself) |
| `VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY` | set to `1` for the sparse-attention decode schedule from before the retune (default `0`, set in the engine) |

The prefill overlap, batch-sharded logits, host-staged all-reduce, the
second-generation decode kernels, the drafter position table and the KV
headroom switches are tensor-parallel only. `DRY=1 ./start.sh` prints the
environment the server would get, so you can check what is on. It runs the
preflight and says whether a real start would install or download, but
installs, downloads and launches nothing, so it also works before the first
install.

### Day to day

| Command | |
|---|---|
| `./start.sh` | preflight → install → download → launch → wait for `/health` |
| `./start.sh restart` | stop, then start (picks up `.env` changes) |
| `./start.sh status` | process, `/health`, KV line, install and checkpoint state |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh smoke` | one chat request and one tool call |
| `./start.sh stop` | stop the server this checkout started |
| `./start.sh update` | `git pull`, reinstall if the engine pin moved, restart |
| `VLLM_COMMIT=<older sha> ./start.sh update` | roll back to an earlier engine, for that run only |
| `DRY=1 ./start.sh` | print what would be installed, downloaded and launched, without doing it |

`stop` only signals the process in `logs/vllm.pid`, after confirming it is the
server this checkout launched; it never searches by process name, so another
vLLM on the same machine is never touched.

The engine pin lives in `start.sh`, so a `git pull` can move it, and `start.sh`
reinstalls whenever it changes: the compiled extensions must match the upstream
commit the fork sits on. Uncomment `VLLM_COMMIT` in `.env` to freeze it.

**Rolling back.** `VLLM_COMMIT=<sha> ./start.sh update` rolls the engine back
for that run only; the next plain `./start.sh` or `restart` sees the pin and
reinstalls this release's engine. To stay rolled back, set
`VLLM_COMMIT=<sha>` in `.env`. That changes the engine only; to go back to an
earlier release's scripts and defaults as well, check out its tag
(`git checkout v1.2.0`, then `./start.sh restart`; back again with
`git checkout main` and `./start.sh update`). Engine pins from 1.1.0 on
(`69c33802d0`, `434dea1a1b`) can be installed; 1.0.x's pin predates a rebase
of the fork branch and cannot.

### CUDA

You need a 13.x toolkit. These commands are for Ubuntu. NVIDIA had no working
`ubuntu2604` repository index when we built this, so we used the `ubuntu2404`
one:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3
```

The install uses upstream's **precompiled** CUDA extensions, which is why it
takes minutes: the fork's patches are Python, Triton and TileLang and touch no
CUDA or C++ source, so the compiled objects are the same, and upstream's
already carry sm_80 code. To compile them yourself:
`BUILD_FROM_SOURCE=1 MAX_JOBS=16 ./install.sh` (full toolkit, ~60 GB of
scratch, one to two hours).

### Link width

Tensor parallelism moves about 9.4 MB per layer between the cards during
prefill and roughly 100 small collectives per decode step. It assumes PCIe Gen2
x16 links; on x4 links it is bus-bound and far slower than the numbers above.
`./start.sh` prints each card's link width in its preflight and warns if any is
narrower than x16. For narrower links, see the [roadmap](#status-and-roadmap).

---

## Status and roadmap

- **Tensor-parallel 4 is the supported layout in 1.3.0.** It gives each request
  the fastest answer and suits one or two interactive users or agents.
- **Pipeline-parallel 4 is in active development** for systems without the x16
  capacitor modification, whose cards run on narrower PCIe links. Each card
  holds a quarter of the layers and passes only activations to the next, so it
  needs far less link bandwidth and leaves more memory for KV. It is aimed at
  many parallel agents and long prompts, and it comes in a later release with
  its own validation and numbers. Until then, another pipeline-parallel recipe
  for these cards is
  [JJ48/glm53-flash-170hx-serving](https://github.com/JJ48/glm53-flash-170hx-serving).
- **Optimisation continues** on both layouts; every change ships with a kill
  switch and measured numbers.

---

## Known limits

- **Ampere only.** Every kernel in the fork is written for sm_80. On newer GPUs
  upstream vLLM's own kernels are better, and forcing these features on
  elsewhere is untested.
- **Wide links for tensor-parallel.** See [Link width](#link-width).
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

## Licence

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
model checkpoint instead and does not use it at all.

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
