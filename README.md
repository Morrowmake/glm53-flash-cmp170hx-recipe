<h1 align="center">GLM-5.3-Flash on 4× NVIDIA CMP 170HX</h1>

<p align="center">
  <strong>by <a href="https://x.com/Morrowmake">Morrowmake</a></strong>
  <br><br>
  <a href="https://x.com/Morrowmake"><img alt="Follow on X" src="https://img.shields.io/badge/Follow-%40Morrowmake-000000?style=flat&logo=x&logoColor=white"></a>
  &nbsp;
  <a href="https://github.com/Morrowmake/vllm-cmp170hx/tree/PIN_PENDING"><img alt="engine" src="https://img.shields.io/badge/engine-vLLM%20fork%20%40%20PIN_PENDING-4b32c3?style=flat"></a>
  &nbsp;
  <img alt="release" src="https://img.shields.io/badge/release-1.4.0-2ea44f?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/recipe-MIT-blue?style=flat">
</p>

**A 320B-parameter MoE with a 262,144-token context, served on four CMP 170HX
cards in two layouts: tensor-parallel at {{TP4_OFF_1U_STRUCT}} tok/s for one
user and {{TP4_OFF_8U_STRUCT}} tok/s across eight, or pipeline-parallel with
{{PP4_PREFILL}} tok/s cold prefill and a {{PP4_KV}}-token KV pool. The weights
are W4A16 and nothing else is cut: the KV cache is full precision, there is no
FP8 anywhere, and nothing is offloaded to CPU or disk. A request sent on its
own gives the same output every time, on every install.**

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

### Release 1.4.0

DFlash2 at k=3, 262,144-token context, the defaults in this repository. Three
columns: tensor-parallel 4 with the cards talking through the host (the
default), tensor-parallel 4 with the optional
[PCIe peer-to-peer](#pcie-peer-to-peer-optional) path, and pipeline-parallel 4
(`LAYOUT=pp4`, {{PP4_P2P_STATE}}).

| | TP4, peer-to-peer off (default) | TP4, peer-to-peer on (optional) | PP4 (`LAYOUT=pp4`) |
|---|---:|---:|---:|
| Decode, 1 user, structured / code / prose | **{{TP4_OFF_1U}} tok/s** | **{{TP4_ON_1U}} tok/s** | **{{PP4_1U}} tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **{{TP4_OFF_8U}} tok/s** | **{{TP4_ON_8U}} tok/s** | **{{PP4_8U}} tok/s** |
| Decode step, 1 / 4 / 6 / 8 users | {{TP4_OFF_STEPS}} ms | {{TP4_ON_STEPS}} ms | {{PP4_STEPS}} ms |
| Cold prefill | **{{TP4_OFF_PREFILL}} tok/s** | **{{TP4_ON_PREFILL}} tok/s** | **{{PP4_PREFILL}} tok/s** |
| Time to first token, 6,217 / 23,255-token prompt | {{TP4_OFF_TTFT}} s | {{TP4_ON_TTFT}} s | {{PP4_TTFT}} s |
| KV pool at 262,144 context | {{TP4_OFF_KV}} tokens ({{TP4_OFF_KV_X}} full-length requests) | {{TP4_ON_KV}} tokens ({{TP4_ON_KV_X}}) | {{PP4_KV}} tokens ({{PP4_KV_X}}) |

All at 180 W per card (a power limit we set on our cards; the scripts never
change power, clock or fan settings), PCIe x16 links, {{RELEASE_BOOTS}}. Decode
tok/s is the per-request streaming rate on three fixed prompt types —
structured, code and prose (400 tokens, temperature 0, median of 5); prose is
slower because the drafter's guesses are accepted less often. Cold prefill is
the median over real-text prompts of 23.9K to 37.9K tokens. The KV pool is the
size the server can actually fill with prefill chunks in flight (see
[Honest KV figures](#measured-while-building-this-release)).

**Quality**, measured on each layout's default:

| | TP4 (peer-to-peer off) | PP4 |
|---|---:|---:|
| Perplexity, fixed 60-document set | **{{TP4_PPL}}** | **{{PP4_PPL}}** |
| GSM8K, all 1,319 problems | **{{TP4_GSM8K}}** | **{{PP4_GSM8K}}** |
| HumanEval, all 164, pass@1 | **{{TP4_HUMANEVAL}}** | **{{PP4_HUMANEVAL}}** |

HumanEval allows 4,096 tokens per reply and scores the last complete fenced code
block of the reply, reasoning included. {{HUMANEVAL_NOTES}}

### Choosing a layout

| | Tensor-parallel 4 (`tp4`, default) | Pipeline-parallel 4 (`pp4`) |
|---|---|---|
| Each card holds | a quarter of every layer | a quarter of the layers |
| Best for | one or two interactive users: the fastest answer per request | many parallel users or clients, long prompts, large shared contexts |
| Prefill | {{TP4_OFF_PREFILL}} tok/s | {{PP4_PREFILL}} tok/s ({{PP4_PREFILL_RATIO}}×) |
| KV pool | {{TP4_OFF_KV}} tokens | {{PP4_KV}} tokens ({{PP4_KV_RATIO}}×) |
| Traffic between cards | ~9.4 MB per layer during prefill, ~100 small collectives per decode step | activations only, once per stage |
| Links | PCIe x16 | built for x4; measured on x16 |

Both run the same model, the same drafter and the same repeatable-output fixes;
switching is `LAYOUT=pp4 ./start.sh restart` and back with `LAYOUT=tp4`.

### Measured while building this release

These come from development runs on the same four cards. They are not the
release table above; each line says what it was measured against.

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
  against a 64-bit reference. {{TP4_PREFILL_GAIN_LINE}}
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
  {{CONTAINER_PARITY_LINE}}
- **A newer upstream.** The fork now sits on upstream vLLM's 0.30.1 development
  line (FlashInfer 0.7.0).

### Decode and prefill in detail

Release 1.4.0, tensor-parallel 4 with the defaults (peer-to-peer off),
temperature 0, median of 5. Against release 1.0.0, one user now decodes at
{{TP4_OFF_1U_STRUCT}} tok/s instead of 238.7 on structured text,
{{TP4_OFF_1U_CODE}} instead of 232.4 on code and {{TP4_OFF_1U_PROSE}} instead
of 165.1 on prose, and cold prefill at ~128k tokens runs at
{{TP4_OFF_PREFILL_128K}} tok/s instead of 2,149.

**Decode**, 400 max tokens. `Stream` is per request,
`(completion_tokens − 1) / (end − first token)`; `Agg` is
`sum(completion_tokens) / wall` across all streams. Each request carries a
unique nonce so nothing hits the prefix cache.

| Prompt type | Users | Stream tok/s | Aggregate tok/s | TTFT (cold) |
|---|---:|---:|---:|---:|
| Structured (count 1→200) | ×1 | **{{D_S1_STREAM}}** | **{{D_S1_AGG}}** | {{D_S1_TTFT}} ms |
|  | ×2 | **{{D_S2_STREAM}}** | **{{D_S2_AGG}}** | {{D_S2_TTFT}} ms |
|  | ×4 | **{{D_S4_STREAM}}** | **{{D_S4_AGG}}** | {{D_S4_TTFT}} ms |
|  | ×8 | **{{D_S8_STREAM}}** | **{{D_S8_AGG}}** | {{D_S8_TTFT}} ms |
| Code (clamp_00…clamp_49) | ×1 | **{{D_C1_STREAM}}** | **{{D_C1_AGG}}** | {{D_C1_TTFT}} ms |
|  | ×2 | **{{D_C2_STREAM}}** | **{{D_C2_AGG}}** | {{D_C2_TTFT}} ms |
|  | ×4 | **{{D_C4_STREAM}}** | **{{D_C4_AGG}}** | {{D_C4_TTFT}} ms |
|  | ×8 | **{{D_C8_STREAM}}** | **{{D_C8_AGG}}** | {{D_C8_TTFT}} ms |
| Prose (hash map) | ×1 | **{{D_P1_STREAM}}** | **{{D_P1_AGG}}** | {{D_P1_TTFT}} ms |
|  | ×2 | **{{D_P2_STREAM}}** | **{{D_P2_AGG}}** | {{D_P2_TTFT}} ms |
|  | ×4 | **{{D_P4_STREAM}}** | **{{D_P4_AGG}}** | {{D_P4_TTFT}} ms |
|  | ×8 | **{{D_P8_STREAM}}** | **{{D_P8_AGG}}** | {{D_P8_TTFT}} ms |

Prose decodes slower than structured or code text because the drafter's guesses
are accepted less often.

**Cold prefill**, unique uncached text, `max_tokens=1`, median of 2,
`prompt tokens / TTFT` measured client side, both layouts.

| Prompt | TP4 TTFT | TP4 tok/s | PP4 TTFT | PP4 tok/s |
|---:|---:|---:|---:|---:|
| ~8k | {{PF_T_8K_TTFT}} s | **{{PF_T_8K}}** | {{PF_P_8K_TTFT}} s | **{{PF_P_8K}}** |
| ~16k | {{PF_T_16K_TTFT}} s | **{{PF_T_16K}}** | {{PF_P_16K_TTFT}} s | **{{PF_P_16K}}** |
| ~32k | {{PF_T_32K_TTFT}} s | **{{PF_T_32K}}** | {{PF_P_32K_TTFT}} s | **{{PF_P_32K}}** |
| ~64k | {{PF_T_64K_TTFT}} s | **{{PF_T_64K}}** | {{PF_P_64K_TTFT}} s | **{{PF_P_64K}}** |
| ~128k | {{PF_T_128K_TTFT}} s | **{{PF_T_128K}}** | {{PF_P_128K_TTFT}} s | **{{PF_P_128K}}** |
| ~250k | {{PF_T_250K_TTFT}} s | **{{PF_T_250K}}** | {{PF_P_250K_TTFT}} s | **{{PF_P_250K}}** |

---

## What runs

| | |
|---|---|
| API | OpenAI-compatible, `http://127.0.0.1:8000/v1`; this machine only, no API key, unless you [change that](#serving-other-machines) |
| Model id | `glm-5.3-flash` |
| Weights | [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) — INT4 weights, FP16 activations, group size 128 |
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), 320B MoE |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), 3 draft tokens per step |
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) `ampere-glm53` @ [`PIN_PENDING`](https://github.com/Morrowmake/vllm-cmp170hx/commit/PIN_PENDING), on upstream vLLM `e55d076f89` |
| Container image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:DIGEST_PENDING` — the engine at that pin, no weights ([docker/](docker/README.md)) |
| Layout | tensor-parallel 4 (`LAYOUT=tp4`, default; assumes PCIe Gen2 x16) or pipeline-parallel 4 (`LAYOUT=pp4`) — see [Choosing a layout](#choosing-a-layout) |
| Context | 262,144 tokens |
| KV cache | full precision, **not quantised**; TP4 {{TP4_OFF_KV}} tokens at 262,144 context ({{TP4_ON_KV}} with peer-to-peer on), PP4 {{PP4_KV}} |
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
| Disk | about 180 GiB (193 GB) for the two checkpoints, plus the engine image ({{IMAGE_SIZE}} compressed) and the kernel compile caches |

### Step by step

`./start.sh` on its own does steps 1–3 in order and skips anything already
done, so running it twice is safe. The same steps one at a time:

```bash
./install.sh       # 1. pull the engine image, pinned by digest ({{IMAGE_SIZE}} compressed)
./download.sh      # 2. fetch the model (~178 GiB) and the drafter (~2.2 GiB) into ./models
./start.sh         # 3. start the container, wait for /health, print the KV pool size
./start.sh smoke   # 4. one chat request and one tool call against the running server
./start.sh stop    # stop it; the image, weights and caches stay, so the next start is quick
```

The container runs the engine image with this repository's `serve.sh` as its
entry point, the checkpoints mounted read-only and the kernel compile caches in
`./cache`. It publishes the API on `127.0.0.1:8000` only. Its output goes to
`logs/serve.log`, as a native start's does, and `./start.sh stop` stops only the
container this checkout started. The first boot builds the compile caches and
takes {{FIRST_BOOT_MIN}} minutes to become healthy; later boots take about
{{LATER_BOOT_MIN}} minutes (weight loading and CUDA-graph capture).

**Smoke test output** on this release looks like this ({{SMOKE_SAMPLE_NOTE}}):

```
{{SMOKE_SAMPLE}}
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

**What you gain.** On this release, the first two columns of the
[results table](#release-140): {{P2P_GAIN_LINE}}

**Turn it on.**

```bash
nvidia-smi topo -p2p r              # every pair must say OK, not GNS
# then set VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1 in .env, or for one run:
VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1 ./start.sh restart
```

`serve.sh` ties the rest to that one variable: it selects the `2stage`
all-reduce kernel (upstream's default crossover is tuned for NVLink), the
caching-allocator mode the peer-to-peer path needs{{NCCL_SYS_README}}.

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
revert. A few work differently: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` is switched
off with `512`, `VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY` is switched *on* (`1`) to
go back to the old schedule, and `VLLM_PP_DRAFT_TAIL_STAGE` takes a stage
number ({{DRAFT_TAIL_README}}).

```bash
VLLM_GLM5_DECODE_KERNELS=0 ./start.sh restart
```

| Variable | Feature | Layout |
|---|---|---|
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
| `VLLM_PP_DRAFT_TAIL_STAGE` | {{DRAFT_TAIL_ROW}} | PP4 |

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

## Licence

**This recipe** — the scripts and the documentation — is MIT, © 2026 Morrowmake.
See [LICENSE](LICENSE).

**The vLLM fork** it installs is Apache-2.0, like upstream vLLM; our patches are
contributed under that licence. The container image carries the same licence
label.

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
