<h1 align="center">GLM-5.3-Flash on 4× NVIDIA CMP 170HX</h1>

<p align="center">
  <strong>by <a href="https://x.com/Morrowmake">Morrowmake</a></strong>
  <br><br>
  <a href="https://x.com/Morrowmake"><img alt="Follow on X" src="https://img.shields.io/badge/Follow-%40Morrowmake-000000?style=flat&logo=x&logoColor=white"></a>
  &nbsp;
  <a href="https://github.com/Morrowmake/vllm-cmp170hx/tree/3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340"><img alt="engine" src="https://img.shields.io/badge/engine-vLLM%20fork%20%40%203a2bf16dae-4b32c3?style=flat"></a>
  &nbsp;
  <img alt="release" src="https://img.shields.io/badge/release-1.6.0-2ea44f?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/recipe-MIT-blue?style=flat">
</p>

**A 320B-parameter MoE with a 262,144-token context, served on four CMP 170HX
cards in two layouts. Native 1.6.0 results: tensor-parallel at 394.0 tok/s for one
user and 797.9 tok/s across eight, or pipeline-parallel with
6,580 tok/s cold prefill and a 1,914,216-token KV pool. The weights
are W4A16 and nothing else is cut: the KV cache is full precision, there is no
FP8 anywhere, and nothing is offloaded to CPU or disk. Single-request
outputs were identical across the repeated checks in [Results](docs/results.md).**

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

## What you need

- **4× NVIDIA CMP 170HX, each exposing 64 GiB**; **PCIe Gen2 x16 links** for
  tensor-parallel 4 (pipeline-parallel 4 is built for narrower links)
- Linux with NVIDIA driver **580 or newer**
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (not needed for the [native install](docs/how-to-use.md#native-install-for-developers))
- `git`, `curl`, `flock`, `setsid`, and `jq` for the smoke test
- about 180 GiB (193 GB) of disk for the two checkpoints, plus the engine image
  (about 10.3 GB compressed) and the kernel compile caches ([details](docs/how-to-use.md#what-you-need))

## Headline numbers

Release 1.6.0, native, DFlash2 (adaptive depth: up to 7 drafts at one request,
up to 5 at two, 3 under load), 262,144-token context, peer-to-peer off (the
default), 180 W per card, PCIe x16 links.

| | Tensor-parallel 4 (default) | Pipeline-parallel 4 (`LAYOUT=pp4`) |
|---|---:|---:|
| Streaming decode, 1 user, structured / code / prose | **394.0 / 377.4 / 180.5 tok/s** | **235.3 / 207.9 / 100.2 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **797.9 / 683.2 / 544.7 tok/s** | **623.1 / 591.0 / 465.8 tok/s** |
| Cold prefill | **2,669 tok/s** | **6,580 tok/s** |
| KV pool at 262,144 context | 1,072,150 tokens | 1,914,216 tokens |
| HumanEval pass@1 / GSM8K (1.6.0 decode kernels, depth ≤ 5; [details](docs/results.md#release-160-faster-decode-for-one-or-two-users)) | 160/164 / 97.04% | 162/164 / 97.04% |

With the optional [PCIe peer-to-peer](docs/how-to-use.md#pcie-peer-to-peer-optional)
path, tensor-parallel 4 measured 437.4 tok/s for one user, 840.7 across eight
and 3,061 tok/s cold prefill. Method, the per-user decode table, the prefill
ladder up to ~250k tokens and quality detail: [Results](docs/results.md).

## Quick start

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe
./start.sh          # pull the engine image, fetch the model, start, wait for /health
./start.sh smoke    # one chat request and one tool call
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-flash",
       "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
       "max_tokens": 400}'
```

`./start.sh` skips anything already done, so running it twice is safe; the
first boot after an install takes about 4–5 minutes longer than later ones.
Any OpenAI-compatible client works with base URL `http://127.0.0.1:8000/v1`
and model `glm-5.3-flash`. The API listens on this machine only and has no
API key unless you [change that](docs/how-to-use.md#serving-other-machines).
Prefer Docker Compose? [`docker-compose.yml`](docker-compose.yml) starts the
same container ([Docker Compose](docs/how-to-use.md#docker-compose)). Python,
streaming and tool-call clients are in [examples/](examples/README.md).

## Choosing a layout

Tensor-parallel 4 (`tp4`, the default) gives one or two interactive users the
fastest answer per request and needs x16 links. Pipeline-parallel 4 (`pp4`)
suits many parallel users, long prompts and x4 links, with 2.47× the prefill
and 1.79× the KV pool. Switch with `LAYOUT=pp4 ./start.sh restart` (or set it
in `.env`); details in [Choosing a layout](docs/results.md#choosing-a-layout).

## Documentation

- [Results](docs/results.md): throughput, quality, decode and prefill in detail
- [How to use this repo](docs/how-to-use.md): install, settings, peer-to-peer,
  kill switches, day to day, updates and rollback, native install, containers
- [Optional compiled Marlin](docs/compiled-marlin.md): defaults, switches, building it
- [How it works](docs/how-it-works.md): what makes it fast and correct, and what runs
- [Engine switches](docs/engine-switches.md): the engine's internal switches, for troubleshooting
- [Status and known limits](docs/known-limits.md)
- [Examples](examples/README.md): curl, Python, streaming, tool calls, layout `.env` files
- [Container image](docker/README.md) and [CHANGELOG.md](CHANGELOG.md)
- Found a problem? [Open an issue](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/issues/new/choose);
  the bug template asks for the version, layout, peer-to-peer setting and log lines we need

## Update and roll back

- `./start.sh update` pulls the new release and engine and restarts; your `.env` is preserved.
- `IMAGE=<earlier image> ./start.sh update` rolls the engine back for that run; set `IMAGE` in `.env` to stay there.
- More, including earlier releases' scripts: [Day to day](docs/how-to-use.md#day-to-day).

---

## Licences

**This recipe** — the scripts and the documentation — is MIT, © 2026 Morrowmake.
See [LICENSE](LICENSE).

**The vLLM fork** is [Apache-2.0](https://github.com/Morrowmake/vllm-cmp170hx/blob/3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340/LICENSE).

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

DFlash2 is the only supported speculative mode; every figure on this page uses
it. Because its licence is non-commercial, commercial users need to make their
own evaluation ([Known limits](docs/known-limits.md)).

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
