# How to use this repo

## What you need

| | |
|---|---|
| GPUs | 4× NVIDIA CMP 170HX, **each exposing 64 GiB** (`nvidia-smi` shows 65,536 MiB). For tensor-parallel 4, **PCIe Gen2 x16 links**; pipeline-parallel 4 is built for narrower links. Stock cards expose less memory and run at x4; getting to 64 GiB and x16 is outside this repository. The preflight stops before any download if a card reports under 60 GiB, and warns below x16 ([Link width](#link-width)) |
| OS and driver | Linux (the commands below are for Ubuntu) with NVIDIA driver **580 or newer**; `nvidia-smi` must list all four cards |
| Container runtime | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), usable by your user, so that `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi` lists all four cards. Not needed for the [native install](#native-install-for-developers) |
| Tools | `git`, `curl`, `flock` and `setsid` (util-linux, on most systems already), `jq` for the smoke test |
| Disk | about 180 GiB (193 GB) for the two checkpoints, plus the engine image (about 10.6 GB compressed) and the kernel compile caches |

## Step by step

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

## Choose the layout

```bash
LAYOUT=pp4 ./start.sh restart     # pipeline-parallel 4, this run
# or set LAYOUT=pp4 in .env and ./start.sh restart to keep it
```

Everything else — the drafter, the context, the kill switches, the API — works
the same in both layouts. `./start.sh status` shows which one is running.

## Talk to it

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

## Serving other machines

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

## Settings

Everything lives in `.env`, copied from [`.env.example`](../.env.example) on first
run; `./start.sh` reads only `.env`. `.env.example` holds the settings people
actually change. Everything else is in
[`.env.advanced.example`](../.env.advanced.example), which you almost certainly
don't need: to use one of its settings, copy the line into your `.env`. The
engine's internal switches are described in [Engine switches](engine-switches.md).
A value on the command line beats `.env` for that run:

```bash
MAX_LEN=131072 ./start.sh restart
```

In `.env.example`:

| Key | Default | What it does |
|---|---|---|
| `LAYOUT` | `tp4` | `tp4` tensor-parallel 4, `pp4` pipeline-parallel 4 ([Choosing a layout](results.md#choosing-a-layout)) |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | where it listens; see [Serving other machines](#serving-other-machines) |
| `API_KEY` | unset | bearer key required on `/v1`; unset means no key |
| `SERVED_MODEL_NAME` | `glm-5.3-flash` | the model id clients send |
| `MODELS_DIR` | `models/` in this repo | where the checkpoints go; give an absolute path |
| `RUNTIME` | `container` | `native` runs the engine from a venv instead ([Native install](#native-install-for-developers)); a checkout that already has a native install keeps it |
| `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE` | `0` | the [peer-to-peer](#pcie-peer-to-peer-optional) switch (TP4) |

Commonly used from `.env.advanced.example`:

| Key | Default | What it does |
|---|---|---|
| `MAX_LEN` | `262144` | context ceiling; lower it for more concurrent full-length requests |
| `MAX_SEQS` | `8` | concurrent requests |
| `GPU_UTIL` | `0.95` | share of each card's memory the engine may use |
| `SPEC_MODE` / `SPEC_N` | `dflash` / `3` | speculative drafter and draft depth; `mtp` uses the MTP head in the model checkpoint, `none` turns speculation off |
| `MAX_BATCHED` | `3460` (TP4), `2312` (PP4) | sets the prefill chunk: 3,456 / 2,304 tokens; `2048` gives 1,152 |
| `FAIR_PREFILL` / `FAIR_CHUNK` | `1` / `384` | fair prefill and its slice size while others decode |
| `PREFILL_CAP` | `0` | upstream's unconditional chunk cap. **Leave it 0**: it cost 15% prefill here and turns the prefill features off |
| `MM_CAP` | `0` | `1` bounds image and video inputs, which returns roughly 150k KV tokens |
| `IMAGE` | this release's image, by digest | the engine image (container) |
| `CONTAINER_NAME` | `glm53-flash` | the container's name |
| `READY_TIMEOUT` | `1800` | seconds `./start.sh` waits for `/health` |
| `EXTRA_ARGS` | unset | appended to the `vllm serve` command line |
| `HF_TOKEN` | unset | a Hugging Face token makes the download faster |

Every setting in both example files is commented out and shows its default;
uncommenting one in `.env` overrides that default. A `.env` you have not edited
therefore picks up a later release's new defaults without changes, and a
`.env` from an earlier release, with settings from either file, keeps working
unchanged. To move to a later release, run `./start.sh update`.

Write each setting as `KEY=value` on its own line. In an unquoted value,
anything after a space and `#` is a note and is ignored; quote the value
(`"..."`) to keep it exactly as written.

## Day to day

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
pulls the matching image whenever it changes. Set `IMAGE` in `.env` (see
[`.env.advanced.example`](../.env.advanced.example)) to
freeze it.

**Updating from 1.4.x.** Run `./start.sh update`. Release 1.5.0 moves the
engine and image pins; your `.env`, including explicit `IMAGE`, `VLLM_COMMIT`
and runtime-switch overrides, is preserved. Remove an old explicit pin only
if you want to follow this release. The image includes optional Marlin:
unset PP4 decode enables it, TP4 decode and both prefill defaults remain off.
Native users must opt in to compilation as described in [Optional compiled Marlin](compiled-marlin.md); without an installed
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

## Advanced

Not needed for a normal install: peer-to-peer, the engine's kill switches,
link width, running the container yourself and the native developer install.

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

Every performance feature can be switched off on its own with one variable,
without a rebuild. The full list, and how to use the switches for
troubleshooting, is in [Engine switches](engine-switches.md).

```bash
VLLM_GLM5_DECODE_KERNELS=0 ./start.sh restart
```

### Link width

Tensor parallelism moves about 9.4 MB per layer between the cards during
prefill and roughly 100 small collectives per decode step. It assumes PCIe Gen2
x16 links; on x4 links it is bus-bound and far slower than the [published numbers](results.md).
Pipeline parallelism passes only each stage's activations to the next card,
which is why it is the layout for narrower links. `./start.sh` prints each
card's link width in its preflight and warns if any is narrower than x16 while
the layout is tensor-parallel.

### Run the container by hand

[`docker/`](../docker/README.md) has the Dockerfile and the build script for the
image, and a `docker run` command with [`container.env`](../docker/container.env)
for running it without `./start.sh`.

### Docker Compose

[`docker-compose.yml`](../docker-compose.yml) at the root of this repository
starts the same container `./start.sh` launches: the release image by digest,
all four cards, the same shared-memory size, the API on `127.0.0.1:8000`
only, the checkpoints mounted read-only, `./cache` for the kernel compile
caches, and the server running as you with the same cache environment, from
[`container.env`](../docker/container.env). `./start.sh` does not use it; use
one or the other.

```bash
./install.sh && ./download.sh      # the image and the two checkpoints
mkdir -p cache
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d
docker compose logs -f             # wait for "Application startup complete"
docker compose down                # stop
```

`LAYOUT=pp4` (in the shell or in `.env`) selects pipeline-parallel 4; any
other setting goes in [`container.env`](../docker/container.env), as for the
`docker run` command.

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
