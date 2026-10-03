# How to use this repo

## What you need

| | |
|---|---|
| GPUs | 4× NVIDIA CMP 170HX, **each exposing 64 GiB** (`nvidia-smi` shows 65,536 MiB). For tensor-parallel 4, **PCIe Gen2 x16 links**; pipeline-parallel 4 is built for narrower links. Stock cards expose less memory and run at x4; getting to 64 GiB and x16 is outside this repository. The preflight stops before any download if a card reports under 60 GiB, and warns below x16 ([Link width](#link-width)) |
| OS and driver | Linux (the commands below are for Ubuntu) with NVIDIA driver **580 or newer**; `nvidia-smi` must list all four cards |
| Container runtime | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), usable by your user, so that `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi` lists all four cards. Not needed for the [native install](#native-install-for-developers) |
| Tools | `git`, `curl`, `flock` and `setsid` (util-linux, on most systems already), `jq` for the smoke test |
| Disk | about 180 GiB (193 GB) for the two checkpoints, plus the engine image ({{NUM:image_compressed_gb}} GB compressed) and the kernel compile caches |

## Step by step

`./start.sh` on its own does steps 1–3 in order and skips anything already
done, so running it twice is safe. The same steps one at a time:

```bash
./install.sh       # 1. pull the engine image, pinned by digest ({{NUM:image_compressed_gb}} GB compressed)
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
container this checkout started. The cold TP4 / PP4 boot takes {{NUM:tp4_boot_cold_s}} /
{{NUM:pp4_boot_cold_s}} s; matching warm-cache seeds reduce that to
{{NUM:tp4_boot_seeded_s}} / {{NUM:pp4_boot_seeded_s}} s.

**The first boot after an install or an update is slower.** FlashInfer 0.7.0
compiles its kernel modules into an empty cache unless a matching seed is
available. The cold/seeded measurements above include this work. Later boots reuse
them. While that build runs, the log can show lines like
`No available shared memory broadcast block found in 60 seconds`; they are
harmless and stop once the build finishes.

The smoke test checks the chat reply, tool call and KV cache; it prints
`smoke: ok` when those checks pass.


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
| `P2P` | `auto` | [content-verified peer access](#pcie-peer-to-peer) for TP and PP; `off` disables, `force` bypasses the check |
| `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE` | `1` after a pass, otherwise `0` | custom all-reduce (TP); an explicit `0` is preserved after a pass |

Commonly used from `.env.advanced.example`:

| Key | Default | What it does |
|---|---|---|
| `MAX_LEN` | `262144` | context ceiling; lower it for more concurrent full-length requests |
| `MAX_SEQS` | `8` | concurrent requests |
| `GPU_UTIL` | `0.95` | share of each card's memory the engine may use |
| `SPEC_N` | `3` | draft depth under load for the DFlash2 drafter (the only supported mode) |
| `MAX_BATCHED` | `3460` (TP4), `2312` (PP4) | sets the prefill chunk: 3,456 / 2,304 tokens; `2048` gives 1,152 |
| `FAIR_PREFILL` / `FAIR_CHUNK` | `1` / `384` | fair prefill and its slice size while others decode |
| `PREFILL_CAP` | `0` | upstream's unconditional chunk cap. **Leave it 0**: it turns the prefill features off |
| `MM_CAP` | `0` | `1` bounds image and video inputs, which reduces the multimodal memory reservation |
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

### Default reasoning effort

`DEFAULT_REASONING_EFFORT` sets the default chat-template kwarg to `low`, `high`
or `max` for requests that omit effort; empty disables this override.
The GLM chat template maps a missing reasoning effort to `max`.
The pinned engine supports `--default-chat-template-kwargs`; request-level
`reasoning_effort` or `chat_template_kwargs.reasoning_effort` takes precedence.
An explicit `--default-chat-template-kwargs` in `EXTRA_ARGS` wins over this
setting. The two boot requests explicitly use `low` regardless of this default.

### Boot check

Startup runs a boot check after `/health`; `BOOT_CHECK=0` disables it.
The temperature-0 prompt is `Reply with exactly OK, with no punctuation or other text.`
The reply's `content`, with surrounding whitespace stripped, must equal `OK`.
A second temperature-0 request streams 768 tokens of arithmetic-function code
with `reasoning_effort=low`, `min_tokens=768` and `ignore_eos=true`.
Model-scoped Prometheus `/metrics` deltas must show at least 64 draft steps,
64 proposed draft tokens and more than zero accepted tokens. Startup fails
and stops its own launch on an incorrect reply, incomplete stream, missing
counters or insufficient drafting. Completion counters settle the first
request before the second snapshot; counter resets fail. Metrics may take
up to 60 seconds to settle. These are the only two boot generation requests;
starting an already-running checkout sends none. Keep user requests out of
startup so the model-wide counters describe the check. Direct `serve.sh` and
Compose launches use the same check. `smoke.sh` retains its two independent
chat/tool requests; it does not repeat the boot check.

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

**Updating from 1.6.x or 1.5.x.** Run `./start.sh update`. Release 1.7.0
moves the engine and image pins. Explicit `.env` values remain in force;
remove explicit `IMAGE` / `VLLM_COMMIT` values to follow the new pins.
P2P defaults to `auto`, enabled only after the content check. RecoverSSM is
on at TP4. Compiled decode stays on when installed; eligible TP4 and PP4
prefill kernels use the new layout-specific switches.
All-reduce flags and cached-boundary reuse default to `1`, and PP4 drafter
width to `0`. Remove explicit old feature overrides to
follow these defaults; see [Engine switches](engine-switches.md).
`BOOT_CHECK=1` and empty `DEFAULT_REASONING_EFFORT` are the new defaults.
`VLLM_BRANCH=ampere` selects the fork branch, while the full commit remains
pinned. Remove an explicit old branch line to follow it.

**Updating from 1.4.x.** Run `./start.sh update`. Release 1.5.0 moves the
engine and image pins; your `.env`, including explicit `IMAGE`, `VLLM_COMMIT`
and runtime-switch overrides, is preserved. Remove an old explicit pin only
if you want to follow this release. The image includes optional Marlin:
release 1.5.0 enabled unset PP4 decode and left unset TP4 decode off.
For current defaults and native compilation, see
[Optional compiled Marlin](compiled-marlin.md); without an installed library,
unset decode remains off. The allocator default remains
`expandable_segments:False`, with explicit allocator overrides preserved.

**Updating from 1.3.x.** Run `./start.sh update`, nothing else. What happens:

1. It pulls this release and restarts into its `start.sh`.
2. Your checkout already has a native install, so it **stays native** (the
   container is the default only for fresh checkouts).
3. The engine pin moved to a new upstream base, so `start.sh` **rebuilds the
   venv** from scratch for the new engine and its own dependency pins (the checkpoints are not touched).
4. It boots the server. **The first boot can be slower** while FlashInfer fills an empty compile cache; the log may show
   `No available shared memory broadcast block found in 60 seconds` lines
   meanwhile, which are harmless.
5. The layout stays tensor-parallel 4. Peer access follows `P2P=auto` unless
   overridden, with a content check before use. Explicit allocator overrides
   are preserved.

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

Advanced settings: peer-to-peer policy, the engine's kill switches,
link width, running the container yourself and the native developer install.

### PCIe peer-to-peer

`P2P=auto` is the default for both container and native starts. Before the
server starts with TP > 1 or PP > 1, the recipe checks every ordered pair of
visible GPUs. It requires peer access to be advertised, then copies random
bytes with `cudaMemcpyPeer`, writes and reads through peer access with an SM
kernel, and verifies a separate-process CUDA IPC write on the allocation's
owner. Every byte is compared on its owner. The read test uses fresh data
written by the owner, so matching writes and reads at a wrong address cannot
pass together.

Nine sizes cover 128 KiB and 512 KiB, including one byte below and above each,
plus 1 MiB, 8 MiB and 32 MiB. Each pair also gets a 32 MiB IPC write. Compilation,
identity lookup, lock waits and the content check share a 120-second budget;
a timed-out probe and its IPC children are killed together, with at most one
additional second allowed for cleanup. The startup banner reports the outcome,
reason, elapsed time and whether the result came from this boot's cache.

A result is cached by driver version/build, visible GPU UUIDs, boot ID and probe
version. Restarts reuse it; a new boot, driver, GPU set or probe invalidates it.
The cache lives at `$HOME/.cache/recipe-p2p` (inside the persistent `/cache`
mount for containers); `P2P_CHECK_CACHE` selects another persistent directory.
A failed content check is cached too. After correcting a configuration within
the same boot, remove the check cache before retrying.

**Topology is not a data-integrity test.** `nvidia-smi topo -p2p r` reporting
`OK` and `cudaDeviceCanAccessPeer` returning true are not enough. A driver can
advertise peer access while writes land in the wrong memory; NCCL may then
hang. The startup content check decides whether this recipe uses P2P.

A pass enables `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1` by default, uses the
release custom all-reduce selection (`1` for flags-in-data), clears `NCCL_P2P_DISABLE`, and sets
`NCCL_P2P_LEVEL=SYS` for TP or PP through `GLM5_NCCL_P2P_SYS=1`.
An explicit custom-all-reduce `0` is preserved after a pass. An explicit
`NCCL_P2P_LEVEL` wins; `GLM5_NCCL_P2P_SYS=0` suppresses the automatic SYS setting.

If any pair does not advertise access, any byte differs, or the check cannot
complete, the engine starts with P2P disabled:
`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0`, `NCCL_P2P_LEVEL` unset and
`NCCL_P2P_DISABLE=1`. This includes PP4 stage transfers. The `[p2p]` banner
says `not advertised`, `advertised but data check failed`, or `check unavailable`
and gives the reason. TP can use the host-staged all-reduce instead.

**Working P2P on CMP 170HX requires a P2P-capable driver build.**
This release uses mainline [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
master, including its 74-SM override, plus the maintainer's P2P changes and
our minimal TRAP31 Booter PLM patch, gated on `ForceP2P=0x11`. The patch makes
peer mappings usable on these cards; advertising peer access alone is insufficient.
The recipe still requires the content check on every new driver/boot identity.
<!-- link: Morrowmake/cmpunlocker after publication -->

Choose the policy in `.env` or for a single start:

```bash
P2P=auto ./start.sh restart   # default: enable only after verification
P2P=off ./start.sh restart    # disable all peer transports, including PP
P2P=force ./start.sh restart  # bypass verification; may hang or corrupt data
```

Use `force` only when deliberately bypassing the check. A dry run initializes
no CUDA devices and leaves auto P2P off until a real start can verify it.
The allocator default stays `expandable_segments:False`, which supports legacy
CUDA IPC allocations. Explicit allocator values are preserved.

Read the last `[p2p]` line in `logs/serve.log`. For TP, a serving custom
all-reduce appears as `['CUSTOM', 'PYNCCL']`; the host-staged path appears as
`['HOSTSHM', 'PYNCCL']`. Then run `./start.sh smoke`.

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
RUNTIME=native ./install.sh    # build ./venv and the pinned vLLM fork
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
`BUILD_FROM_SOURCE=1 MAX_JOBS=16 ./install.sh` (full toolkit and build scratch space required).
