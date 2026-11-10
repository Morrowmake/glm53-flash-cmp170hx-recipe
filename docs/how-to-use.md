# How to use this repo

## What you need

| | |
|---|---|
| GPUs | 4× NVIDIA CMP 170HX, **each exposing 64 GiB** (`nvidia-smi` shows 65,536 MiB). For tensor-parallel 4, **PCIe Gen2 x16 links**; pipeline-parallel 4 is built for narrower links. Stock cards expose less memory and run at x4; getting to 64 GiB and x16 is outside this repository. The preflight stops before any download if a card reports under 60 GiB, and warns below x16 ([Link width](#link-width)) |
| OS and driver | Linux (the commands below are for Ubuntu) with NVIDIA driver **580 or newer**; `nvidia-smi` must list all four cards |
| Container runtime | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), usable by your user, so that `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi` lists all four cards. This and the driver are a one-time host setup; the engine and its Python environment come in the image |
| Tools | for `./start.sh`: `git`, `curl`, `flock` and `setsid` (util-linux, on most systems already), Python 3 for the boot check, `jq` for the smoke test. Docker Compose needs only Docker |
| Disk | about 180 GiB (193 GB) for the two checkpoints, plus the engine image and the kernel compile caches |

## Step by step

`./start.sh` on its own does steps 1–3 in order and skips anything already
done, so running it twice is safe. The same steps one at a time:

```bash
./install.sh       # 1. pull the engine image, pinned by digest
./download.sh      # 2. fetch the model (~178 GiB) and the drafter (~2.2 GiB) into ./models, at the pinned revisions
./start.sh         # 3. start the container, wait for /health, print the KV pool size
./start.sh smoke   # 4. one chat request and one tool call against the running server
./start.sh stop    # stop it; the image, weights and caches stay, so the next start is quick
```

The container runs the engine image with this repository's `serve.sh` as its
entry point, as your user rather than root, with the checkpoints mounted
read-only and the kernel compile caches in `./cache` (owned by you). It
publishes the API on `127.0.0.1:8000` only. Its output goes to
`logs/serve.log`, and `./start.sh stop` stops only the container this checkout
started. The download runs in the same image as you, so the checkpoints are
yours too; an interrupted download resumes on the next run, and a checkpoint
counts as present only when every file its index names is there.

Both checkpoints are pinned to the Hugging Face revisions this release was
validated with: `TARGET_REVISION` (`5681f7c2dc6436d63f82ab6fb892b4f5062b8304`)
and `DRAFTER_REVISION` (`bf582e4eacc1810f76656d1811693ff6c6737d2a`), for
`./start.sh` and Docker Compose alike. Checkpoints you already have are kept.

**The first boot after an install or an update is slower.** The image ships
no warm-cache seed (seed: no): it starts compile on first boot (cold).
FlashInfer compiles its kernel modules into `./cache`, and later boots reuse
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
       "max_tokens": 4096}'
```

Any OpenAI-compatible client works with base URL `http://127.0.0.1:8000/v1`
and model `glm-5.3-flash`. GLM-5.3 always reasons before it answers: its chat
template has no thinking switch, and thinking off is not supported. The
reasoning comes back in the message's `reasoning` field (streamed as
`delta.reasoning`) and the answer in `content`.
`"chat_template_kwargs": {"enable_thinking": false}` and `thinking: false` are
ignored, with a warning in the server log. `reasoning_effort` of `low`, `high`
or `max` sets how much it thinks; do not send `reasoning_effort: "none"`.

**Set `max_tokens` for the reasoning plus the answer.** If the limit is
reached while the model is still reasoning, `content` is empty and
`finish_reason` is `length`; a client that reads only `content` then shows an
empty reply.

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
| `P2P` | `auto` | [content-verified peer access](#pcie-peer-to-peer) for TP and PP; `off` disables, `force` bypasses the check |
| `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P` | TP4: `1` after a pass, otherwise `0` | custom all-reduce (TP); an explicit `0` is preserved after a pass |

Commonly used from `.env.advanced.example`:

| Key | Default | What it does |
|---|---|---|
| `MAX_LEN` | `262144` | context ceiling; lower it for more concurrent full-length requests |
| `MAX_SEQS` | `8` | concurrent requests |
| `GPU_UTIL` | `0.95` | share of each card's memory the engine may use |
| `SPEC_N` | unset | draft tokens per step for `mtp` only; DFlash2 (the only supported mode) uses the layout's adaptive depth, up to 7 ([Draft depth](engine-switches.md#draft-depth)) |
| `MAX_BATCHED` | `3463` (TP4), `2312` (PP4) | sets the prefill chunk: 3,456 / 2,304 tokens; `2048` gives 1,152 |
| `FAIR_PREFILL` / `FAIR_CHUNK` | `1` / `384` | fair prefill and its slice size while others decode |
| `PREFILL_CAP` | `0` | upstream's unconditional chunk cap. **Leave it 0**: it turns the prefill features off |
| `MM_CAP` | `0` | `1` bounds image and video inputs, which reduces the multimodal memory reservation. Each image, and each whole video, is capped at `MM_MAX_IMAGE_TOKENS` tokens (default `MM_MAX_PIXELS / 784` = 1280) |
| `IMAGE` | this release's image, by digest | the engine image (container) |
| `CONTAINER_NAME` | `glm53-flash` | the container's name |
| `READY_TIMEOUT` | `1800` | seconds `./start.sh` waits for `/health` |
| `EXTRA_ARGS` | unset | appended to the `vllm serve` command line |
| `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES` | `715827880` | upper limit on decoded images, video frames and audio per request (about 683 MiB, about 115 frames of 1080p); larger requests get HTTP 400. Set it in `.env` (under Docker Compose, in [`container.env`](../docker/container.env)) |
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
| `./start.sh status` | layout, container, `/health`, KV line, install and checkpoint state |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh smoke` | one chat request and one tool call |
| `./start.sh stop` | stop the server this checkout started |
| `./start.sh restart` after a crash | `./start.sh` does not restart a crashed server; Docker Compose does, up to 3 times |
| `./start.sh update` | `git pull`, pull the new engine if the pin moved, restart |
| `DRY=1 ./start.sh` | print what would be pulled, downloaded and launched, without doing it |

`stop` only stops the container recorded in `logs/container.id`, after
confirming it carries this checkout's label; it never matches by name alone,
so another container or vLLM on the same machine is never touched.

The engine pin lives in `start.sh`, so a `git pull` can move it, and `start.sh`
pulls the matching image whenever it changes.

**Updating from an earlier release.** Run `./start.sh update`: it pulls this
release, restarts into its `start.sh`, pulls the new image if the pin moved and
restarts the server. Your `.env` is kept and its explicit values stay in force;
each release's [CHANGELOG](../CHANGELOG.md) entry names the lines to delete to
follow its new defaults. With Docker Compose: `git pull`, then
`docker compose up -d`.

**Updating from 1.7.x to 1.8.0.** Release 1.8.0 moves the engine to a new
upstream vLLM base; `./start.sh update` pulls its image. Then:

- **Size `max_tokens` for the reasoning plus the answer** in your clients.
  Clients that send thinking off now get the reasoning in `reasoning` instead
  of `content` ([Talk to it](#talk-to-it)).
- In `.env`, delete any `VLLM_BRANCH=ampere-glm53` line: that fork branch is
  retired. Delete explicit `IMAGE` or `VLLM_COMMIT` lines to follow this
  release's image.
- Rename any engine switch set under an old name in `.env` or
  [`container.env`](../docker/container.env); old names are ignored with a
  warning at startup ([Engine switches](engine-switches.md#renamed-and-retired-settings)).
  Delete a `SPEC_N` line: with DFlash2, `serve.sh` refuses it.
- Delete an `EXTRA_ARGS` line carrying `max_pixels` and `max_image_tokens` if
  you added it for `MM_CAP=1` on 1.7.2; `MM_CAP=1` now does this itself. To
  raise the cap, set `MM_MAX_IMAGE_TOKENS`.

**From a native install.** Earlier releases could run the engine natively from
`venv/`. This release runs it only in the container, and `./start.sh update`
(or `restart`) moves the checkout over:

1. It checks Docker first. Without a usable Docker and NVIDIA Container
   Toolkit it stops with a message and leaves the native server running;
   install them ([What you need](#what-you-need)) and run `./start.sh update`
   again.
2. It pulls the image, stops the native server (only the process recorded in
   `logs/vllm.pid`, after confirming it is this checkout's server) and starts
   the container with your `.env` and the checkpoints already in `MODELS_DIR`.
   Complete checkpoints are not downloaded again.
3. It says what it did and deletes nothing: `venv/` and `vllm-src/` stay until
   you remove them, and `./start.sh status` mentions them while they exist.
4. The first container start compiles its kernel caches into `./cache`, so it
   is slower; `No available shared memory broadcast block found in 60 seconds`
   lines meanwhile are harmless.

`RUNTIME`, `VLLM_COMMIT` and `VLLM_BUILD_AMPERE_MARLIN` in `.env` no longer do
anything; `./start.sh` warns about them until you delete them. A plain
`./start.sh` while the native server is still running refuses and points to
`./start.sh update`.

Older releases are git tags; their images may be retired.

## Advanced

Advanced settings: peer-to-peer policy, the engine's kill switches,
link width, running the container yourself and Docker Compose.

### PCIe peer-to-peer

`P2P=auto` is the default for every start. Before the
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

A pass enables `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1` under TP by default, uses the
release custom all-reduce selection (`1` for flags-in-data), clears `NCCL_P2P_DISABLE`, and sets
`NCCL_P2P_LEVEL=SYS` through `GLM5_NCCL_P2P_SYS=1`: under TP while the custom
all-reduce is on, and under PP for the stage transfers.
An explicit custom-all-reduce `0` is preserved after a pass. An explicit
`NCCL_P2P_LEVEL` wins; `GLM5_NCCL_P2P_SYS=0` suppresses the automatic SYS setting.

If any pair does not advertise access, any byte differs, or the check cannot
complete, the engine starts with P2P disabled:
`VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0`, `NCCL_P2P_LEVEL` unset and
`NCCL_P2P_DISABLE=1`. This includes PP4 stage transfers. The `[p2p]` banner
says `not advertised`, `advertised but data check failed`, or `check unavailable`
and gives the reason. TP can use the host-staged all-reduce instead.

**Working P2P on CMP 170HX requires a P2P-capable driver build.**
This release uses mainline [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
master, including its 74-SM override, plus the maintainer's P2P changes and
our minimal TRAP31 Booter PLM patch, gated on `ForceP2P=0x11`. The patch makes
peer mappings usable on these cards; advertising peer access alone is insufficient.
The recipe still requires the content check on every new driver/boot identity.
The ready-to-install build is [Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker): mainline master with these changes, installed the same way as mainline.

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

**AMD hosts: P2P that passes once, then fails after every stop.** If the
content check passes on a fresh boot but fails after any server stop or crash
(`cudaMemcpyPeer mismatch`, NCCL hangs in `_init_comm`) and only a reboot
helps, look for `AMD-Vi: Event logged [IO_PAGE_FAULT` from the GPUs in
`dmesg`. With the IOMMU in translated mode (`DMA-FQ` domains, lazy
invalidation), peer DMA goes through IOMMU mappings that a process exit tears
down. Put the GPUs in identity domains with the kernel parameter `iommu=pt`
(for GRUB: add it to `GRUB_CMDLINE_LINUX_DEFAULT`, run `update-grub`,
reboot; `cat /sys/kernel/iommu_groups/*/type` then shows `identity`). Seen on
EPYC 7003 under a Proxmox kernel; after the change, stops, crashes and
restarts leave P2P working and `IO_PAGE_FAULT` stays at zero.

### Kill switches

Every performance feature can be switched off on its own with one variable,
without a rebuild. The full list, and how to use the switches for
troubleshooting, is in [Engine switches](engine-switches.md).

```bash
VLLM_AMPERE_DECODE_KERNELS=0 ./start.sh restart
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
is the other one-command start. From a fresh clone, after the one-time host
setup ([What you need](#what-you-need)):

```bash
docker compose up                  # download, then serve in the foreground
docker compose up -d               # the same in the background
docker compose ps                  # the server shows (healthy) once /health answers
docker compose logs -f             # follow it
docker compose down                # stop (allows 120 s for a clean shutdown)
```

Two services run in order:

- **`prepare`** makes `./cache` and the model directory (`MODELS_DIR`, default
  `./models`) writable by the serving user, then downloads the two checkpoints
  as that user, at the pinned revisions (~180 GiB). An interrupted download resumes; complete
  checkpoints are skipped without contacting Hugging Face, so later starts go
  straight to the server. Docker creates a missing directory as root; `prepare`
  hands such a directory to the serving user, but never changes the owner of
  existing checkpoints or of anything else in the model directory, and never
  deletes anything. It must finish successfully before the server starts.
- **The server** is the same container `./start.sh` launches: the release image
  by digest, all four cards, the same shared-memory size, the API on
  `127.0.0.1:8000` only, the checkpoints mounted read-only, `./cache` for the
  kernel compile caches and the settings in
  [`container.env`](../docker/container.env), with this repository's
  `serve.sh` as its entry point. It switches to the serving user before
  `serve.sh` runs, so it never serves as root. It reports healthy once
  `/health` answers (`docker compose ps`).
- **Restarts.** If the server exits with an error, Compose starts it again, up
  to 3 times (`restart: on-failure:3`); `docker compose stop` and `down` stop it
  for good. `./start.sh` does not restart a crashed server: run
  `./start.sh restart`.

The serving user is the owner of this checkout, so it works for any user with
no settings; set `HOST_UID` and `HOST_GID` (in the shell or in `.env`) to use
another. `LAYOUT=pp4` (in the shell or in `.env`) selects pipeline-parallel 4,
`MODELS_DIR` moves the checkpoints and `HF_TOKEN` speeds up the download; any
other setting goes in [`container.env`](../docker/container.env), as for the
`docker run` command. `./start.sh` does not use this file; use one or the
other (they share the container name, so a second start fails rather than
doubling up on the cards).
