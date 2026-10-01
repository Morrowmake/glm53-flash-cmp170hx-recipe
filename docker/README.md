# Container image

The engine in a container: the way this recipe runs by default. `./start.sh`
pulls the image, starts it and manages it for you (see the
[How to use this repo](../docs/how-to-use.md)); this page describes the
image, how to build it, and how to run it by hand without `./start.sh`.

## What the image contains

- **Engine:** the [vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx/tree/3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340)
  pinned at `3a2bf16dae` (1.6.0, on upstream `e55d076f89`), installed
  the way the native install does it: Python 3.12, torch 2.13.0 (CUDA 13.0
  build), the fork installed editable in `/opt/venv` with upstream's
  precompiled extensions, then the runtime extras the engine pins (FlashInfer
  0.7.0, humming-kernels 0.1.16, TileLang). The base `e55d076f89` has no nightly
  wheel of its own, so the extensions come from upstream `b6761e8ded`'s wheel,
  whose C++, CUDA and Rust sources are identical. The installed package list
  is in `/opt/venv/requirements.lock.txt`.
- **Optional Marlin:** a distinct compiled CUDA/C++ library, included for both
  layouts without replacing the upstream base extensions.
- **Base:** `nvidia/cuda:13.3.1-devel-ubuntu24.04`, pinned by digest. It is the
  devel image because Triton, TileLang, FlashInfer and the host-memory
  all-reduce compile kernels at run time and need `nvcc` and `g++`.
- **Entry point:** the `vllm` CLI. The recipe runs it with this repository's
  [`serve.sh`](../serve.sh) mounted and used as the entry point instead, so the
  layout and every setting are chosen at run time. No launch arguments or
  settings are baked in.
- **No weights.** The two checkpoints are mounted from the host.

Release image, built from the pinned engine; release acceptance installs and updates from this exact published digest:

```
ghcr.io/morrowmake/vllm-cmp170hx@sha256:cc26c8abb63953a37c6c1861cadc9188e99c1cceab403600b5822740f3a82ae0
```

Its OCI version is `1.6.0-3a2bf16dae`.

**Measured in 1.4.0:** On the same four cards at 180 W with peer-to-peer off, the image runs at the native install's speed: 15.60 / 29.16 / 43.99 ms per decode step at 1 / 4 / 8 users (native 15.87 / 29.51 / 44.45) and 2,672 tokens/s cold prefill (native 2,670). (Release 1.3.x's image ran within
0.4 % of the native install on decode at 1 / 4 / 8 users and on cold prefill.)

## Build it

**Optional Marlin:** both builders use the release engine pin, which includes
`csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py`. Missing support
fails the build rather than omitting the library. To build from another local
source, supply its full commit and branch explicitly:

```bash
VLLM_COMMIT=<full-engine-commit> CLONE_FROM=/path/to/source CLONE_BRANCH=<source-branch> \
  STAGE=/path/to/new-owned-staging-directory ./docker/build.sh
```

The builder replaces `STAGE`: choose a new dedicated path. It produces a local
OCI image in `docker/out/`, without publishing it. Set `OUT=/path/to/new-output`
to preserve earlier image evidence. Local `CLONE_FROM` sources use Git transport
cloning (`--no-local`), never object-directory copying or hardlinks. Only the
selected history and explicit upstream version tag are retained during the build;
objects newer than an overridden pin are pruned. Before removal, the actual
source HEAD, self-contained Git stores and absence of unreachable objects are
checked, including submodule stores. After version construction, both builders
remove all Git metadata **before** copying or archiving the application payload
into any final runtime layer. No contaminated application layer is reused.

The final source tree remains editable Python source, but is not a Git checkout.
Its root-owned, read-only `/opt/vllm-src/provenance.json` records `schema` (1),
`engine_commit` (full hash), `engine_repository` (public fork URL) and
`engine_version` (installed version). The OCI revision label retains the same
commit. Native installations remain Git checkouts and keep their update behavior.
Layer assembly rejects retained Git metadata/object stores and forbidden build
paths, including standard nested gzip, bzip2, xz, tar and zip payloads. Set
`IMAGE_FORBIDDEN_STRINGS` to a JSON array of additional identifying strings for
the rootless build. Do not pass private identifier lists through Docker build
arguments or labels; use a separate private audit of the resulting image. Broad
name matches still need attribution review; upstream copyright/author notices
are retained.
For Docker, add build arguments `VLLM_REPO`, `VLLM_BRANCH`, `VLLM_COMMIT` naming
a source reachable inside the build.
The revision label identifies the supplied source. Both builders compile one
sm_80 library without GPUs and check operator registrations. Image configuration
leaves `VLLM_GLM5_MARLIN_DECODE_CUDA` and `VLLM_GLM5_MARLIN_PREFILL_CUDA` unset.
With the mounted `serve.sh`, TP4 and PP4 decode default on when
installed, and compiled prefill defaults off. Explicit 0/1 values override each
switch independently. The bare `vllm` entry point retains engine-code defaults;
the recipe's layout defaults require its launcher. Missing libraries default
unset PP4 decode off with a banner; enabled paths must pass compatibility checks.
TP4/PP4 changes never rebuild the library. Other engine kernels still use the
existing toolkit. Validate real-image startup before selecting a new `IMAGE` pin.

Build the release source with Docker:

```bash
bash docker/build-docker.sh vllm-cmp170hx:1.6.0-3a2bf16dae
```

Both build paths print the total filesystem layer count, including the base,
and fail above 123. OCI assembly checks before compression; the Docker wrapper
checks the final image's `RootFS.Layers` after building. Use the wrapper for
Docker builds; a bare `docker build` bypasses this check. The troubleshooting
kill switch is `IMAGE_LAYER_COUNT_CHECK=0` (default 1).

The build needs no GPU. It clones the fork at the pinned commit and downloads
torch, the precompiled extensions and the runtime extras, so it takes a while
and needs network access.

The pinned release artifact used the rootless application-layer assembly
approach in [`build.sh`](build.sh). The current builder prepares the runtime
tree on the host, creates separate deterministic layers and writes an OCI
layout without a Docker daemon. [crane](https://github.com/google/go-containerregistry)
fetches the pinned base and publishes the retained OCI without recompression.
See [release procedure](RELEASE.md) for the dependency lock, compression
switches, cache-seed gates and publication. Rebuilding with either
builder produces a separate artifact; it does not reproduce or revalidate the
frozen digest automatically. Validate the resulting provenance, layers and
runtime before distributing any rebuilt image.

## Run it by hand

You need:

- the four cards and NVIDIA driver **580 or newer**
  ([What you need](../docs/how-to-use.md#what-you-need));
- Docker with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  so that `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists all four cards;
- the two checkpoints on the host. `./download.sh` puts them in `./models`, or
  fetch `canada-quant/GLM-5.3-Flash-W4A16-MTP` and
  `incoai/GLM-5.3-Flash-DFlash2` with any Hugging Face client.

From the root of this repository:

```bash
IMAGE=ghcr.io/morrowmake/vllm-cmp170hx@sha256:cc26c8abb63953a37c6c1861cadc9188e99c1cceab403600b5822740f3a82ae0
MODELS=$PWD/models                  # holds GLM-5.3-Flash-W4A16-MTP and GLM-5.3-Flash-DFlash2
CACHE=$PWD/cache                    # kernel compile caches, kept between starts
mkdir -p "$CACHE"

docker run -d --name glm53-flash --user "$(id -u):$(id -g)" --workdir /cache \
  --gpus all --shm-size 16g \
  -p 127.0.0.1:8000:8000 \
  --env-file docker/container.env \
  -v "$PWD/serve.sh:/recipe/serve.sh:ro" \
  -v "$PWD/boot_check.py:/recipe/boot_check.py:ro" \
  -v "$MODELS/GLM-5.3-Flash-W4A16-MTP:/models/GLM-5.3-Flash-W4A16-MTP:ro" \
  -v "$MODELS/GLM-5.3-Flash-DFlash2:/models/GLM-5.3-Flash-DFlash2:ro" \
  -v "$CACHE:/cache" \
  --entrypoint /bin/bash "$IMAGE" /recipe/serve.sh

docker logs -f glm53-flash          # wait for "Application startup complete"
curl http://127.0.0.1:8000/health
```

If you built the image yourself, use your tag in place of `$IMAGE`. The same
launch as a Compose file is [`docker-compose.yml`](../docker-compose.yml) at the
root of this repository ([Docker Compose](../docs/how-to-use.md#docker-compose)).

**Layout.** [`container.env`](container.env) sets `LAYOUT=tp4`; change it to
`LAYOUT=pp4` for pipeline-parallel 4. `serve.sh` inside the container turns
the layout and any other setting from [`.env.example`](../.env.example) or
[`.env.advanced.example`](../.env.advanced.example) that
you add to `container.env` into the same flags as a native start, so the
server is the same either way.

Inside the container the server listens on every interface;
`-p 127.0.0.1:8000:8000` keeps it reachable from this machine only, as
`./start.sh` does. It has **no API key**: to serve other machines, publish the
port on another address and add `-e API_KEY=<a long random string>`.

The server runs as you (`--user`), not root: Triton, FlashInfer, TileLang,
torch and vLLM write their compile caches under the cache mount (`HOME=/cache`
and the cache paths in `container.env`), so `./cache` stays yours. The first
start compiles kernels there and takes several minutes; later starts reuse
them. Optional device-specific seeds are described in the
[release procedure](RELEASE.md); `VLLM_IMAGE_CACHE_SEED=0` disables them. Stop it with `docker stop -t 120 glm53-flash`
and wait until `nvidia-smi` shows the cards empty before starting again. The
API is the same as always ([Talk to it](../docs/how-to-use.md#talk-to-it)).

**PCIe peer-to-peer is off** (`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0`). If peer
access already works on your cards
([PCIe peer-to-peer](../docs/how-to-use.md#pcie-peer-to-peer-optional)), set it to `1` in
`container.env`; `serve.sh` picks the matching allocator. The container changes
no driver setting.

The [engine switches](../docs/engine-switches.md) work the
same way: set the variable in `container.env` or add `-e NAME=0`.
