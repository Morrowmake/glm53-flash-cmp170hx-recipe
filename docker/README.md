# Container image

The engine in a container: the way this recipe runs by default. `./start.sh`
pulls the image, starts it and manages it for you (see the
[main README](../README.md#how-to-use-this-repo)); this page describes the
image, how to build it, and how to run it by hand without `./start.sh`.

## What the image contains

- **Engine:** the [vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx/tree/PIN_PENDING)
  pinned at `PIN_PENDING` (the 1.4.0 pin, on upstream `e55d076f89`), installed
  the way the native install does it: Python 3.12, torch 2.13.0 (CUDA 13.0
  build), the fork installed editable in `/opt/venv` with upstream's
  precompiled extensions, then the runtime extras the engine pins (FlashInfer
  0.7.0, humming-kernels 0.1.16, TileLang). The base `e55d076f89` has no nightly
  wheel of its own, so the extensions come from upstream `b6761e8ded`'s wheel,
  whose C++, CUDA and Rust sources are identical. The installed package list
  is in `/opt/venv/requirements.lock.txt`.
- **Base:** `nvidia/cuda:13.3.1-devel-ubuntu24.04`, pinned by digest. It is the
  devel image because Triton, TileLang, FlashInfer and the host-memory
  all-reduce compile kernels at run time and need `nvcc` and `g++`.
- **Entry point:** the `vllm` CLI. The recipe runs it with this repository's
  [`serve.sh`](../serve.sh) mounted and used as the entry point instead, so the
  layout and every setting are chosen at run time. No launch arguments or
  settings are baked in.
- **No weights.** The two checkpoints are mounted from the host.

Image (compressed size {{IMAGE_SIZE}}), pulled by digest to get exactly the
tested image:

```
ghcr.io/morrowmake/vllm-cmp170hx@sha256:DIGEST_PENDING
```

Also tagged `ghcr.io/morrowmake/vllm-cmp170hx:1.4.0-PIN_PENDING`.

**Measured:** {{CONTAINER_PARITY_LINE}} (Release 1.3.x's image ran within
0.4 % of the native install on decode at 1 / 4 / 8 users and on cold prefill.)

## Build it

```bash
docker build -t vllm-cmp170hx:1.4.0-PIN_PENDING docker/
```

The build needs no GPU. It clones the fork at the pinned commit and downloads
torch, the precompiled extensions and the runtime extras, so it takes a while
and needs network access.

The release image was assembled with [`build.sh`](build.sh) instead, which
runs the same steps on the host without a Docker daemon and appends the result
to the pinned base as one layer with
[crane](https://github.com/google/go-containerregistry). The contents are the
same; the digest of a `docker build` will differ.

## Run it by hand

You need:

- the four cards and NVIDIA driver **580 or newer**
  ([What you need](../README.md#what-you-need));
- Docker with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  so that `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists all four cards;
- the two checkpoints on the host. `./download.sh` puts them in `./models`, or
  fetch `canada-quant/GLM-5.3-Flash-W4A16-MTP` and
  `incoai/GLM-5.3-Flash-DFlash2` with any Hugging Face client.

From the root of this repository:

```bash
IMAGE=ghcr.io/morrowmake/vllm-cmp170hx@sha256:DIGEST_PENDING
MODELS=$PWD/models                  # holds GLM-5.3-Flash-W4A16-MTP and GLM-5.3-Flash-DFlash2
CACHE=$PWD/cache                    # kernel compile caches, kept between starts
mkdir -p "$CACHE"

docker run -d --name glm53-flash --gpus all --shm-size 16g \
  -p 127.0.0.1:8000:8000 \
  --env-file docker/container.env \
  -v "$PWD/serve.sh:/recipe/serve.sh:ro" \
  -v "$MODELS/GLM-5.3-Flash-W4A16-MTP:/models/GLM-5.3-Flash-W4A16-MTP:ro" \
  -v "$MODELS/GLM-5.3-Flash-DFlash2:/models/GLM-5.3-Flash-DFlash2:ro" \
  -v "$CACHE:/root/.cache" \
  --entrypoint /bin/bash "$IMAGE" /recipe/serve.sh

docker logs -f glm53-flash          # wait for "Application startup complete"
curl http://127.0.0.1:8000/health
```

If you built the image yourself, use your tag in place of `$IMAGE`.

**Layout.** [`container.env`](container.env) sets `LAYOUT=tp4`; change it to
`LAYOUT=pp4` for pipeline-parallel 4. `serve.sh` inside the container turns
the layout and any other setting from [`.env.example`](../.env.example) that
you add to `container.env` into the same flags as a native start, so the
server is the same either way.

Inside the container the server listens on every interface;
`-p 127.0.0.1:8000:8000` keeps it reachable from this machine only, as
`./start.sh` does. It has **no API key**: to serve other machines, publish the
port on another address and add `-e API_KEY=<a long random string>`.

The first start compiles kernels into the cache mount and takes several
minutes; later starts reuse them. Stop it with `docker stop -t 120 glm53-flash`
and wait until `nvidia-smi` shows the cards empty before starting again. The
API is the same as always ([Talk to it](../README.md#talk-to-it)).

**PCIe peer-to-peer is off** (`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0`). If peer
access already works on your cards
([PCIe peer-to-peer](../README.md#pcie-peer-to-peer-optional)), set it to `1` in
`container.env`; `serve.sh` picks the matching allocator. The container changes
no driver setting.

The kill switches in the [main README](../README.md#kill-switches) work the
same way: set the variable in `container.env` or add `-e NAME=0`.
