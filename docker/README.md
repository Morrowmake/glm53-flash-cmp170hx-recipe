# Container image

The same engine as the native install, in a container. The native install
(`./start.sh`, see the [main README](../README.md#how-to-use-this-repo)) remains
the primary path: it is where every number on the main page was measured, and
it has the preflight, the smoke test and `./start.sh update`. Use the image if
you would rather keep the engine and its Python environment out of the host.

## What the image contains

- **Engine:** the [vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx/tree/0ed7d3e7f3f855646a139701598a9b40d5745688)
  pinned at `0ed7d3e7f3` (the 1.3.0 pin), installed the way `install.sh`
  installs it: Python 3.12, torch 2.13.0 (CUDA 13.0 build), the fork installed
  editable in `/opt/venv` with upstream's precompiled extensions, then the same
  runtime extras (FlashInfer, TileLang). The installed package list is in
  `/opt/venv/requirements.lock.txt`.
- **Base:** `nvidia/cuda:13.3.1-devel-ubuntu24.04`, pinned by digest. It is the
  devel image because Triton, TileLang, FlashInfer and the host-memory
  all-reduce compile kernels at run time and need `nvcc` and `g++`.
- **Entry point:** the `vllm` CLI. No launch arguments or settings are baked
  in; they are given at run time.
- **No weights.** The two checkpoints are mounted from the host.

Image name (compressed size about 10.35 GB):

```
ghcr.io/morrowmake/vllm-cmp170hx:1.3.0-0ed7d3e7f3
```

Digest (pull by digest to get exactly the tested image):

```
docker pull ghcr.io/morrowmake/vllm-cmp170hx@sha256:ee978fb3e3d11cf8577a014539a7ad4e2a8dff95163fc5d96c0a06fcf9c64640
```

**Measured:** on the same four cards at 180 W with peer-to-peer off, the image
runs at the native install's speed: 15.79 / 30.16 / 44.89 ms per decode step
at 1 / 4 / 8 users (native 15.84 / 30.09 / 44.71) and 2,488 tokens/s cold
prefill (native 2,484), all within 0.4 %.

## Build it

```bash
docker build -t vllm-cmp170hx:1.3.0-0ed7d3e7f3 docker/
```

The build needs no GPU. It clones the fork at the pinned commit and downloads
torch and the runtime extras, so it takes a while and needs network access.

The release image was assembled with [`build.sh`](build.sh) instead, which
runs the same steps on the host without a Docker daemon and appends the result
to the pinned base as one layer with
[crane](https://github.com/google/go-containerregistry). The contents are the
same; the digest of a `docker build` will differ.

## Run it

You need:

- the four cards and NVIDIA driver **580 or newer**, as for the native install
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
MODELS=$PWD/models                  # holds GLM-5.3-Flash-W4A16-MTP and GLM-5.3-Flash-DFlash2
CACHE=$PWD/cache                    # kernel compile caches, kept between starts
mkdir -p "$CACHE"

docker run -d --name glm53-flash --gpus all --shm-size 16g \
  -p 127.0.0.1:8000:8000 \
  --env-file docker/container.env \
  -v "$MODELS/GLM-5.3-Flash-W4A16-MTP:/models/GLM-5.3-Flash-W4A16-MTP:ro" \
  -v "$MODELS/GLM-5.3-Flash-DFlash2:/models/GLM-5.3-Flash-DFlash2:ro" \
  -v "$CACHE:/root/.cache" \
  ghcr.io/morrowmake/vllm-cmp170hx:1.3.0-0ed7d3e7f3 \
  serve /models/GLM-5.3-Flash-W4A16-MTP \
  --served-model-name glm-5.3-flash --host 0.0.0.0 --port 8000 \
  --pipeline-parallel-size 1 --tensor-parallel-size 4 \
  --max-model-len 262144 --max-num-seqs 8 \
  --max-num-batched-tokens 3460 --long-prefill-token-threshold 0 \
  --prefill-chunk-with-decodes 384 --max-num-partial-prefills 2 \
  --gpu-memory-utilization 0.95 --trust-remote-code \
  --reasoning-parser glm47 --enable-auto-tool-choice --tool-call-parser glm47 \
  --speculative-config '{"method":"dflash","model":"/models/GLM-5.3-Flash-DFlash2","num_speculative_tokens":3}'

docker logs -f glm53-flash          # wait for "Application startup complete"
curl http://127.0.0.1:8000/health
```

If you built the image yourself, use your tag (`vllm-cmp170hx:1.3.0-0ed7d3e7f3`)
in place of the `ghcr.io` name.

The arguments and [`container.env`](container.env) are `serve.sh`'s defaults
for this release, written out, because the container runs `vllm` directly and
not `serve.sh`. Inside the container the server listens on every interface;
`-p 127.0.0.1:8000:8000` keeps it reachable from this machine only, as the
native install does. It has **no API key**: to serve other machines, publish the
port on another address and add `-e VLLM_API_KEY=<a long random string>`.

The first start compiles kernels into the cache mount and takes several
minutes; later starts reuse them. Stop it with `docker stop -t 120 glm53-flash`
and wait until `nvidia-smi` shows the cards empty before starting again. The
API is the same as the native install's ([Talk to it](../README.md#talk-to-it)).

**PCIe peer-to-peer is off** (`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0`), as in
the native install. If peer access already works on your cards
([PCIe peer-to-peer](../README.md#pcie-peer-to-peer-optional)), set
`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` in `container.env`; the
container changes no driver setting.

The kill switches in the [main README](../README.md#kill-switches) work the
same way: set the variable in `container.env` or add `-e NAME=0`.
