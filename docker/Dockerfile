# vLLM (Morrowmake/vllm-cmp170hx, branch ampere) for GLM-5.3-Flash W4A16
# on 4x NVIDIA CMP 170HX, with optional sm_80 Marlin.
#
# The engine is installed the way the recipe's native install does it: Python
# 3.12, torch 2.13.0 (CUDA 13.0 build), the fork checked out at the pinned
# commit and installed editable with upstream's precompiled extensions
# (VLLM_USE_PRECOMPILED=1), then the runtime extras the engine pins in
# requirements/cuda.txt (FlashInfer 0.7.0, humming-kernels 0.1.16, TileLang
# 0.1.12). The pin sits on upstream e55d076f89, which has no nightly wheel of
# its own; the extensions come from the wheel of b6761e8ded, whose C++, CUDA
# and Rust sources are identical to e55d076f89 (the two commits between them
# are Python and tests). Model weights are not in the image; mount them (see
# README.md in this folder for the run command).
#
# The CUDA 13.3.1 devel base matches the toolkit the release was validated
# with (Triton, TileLang, FlashInfer and the host-memory all-reduce extension
# compile against it at runtime, so nvcc and g++ must be present).
# NVIDIA_REQUIRE_CUDA is relaxed to cuda>=13.0 so the container starts on any
# driver 580 or newer, the same requirement as the recipe.
#
# Entry point: the vllm CLI. The recipe runs the image with its own serve.sh
# mounted and used as the entry point, so the layout (LAYOUT=tp4 or pp4) and
# every setting are chosen at run time (container.env in this folder).
#
#   bash docker/build-docker.sh vllm-cmp170hx:1.6.0-3a2bf16dae
# The wrapper validates the final layer count; Dockerfile instructions cannot
# inspect the inherited base image manifest.

ARG BASE=nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87

ARG IMAGE_LAYERED=1

ARG VLLM_COMMIT=3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340

FROM ${BASE} AS build
ARG PYTHON_VERSION=3.12
ARG PYTHON_INSTALL_VERSION=3.12.14
ARG UV_VERSION=0.12.11
COPY CONSTRAINTS /tmp/CONSTRAINTS
ENV UV_CONSTRAINT=/tmp/CONSTRAINTS
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python-dl \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh
RUN uv python install "${PYTHON_INSTALL_VERSION}" \
 && mv /opt/python-dl/cpython-${PYTHON_VERSION}.*-linux-x86_64-gnu /opt/python \
 && rm -rf /opt/python-dl \
 && grep -rlI --null /opt/python-dl/ /opt/python \
    | xargs -0 -r sed -i -E "s#/opt/python-dl/cpython-[^/\"' ]+#/opt/python#g" \
 && uv venv --relocatable --python /opt/python/bin/python${PYTHON_VERSION} /opt/venv
ENV VIRTUAL_ENV=/opt/venv
# Dependencies are independent of the engine pin and install from the lock.
RUN uv pip install --extra-index-url https://flashinfer.ai/whl/ -r /tmp/CONSTRAINTS
ARG VLLM_REPO=https://github.com/Morrowmake/vllm-cmp170hx.git
ARG VLLM_BRANCH=ampere
ARG VLLM_COMMIT
ARG VLLM_WHEEL_COMMIT=b6761e8ded57ef85b708f34af8cab1649eae1069
# Full transport clone: version construction and wheel lookup need history.
RUN git clone --no-local --no-tags --single-branch --branch "${VLLM_BRANCH}" "${VLLM_REPO}" /opt/vllm-src \
 && git -C /opt/vllm-src checkout -B "${VLLM_BRANCH}" "${VLLM_COMMIT}" \
 && git -C /opt/vllm-src update-ref "refs/remotes/origin/${VLLM_BRANCH}" "${VLLM_COMMIT}" \
 && git -C /opt/vllm-src fetch --no-tags https://github.com/vllm-project/vllm.git \
            refs/tags/v0.30.1rc0:refs/tags/v0.30.1rc0 \
 && git -C /opt/vllm-src merge-base --is-ancestor v0.30.1rc0 HEAD \
 && git -C /opt/vllm-src submodule update --init --recursive --depth 1 \
 && git -C /opt/vllm-src reflog expire --expire=now --all \
 && git -C /opt/vllm-src repack -ad \
 && git -C /opt/vllm-src prune --expire=now
RUN VLLM_BUILD_AMPERE_MARLIN=0 VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_WHEEL_COMMIT}" \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    uv pip install --no-deps -e /opt/vllm-src \
 && uv pip freeze > /opt/venv/requirements.lock.txt
COPY runtime_tree.py payload.py cache_seed.py rootfs_smoke.py /tmp/
RUN test -f /opt/vllm-src/csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py \
    || (echo "Set VLLM_REPO, VLLM_BRANCH and VLLM_COMMIT to a source with optional Marlin support" >&2; exit 1)
RUN cd /opt/vllm-src && CUDA_VISIBLE_DEVICES= VLLM_BUILD_AMPERE_MARLIN=1 \
    PATH=/opt/venv/bin:$PATH /opt/venv/bin/python \
    csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py --out /opt/vllm-src/vllm \
    --build-dir /tmp/ampere-marlin-build \
 && rm -rf /tmp/ampere-marlin-build
RUN CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -c "from vllm.ampere_marlin import require_extension; require_extension()"
RUN CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -c "import torch, vllm, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops, triton, tilelang, flashinfer, humming; from vllm.v1.attention.backends.mla import triton_mla_sparse; from vllm.v1.attention.ops import triton_mqa_logits, triton_e4m3; from vllm.v1.worker.gpu import prologue_fuse; from vllm.distributed.device_communicators import host_shm_all_reduce; print(vllm.__version__, torch.__version__, torch.version.cuda)" \
 && find /opt/python /opt/venv /opt/vllm-src -name __pycache__ -type d -prune -exec rm -rf {} + \
 && /opt/venv/bin/python /tmp/runtime_tree.py finalize /opt "${VLLM_COMMIT}" \
 && /opt/venv/bin/python /tmp/runtime_tree.py check /opt "${VLLM_COMMIT}"

RUN /opt/venv/bin/python /tmp/payload.py /opt /payload /tmp/CONSTRAINTS \
 && mkdir -p /payload/engine/opt/image-tools \
 && cp /tmp/cache_seed.py /tmp/rootfs_smoke.py /payload/engine/opt/image-tools/

FROM ${BASE} AS runtime-1
COPY --from=build /payload/python/ /
COPY --from=build /payload/dependencies/ /
COPY --from=build /payload/native/ /
COPY --from=build /payload/engine/ /

FROM ${BASE} AS runtime-0
COPY --from=build /opt/ /opt/
COPY --from=build /payload/engine/opt/image-tools/ /opt/image-tools/

FROM runtime-${IMAGE_LAYERED}
ARG VLLM_COMMIT
ARG RELEASE=1.6.0
ENV PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_REQUIRE_CUDA=cuda>=13.0
LABEL org.opencontainers.image.title="vllm-cmp170hx" \
      org.opencontainers.image.description="vLLM fork for GLM-5.3-Flash W4A16 on 4x NVIDIA CMP 170HX (sm_80) with optional sm_80 Marlin; weights not included" \
      org.opencontainers.image.source="https://github.com/Morrowmake/vllm-cmp170hx" \
      org.opencontainers.image.revision="${VLLM_COMMIT}" \
      org.opencontainers.image.url="https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe" \
      org.opencontainers.image.version="${RELEASE}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04" \
      org.opencontainers.image.base.digest="sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
EXPOSE 8000
ENTRYPOINT ["/opt/venv/bin/vllm"]
