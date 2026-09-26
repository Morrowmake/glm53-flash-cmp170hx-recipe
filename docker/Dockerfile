# vLLM (Morrowmake/vllm-cmp170hx, branch ampere-glm53) for GLM-5.3-Flash W4A16
# on 4x NVIDIA CMP 170HX, as pinned by glm53-flash-cmp170hx-recipe 1.4.0.
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
#   docker build -t vllm-cmp170hx:1.4.0-PIN_PENDING docker/

ARG BASE=nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87

FROM ${BASE} AS build
ARG VLLM_REPO=https://github.com/Morrowmake/vllm-cmp170hx.git
ARG VLLM_BRANCH=ampere-glm53
ARG VLLM_COMMIT=PIN_PENDING
ARG VLLM_WHEEL_COMMIT=b6761e8ded57ef85b708f34af8cab1649eae1069
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.11
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python-dl \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh
RUN uv python install "${PYTHON_VERSION}" \
 && mv /opt/python-dl/cpython-${PYTHON_VERSION}.*-linux-x86_64-gnu /opt/python \
 && rm -rf /opt/python-dl \
 && grep -rlI --null /opt/python-dl/ /opt/python \
    | xargs -0 -r sed -i -E "s#/opt/python-dl/cpython-[^/\"' ]+#/opt/python#g" \
 && uv venv --relocatable --python /opt/python/bin/python${PYTHON_VERSION} /opt/venv
# Blobless, not shallow: the precompiled-wheel lookup needs the history.
RUN git clone --filter=blob:none --branch "${VLLM_BRANCH}" "${VLLM_REPO}" /opt/vllm-src \
 && git -C /opt/vllm-src checkout -B "${VLLM_BRANCH}" "${VLLM_COMMIT}" \
 && git -C /opt/vllm-src submodule update --init --recursive --depth 1
ENV VIRTUAL_ENV=/opt/venv
RUN uv pip install --extra-index-url https://flashinfer.ai/whl/ \
        "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"
RUN VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_WHEEL_COMMIT}" \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    uv pip install --extra-index-url https://flashinfer.ai/whl/ -e /opt/vllm-src
RUN uv pip install --extra-index-url https://flashinfer.ai/whl/ \
        -r /opt/vllm-src/requirements/cuda.txt \
        "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0" \
        "flashinfer-python==0.7.0" "flashinfer-cubin==0.7.0" \
        "tilelang==0.1.12" "apache-tvm-ffi==0.1.11" "ninja" "huggingface_hub[hf_xet]>=1.0" \
 && uv pip freeze > /opt/venv/requirements.lock.txt
RUN CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -c "import torch, vllm, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops, triton, tilelang, flashinfer, humming; from vllm.v1.attention.backends.mla import triton_mla_sparse; from vllm.v1.attention.ops import triton_mqa_logits, triton_e4m3; from vllm.v1.worker.gpu import prologue_fuse; from vllm.distributed.device_communicators import host_shm_all_reduce; print(vllm.__version__, torch.__version__, torch.version.cuda)" \
 && find /opt/python /opt/venv /opt/vllm-src -name __pycache__ -type d -prune -exec rm -rf {} + \
 && rm -rf /opt/vllm-src/.git/logs

FROM ${BASE}
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/vllm-src /opt/vllm-src
ENV PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_REQUIRE_CUDA=cuda>=13.0
LABEL org.opencontainers.image.title="vllm-cmp170hx" \
      org.opencontainers.image.description="vLLM fork for GLM-5.3-Flash W4A16 on 4x NVIDIA CMP 170HX (sm_80) as pinned by glm53-flash-cmp170hx-recipe 1.4.0; weights not included" \
      org.opencontainers.image.source="https://github.com/Morrowmake/vllm-cmp170hx" \
      org.opencontainers.image.revision="PIN_PENDING" \
      org.opencontainers.image.url="https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/tree/v1.4.0" \
      org.opencontainers.image.version="1.4.0-PIN_PENDING" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04" \
      org.opencontainers.image.base.digest="sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
EXPOSE 8000
ENTRYPOINT ["/opt/venv/bin/vllm"]
