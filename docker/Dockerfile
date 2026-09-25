# vLLM (Morrowmake/vllm-cmp170hx, branch ampere-glm53) for GLM-5.3-Flash W4A16
# on 4x NVIDIA CMP 170HX, as pinned by glm53-flash-cmp170hx-recipe 1.3.0.
#
# The engine is installed the way the recipe's install.sh installs it: Python
# 3.12, torch 2.13.0 (CUDA 13.0 build), the fork checked out at the pinned
# commit and installed editable with upstream's precompiled extensions
# (VLLM_USE_PRECOMPILED=1), then the same runtime extras. Model weights are not
# in the image; mount them (see README.md in this folder for the run command).
#
# The CUDA 13.3.1 devel base matches the toolkit the release was validated
# with (Triton, TileLang, FlashInfer and the host-memory all-reduce extension
# compile against it at runtime, so nvcc and g++ must be present).
# NVIDIA_REQUIRE_CUDA is relaxed to cuda>=13.0 so the container starts on any
# driver 580 or newer, the same requirement as the recipe.
#
# Entry point: the vllm CLI. Arguments and environment are given at run time
# (container.env in this folder holds the release defaults).
#
#   docker build -t vllm-cmp170hx:1.3.0-0ed7d3e7f3 docker/

ARG BASE=nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87

FROM ${BASE} AS build
ARG VLLM_REPO=https://github.com/Morrowmake/vllm-cmp170hx.git
ARG VLLM_BRANCH=ampere-glm53
ARG VLLM_COMMIT=0ed7d3e7f3f855646a139701598a9b40d5745688
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
RUN VLLM_USE_PRECOMPILED=1 uv pip install --extra-index-url https://flashinfer.ai/whl/ -e /opt/vllm-src
RUN uv pip install --extra-index-url https://flashinfer.ai/whl/ \
        "flashinfer-python==0.6.18.post1" "flashinfer-cubin==0.6.18.post1" \
        "tilelang==0.1.12" "apache-tvm-ffi==0.1.11" "ninja" "huggingface_hub[hf_xet]>=1.0" \
 && uv pip freeze > /opt/venv/requirements.lock.txt
RUN CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -c "import torch, vllm, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops, triton, tilelang, flashinfer; from vllm.v1.attention.backends.mla import triton_mla_sparse; from vllm.v1.attention.ops import triton_mqa_logits, triton_e4m3; from vllm.v1.worker.gpu import prologue_fuse; from vllm.distributed.device_communicators import host_shm_all_reduce; print(vllm.__version__, torch.__version__, torch.version.cuda)" \
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
      org.opencontainers.image.description="vLLM fork for GLM-5.3-Flash W4A16 on 4x NVIDIA CMP 170HX (sm_80) as pinned by glm53-flash-cmp170hx-recipe 1.3.0; weights not included" \
      org.opencontainers.image.source="https://github.com/Morrowmake/vllm-cmp170hx" \
      org.opencontainers.image.revision="0ed7d3e7f3f855646a139701598a9b40d5745688" \
      org.opencontainers.image.url="https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/tree/v1.3.0" \
      org.opencontainers.image.version="1.3.0-0ed7d3e7f3" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04" \
      org.opencontainers.image.base.digest="sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
EXPOSE 8000
ENTRYPOINT ["/opt/venv/bin/vllm"]
