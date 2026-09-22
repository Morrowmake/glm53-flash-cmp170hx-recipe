#!/usr/bin/env bash
# install.sh — build the venv and the pinned vLLM fork.
#
# Creates ./venv on Python 3.12, clones https://github.com/Morrowmake/vllm at
# the commit pinned in .env (or start.sh's default), installs it editable with
# torch 2.13.0+cu130 and the runtime extras, then verifies every import.
#
# Already built at that commit? Exits 0 without touching it.
# Rebuild the venv: VENV_CLEAR=1 ./install.sh
# Reinstall anyway:  FORCE_INSTALL=1 ./install.sh
# Compile the CUDA extensions instead of using upstream's precompiled ones:
#                    BUILD_FROM_SOURCE=1 MAX_JOBS=16 ./install.sh
#
# Equivalent to: ./start.sh install
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" install
