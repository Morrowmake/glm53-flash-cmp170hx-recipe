#!/usr/bin/env bash
# download.sh — fetch the two checkpoints into ./models.
#
#   canada-quant/GLM-5.3-Flash-W4A16-MTP    ~178 GB, 21 files   (target)
#   incoai/GLM-5.3-Flash-DFlash2            ~2.2 GB,  5 files   (drafter)
#
# Budget ~185 GB of free disk. Needs the venv, so run ./install.sh first.
#
# Already present? Exits 0. Re-fetch:  REFRESH_WEIGHTS=1 ./download.sh
# Skip the drafter:                    SPEC_MODE=mtp ./download.sh
#
# Equivalent to: ./start.sh download
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" download
