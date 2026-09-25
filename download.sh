#!/usr/bin/env bash
# download.sh — fetch the two checkpoints into ./models.
#
#   canada-quant/GLM-5.3-Flash-W4A16-MTP    ~178 GiB (191 GB), 22 files   (target)
#   incoai/GLM-5.3-Flash-DFlash2            ~2.2 GiB (2.3 GB),   5 files   (drafter)
#
# Budget ~185 GiB of free disk. Needs the venv, so run ./install.sh first.
# Runs the preflight first, so it stops before downloading if a card reports
# under 60 GiB or the disk is too small.
#
# Already present? Exits 0. Re-fetch:  REFRESH_WEIGHTS=1 ./download.sh
# Skip the drafter:                    SPEC_MODE=mtp ./download.sh
#
# Equivalent to: ./start.sh download
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" download
