# Changelog

## 1.4.0 — {{RELEASE_DATE}}

**Install:** `./start.sh` (see the README). This release runs the engine in a
container by default.

**Updating from 1.3.x:** `./start.sh update`, nothing else. The checkout stays
native (it already has a native install); the venv is rebuilt from scratch for
the new engine, because the pin moved to a new upstream base; the first boot
is about 4–5 minutes slower while FlashInfer compiles two kernel modules
(below); the layout stays tensor-parallel 4. The KV line then reads 1,156,635
tokens with peer-to-peer off instead of 1,174,567 (the corrected figure,
below). To move to the container afterwards: install Docker and the NVIDIA
Container Toolkit, set `RUNTIME=container` in `.env` and `./start.sh restart`
(it stops the native server first).

**Two layouts.** New setting `LAYOUT`: `tp4` (tensor-parallel 4, the default,
as before) or `pp4` (pipeline-parallel 4). Pipeline-parallel 4 gives each card a
quarter of the layers and passes only activations between them: it prefills
2.35× faster than TP4, holds 2.01× the KV, suits
many parallel users and long prompts, and needs far less link bandwidth, which
makes it the layout for cards limited to x4 links. Its numbers were measured on
our x16 cards; x4 links are not measured yet. It runs with the DFlash2 drafter
like TP4 (`SPEC_MODE` now defaults to `dflash` in both layouts). `PP` and `TP`
still override the layout.

**Container first.** `./start.sh` now runs the engine image
`ghcr.io/morrowmake/vllm-cmp170hx@sha256:80bf2f40c1d40c6d20ae5ac101173f77bdd89ca76099c68330949e4d2cbbd89f` (the fork at the new
pin on `nvidia/cuda:13.3.1-devel-ubuntu24.04`, no weights) with this
repository's `serve.sh` as its entry point, as the invoking user rather than
root, with the checkpoints mounted read-only and the compile caches in
`./cache`, owned by that user. It needs Docker with the NVIDIA Container
Toolkit and driver 580 or newer. `install`, `download`, `stop`, `status`,
`logs`, `smoke`, `update` and `DRY=1` all work as before; `stop` only stops the
container this checkout started (recorded in `logs/container.id` and labelled
with the checkout). The native install stays available with `RUNTIME=native`,
and a checkout that already has one keeps using it unless `RUNTIME` says
otherwise. {{CONTAINER_PARITY_LINE}}

**Engine pin moves to `9cdecd00a4`**, on upstream `e55d076f89` (vLLM 0.30.1
development line). Installed version `0.30.1rc1.dev282+g9cdecd00a.precompiled`. The engine now pins
FlashInfer 0.7.0 and humming-kernels 0.1.16. The fork still adds no C++ or CUDA
source; the base has no nightly wheel of its own, so the compiled extensions
come from the wheel of upstream `b6761e8ded`, whose C++, CUDA and Rust sources
are identical. A native install takes the runtime extras from the engine's own
`requirements/cuda.txt`, and rebuilds the venv when those requirements change
with the pin.

**Slower first boot.** On the new base, FlashInfer 0.7.0 compiles two of its
kernel modules (top-k, about 160 s, and sampling, about 60 s) into an empty
cache, so the first boot after installing or updating to 1.4.0 takes about 4–5
minutes longer; later boots reuse the cache. During that build the log can show
`No available shared memory broadcast block found in 60 seconds`; the lines are
harmless.

### Measured with this release's defaults

180 W per card, PCIe x16 links, one server start per column.

| | TP4, peer-to-peer off (default) | TP4, peer-to-peer on (optional) | PP4, peer-to-peer off |
|---|---:|---:|---:|
| Decode step at 1 / 4 / 6 / 8 users, ms | 15.87 / 29.51 / 38.91 / 44.45 | 15.29 / 28.23 / 35.94 / 40.39 | 29.35 at 1 user |
| Decode, 1 user, structured / code / prose | 264.4 / 258.5 / 189.9 tok/s | 275.8 / 274.0 / 204.2 tok/s | 142.2 / 138.6 / 100.7 tok/s |
| Decode, 8 users, aggregate | 763.1 / 684.7 / 512.0 tok/s | 848.3 / 726.9 / 570.0 tok/s | 556.3 / 503.5 / 376.8 tok/s |
| Cold prefill | 2,670 tok/s | 3,062 tok/s | 6,264 tok/s |
| TTFT, 6,217 / 23,255 tokens | 2.37 / 8.67 s | 2.05 / 7.49 s | 1.49 / 3.79 s |
| KV pool at 262,144 | 1,156,635 tokens, 4.41x | 1,177,646 tokens, 4.49x | 2,320,328 tokens, 8.85x |

PP4 with peer-to-peer on: 141.0 / 139.4 / 105.1 tok/s for one user, 532.7 /
493.9 / 377.8 across eight, cold prefill 6,606 tok/s, the same KV pool.

Quality: TP4 perplexity 3.2781, GSM8K 0.972, HumanEval
0.9634; PP4 perplexity 3.2735, GSM8K 0.970, HumanEval
0.9878.

### What changed

**Prefill kernels.** Under PP4, new Ampere kernels for the shapes that layout
runs (64 heads, whole experts): linear-attention (KDA) chunked prefill,
sparse-attention prefill and a split-block Marlin MoE prefill
(`VLLM_GLM5_PP_KDA_PREFILL`, `VLLM_GLM5_PP_SPARSE_MLA_PREFILL`,
`VLLM_GLM5_PP_MARLIN_PREFILL`). Under TP4, the linear-attention prefill at 16
heads (1.42× per prompt per card) and the split-block MoE prefill at TP4 shards
(1.24×) (`VLLM_GLM5_TP4_KDA_PREFILL`, `VLLM_GLM5_TP4_MARLIN_PREFILL`). In both
layouts, the sparse-attention prefill gather no longer reads the same cache row
for every empty top-k slot, with bitwise-identical output
(`VLLM_GLM5_SMLA_PREFILL_PRED_LOAD`). Each kernel is at least as accurate as the
code it replaces against a 64-bit reference on real inputs. Measured in
development: PP4 cold prefill +19.5% (5,520 → 6,586–6,613 tok/s), TTFT on a
23,255-token prompt 4.27 → 3.64 s, decode unchanged. Under TP4, against 1.3.0:
cold prefill 2,484 → 2,670 tok/s (+7.5%) with peer-to-peer off and
2,490 → 3,062 (+23%) with it on (below).
All on by default, each a kill switch at 0.

**Pipeline.** Under PP4, decoding requests are spread over every in-flight
micro-batch (`VLLM_PP_SPREAD_DECODES`), the stage hand-off is packed into one
transfer without a metadata exchange (`VLLM_PP_PACKED_HOP`,
`VLLM_PP_HOP_NO_METADATA`), the drafter's inputs are synchronised more finely
(`VLLM_PP_SPLIT_DRAFT_EVENT`) and its input projection is folded
(`VLLM_GLM5_PP_FOLD_DRAFT_FC`). Together: +7.3% aggregate decode at four users,
+2.0% cold prefill; outputs identical apart from the fold, which is checked
against a 64-bit reference. KV block size 4,608 under PP4 with
DFlash2 (`BLOCK_SIZE`), so the drafter shares the KV layout.

**Honest KV figures.** With prefill chunks in flight, a request can hold more
linear-attention state and drafter window blocks than the engine reserved for
it, so the KV pool it reported was larger than the server could fill: under
TP4, 1.3.x's published figure was 17,932 tokens (1.5%) too high. Near a full
pool that could only lead to a request being paused and resumed, never to
wrong output. The reserve now matches (`VLLM_KV_MAMBA_INFLIGHT_STATES`,
`VLLM_KV_SWA_INFLIGHT_SCRATCH`, both on, in both layouts; 0 reports the old
figure), and every KV figure in the README is the corrected one. Outputs
cannot change.

**The same output on every install.** The linear-attention prefill kernels
picked their tile configuration by timing the options in each new process, so
a fresh install or a cleared compile cache could compute in a different order.
They now use pinned configurations (`VLLM_GLM5_FLA_PIN_AUTOTUNE`, on), exactly
as accurate as the tuned ones against a 64-bit reference and within 0.03% of
their speed on one card.

**Second-generation decode kernels and KV headroom in both layouts.**
`VLLM_GLM5_DECODE_IDX_GLUE`, `VLLM_GLM5_DECODE_KDA_V2`,
`VLLM_GLM5_DECODE_MOE_ROUTE_V2`, `VLLM_GLM5_DECODE_MHC_V2`,
`VLLM_GLM5_DRAFTER_ROPE_FIT`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` and
`VLLM_GLM5_INDEXER_DECODE_ROWS` are now set under PP4 too; the drafter
selector shard stays TP only.

**NCCL over peer-to-peer.** With peer-to-peer on under TP4
(`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1`), `serve.sh` now also sets
`NCCL_P2P_LEVEL=SYS`, so NCCL carries the large prefill collectives card to
card instead of through host memory: +13.7% cold prefill with identical
outputs. `GLM5_NCCL_P2P_SYS=0` turns it off; an explicit `NCCL_P2P_LEVEL` wins.
Nothing changes with peer-to-peer off (the default) or under PP4.

**Release tags on the fork.** Every commit a release has pinned is kept under
a `glm53-recipe-<version>` tag on the fork, so older releases, and
`VLLM_COMMIT=<older pin>` rollbacks, keep installing after the fork's branch
moves to a new upstream base. A native install fetches those tags when a pin
is no longer on the branch.

**Settings.** New: `LAYOUT`, `RUNTIME`, `IMAGE`, `CONTAINER_NAME`, `SHM_SIZE`,
`CACHE_DIR`, `BLOCK_SIZE` (PP4), `VLLM_PRECOMPILED_WHEEL_COMMIT` (native),
`GLM5_NCCL_P2P_SYS`, and the kill switches above. Changed defaults:
`VLLM_COMMIT` (the new pin), `SPEC_MODE` under `PP=4` (`mtp` → `dflash`), and
the flags now set in both layouts. A `VLLM_COMMIT` or `IMAGE` you have
uncommented in `.env` still wins, as it always has (that is how the engine is
frozen), and so does a `PP`/`TP` pair over `LAYOUT`.

**Rolling back.** Container: `IMAGE=<image> ./start.sh update` for one run, or
`IMAGE=` in `.env`; 1.3.x's image is
`ghcr.io/morrowmake/vllm-cmp170hx@sha256:ee978fb3e3d11cf8577a014539a7ad4e2a8dff95163fc5d96c0a06fcf9c64640`.
Native: `VLLM_COMMIT=0ed7d3e7f3` the same way (the venv is rebuilt for the
older engine's requirements). For 1.3.1's scripts and defaults as well,
`git checkout v1.3.1` and `./start.sh restart` (1.3.x installs natively); back
again with `git checkout main` and `./start.sh update`.

## 1.3.1 — 2026-09-26

**Documentation only. No engine change:** the pin stays at `0ed7d3e7f3`, and
`install.sh`, `serve.sh` and every default are unchanged, so `./start.sh update`
from 1.3.0 reinstalls nothing and the 1.3.0 results stand.

**Container image.** New [`docker/`](docker/README.md) folder: the Dockerfile
and the build script for `ghcr.io/morrowmake/vllm-cmp170hx:1.3.0-0ed7d3e7f3`
(the fork at the 1.3.0 pin on `nvidia/cuda:13.3.1-devel-ubuntu24.04`, no
weights), `container.env` with `serve.sh`'s defaults written out (peer-to-peer
off), and the `docker run` command with the two checkpoints and the compile
cache mounted. Needs the NVIDIA Container Toolkit and driver 580 or newer. The
README links it under *Run in a container*; the native install remains the
primary path. The image is published at
`ghcr.io/morrowmake/vllm-cmp170hx@sha256:ee978fb3e3d11cf8577a014539a7ad4e2a8dff95163fc5d96c0a06fcf9c64640`
and runs within 0.4 % of the native install (decode at 1 / 4 / 8 users and
cold prefill).

## 1.3.0 — 2026-09-25

**Install:** `./start.sh` (see the README). Later releases update with
`./start.sh update`: it pulls, reinstalls the engine if its pin moved and
restarts the server. Every setting in `.env.example` is now commented out and
shows its default; uncommenting one overrides that default, so a later
release's new defaults apply without editing `.env`. Each note in
`.env.example` now sits on its own line, and a note left after an unquoted
value (`KEY=value  # note`) is no longer read into the value.

**Engine pin moves to `0ed7d3e7f3`** (98 commits on upstream `496c6472cb`).
Installed version `0.29.1rc1.dev617+g0ed7d3e7f.precompiled`. As before, the
fork adds no C++ or CUDA source, so the install uses upstream's precompiled
extensions for that base.

**Tensor-parallel only.** 1.3.0 supports one layout, tensor-parallel 4
(`PP=1`, `TP=4`), as 1.2.0 did. Pipeline-parallel 4 is planned (below).

### Measured with this release's defaults

180 W per card, one server start per column.

| | Peer-to-peer off (default) | Peer-to-peer on (optional) |
|---|---:|---:|
| Decode step at 1 / 4 / 6 / 8 users, ms | 15.841 / 30.092 / 38.778 / 44.705 | 15.250 / 28.015 / 35.549 / 41.921 |
| Decode, 1 user, structured / code / prose | 264.6 / 260.4 / 188.4 tok/s | 274.8 / 273.0 / 197.9 tok/s |
| Decode, 8 users, aggregate | 745.4 / 664.5 / 519.8 tok/s | 808.1 / 720.5 / 556.3 tok/s |
| Cold prefill | 2,484 tok/s | 2,490 tok/s |
| TTFT, 6,217 / 23,255 tokens | 2.50 / 8.97 s | 2.49 / 8.94 s |
| KV pool at 262,144 | 1,174,567 tokens, 4.48x | 1,187,776 tokens, 4.53x |

Quality, peer-to-peer off: perplexity 3.2858 on the fixed 60-document set
(bit-identical with peer-to-peer on); GSM8K 0.975 on all 1,319 problems, none
truncated; HumanEval pass@1 0.9573 (157/164) at 4,096 tokens per reply, scoring
the last complete code block of the reply (the first block gives 0.8537).

Against 1.0.0, one user: 238.7 → 264.6 tok/s
structured, 232.4 → 260.4 code, 165.1 → 188.4 prose. Cold-prefill ladder from
~8k to ~250k tokens: 2,443 / 2,502 / 2,498 / 2,479 / 2,425 / 2,326 tok/s.

### What changed

**Same request, same output.** A request sent on its own now returns the same
tokens and log-probabilities on every repeat and across restarts. Four sources
of run-to-run variation are fixed: MoE block alignment in a fixed order,
CUDA-graph padding rows kept out of the MoE, and the sparse-attention indexer's
top-k made consistent on ties and returned in a fixed order. New switches, all
on by default, each a kill switch at 0: `VLLM_GLM5_DETERMINISTIC_MOE_ALIGN`,
`VLLM_GLM5_MOE_MASK_PADDING`, `VLLM_GLM5_TOPK_TIEFIX`, `VLLM_GLM5_TOPK_SORTED`,
`VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS` (8). Measured on against off: step time
within restart noise, prefill −0.3%. `serve.sh` refuses
`VLLM_MOE_SKIP_PADDING=0` together with `VLLM_GLM5_MOE_MASK_PADDING=1`.

**Second-generation decode kernels**, on by default under TP, each a kill
switch at 0: `VLLM_GLM5_DECODE_IDX_GLUE`, `VLLM_GLM5_DECODE_KDA_V2`,
`VLLM_GLM5_DECODE_MOE_ROUTE_V2`, `VLLM_GLM5_DECODE_MHC_V2`,
`VLLM_GLM5_DRAFTER_ROPE_FIT`, plus thin-GEMM rows for 24-row batches. Measured
together with peer-to-peer on: decode step 16.26 → 15.05 ms at one user,
29.98 → 27.76 at four, 42.00 → 34.62 at six, 43.92 → 41.55 at eight; cold
prefill flat.

**KV headroom**, on by default under TP: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`
(512, upstream's value, switches it off), `VLLM_GLM5_DRAFTER_SELECTOR_SHARD=1`,
`VLLM_GLM5_INDEXER_DECODE_ROWS=1`; the engine's own
`VLLM_GLM5_INDEXER_GATHER_CLAMP` (on) is documented as a kill switch. Together
+39,626 KV tokens (+3.45%, measured with peer-to-peer on), outputs
bit-identical, no step-time cost.

**Accuracy.** Every custom kernel on the default path is now checked on real
inputs against a 64-bit reference, side by side with the code it replaces. The
mHC pre-norm projection in the prefill kernels (`VLLM_GLM5_PREFILL_KERNELS`,
shipped since 1.0.0) lost precision in its tensor-core accumulation: its mean
error was 22.8× upstream's and is now 0.93×, for about +0.9 ms per 3,456-token
chunk. The new mHC decode and indexer-glue kernels had the same defect and were
fixed before release.

**KV-pool fix.** With speculative decoding, a rejected draft could overwrite
the per-request tail of the sparse-attention key pool, leaving a wrong pool key
in the indexer cache past 2,048 tokens of context. The tail ring is now sized
for the draft depth, at no KV cost.

**Startup fix.** Startup no longer hangs when the host-memory all-reduce fails
to set up on some cards (peer-to-peer off); all cards now agree and fall back
together.

**3,456-token prefill chunks.** The default `MAX_BATCHED` goes from 2048
(1,152-token chunks) to 3460 (3,456-token chunks). 2,304-token chunks measured
+8.0% cold prefill against 1,152; 3,456 adds +2.4% on top at 180 W per card,
for about 35,600 fewer KV tokens (−3%), with decode step time unchanged.
`MAX_BATCHED=2048` restores the previous chunking.

**Optional PCIe peer-to-peer.** Where the driver advertises peer access, the TP
all-reduce can run in device memory rather than staging through the host:
`VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=1`, with `VLLM_CUSTOM_ALLREDUCE_ALGO` and
the caching-allocator choice tied to the same variable. On this release: step
time −3.7% / −6.9% / −8.3% / −6.2% at 1 / 4 / 6 / 8 users, cold prefill
unchanged, +13,209 KV tokens. **It ships at 0**, because it needs peer-to-peer
enabled at the driver level; at 0 the recipe needs no driver change. See
"PCIe peer-to-peer (optional)" in the README.

**Prefill overlap beside a live CustomAllreduce.** The overlap no longer stands
down when a CustomAllreduce is present; its split collectives ride NCCL
(`VLLM_GLM5_PREFILL_OVERLAP_BACKEND`, code default `nccl`). That is what
removes the prefill cost the peer-to-peer path used to carry.

The variables the engine reads directly (`VLLM_GLM5_PROLOGUE_FUSE` and its
three sub-switches, `VLLM_GLM5_AUX_HIDDEN_TENSOR`,
`VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR` and `VLLM_CUSTOM_ALLREDUCE_ALGO`) are now
declared, so starting the server no longer prints unknown-variable warnings.

**The API listens on 127.0.0.1 by default.** `HOST` in `.env` was never
passed to the server, which listened on every interface; `serve.sh` now passes
`--host "${HOST:-127.0.0.1}"`. To serve other machines, set `HOST=0.0.0.0` and
set the new `API_KEY` (every `/v1` request must then carry it as a bearer
token; `smoke` and `status` send it). With no `API_KEY` there is no key.

**`SERVED_MODEL_NAME` now works.** It was documented but `serve.sh` read
`SERVED_NAME`, so the model always served as `glm-5.3-flash`. Both names are
read now (`SERVED_MODEL_NAME` wins), and `smoke` uses the same name.

**`stop`, `restart` and `update` no longer wait on the running server.** The
server inherited the lock `start.sh` takes, so a later `stop` could give up
and `restart`/`update` refused while it ran. The server no longer holds it,
and if a server started by an earlier release still does, `start.sh` sees that
only that server holds it and carries on; a command that is really running
still makes the checkout busy.

**Preflight.** It now stops, before any download, if a card reports less than
60 GiB (`MIN_GPU_MIB` overrides), and `./start.sh download` runs it too. Disk
figures are given in GiB.

**Scripts.** `DRY=1 ./start.sh` (and `DRY=1 ./serve.sh`) now print the
environment the server would get as well as the command, and `DRY=1
./start.sh` no longer installs or downloads anything first: it says what a
real start would do. With `stop`, `restart` or `update`, `DRY=1` takes the
checkout's lock and says what it would stop, without stopping, pulling or
installing anything. `smoke.sh` no longer needs `bc`. `smoke.sh` no longer
sends `enable_thinking: false`: the chat request uses `reasoning_effort: "low"`
and the tool call uses the model's default reasoning, and the smoke test now
fails if no tool call is parsed.

**README rewritten**: results first, what makes it fast and correct, a
step-by-step guide including the peer-to-peer on/off choice, and the roadmap.

**Rolling back.** To go back to the previous engine, set `VLLM_COMMIT=434dea1a1b`
in `.env` and run `./start.sh update`. A prefix assignment
(`VLLM_COMMIT=434dea1a1b ./start.sh update`) lasts one run: the next plain
start or restart reinstalls `0ed7d3e7f3`. That rolls back the engine only; for
the 1.2.0 scripts and defaults as well, `git checkout v1.2.0` and
`./start.sh restart` (back again: `git checkout main`, then `./start.sh update`).
Engine pins from 1.1.0 on (`69c33802d0`, `434dea1a1b`) are on the fork branch;
1.0.x's `cf80da1839` predates a rebase of the branch and a fresh install cannot
fetch it.

### Planned

**Pipeline-parallel 4** (`PP=4`), for systems without the x16 capacitor
modification, whose cards run on narrower PCIe links, and for many parallel
agents and long prompts. It is in active development and comes in a later
release with its own validation and numbers.


## 1.2.0 — 2026-09-22

**Engine pin moves to `434dea1a1b`** (42 commits on upstream `496c6472cb`).
Installed version `0.29.1rc1.dev561+g434dea1a1.precompiled`.

**Correctness.** 64-bit KV row offsets in the sparse-attention kernels, closing
a silent-corruption risk above roughly 4.2M KV tokens; a vocabulary clamp in
three sampler kernels; a 512 MiB transient freed in the indexer's chunk loop.

**Performance.** The sparse-MLA decode schedule is retuned — wider KV tile, two
pipeline stages, head tile sized to the rank — with the old schedule available
as `VLLM_GLM5_SPARSE_MLA_DECODE_LEGACY=1`. A new flag,
`VLLM_GLM5_SHARED_EXPERT_REORDER` (default 1, kill switch 0), submits the MoE
shared experts after the routed dispatch so the two overlap: shared-expert GEMM
time overlapping the routed kernels 0.01% -> 73.3% on a rank-0 decode trace.

Measured against the previous pin with alternating restarts: ms/step at one
stream 17.16 -> 17.01 (-0.85%), at four streams 32.27 -> 32.10 (-0.53%), cold
prefill flat at ~2,238 tok/s, TTFT 2.59 s at 6.2K and 9.74 s at 23K, GSM8K
0.98-1.00, KV pool 1,160,192 tokens (4.43x). A 262,143-token prompt — the
largest a 262,144-token context accepts — is served in 126 s, and a
200,043-token prompt in 94 s. Single-stream decode on this commit's parent
tree averaged 169.5, 167.5 and 172.0 tok/s over three runs; the decode rows in
the README come from a different protocol and are unchanged.


## 1.1.1 — 2026-09-22

Fork repository renamed to `vllm-cmp170hx`; no code change. Every clone URL
and link now names it directly rather than leaning on GitHub's redirect from
the old path. The commit pin is unchanged at `69c33802d0`, so an existing
install is already correct and does not need rebuilding — only a checkout's
`origin` remote is worth repointing.


## 1.1.0 — 2026-09-22

**Engine pin moves to `69c33802d0`.** The fork branch was rebased onto upstream
main (`496c6472cb`), carrying 32 commits, and picked up two drafter
cache-handling fixes along the way. Installed version string is now
`0.29.1rc1.dev551+g69c33802d.precompiled` — taken from a clean install of the
pin, not from a running server, because a server can be ahead of what the fork
publishes.

Validated on four sm_80 cards against the previous pin: ms/step at one stream
17.11 vs 17.61, at four streams 32.04 vs 32.86, cold prefill 2,243 tok/s,
TTFT on a 23,255-token prompt 9.77 s, GSM8K 0.980 on a 50-problem sample, KV pool 1,160,192
tokens at a 262,144-token context.

`./start.sh install` needs no change for the new pin, but **a pin bump is not
a `git checkout`**. The fork adds no C++ of its own, so the compiled
extensions always come from upstream — but upstream's own ABI moved across
this rebase (`_moe_C::moe_align_block_size` gained a `scatter_idx` argument),
so extensions built for the old base will not run the new Python. `start.sh`
handles this: it records the installed commit in `venv/.recipe-stamp`, so a
changed pin re-runs the install and downloads the extensions for the new base
commit. Updating the checkout without reinstalling fails at engine start with
`expected at most 7 argument(s) but received 8`.


## 1.0.4 — 2026-09-22

Tensor-parallel only. The pipeline-parallel material is replaced by one
statement about link width and a pointer to a recipe that covers it.


## 1.0.3 — 2026-09-22

Renamed the closing README heading to "Source".


## 1.0.2 — 2026-09-22

Title names the model and the cards so the repo can be found by either; the
hook moves to a subtitle under the byline.


## 1.0.1 — 2026-09-22

README leads with what the machine does: the model, the rate and what the cards
cost, with a byline and badges. Documentation trimmed to what the thing is and
what it is worth, with nothing about how it was made.


## 1.0.0 — 2026-09-22

First public cut of the recipe.

**Engine.** Pinned to [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
`ampere-glm53` at commit `cf80da1839`, "[GLM-5.3-Flash] Host-staged all-reduce
for PCIe-only nodes without P2P". (That commit predates a later rebase of the
branch and can no longer be installed; roll back no further than 1.1.0.) Seven sm_80 features, each off by default in
the engine and turned on by `serve.sh`, each with a one-variable kill switch:
Ampere sparse-MLA / indexer / kpool backends, TP prefill comm/compute overlap,
host-staged all-reduce for nodes without peer access, thin-M BF16 GEMM, sm_80
decode kernels, sm_80 prefill kernels, fused decode prologue with
batch-sharded logits, and decode-aware fair chunked prefill. Pipeline
parallelism is enabled in the engine but not supported by this recipe.

**Serving.** GLM-5.3-Flash at W4A16 (group size 128) across four GPUs, TP=4,
262,144-token context, DFlash2 speculation at k=3, glm47 reasoning and
tool-call parsers, vision and video on. No FP8, no KV-cache quantisation, no
offload. KV pool 1,158,144 tokens at `--gpu-memory-utilization 0.95`.

**Interface.** One entry point, `./start.sh`, with `install`, `download`,
`stop`, `restart`, `status`, `logs`, `update`, `smoke` and `help`. Every step
is idempotent and skipped when already done. Configuration in `.env`, copied
from `.env.example` on first run; a prefix env assignment overrides it for
every key. `install.sh`, `download.sh` and `stop.sh` are thin wrappers.

**Results.** Decode and cold-prefill measurements taken 2026-09-18 against
MiaAI-Lab's published 2x DGX Spark figures, using their benchmark prompts.

**Known limits.** Tensor-parallel only. These features target TP=4 shapes
and the recipe assumes PCIe Gen2 x16 links between the cards; for a
pipeline-parallel recipe on the same hardware see
[JJ48/glm53-flash-170hx-serving](https://github.com/JJ48/glm53-flash-170hx-serving).