# Results

## Release 1.6.0: faster decode for one or two users

Engine changes, all on by default:

- **Adaptive draft depth.** The DFlash2 drafter's guesses are verified up to
  seven at a time when one request is running, up to five at two, and three (the
  previous fixed depth) under heavier load. Within those limits each request's
  depth follows its own recent acceptance and the measured cost of each depth on
  this hardware, so text that drafts well (code, structured output) verifies
  deep and prose stays shallow. Under tensor-parallel 4 the drafter itself
  shrinks to the verified depth. Kill switches in
  [engine switches](engine-switches.md).
- **Compiled Marlin decode on by default, original reduction order.** The
  optional library's decode kernels now run in both layouts, in a faster order
  that sums the first MoE projection in four fixed K slices. Decoded text can
  differ from the released order; quality was checked per answer (below).
  `VLLM_GLM5_MARLIN_DECODE_VARIANT=exact` keeps the previous order
  ([Optional compiled Marlin](compiled-marlin.md)).

**The trade: KV pool.** Deeper drafts reserve more per-request scratch, so the
KV pool at 262,144 context is 1,072,150 tokens under TP4 (−8.9 % against 1.5.x)
and 1,914,216 under PP4 (−18.0 %). Cold prefill is unchanged.

**Against 1.5.x, same protocol:** one user +47.4 / +44.7 / −3.0 % (structured /
code / prose) under TP4 and +65.9 / +48.7 / −3.2 % under PP4; eight users
+5.1 / +1.3 / +2.6 % (TP4) and +3.3 / +2.4 / +4.7 % (PP4). Prose, whose drafts
are accepted least often, stays within a few percent of 1.5.x.

**Quality**, fixed-batch HumanEval (164) and GSM8K (1,319), compared per
answer with the previous release. **TP4, measured on the 1.6.0 release build**
(adaptive depth up to 7, drafter width on): **162/164 and 1,285/1,319 (97.42 %)**
(previous 162 and 1,281; HumanEval 2 losses and 2 gains, McNemar p 1.0; GSM8K
6 gains and 2 losses, p 0.29). **PP4, measured on the 1.6.0 decode kernels with
adaptive depth up to 5** (no PP4 run on the final build): **162/164 and
1,280/1,319** (previous 163 and 1,284; p 1.0 and 0.22). Earlier TP4 runs on the
way to this release scored 160/164 and 1,280/1,319. Individual answers flip in
both directions when decoded text changes; no suite shows a significant net loss.

**Long-context and repeatability checks.** The full tier-2 long-context and
repeatability checks ran on the same engine build: needle retrieval 30/30 up
to 262K tokens, copy fidelity no worse than the previous baseline, no
cross-request leaks, and bit-identical outputs for the same batch run twice in
eager and CUDA-graph modes, in both layouts. The README numbers and the needle
and repeatability checks were re-measured on the final candidate: needle
retrieval at 32K and 262K, and the same request repeated alone gives identical
output (12 of 12 prompts) in both layouts.

## Release 1.6.0 throughput

Measured on 2026-10-01 with the released engine and launcher defaults, the
same workloads and client as the 1.5.0 table below: DFlash2 with adaptive
depth, 262,144-token context, 180 W per card, PCIe x16 links, one server start
per column, decode median of five runs.

| | TP4, peer-to-peer off (default) | TP4, peer-to-peer on (optional) | PP4, peer-to-peer off (`LAYOUT=pp4`) |
|---|---:|---:|---:|
| Streaming decode, 1 user, structured / code / prose | **394.0 / 377.4 / 180.5 tok/s** | **437.4 / 404.2 / 198.6 tok/s** | **235.3 / 207.9 / 100.2 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **797.9 / 683.2 / 544.7 tok/s** | **840.7 / 769.4 / 589.5 tok/s** | **623.1 / 591.0 / 465.8 tok/s** |
| Cold prefill | **2,669 tok/s** | **3,061 tok/s** | **6,580 tok/s** |
| One-token response time, 6,217 / 23,255-token prompt | 2.38 / 8.67 s | 2.04 / 7.49 s | 1.46 / 3.63 s |
| KV pool at 262,144 context | 1,072,150 tokens (4.09 full-length requests) | 1,073,093 tokens (4.09) | 1,914,216 tokens (7.30) |

Of the 855 measured decode requests, one (TP4 peer-to-peer on, structured,
eight users) stopped at 262 tokens; all others ran to the 400-token cap.

## Optional Marlin and image packaging (1.5.0)

Release 1.5.0 adds optional compiled Marlin and a Git-free runtime image.
The throughput tables below were remeasured on the native 1.5.0 engine with
the optional library installed. Quality results retain their explicitly named
release scope; throughput measurements are not quality or universal exactness claims.
See [Optional compiled Marlin](compiled-marlin.md) for defaults and limits.

## Allocator compatibility default (1.4.3)

With DFlash2 k=3 and a 262,144-token context, the new
`expandable_segments:False` default measured **1,176,646 KV tokens under
TP4/peer-to-peer off** and **2,334,498 under PP4**: respectively +20,011 tokens
(+20 physical blocks) and +14,170 (+4) against True on the same engine.
These are not peer-to-peer gains. Explicit allocator overrides remain intact.

Four alternating True/False starts per layout passed startup checks, short/long
decode-drift and KL-divergence comparisons, and single-request repeatability,
with no new GPU Xid errors. TP4 paired decode-step cost was +0.38–0.83% at
1/4/6/8 users; PP4 paired means at 1/4/8 differed by at most 0.21%. PP4 at six
users was scheduling-confounded; no speed conclusion is drawn there.
Fixed-batch repeatability also passed under TP4 and PP4 in both eager and
CUDA-graph modes. Native update checks passed with `.env` preserved and no
engine reinstall. Validation for this change is DFlash-only; conflicting
allocator aliases were not GPU-validated.
It does not establish a root-cause fix for driver or virtualisation failures.

## Quality (1.4.3)

Full native fixed-batch results with DFlash2 k=3, repeated twice in each configuration:

| Layout | HumanEval, pass@1 (164 tasks) | GSM8K (1,319 problems) |
|---|---:|---:|
| TP4, peer-to-peer off | **162/164 — 98.78%** | **1,281/1,319 — 97.12%** |
| PP4, peer-to-peer off | **163/164 — 99.39%** | **1,284/1,319 — 97.35%** |

The 1.4.0 baseline and 1.4.3 configuration matched exactly on generated token
IDs, answers and scores in both full repeats of each same-layout comparison.
The PP4 comparisons also isolated the engine update, draft-tail enablement
and allocator default: all were exact, with no per-task gains or losses.
This is evidence for those measured paths, not universal equivalence across
layouts, batches or workloads.

These results use fixed-order batches of up to eight, identical prompts and
generation settings within each comparison, and a corrected HumanEval scorer
that executes the extracted completion with its supplied task context.
HumanEval allows 4,096 tokens per reply; GSM8K allows 3,072. Length-capped
outputs remain in the scores. These controlled results supersede the earlier
unmatched quality figures; those separate runs are not release-change controls.

**Rolling concurrency, separately measured:** on the unchanged 1.4.0
TP4/peer-to-peer-on server, HumanEval scored **162/164 twice serially** and
**163/164, 162/164 and 162/164** with up to eight continuously replenished
requests in original, shuffled and reversed order. There was no demonstrated
aggregate accuracy loss, but individual tasks both improved and worsened.
This is not a 1.4.3 or PP4 comparison, nor proof of batch invariance.

## Release 1.5.0 throughput (previous release)

Measured on 2026-09-30 using the original throughput workloads and the released
engine and launcher. These are native measurements, not a new container-speed
comparison. The optional library was installed: compiled decode uses its release
defaults (off for TP4, on for PP4), compiled prefill is off, and
`expandable_segments:False` applies to all three columns. Native installations
without the optional library do not use the measured PP4 compiled-decode path.

DFlash2 at k=3, 262,144-token context, the defaults of release 1.5.0. Three
columns: tensor-parallel 4 with the cards talking through the host (the
default), tensor-parallel 4 with the optional
[PCIe peer-to-peer](how-to-use.md#pcie-peer-to-peer-optional) path, and pipeline-parallel 4
(`LAYOUT=pp4`, peer-to-peer off: the setup of a stock driver on x4 cards,
which is what this layout is for).

| | TP4, peer-to-peer off (default) | TP4, peer-to-peer on (optional) | PP4, peer-to-peer off (`LAYOUT=pp4`) |
|---|---:|---:|---:|
| Streaming decode, 1 user, structured / code / prose | **267.3 / 260.9 / 186.0 tok/s** | **282.0 / 274.0 / 198.5 tok/s** | **141.8 / 139.8 / 103.5 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **758.9 / 674.1 / 531.1 tok/s** | **816.8 / 741.1 / 563.7 tok/s** | **603.0 / 577.2 / 445.1 tok/s** |
| Cold prefill | **2,657 tok/s** | **3,053 tok/s** | **6,575 tok/s** |
| One-token response time, 6,217 / 23,255-token prompt | 2.39 / 8.73 s | 2.05 / 7.52 s | 1.46 / 3.64 s |
| KV pool at 262,144 context | 1,176,646 tokens (4.49 full-length requests) | 1,177,646 tokens (4.49) | 2,334,498 tokens (8.91) |

All at 180 W per card (a power limit we set on our cards; the scripts never
change power, clock or fan settings), PCIe x16 links, one server start per
column. Decode uses three fixed prompt types—structured, code and prose—with
a 400-token cap, temperature 0 and median of five runs. Single-user tok/s
measures streaming decode after the first token. Eight-user aggregate is actual
completion tokens divided by concurrent batch wall time, including prefill and
client overhead; it is not the per-stream rate multiplied by eight.

Two of the 855 measured decode requests stopped at 298 tokens rather than the
400-token cap: one structured eight-user request under TP4/P2P-on and one under
PP4. Both began with instruction-analysis prose, misinterpreting the original
`(stream 4/8)` prompt suffix as splitting the counting task. They remain included
using their actual token counts, with no selective rerun or exclusion. These are
throughput observations, not evidence of output correctness or a quality comparison.

Cold prefill is the median of nine rates on the same three real-text prompts
of 23,945, 34,299 and 37,905 tokens, with zero observed prefix-cache hits.
The one-token response row retains the original nonstreaming elapsed-time
measurement (median of three); it includes response handling and is distinct
from streamed time to first token. The KV pool accounts for prefill chunks in
flight (see [Honest KV figures](#measured-while-building-this-release)).

For historical reference, PP4 with peer-to-peer on was measured in 1.4.0 at
141.0 / 139.4 / 105.1 tok/s for one user, 532.7 / 493.9 / 377.8 across eight,
6,606 tok/s cold prefill. That fourth configuration was not remeasured here.
Older release comparisons below retain their original scope.

Historical perplexity on the fixed 60-document set was **3.2781 under TP4**
and **3.2735 under PP4** (1.4.0); it was not remeasured for the allocator change.
Current HumanEval and GSM8K results are in [Quality (1.4.3)](#quality-143).

## Choosing a layout

Prefill and KV figures compare the native 1.6.0 peer-to-peer-off columns above.

| | Tensor-parallel 4 (`tp4`, default) | Pipeline-parallel 4 (`pp4`) |
|---|---|---|
| Each card holds | a quarter of every layer | a quarter of the layers |
| Best for | one or two interactive users: the fastest answer per request | many parallel users or clients, long prompts, large shared contexts |
| Prefill | 2,669 tok/s | 6,580 tok/s (2.47×) |
| KV pool | 1,072,150 tokens | 1,914,216 tokens (1.79×) |
| Traffic between cards | ~9.4 MB per layer during prefill, ~100 small collectives per decode step | activations only, once per stage |
| Links | PCIe x16 | built for x4; measured on x16 |

Both run the same model, the same drafter and the same repeatable-output fixes;
switching is `LAYOUT=pp4 ./start.sh restart` and back with `LAYOUT=tp4`.

## Measured while building this release

These are historical 1.4.0/1.4.1 measurements on the same four cards, not
remeasurements of the new allocator default. They are not the release table
above; each line says what it was measured against.

- **Pipeline-parallel prefill +19.5%.** New prefill kernels for 64 heads and
  whole experts — the linear-attention (KDA) prefill, the sparse-attention
  prefill and the MoE prefill — took PP4 cold prefill from 5,520 to 6,586–6,613
  tok/s, and time to first token on a 23,255-token prompt from 4.27 to 3.64 s,
  with decode unchanged. Each kernel is at least as accurate as the code it
  replaces against a 64-bit reference on real inputs.
- **A faster pipeline.** Decodes spread over every in-flight micro-batch and a
  leaner hand-off between stages: +7.3% aggregate decode at four users and
  +2.0% cold prefill under PP4, outputs identical apart from the drafter's
  folded input projection, which is checked against a 64-bit reference.
- **Tensor-parallel prefill kernels.** The linear-attention prefill runs 1.42×
  faster per prompt per card and the MoE prefill 1.24×, as accurate as before
  against a 64-bit reference.
- **Against release 1.3.0**, tensor-parallel 4: cold prefill 2,484 → 2,670
  tok/s (+7.5%) with peer-to-peer off and 2,490 → 3,062 (+23%) with it on,
  where NCCL now also runs card to card (worth +13.7% on its own, outputs
  identical).
  Eight-user aggregate 745.4 → 763.1 tok/s structured (+2.4%) off and
  808.1 → 848.3 (+5.0%) on; one user unchanged (264.6 → 264.4 structured).
- **Honest KV figures.** With prefill chunks in flight, a request can hold more
  linear-attention state than the engine used to reserve for it, so the KV pool
  it reported was larger than the server could actually fill: under TP4,
  release 1.3.x's figure was 17,932 tokens (1.5%) too high. Near a full pool
  that could only mean a request being paused and resumed, never wrong output,
  but the published number should be one you can use. The reserve now
  matches, and every KV figure on this page is the corrected one.
- **The same output on every install.** The linear-attention prefill kernels
  used to pick their tile configuration by timing the options in each new
  process, so a fresh install or a cleared cache could compute in a different
  order from ours. They now use pinned configurations, exactly as accurate as
  the tuned ones against a 64-bit reference (1.000×) and within 0.03% of their
  speed on one card.
- **Container at native speed.** Release 1.3.x's image ran within 0.4% of the
  native install on decode at 1 / 4 / 8 users and on cold prefill.
  On the same four cards at 180 W with peer-to-peer off, the image runs at the native install's speed: 15.60 / 29.16 / 43.99 ms per decode step at 1 / 4 / 8 users (native 15.87 / 29.51 / 44.45) and 2,672 tokens/s cold prefill (native 2,670).
- **A newer upstream.** The fork now sits on upstream vLLM's 0.30.1 development
  line (FlashInfer 0.7.0).

## Decode and prefill in detail

Release 1.6.0, native tensor-parallel 4 with peer-to-peer off and the
configuration described above: temperature 0, median of five, original
structured/code/prose prompts and cache-busting nonces.

**Decode**, 400 max tokens. `Stream` is per request,
`(completion_tokens − 1) / (end − first token)`; concurrent aggregate is
`sum(completion_tokens) / batch wall`, including prefill and client overhead.
At one user the original client's aggregate field duplicates its streaming
rate, so it is omitted here rather than presented as end-to-end throughput.
Each request carries a unique nonce to prevent prefix-cache reuse.

| Prompt type | Users | Stream tok/s | Aggregate tok/s | TTFT (cold) |
|---|---:|---:|---:|---:|
| Structured (count 1→200) | ×1 | **394.0** | — | 0.066 s |
|  | ×2 | **260.3** | **469.2** | 0.110 s |
|  | ×4 | **163.9** | **595.2** | 0.250 s |
|  | ×8 | **109.8** | **797.9** | 0.306 s |
| Code (clamp_00…clamp_49) | ×1 | **377.4** | — | 0.156 s |
|  | ×2 | **215.2** | **360.5** | 0.247 s |
|  | ×4 | **148.5** | **513.7** | 0.389 s |
|  | ×8 | **99.1** | **683.2** | 0.617 s |
| Prose (hash map) | ×1 | **180.5** | — | 0.068 s |
|  | ×2 | **149.4** | **276.0** | 0.160 s |
|  | ×4 | **115.3** | **418.4** | 0.254 s |
|  | ×8 | **75.2** | **544.7** | 0.312 s |

Prose decodes slower than structured or code text because the drafter's guesses
are accepted less often.

**Cold prefill by prompt length**, native release 1.6.0 pipeline-parallel 4
(peer-to-peer off), original real-text corpus and unique uncached windows,
`max_tokens=1`, median of two, `prompt tokens / streamed TTFT` measured client
side. All twelve requests recorded zero prefix-cache hits. (Tensor-parallel 4
runs at 2,669 tok/s on the separate 24K–38K prompts of the results table.)

| Prompt | TTFT | tok/s |
|---:|---:|---:|
| ~8k | 1.89 s | **4,215** |
| ~16k | 2.76 s | **5,789** |
| ~32k | 4.80 s | **6,645** |
| ~64k | 9.03 s | **7,097** |
| ~128k | 17.78 s | **7,176** |
| ~250k | 35.45 s | **7,061** |
