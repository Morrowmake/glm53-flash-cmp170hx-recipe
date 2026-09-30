# Results

## Release 1.6.0: faster decode for one or two users

Two engine changes, both on by default in both layouts:

- **Load-adaptive draft depth.** The DFlash2 drafter's guesses are verified
  five at a time when one request is running, four at two, and three (the
  previous fixed depth) under heavier load. More accepted tokens per step for a
  single user; unchanged behaviour under load. Kill switch
  `VLLM_GLM5_DFLASH_ADAPTIVE_K=0` ([engine switches](engine-switches.md)).
- **Compiled Marlin decode on by default, original reduction order.** The
  optional library's decode kernels now run in both layouts, in a faster order
  that sums the first MoE projection in four fixed K slices. Decoded text can
  differ from the released order; quality was checked per answer (below).
  `VLLM_GLM5_MARLIN_DECODE_VARIANT=exact` keeps the previous order
  ([Optional compiled Marlin](compiled-marlin.md)).

**The trade: KV pool.** Deeper drafts reserve more per-request scratch, so the
KV pool at 262,144 context is 1,125,277 tokens under TP4 (−4.4 % against
1.5.x) and 2,064,638 under PP4 (−11.6 %; about 2,000 tokens of that is the
compiled decode's scratch). Cold prefill is unchanged.

**Against 1.5.x, same protocol:** one user +27.0 / +22.2 / −0.5 %
(structured / code / prose) under TP4 and +31.6 / +20.5 / −5.3 % under PP4;
eight users +1.4 / +2.7 / +2.7 % (TP4) and +0.5 / +2.0 / +6.1 % (PP4).
Prose gains least: its drafts are accepted less often, so a deeper draft adds
step time without adding accepted tokens.

**Quality**, fixed-batch HumanEval (164) and GSM8K (1,319), compared per
answer with the previous release: TP4 **160/164 and 1,280/1,319** (previous
162 and 1,281; 3 losses and 1 gain on HumanEval, McNemar p 0.63), PP4
**162/164 and 1,280/1,319** (previous 163 and 1,284; p 1.0 and 0.22). Individual
answers flip in both directions when decoded text changes; no suite shows a
significant net loss.

**Long-context and repeatability checks.** The full tier-2 long-context and
repeatability checks ran on the same engine build: needle retrieval 30/30 up
to 262K tokens, copy fidelity no worse than the previous baseline, no
cross-request leaks, and bit-identical outputs for the same batch run twice in
eager and CUDA-graph modes, in both layouts. The README numbers and the needle
and repeatability checks were re-measured on the final candidate.

## Release 1.6.0 throughput

Measured on 2026-10-01 with the released engine and launcher defaults, the
same workloads and client as the 1.5.0 table below: DFlash2 with load-adaptive
depth, 262,144-token context, 180 W per card, PCIe x16 links, one server start
per column, decode median of five runs.

| | TP4, peer-to-peer off (default) | TP4, peer-to-peer on (optional) | PP4, peer-to-peer off (`LAYOUT=pp4`) |
|---|---:|---:|---:|
| Streaming decode, 1 user, structured / code / prose | **339.4 / 318.7 / 185.0 tok/s** | **372.6 / 350.3 / 199.4 tok/s** | **186.6 / 168.5 / 98.0 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **769.7 / 692.1 / 545.7 tok/s** | **821.2 / 749.2 / 591.9 tok/s** | **606.3 / 588.5 / 472.2 tok/s** |
| Cold prefill | **2,671 tok/s** | **3,056 tok/s** | **6,586 tok/s** |
| One-token response time, 6,217 / 23,255-token prompt | 2.38 / 8.68 s | 2.06 / 7.49 s | 1.46 / 3.64 s |
| KV pool at 262,144 context | 1,125,277 tokens (4.29 full-length requests) | 1,126,248 tokens (4.30) | 2,064,638 tokens (7.88) |

Of the 855 measured decode requests, one (TP4 peer-to-peer off, structured,
four users) stopped at 340 tokens; all others ran to the 400-token cap.

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
| Prefill | 2,671 tok/s | 6,586 tok/s (2.47×) |
| KV pool | 1,125,277 tokens | 2,064,638 tokens (1.83×) |
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
| Structured (count 1→200) | ×1 | **339.4** | — | 0.063 s |
|  | ×2 | **239.0** | **436.3** | 0.111 s |
|  | ×4 | **156.8** | **565.0** | 0.256 s |
|  | ×8 | **106.1** | **769.7** | 0.315 s |
| Code (clamp_00…clamp_49) | ×1 | **318.7** | — | 0.158 s |
|  | ×2 | **192.7** | **339.5** | 0.250 s |
|  | ×4 | **141.4** | **495.9** | 0.393 s |
|  | ×8 | **101.1** | **692.1** | 0.618 s |
| Prose (hash map) | ×1 | **185.0** | — | 0.067 s |
|  | ×2 | **153.4** | **276.8** | 0.161 s |
|  | ×4 | **112.4** | **416.0** | 0.255 s |
|  | ×8 | **74.2** | **545.7** | 0.316 s |

Prose decodes slower than structured or code text because the drafter's guesses
are accepted less often.

**Cold prefill by prompt length**, native release 1.6.0 pipeline-parallel 4
(peer-to-peer off), original real-text corpus and unique uncached windows,
`max_tokens=1`, median of two, `prompt tokens / streamed TTFT` measured client
side. All twelve requests recorded zero prefix-cache hits. (Tensor-parallel 4
runs at 2,671 tok/s on the separate 24K–38K prompts of the results table.)

| Prompt | TTFT | tok/s |
|---:|---:|---:|
| ~8k | 1.90 s | **4,197** |
| ~16k | 2.77 s | **5,762** |
| ~32k | 4.83 s | **6,604** |
| ~64k | 9.06 s | **7,068** |
| ~128k | 17.79 s | **7,172** |
| ~250k | 35.41 s | **7,067** |
