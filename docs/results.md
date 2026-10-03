# Results

## Release 1.7.0

Measured 2026-10-03 on 4× CMP 170HX. Release 1.7.0 results use the final
engine `c1ce6491efe53934119d306d0a0501b475458e9b` and its release defaults
on 74 SMs per card, mainline cmpunlocker plus the minimal P2P patch, verified
P2P, PCIe x16 links, 262,144-token context and 180 W per card.
P2P-off results are no longer published. PP4 on x4 links is not measured.

RecoverSSM is enabled at TP4; PP4 keeps its existing state-storage path.
Compiled Marlin decode and eligible compiled prefill are included; native
installs must opt into building the optional library to use those paths.
All-reduce flags and cached-boundary reuse default to `1`; PP4 drafter
width defaults to `0`.

**Quality**, fixed-order batches of up to eight. HumanEval allows 4,096
reply tokens; GSM8K allows 3,072. Length-capped outputs remain in the scores.
The final release commit `c1ce6491ef` uses the same scorer and contract as
1.6.0's published quality, except that 1.7.0 requests carry the shipped
`tool_choice="none"` ban.

| Layout | Published 1.6.0 HumanEval | 1.7.0 HumanEval | Published 1.6.0 GSM8K | 1.7.0 GSM8K |
|---|---:|---:|---:|---:|
| TP4 | 160/164 | 160/164 | 1,280/1,319 | 1,282/1,319 |
| PP4 | 162/164 | 163/164 | 1,280/1,319 | 1,284/1,319 |

Neither layout shows a significant change (paired two-sided McNemar p ≥ 0.34).

**KL divergence versus 1.6.0**, measured with `kld_probe`: short probes use
60 × 2,048 tokens; long probes use 8 × 16,384 tokens. Each cell reports
mean KL / top-1 agreement.

| Comparison | Short | Long |
|---|---:|---:|
| 1.7.0 TP4 vs 1.6.0 TP4 | 0.0169 / 95.3 % | 0.0084 / 97.1 % |
| 1.7.0 PP4 vs 1.6.0 PP4 | 0.0156 / 95.4 % | 0.0086 / 97.0 % |
| Reference: 1.6.0 TP4 vs 1.6.0 PP4 | 0.0174 / 95.3 % | 0.0092 / 97.0 % |

The change in output distribution is no larger than the difference between
1.6.0's two layouts; perplexity is unchanged within 0.002.

Cached and fresh runs of the same prompt can differ at near-ties at TP4
because prefix hits change the prefill chunk layout, as with batching.

## Release 1.7.0 throughput

DFlash2 with calibrated adaptive depth. Decode uses structured, code and prose
prompts under the MiaAI-Lab protocol: temperature 0, thinking off and a
400-token cap. One-user throughput is the median of all ten streaming decode
rates from five runs on each of two boots, measured after the first token.
Two-, four- and eight-user aggregate throughput is the median of three runs
on one boot, using actual completion tokens divided by concurrent batch wall
time, including prefill and client overhead.

| | TP4, peer-to-peer on |
|---|---:|
| Streaming decode, 1 user, structured / code / prose | **482.2 / 447.7 / 210.6 tok/s** |
| Decode, 2 users, aggregate, structured / code / prose | **588.0 / 464.6 / 311.3 tok/s** |
| Decode, 4 users, aggregate, structured / code / prose | **691.2 / 584.9 / 485.8 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **948.6 / 834.1 / 684.6 tok/s** |
| Cold prefill | **3,197 tok/s** |
| KV pool at 262,144 context | 1,199,570 tokens (4.58 full-length requests) |

Container TP4 cold first start: 1,198,522 tokens, about 0.1 % fewer than native's 1,199,570; seeded starts match.

## Pipeline-parallel 4 throughput

| | PP4, peer-to-peer on |
|---|---:|
| Streaming decode, 1 user, structured / code / prose | **231.1 / 212.4 / 99.8 tok/s** |
| Decode, 2 users, aggregate, structured / code / prose | **331.0 / 277.9 / 189.8 tok/s** |
| Decode, 4 users, aggregate, structured / code / prose | **392.0 / 363.5 / 301.7 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **649.5 / 600.3 / 486.7 tok/s** |
| Cold prefill | **7,444 tok/s** |
| KV pool at 262,144 context | 1,914,216 tokens (7.30 full-length requests) |

Cold prefill is the median of nine rates over the same three real-text
prompts (23,945, 34,299 and 37,905 tokens), with no prefix-cache hits.

## Compared with 1.6.0

The published 1.6.0 figures below were measured on the earlier 70-SM driver.

| Metric | Published 1.6.0 | 1.7.0 change |
|---|---:|---:|
| TP4, 1 user, structured / code / prose (tok/s) | 437.4 / 404.2 / 198.6 | +10 % / +11 % / +6 % |
| TP4, 8 users, aggregate, structured / code / prose (tok/s) | 840.7 / 769.4 / 589.5 | +8 to +16 % |
| PP4, 1 user, structured / code / prose (tok/s) | 235.3 / 207.9 / 100.2 | flat |
| PP4, 8 users, aggregate, structured / code / prose (tok/s) | 623.1 / 591.0 / 465.8 | +2 to +5 % |
| TP4 cold prefill (tok/s) | 3,061 | +4 % |
| PP4 cold prefill (tok/s) | 6,580 | +13 % |
| TP4 KV pool (tokens) | 1,073,093 | +12 % |

## Choosing a layout

PP4 passes activations between stages; TP4 exchanges data after each layer.

| | Tensor-parallel 4 (`tp4`, default) | Pipeline-parallel 4 (`pp4`) |
|---|---|---|
| Each card holds | a quarter of every layer | a quarter of the layers |
| Best for | one or two interactive users: the fastest answer per request | many parallel users or clients, long prompts, large shared contexts |
| Traffic between cards | ~9.4 MB per layer during prefill, ~100 small collectives per decode step | activations only, once per stage |
| Links | PCIe x16 | built for x4; measured on x16 |

Both run the same model, the same drafter and the same repeatable-output fixes;
switching is `LAYOUT=pp4 ./start.sh restart` and back with `LAYOUT=tp4`.
