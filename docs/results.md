# Results

## Release 1.7.0

Measured 2026-10-03 on 4× CMP 170HX. Release 1.7.0 results use the final
engine `caaf6afe8ee8c29b32ef3b77284656ca87a0937e` and its release defaults
on 74 SMs per card, mainline cmpunlocker plus the minimal P2P patch, verified
P2P, PCIe x16 links, 262,144-token context and 180 W per card.
P2P-off results are no longer published. PP4 on x4 links is not measured.

RecoverSSM is enabled at TP4; PP4 keeps its existing state-storage path.
Compiled Marlin decode and eligible compiled prefill are included; native
installs must opt into building the optional library to use those paths.
All-reduce flags and cached-boundary reuse default to `1`; PP4 drafter
width defaults to `0`.

**Quality**, fixed-order batches of up to eight, scored per answer against the
baseline on the same driver and SM count. HumanEval allows 4,096 reply tokens;
GSM8K allows 3,072. Length-capped outputs remain in the scores.
TP4: HumanEval {{NUM:tp4_humaneval_passed}}/164, GSM8K
{{NUM:tp4_gsm8k_passed}}/1,319 ({{NUM:tp4_gsm8k_pct}}%). PP4:
{{NUM:pp4_humaneval_passed}}/164 and {{NUM:pp4_gsm8k_passed}}/1,319
({{NUM:pp4_gsm8k_pct}}%). Net gains in percentage points, TP4 HumanEval/GSM8K:
{{NUM:tp4_humaneval_net_pp}} / {{NUM:tp4_gsm8k_net_pp}}; PP4:
{{NUM:pp4_humaneval_net_pp}} / {{NUM:pp4_gsm8k_net_pp}}.
Paired two-sided McNemar p values, TP4 HumanEval/GSM8K:
{{NUM:tp4_humaneval_p}} / {{NUM:tp4_gsm8k_p}}; PP4:
{{NUM:pp4_humaneval_p}} / {{NUM:pp4_gsm8k_p}}.

**KL divergence:** TP4 {{NUM:tp4_kl_divergence}}, PP4 {{NUM:pp4_kl_divergence}}.

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
