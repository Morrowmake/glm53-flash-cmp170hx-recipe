# Results

## Release 1.7.0

All results below belong to the final engine `caaf6afe8ee8c29b32ef3b77284656ca87a0937e` and its release defaults
on 74 SMs per card, mainline cmpunlocker plus the minimal P2P patch, verified
P2P, PCIe x16 links, 262,144-token context and 180 W per card.
P2P-off results are no longer published. PP4 on x4 links is not measured.

RecoverSSM increases TP4 KV by {{NUM:tp4_recover_kv_gain_pct}}%.
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

**Long context and repeatability.** Needle retrieval TP4
{{NUM:tp4_needle_passed}}/{{NUM:tp4_needle_total}}, PP4
{{NUM:pp4_needle_passed}}/{{NUM:pp4_needle_total}} up to 262K tokens.
Single-request repeatability TP4 {{NUM:tp4_repeatability_passed}}/
{{NUM:tp4_repeatability_total}}, PP4 {{NUM:pp4_repeatability_passed}}/
{{NUM:pp4_repeatability_total}}. Same-batch repeatability mismatches in eager /
graph mode: TP4 {{NUM:tp4_repeat_eager_mismatches}} /
{{NUM:tp4_repeat_graph_mismatches}}; PP4 {{NUM:pp4_repeat_eager_mismatches}} /
{{NUM:pp4_repeat_graph_mismatches}}. Copy-fidelity mismatches TP4/PP4:
{{NUM:tp4_copy_mismatches}} / {{NUM:pp4_copy_mismatches}};
cross-request leak failures: {{NUM:leak_failures}}; new Xid events:
{{NUM:validation_xid_delta}}.

Cached and fresh runs of the same prompt can differ at near-ties at TP4
because prefix hits change the prefill chunk layout, as with batching.
This is separate from repeating the same request with the same cache and
batch conditions. Cached-boundary checks passed {{NUM:boundary_passed}}/
{{NUM:boundary_total}} cases. Resumed-prefill stress passed
{{NUM:resume_stress_passed}}/{{NUM:resume_stress_total}} cases;
state-index check failures: {{NUM:state_index_failures}}.

## Release 1.7.0 throughput

DFlash2 with calibrated adaptive depth. Decode uses structured, code and prose
prompts, temperature 0, a 400-token cap and the median of five runs.
One-user throughput measures streaming decode after the first token. Eight-user
aggregate uses actual completion tokens divided by concurrent batch wall time,
including prefill and client overhead.

| | TP4, peer-to-peer on |
|---|---:|
| Streaming decode, 1 user, structured / code / prose | **{{NUM:tp4_c1_structured_tps}} / {{NUM:tp4_c1_code_tps}} / {{NUM:tp4_c1_prose_tps}} tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **{{NUM:tp4_c8_structured_tps}} / {{NUM:tp4_c8_code_tps}} / {{NUM:tp4_c8_prose_tps}} tok/s** |
| Cold prefill | **{{NUM:tp4_prefill_tps}} tok/s** |
| One-token response time, 6,217 / 23,255-token prompt | {{NUM:tp4_response_6217_s}} / {{NUM:tp4_response_23255_s}} s |
| KV pool at 262,144 context | {{NUM:tp4_kv_tokens}} tokens ({{NUM:tp4_kv_context_ratio}}) |

## Pipeline-parallel 4 throughput

| | PP4, peer-to-peer on |
|---|---:|
| Streaming decode, 1 user, structured / code / prose | **{{NUM:pp4_c1_structured_tps}} / {{NUM:pp4_c1_code_tps}} / {{NUM:pp4_c1_prose_tps}} tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **{{NUM:pp4_c8_structured_tps}} / {{NUM:pp4_c8_code_tps}} / {{NUM:pp4_c8_prose_tps}} tok/s** |
| Cold prefill | **{{NUM:pp4_prefill_tps}} tok/s** |
| One-token response time, 6,217 / 23,255-token prompt | {{NUM:pp4_response_6217_s}} / {{NUM:pp4_response_23255_s}} s |
| KV pool at 262,144 context | {{NUM:pp4_kv_tokens}} tokens ({{NUM:pp4_kv_context_ratio}}) |

Cold prefill is the median of nine rates over the same three real-text
prompts (23,945, 34,299 and 37,905 tokens), with no prefix-cache hits.
The one-token response row is nonstreaming elapsed time (median of three),
including response handling. Early-stopped decode requests / all measured
requests: {{NUM:decode_early_stopped}} / {{NUM:decode_requests}}; their actual
token counts remain included without selective reruns.

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
