# Results

## Release 1.8.0

Measured 2026-10-16 on 4× CMP 170HX. Release 1.8.0 results use the
engine `819146023facb31c04f1f8512e52464835a10d40`, rebased on upstream vLLM
`4e0681ed56`, and its release defaults on 74 SMs per card, mainline
cmpunlocker plus the minimal P2P patch, verified P2P, PCIe x16 links,
262,144-token context and 180 W per card. P2P-off results are not published.
PP4 on x4 links is not measured.

RecoverSSM is enabled at TP4; PP4 keeps its existing state-storage path.
Compiled Marlin decode and eligible compiled prefill are included.
All-reduce flags and cached-boundary reuse default to `1`; PP4 drafter
width defaults to `0`.

**Quality**, fixed-order batches of up to eight. HumanEval allows 4,096
reply tokens; GSM8K allows 3,072. Length-capped outputs remain in the scores.
Scored on the answer after the reasoning (`content`), the same way for both
releases; 1.7.2 was re-scored that way from its stored replies, so its figures
here differ from the 160/164 and 1,282/1,319 published with 1.7.0.

| Layout | 1.7.2 HumanEval | 1.8.0 HumanEval | 1.7.2 GSM8K | 1.8.0 GSM8K |
|---|---:|---:|---:|---:|
| TP4 | 154/164 | 151/164 | 1,284/1,319 | 1,286/1,319 |

Neither suite shows a significant change (paired two-sided McNemar p = 0.45
for HumanEval, 0.75 for GSM8K). Quality was measured at TP4; PP4 runs the same
engine and checkpoints.

**KL divergence versus 1.7.2**, measured with `kld_probe`: short probes use
60 × 2,048 tokens; long probes use 8 × 16,384 tokens. Each cell reports
mean KL / top-1 agreement.

| Comparison | Short | Long |
|---|---:|---:|
| 1.8.0 TP4 vs 1.7.2 TP4 | 0.0158 / 95.4 % | 0.0083 / 97.0 % |
| 1.8.0 PP4 vs 1.7.2 PP4 | 0.0000 / 100.0 % | 0.00014 / 99.8 % |

At PP4, short-prompt output is identical to 1.7.2. Both TP4 checks are within
the release limits against 1.7.2 (mean KL 0.0228 short, 0.0124 long).

Cached and fresh runs of the same prompt can differ at near-ties at TP4
because prefix hits change the prefill chunk layout, as with batching.

## Release 1.8.0 throughput

DFlash2 with calibrated adaptive depth. Decode uses structured, code and prose
prompts under the MiaAI-Lab protocol: temperature 0, requests with thinking
off (GLM-5.3 still reasons, and its reasoning tokens count; see
[Talk to it](how-to-use.md#talk-to-it)) and a 400-token cap. One-user
throughput is the median of five streaming decode rates on one boot, measured
after the first token. Two-, four- and eight-user aggregate throughput is the
median of three runs on the same boot, using actual
completion tokens divided by concurrent batch wall time, including prefill and
client overhead.

| | TP4, peer-to-peer on | PP4, peer-to-peer on |
|---|---:|---:|
| Streaming decode, 1 user, structured / code / prose | **482.6 / 442.1 / 232.1 tok/s** | **232.4 / 209.9 / 100.1 tok/s** |
| Decode, 2 users, aggregate, structured / code / prose | **606.5 / 460.3 / 314.7 tok/s** | **343.8 / 273.4 / 195.5 tok/s** |
| Decode, 4 users, aggregate, structured / code / prose | **717.7 / 618.7 / 501.4 tok/s** | **409.2 / 382.8 / 314.2 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **943.7 / 835.0 / 667.2 tok/s** | **664.1 / 608.8 / 469.2 tok/s** |
| Cold prefill | **3,161 tok/s** | **7,419 tok/s** |
| KV pool at 262,144 context | 1,203,765 tokens (4.59 full-length requests) | 2,088,960 tokens (7.97 full-length requests) |

Each layout was measured on its own boot. Cold prefill is the median of nine
rates over the same three real-text prompts (23,945, 34,299 and 37,905
tokens), with no prefix-cache hits.

One-user prose varies most between runs (208.8–234.5 tok/s across the five
TP4 samples).

## Fork measurements (2026-11-10)

Same engine and image as 1.8.0 (`819146023f`, digest `sha256:18a2f262…`),
4× CMP 170HX at 200 W, PCIe Gen2 x16, verified P2P, TP4, DFlash2,
`MAX_SEQS=16`, `GPU_UTIL=0.95`. Decode uses the MiaAI-Lab script
(`tests/bench_decode.py`, unmodified apart from host and model name): 400
tokens, temperature 0, thinking off; one user is the median streaming decode
rate of five runs, N users the median aggregate rate of three. Measured with
an LMCache KV connector attached (it does not change decode). The 16-user row
is new. The base model is the release checkpoint; the LoRA rows add a rank-1
adapter on `o_proj`, `down_proj` and the experts.

| tok/s, structured / code / prose | 1.8.0 (Morrowmake) | Fork run, base model | Same, with a LoRA adapter |
|---|---:|---:|---:|
| 1 user | 482.6 / 442.1 / 232.1 | 530.0 / 408.5 / 212.1 | 431.2 / 329.5 / 191.4 |
| 2 users | 606.5 / 460.3 / 314.7 | 631.3 / 503.7 / 358.1 | 537.6 / 431.7 / 311.5 |
| 4 users | 717.7 / 618.7 / 501.4 | 818.0 / 685.3 / 548.6 | 680.4 / 567.6 / 443.0 |
| 8 users | 943.7 / 835.0 / 667.2 | 1,158.0 / 1,007.1 / 729.5 | 975.9 / 804.5 / 624.3 |
| 16 users | — | 1,200.9 / 1,020.6 / 777.4 | 1,124.1 / 939.4 / 731.9 |
| Cold prefill | 3,161 | 3,141 without LMCache, 2,565 with | 2,554 with LMCache |

The decode rows are within run-to-run variation of 1.8.0 or above it; the
engine and its settings are the same, so the difference is the host (power
limit, CPU, P2P path), not the fork's patches, which only change startup.
From 8 to 16 users the aggregate grows little (+4 % structured): decode is
compute-bound at that point. Cold prefill here uses three synthetic prompts of
~67k tokens with a fresh prefix each (the release's real-text prompts are not
published), median of nine; the LMCache connector costs ~19 % of cold prefill
because it writes every prefilled chunk to its CPU tier.

### Restart time

Measured on the same host with the same image and `serve.sh`, from
`systemctl restart` to `/health` 200, model already in the page cache:

| | Release behaviour | With the fork's start changes |
|---|---:|---:|
| TP4, base model, `MAX_SEQS=16` | 252 s | 147–155 s |

Breakdown of the saving: workers started together (~45 s), bytecode cache
(~8 s for each of API server, engine core and workers), startup plan
(engine init 63 s → 39 s; the KV pool is 7 % smaller on such starts). See [patches/README.md](../patches/README.md).

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
