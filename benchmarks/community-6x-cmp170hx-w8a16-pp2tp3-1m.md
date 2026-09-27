# Community result: 6x CMP 170HX, W8A16, PP2 x TP3, 1M context

This is a field report from a Windows/WSL2 host running a six-card adaptation
of this recipe. It is not a release benchmark and should not be read as an
apples-to-apples replacement for the four-card W4A16 tables in the main
README. The checkpoint, quantization, engine revision, topology and context
length all differ.

The main reason for publishing it is to document the observed gap and expose
the configuration behind it. Using the same public prompt constants, sampling,
400-token limit and decode-rate formula, this system reaches
**177.0 / 147.0 / 123.2 tok/s** for structured text, code and prose. The
approximately 190 tok/s reference is the project's TP4 W4A16 prose result;
this PP2 x TP3 W8A16 system reaches 123.2 tok/s on that prompt.

## Hardware and host

Measured on 2026-09-27.

| Component | Configuration |
|---|---|
| GPUs used by GLM | 6x NVIDIA CMP 170HX 64G, sm_80, 384 GiB total HBM2e |
| PCIe links | Gen2 x16 on all six CMP cards |
| Other GPU | RTX 3090 24G present but excluded from `CUDA_VISIBLE_DEVICES` |
| CPU | AMD Ryzen Threadripper PRO 3995WX, 64 cores / 128 threads |
| Memory | 512 GiB, 8 DIMMs |
| Mainboard | ASUS Pro WS WRX80E-SAGE SE WIFI |
| Host OS | Windows 11 Pro for Workstations, build 26200 |
| Linux environment | WSL 2.6.3, Ubuntu 24.04.4, kernel 6.6.87.2 |
| NVIDIA driver | 616.92 |
| Docker Engine | 29.1.3 |

The current `nvidia-smi` power limit is 250 W per CMP card. The strict speed
receipt did not record power, so this report does not claim a benchmark power
cap. No clock or fan changes were made for these runs. The project's published
tables were measured at 180 W per card, so this is not a controlled
watt-for-watt comparison.

With the model idle and loaded, observed memory use was 62.7--62.9 GiB on each
CMP card. GPU identifiers, PCI bus addresses, hostnames, local paths and
container IDs are deliberately omitted.

## Model and serving configuration

| Setting | Value |
|---|---|
| Served model | `GLM-5.3-Flash-UNCENSORED` |
| Target checkpoint | `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` at `d21b19569d30e6f471c433b11e672b3bbb80552a` |
| Architecture | Approximately 320B-parameter MoE; 45 layers, 288 routed experts, top-8 routing |
| Target size | 62 shards, 328,337,456,936 weight bytes |
| Quantization | W8A16: block FP8 E4M3 weights through FP8 Marlin, BF16 activations, 128x128 weight blocks; checkpoint metadata uses `activation_scheme=dynamic` |
| Draft checkpoint | `incoai/GLM-5.3-Flash-DFlash2` at `bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| Speculative decoding | DFlash2, 3 draft tokens per step, selector top-k 16 |
| Parallel layout | pipeline parallel 2 x tensor parallel 3 |
| Layer partition | 24 / 21 target layers |
| TP head geometry | 64 checkpoint heads padded to 66 for TP3, 22 local heads per rank |
| Context | 1,048,576 tokens |
| Scheduler | `max_num_seqs=4`, `max_num_batched_tokens=1024` |
| Fair prefill | 128-token cap while another request decodes; one partial prefill |
| Memory fraction | `gpu_memory_utilization=0.945` |
| Prefix cache | enabled |
| CUDA graphs | `FULL_DECODE_ONLY`, capture sizes 4, 8 and 16 |
| Multimodal limit | up to 32 images, video disabled |
| Interconnect path | NCCL P2P disabled; SHM and the host-staged small all-reduce path enabled |
| KV pool at startup | 1,160,502 tokens, approximately 1.11 full 1M requests |

The target has hidden size 4096 and MoE intermediate size 2048. Local overlays
enable the uneven TP3 path, H22/M4 decode shapes, host-staged all-reduce and
the retained exact-greedy fast path.

## Engine provenance

The running image reports:

```text
vLLM 0.29.1rc1.dev619+g37e4d5814.d20260925.precompiled
```

The source tree starts from this project's older engine commit
`0ed7d3e7f3f855646a139701598a9b40d5745688`, followed by six local commits:

```text
3f3a0fe9d Enable GLM5Next auxiliary-state relay over pipeline parallelism
37e4d5814 Load target embeddings for GLM speculative PP
a38015e48 Fix GLM FP8 loading on pipeline stages
9c00ded14 Filter GLM fastsafetensors loads by PP stage
be5182ac6 Reduce GLM FP8 Marlin startup memory
9119ffb98 Support DFlash KV overflow lanes under PP
```

This branch is based on the recipe's earlier engine pin `0ed7d3e7f3` and does
not include recipe 1.4.1's `378c37b009` changes. The two heads are divergent,
so no performance claim from the 1.4.1 pin is attributed to this result.

## Prompt-matched uncached decode

This section uses the public prompt constants from
[MiaAI-Lab's benchmark repository](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks),
which are also the prompt types identified in this project's README:

- structured: count from 1 to 200;
- code: implement `clamp_range`;
- prose: explain hash maps, collisions, resizing and complexity.

The driver uses MiaAI-Lab harness revision
`f4970207e9fb2bdeac40d88b7cbef18c98aea310`, with two request-side changes to
follow this project's documented behavior: it removes
`chat_template_kwargs.enable_thinking=false` and appends a unique UUID nonce
to every prompt. Model-native reasoning is therefore streamed and each
measured request misses the prefix cache; the API reported `cached_tokens=0`
for all 15 runs.

Protocol: one stream, 400 maximum completion tokens, temperature 0, top-p 1,
five measured runs per prompt. Structured and code each had one excluded
32-token warm-up request; prose did not. Decode rate is
`(completion_tokens - 1) / (last_token_time - first_token_time)`. All measured
runs returned HTTP 200, generated 400 tokens, finished by the length cap and
contained no NaN marker. This is a throughput receipt; the outputs were not
scored for instruction-following correctness.

| Prompt | Median tok/s | Range tok/s | Uncached TTFT median | Draft accept ratio | Accepted drafts/step |
|---|---:|---:|---:|---:|---:|
| Structured | **177.04** | 166.48--181.38 | 0.319 s | 0.9519 | 2.856 |
| Code | **146.97** | 142.79--150.78 | 0.321 s | 0.7419 | 2.226 |
| Prose | **123.22** | 119.41--127.46 | 0.316 s | 0.5593 | 1.678 |

For orientation only, recipe 1.4.1 reports the following values. Those numbers
were not remeasured on this host.

| Prompt | This host, W8 PP2xTP3 | README TP4, P2P off | README PP4, P2P off |
|---|---:|---:|---:|
| Structured | 177.0 | 264.4 | 141.7 |
| Code | 147.0 | 258.5 | 138.4 |
| Prose | 123.2 | 189.9 | 100.3 |

The observed rates exceed the published PP4 values but remain materially below
TP4. This table does not isolate the cause: W8 versus W4 weight traffic, TP3
collectives, the H22 kernel path, divergent engine branches, the different
checkpoint, the 1M-context memory plan and the uncontrolled power cap all
change at once.

## Long max-reasoning workload

A second suite describes the interactive workload used on this host. It is
included because these are the 105--118 tok/s numbers visible in normal use,
but it is **not comparable** to the 400-token native-reasoning table above.

Protocol: Python implementation, Chinese technical writing and React/CSS
frontend prompts; three runs each; one stream; 2,048 output tokens;
`reasoning_effort=max`; temperature 0; top-p 1; top-k -1; no penalties; unique
nonce and seed per run. All nine runs reached the 2,048-token length cap while
still in the separate reasoning field, so these are reasoning-token decode
rates rather than completed-answer rates.

| Workload | Median decode tok/s | Median end-to-end tok/s | Median TTFT | Median inferred speculative round |
|---|---:|---:|---:|---:|
| Python | **114.01** | 112.25 | 0.316 s | 22.088 ms |
| Chinese technical writing | **104.97** | 103.40 | 0.339 s | 21.826 ms |
| Frontend | **117.68** | 115.41 | 0.345 s | 22.018 ms |
| All nine runs | **112.30** | 110.44 | 0.339 s | - |

The inferred speculative round is decode time divided by the speculative-draft
counter delta; it is a serving-cycle ratio, not a kernel measurement or
per-output-token latency.

The highest recorded 2,048-token run was 130.00 tok/s on one frontend sample;
it is a peak, not the retained headline. A bracketed A1/A2 rerun of the same
server produced 112.298 and 112.280 tok/s overall medians. All 9/9 token-ID
sequences matched across those two server epochs, with 0.189% median inferred
speculative-round timing drift.

Raw SSE events and model reasoning are not attached because they contain full
generated traces and local runtime identifiers. Integrity hashes retained by
the operator are:

```text
structured.json aa3177c2a40b0a6924a0ffbdfb7a3393e1ddc9ef023e8e715787f413968c2543
coding.json     48d80c1bfc76aac656febe83fc80d661c1cfcf22ad9c9de97b32be795ab6a35f
prose.json      94aa2e0aa303f18c72f03ca43dccc343ea0ad3fb981f0fc65a9808a64bc08799
A1.json      c625a1974c2183a08d7d499e8441918cd036de0bef02ad70b4953c14d8ec234b
A2.json      afbf0809945f27331d533b911bb6abbb087cb9b855a6e3391d7559a561155022
analysis.json 9c2fdf89a8691fd0aef8c4c7290b47a43ce34fb3074fbe76d269ea4f55a56651
```

## Questions this result leaves open

1. Which 1.4.1 decode changes are expected to apply to local-head geometry
   H22, rather than the H16/H32 paths used by the four-card layouts?
2. Is W8 weight bandwidth plus TP3 communication enough to explain the
   remaining TP4 gap, or is an H22 fallback visible in this configuration?
3. Is there a preferred upstream benchmark receipt format for a future strict
   A/B of the current engine pin versus `378c37b009` on this six-card host?
