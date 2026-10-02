# Release 1.7.0 fill-in list

Temporary release preparation file; delete before release. Values remain
unresolved until the named decision or measurement is complete. Do not reuse
numbers from another driver, SM count or engine tree. The preparation checkout
is not runnable as a release while slots remain.

Fill identical tokens with identical values throughout the checkout. Use
`tools/set-pin.sh <40-hex commit>` for the engine pin. Image digest comes from
the final built artifact, not the engine hash. Preserve earlier changelog
provenance. Step 5 is calibration and deciding A/Bs; step 6 is grouped validation;
image acceptance covers full fresh/update installs in both layouts.

90 distinct release slots, 177 release occurrences outside this inventory. Runtime template expressions are listed separately by purpose below and are not fill-in slots.

| Placeholder | Fills from | Locations |
|---|---|---|
| `{{$k}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `start.sh:448` |
| `{{.Id}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `start.sh:937` |
| `{{.State.Running}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `start.sh:833` |
| `{{DEFAULT:all_reduce_flags}}` | Step 5 decision: TP4 flags-in-data all-reduce A/B; fill 0 or 1. | `CHANGELOG.md:18`, `docs/engine-switches.md:83`, `docs/how-it-works.md:71`, `docs/how-to-use.md:202`, `docs/how-to-use.md:295`, `docs/results.md:13`, `serve.sh:246` |
| `{{DEFAULT:cached_boundary}}` | Step 5 decision: cached-boundary A/B, then step 6 boundary checks; fill 0 or 1. | `CHANGELOG.md:21`, `docs/engine-switches.md:85`, `docs/how-it-works.md:74`, `docs/how-to-use.md:204`, `docs/results.md:15`, `serve.sh:551` |
| `{{DEFAULT:depth2}}` | Step 5 decision: depth-2 policy A/B with matching cost tables, then step 6 grouped validation; fill 0 or 1. | `CHANGELOG.md:23`, `docs/engine-switches.md:87`, `serve.sh:554` |
| `{{DEFAULT:draft_skip}}` | Step 5 decision: TP4 confidence-skip A/B; fill 0 or 1. PP4 stays off because the current implementation requires TP-only. | `CHANGELOG.md:20`, `docs/engine-switches.md:84`, `docs/how-it-works.md:73`, `docs/how-to-use.md:203`, `docs/results.md:14`, `serve.sh:545` |
| `{{DEFAULT:kda_v2_deep}}` | Step 5 decision: deep KDA v2 A/B, then step 6 grouped validation; fill 0 or 1. | `CHANGELOG.md:24`, `docs/engine-switches.md:97`, `serve.sh:240` |
| `{{DEFAULT:pp4_drafter_width}}` | Step 5 decision: PP4 drafter-width A/B after calibration; fill 0 or 1. | `CHANGELOG.md:22`, `docs/engine-switches.md:82`, `docs/how-it-works.md:75`, `docs/how-to-use.md:205`, `docs/results.md:16`, `serve.sh:538` |
| `{{DEFAULT:pp4_partition}}` | Step 5 decision: PP4 partition A/B before calibration; fill the selected comma-separated layer counts. | `docs/engine-switches.md:88`, `serve.sh:165` |
| `{{NCCL_P2P_DISABLE-unset}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `docker/tests/test_p2p_check.py:183`, `docker/tests/test_p2p_check.py:256` |
| `{{NCCL_P2P_LEVEL-unset}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `docker/tests/test_p2p_check.py:183`, `docker/tests/test_p2p_check.py:256` |
| `{{NUM:boundary_passed}}` | Step 6 validation: cached-boundary validation, passed case count. | `docs/results.md:47` |
| `{{NUM:boundary_total}}` | Step 6 validation: cached-boundary validation, total case count. | `docs/results.md:48` |
| `{{NUM:decode_early_stopped}}` | Step 6 validation: README throughput leg, number of requests ending below the cap. | `docs/results.md:82` |
| `{{NUM:decode_requests}}` | Step 6 validation: README throughput leg, total measured decode requests. | `docs/results.md:82` |
| `{{NUM:image_compressed_gb}}` | Image build: compressed bytes of the exact final distributed image, converted to decimal GB. | `README.md:57`, `docker/RELEASE.md:80`, `docs/how-to-use.md:11`, `docs/how-to-use.md:19` |
| `{{NUM:image_digest}}` | Image build + full image acceptance: exact final OCI manifest SHA-256, 64 hex characters without the sha256 prefix. Includes the selected seed layer when shipped. | `.env.advanced.example:34`, `CHANGELOG.md:5`, `docker-compose.yml:21`, `docker/README.md:39`, `docker/README.md:141`, `docker/RELEASE.md:82`, `docs/how-it-works.md:92`, `start.sh:157` |
| `{{NUM:image_layer_count}}` | Image build: final filesystem layer count including base and seed; must pass the layer-count gate. | `docker/RELEASE.md:81` |
| `{{NUM:leak_failures}}` | Step 6 validation: cross-request leak leg, failure count. | `docs/results.md:41` |
| `{{NUM:pp4_adaptive_costs_multi}}` | Step 5 calibration on the chosen PP4 partition and frozen tree: complete comma-separated relative cost table in the parser order matching the selected depth-2 setting, multi-request. | `serve.sh:560` |
| `{{NUM:pp4_adaptive_costs}}` | Step 5 calibration on the chosen PP4 partition and frozen tree: complete comma-separated relative cost table in the parser order matching the selected depth-2 setting, single-request. | `serve.sh:559` |
| `{{NUM:pp4_boot_cold_s}}` | Image build acceptance: PP4 cold/seeded health + boot-check startup elapsed seconds on the exact image; retain cache identity and smoke evidence. | `CHANGELOG.md:33`, `docker/RELEASE.md:79`, `docs/how-to-use.md:32` |
| `{{NUM:pp4_boot_seeded_s}}` | Image build acceptance: PP4 cold/seeded health + boot-check startup elapsed seconds on the exact image; retain cache identity and smoke evidence. | `CHANGELOG.md:34`, `docker/RELEASE.md:79`, `docs/how-to-use.md:33` |
| `{{NUM:pp4_c1_code_tps}}` | Step 6 README throughput leg: PP4 one-user streaming decode, code prompt, tok/s; median of five. | `docs/results.md:72` |
| `{{NUM:pp4_c1_prose_tps}}` | Step 6 README throughput leg: PP4 one-user streaming decode, prose prompt, tok/s; median of five. | `docs/results.md:72` |
| `{{NUM:pp4_c1_structured_tps}}` | Step 6 README throughput leg: PP4 one-user streaming decode, structured prompt, tok/s; median of five. | `docs/results.md:72` |
| `{{NUM:pp4_c8_code_tps}}` | Step 6 README throughput leg: PP4 eight-user aggregate completion rate, code prompt, tok/s; median of five. | `docs/results.md:73` |
| `{{NUM:pp4_c8_prose_tps}}` | Step 6 README throughput leg: PP4 eight-user aggregate completion rate, prose prompt, tok/s; median of five. | `docs/results.md:73` |
| `{{NUM:pp4_c8_structured_tps}}` | Step 6 README throughput leg: PP4 eight-user aggregate completion rate, structured prompt, tok/s; median of five. | `docs/results.md:73` |
| `{{NUM:pp4_copy_mismatches}}` | Step 6 long-context copy-fidelity leg: PP4 mismatch count against its baseline. | `docs/results.md:40` |
| `{{NUM:pp4_gsm8k_net_pp}}` | Step 6 paired quality leg: PP4 GSM8K net gain in percentage points versus the same-driver baseline. | `docs/results.md:26` |
| `{{NUM:pp4_gsm8k_passed}}` | Step 6 paired quality leg: PP4 GSM8K count of passed tasks; retain capped outputs. | `README.md:80`, `docs/results.md:23` |
| `{{NUM:pp4_gsm8k_pct}}` | Step 6 paired quality leg: PP4 GSM8K percent correct on all 1,319 problems. | `README.md:80`, `docs/results.md:24` |
| `{{NUM:pp4_gsm8k_p}}` | Step 6 paired quality leg: PP4 GSM8K two-sided paired McNemar p value. | `docs/results.md:29` |
| `{{NUM:pp4_humaneval_net_pp}}` | Step 6 paired quality leg: PP4 HumanEval net gain in percentage points versus the same-driver baseline. | `docs/results.md:26` |
| `{{NUM:pp4_humaneval_passed}}` | Step 6 paired quality leg: PP4 HumanEval count of passed tasks; retain capped outputs. | `README.md:79`, `docs/results.md:23` |
| `{{NUM:pp4_humaneval_p}}` | Step 6 paired quality leg: PP4 HumanEval two-sided paired McNemar p value. | `docs/results.md:29` |
| `{{NUM:pp4_kv_context_ratio}}` | Step 6 validation boot: measured PP4 KV pool divided by 262,144. | `docs/results.md:76` |
| `{{NUM:pp4_kv_tokens}}` | Step 6 validation boot: PP4 reported usable KV tokens at 262,144 context with final release defaults. | `docs/how-it-works.md:95`, `docs/results.md:76` |
| `{{NUM:pp4_needle_passed}}` | Step 6 long-context needle leg: PP4 passed/total retrieval cases through the context ceiling. | `docs/results.md:33` |
| `{{NUM:pp4_needle_total}}` | Step 6 long-context needle leg: PP4 passed/total retrieval cases through the context ceiling. | `docs/results.md:33` |
| `{{NUM:pp4_prefill_tps}}` | Step 6 README throughput leg: PP4 cold prefill, median of nine real-text rates, no cache hits, tok/s. | `docs/results.md:74` |
| `{{NUM:pp4_repeat_eager_mismatches}}` | Step 6 repeatability leg: PP4 mismatch count in the named eager/graph mode. | `docs/results.md:38` |
| `{{NUM:pp4_repeat_graph_mismatches}}` | Step 6 repeatability leg: PP4 mismatch count in the named eager/graph mode. | `docs/results.md:39` |
| `{{NUM:pp4_repeatability_passed}}` | Step 6 repeatability leg: PP4 passed/total single-request repeatability checks, matching cache and batch conditions. | `docs/results.md:35` |
| `{{NUM:pp4_repeatability_total}}` | Step 6 repeatability leg: PP4 passed/total single-request repeatability checks, matching cache and batch conditions. | `docs/results.md:36` |
| `{{NUM:pp4_response_23255_s}}` | Step 6 README throughput leg: PP4 nonstreaming one-token elapsed seconds at 23255 prompt tokens, median of three. | `docs/results.md:75` |
| `{{NUM:pp4_response_6217_s}}` | Step 6 README throughput leg: PP4 nonstreaming one-token elapsed seconds at 6217 prompt tokens, median of three. | `docs/results.md:75` |
| `{{NUM:pp4_vs_tp4_kv_ratio}}` | Step 6 throughput/KV validation: ratio of the PP4 and TP4 measured KV token pools. | `README.md:117`, `docs/how-it-works.md:20` |
| `{{NUM:pp4_vs_tp4_prefill_ratio}}` | Step 6 throughput/KV validation: ratio of the PP4 and TP4 measured cold-prefill rates. | `README.md:116` |
| `{{NUM:resume_stress_passed}}` | Step 6 validation: resumed-prefill stress, passed case count across both layouts. | `docs/results.md:49` |
| `{{NUM:resume_stress_total}}` | Step 6 validation: resumed-prefill stress, total case count across both layouts. | `docs/results.md:49` |
| `{{NUM:skip_intercept_1}}` | Step 5 confidence-skip A/B: selected frozen estimator intercept_1 from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:6` |
| `{{NUM:skip_intercept_2}}` | Step 5 confidence-skip A/B: selected frozen estimator intercept_2 from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:7` |
| `{{NUM:skip_intercept_3}}` | Step 5 confidence-skip A/B: selected frozen estimator intercept_3 from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:8` |
| `{{NUM:skip_intercept_4}}` | Step 5 confidence-skip A/B: selected frozen estimator intercept_4 from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:9` |
| `{{NUM:skip_intercept_5}}` | Step 5 confidence-skip A/B: selected frozen estimator intercept_5 from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:10` |
| `{{NUM:skip_intercept_6}}` | Step 5 confidence-skip A/B: selected frozen estimator intercept_6 from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:11` |
| `{{NUM:skip_intercept_7}}` | Step 5 confidence-skip A/B: selected frozen estimator intercept_7 from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:12` |
| `{{NUM:skip_slope}}` | Step 5 confidence-skip A/B: selected frozen estimator slope from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:4` |
| `{{NUM:skip_threshold}}` | Step 5 confidence-skip A/B: selected frozen estimator threshold from the calibrated version-1 coefficient file; retain the validated 7-position order. If skip ships off, keep the selected calibrated file or remove its unused mount and override together. | `draft-skip-coefficients.json:14` |
| `{{NUM:state_index_failures}}` | Step 6 validation: state-index-check stress leg, failure count. | `docs/results.md:50` |
| `{{NUM:tp4_adaptive_costs_multi}}` | Step 5 calibration on the chosen TP4 partition and frozen tree: complete comma-separated relative cost table in the parser order matching the selected depth-2 setting, multi-request. | `serve.sh:560` |
| `{{NUM:tp4_adaptive_costs}}` | Step 5 calibration on the chosen TP4 partition and frozen tree: complete comma-separated relative cost table in the parser order matching the selected depth-2 setting, single-request. | `serve.sh:559` |
| `{{NUM:tp4_boot_cold_s}}` | Image build acceptance: TP4 cold/seeded health + boot-check startup elapsed seconds on the exact image; retain cache identity and smoke evidence. | `CHANGELOG.md:32`, `README.md:98`, `docker/RELEASE.md:78`, `docs/how-to-use.md:31` |
| `{{NUM:tp4_boot_seeded_s}}` | Image build acceptance: TP4 cold/seeded health + boot-check startup elapsed seconds on the exact image; retain cache identity and smoke evidence. | `CHANGELOG.md:33`, `README.md:99`, `docker/RELEASE.md:78`, `docs/how-to-use.md:33` |
| `{{NUM:tp4_c1_code_tps}}` | Step 6 README throughput leg: TP4 one-user streaming decode, code prompt, tok/s; median of five. | `README.md:66`, `docs/results.md:62` |
| `{{NUM:tp4_c1_prose_tps}}` | Step 6 README throughput leg: TP4 one-user streaming decode, prose prompt, tok/s; median of five. | `README.md:67`, `docs/results.md:62` |
| `{{NUM:tp4_c1_structured_tps}}` | Step 6 README throughput leg: TP4 one-user streaming decode, structured prompt, tok/s; median of five. | `README.md:17`, `README.md:65`, `docs/results.md:62` |
| `{{NUM:tp4_c8_code_tps}}` | Step 6 README throughput leg: TP4 eight-user aggregate completion rate, code prompt, tok/s; median of five. | `README.md:69`, `docs/results.md:63` |
| `{{NUM:tp4_c8_prose_tps}}` | Step 6 README throughput leg: TP4 eight-user aggregate completion rate, prose prompt, tok/s; median of five. | `README.md:70`, `docs/results.md:63` |
| `{{NUM:tp4_c8_structured_tps}}` | Step 6 README throughput leg: TP4 eight-user aggregate completion rate, structured prompt, tok/s; median of five. | `README.md:17`, `README.md:68`, `docs/results.md:63` |
| `{{NUM:tp4_copy_mismatches}}` | Step 6 long-context copy-fidelity leg: TP4 mismatch count against its baseline. | `docs/results.md:40` |
| `{{NUM:tp4_gsm8k_net_pp}}` | Step 6 paired quality leg: TP4 GSM8K net gain in percentage points versus the same-driver baseline. | `docs/results.md:25` |
| `{{NUM:tp4_gsm8k_passed}}` | Step 6 paired quality leg: TP4 GSM8K count of passed tasks; retain capped outputs. | `README.md:78`, `docs/results.md:22` |
| `{{NUM:tp4_gsm8k_pct}}` | Step 6 paired quality leg: TP4 GSM8K percent correct on all 1,319 problems. | `README.md:79`, `docs/results.md:22` |
| `{{NUM:tp4_gsm8k_p}}` | Step 6 paired quality leg: TP4 GSM8K two-sided paired McNemar p value. | `docs/results.md:28` |
| `{{NUM:tp4_humaneval_net_pp}}` | Step 6 paired quality leg: TP4 HumanEval net gain in percentage points versus the same-driver baseline. | `docs/results.md:25` |
| `{{NUM:tp4_humaneval_passed}}` | Step 6 paired quality leg: TP4 HumanEval count of passed tasks; retain capped outputs. | `README.md:78`, `docs/results.md:21` |
| `{{NUM:tp4_humaneval_p}}` | Step 6 paired quality leg: TP4 HumanEval two-sided paired McNemar p value. | `docs/results.md:28` |
| `{{NUM:tp4_kv_context_ratio}}` | Step 6 validation boot: measured TP4 KV pool divided by 262,144. | `docs/results.md:66` |
| `{{NUM:tp4_kv_tokens}}` | Step 6 validation boot: TP4 reported usable KV tokens at 262,144 context with final release defaults. | `CHANGELOG.md:12`, `README.md:18`, `README.md:72`, `docs/how-it-works.md:95`, `docs/results.md:66` |
| `{{NUM:tp4_needle_passed}}` | Step 6 long-context needle leg: TP4 passed/total retrieval cases through the context ceiling. | `docs/results.md:32` |
| `{{NUM:tp4_needle_total}}` | Step 6 long-context needle leg: TP4 passed/total retrieval cases through the context ceiling. | `docs/results.md:32` |
| `{{NUM:tp4_prefill_tps}}` | Step 6 README throughput leg: TP4 cold prefill, median of nine real-text rates, no cache hits, tok/s. | `README.md:18`, `README.md:71`, `docs/results.md:64` |
| `{{NUM:tp4_recover_kv_gain_pct}}` | Step 6 validation: TP4 RecoverSSM on/off KV pool comparison at matching settings; percentage gain from the measured pools. | `CHANGELOG.md:13`, `docs/results.md:10` |
| `{{NUM:tp4_repeat_eager_mismatches}}` | Step 6 repeatability leg: TP4 mismatch count in the named eager/graph mode. | `docs/results.md:37` |
| `{{NUM:tp4_repeat_graph_mismatches}}` | Step 6 repeatability leg: TP4 mismatch count in the named eager/graph mode. | `docs/results.md:38` |
| `{{NUM:tp4_repeatability_passed}}` | Step 6 repeatability leg: TP4 passed/total single-request repeatability checks, matching cache and batch conditions. | `README.md:21`, `docs/results.md:34` |
| `{{NUM:tp4_repeatability_total}}` | Step 6 repeatability leg: TP4 passed/total single-request repeatability checks, matching cache and batch conditions. | `README.md:21`, `docs/results.md:35` |
| `{{NUM:tp4_response_23255_s}}` | Step 6 README throughput leg: TP4 nonstreaming one-token elapsed seconds at 23255 prompt tokens, median of three. | `docs/results.md:65` |
| `{{NUM:tp4_response_6217_s}}` | Step 6 README throughput leg: TP4 nonstreaming one-token elapsed seconds at 6217 prompt tokens, median of three. | `docs/results.md:65` |
| `{{NUM:validation_xid_delta}}` | Step 6 validation: new Xid-event count across the grouped validation. | `docs/results.md:42` |
| `{{PIN}}` | Step 6 acceptance on the frozen tree, then the exact full 40-hex fork commit selected for release; run tools/set-pin.sh with it. Historical changelog pins remain unchanged. | `.env.advanced.example:40`, `CHANGELOG.md:5`, `README.md:8`, `README.md:8`, `README.md:147`, `docker/Dockerfile:25`, `docker/Dockerfile:33`, `docker/README.md:10`, `docker/README.md:11`, `docker/README.md:42`, `docker/README.md:100`, `docker/build.sh:20`, `docs/compiled-marlin.md:44`, `docs/how-it-works.md:91`, `docs/how-it-works.md:91`, `docs/results.md:5`, `start.sh:155` |
| `{{end}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `start.sh:448` |
| `{{index .Config.Labels \"$LABEL\"}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `start.sh:833` |
| `{{model_name="fixture",engine="0"}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `docker/tests/test_boot_check.py:43` |
| `{{model_name="fixture",finished_reason="stop"}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `docker/tests/test_boot_check.py:47` |
| `{{model_name="other"}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `docker/tests/test_boot_check.py:46` |
| `{{range $k, $v := .Runtimes}}` | Runtime formatting syntax (Docker Go template or test fixture); no release measurement or substitution. Leave unchanged. | `start.sh:448` |

Conditional documentation edits: drop the TensorFold credit if TP4 skip ships
off; drop the cached-boundary credit if cached boundary ships off. The HTML
comments identify both lines. Replace the two driver-link comments only after
the P2P-capable source repository is published with separate approval.

Finish: resolve slots, check the engine and image provenance, confirm launcher
settings match step 5, run CPU checks and full image acceptance, review the
issue reply, then delete this inventory. Nothing in this file authorizes
publication. The issue reply describes the observed crash path and fixes;
its exact original trigger remains unconfirmed.

## Switch inventory

Read-only comparison: `3a2bf16dae` (1.6.0) to frozen candidate
`0954f02db6d6738ce6d57acccf4be19d27caa3a7`, and independently to
`bundle-1003` at `09ade8b09e5c824f67947ec16d51bde002b2320d`.
The branch moved during review and was rechecked at this exact newer head. Compared `vllm/envs.py`, all tracked
Python `os.getenv` / `os.environ` read sites, helper-based reads, and native
read sites. There are 24 new or changed runtime settings in the frozen candidate,
and 25 in the newer bundle head (the first 25 rows): it adds PP4 compiled
prefill, default `1`. No other direct-read changes or `VLLM_MOE_*` default
changes were found. Remaining rows show existing release policy controls. Explicit user overrides remain supported.

| switch | fork default | release value | where set | decided or placeholder |
|---|---|---|---|---|
| `VLLM_CUSTOM_ALLREDUCE_FLAGS` | 0 | TP4: {{DEFAULT:all_reduce_flags}}; PP4: 0 | serve.sh TP4 export; release-features all_reduce_flags | placeholder |
| `VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES` | 262144 | 262144 | serve.sh export; release-features all_reduce_max_bytes | decided |
| `VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR` | None | unset (engine cache fallback) | fork; no export needed; release-features all_reduce_build_dir | decided |
| `VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S` | 60 seconds | 60 seconds (overrun traps) | serve.sh export; release-features all_reduce_wait_s | decided |
| `VLLM_GLM5_PREFILL_PACK_BF16X2` | 1 | 1 | serve.sh export; release-features pack_bf16x2 | decided |
| `VLLM_GLM5_INDEXER_DECODE_RAW_K` | 1 | 1 | serve.sh export; release-features indexer_raw_k | decided |
| `VLLM_GLM5_STATE_INDEX_CHECK` | 0 | 0 (diagnostic off) | serve.sh export; release-features state_index_check | decided |
| `VLLM_GLM5_DECODE_MHC_V2_FN_BF16` | 1 | 1 | serve.sh export; release-features mhc_fn_bf16 | decided |
| `VLLM_GLM5_DECODE_MHC_V3` | 1 | 1 | serve.sh export; release-features mhc_v3 | decided |
| `VLLM_GLM5_DECODE_KDA_V2_DEEP` | 0 | {{DEFAULT:kda_v2_deep}} | serve.sh export; release-features kda_deep | placeholder |
| `VLLM_GLM5_KDA_RECOVER` | 0 | TP4: 1; PP4: 0 | serve.sh TP4 export; release-features recover | decided |
| `VLLM_GLM5_DECODE_KDA_STEP_TILE` | 0 | 1 | serve.sh export; release-features step_tile | decided |
| `VLLM_GLM5_THIN_GEMM_V74` | 1 | 1 | serve.sh export; release-features thin_gemm_v74 | decided |
| `VLLM_GLM5_TOOL_CHOICE_NONE_MASK` | 1 | 1 | serve.sh export; release-features tool_choice_none_mask | decided |
| `VLLM_GLM5_DFLASH_BOUNDARY_CACHE` | 0 | {{DEFAULT:cached_boundary}} | serve.sh DFlash export; draft-policy boundary | placeholder |
| `VLLM_GLM5_DFLASH_DEPTH2` | 0 | {{DEFAULT:depth2}} | serve.sh DFlash export; draft-policy depth2 | placeholder |
| `VLLM_GLM5_DFLASH_SKIP` | 0 | TP4: {{DEFAULT:draft_skip}}; PP4: 0 | serve.sh DFlash export; draft-policy skip | placeholder |
| `VLLM_GLM5_DFLASH_SKIP_COEFFICIENTS` | empty string | draft-skip-coefficients.json beside launcher | serve.sh DFlash export; draft-policy skip_coefficients | placeholder (coefficient values) |
| `VLLM_GLM5_DFLASH_CONFIDENCE_LOG` | empty string | unset (logging off) | fork; no export needed; draft-policy confidence_log | decided |
| `VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED` | 1 | 1 (eligible TP4 shapes, installed op) | serve.sh export; release-features tp4_marlin_prefill_compiled | decided |
| `VLLM_GLM5_MARLIN_PREFILL_CUDA` | empty string; was boolean 0 | unset (removed; ignored even if set) | fork; no export needed; release-features removed_marlin_prefill_cuda | decided |
| `VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS` | 0 | PP4: 1; TP4: 0 | serve.sh per-layout export; release-features pp_marlin_mid_rows | decided |
| `VLLM_GLM5_MARLIN_DECODE_PP_MULTI` | 0 | PP4: 1; TP4: 0 | serve.sh per-layout export; release-features pp_marlin_multi | decided |
| `VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS` | 9-32 | 9-32 | serve.sh export; release-features pp_marlin_multi_rows | decided |
| `VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED` | absent at frozen; 1 at bundle head | 1 (eligible PP4 shapes, installed op) | serve.sh export; release-features pp4_marlin_prefill_compiled | decided; requires newer tree |
| `VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH` | 0 (unchanged) | TP4: 1; PP4: {{DEFAULT:pp4_drafter_width}} | serve.sh DFlash export; draft-policy width | placeholder at PP4 |
| `VLLM_GLM5_DFLASH_ADAPTIVE_K` | 0 (unchanged) | 1 | serve.sh DFlash export; draft-policy adaptive_k | decided |
| `VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS` | 5,4 (unchanged) | 7,5 | serve.sh DFlash export; draft-policy depths | decided |
| `VLLM_GLM5_DFLASH_ADAPTIVE_K_ACCEPT` | 0 (unchanged) | 1 | serve.sh DFlash export; draft-policy accept | decided |
| `VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS` | empty string (unchanged) | {{NUM:tp4_adaptive_costs}} / {{NUM:pp4_adaptive_costs}} | serve.sh per-layout export; draft-policy costs | placeholder |
| `VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS_MULTI` | empty string (unchanged) | {{NUM:tp4_adaptive_costs_multi}} / {{NUM:pp4_adaptive_costs_multi}} | serve.sh per-layout export; draft-policy costs_multi | placeholder |
| `VLLM_GLM5_MARLIN_DECODE_CUDA` | 0 (unchanged) | 1 if library installed, otherwise 0 | serve.sh discovery/export; release-features marlin_decode | decided |
| `VLLM_GLM5_MARLIN_DECODE_VARIANT` | orig (unchanged) | orig | serve.sh export; release-features marlin_variant | decided |

`VLLM_MOE_SKIP_PADDING` stays at its unchanged fork default `1`; draft skip
also activates padding handling in the new tree. `VLLM_GLM5_MOE_MASK_PADDING`
is exported as `1` by the existing launcher block and is required by skip.
Existing parent gates still apply to thin GEMM, mHC and prefill kernels;
compiled PP4 decode row switches require `VLLM_GLM5_MARLIN_DECODE_CUDA=1`.

Frozen-tree differences requiring exports: KDA step tile `0` -> `1`, TP4
RecoverSSM `0` -> `1`, PP4 compiled decode mid/multi rows `0` -> `1`, and
DFlash adaptive policy, width, costs and coefficient file. Flags-in-data,
draft skip, cached boundary, depth 2 and deep KDA v2 remain explicit decision
slots rather than inheriting the fork's `0`. The 60-second all-reduce wait
limit is retained. The state-index diagnostic stays off in release defaults;
its validation leg must opt in. PP4 compiled wide-tile prefill is absent from
the frozen candidate but present in the newer bundle head, with default `1`
and a changed optional-library prefill signature. Select the final validated
tree and rebuild its matching library before release; an export alone does
not add this path to the older tree. Recheck the inventory if the head moves.

Run `python3 tools/check-release.py` for full-tree wording, per-commit added
lines, messages, trailers and identity. Its sole wording exception matches
the exact frozen dependency line in `docker/CONSTRAINTS` (a package name);
other paths, versions, prefixes and suffixes are rejected.
