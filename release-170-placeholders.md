# Release 1.7.0 fill-in list

Temporary release preparation file; delete before release. Only measured
numbers and the final image digest remain. Use measurements from this release's
engine, driver and SM count; do not reuse earlier results.

The engine is pinned to `caaf6afe8ee8c29b32ef3b77284656ca87a0937e`.
Launcher decisions and calibrated cost tables are filled. Check both layouts
with `python3 tools/check-env-match.py <validation-json>`; the reference stays
outside the recipe. The check lists ignored keys and uses a CPU installation
fixture for optional-library discovery. For a full leg comparison, use
`python3 tools/check-env-match.py --effective <effective-env-json> --layout tp4`
(or `pp4`), with `serve.log` beside the JSON. `--fork <fork-checkout>` selects a
checkout containing the pinned fork commit. This mode uses `P2P=force`, checks
all environment keys and the logged serve arguments, and prints each permitted
difference with its reason and source evidence for fork defaults.

69 distinct release slots, 114 occurrences outside this inventory.

| Placeholder | Fills from | Locations |
|---|---|---|
| `{{NUM:boundary_passed}}` | Step 6 validation: cached-boundary validation, passed case count. | `docs/results.md:45` |
| `{{NUM:boundary_total}}` | Step 6 validation: cached-boundary validation, total case count. | `docs/results.md:46` |
| `{{NUM:decode_early_stopped}}` | Step 6 validation: README throughput leg, number of requests ending below the cap. | `docs/results.md:80` |
| `{{NUM:decode_requests}}` | Step 6 validation: README throughput leg, total measured decode requests. | `docs/results.md:80` |
| `{{NUM:image_compressed_gb}}` | Image build: compressed bytes of the exact final distributed image, converted to decimal GB. | `README.md:57`, `docker/RELEASE.md:80`, `docs/how-to-use.md:11`, `docs/how-to-use.md:19` |
| `{{NUM:image_digest}}` | Image build + full image acceptance: exact final OCI manifest SHA-256, 64 hex characters without the sha256 prefix. Includes the selected seed layer when shipped. | `.env.advanced.example:34`, `CHANGELOG.md:5`, `docker-compose.yml:21`, `docker/README.md:39`, `docker/README.md:141`, `docker/RELEASE.md:82`, `docs/how-it-works.md:90`, `start.sh:157` |
| `{{NUM:image_layer_count}}` | Image build: final filesystem layer count including base and seed; must pass the layer-count gate. | `docker/RELEASE.md:81` |
| `{{NUM:leak_failures}}` | Step 6 validation: cross-request leak leg, failure count. | `docs/results.md:39` |
| `{{NUM:pp4_boot_cold_s}}` | Image build acceptance: PP4 cold/seeded health + boot-check startup elapsed seconds on the exact image; retain cache identity and smoke evidence. | `CHANGELOG.md:30`, `docker/RELEASE.md:79`, `docs/how-to-use.md:32` |
| `{{NUM:pp4_boot_seeded_s}}` | Image build acceptance: PP4 cold/seeded health + boot-check startup elapsed seconds on the exact image; retain cache identity and smoke evidence. | `CHANGELOG.md:31`, `docker/RELEASE.md:79`, `docs/how-to-use.md:33` |
| `{{NUM:pp4_c1_code_tps}}` | Step 6 README throughput leg: PP4 one-user streaming decode, code prompt, tok/s; median of five. | `docs/results.md:70` |
| `{{NUM:pp4_c1_prose_tps}}` | Step 6 README throughput leg: PP4 one-user streaming decode, prose prompt, tok/s; median of five. | `docs/results.md:70` |
| `{{NUM:pp4_c1_structured_tps}}` | Step 6 README throughput leg: PP4 one-user streaming decode, structured prompt, tok/s; median of five. | `docs/results.md:70` |
| `{{NUM:pp4_c8_code_tps}}` | Step 6 README throughput leg: PP4 eight-user aggregate completion rate, code prompt, tok/s; median of five. | `docs/results.md:71` |
| `{{NUM:pp4_c8_prose_tps}}` | Step 6 README throughput leg: PP4 eight-user aggregate completion rate, prose prompt, tok/s; median of five. | `docs/results.md:71` |
| `{{NUM:pp4_c8_structured_tps}}` | Step 6 README throughput leg: PP4 eight-user aggregate completion rate, structured prompt, tok/s; median of five. | `docs/results.md:71` |
| `{{NUM:pp4_copy_mismatches}}` | Step 6 long-context copy-fidelity leg: PP4 mismatch count against its baseline. | `docs/results.md:38` |
| `{{NUM:pp4_gsm8k_net_pp}}` | Step 6 paired quality leg: PP4 GSM8K net gain in percentage points versus the same-driver baseline. | `docs/results.md:24` |
| `{{NUM:pp4_gsm8k_passed}}` | Step 6 paired quality leg: PP4 GSM8K count of passed tasks; retain capped outputs. | `README.md:80`, `docs/results.md:21` |
| `{{NUM:pp4_gsm8k_pct}}` | Step 6 paired quality leg: PP4 GSM8K percent correct on all 1,319 problems. | `README.md:80`, `docs/results.md:22` |
| `{{NUM:pp4_gsm8k_p}}` | Step 6 paired quality leg: PP4 GSM8K two-sided paired McNemar p value. | `docs/results.md:27` |
| `{{NUM:pp4_humaneval_net_pp}}` | Step 6 paired quality leg: PP4 HumanEval net gain in percentage points versus the same-driver baseline. | `docs/results.md:24` |
| `{{NUM:pp4_humaneval_passed}}` | Step 6 paired quality leg: PP4 HumanEval count of passed tasks; retain capped outputs. | `README.md:79`, `docs/results.md:21` |
| `{{NUM:pp4_humaneval_p}}` | Step 6 paired quality leg: PP4 HumanEval two-sided paired McNemar p value. | `docs/results.md:27` |
| `{{NUM:pp4_kv_context_ratio}}` | Step 6 validation boot: measured PP4 KV pool divided by 262,144. | `docs/results.md:74` |
| `{{NUM:pp4_kv_tokens}}` | Step 6 validation boot: PP4 reported usable KV tokens at 262,144 context with final release defaults. | `docs/how-it-works.md:93`, `docs/results.md:74` |
| `{{NUM:pp4_needle_passed}}` | Step 6 long-context needle leg: PP4 passed/total retrieval cases through the context ceiling. | `docs/results.md:31` |
| `{{NUM:pp4_needle_total}}` | Step 6 long-context needle leg: PP4 passed/total retrieval cases through the context ceiling. | `docs/results.md:31` |
| `{{NUM:pp4_prefill_tps}}` | Step 6 README throughput leg: PP4 cold prefill, median of nine real-text rates, no cache hits, tok/s. | `docs/results.md:72` |
| `{{NUM:pp4_repeat_eager_mismatches}}` | Step 6 repeatability leg: PP4 mismatch count in the named eager/graph mode. | `docs/results.md:36` |
| `{{NUM:pp4_repeat_graph_mismatches}}` | Step 6 repeatability leg: PP4 mismatch count in the named eager/graph mode. | `docs/results.md:37` |
| `{{NUM:pp4_repeatability_passed}}` | Step 6 repeatability leg: PP4 passed/total single-request repeatability checks, matching cache and batch conditions. | `docs/results.md:33` |
| `{{NUM:pp4_repeatability_total}}` | Step 6 repeatability leg: PP4 passed/total single-request repeatability checks, matching cache and batch conditions. | `docs/results.md:34` |
| `{{NUM:pp4_response_23255_s}}` | Step 6 README throughput leg: PP4 nonstreaming one-token elapsed seconds at 23255 prompt tokens, median of three. | `docs/results.md:73` |
| `{{NUM:pp4_response_6217_s}}` | Step 6 README throughput leg: PP4 nonstreaming one-token elapsed seconds at 6217 prompt tokens, median of three. | `docs/results.md:73` |
| `{{NUM:pp4_vs_tp4_kv_ratio}}` | Step 6 throughput/KV validation: ratio of the PP4 and TP4 measured KV token pools. | `README.md:117`, `docs/how-it-works.md:20` |
| `{{NUM:pp4_vs_tp4_prefill_ratio}}` | Step 6 throughput/KV validation: ratio of the PP4 and TP4 measured cold-prefill rates. | `README.md:116` |
| `{{NUM:resume_stress_passed}}` | Step 6 validation: resumed-prefill stress, passed case count across both layouts. | `docs/results.md:47` |
| `{{NUM:resume_stress_total}}` | Step 6 validation: resumed-prefill stress, total case count across both layouts. | `docs/results.md:47` |
| `{{NUM:state_index_failures}}` | Step 6 validation: state-index-check stress leg, failure count. | `docs/results.md:48` |
| `{{NUM:tp4_boot_cold_s}}` | Image build acceptance: TP4 cold/seeded health + boot-check startup elapsed seconds on the exact image; retain cache identity and smoke evidence. | `CHANGELOG.md:29`, `README.md:98`, `docker/RELEASE.md:78`, `docs/how-to-use.md:31` |
| `{{NUM:tp4_boot_seeded_s}}` | Image build acceptance: TP4 cold/seeded health + boot-check startup elapsed seconds on the exact image; retain cache identity and smoke evidence. | `CHANGELOG.md:30`, `README.md:99`, `docker/RELEASE.md:78`, `docs/how-to-use.md:33` |
| `{{NUM:tp4_c1_code_tps}}` | Step 6 README throughput leg: TP4 one-user streaming decode, code prompt, tok/s; median of five. | `README.md:66`, `docs/results.md:60` |
| `{{NUM:tp4_c1_prose_tps}}` | Step 6 README throughput leg: TP4 one-user streaming decode, prose prompt, tok/s; median of five. | `README.md:67`, `docs/results.md:60` |
| `{{NUM:tp4_c1_structured_tps}}` | Step 6 README throughput leg: TP4 one-user streaming decode, structured prompt, tok/s; median of five. | `README.md:17`, `README.md:65`, `docs/results.md:60` |
| `{{NUM:tp4_c8_code_tps}}` | Step 6 README throughput leg: TP4 eight-user aggregate completion rate, code prompt, tok/s; median of five. | `README.md:69`, `docs/results.md:61` |
| `{{NUM:tp4_c8_prose_tps}}` | Step 6 README throughput leg: TP4 eight-user aggregate completion rate, prose prompt, tok/s; median of five. | `README.md:70`, `docs/results.md:61` |
| `{{NUM:tp4_c8_structured_tps}}` | Step 6 README throughput leg: TP4 eight-user aggregate completion rate, structured prompt, tok/s; median of five. | `README.md:17`, `README.md:68`, `docs/results.md:61` |
| `{{NUM:tp4_copy_mismatches}}` | Step 6 long-context copy-fidelity leg: TP4 mismatch count against its baseline. | `docs/results.md:38` |
| `{{NUM:tp4_gsm8k_net_pp}}` | Step 6 paired quality leg: TP4 GSM8K net gain in percentage points versus the same-driver baseline. | `docs/results.md:23` |
| `{{NUM:tp4_gsm8k_passed}}` | Step 6 paired quality leg: TP4 GSM8K count of passed tasks; retain capped outputs. | `README.md:78`, `docs/results.md:20` |
| `{{NUM:tp4_gsm8k_pct}}` | Step 6 paired quality leg: TP4 GSM8K percent correct on all 1,319 problems. | `README.md:79`, `docs/results.md:20` |
| `{{NUM:tp4_gsm8k_p}}` | Step 6 paired quality leg: TP4 GSM8K two-sided paired McNemar p value. | `docs/results.md:26` |
| `{{NUM:tp4_humaneval_net_pp}}` | Step 6 paired quality leg: TP4 HumanEval net gain in percentage points versus the same-driver baseline. | `docs/results.md:23` |
| `{{NUM:tp4_humaneval_passed}}` | Step 6 paired quality leg: TP4 HumanEval count of passed tasks; retain capped outputs. | `README.md:78`, `docs/results.md:19` |
| `{{NUM:tp4_humaneval_p}}` | Step 6 paired quality leg: TP4 HumanEval two-sided paired McNemar p value. | `docs/results.md:26` |
| `{{NUM:tp4_kv_context_ratio}}` | Step 6 validation boot: measured TP4 KV pool divided by 262,144. | `docs/results.md:64` |
| `{{NUM:tp4_kv_tokens}}` | Step 6 validation boot: TP4 reported usable KV tokens at 262,144 context with final release defaults. | `CHANGELOG.md:12`, `README.md:18`, `README.md:72`, `docs/how-it-works.md:93`, `docs/results.md:64` |
| `{{NUM:tp4_needle_passed}}` | Step 6 long-context needle leg: TP4 passed/total retrieval cases through the context ceiling. | `docs/results.md:30` |
| `{{NUM:tp4_needle_total}}` | Step 6 long-context needle leg: TP4 passed/total retrieval cases through the context ceiling. | `docs/results.md:30` |
| `{{NUM:tp4_prefill_tps}}` | Step 6 README throughput leg: TP4 cold prefill, median of nine real-text rates, no cache hits, tok/s. | `README.md:18`, `README.md:71`, `docs/results.md:62` |
| `{{NUM:tp4_recover_kv_gain_pct}}` | Step 6 validation: TP4 RecoverSSM on/off KV pool comparison at matching settings; percentage gain from the measured pools. | `CHANGELOG.md:13`, `docs/results.md:10` |
| `{{NUM:tp4_repeat_eager_mismatches}}` | Step 6 repeatability leg: TP4 mismatch count in the named eager/graph mode. | `docs/results.md:35` |
| `{{NUM:tp4_repeat_graph_mismatches}}` | Step 6 repeatability leg: TP4 mismatch count in the named eager/graph mode. | `docs/results.md:36` |
| `{{NUM:tp4_repeatability_passed}}` | Step 6 repeatability leg: TP4 passed/total single-request repeatability checks, matching cache and batch conditions. | `README.md:21`, `docs/results.md:32` |
| `{{NUM:tp4_repeatability_total}}` | Step 6 repeatability leg: TP4 passed/total single-request repeatability checks, matching cache and batch conditions. | `README.md:21`, `docs/results.md:33` |
| `{{NUM:tp4_response_23255_s}}` | Step 6 README throughput leg: TP4 nonstreaming one-token elapsed seconds at 23255 prompt tokens, median of three. | `docs/results.md:63` |
| `{{NUM:tp4_response_6217_s}}` | Step 6 README throughput leg: TP4 nonstreaming one-token elapsed seconds at 6217 prompt tokens, median of three. | `docs/results.md:63` |
| `{{NUM:validation_xid_delta}}` | Step 6 validation: new Xid-event count across the grouped validation. | `docs/results.md:40` |

Other double-brace expressions are runtime formatting syntax in Docker commands
and test fixtures, not release slots. Leave them unchanged.

Finish the measured slots, verify image provenance and full fresh/update
acceptance in both layouts, then delete this inventory. Driver-link comments
remain until the source repository is published with separate approval.
Nothing in this file authorizes publication.
