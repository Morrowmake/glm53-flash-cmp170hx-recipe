# Release 1.7.0 fill-in list

Temporary release preparation file; delete before release. Only GSM8K/HumanEval quality, KL divergence
and the final image digest remain. Use measurements from this release's
engine, driver and SM count for pending results.

The engine is pinned to `c1ce6491efe53934119d306d0a0501b475458e9b`.
Launcher decisions and calibrated cost tables are filled. Check both layouts
with `python3 tools/check-env-match.py <validation-json>`; the reference stays
outside the recipe. The check lists ignored keys and uses a CPU installation
fixture for optional-library discovery. For a full leg comparison, use
`python3 tools/check-env-match.py --effective <effective-env-json> --layout tp4`
(or `pp4`), with `serve.log` beside the JSON. `--fork <fork-checkout>` selects a
checkout containing the pinned fork commit. This mode uses `P2P=force`, checks
all environment keys and the logged serve arguments, and prints each permitted
difference with its reason and source evidence for fork defaults.

17 distinct release slots, 30 occurrences outside this inventory.

| Placeholder | Fills from | Locations |
|---|---|---|
| `{{NUM:image_digest}}` | Image build + full image acceptance: exact final OCI manifest SHA-256, 64 hex characters without the sha256 prefix. Includes the selected seed layer when shipped. | `.env.advanced.example:34`, `CHANGELOG.md:5`, `docker-compose.yml:21`, `docker/README.md:39`, `docker/README.md:141`, `docker/RELEASE.md:78`, `docs/how-it-works.md:90`, `start.sh:157` |
| `{{NUM:pp4_gsm8k_net_pp}}` | Step 6 paired quality leg: PP4 GSM8K net gain in percentage points versus the same-driver baseline. | `docs/results.md:25` |
| `{{NUM:pp4_gsm8k_p}}` | Step 6 paired quality leg: PP4 GSM8K two-sided paired McNemar p value. | `docs/results.md:28` |
| `{{NUM:pp4_gsm8k_passed}}` | Step 6 paired quality leg: PP4 GSM8K count of passed tasks; retain capped outputs. | `README.md:83`, `docs/results.md:22` |
| `{{NUM:pp4_gsm8k_pct}}` | Step 6 paired quality leg: PP4 GSM8K percent correct on all 1,319 problems. | `README.md:83`, `docs/results.md:23` |
| `{{NUM:pp4_humaneval_net_pp}}` | Step 6 paired quality leg: PP4 HumanEval net gain in percentage points versus the same-driver baseline. | `docs/results.md:25` |
| `{{NUM:pp4_humaneval_p}}` | Step 6 paired quality leg: PP4 HumanEval two-sided paired McNemar p value. | `docs/results.md:28` |
| `{{NUM:pp4_humaneval_passed}}` | Step 6 paired quality leg: PP4 HumanEval count of passed tasks; retain capped outputs. | `README.md:82`, `docs/results.md:22` |
| `{{NUM:pp4_kl_divergence}}` | Release KL-divergence validation: measured result for PP4. | `docs/results.md:30` |
| `{{NUM:tp4_gsm8k_net_pp}}` | Step 6 paired quality leg: TP4 GSM8K net gain in percentage points versus the same-driver baseline. | `docs/results.md:24` |
| `{{NUM:tp4_gsm8k_p}}` | Step 6 paired quality leg: TP4 GSM8K two-sided paired McNemar p value. | `docs/results.md:27` |
| `{{NUM:tp4_gsm8k_passed}}` | Step 6 paired quality leg: TP4 GSM8K count of passed tasks; retain capped outputs. | `README.md:81`, `docs/results.md:21` |
| `{{NUM:tp4_gsm8k_pct}}` | Step 6 paired quality leg: TP4 GSM8K percent correct on all 1,319 problems. | `README.md:82`, `docs/results.md:21` |
| `{{NUM:tp4_humaneval_net_pp}}` | Step 6 paired quality leg: TP4 HumanEval net gain in percentage points versus the same-driver baseline. | `docs/results.md:24` |
| `{{NUM:tp4_humaneval_p}}` | Step 6 paired quality leg: TP4 HumanEval two-sided paired McNemar p value. | `docs/results.md:27` |
| `{{NUM:tp4_humaneval_passed}}` | Step 6 paired quality leg: TP4 HumanEval count of passed tasks; retain capped outputs. | `README.md:81`, `docs/results.md:20` |
| `{{NUM:tp4_kl_divergence}}` | Release KL-divergence validation: measured result for TP4. | `docs/results.md:30` |

Other double-brace expressions are runtime formatting syntax in Docker commands
and test fixtures, not release slots. Leave them unchanged.

Finish quality and KL validation, verify image provenance and full fresh/update
acceptance in both layouts, then delete this inventory. Driver-link comments
remain until the source repository is published with separate approval.
Nothing in this file authorizes publication.
