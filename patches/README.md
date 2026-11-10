# Engine patches: faster restarts

Two files of the release engine (`819146023facb31c04f1f8512e52464835a10d40`,
image digest `sha256:18a2f262…cdbe14`) are replaced at start by read-only bind
mounts. They are whole-file copies of the engine's own files with a small
change each, so they only fit this engine build. `ENGINE_PATCHES=0` turns both
mounts off (`ENGINE_PATCHES=0 ./start.sh restart`); for Compose, remove the two
`patches/` lines from `docker-compose.yml`.

| File | Change | Saves per start |
|---|---|---|
| `vllm/v1/executor/multiproc_executor.py` | Start the four workers from threads at once. With the `spawn` start method `proc.start()` blocks while the child imports vLLM (~15 s), so the release starts them one after another. The `fork` path keeps the original serial loop. | ~45 s |
| `vllm/v1/worker/startup_plan.py` | Two fixes to the engine's startup plan (`VLLM_ENABLE_STARTUP_PLAN=1`, now on by default; see below). | ~10–35 s |

## Startup plan fixes

With `VLLM_ENABLE_STARTUP_PLAN=1` each worker saves its measured KV-cache size
and later starts skip memory profiling and the CUDA-graph memory estimate.

1. **Hang.** Each rank decided alone whether to use its plan. When free memory
   differed slightly between cards, one rank skipped profiling while the
   others profiled; their collectives no longer matched and the engine hung
   after "Applying persisted startup plan" (seen with a co-tenant process that
   held ~0.5 GiB on three of four cards). The patch MIN-all-reduces the
   decision over the world group: either every rank uses its plan or none does.
2. **Plan almost never applied.** A plan is used only if free memory is at
   least the recorded baseline. Free memory moves by a few MiB between boots
   (62.07 vs 62.09 GiB here), so the plan was usually discarded. The patch
   saves the baseline and the KV size both lowered by
   `VLLM_STARTUP_PLAN_MARGIN_MB` (default 512), so a normal restart applies it
   and the safety gate is unchanged.

**Cost:** a start that uses a plan has a smaller KV pool: 992,941 instead of
1,066,492 tokens at TP4, `MAX_SEQS=16` (−7 %, still 3.8 full 262k-token
requests). Half of that is the engine itself (it records the size that fits
`--gpu-memory-utilization`, below what full profiling allocates), half is the
margin. `VLLM_ENABLE_STARTUP_PLAN=0` keeps the full pool and profiles every
start.

Tested on TP4: all ranks apply the plan (engine init 63 s → 39 s), and with one
rank's plan made unusable all four fall back to full profiling and start.

## Bytecode cache

Not a patch, a setting: the image ships the engine source without `.pyc` files
and the release sets `PYTHONDONTWRITEBYTECODE=1`, so every process (API
server, engine core, four workers) compiles vLLM from source on every start.
`PYTHONPYCACHEPREFIX=/cache/pycache` keeps the bytecode in the cache mount:
`import vllm.entrypoints.openai.api_server` drops from 17.7 s to 9.0 s.
