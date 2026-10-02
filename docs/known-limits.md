# Status and known limits

## Status and roadmap

- **Two layouts in 1.4.0.** Tensor-parallel 4 for the fastest single answer on
  x16 links; pipeline-parallel 4 for many parallel users, long prompts and
  cards limited to x4 links.
- **Pipeline-parallel on x4 links** is next to be measured: every PP4 number on
  this page comes from x16 links.
- **Optimisation continues** on both layouts; every change ships with a kill
  switch and measured numbers.

## Known limits

- **Ampere only.** Every kernel in the fork is written for sm_80. On newer GPUs
  upstream vLLM's own kernels are better, and forcing these features on
  elsewhere is untested.
- **Wide links for tensor-parallel.** See [Link width](how-to-use.md#link-width).
- **Pipeline-parallel is measured on x16 links.** It needs far less link
  bandwidth by design, but its numbers on x4 links are not measured yet.
- **Repeatability depends on matching cache and batch conditions.** A request
  on its own is checked for repeatability. Cached and fresh runs of the same
  prompt can differ at near-ties at TP4 because prefix hits change the prefill
  chunk layout, as with batching. When requests share a batch, the result can differ
  slightly from the same request alone, by about as much as with every
  optimisation switched off.
- **Context and KV are a trade.** Raising `MAX_LEN` lowers how many full-length
  requests fit at once. With `MM_CAP=0` (the default) the memory profiler also
  reserves additional memory for a context-filling video.
- **DFlash2 is the only supported speculative mode.** The launcher still
  accepts other `SPEC_MODE` values, but they are unsupported and untested: no
  validation, no issue support. The DFlash2 drafter's licence is
  non-commercial (CC BY-NC-ND 4.0), so commercial users need their own
  evaluation.
- **Host-staged all-reduce is only for cards without peer access.** Where
  peer-to-peer works, it stands aside for the device-memory path.
