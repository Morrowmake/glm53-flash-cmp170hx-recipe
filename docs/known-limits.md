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
- **Repeatable per request, not per batch.** A request on its own gives the
  same result every time. When requests share a batch, the result can differ
  slightly from the same request alone, by about as much as with every
  optimisation switched off.
- **Context and KV are a trade.** Raising `MAX_LEN` lowers how many full-length
  requests fit at once. With `MM_CAP=0` (the default) the memory profiler also
  reserves room for a context-filling video, roughly 150k KV tokens.
- **The DFlash2 checkpoint is needed for the default mode.** `SPEC_MODE=mtp`
  uses the MTP head inside the model checkpoint, and `none` turns speculation
  off; both are slower.
- **Host-staged all-reduce is only for cards without peer access.** Where
  peer-to-peer works, it stands aside for the device-memory path.
