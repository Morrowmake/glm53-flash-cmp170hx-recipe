# Optional compiled Marlin

The launcher selects TP4 compiled decode off, PP4 compiled decode on
when the optional library is installed, and compiled prefill off in both layouts.
The effective `PP=4 TP=1` configuration selects the PP4 default even when those
dimensions override `LAYOUT`. One `vllm._ampere_marlin_C` library serves both
layouts; changing layout never rebuilds it. The independent switches override
these defaults explicitly:

```bash
VLLM_GLM5_MARLIN_DECODE_CUDA=1 ./start.sh restart
VLLM_GLM5_MARLIN_PREFILL_CUDA=1 ./start.sh restart
```

An explicit `0` or `1` in the environment or `.env` remains authoritative across
updates. Leave the flags commented out to follow the defaults. Unset PP4 decode
uses CPU-only module discovery, without importing the extension or probing CUDA.
If the library is absent it defaults off with a banner; if present it defaults on
and startup must validate compatibility. Explicit `1` with an absent or incompatible
library, or default-on with an incompatible library, fails before serving, never
silently disabling the requested feature. Both flags off means no extension load.
`DRY=1` prints the resolved flags and command but skips compatibility validation.

The compiled decode path is limited to eligible small batches in TP4 and PP4.
The compiled prefill path is PP4-only, within the engine's validated shape and
token bounds; TP4 prefill retains the released implementation. Unsupported
shapes use the released paths. These are not universal kernel replacements,
and no whole-server speedup is claimed here.

The pinned container image includes the optional library. To build it natively,
use `RUNTIME=native VLLM_BUILD_AMPERE_MARLIN=1 ./start.sh install`.
Native compilation is opt-in; it is independent of runtime enablement.

Normal native installs use the precompiled base engine without compiling this
library. Opting in requires a CUDA toolkit (`CUDA_HOME`) and C++ compiler.
`install` and `update` add the library even at an unchanged engine pin. A separate
stamp includes source, requirements, Python/torch ABI, compiler/toolkit versions
and binary digest; matching builds are reused. Engine reinstalls invalidate the
optional binary, without demanding its toolkit when build is off. Failed rebuilds
cannot leave a stale library in service. Enabling a runtime switch without the
library fails before serving, never silently falling back. Both switches off
means no optional-extension load. Existing `.env` and explicit pin overrides are preserved.

The [container build](../docker/README.md#build-it) prebuilds the same library with
no visible GPUs. Marlin needs no startup compiler or layout-specific rebuild;
other engine kernels still need the image's existing toolkit.
