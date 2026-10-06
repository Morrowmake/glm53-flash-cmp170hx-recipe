# Optional compiled Marlin

The launcher turns compiled decode on in both layouts when the optional
library is installed. The effective `TP=4 PP=1` or `PP=4 TP=1` configuration selects the default even when
those dimensions override `LAYOUT`. One `vllm._ampere_marlin_C` library serves
both layouts; changing layout never rebuilds it. The decode switch
overrides the default explicitly:

```bash
VLLM_GLM5_MARLIN_DECODE_CUDA=0 ./start.sh restart
```

Compiled decode has two reduction orders in the same library, chosen by
`VLLM_GLM5_MARLIN_DECODE_VARIANT`. `orig` (the default) splits the first MoE
projection four ways along K; it is faster, and because it adds in a different
order the decoded text can differ from the released kernels, within their
accuracy bounds. `exact` keeps the released summation order. The startup log
names the active variant:

```bash
VLLM_GLM5_MARLIN_DECODE_VARIANT=exact ./start.sh restart
```

An explicit `0` or `1` in the environment or `.env` remains authoritative across
updates. Leave the flag commented out to follow the defaults. Unset decode
uses CPU-only module discovery, without importing the extension or probing CUDA.
If the library is absent it defaults off with a banner; if present it defaults on
and startup must validate compatibility. Explicit `1` with an absent or incompatible
library, or default-on with an incompatible library, fails before serving, never
silently disabling the requested feature. Decode off means no extension load.
`DRY=1` prints the resolved settings and command but skips compatibility validation.

The compiled decode path is limited to eligible small batches in TP4 and PP4.
Unsupported shapes use the released paths. This is not a universal kernel
replacement, and no whole-server speedup is claimed here.

The release includes TP4 compiled prefill tiles
(`VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED`, default on) and PP4 compiled
wide-tile prefill (`VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED`, default on).
Shapes outside their gates use the existing split paths. In 1.7.2, prefill
work blocks start in dependency order to avoid stalls; use the matching
engine pin and rebuilt library.
The old `VLLM_GLM5_MARLIN_PREFILL_CUDA` switch is ignored. The compiled library
is rebuilt from `ab60b723ada254a442a4ba5ff27bf837aa27ef83`; decode entry points retain their established outputs.
The published [Results](results.md) remain the 1.7.0 measurements until
1.7.2 validation is complete.

The pinned container image includes the optional library. To build it natively,
use `RUNTIME=native VLLM_BUILD_AMPERE_MARLIN=1 ./start.sh install`.
Native compilation is opt-in; it is independent of runtime enablement.

Normal native installs use the precompiled base engine without compiling this
library. Opting in requires a CUDA toolkit (`CUDA_HOME`) and C++ compiler.
`install` and `update` add the library even at an unchanged engine pin. A separate
stamp includes source, requirements, Python/torch ABI, compiler/toolkit versions
and binary digest; matching builds are reused. Engine reinstalls invalidate the
optional binary, without demanding its toolkit when build is off. Failed rebuilds
cannot leave a stale library in service. Enabling the runtime switch without the
library fails before serving, never silently falling back. Decode off
means no optional-extension load. Existing `.env` and explicit pin overrides are preserved.

The [container build](../docker/README.md#build-it) prebuilds the same library with
no visible GPUs. Marlin needs no startup compiler or layout-specific rebuild;
other engine kernels still need the image's existing toolkit.
