# Hardware notes

Everything in this directory is about getting four CMP 170HX cards into a state
where the software in this repo has something to run on. **None of it is part of
the recipe proper**, and the unlock in particular is your problem, not ours —
see the warning at the bottom.

## The card

The CMP 170HX is a GA100 die — the same silicon as an A100 — sold as a mining
part with compute throughput, memory capacity and BAR1 size cut down in OTP
fuses and firmware-enforced registers. Out of the box it presents ~8 GB and a
fraction of the FP16 rate, and it is useless for this workload.

Unlocked, each card is a 64 GB HBM2e sm_80 GPU. That is what this recipe
assumes: four of them, 256 GB of HBM2e total, which is what makes a 320B MoE at
W4A16 with a 1.15M-token KV pool fit at all.

## Unlocking

We used **[cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)**. It
restores the fused-off features by opening the Platform Lock Manager registers
and writing the memory and compute unlock values, and it ships a patched build
of NVIDIA's open kernel module that redoes the BAR1 resize on every load. On our
box the patched module lands at
`/lib/modules/$(uname -r)/updates/cmpunlocker/nvidia.ko` and announces itself in
the kernel log:

```
NVRM: CMP BAR1: before CYA=0x00000026 CFG=0x80000000 CAP=0x00000400
NVRM: CMP BAR1: after  CYA=0x00000030 CFG=0x8000000a CAP=0x001ffc00
NVRM: CMP BAR1: resize enabled, will attempt up to 64 GB
NVRM: CMP BAR1: final size = 65536 MB
```

If you do not see those four lines for each card, the rest of this repo will not
work. `nvidia-smi` must report `65536 MiB` per GPU before you go any further.

The project moves fast and has a lot of forks. Read its README, not ours.

### Module options

Two files in `/etc/modprobe.d/` matter:

```
# /etc/modprobe.d/cmp-unsupported-gpu.conf
options nvidia NVreg_OpenRmEnableUnsupportedGpus=1

# /etc/modprobe.d/cmp-pcie-gen2.conf
options nvidia NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x1"
```

The first lets the open kernel module bind a GPU it does not officially support.
The second pins the link to Gen 2, which is what these cards negotiate reliably;
letting the driver try for more produced link training failures here.

## Power and cooling

`cmp-power-limit.service` sets persistence mode and a 180 W board limit at boot.
Every number in the top-level README was measured at 180 W. The cards are
passive — no fan, no shroud — so they need chassis airflow that you arrange
yourself; `cmp-fan-controller.service` is included as a stub unit for whatever
controller script fits your chassis, because ours is machine-specific and not
worth publishing.

Persistence mode matters for more than startup latency: without it the driver
tears down and rebuilds GPU state between processes, which on these cards costs
tens of seconds per server start.

## Topology

Four cards on PCIe Gen 2 x16, all on one NUMA node, **no peer-to-peer**:

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      PHB     NODE    NODE    0-111           0
GPU1    PHB      X      NODE    NODE    0-111           0
GPU2    NODE    NODE     X      PHB     0-111           0
GPU3    NODE    NODE    PHB      X      0-111           0

PHB  = through a PCIe host bridge
NODE = through the interconnect between host bridges in one NUMA node
```

`nvidia-smi topo -p2p r` reports `GNS` ("GPU not supported") on every off-diagonal
pair — the driver refuses peer access outright, so vLLM's `CustomAllreduce`
disables itself ("not supported
on more than two PCIe-only GPUs") and NCCL falls back to a shared-memory ring
that pays `2(N-1)` sequential host hops per message. That is the single reason
`VLLM_GLM5_HOST_ALLREDUCE` exists — see the top-level README.

Gen 2 x16 is about 8 GB/s per direction per card. It is a poor interconnect by
any modern standard, and the whole serve configuration is shaped around not
using it very much.

## Host

For reference, the machine these numbers came from:

| | |
|---|---|
| CPU | AMD EPYC 7663, 56 cores / 112 threads |
| RAM | 247 GB |
| OS | Ubuntu 26.04 LTS, kernel 7.0 |
| Driver | 610.57.04 (NVIDIA open kernel module, patched by cmpunlocker) |
| CUDA | 13.3 toolkit at `/usr/local/cuda-13.3` |
| GPUs | 4x CMP 170HX, 65536 MiB each, 180 W, persistence on, PCIe Gen 2 x16 |

Host RAM is not incidental. `VLLM_GLM5_HOST_ALLREDUCE` stages collectives
through a `/dev/shm` segment, and model load pulls 178 GB off disk; 128 GB would
be tight and 64 GB would not work.

---

## This is your responsibility

Unlocking a CMP 170HX is out of scope for this repository. It modifies firmware
state on hardware NVIDIA sold with those features disabled, it can brick a card,
it certainly voids whatever warranty you imagine you have, and nothing here is
endorsed by or affiliated with NVIDIA. We are documenting the configuration our
measurements came from, not recommending that you reproduce it. If you go ahead,
you accept the whole of that risk yourself.
