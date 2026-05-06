# ROCDL / LLVM Intrinsic Coverage for RDNA 3.5 (gfx1150)

> What LLVM / MLIR ROCDL intrinsics are available for RDNA 3.5 (gfx1150 / gfx1151 / gfx1152 /
> gfx1153 / Strix Point) targets, and what is needed for FlyDSL to add an RDNA 3.5 backend.

## A. LLVM source tree location

The pinned LLVM commit FlyDSL builds against is `7f77ca0dbda4abbf9af06537b2c475f20ccd6007`
(recorded in [`thirdparty/llvm-hash.txt`](../../thirdparty/llvm-hash.txt)). The build script
[`scripts/build_llvm.sh`](../../scripts/build_llvm.sh) shallow-clones LLVM to
`${parent_of_FlyDSL}/llvm-project`, which on this workstation resolves to
`/home/aup/llvm-project`.

To inspect the source locally (after building), the relevant files are:

- `${LLVM_SRC_DIR}/llvm/include/llvm/IR/IntrinsicsAMDGPU.td`
- `${LLVM_SRC_DIR}/mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td`
- `${LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/AMDGPU.td` (subtarget features and per-chip predicates)
- `${LLVM_SRC_DIR}/llvm/lib/Target/AMDGPU/GCNProcessors.td` (per-`gfxXXXX` `ProcessorModel`
  definitions)

Cross-reference for the on-die ISA mnemonics used in the tables below:
[`rdna35_instruction_set_architecture.pdf`](../../rdna35_instruction_set_architecture.pdf),
section 16.10 ("VOP3P Instructions"), pages 381–383.

---

## B. WMMA intrinsic table

Column conventions:

- **gfx1100** = RDNA 3 desktop (RX 7900 XTX / W7900). LLVM `FeatureISAVersion11_0_Common` enables
  `FeatureWMMA256bInsts` (A and B held with duplication across the wave).
- **gfx1150** = RDNA 3.5 (Strix Point Radeon 890M, 8060S, 860M iGPU). LLVM
  `FeatureISAVersion11_5_Common` also enables `FeatureWMMA256bInsts`. Per the RDNA 3.5 ISA
  Reference Guide pages 381–383, only six VOP3P WMMA opcodes exist (V_WMMA_F32_16X16X16_F16,
  V_WMMA_F32_16X16X16_BF16, V_WMMA_F16_16X16X16_F16, V_WMMA_BF16_16X16X16_BF16,
  V_WMMA_I32_16X16X16_IU8, V_WMMA_I32_16X16X16_IU4).
- **gfx1201** = RDNA 4 (RX 9070 XT / R9700). `FeatureISAVersion12.Features` enables
  `FeatureWMMA128bInsts` + `FeatureFP8ConversionInsts` — A and B are not duplicated, FP8/BF8
  K=16 WMMA added, K=32 i32_iu4 added.
- **gfx1250** = next-gen APU (referred to as MI450 in [`CLAUDE.md`](../../CLAUDE.md)).
  `FeatureISAVersion12_50_Common.Features` enables `FeatureGFX1250Insts` +
  `FeatureTransposeLoadF4F6Insts` + `FeatureFP8E5M3Insts`. Adds the entire
  "ModsAll_Reuse" / "ModsC" / "ModsABClamp" / "Scale" WMMA family with K=4 (f32), K=32
  (f16/bf16), K=64/128 (fp8/bf8), K=64 (iu8), and the scaled WMMAs.

Legend: ✓ = available; — = absent; (dup) = available but encoded with duplicated A/B operand
layout; "scaled-only" indicates that only the WMMA_SCALE variant exists at that size.

| LLVM intrinsic | gfx1100 | gfx1150 | gfx1201 | gfx1250 | MLIR ROCDL Op |
| --- | --- | --- | --- | --- | --- |
| `llvm.amdgcn.wmma.f32.16x16x16.f16` | ✓ (dup) | ✓ (dup) | ✓ | — (replaced by K=32) | `rocdl.wmma.f32.16x16x16.f16` |
| `llvm.amdgcn.wmma.f32.16x16x16.bf16` | ✓ (dup) | ✓ (dup) | ✓ | — | `rocdl.wmma.f32.16x16x16.bf16` |
| `llvm.amdgcn.wmma.f16.16x16x16.f16` | ✓ (dup, opsel) | ✓ (dup, opsel) | ✓ | — | `rocdl.wmma.f16.16x16x16.f16` |
| `llvm.amdgcn.wmma.bf16.16x16x16.bf16` | ✓ (dup, opsel) | ✓ (dup, opsel) | ✓ | — | `rocdl.wmma.bf16.16x16x16.bf16` |
| `llvm.amdgcn.wmma.i32.16x16x16.iu8` | ✓ (dup) | ✓ (dup) | ✓ | — | `rocdl.wmma.i32.16x16x16.iu8` |
| `llvm.amdgcn.wmma.i32.16x16x16.iu4` | ✓ (dup) | ✓ (dup) | ✓ | — | `rocdl.wmma.i32.16x16x16.iu4` |
| `llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8` | — | — | ✓ | — | `rocdl.wmma.f32.16x16x16.fp8_fp8` |
| `llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8` | — | — | ✓ | — | `rocdl.wmma.f32.16x16x16.fp8_bf8` |
| `llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8` | — | — | ✓ | — | `rocdl.wmma.f32.16x16x16.bf8_fp8` |
| `llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8` | — | — | ✓ | — | `rocdl.wmma.f32.16x16x16.bf8_bf8` |
| `llvm.amdgcn.wmma.i32.16x16x32.iu4` | — | — | ✓ | — | `rocdl.wmma.i32.16x16x32.iu4` |
| `llvm.amdgcn.wmma.f32.16x16x4.f32` | — | — | — | ✓ | `rocdl.wmma.f32.16x16x4.f32` |
| `llvm.amdgcn.wmma.f32.16x16x32.f16` | — | — | — | ✓ | `rocdl.wmma.f32.16x16x32.f16` |
| `llvm.amdgcn.wmma.f32.16x16x32.bf16` | — | — | — | ✓ | `rocdl.wmma.f32.16x16x32.bf16` |
| `llvm.amdgcn.wmma.f16.16x16x32.f16` | — | — | — | ✓ | `rocdl.wmma.f16.16x16x32.f16` |
| `llvm.amdgcn.wmma.bf16.16x16x32.bf16` | — | — | — | ✓ | `rocdl.wmma.bf16.16x16x32.bf16` |
| `llvm.amdgcn.wmma.bf16f32.16x16x32.bf16` | — | — | — | ✓ | `rocdl.wmma.bf16f32.16x16x32.bf16` |
| `llvm.amdgcn.wmma.f32.16x16x64.{fp8,bf8}.{fp8,bf8}` (4 × ) | — | — | — | ✓ | `rocdl.wmma.f32.16x16x64.{fp8_fp8,fp8_bf8,bf8_fp8,bf8_bf8}` |
| `llvm.amdgcn.wmma.f16.16x16x64.{fp8,bf8}.{fp8,bf8}` (4 × ) | — | — | — | ✓ | `rocdl.wmma.f16.16x16x64.*` |
| `llvm.amdgcn.wmma.f32.16x16x128.{fp8,bf8}.{fp8,bf8}` (4 × ) | — | — | — | ✓ | `rocdl.wmma.f32.16x16x128.*` |
| `llvm.amdgcn.wmma.f16.16x16x128.{fp8,bf8}.{fp8,bf8}` (4 × ) | — | — | — | ✓ | `rocdl.wmma.f16.16x16x128.*` |
| `llvm.amdgcn.wmma.i32.16x16x64.iu8` | — | — | — | ✓ | `rocdl.wmma.i32.16x16x64.iu8` |
| `llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4` | — | — | — | scaled-only | `rocdl.wmma.scale.f32.16x16x128.f8f6f4` |
| `llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4` | — | — | — | scaled-only | `rocdl.wmma.scale16.f32.16x16x128.f8f6f4` |
| `llvm.amdgcn.wmma.scale.f32.32x16x128.f4` | — | — | — | scaled-only | `rocdl.wmma.scale.f32.32x16x128.f4` |
| `llvm.amdgcn.wmma.scale16.f32.32x16x128.f4` | — | — | — | scaled-only | `rocdl.wmma.scale16.f32.32x16x128.f4` |

Source for the comment-grouped target gating:
`mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td` at the pinned commit, which groups the WMMA
intrinsics under three explicit comment dividers — `// Available from gfx11`,
`// Available from gfx12`, `// Available from gfx1250`. Cross-referenced with
`llvm/lib/Target/AMDGPU/AMDGPU.td` `FeatureISAVersion11_5_Common` (which contains
`FeatureWMMA256bInsts` but **not** `FeatureWMMA128bInsts`, `FeatureFP8ConversionInsts`, or
`FeatureGFX1250Insts`).

Also note the SWMMAC (sparse, 2:4) family, which `ROCDLOps.td` gates as
`// Available from gfx12`:

| LLVM intrinsic | gfx1100 | gfx1150 | gfx1201 | gfx1250 | MLIR ROCDL Op |
| --- | --- | --- | --- | --- | --- |
| `llvm.amdgcn.swmmac.f32.16x16x32.{f16,bf16}` | — | — | ✓ | ✓ | `rocdl.swmmac.f32.16x16x32.{f16,bf16}` |
| `llvm.amdgcn.swmmac.{f16,bf16}.16x16x32.{f16,bf16}` | — | — | ✓ | ✓ | `rocdl.swmmac.{f16,bf16}.16x16x32.{f16,bf16}` |
| `llvm.amdgcn.swmmac.i32.16x16x32.iu{4,8}` | — | — | ✓ | ✓ | `rocdl.swmmac.i32.16x16x32.iu{4,8}` |
| `llvm.amdgcn.swmmac.i32.16x16x64.iu4` | — | — | ✓ | ✓ | `rocdl.swmmac.i32.16x16x64.iu4` |
| `llvm.amdgcn.swmmac.f32.16x16x32.{fp8,bf8}.{fp8,bf8}` | — | — | ✓ | ✓ | `rocdl.swmmac.f32.16x16x32.{fp8,bf8}.{fp8,bf8}` |
| `llvm.amdgcn.swmmac.{f32,f16,bf16,bf16f32}.16x16x64.{f16,bf16}` | — | — | — | ✓ | `rocdl.swmmac.*.16x16x64.*` |
| `llvm.amdgcn.swmmac.{f32,f16}.16x16x128.{fp8,bf8}.{fp8,bf8}` | — | — | — | ✓ | `rocdl.swmmac.*.16x16x128.*` |
| `llvm.amdgcn.swmmac.i32.16x16x128.iu8` | — | — | — | ✓ | `rocdl.swmmac.i32.16x16x128.iu8` |

**No SWMMAC of any kind on gfx1100/gfx1150** — sparse WMMA was introduced in RDNA 4 (gfx12).

---

## C. Transpose-load / DS_LOAD_TR intrinsic table

`ROCDLOps.td` separates the transpose-load intrinsics into two groups:

1. `// LDS transpose intrinsics (available in GFX950)` — these wrap `llvm.amdgcn.ds.read.tr*`
   and target only gfx950 (CDNA 4 / MI350). They take a single LDS pointer and return a
   transposed register.
2. `// Glb/DS load-transpose intrinsics (available in GFX1250+)` — these wrap
   `llvm.amdgcn.{ds,global}.load.tr*` and are gated by the `gfx1250-insts` subtarget feature.

There is **also** a separate gfx1200/gfx1201 (RDNA 4) family of transpose-loads that the comment
in `ROCDLOps.td` does not call out explicitly but that AMDGPU.td gates with
`FeatureTransposeLoadF4F6Insts` (description: *"Has ds_load_tr4/tr6 and global_load_tr4/tr6
instructions"*). `FeatureTransposeLoadF4F6Insts` is a member of `FeatureISAVersion12.Features`
(gfx1200/gfx1201) and `FeatureISAVersion12_50_Common.Features` (gfx1250). It is **not** in
`FeatureISAVersion11_5_Common` (gfx1150) nor in any GFX11 feature set.

| Intrinsic | gfx1100 | gfx1150 | gfx1201 | gfx1250 | MLIR ROCDL Op |
| --- | --- | --- | --- | --- | --- |
| `llvm.amdgcn.ds.read.tr4.b64` | — | — | — | — *(gfx950 / CDNA 4 only)* | `rocdl.ds.read.tr4.b64` |
| `llvm.amdgcn.ds.read.tr8.b64` | — | — | — | — *(gfx950 only)* | `rocdl.ds.read.tr8.b64` |
| `llvm.amdgcn.ds.read.tr6.b96` | — | — | — | — *(gfx950 only)* | `rocdl.ds.read.tr6.b96` |
| `llvm.amdgcn.ds.read.tr16.b64` | — | — | — | — *(gfx950 only)* | `rocdl.ds.read.tr16.b64` |
| `llvm.amdgcn.global.load.tr4.b64` | — | — | ✓ | ✓ | `rocdl.global.load.tr4.b64` |
| `llvm.amdgcn.global.load.tr8.b64` | — | — | ✓ | ✓ | `rocdl.global.load.tr8.b64` |
| `llvm.amdgcn.global.load.tr6.b96` | — | — | ✓ | ✓ | `rocdl.global.load.tr6.b96` |
| `llvm.amdgcn.global.load.tr.b128` (8 bits in / 128 bits out) | — | — | ✓ | ✓ | `rocdl.global.load.tr8.b128` |
| `llvm.amdgcn.ds.load.tr4.b64` | — | — | ✓ | ✓ | `rocdl.ds.load.tr4.b64` |
| `llvm.amdgcn.ds.load.tr8.b64` | — | — | ✓ | ✓ | `rocdl.ds.load.tr8.b64` |
| `llvm.amdgcn.ds.load.tr6.b96` | — | — | ✓ | ✓ | `rocdl.ds.load.tr6.b96` |
| `llvm.amdgcn.ds.load.tr16.b128` | — | — | ✓ | ✓ | `rocdl.ds.load.tr16.b128` |

**Confirmation from the user-supplied PDF:** Section 16.15 (LDS Instructions, page 535) and
16.20.3 (Global Instructions, page 631) of the RDNA 3.5 ISA PDF do not list any transpose-load
opcode. This is consistent with the LLVM gating: `FeatureTransposeLoadF4F6Insts` is absent from
the GFX11.5 feature vector. **RDNA 3.5 has no hardware transpose load anywhere in its ISA.**

---

## D. Buffer / Global load-store intrinsic differences

The CDNA3 `BufferCopy`, `BufferCopyLDS`, and `BufferAtomic` atoms in
[`lib/Dialect/FlyROCDL/CDNA3/CopyAtom.cpp`](../../lib/Dialect/FlyROCDL/CDNA3/CopyAtom.cpp) emit:

- `ROCDL::RawPtrBufferLoadOp` → `llvm.amdgcn.raw.ptr.buffer.load` — gfx9 onward, including all of
  gfx1100/1150/1201/1250. **Available on gfx1150.**
- `ROCDL::RawPtrBufferStoreOp` → `llvm.amdgcn.raw.ptr.buffer.store` — same. **Available on
  gfx1150.**
- `ROCDL::RawPtrBufferAtomicFaddOp` → `llvm.amdgcn.raw.ptr.buffer.atomic.fadd` — gated by
  `FeatureAtomicFaddRtnInsts` / `FeatureAtomicFaddNoRtnInsts`. Both are in
  `FeatureISAVersion11_Common` so **available on gfx1150 for f32**. f64 fadd requires
  `FeatureFlatBufferGlobalAtomicFaddF64Inst`, which gfx1150 does not have.
- `ROCDL::RawPtrBufferAtomicFmaxOp` → `llvm.amdgcn.raw.ptr.buffer.atomic.fmax` — gated by
  `FeatureAtomicFMinFMaxF32GlobalInsts` (in 11_Common ⇒ **OK on gfx1150**) for f32. For f64 it
  needs `FeatureAtomicFMinFMaxF64GlobalInsts`, **not** present on gfx1150.
- `ROCDL::RawPtrBufferAtomicSmaxOp` / `RawPtrBufferAtomicUminOp` — generic integer buffer
  atomics; available everywhere from gfx9. **OK on gfx1150.**
- `ROCDL::RawPtrBufferLoadLdsOp` → `llvm.amdgcn.raw.ptr.buffer.load.lds`. This is the GMEM-→-LDS
  direct load (*buffer_load_dword lds*). The `FeatureVMemToLDSLoad` subtarget feature
  ("global_load w/lds bit, buffer_load w/lds bit, or global_load_lds") **is in `FeatureGFX9` and
  `FeatureGFX10` but is _NOT_ in `FeatureGFX11`**. RDNA 3 / RDNA 3.5 / RDNA 4 do not have
  buffer/global LDS-direct load — gfx1250 reintroduces it under different intrinsics
  (`global.load.async.to.lds.{b8,b32,b64,b128}`). **`raw.ptr.buffer.load.lds` is therefore _not_
  available on gfx1150.**

For the `make.buffer.rsrc` infrastructure FlyDSL uses to build BufferDesc pointers
(`ROCDL::MakeBufferRsrcOp`, address space 8): supported on gfx9+ including gfx1150. Note that on
gfx1250 the `numRecords` field changed from i32 to i64 (45-bit num_records) per the AMDGPU LLVM
doc, so the same atom needs slight re-typing for gfx1250 but works as-is on gfx1150.

`ROCDL::GlobalPrefetchOp` and `ROCDL::FlatPrefetchOp`: gfx1250+ only (`FeatureVmemPrefInsts` is
in `FeatureISAVersion12_50_Common.Features` only). **Not available on gfx1150.**

`ROCDL::TensorLoadToLDSOp` / `TensorStoreFromLDSOp` (TDM): gfx1250+ only.

The cluster ops (`rocdl.cluster.workgroup.id.{x,y,z}`,
`rocdl.cluster.load.async.to.lds.{b8,b32,b64,b128}`): gfx1250+ only.

Async/asyncmark machinery (`rocdl.s.wait.asynccnt`, `rocdl.asyncmark`,
`rocdl.wait.asyncmark`, `rocdl.global.load.async.to.lds.*`): all gfx1250+ except
`rocdl.asyncmark` itself which the comment claims to be gfx9+, but its actual usefulness is tied
to gfx1250 async loads.

The `rocdl.s.wait.{loadcnt,storecnt,dscnt,expcnt}` family: gfx12+ only (RDNA 4 introduced split
counters). On gfx1150, you must continue to use `rocdl.s.waitcnt` with the GFX11-format
bitfield.

Workgroup barrier (`rocdl.s.barrier`): present everywhere, but the `rocdl.s.barrier.signal` /
`s.barrier.wait` / `s.barrier.signal.isfirst` / `s.get.barrier.state` family is gfx12+. The
`rocdl.s.barrier.{init,signal.var,join,leave}` and `wakeup.barrier` family is gfx12.5+
(gfx1250+) only. On gfx1150 you fall back to plain `rocdl.s.barrier` or `rocdl.barrier`
(lowered to `s_barrier`).

---

## E. Gap analysis — gfx1250 atoms that gfx1150 cannot run

Mapping the operations actually emitted by
[`lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp`](../../lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp)
(the only file under `lib/Dialect/FlyROCDL/GFX1250/` — gfx1250 has no separate `CopyAtom.cpp`
and instead reuses `CDNA3/CopyAtom.cpp` and `CDNA4/CopyAtom.cpp`) plus the ROCDL ops emitted
directly from Python kernels ([`python/flydsl/expr/rocdl.py`](../../python/flydsl/expr/rocdl.py)):

| ROCDL op emitted by gfx1250 path | Available on gfx1150? | Proposed alternative for an RDNA 3.5 backend |
| --- | --- | --- |
| `rocdl.wmma.f32.16x16x4.f32` | No (gfx1250 only). | No equivalent in gfx1150 ISA; emulate with 4 × `rocdl.wmma.f32.16x16x16.f16` after an f32→f16 down-cast, or fall back to scalar V_FMA / V_DOT2 sequences. Realistically, drop f32 × f32 WMMA from the RDNA 3.5 atom entirely (RDNA 3.5 has no f32-input matrix instruction). |
| `rocdl.wmma.f32.16x16x32.f16` (and bf16) | No. | Use `rocdl.wmma.f32.16x16x16.f16` (or `.bf16`) twice with the K dim split into two halves — same lane layout, two MMAs per K=32 tile. |
| `rocdl.wmma.f16.16x16x32.f16`, `rocdl.wmma.bf16.16x16x32.bf16`, `rocdl.wmma.bf16f32.16x16x32.bf16` | No. | Use the K=16 opsel variants (`rocdl.wmma.f16.16x16x16.f16` / `rocdl.wmma.bf16.16x16x16.bf16`) and double the K loop. |
| `rocdl.wmma.f32.16x16x64.{fp8,bf8}.{fp8,bf8}`, etc. K=128 forms | No (gfx1250 only). gfx1201 has only K=16 fp8/bf8 WMMA, gfx1150 has none. | gfx1150 has zero FP8/BF8 hardware support. **Recommend**: refuse to lower an FP8 MMA on gfx1150 (compile error). |
| `rocdl.wmma.i32.16x16x64.iu8` | No. | Use `rocdl.wmma.i32.16x16x16.iu8` four times across the K dim. |
| `rocdl.wmma.i32.16x16x32.iu4` (RDNA 4 only) | No. | Use `rocdl.wmma.i32.16x16x16.iu4` twice. |
| `rocdl.wmma.scale.*`, `rocdl.wmma.scale16.*` | No (gfx1250 only). | No counterpart in gfx1150 ISA. The scaled-WMMA mxfp4/fp6/fp8 family is gfx1250-exclusive. |
| `rocdl.ds.load.tr16.b128`, `rocdl.ds.load.tr*`, `rocdl.global.load.tr.*` | No (gfx1250+ for `ds.load.tr*`/`global.load.tr*`; gfx950 only for `ds.read.tr*`). | gfx1150 has no transpose-load instruction class at all. Replace with manual lane-permute: load contiguous data with regular `ds_read_b32`/`ds_read_b64` and apply `rocdl.ds.bpermute` / `rocdl.permlanex16` / `rocdl.update.dpp` to redistribute bytes between lanes. |
| `rocdl.cluster.workgroup.id.{x,y,z}`, `rocdl.cluster.load.async.to.lds.{b8,b32,b64,b128}`, `rocdl.s.barrier.{init,signal.var,join,leave}`, `rocdl.s.wakeup.barrier`, `rocdl.s.wait.tensorcnt`, `rocdl.s.wait.asynccnt`, `rocdl.tensor.{load.to.lds,store.from.lds}` (TDM), `rocdl.global.prefetch`, `rocdl.flat.prefetch`, `rocdl.global.load.async.to.lds.*` | No. All gfx1250+. | Drop the cluster/TDM/async/prefetch features from the RDNA 3.5 path. Replace TDM with the existing per-lane CDNA3 `RawPtrBufferLoadOp` flow. |
| `rocdl.s.wait.{loadcnt,storecnt,dscnt,expcnt}` (gfx12+ only) | No. | Use single-counter `rocdl.s.waitcnt` with the GFX11 bitfield encoding. |
| `rocdl.cvt.f32.{bf8,fp8}`, `rocdl.cvt.pk.f32.{bf8,fp8}`, `rocdl.cvt.pk.{bf8,fp8}.f32`, `rocdl.cvt.sr.{bf8,fp8}.f32`, scaled cvt family `rocdl.cvt.scale.pk*`, `rocdl.cvt.scalef32.*` | No on gfx1150 (`FeatureFP8ConversionInsts` absent). | No native FP8 converter. Software cast via bit manipulation if needed. |
| `rocdl.raw.ptr.buffer.load.lds` (used by `CopyOpCDNA3BufferCopyLDS`) | No on gfx11/12 (`FeatureVMemToLDSLoad` is only in gfx9 and gfx10). | Issue a regular `rocdl.raw.ptr.buffer.load` to a register, then a regular LDS store via `llvm.store` to a `!llvm.ptr<3>`. (This is what [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) does.) |
| `rocdl.disable_xdl_arb_stall` (sets SCHED_MODE bit 4 via `s_setreg`) | No-op equivalent — gfx1150 has no XDL arbitration. | Remove the call from the RDNA 3.5 path. |
| `rocdl.wave_id` (architected SGPR read on gfx1250) | No (gfx1250 `FeatureArchitectedSGPRs` only). | Compute `wave_id_in_workgroup = thread_id / wavefront_size`; or use `rocdl.mbcnt.lo`/`mbcnt.hi`. |

---

## F. Recommended ROCDL ops for an RDNA 3.5 `MmaOp` / `CopyOp` implementation

Add the following two atom types under `lib/Dialect/FlyROCDL/RDNA3/`. Most of the design can also
be reused for gfx1100-1103 (RDNA 3 desktop) since they share `WMMA256bInsts`.

**`MmaOpRDNA3_WMMAType`** — wave32, M = N = K = 16 only:

| (M, N, K) | (elemTyA, elemTyB, elemTyAcc) | ROCDL op to emit |
| --- | --- | --- |
| 16 × 16 × 16 | (f16, f16, f32) | `ROCDL::wmma_f32_16x16x16_f16` |
| 16 × 16 × 16 | (bf16, bf16, f32) | `ROCDL::wmma_f32_16x16x16_bf16` |
| 16 × 16 × 16 | (f16, f16, f16) | `ROCDL::wmma_f16_16x16x16_f16` (with `op_sel` attr) |
| 16 × 16 × 16 | (bf16, bf16, bf16) | `ROCDL::wmma_bf16_16x16x16_bf16` (with `op_sel` attr) |
| 16 × 16 × 16 | (i8, i8, i32) | `ROCDL::wmma_i32_16x16x16_iu8` (with sign and clamp attrs) |
| 16 × 16 × 16 | (i4, i4, i32) | `ROCDL::wmma_i32_16x16x16_iu4` (with sign and clamp attrs) |

Operand-vector-size convention on RDNA 3 / 3.5 (`WMMA256bInsts`): A and B are
`vector<16xf16>` (or `vector<16xbf16>` / `vector<16xi16>` for bf16 / `vector<4xi32>` packed for
i8 / `vector<4xi32>` packed for i4). C/D for f32 and i32 outputs is `vector<8xf32>` /
`vector<8xi32>`. C/D for f16/bf16 outputs is `vector<16xf16>` / `vector<16xbf16>` because of the
duplicated even-or-odd-half-of-VGPR layout selected by the `op_sel` flag.

`getThrLayout()`: `FxLayout(FxC(32), FxC(1))` (wave32). `getThrValLayoutAB()`: each lane holds
K = 16 elements **with duplication** — lanes 0-15 hold the same data as lanes 16-31 (the "256b"
layout), so the tile is `16 (M) × 16 (K)` with stride `(1, 16)` and each value is replicated
across two 16-lane groups. C/D layout is the gfx11 *packed* layout: 8 elements/lane for f32 acc,
16 elements/lane (op_sel halves) for f16 acc — exactly what
[`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) already builds by hand.

**`CopyOpRDNA3BufferCopyType`** — same shape as the existing `CopyOpCDNA3BufferCopyType` but
with the RDNA 3 / 3.5 buffer-resource flag word:

- Emit `ROCDL::RawPtrBufferLoadOp` and `ROCDL::RawPtrBufferStoreOp` exactly as
  `CDNA3/CopyAtom.cpp` does.
- Use the same `BufferFatPtr::bufferRsrc` builder as CDNA3.
- The flag word in the buffer resource needs `OOB_SELECT = 3, INDEX_STRIDE = 0,
  ADD_TID_ENABLE = 0` (RDNA-style). Look at
  [`python/flydsl/expr/buffer_ops.py::_get_buffer_flags()`](../../python/flydsl/expr/buffer_ops.py)
  which is already RDNA-aware (gated by `is_rdna_arch()`).
- **Do not emit `RawPtrBufferLoadLdsOp`** — buffer→LDS direct load is unavailable on gfx11.
  Implement `BufferCopyLDS` semantics by chaining `RawPtrBufferLoadOp` (to register) then a
  plain `LLVM::StoreOp` to `!llvm.ptr<3>`.
- Atomic add f32, atomic max f32, atomic max i32, atomic min u32 work via the same
  `RawPtrBufferAtomic*Op` ops as gfx942.

**`MmaOpRDNA3_SWMMACType`** — **omit**. SWMMAC is gfx12+ only.

**`CopyOpRDNA3LdsReadTransposeType`** — **omit**. RDNA 3.5 has no `ds_read_tr*` / `ds_load_tr*`.

Other RDNA 3.5 specifics worth wiring in but outside the atom layer:

- `is_rdna_arch()` in [`python/flydsl/runtime/device.py`](../../python/flydsl/runtime/device.py)
  already returns `True` for `gfx115*`, so the existing `wave64 = false`, `warp_size = 32` paths
  in [`python/flydsl/compiler/backends/rocm.py`](../../python/flydsl/compiler/backends/rocm.py)
  route correctly.
- LDS capacity for gfx1150-1153 is **64 KB per WGP** (per AMD's RDNA 3.5 ISA Reference Guide
  section 12.1.2), not the 327680 bytes that gfx1250 has. Add `"gfx1150": 65536, "gfx1151":
  65536, "gfx1152": 65536, "gfx1153": 65536` entries to `SMEM_CAPACITY_MAP` in
  [`python/flydsl/utils/smem_allocator.py`](../../python/flydsl/utils/smem_allocator.py).
- Wave-uniform `wave_id` is **not** available as an SGPR on gfx11.5. Compute it from `tid // 32`.
- `rocdl.s.barrier` is documented as deprecated on gfx12+ but is the correct primitive on
  gfx11 / gfx11.5.
- `rocdl.sched.barrier` and `rocdl.sched.group.barrier` (and `rocdl.iglp.opt`) are all gfx9+ —
  usable on gfx1150, but the IGLP-1 strategy in particular targets MFMA pipelines; on gfx1150
  you should not invoke IGLP and instead author scheduling barriers manually if needed.

[`tests/arch_compat.py`](../../tests/arch_compat.py) already lists tests that work on RDNA. The
new atom types should be opted into by `RDNA_COMPATIBLE_EXAMPLES` and any new RDNA 3.5 GEMM
kernels following the structure of [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py)
and [`kernels/rdna_fp8_preshuffle_gemm.py`](../../kernels/rdna_fp8_preshuffle_gemm.py).

---

## G. Caveats / unverified claims

1. The RDNA 3.5 ISA PDF
   ([`rdna35_instruction_set_architecture.pdf`](../../rdna35_instruction_set_architecture.pdf))
   confirms only the six VOP3P WMMA opcodes and the absence of any `DS_LOAD_TR*` or
   `GLOBAL_LOAD_TR*` opcodes. The PDF is dated 23 July 2024 and is for "RDNA3.5"; AMD is known
   to issue silent updates, so a newer revision may add features. **Verified by reading the
   PDF.**
2. The LLVM intrinsic gating I describe is from
   `mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td` and `llvm/lib/Target/AMDGPU/AMDGPU.td` at the
   **specific** commit `7f77ca0dbda4abbf9af06537b2c475f20ccd6007` that FlyDSL pins. Other LLVM
   revisions may differ — in particular, the `wmma.scale16.*` ops were a 2025 addition.
   **Verified by web-fetching the raw TableGen at that commit.**
3. The mapping "ROCDL op name → enabling subtarget feature" was inferred from the comment
   dividers in `ROCDLOps.td` (`// Available from gfx11`, `// Available from gfx12`,
   `// Available from gfx1250`). The strict gating is in `IntrinsicsAMDGPU.td` (a much larger
   file). For individual intrinsics the predicate is typically of the form
   `Predicate<"Subtarget->hasWMMA256bInsts()">`. The mapping in section B is therefore
   **expected but not personally verified at the per-intrinsic predicate level**. For a critical
   change, grep `IntrinsicsAMDGPU.td` for each intrinsic name and cross-check the predicate.
4. The claim that `raw.ptr.buffer.load.lds` is unavailable on gfx11 / gfx11.5 is based on
   `FeatureVMemToLDSLoad` being absent from `FeatureGFX11`. **Verified from the `FeatureGFX11 :
   GCNSubtargetFeatureGeneration` definition in AMDGPU.td.** The FlyDSL kernel `rdna_f16_gemm.py`
   already deliberately avoids this op (it does explicit `lds_vec_ptr.store(...)` instead of
   `buffer_load_to_lds`), which corroborates the gating.
5. The claim that `rocdl.s.barrier.signal` / `s.barrier.wait` and
   `rocdl.s.wait.{load,store,ds,exp}cnt` are gfx12+ is from the `let description = [{ Available
   on gfx12+. }]` text inside each op definition in `ROCDLOps.td`. **Verified directly.**
6. RDNA 4 (gfx1200/gfx1201) is described above as having the full f8/bf8 K=16 WMMA family and
   the `ds_load_tr` / `global_load_tr` family. This is consistent with
   `FeatureISAVersion12.Features` containing `FeatureFP8ConversionInsts`,
   `FeatureWMMA128bInsts`, and `FeatureTransposeLoadF4F6Insts`. **Verified at the feature list
   level.**
7. gfx1250 is documented in the LLVM `AMDGPUUsage.html` page under "GCN GFX12.5" with no
   associated example product; AMD has not yet shipped a public part with this LLVM-target, so
   the existence of the K=128 scaled WMMA is **verified at the LLVM TableGen level only**, not
   against released hardware.
8. `lib/Dialect/FlyROCDL/GFX1250/` directory contains only `MmaAtom.cpp`. The gfx1250 copy-atom
   layer reuses `CDNA3/CopyAtom.cpp` (raw buffer ops) and `CDNA4/CopyAtom.cpp` (LDS read
   transpose). The kernel-level Python in `kernels/wmma_gemm_gfx1250.py`, etc. reaches around
   the atom layer and emits `rocdl.ds.load.tr16.b128`, `rocdl.tensor.load.to.lds`,
   `rocdl.cluster.load.async.to.lds.*`, etc. directly via the Python ROCDL bindings. **None of
   those direct emits are valid on gfx1150** — the gap analysis in §E covers them.
