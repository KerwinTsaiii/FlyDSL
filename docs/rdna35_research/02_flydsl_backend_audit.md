# FlyDSL Backend Architecture — RDNA 3.5 Readiness Audit

> Comprehensive audit of the FlyROCDL backend dialect with a special focus on what already
> exists for wave32 + WMMA, and the path to add an `RDNA3` sub-target for gfx1150 (RDNA 3.5).

## A. Executive summary

- **FlyROCDL sub-target layout is small and explicit:** TableGen defines exactly three MMA atom
  types ([`MmaAtom.td`](../../include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td)) — CDNA3 MFMA,
  CDNA4 MFMA scale, GFX1250 WMMA — and four copy-atom types
  ([`CopyAtom.td`](../../include/flydsl/Dialect/FlyROCDL/IR/CopyAtom.td)) — three CDNA3 buffer
  variants plus CDNA4 LDS read-transpose. Implementation lives under
  [`lib/Dialect/FlyROCDL/CDNA3/`](../../lib/Dialect/FlyROCDL/CDNA3/),
  [`CDNA4/`](../../lib/Dialect/FlyROCDL/CDNA4/), and
  [`GFX1250/`](../../lib/Dialect/FlyROCDL/GFX1250/) only — **there is no separate `RDNA4/`
  backend folder**; RDNA4-style kernels in Python often call **upstream ROCDL WMMA ops
  directly**, bypassing the FlyROCDL `MmaOpGFX1250_WMMA` atom for some shapes.

- **Wave32 + WMMA today:** The only Fly dialect MMA type wired for **wave32** is
  [`MmaOpGFX1250_WMMA`](../../lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp) (`getThrLayout()` uses 32
  threads per MMA). **TDM**
  ([`python/flydsl/expr/rocdl/tdm_ops.py`](../../python/flydsl/expr/rocdl/tdm_ops.py)) and
  **WMMA_SCALE** helpers in [`python/flydsl/expr/rocdl.py`](../../python/flydsl/expr/rocdl.py)
  are documented and used as **gfx1250 / MI450-specific** paths. **LDS transpose**
  (`ds_read_tr*`) is modeled as **CDNA4** copy atoms, not GFX1250.

- **Conversion pipeline is architecture-agnostic:**
  [`lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`](../../lib/Conversion/FlyToROCDL/FlyToROCDL.cpp)
  lowers Fly atoms to LLVM/ROCDL generically; **no `gfx942` / `gfx950` / `gfx1250` branching**
  appears there. **Wave size for codegen** is chosen in the Python ROCm backend via
  `rocdl-attach-target{... wave64=...}`
  ([`python/flydsl/compiler/backends/rocm.py`](../../python/flydsl/compiler/backends/rocm.py)
  lines 49–62), using
  [`is_rdna_arch()`](../../python/flydsl/runtime/device.py) — **`gfx1150` is already treated as
  RDNA** because it starts with `gfx11`, so **warp 32** is already selected without new code.

- **Gaps for RDNA 3.5 (gfx1150):**
  1. **`is_rdna_arch` does not list `gfx1150` explicitly** but **prefix `gfx11` already matches**;
     validate Strix Point reports `gfx1150` vs `gfx1151` etc.
  2. **`SMEM_CAPACITY_MAP`** has no `gfx115*` entry — LDS limit checks may silently skip
     ([`python/flydsl/utils/smem_allocator.py`](../../python/flydsl/utils/smem_allocator.py)
     ~222–251).
  3. **Two WMMA stories:** Layout/TiledMma path uses **GFX1250 atom** with **K=32** for fp16/bf16
     and **ROCDL `wmma_*_16x16x32_*`** in C++; **RDNA4 example**
     [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) uses
     **`wmma_f32_16x16x16_*`** directly — **hardware/ISA alignment for RDNA 3.5 must be
     confirmed** (reuse which path).
  4. **Tests** often hard-gate on `gfx120*` or `gfx1250`, not `gfx115*`.

- **Biggest unknowns:** Whether RDNA 3.5 WMMA mnemonics and K-tile conventions match
  **GFX1250 (16×16×32)** vs **RDNA4-style (16×16×16)** vs **RDNA3**; whether **TDM /
  WMMA_SCALE** exist on Strix; exact **LDS capacity** for gfx115x for `SmemAllocator`
  validation.

---

## B. Atom type inventory

**TableGen class hierarchy** ([`Dialect.td`](../../include/flydsl/Dialect/FlyROCDL/IR/Dialect.td)):

- `FlyROCL_MmaOp` — stateless MMA (`Fly_MmaOpTypeInterface`).
- `FlyROCL_StatefulMmaOp` — stateful MMA (e.g. CDNA4 scales).
- `FlyROCL_CopyOp` — stateless copy.
- `FlyROCL_StatefulCopyOp` — stateful buffer copy (`soffset`, `imm_offset`, etc.).
- **`AtomStateField`** ([`Atom.td`](../../include/flydsl/Dialect/FlyROCDL/IR/Atom.td)):
  `soffset`, `imm_offset`, `scale_a`, `scale_b`.

[`Ops.td`](../../include/flydsl/Dialect/FlyROCDL/IR/Ops.td) is effectively empty (header only).

### Table: atoms (compact)

| Sub-target | Atom mnemonic (TD) | C++ file | Threads (wave) | dtypes / ROCDL surface | Stateful? | Notes |
|------------|----------------------|----------|----------------|-------------------------|-----------|--------|
| CDNA3 | `cdna3.mfma` | [`CDNA3/MmaAtom.cpp`](../../lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp) | **64** | f32/f16/bf16/f8 pairs → f32 acc; **ROCDL `mfma_*`** | No | MFMA family |
| CDNA4 | `cdna4.mfma_scale` | [`CDNA4/MmaAtom.cpp`](../../lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp) | **64** | f8/f6/f4 element types → f32; **`mfma_scale_f32_*_f8f6f4`** | Yes (`scale_a`, `scale_b`) | Scaled MFMA |
| GFX1250 | `gfx1250.wmma` | [`GFX1250/MmaAtom.cpp`](../../lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp) | **32** | f32×f32 K=4; f16/bf16 K=32; f8 pairs K 64/128; i8 K=64; **`wmma_f32_*` / `wmma_f16_*` / `wmma_i32_*`** | No | Wave32 layouts in `gfx1250::` namespace |
| CDNA3 | `cdna3.buffer_copy` | [`CDNA3/CopyAtom.cpp`](../../lib/Dialect/FlyROCDL/CDNA3/CopyAtom.cpp) | thr layout `1` | **`RawPtrBufferLoad`/`Store`** | Yes (`soffset`) | Universal buffer/tensor path in Python [`universal.py`](../../python/flydsl/expr/rocdl/universal.py) |
| CDNA3 | `cdna3.buffer_copy_lds` | same | 1 | **`RawPtrBufferLoadLds`** | Yes (`soffset`, `imm_offset`) | Buffer → LDS |
| CDNA3 | `cdna3.buffer_atomic` | same | 1 | **Atomics (`RawPtrBufferAtomicFadd`, etc.)** | Yes (`soffset`) | Limited ops (Add/Max/Min) |
| CDNA4 | `cdna4.lds_read_trans` | [`CDNA4/CopyAtom.cpp`](../../lib/Dialect/FlyROCDL/CDNA4/CopyAtom.cpp) | **`FxC(16)`** | **`ds_read_tr4_b64`**, `ds_read_tr8_b64`, `ds_read_tr6_b96`, `ds_read_tr16_b64` | No (static rebuild) | `getThrLayout` 16-thread — tuned for CDNA-style tiling |

**Per-directory file list:**

- **[CDNA3](../../lib/Dialect/FlyROCDL/CDNA3/):** `MmaAtom.cpp` — MFMA + layouts (wave **64**).
  `CopyAtom.cpp` — buffer copy/load-LDS/atomic (stateful where noted).
- **[CDNA4](../../lib/Dialect/FlyROCDL/CDNA4/):** `MmaAtom.cpp` — MFMA scale.
  `CopyAtom.cpp` — LDS read-transpose (**not** "WMMA"; uses **DS read-transpose** intrinsics).
- **[GFX1250](../../lib/Dialect/FlyROCDL/GFX1250/):** `MmaAtom.cpp` — **WMMA only**;
  **no GFX1250 CopyAtom** in TableGen — loads/stores reuse **CDNA3 buffer copy** atoms from
  Python helpers.

**RDNA4 directory?** **No.** **gfx12 / WMMA outside GFX1250:** Yes —
**`from .._mlir.dialects.rocdl import *`** exposes upstream WMMA ops;
[`python/flydsl/expr/rocdl.py`](../../python/flydsl/expr/rocdl.py) wraps
**`wmma_f32_16x16x16_*`**, **`wmma_*_16x16x32_*`** (via ODS names matching GFX1250 emits), FP8
**`wmma_f32_16x16x16_*` gfx12**, **`wmma_scale_*` gfx1250**, MFMA wrappers, sched barriers,
cluster ops, etc.

---

## C. Conversion pipeline observations

**Passes** ([`include/flydsl/Conversion/FlyToROCDL/Passes.td`](../../include/flydsl/Conversion/FlyToROCDL/Passes.td)):

- `convert-fly-to-rocdl` — lowers Fly (+ FlyROCDL atom **metadata**) to LLVM/ROCDL.
- `fly-rocdl-cluster-attr` — fixes `amdgpu-cluster-dims` on `llvm.func` after GPU lowering.

**Orchestration** ([`python/flydsl/compiler/backends/rocm.py`](../../python/flydsl/compiler/backends/rocm.py)
lines 36–91):

| Stage | Behavior |
|--------|----------|
| Early Fly passes | `fly-rewrite-func-signature` → … → `fly-convert-atom-call-to-ssa-form` → … → **`convert-fly-to-rocdl`** |
| GPU lowering | **`convert-gpu-to-rocdl{chipset=<arch> ...}`** — **chip string comes from `GPUTarget.arch`** (`compile_hints`/env) |
| Target attach | **`rocdl-attach-target`** with **`chip=<arch>`**, **`wave64`** = **`false`** iff [`is_rdna_arch(chip)`](../../python/flydsl/runtime/device.py) (lines 61–62) |

**FlyToROCDL.cpp:** Lowers `MakeCopyAtom`/`MakeMmaAtom`, `CopyAtomCall`/`MmaAtomCall` (memory and
SSA paths), buffer fat pointers — **no sub-target switches** in the sampled file (~383–876). Atom
behavior is delegated to **`emitAtomCall*`** on each **concrete FlyROCDL type**
(`MmaAtom.cpp` / `CopyAtom.cpp`).

**Wave size:**
- **Python `GPUTarget.warp_size`:** 32 vs 64 from `is_rdna_arch`
  ([`rocm.py`](../../python/flydsl/compiler/backends/rocm.py) lines 21–27).
- **LLVM ROCm backend:** **`wave64` flag** on `rocdl-attach-target` (same file).
- **MMA tile width:** Embedded in **`getThrLayout()`** per atom (e.g. 64 vs 32) — must match
  runtime wave and kernel tile/rasterization assumptions.

---

## D. Runtime / arch detection observations

| Concern | Location | Behavior |
|---------|----------|----------|
| Primary arch string | [`get_rocm_arch()`](../../python/flydsl/runtime/device.py) 40–50 | **`FLYDSL_GPU_ARCH`** or **`HSA_OVERRIDE_GFX_VERSION`**, else `rocm_agent_enumerator -name`; **default fallback `gfx942`** if detection fails (32–37) |
| RDNA vs CDNA | [`is_rdna_arch()`](../../python/flydsl/runtime/device.py) 77–94 | **`gfx10*`**, **`gfx11*`**, **`gfx120*`** → RDNA-style (wave32 + buffer flags) — **`gfx1150` matches `gfx11*`** |
| Buffer V# flags | [`_get_buffer_flags()`](../../python/flydsl/expr/buffer_ops.py) 35–68 | Uses `is_rdna_arch(arch)`; RDNA adds bit 24 + OOB_SELECT (lines 56–68) |

**Repo search (no edits):**

| Pattern | Hits of note |
|---------|----------------|
| `gfx1150`, `gfx1151`, … | **None** in repo |
| `RDNA3` / `rdna3` (case variants) | Comments in [`rocdl.py`](../../python/flydsl/expr/rocdl.py) (~253 WMMA header); **`test_fused_rope_cache.py`** mentions RDNA gfx10/11/12 FP8 behavior |
| `WMMA` / `wmma` | Widespread: tests, kernels, [`rocdl.py`](../../python/flydsl/expr/rocdl.py), [`tdm_ops.py`](../../python/flydsl/expr/rocdl/tdm_ops.py) |
| `wave32` / `WAVE32` | Docs/comments; **logical** via `warp_size` / `wave64` |
| `is_rdna_arch` | [`device.py`](../../python/flydsl/runtime/device.py), [`buffer_ops.py`](../../python/flydsl/expr/buffer_ops.py), [`rocm.py`](../../python/flydsl/compiler/backends/rocm.py) |
| `gfx1100`–`gfx1103` | **No matches** |

**Backend selection:**
[`RocmBackend.detect_target()` / `make_target()`](../../python/flydsl/compiler/backends/rocm.py) —
single ROCm path; arch is the ROCm GFX string.

**CI / staging:** Wheels and promote workflows reference **`gfx942-gfx950`** paths (e.g.
[`.github/workflows/build-whl.yaml`](../../.github/workflows/build-whl.yaml), `promote.yaml`);
**[`.github/runner-config.yml`](../../.github/runner-config.yml)** lists **`gpu_arch: gfx1201`** —
RDNA4-class CI, not gfx115x.

---

## E. Tests / examples that exercise wave32 + WMMA today

**Strict gfx1250:**

- [`tests/kernels/test_wmma_gemm_gfx1250.py`](../../tests/kernels/test_wmma_gemm_gfx1250.py) —
  skips unless `arch == "gfx1250"`.
- [`tests/kernels/test_moe_gemm_wmma_gfx1250.py`](../../tests/kernels/test_moe_gemm_wmma_gfx1250.py) —
  module skip unless `startswith("gfx1250")`.
- [`tests/kernels/test_moe_gemm_mxscale_gfx1250.py`](../../tests/kernels/test_moe_gemm_mxscale_gfx1250.py) —
  same.
- [`tests/kernels/test_gemm_fp8fp4_gfx1250.py`](../../tests/kernels/test_gemm_fp8fp4_gfx1250.py) —
  WMMA_SCALE / MXFP pipelines.

**RDNA4 (gfx120x) WMMA-style (direct ROCDL WMMA in kernel, not necessarily FlyROCDL GFX1250
atom):**

- [`tests/kernels/test_rdna_gemm.py`](../../tests/kernels/test_rdna_gemm.py) — `_requires_rdna4()`
  requires **`ARCH.startswith("gfx120")`**; uses
  [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py).

**Architecture filters elsewhere:**

- [`tests/kernels/conftest.py`](../../tests/kernels/conftest.py) +
  [`tests/arch_compat.py`](../../tests/arch_compat.py) — skips **MFMA-heavy** tests on non-CDNA
  (`gfx9` check); **`RDNA_COMPATIBLE_EXAMPLES`** only lists `01-vectorAdd.py`,
  `02-tiledCopy.py` — **tiled MMA example not in RDNA whitelist**.

**Examples:**

- [`examples/03-tiledMma.py`](../../examples/03-tiledMma.py) — **`MFMA(16,16,4)`**,
  `block=(256,1,1)` — **CDNA-oriented** (256 threads ↔ 4×64-lane waves typical for MFMA path);
  **not a wave32 WMMA example**.

---

## F. Mapping: Python Layout → TiledMma → MmaAtom → ROCDL (wave-size)

1. **`make_mma_atom(...)`** builds a **`!fly.mma_atom<..., fly_rocdl.op>`** wrapping the
   FlyROCDL type (e.g. `gfx1250.wmma`).
2. **`make_tiled_mma`** combines atom + tiling; **`thr_slice(thread_id)`** uses atom
   **`getThrLayout()`** / operand layouts (**32 or 64** lanes per MMA from the atom).
3. **`fly.gemm`** / atom calls lower through **`convert-fly-to-rocdl`** to **`emitAtomCallSSA`**
   on the concrete type → **ROCDL dialect ops** (e.g. `ROCDL::wmma_*` in
   [`GFX1250/MmaAtom.cpp`](../../lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp)).

**Where wave size lives:**

| Layer | Role |
|-------|------|
| **`getThrLayout()` on MMA atom** | Declares threads participating in **one MMA** (64 vs **32**) |
| **Kernel `block=` / tiling** | Must match warp count × **hardware wave size** (e.g. 128 threads = 4×32 on RDNA) |
| **`rocdl-attach-target` `wave64=`** | Tells AMDGPU backend **default execution width** |

**Concrete tension:** [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) uses
**`WAVE_SIZE = 32`**, **`wmma_f32_16x16x16_*`**, **`WMMA_K = 16`**. FlyROCDL **GFX1250** path
uses **`K=32`** for f16/bf16 and lowers to **`wmma_*_16x16x32_*`**
([`GFX1250/MmaAtom.cpp`](../../lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp) lines 287–294).
**`examples/03-tiledMma.py`** assumes **MFMA** and **256-thread block**, which is aligned with
**wave64**/CDNA, not a minimal wave32 WMMA launch.

---

## G. Files likely to change for an "RDNA35" / gfx1150 sub-target (one line each)

| Path | Rationale |
|------|-----------|
| [`include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td`](../../include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td) | New mnemonic if RDNA35 cannot alias GFX1250 WMMA type. |
| [`lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp`](../../lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp) or new `RDNA3/` | `verify()` + `emitAtomCallSSA` + layouts if ISA differs from MI450. |
| [`lib/Dialect/FlyROCDL/CMakeLists.txt`](../../lib/Dialect/FlyROCDL/CMakeLists.txt) | Register new `.cpp`. |
| [`python/flydsl/expr/rocdl/universal.py`](../../python/flydsl/expr/rocdl/universal.py) | Expose **`WMMA()`** factory to new mlir type if not aliased. |
| [`python/flydsl/expr/rocdl.py`](../../python/flydsl/expr/rocdl.py) | Same + any new `wmma_*` wrappers LLVM exposes for gfx115x. |
| [`python/flydsl/runtime/device.py`](../../python/flydsl/runtime/device.py) | Optionally **document/test gfx115***; clarify if **`gfx1150`/`gfx120*`** share same RDNA classification for buffer flags. |
| [`python/flydsl/utils/smem_allocator.py`](../../python/flydsl/utils/smem_allocator.py) | **`SMEM_CAPACITY_MAP["gfx1150"]`** (or family key) so LDS checks are correct. |
| [`python/flydsl/compiler/backends/rocm.py`](../../python/flydsl/compiler/backends/rocm.py) | Only if **`is_rdna_arch`** or **`wave64`** needs a gfx115 exception (currently unlikely). |
| [`tests/kernels/test_*_gfx1250.py`](../../tests/kernels/) and [`test_rdna_gemm.py`](../../tests/kernels/test_rdna_gemm.py) | Duplicate or broaden gates for **gfx115x** correctness. |
| [`tests/arch_compat.py`](../../tests/arch_compat.py) | Optional: whitelist examples/kernels valid on gfx115x. |
| [`kernels/wmma_gemm_gfx1250.py`](../../kernels/wmma_gemm_gfx1250.py), [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) | Port/branch if one path fits RDNA3.5 better than the other. |
| `.github/workflows/*` | CI runners / wheel tags if gfx115x verification is added. |

---

## H. Open questions for humans

1. **ISA:** On gfx1150, do WMMA opcodes align with **gfx1250 MI450 (`16×16×32` intrinsic names in
   ROCm)** or **RDNA3/4 (`16×16×16`)** as in
   [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py)? LLVM/AMDGPU naming determines
   whether **FlyROCDL GFX1250** can be reused unchanged.
2. **TDM / WMMA_SCALE / cluster:** Are **`tdm_ops.py`** pipelines and **`wmma_scale_*`** available
   on Strix Point, or **MI450-only**?
3. **LDS limits:** Preferred **per-WGP LDS capacity** value for **`SMEM_CAPACITY_MAP`** for
   gfx115x?
4. **Testing:** Hardware availability for **automated gfx115*** vs reliance on
   **`HSA_OVERRIDE_GFX_VERSION`** / simulator?
