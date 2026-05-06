# FlyDSL Python Layer — RDNA 3.5 Wiring Audit

> Comprehensive analysis of the FlyDSL Python layer to identify everything that gates kernels by
> GPU sub-target / wave-size, with the explicit goal of wiring up an RDNA 3.5 (gfx1150) path.

## A. Executive summary

- **Wave size is not a per-kernel Python knob**:
  [`RocmBackend.detect_target`](../../python/flydsl/compiler/backends/rocm.py) sets
  `GPUTarget.warp_size` to **32** when
  [`is_rdna_arch`](../../python/flydsl/runtime/device.py) is true (`gfx10*`, `gfx11*`,
  `gfx120*`), else **64**, and passes **`wave64=false/true`** and **`chip=<arch>`** into the
  MLIR ROCm pipeline. There is **no** `warp_size` / `num_warps` hook in the compiler itself —
  `num_warps` in autotune is only forwarded as **caller kwargs** if your `@jit` function
  accepts them.
- **Sub-target split in Python is by MLIR atom types and ops**, not by a central dispatcher:
  CDNA3 buffer/MFMA atoms ([`CopyOpCDNA3*`](../../python/flydsl/expr/rocdl/universal.py),
  [`MmaOpCDNA3_MFMAType`](../../python/flydsl/expr/rocdl.py)),
  CDNA4 LDS-transpose + scaled MFMA ([`cdna4.py`](../../python/flydsl/expr/rocdl/cdna4.py)),
  GFX1250-class WMMA + TDM/cluster helpers
  ([`universal.WMMA`](../../python/flydsl/expr/rocdl/universal.py) → `MmaOpGFX1250_WMMAType`,
  [`tdm_ops.py`](../../python/flydsl/expr/rocdl/tdm_ops.py)).
  **gfx1150 is already classified as RDNA** for buffer flags and wave32 pipeline; anything
  **gfx1250-only** (TDM, WMMA_SCALE, cluster ops) still needs validation against RDNA 3.5
  hardware/LLVM.
- **Production [`kernels/`](../../kernels)** mix portable helpers
  ([`get_warp_size()`](../../kernels/kernels_common.py)) with **hard-coded wave64** MFMA stacks
  (`preshuffle_gemm`, `moe_gemm_2stage`, etc.) and **wave32** WMMA paths
  (`rdna_*`, `*_gfx1250.py`). A PoC for **gfx1150** should start from **RDNA WMMA-style**
  kernels/tests, not CDNA MFMA kernels.
- **[`python/flydsl/expr/rocdl.py`](../../python/flydsl/expr/rocdl.py) vs
  [`python/flydsl/expr/rocdl/`](../../python/flydsl/expr/rocdl/)**: the package
  **`expr/rocdl/`** is what
  [`flydsl.expr`](../../python/flydsl/expr/__init__.py) imports (`from . import rocdl`). If both
  the file and the directory exist, Python resolves **`flydsl.expr.rocdl` to the package**, so
  the top-level **`rocdl.py`** file is likely **dead or legacy duplicate** — worth confirming in
  your tree, but user code should treat **`expr/rocdl/__init__.py` + `universal.py` + …** as the
  live API.
- **`pip install -e .`**: [`setup.py`](../../setup.py) is **not** tying the build to a GPU arch
  (no `--offload-arch` / `HSA_OVERRIDE_GFX_VERSION` in setup); install success on gfx1150
  depends on **building the embedded LLVM/MLIR stack and AMDGPU support for your chip** in the
  C++ toolchain, not on Python arch checks.

---

## B. Arch dispatch table

| Site | Note |
|------|------|
| [`device.py` L13–L37](../../python/flydsl/runtime/device.py) | `_arch_from_rocm_agent_enumerator`, default fallback **`gfx942`** if probe fails. |
| [`device.py` L40–L50](../../python/flydsl/runtime/device.py) | **`get_rocm_arch()`**: `FLYDSL_GPU_ARCH` or `HSA_OVERRIDE_GFX_VERSION` (supports `9.4.2` → `gfx942`); else hardware. |
| [`device.py` L77–L94](../../python/flydsl/runtime/device.py) | **`is_rdna_arch`**: `gfx10*`, `gfx11*` (**includes gfx1150**), `gfx120*`; else CDNA-style. |
| [`rocm.py` L18–L27](../../python/flydsl/compiler/backends/rocm.py) | **`warp_size` 32 vs 64** from `is_rdna_arch(arch)`; `make_target(arch)` same rule. |
| [`rocm.py` L36–L61](../../python/flydsl/compiler/backends/rocm.py) | **`chip`** and **`wave64`** in `rocdl-attach-target` / pipeline; `convert-gpu-to-rocdl{chipset=...}`. |
| [`env.py` L219–L227](../../python/flydsl/utils/env.py) | **`FLYDSL_COMPILE_ARCH`** overrides compile arch (`compile.arch`). |
| [`buffer_ops.py` L35–L69](../../python/flydsl/expr/buffer_ops.py) | **`_get_buffer_flags`**: RDNA extra bits via `is_rdna_arch` (bit 24, OOB_SELECT). |
| [`typing.py` L61–L74](../../python/flydsl/expr/typing.py) | **`default_f8_type()`**: gfx95* / **gfx12** → `Float8E4M3FN`, else `Float8E4M3FNUZ`; **gfx11 not special-cased**. |
| [`smem_allocator.py` L222–L251](../../python/flydsl/utils/smem_allocator.py) | **`SMEM_CAPACITY_MAP`**: gfx942, gfx950, gfx1201, gfx1250; **`gfx1150` absent** → `check_smem_capacity` **no enforced limit**. |
| [`kernels_common.py` L71–L78](../../kernels/kernels_common.py) | **`get_warp_size()`**: delegates to `is_rdna_arch`. |
| [`kernels/preshuffle_gemm.py`](../../kernels/preshuffle_gemm.py) (multiple) | `get_rocm_arch`, **wave_size = 64**, gfx942/gfx950/gfx12 branches, MFMA paths. |
| [`kernels/preshuffle_gemm_v2.py`](../../kernels/preshuffle_gemm_v2.py) | `thread_id.x // 64`, gfx942/gfx950. |
| [`kernels/moe_gemm_2stage.py`](../../kernels/moe_gemm_2stage.py) | gfx950 MFMA, layout **`(4, 64)`** wave64 tiling. |
| [`kernels/hgemm_splitk.py`](../../kernels/hgemm_splitk.py) | Uses `get_rocm_arch()`. |
| [`kernels/rmsnorm_kernel.py`](../../kernels/rmsnorm_kernel.py) / [`layernorm_kernel.py`](../../kernels/layernorm_kernel.py) | **`USE_HW_CVT_PK_BF16_F32`** for gfx950(gfx95*); **`WARP_SIZE = get_warp_size()`**. |
| [`kernels/wmma_gemm_gfx1250.py`](../../kernels/wmma_gemm_gfx1250.py) | **`assert arch.startswith("gfx1250")`**. |
| [`kernels/moe_gemm_2stage_common_gfx1250.py` L19–L22](../../kernels/moe_gemm_2stage_common_gfx1250.py) | **`_require_gfx1250()`**. |
| [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) / [`rdna_fp8_preshuffle_gemm.py`](../../kernels/rdna_fp8_preshuffle_gemm.py) | **Wave32** (`tid // 32`), WMMA, gfx120x-oriented comments. |

(Other kernels import `get_rocm_arch` for dtype or guards: `flash_attn_func`, `pa_decode_fp8`,
`moe_blockscale_2stage`, `mla_fwd*`, `mixed_moe_gemm_2stage`, `blockscale_preshuffle_gemm`,
`gemm_fp8fp4_gfx1250`, etc.)

---

## C. Python atom constructor inventory

**Public modules under [`python/flydsl/expr/rocdl/`](../../python/flydsl/expr/rocdl/):**

| File | Role |
|------|------|
| [`__init__.py`](../../python/flydsl/expr/rocdl/__init__.py) | Main umbrella: re-exports **`from ...dialects.rocdl import *`**, traced MFMA wrappers, gfx1250 cluster/TDM entry points, imports **`universal`**, **`inline_asm`**, subpackage **`cdna4`**. |
| [`universal.py`](../../python/flydsl/expr/rocdl/universal.py) | **CDNA3** `BufferCopy` / `BufferCopyLDS` / `BufferAtomic`; **CDNA3** `MFMA` (`MmaOpCDNA3_MFMAType`); **GFX1250-class** `WMMA` (`MmaOpGFX1250_WMMAType`); **`make_buffer_tensor`** with [`_get_buffer_flags`](../../python/flydsl/expr/buffer_ops.py). |
| [`cdna4.py`](../../python/flydsl/expr/rocdl/cdna4.py) | **`LDSReadTrans*`** → `CopyOpCDNA4LdsReadTransposeType`; **`MFMA_Scale`** → `MmaOpCDNA4_MFMAScaleType`. |
| [`tdm_ops.py`](../../python/flydsl/expr/rocdl/tdm_ops.py) | **gfx1250 TDM** descriptors + `tensor_load_to_lds` / `tensor_store_from_lds` / `s_wait_tensorcnt` / `global_prefetch`. |
| [`inline_asm.py`](../../python/flydsl/expr/rocdl/inline_asm.py) | **gfx9** `cvt_off_f32_i4`; **gfx950** `cvt_pk_bf16_f32` inline asm. |

**Recommended user-facing constructors (by sub-target):**

| Constructor | MLIR / dialect target | Typical ISA |
|-------------|------------------------|-------------|
| **`MFMA(...)`** / **`fx.rocdl.MFMA`** | `MmaOpCDNA3_MFMAType` | CDNA3 (gfx942-class) |
| **`MFMA_Scale`** ([`cdna4`](../../python/flydsl/expr/rocdl/cdna4.py)) | `MmaOpCDNA4_MFMAScaleType` | CDNA4 (gfx950) |
| **`LDSReadTrans*`** | `CopyOpCDNA4LdsReadTransposeType` | CDNA4 LDS transpose loads |
| **`WMMA(...)`** / **`fx.rocdl.WMMA`** | `MmaOpGFX1250_WMMAType` | **Named GFX1250 in bindings**; used on RDNA paths in repo (also raw `wmma_*` ROCDL ops in [`expr/rocdl.py`](../../python/flydsl/expr/rocdl.py) duplicates). |
| **`BufferCopy*`** / LDS / Atomic | `CopyOpCDNA3*` | CDNA3 buffer path (still used broadly; V# flags patched for RDNA via `_get_buffer_flags`) |
| **TDM helpers** ([`tdm_ops`](../../python/flydsl/expr/rocdl/tdm_ops.py)) | ROCDL TDM intrinsics | **gfx1250**-documented hardware |
| **`wmma_scale_*`** ([`__init__.py`](../../python/flydsl/expr/rocdl/__init__.py)) | WMMA scale ops | **gfx1250** (wave32 operand shapes in docstrings) |

**What an "RDNA35" constructor might look like (planning only):**

- If gfx1150 **shares** WMMA encoding/semantics with the existing **`MmaOpGFX1250_WMMAType`**
  lowering, Python may only need **documentation + tests**, no new wrapper.
- If LLVM/MLIR exposes a **distinct chip feature** or atom metadata, mirror the existing pattern:
  new **`MmaOpRDNA3*Type`** (C++/TableGen) + thin Python **`WMMA_RDNA3(...)`** or
  **`RDNA3.WMMA(...)`** next to
  [`universal.WMMA`](../../python/flydsl/expr/rocdl/universal.py).

**Umbrella / dispatch:**
[`expr/rocdl/__init__.py`](../../python/flydsl/expr/rocdl/__init__.py) does **not** branch on
arch; it **always** exposes all symbols — **correct usage is on the kernel author**. No
automatic `gfx1150` → atom routing exists in Python today.

---

## D. Wave-size / wave64 hard-coding

| Location | Pattern |
|----------|---------|
| [`RocmBackend`](../../python/flydsl/compiler/backends/rocm.py) | **`wave64` MLIR flag** tied to **`is_rdna_arch`**. |
| [`GPUTarget.warp_size`](../../python/flydsl/compiler/backends/base.py) | 32 vs 64 for cache key / tooling; **not** threaded into every layout API automatically. |
| [`kernels_common.get_warp_size`](../../kernels/kernels_common.py) | Portable **32/64** from arch. |
| [`softmax_kernel.py`](../../kernels/softmax_kernel.py), [`layernorm_kernel.py`](../../kernels/layernorm_kernel.py), [`topk_gating_softmax_kernel.py`](../../kernels/topk_gating_softmax_kernel.py) | Use **`get_warp_size()`** (RDNA35-friendly). |
| [`preshuffle_gemm.py`](../../kernels/preshuffle_gemm.py) L232–L236, L432–L436 | **`wave_size = 64`**, layouts **`(4, wave_size)`**, MFMA math. |
| [`preshuffle_gemm_v2.py`](../../kernels/preshuffle_gemm_v2.py) L370 | **`thread_id("x") // 64`**. |
| [`moe_gemm_2stage.py`](../../kernels/moe_gemm_2stage.py) L399 | Layout **`(4, 64)`** for thread→wave/lane. |
| [`rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py), [`rdna_fp8_preshuffle_gemm.py`](../../kernels/rdna_fp8_preshuffle_gemm.py) | **`tid // 32`**, WMMA 16×16. |
| [`examples/03-tiledMma.py`](../../examples/03-tiledMma.py) | **`MFMA`**, **`block=(256,1,1)`** — assumes **CDNA-style** 256-thread block / wave64 tiling (not RDNA WMMA). |
| [`examples/04-preshuffle_gemm.py`](../../examples/04-preshuffle_gemm.py) | **`MFMA(16,16,16,f16)`**, sched barriers **`sched_mfma`**, **`VectorType.get([64], f16)`** — **wave64 / CDNA** oriented. |

---

## E. Kernels in `/kernels/` — RDNA 3.5 readiness (high level)

| Kernel / family | As-is on RDNA 3.5? | Notes |
|-----------------|-------------------|--------|
| **softmax / layernorm / topk_gating_softmax** | **Likely yes** (wave-agnostic math) | Use **`get_warp_size()`**; gfx950-only fast paths for **`cvt_pk_bf16_f32`** — may fall back on gfx1150 unless ISA matches. |
| **rmsnorm** | Partial | Same as layernorm for warp; **HW cvt** gate is gfx950-specific. |
| **preshuffle_gemm / v2, blockscale_*, mfma pipelines** | **No** | **MFMA + wave64** assumptions; need **WMMA** port or run only on CDNA. |
| **moe_gemm_2stage (CDNA path)** | **No** | **MFMA**, **`(4,64)`** layouts. |
| **moe / gemm *_gfx1250** | **No** (gfx1250-only) | Hard **`gfx1250`** asserts; TDM/WMMA_SCALE/cluster — **reuse only after** ISA/capability audit for gfx1150. |
| **rdna_f16_gemm, rdna_fp8_preshuffle_gemm** | **Best templates** | **Wave32 + WMMA**; target **gfx120x** in comments — **closest PoC base** for gfx1150 if WMMA matches. |
| **wmma_gemm_gfx1250** | **No** as-is | Asserts gfx1250; **TDM** may not exist on RDNA3.5. |
| **pa_decode_fp8, flash_attn, mla_*, mixed_moe** | **Case-by-case** | Mostly **CDNA-tuned**; check MFMA and **wave64** splits before use. |
| **hgemm_splitk** | Check | Uses arch string; verify MFMA vs WMMA path. |

---

## F. Files to add or edit for an RDNA 3.5 PoC (full paths + rationale)

| Path | Rationale |
|------|-----------|
| [`python/flydsl/runtime/device.py`](../../python/flydsl/runtime/device.py) | Optional explicit **gfx1150** in docs/tests; confirm **`is_rdna_arch`** still matches your LLVM "wave32 default" story. |
| [`python/flydsl/utils/smem_allocator.py`](../../python/flydsl/utils/smem_allocator.py) | Add **`gfx1150`** (and any alias) to **`SMEM_CAPACITY_MAP`** if you want **compile-time LDS checks**. |
| [`python/flydsl/expr/typing.py`](../../python/flydsl/expr/typing.py) | Extend **`default_f8_type()`** if gfx1150 F8 semantics match gfx12 OCP vs MI300 NUZ. |
| [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) or new **`kernels/rdna35_*`** | PoC **WMMA GEMM** starting from RDNA examples; drop **gfx120** string checks if you generalize to **gfx11**. |
| [`tests/kernels/test_rdna_gemm.py`](../../tests/kernels/test_rdna_gemm.py) | Parametrize **`gfx120` / `gfx1150`** once golden shapes/ops validated. |
| [`tests/arch_compat.py`](../../tests/arch_compat.py) | Whitelist examples/tests that should run on **gfx11** if you add new demos. |
| **C++ / TableGen** (out of tree here) | New **`MmaOp*`** / lowering if WMMA on gfx1150 **≠** existing **`MmaOpGFX1250_WMMAType`**. |

Clean up or remove [`python/flydsl/expr/rocdl.py`](../../python/flydsl/expr/rocdl.py) if confirmed
**shadowed** by the **`rocdl/`** package — reduces confusion when adding RDNA35 wrappers.

---

## G. Open questions

1. **Does gfx1150 WMMA match GFX1250's `MmaOpGFX1250_WMMAType` lowering in your LLVM build**, or
   does it follow RDNA3 (gfx11) WMMA more closely — i.e. can you reuse the **same** Python
   `WMMA()` atom or do you need a new MLIR type name?
2. **TDM, cluster barriers, `wave_id()` TTMP path, `wmma_scale_*`, `global_prefetch`**: which are
   **legal on gfx1150** vs **gfx1250-only** in your ISA/LLVM snapshot?
3. **`default_f8_type`** for **gfx11**: should gfx1150 use **OCP E4M3FN** (like gfx12) or
   **FNUZ** (like gfx94*)?
4. **Per-CU/WGP LDS limit** for gfx1150: what value belongs in **`SMEM_CAPACITY_MAP`**?
5. **Duplicate [`rocdl.py`](../../python/flydsl/expr/rocdl.py)**: confirm import resolution in
   your environment and whether the file should be deleted or merged to avoid **drift** when you
   add RDNA35 APIs.

---

## Supplement: `@flyc.kernel` / `@flyc.jit` and compiler files

**Target selection:**
[`JitFunction.__call__`](../../python/flydsl/compiler/jit_function.py) calls `get_backend()`
(no explicit arch arg) →
[`RocmBackend.detect_target`](../../python/flydsl/compiler/backends/rocm.py) →
**`FLYDSL_COMPILE_ARCH`** or [`get_rocm_arch()`](../../python/flydsl/runtime/device.py).
[`MlirCompiler.compile(..., arch=backend.target.arch)`](../../python/flydsl/compiler/jit_function.py)
passes that string into the pipeline. **Subgroup / wave size** for IR is driven by backend
**`wave64`** flag, not a Python `@kernel` decorator option.

**`[`/home/aup/FlyDSL/python/flydsl/compiler/`](../../python/flydsl/compiler)` — one-line
summaries**

| File | Summary |
|------|---------|
| [`__init__.py`](../../python/flydsl/compiler/__init__.py) | Public API: `jit`, `compile`, `kernel`, `get_backend`, `GPUTarget`. |
| [`jit_function.py`](../../python/flydsl/compiler/jit_function.py) | `@jit`, caching, MLIR module build, `get_backend()`, `MlirCompiler.compile`, `JitFunction.compile_hints`. |
| [`kernel_function.py`](../../python/flydsl/compiler/kernel_function.py) | `@kernel`, `KernelLauncher`, GPU module/func creation, **`known_block_size`**, compile-hints thread-local. |
| [`jit_executor.py`](../../python/flydsl/compiler/jit_executor.py) | `ExecutionEngine`, resolve JIT `.so` paths from backend. |
| [`jit_argument.py`](../../python/flydsl/compiler/jit_argument.py) | Torch/LPack → structured JIT arguments and types. |
| [`mlir_utils.py`](../../python/flydsl/compiler/mlir_utils.py) | Convert Python values to MLIR attributes. |
| [`llvm_options.py`](../../python/flydsl/compiler/llvm_options.py) | Context manager for LLVM `cl::opt` overrides. |
| [`protocol.py`](../../python/flydsl/compiler/protocol.py) | `fly_types`, `fly_construct`, pointer protocols. |
| [`ast_rewriter.py`](../../python/flydsl/compiler/ast_rewriter.py) | AST rewrite for tracing (`scf`, const_expr, etc.). |
| [`backends/__init__.py`](../../python/flydsl/compiler/backends/__init__.py) | `get_backend(name, arch=)`, register ROCm backend. |
| [`backends/base.py`](../../python/flydsl/compiler/backends/base.py) | `GPUTarget`, abstract pipeline / fingerprint API. |
| [`backends/rocm.py`](../../python/flydsl/compiler/backends/rocm.py) | ROCm pass pipeline, **chip + wave64**, HIP fatbin options. |

**Autotune:** [`autotune.py`](../../python/flydsl/autotune.py) is **not** arch-gated; it
benchmarks `Config` lists and applies **`waves_per_eu` / `maxnreg`** via
[`CompilationContext.compile_hints`](../../python/flydsl/compiler/kernel_function.py).
**`num_warps`** is only passed through if the **`@jit` function's signature** accepts it.

---

## Supplement: Tests relevant for RDNA 3.5

| File | Relevance |
|------|-----------|
| [`tests/kernels/test_rdna_gemm.py`](../../tests/kernels/test_rdna_gemm.py) | **Primary template**: WMMA + wave32; currently skips unless **`gfx120*`** — extend for **gfx1150**. |
| [`tests/kernels/test_fused_rope_cache.py`](../../tests/kernels/test_fused_rope_cache.py) | RDNA vs CDNA **fp8 encoding** skip logic (`_IS_RDNA`). |
| [`tests/kernels/test_wmma_gemm_gfx1250.py`](../../tests/kernels/test_wmma_gemm_gfx1250.py), [`test_moe_gemm_wmma_gfx1250.py`](../../tests/kernels/test_moe_gemm_wmma_gfx1250.py), [`test_moe_gemm_mxscale_gfx1250.py`](../../tests/kernels/test_moe_gemm_mxscale_gfx1250.py), [`test_gemm_fp8fp4_gfx1250.py`](../../tests/kernels/test_gemm_fp8fp4_gfx1250.py) | **gfx1250**-gated; reuse patterns **after** confirming gfx1150 has same ops. |
| [`tests/kernels/benchmark_common.py`](../../tests/kernels/benchmark_common.py) | WMMA benchmarks gated on **`gfx120`**. |
| [`tests/arch_compat.py`](../../tests/arch_compat.py) | CDNA-only test set vs **RDNA-compatible examples**. |

---

## Supplement: Examples

| Example | Arch / wave notes |
|---------|-------------------|
| [`examples/03-tiledMma.py`](../../examples/03-tiledMma.py) | **`MFMA`**, `block=(256,1,1)` — **CDNA-oriented**; not a RDNA35 WMMA template. |
| [`examples/04-preshuffle_gemm.py`](../../examples/04-preshuffle_gemm.py) | **MFMA 16×16×16 f16**, `sched_mfma`, **vector 64** fragment — **CDNA / wave64** style. |

---

## Supplement: Build / install vs gfx1150

- [`setup.py`](../../setup.py) **does not** bake **`-mcpu` / `--offload-arch` /
  `HSA_OVERRIDE_GFX_VERSION`** into the Python packaging logic; those remain **runtime/compile
  env** concerns ([`device.py`](../../python/flydsl/runtime/device.py), docs).
  **`pip install -e .`** on a gfx1150 **host** succeeds if the **prebuilt or locally built**
  MLIR/LLVM stack supports that target — not because Python detects the GPU.
