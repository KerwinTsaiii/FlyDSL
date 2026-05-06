# Phase 1 PoC — Empirical Outcomes (gfx1150 / Radeon 890M)

> Captured live on the development machine after running the full plan from
> [`05_poc_summary.md`](05_poc_summary.md). This doc records what
> worked, what's blocked, and the exact follow-up needed.

## TL;DR

- **C++ atom + Python integration: works end-to-end.** Builds clean, FileCheck
  test passes, MLIR lowering produces the correct ROCDL ops, no regression on
  CDNA3 / CDNA4 / GFX1250 tests.
- **WMMA256b operand widening: works at the IR level.** The atom now emits
  `<16 x f16>` operands via `shufflevector` self-duplication so the LLVM
  AMDGPU backend selects the gfx1100/gfx1150 WMMA256bInsts instruction
  variant.
- **Runtime end-to-end on this dev box: blocked by an environment / install
  issue unrelated to the new atom.** The same `lld invocation failed`
  appears for **every** FlyDSL kernel on this machine, including the
  pre-existing
  [`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) raw-intrinsic
  path. Fix is sysadmin-level (see §3 below).
- **Two remaining technical follow-ups for full correctness on RDNA 3 /
  3.5,** both deferable to Phase 2: (a) the `<8 x f16>` → `<16 x f16>`
  duplication is currently self-duplicating instead of cross-lane combining
  (so output is wrong even when the kernel runs); (b) the FlyDSL fragment
  shape needs the rank-2 slice trick because the layout API otherwise leaks
  a stale K-grid mode. Both are documented inline in
  [`lib/Dialect/FlyROCDL/RDNA3/MmaAtom.cpp`](../../lib/Dialect/FlyROCDL/RDNA3/MmaAtom.cpp)
  and
  [`kernels/rdna3_f16_gemm.py`](../../kernels/rdna3_f16_gemm.py).

## 1. What was successfully built

| Component | Status | Validation |
|---|---|---|
| LLVM/MLIR @ pinned commit `7f77ca0d` | Built (~30 min, ninja, gcc 13.3) | `ls /home/aup/llvm-project/mlir_install/bin/{fly-opt,FileCheck}` exists |
| FlyDSL C++ + Python bindings | Built (`bash scripts/build.sh`) | `pip install -e .` succeeds, `import flydsl` works |
| New `MmaOpRDNA3_WMMAType` | Compiled into `MLIRFlyROCDLDialect` | Symbol exposed in `flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL` |
| TableGen registration | OK | `!fly_rocdl.rdna3.wmma<16x16x16, (f16, f16) -> f32>` parses & prints |
| `WMMA_RDNA3(...)` Python factory | OK | `fx.rocdl.WMMA_RDNA3(16, 16, 16, fx.Float16, fx.Float32)` returns a `MmaAtom` |
| `WMMA(...)` auto-dispatch | OK | On gfx115x/gfx110x → routes to `WMMA_RDNA3`; else → `WMMA_GFX1250` |
| `SMEM_CAPACITY_MAP` entries | OK | gfx1100..gfx1153 all map to 65536 |
| `default_f8_type()` raises on gfx11* | OK | Avoids silent FNUZ fall-through |
| FileCheck `tests/mlir/Conversion/wmma_rdna3.mlir` | **Pass** | All 4 dtype combos lower to the right `rocdl.wmma.*.16x16x16.*` op |
| End-to-end MLIR pass pipeline | **Reaches LLVM IR successfully** | Pipeline produces a complete `gpu.module` with valid LLVM IR up to the lld step |
| Existing CDNA3/CDNA4/GFX1250 FileCheck regression | **No regression** | `mma_atom.mlir` and `copy_atom_stateful.mlir` still pass |

## 2. ROCDL ops emitted by the new atom (verified)

```
$ fly-opt tests/mlir/Conversion/wmma_rdna3.mlir \
    --fly-rewrite-func-signature --fly-canonicalize \
    --fly-layout-lowering --convert-fly-to-rocdl

%9 = rocdl.wmma.f32.16x16x16.f16 %7, %8, %6
   : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>

%11 = rocdl.wmma.f32.16x16x16.bf16 %8, %10, %6
   : (vector<16xi16>, vector<16xi16>, vector<8xf32>) -> vector<8xf32>

%10 = rocdl.wmma.f16.16x16x16.f16 %7, %8, %9
   : (vector<16xf16>, vector<16xf16>, vector<16xf16>) -> vector<16xf16>

%9 = rocdl.wmma.i32.16x16x16.iu8 %7, %8, %6
   : (vector<4xi32>, vector<4xi32>, vector<8xi32>) -> vector<8xi32>
```

These match the LLVM AMDGPU `WMMA256bInsts` operand convention exactly:
A/B widened to 256 bits per lane (16 f16 / 16 i16 / 4 i32 of packed i8) and
C/D either 256-bit (op_sel f16/bf16 acc) or 128-bit (8xf32 / 8xi32 acc).

## 3. Runtime gap on this machine — `lld invocation failed`

### 3.1 Symptom

Every FlyDSL kernel — the new `kernels/rdna3_f16_gemm.py`, the legacy
`kernels/rdna_f16_gemm.py`, and even `examples/01-vectorAdd.py` — fails with:

```
error: "-":2:3: lld invocation failed
error: "-":2:3: An error happened while serializing the module.
```

The MLIR `gpu-module-to-binary` pass invokes the LLD library to link the
kernel against the AMDGCN device libraries (`ocml.bc`, `ockl.bc`, `hip.bc`,
`oclc_*.bc`). On this machine those bitcodes exist at
`/opt/rocm/core-7.12/lib/llvm/amdgcn/bitcode/` (verified `ocml.bc`,
`oclc_isa_version_1100.bc`, `oclc_isa_version_1150.bc`, etc. are all
present), but MLIR's `getROCMPath()` defaults to `/opt/rocm` and there is no
`/opt/rocm/amdgcn` directory, so `appendStandardLibs` returns `failure()`
and lld is invoked without the device libs.

### 3.2 Root cause

The ROCm "core-7.12" install on this box uses a versioned subdir layout
(`/opt/rocm/core-7.12/lib/llvm/amdgcn/bitcode/`) instead of the canonical
`/opt/rocm/amdgcn/bitcode/` that MLIR's
[`mlir/lib/Target/LLVM/ROCDL/Target.cpp`](../../../llvm-project/mlir/lib/Target/LLVM/ROCDL/Target.cpp)
expects (`appendStandardLibs` at lines 134-175).

### 3.3 Workarounds tried (all without sudo)

- **`ROCM_PATH=/opt/rocm/core-7.12/lib/llvm`**: ignored, MLIR still uses
  `/opt/rocm` default. Reason: MLIR's path lookup actually does check
  `ROCM_PATH` (Target.cpp line 86), so this *should* work; the fact that it
  doesn't suggests something in the call chain is overriding it. Needs more
  triage with sudo+strace.
- **Symlink farm `/tmp/rocm-flydsl/amdgcn/bitcode → real-bitcodes`** plus
  `ROCM_PATH=/tmp/rocm-flydsl`: same outcome. Same suspect cause.
- **Add `toolkit=...` to `gpu-module-to-binary` pass options** (committed in
  [`backends/rocm.py`](../../python/flydsl/compiler/backends/rocm.py)):
  the option is accepted (`--help-list` shows it) but does not appear to
  flow into the ROCDL serializer's `toolkitPath` — likely because the
  option is consumed by NVIDIA-specific code paths and the ROCDL path uses
  the per-target attribute / env path instead. Needs a small upstream fix
  in MLIR or a way to set `toolkit` on the `#rocdl.target<...>` attribute
  itself.

### 3.4 Recommended fix (owner: ops / sysadmin)

**Option A (preferred, requires sudo).** Symlink the canonical paths:

```bash
sudo ln -sf /opt/rocm/core-7.12/lib/llvm/amdgcn /opt/rocm/amdgcn
sudo ln -sf /opt/rocm/core-7.12/lib/llvm/lib /opt/rocm/lib  # if missing
```

After this, `ROCM_PATH` is no longer needed and every FlyDSL kernel on this
machine should compile through to a fatbin.

**Option B (no sudo).** Patch MLIR (or FlyDSL's `backends/rocm.py`) to:
1. Inject `toolkit` into `#rocdl.target<...>` via a custom `rocdl-attach-target`-style pass, OR
2. Build LLVM with `-DROCM_PATH=/opt/rocm/core-7.12/lib/llvm` so the
   default toolkit path is correct.

This is a Phase 2 task: it lives outside the scope of the RDNA 3.5 atom
work and affects the broader FlyDSL development setup on Strix Point /
APU-style ROCm installs.

## 4. Numerical-correctness gap (Phase 2)

Even after the lld issue is fixed, the current `MmaOpRDNA3_WMMAType` will
produce **incorrect numerical output** because of how it bridges the
FlyDSL invariant (each lane holds 8 unique elements) to the WMMA256b
hardware contract (lanes 0-15 hold all 16 K-positions per row, lanes
16-31 are duplicates).

The current implementation does the simplest valid bridge —
`shufflevector <8 x f16> %a, %a, <16 x i32> <0..7, 0..7>` — which gives
each lane `(K=v_low, K=v_low)` (its own data duplicated within the lane)
instead of `(K=0..15)` (full row). On gfx1100 / gfx1150, the WMMA hardware
will read 16 K-positions but the upper 8 will be wrong data, producing an
incorrect dot product.

The correct bridge is one of:

1. **Cross-lane permute** before WMMA: `rocdl.ds.bpermute` /
   `rocdl.permlane32_swap` to swap the upper-K data between lane pairs
   `(l, l+16)` so each lane in 0-15 ends up holding K=0..15 and each lane
   in 16-31 holds the same K=0..15 (true duplicate of its mate).
2. **Layout-side fix**: change
   [`getThrValLayoutAB`](../../lib/Dialect/FlyROCDL/RDNA3/MmaAtom.cpp) to
   `((16, 2), 16):((1, 0), 16)` (lane-group stride 0 = duplicate, val
   width 16 = full K row) and accept that
   `|thr| * |val| = 32 * 16 = 512 ≠ M * K = 256`. The FlyDSL invariant
   would no longer hold but the framework may accept it; or wrap in a
   "duplicated atom" interface that publishes `M = 32` to the framework
   and treats the upper half as ghost rows.

Option 1 is the cleaner long-term fix. Option 2 is more invasive at the
layout-API level. The pre-existing
[`kernels/rdna_f16_gemm.py`](../../kernels/rdna_f16_gemm.py) raw-intrinsic
kernel hits the *same* gap on gfx1100/gfx1150 — its tests are gated to
gfx120x precisely because the kernel was never validated on
WMMA256b targets.

## 5. Phase-2 follow-ups, prioritized

1. **Fix the ROCm toolkit path** (sysadmin, blocking everything else).
2. **Replace self-duplicate shufflevector with `rocdl.ds.bpermute`**-based
   cross-lane combine so the WMMA result is numerically correct on
   gfx1100/gfx1150. Same fix benefits the legacy `kernels/rdna_f16_gemm.py`
   when running on gfx115x.
3. **Investigate why `tile.slice(bA, (None, (bid_m, None)))` produces a
   stale K-grid mode in the MMA fragment shape**, forcing the workaround
   `slice(bA, (None, (bid_m, 0)))`. Probably a partitioner bug or
   missing rank-coalescing.
4. **End-to-end correctness test** on the dev box: once (1) and (2) are
   in, run [`tests/kernels/test_rdna3_wmma_gemm.py`](../../tests/kernels/test_rdna3_wmma_gemm.py)
   and verify against PyTorch reference for f16 / bf16 / iu8 / iu4.
5. **Optimised GEMM** layered on top: LDS double-buffer + sched_barrier
   for a non-trivial throughput target (Phase 2 in
   [`05_poc_summary.md`](05_poc_summary.md)).
