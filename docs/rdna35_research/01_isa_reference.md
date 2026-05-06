# RDNA 3.5 ISA — Reference Notes for FlyDSL Backend

> **Source.** Distilled from the AMD *RDNA 3.5 Instruction Set Architecture Reference Guide*
> (23 July 2024, 644 pp.) located at
> [`rdna35_instruction_set_architecture.pdf`](../../rdna35_instruction_set_architecture.pdf).
> Page citations refer to the logical PDF page (e.g. "p. 75"), which appears in the running footer
> of every page.

---

## TL;DR for FlyDSL backend porting

RDNA 3.5 (gfx1150) is **essentially RDNA 3.0 (gfx1100) with minor SALU/scheduler additions and CU
count tuning**. Its WMMA set is the original 6-opcode RDNA 3 set (F16, BF16, IU8, IU4 only); it
has **no FP8, no FP4, no scaled WMMA, no sparse SWMMAC, no `ds_load_tr_*`, no TDM, no
split-barriers, no async global→LDS copies**.

If you have an RDNA 4 / gfx1250 backend working, almost everything new there must be removed when
targeting gfx1150 — what remains is a wave32 + WMMA + raw-DS-load + raw-buffer-load codegen path,
which is the same layer of code as the existing FlyDSL "RDNA 4 / gfx1250 minus the
dtm/scale/fp4/ds_load_tr/sparse" subset.

---

## 1. Architectural overview

### 1.1 Wave / SIMD / register-file

| Property | Value | Source |
|---|---|---|
| Wave size | **wave32 + wave64** both supported, selected per shader | p. 9 (§2.1) |
| SIMD width | **SIMD32** — 4 SIMD32 per WGP, 2 SIMD32 per CU | p. 11 (§2.3), p. 4–5 (terminology) |
| VGPRs per wave | 256 × 32-bit (V0…V255), one DWORD per lane | p. 13 (Tbl. 4) |
| 16-bit VGPR addressing | 32-bit V0 splits into V0.L (bits 15:0) and V0.H (bits 31:16) | p. 67 (§7.4) |
| SGPRs per wave | 106 normal SGPRs + VCC (in S106/107) + 16 TTMP | p. 15 (§3.3.1.1) |
| LDS per WGP | **128 KiB total** organised as 64 banks × 512 entries × 4 B | p. 6 (§1.2.2.1), p. 120 (§12.1) |
| LDS per work-group | **max 64 KiB** (work-group cap) | p. 6 (§1.2.2.1) |
| LDS bank organisation | 2 × 32-bank halves (one half affiliated per CU pair) | p. 120 (§12.1) |
| Max work-groups / WGP | 32 work-groups, ≤ 1024 WIs/WG | p. 11 (§2.3) |
| GDS | 4 KiB device-wide scratch | p. 7 (§1.2.2.2) |
| EXEC mask | 64 bits (wave64); upper 32 bits ignored in wave32 | p. 14 (§3.2.2) |

### 1.2 Memory hierarchy (gfx1150 / Strix Point — Radeon 890M)

The ISA itself only exposes "L0/L1/L2 + (optional) MALL"; the per-product sizes come from the
Strix Point silicon datasheet / Chips & Cheese reverse-engineering, **NOT** the ISA PDF (which is
product-agnostic).

| Cache | Size | Scope | ISA bit that targets it |
|---|---|---|---|
| L0 (vector) | 32 KiB / WGP | per-WGP, write-combining | `GLC` (p. 36 §4.1.1) |
| L0 scalar (Kcache) | 16 KiB / WGP | per-WGP, scalar reads | `S_DCACHE_INV` (p. 78–79 §8.1.3) |
| L1 (GL1) | 256 KiB / Shader Array (4 WGPs share) | shared across WGPs in same SA | `S_GL1_INV`, `BUFFER_GL1_INV` |
| L2 (GL2) | 2 MiB shared across the iGPU | device-wide | `SLC` (p. 37 §4.1.1) |
| MALL / Infinity Cache | **NOT PRESENT on gfx1150** (no DLC on Strix Point — the bit is encoded but ignored) | — | `DLC` is encoded but doesn't do anything on gfx1150 (p. 37) |

The ISA PDF defines the cache-control table generically (p. 36–38 §4.1.1) — for codegen treat
`DLC=0` as the safe default on gfx1150.

### 1.3 gfx1150 / 1151 / 1152 / 1153 product mapping

The ISA PDF cover says simply "RDNA3.5"; it does **not** name product targets. Product mapping
(per AMD LLVM `AMDGPU.td`, Mesa, ROCm, Chips & Cheese):

| llc target | Internal | Product | Configuration |
|---|---|---|---|
| `gfx1150` | Strix Point | Ryzen AI 300 (Strix Point); Radeon 890M | 8 WGPs / **16 CUs**, wave32, 128 KiB VGPR/SIMD |
| `gfx1151` | Strix Halo | Ryzen AI Max (Strix Halo); Radeon 8060S | larger, 192 KiB VGPR/SIMD (like high-end RDNA 3) |
| `gfx1152` | Krackan Point | Ryzen AI 300 (Krackan); Radeon 860M / 840M | smaller variant |
| `gfx1153` | Medusa Point | next-gen successor to Strix; Radeon 820M | reported but late-2025 era |

All four share the **same RDNA 3.5 ISA** described in this PDF — the differences are register-file
size and CU/WGP count, not opcodes. So a single FlyDSL gfx1150 backend works for all four after
retargeting `--mcpu=`.

### 1.4 What changed RDNA 3.5 vs RDNA 3.0 (gfx1100) and vs RDNA 4 (gfx1201)

The PDF doesn't include a "what's new" page; combining the opcode tables (§16.1–16.20) with public
LLVM AMDGPU sources gives:

| Capability | RDNA 3.0 (gfx1100) | RDNA 3.5 (gfx1150) | RDNA 4 (gfx1201) |
|---|---|---|---|
| WMMA opcodes | F16/BF16/IU8/IU4 (6 ops) | **same** (F16/BF16/IU8/IU4, 6 ops) | + FP8/BF8 + 16x16x32 K-shape + SWMMAC sparse |
| SALU FP add/mul/cvt | NOT in gfx1100 | **NEW**: `S_ADD_F32`, `S_MUL_F32`, `S_FMAC_F32`, `S_CVT_F16_F32`, `S_FMAC_F16`, etc. | same as 3.5 |
| `s_singleuse_vdst` register-cache hint | no | **NEW** (per LLVM) | yes |
| `s_delay_alu` | yes | yes (same encoding) | yes (extended) |
| VOPD dual-issue | yes (wave32) | yes (wave32, same op set) | yes (slightly extended) |
| `ds_load_tr_*` (transpose load) | NO | **NO** | YES (`ds_load_tr16_b64`, `ds_load_tr8_b64`, etc.) |
| Split-barrier (`s_barrier_init/signal/wait`) | NO | NO | YES |
| FP8 / FP4 / scaled WMMA | NO | NO | partial (FP8 yes, FP4/scale only on gfx1250) |

The bottom line: **RDNA 3.5 is essentially "RDNA 3.0 + scalar-FP + s_singleuse_vdst" — it does NOT
pick up any of RDNA 4's matrix or memory primitives.**

---

## 2. Matrix instructions (WMMA)

### 2.1 The complete WMMA opcode list (VOP3P, p. 374, opcodes 64–69)

All WMMA on RDNA 3.5 are 16×16×16 only and use the **VOP3P** encoding (64-bit instruction word),
with the standard `OP/OPSEL/OPSEL_HI/NEG/NEG_HI/CLMP/SRC0/SRC1/SRC2/VDST` fields described
p. 68–69 (§7.5).

| Mnemonic | OP # | A dtype | B dtype | C/D dtype | M/N/K | LLVM intrinsic (gfx11/gfx115) |
|---|---:|---|---|---|---|---|
| `V_WMMA_F32_16X16X16_F16` | 64 | F16 | F16 | F32 | 16,16,16 | `llvm.amdgcn.wmma.f32.16x16x16.f16` |
| `V_WMMA_F32_16X16X16_BF16` | 65 | BF16 | BF16 | F32 | 16,16,16 | `llvm.amdgcn.wmma.f32.16x16x16.bf16` |
| `V_WMMA_F16_16X16X16_F16` | 66 | F16 | F16 | F16 | 16,16,16 | `llvm.amdgcn.wmma.f16.16x16x16.f16` |
| `V_WMMA_BF16_16X16X16_BF16` | 67 | BF16 | BF16 | BF16 | 16,16,16 | `llvm.amdgcn.wmma.bf16.16x16x16.bf16` |
| `V_WMMA_I32_16X16X16_IU8` | 68 | IU8 (signed-or-unsigned per `NEG[0]`/`NEG[1]`) | IU8 | I32 | 16,16,16 | `llvm.amdgcn.wmma.i32.16x16x16.iu8` |
| `V_WMMA_I32_16X16X16_IU4` | 69 | IU4 (signed-or-unsigned per `NEG[0]`/`NEG[1]`) | IU4 | I32 | 16,16,16 | `llvm.amdgcn.wmma.i32.16x16x16.iu4` |

**Pseudo-code (p. 381–383, §16.10):** every WMMA forces `EXEC` to all-1s internally for the matrix
issue and restores after:

```text
saved_exec = EXEC
EXEC      = -1
D0(16x16) = S0(16x16) * S1(16x16) + S2(16x16)
EXEC      = saved_exec
```

### 2.2 WMMA encoding & operand rules (p. 75, §7.9; p. 68, §7.5)

* All three sources **must be VGPRs** (`SRC0`, `SRC1`, `SRC2`); SGPRs are illegal as A/B/C in
  WMMA. Inline constants are allowed **only for the C matrix**, and for F16/BF16 the inline value
  is replicated low→high (p. 75, §7.9).
* `NEG` field is repurposed for IU types: `NEG[0]` = signed(1)/unsigned(0) for SRC0 (A),
  `NEG[1]` = same for SRC1 (B); `NEG[2]` and `NEG_HI[2:0]` must be 0. For F16/BF16 WMMA:
  `NEG[1:0]` apply to SRC1 / SRC0 low-16, `NEG_HI[1:0]` apply to SRC1 / SRC0 high-16,
  `{NEG_HI[2], NEG[2]}` are `{ABS, NEG}` on SRC2 (the C matrix). Destination is signed for IU
  types (p. 75, §7.9).
* `OPSEL[0]` and `OPSEL[1]` are **unused** for WMMA. `OPSEL[2]` selects upper-vs-lower half of
  the VGPR for the C/D matrix only on the 16-bit-output variants (`F16_16x16x16_F16`,
  `BF16_16x16x16_BF16`) (p. 68 §7.5, "OPSEL[0] and [1] are unused for WMMA ops, and OPSEL[2] is
  used only with WMMA ops with 16-bit output…").
* `CLMP` is not supported on WMMA (clamp bit is "ignored" for WMMA per p. 62, §7.2.3.1).
* WMMA **does not generate ALU exceptions** (p. 75 §7.9).
* Round-to-nearest-even rounding only (no other rounding modes); denorm flushing follows MODE
  register (p. 75 §7.9).
* Output modifiers (OMOD) are **not supported** on WMMA.

### 2.3 Matrix layout in VGPRs (p. 75–76, §7.9)

For **wave32** at M=N=K=16:
* Matrix A is **column-major** in VGPRs.
* Matrices B, C, D are **row-major** in VGPRs.
* Each lane in lanes 0–15 holds 8 elements of A and 8 of B.
* **In RDNA 3.5 the A and B matrix data is replicated**: lanes 0-15's data must also appear in
  lanes 16-31 (and again in 32-47 / 48-63 for wave64). The PDF says: "lanes 0–15 data are
  replicated into lanes 16–31 (for wave64: also into lanes 32–47 and 48–63)" — p. 75.

  This is the **same convention as RDNA 3.0** and is the key way it differs from RDNA 4 /
  gfx1201, where each lane just loads 8 elements without duplication (LLVM `WMMA128bInsts` vs
  RDNA 3.5's `WMMA256bInsts`). When porting from a gfx1250 backend, you must re-introduce the
  duplication step (e.g. via `v_dual_mov_b32` or a `ds_bpermute`-style broadcast) before issuing
  WMMA on gfx1150.

* C and D matrices for the F32-acc variants store one F32 per lane per accumulator row, so each
  lane in 0–15 holds 8 F32 accumulators (256 bits = 8 VGPRs) per 16x16 tile.
* For F16/BF16-acc variants, C and D pack two 16-bit values per VGPR; `OPSEL[2]` chooses upper or
  lower half.

The detailed lane→element mapping is given in Figure 4 (p. 75–76) of the PDF; the AMD
`amd_matrix_instruction_calculator` is the canonical reference for the exact `(lane, vreg)` ↔
`(row, col)` map.

### 2.4 WMMA scheduling rules (p. 77, §7.9.1)

| First inst | Second inst | Required between |
|---|---|---|
| `WMMA` | `WMMA` whose A or B overlaps first WMMA's D | **1 V_NOP or independent VALU op** (required for correctness) |
| `WMMA` | `WMMA` re-using D as C (same kind) | none required (typical accumulator pattern) |
| `WMMA` | `WMMA` re-using D as C but **different** WMMA type or with IMOD on SRC2 | hardware may stall (perf hint, not correctness) |
| `WMMA` | VALU that reads D | hardware may stall VALU (perf hint, not correctness) |

The "1 NOP between WMMAs whose D feeds the next A or B" is **the only correctness rule** —
accumulator chains (D = D + A·B as the loop body) require no NOPs and run back-to-back.

### 2.5 SWMMAC (sparse) — NOT PRESENT in RDNA 3.5

A search through §16.10 (VOP3P) p. 373–383 finds **no `V_SWMMAC_*` opcodes**. The sparse 4:2 /
2:1 matrix multiply primitives appear only in RDNA 4 / gfx1170+ and gfx1250.

### 2.6 Side-by-side comparison (RDNA 3.0 vs 3.5 vs 4 vs CDNA)

| Capability | CDNA3 (MFMA, wave64) | CDNA4 (MFMA, wave64, scaled FP4) | RDNA 3.0 gfx1100 | **RDNA 3.5 gfx1150** | RDNA 4 gfx1201 | gfx1250 |
|---|---|---|---|---|---|---|
| Native instr family | MFMA | MFMA + scale | WMMA | **WMMA** | WMMA + SWMMAC | WMMA + SWMMAC + TDM |
| Wave size | 64 | 64 | 32 | **32** | 32 | 32 |
| Encoding | VOP3P-like (gfx9-VOP3P) | VOP3P-like | VOP3P | **VOP3P** | VOP3P (`_gfx1170` re-encode) | VOP3P |
| F16 → F32 | ✅ | ✅ | ✅ (16x16x16) | **✅ (16x16x16)** | ✅ (16x16x16, **also 16x16x32**) | ✅ (more shapes) |
| BF16 → F32 | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ |
| F16 → F16 | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ |
| BF16 → BF16 | ✅ | ✅ | ✅ | **✅** | ✅ | ✅ |
| INT8 → I32 | ✅ | ✅ | ✅ (IU8) | **✅ (IU8)** | ✅ | ✅ |
| INT4 → I32 | ✅ | ✅ | ✅ (IU4) | **✅ (IU4)** | ✅ | ✅ |
| FP8 (E4M3/E5M2) | ✅ | ✅ | ❌ | **❌** | ✅ | ✅ |
| FP4 (E2M1) | ❌ | ✅ | ❌ | **❌** | ❌ | ✅ |
| Block-scale variants | ❌ | ✅ | ❌ | **❌** | ❌ | ✅ |
| Sparse (SWMMAC) | ❌ | ❌ | ❌ | **❌** | ✅ | ✅ |
| Inline-modifier on A/B | NEG only | NEG/scale | NEG (sign for IU) | **NEG (sign for IU)** | same as 3.5 | same |
| Operand replication required (lanes 0–15 → 16–31) | n/a (wave64, MFMA layout) | n/a | **YES** (`WMMA256bInsts`) | **YES** (`WMMA256bInsts`) | NO (`WMMA128bInsts`) | NO |

---

## 3. Buffer / global memory ops

### 3.1 Vectorized loads/stores (the ones a DSL backend cares about)

All listed in §16.16 (MUBUF, p. 573–) and §16.20 (FLAT/GLOBAL/SCRATCH, p. 614–644). The full set
of "useful for vector loads" mnemonics:

| Family | Mnemonics (DWORD vector loads/stores) | Page |
|---|---|---|
| `BUFFER` (MUBUF, indexed, uses V#) | `BUFFER_LOAD_{B32, B64, B96, B128}`, `BUFFER_STORE_{B32, B64, B96, B128}` | p. 84–85, p. 577 (§16.16) |
| `BUFFER` byte/short | `BUFFER_LOAD_{U8, I8, U16, I16}`, `BUFFER_LOAD_D16_B16`, `BUFFER_LOAD_D16_HI_B16`, `BUFFER_STORE_{B8, B16}`, `BUFFER_STORE_D16_HI_{B8, B16}` | p. 84–85 |
| `BUFFER` formatted | `BUFFER_LOAD_FORMAT_{X, XY, XYZ, XYZW}`, plus `BUFFER_LOAD_D16_FORMAT_*` and `BUFFER_LOAD_D16_HI_FORMAT_X` | p. 84–85 |
| `BUFFER` atomic | `BUFFER_ATOMIC_{ADD,SUB,AND,OR,XOR,MIN,MAX,SWAP,CMPSWAP,CSUB,INC,DEC}_{U32,U64,B32,B64,F32,I32,I64,U32,U64}` | p. 85 (§9.1), p. 587 (§16.16) |
| `GLOBAL` (vmem, no V#) | `GLOBAL_LOAD_{B32, B64, B96, B128}`, `GLOBAL_STORE_{B32, B64, B96, B128}` | p. 631–633 (§16.20.3) |
| `GLOBAL` byte/short/D16 | `GLOBAL_LOAD_{U8, I8, U16, I16}`, `GLOBAL_LOAD_D16_*`, `GLOBAL_STORE_D16_HI_*` | p. 633–634 |
| `GLOBAL` "ADDTID" | `GLOBAL_LOAD_ADDTID_B32`, `GLOBAL_STORE_ADDTID_B32` (no VGPR address; SGPR base + lane id × 4) | p. 116 (§11.1.2), p. 634 |
| `GLOBAL` atomic | `GLOBAL_ATOMIC_{SWAP,CMPSWAP,ADD,SUB,CSUB,MIN,MAX,AND,OR,XOR,INC,DEC}_{B32,B64,U32,U64,I32,I64,F32}` | p. 635–643 |
| `SCRATCH` | `SCRATCH_LOAD_{U8,I8,U16,I16,B32,B64,B96,B128}`, `SCRATCH_STORE_{B8,B16,B32,B64,B96,B128}`, plus `D16` variants | p. 626–630 (§16.20.2) |
| `FLAT` | `FLAT_LOAD/STORE_{B32,B64,B96,B128}` (and bytes/D16/atomics) — auto-routes by aperture | p. 614 (§16.20.1) |

> **No `BUFFER_LOAD_DWORDx{1,2,3,4}` mnemonics anymore** — RDNA 3+ renamed everything to
> `_B32 / _B64 / _B96 / _B128`. Previously gcn-style mnemonics like `BUFFER_LOAD_DWORDX4` are
> aliases in the assembler but the canonical form is `_B128`.

### 3.2 Cache control bits (p. 36–38, §4.1.1)

RDNA 3.5 uses the **legacy GLC/SLC/DLC three-bit scheme**, **NOT** the newer CPOL field that
RDNA 4 / CDNA4 introduce.

| Bit | Cache it controls | Behaviour |
|---|---|---|
| `GLC` | L0 (graphics first-level / texture-cache) | 0 = CU scope (work-group); 1 = DEVICE scope (forces L0 miss, reads from L2). For atomics: 0 = no return, 1 = return pre-op value. |
| `SLC` | L2 (graphics L2) | Temporal hint. 0 = LRU regular; 1 = STREAM non-temporal hint. |
| `DLC` | MALL / Infinity Cache (when present) | Temporal hint. 0 = regular; 1 = non-temporal. **Ignored on gfx1150** (no MALL). |

The complete shader-load policy table (p. 37) shows the full SRD.LLC_NOALLOC × DLC × SLC × GLC
matrix; for FlyDSL the relevant defaults are:

* Plain vector load — `GLC=0, SLC=0, DLC=0` (CU scope, LRU all caches)
* Streaming load (e.g. K-loop GEMM input) — `GLC=0, SLC=1, DLC=0` (CU scope, non-temporal at L2)
* Coherent load (last-iteration acquire) — `GLC=1, SLC=0, DLC=0` (DEVICE scope)
* Plain store — `SLC=0, DLC=0` (always device scope)

**Note for codegen:** there is no separate "scope=AGENT" or "scope=SYSTEM" semantic in the ISA;
agent vs device coherency is handled by the L2 (`SLC=0/1`). RDNA 3.5 has no `s_inv_l2` style
scalar instruction — use `BUFFER_GL0_INV` (L0 invalidate) and `BUFFER_GL1_INV` (L1 invalidate)
explicitly (p. 85). `S_GL1_INV` and `S_DCACHE_INV` exist as SMEM ops (p. 78–79, 265).

### 3.3 Buffer addressing and resource constants (p. 89–93, §9.4)

Buffer instructions take a 128-bit V# (4 SGPRs, aligned to a 4-SGPR boundary in `SRSRC[4:0]`)
plus optional VGPR(`VADDR`)/SGPR(`SOFFSET`) offsets. `IDXEN`/`OFFEN` flags select index/offset
modes. The PDF gives a final-address formula and bounds-checking rules at p. 89–93. **This is the
same V# layout as RDNA 3.0**, so any FlyDSL CDNA/RDNA 4 buffer-resource construction code reuses
1:1.

Alignment rules (p. 93, §9.5): natural alignment for `B16`/`B32` etc.; misaligned access returns
`MEMVIOL`. (CDNA's "misaligned-but-slow" is **not** allowed here.)

---

## 4. LDS ops

### 4.1 The big finding: NO `DS_LOAD_TR_*` in RDNA 3.5

**There is no transpose-load instruction on gfx1150**. Manually verified by reading every opcode
in §16.15 (LDS & GDS Instructions, p. 535–571): the LDS opcode space goes 0–179 + 222–223 +
254–255, with no `_TR` mnemonic. The summary table on p. 128–129 (§12.5) also lists every LDS op
and contains no transpose load.

This is a critical gap vs gfx1250: on gfx1250 the WMMA-feeding fast path is
`ds_load_tr16_b64 / ds_load_tr8_b64 / ds_load_tr_b6` to put data into the right lane layout in
one shot. On gfx1150 you need either:
1. **Bank-conflict-friendly LDS layout + plain `DS_LOAD_B64/B128`** with the matrix-A duplication
   done in registers (`v_dual_mov_b32` from VGPR to VGPR), or
2. **`DS_BPERMUTE_B32` post-load** to shuffle the elements into the lane positions WMMA expects.

Strategy (1) is what RDNA 3.0 GEMM kernels use; FlyDSL's existing RDNA 3.0 code path (if any)
maps directly. No FlyDSL "LDS_TR atom" wrapper can be lowered on gfx1150 — for this target the
lowering has to fall back to plain `DS_LOAD_B*` + register fix-up.

### 4.2 The full LDS op set you can rely on

| Family | Mnemonics | OP # range | Page |
|---|---|---|---|
| Plain DWORD/qword load | `DS_LOAD_B32` (54), `DS_LOAD_B64` (118), `DS_LOAD_B96` (254), `DS_LOAD_B128` (255) | 54, 118, 254, 255 | p. 549, 562, 571 |
| Two-address loads | `DS_LOAD_2ADDR_{B32, B64}` (55, 119), `DS_LOAD_2ADDR_STRIDE64_{B32, B64}` (56, 120) | 55, 56, 119, 120 | p. 549, 562 |
| Byte/short load | `DS_LOAD_{U8, I8, U16, I16}` (57–60), `DS_LOAD_{U8, I8, U16}_D16`/`_D16_HI` (162–167) | 57–60, 162–167 | p. 550, 566 |
| Plain store | `DS_STORE_B32` (13), `DS_STORE_B64` (77), `DS_STORE_B96` (222), `DS_STORE_B128` (223) | 13, 77, 222, 223 | p. 537, 555, 571 |
| Two-address store | `DS_STORE_2ADDR_{B32, B64}` (14, 78), `DS_STORE_2ADDR_STRIDE64_{B32, B64}` (15, 79) | 14, 15, 78, 79 | p. 538, 555 |
| Byte/short store | `DS_STORE_B8` (30), `DS_STORE_B16` (31), `DS_STORE_B8_D16_HI` (160), `DS_STORE_B16_D16_HI` (161) | 30, 31, 160, 161 | p. 540, 565 |
| Lane permute (cross-thread shuffle, no LDS storage) | `DS_PERMUTE_B32` (178), `DS_BPERMUTE_B32` (179), `DS_SWIZZLE_B32` (53) | 53, 178, 179 | p. 569, 547, 130 |
| Add-thread-id base | `DS_LOAD_ADDTID_B32` (177), `DS_STORE_ADDTID_B32` (176) | 176, 177 | p. 568 |
| Atomics (32-bit) | `DS_{ADD,SUB,RSUB,INC,DEC,MIN,MAX,AND,OR,XOR,MSKOR,CMPSTORE}_{U32,I32,F32,B32}` and `_RTN` variants | 0–52 + 32–52 | p. 130, 537–545 |
| Atomics (64-bit) | `DS_{ADD,SUB,...}_{U64,I64,F64,B64}` + `_RTN` | 64–115 | p. 553–561 |
| Atomic FP32 add | `DS_ADD_F32` (21), `DS_ADD_RTN_F32` (121) | 21, 121 | p. 540, 562 |
| Min/max FP | `DS_MIN_F32`/`MAX_F32` (18, 19), and `_F64` | 18, 19, 82, 83 | p. 539, 556 |
| BVH stack (ray tracing) | `DS_BVH_STACK_RTN_B32` (173) | 173 | p. 567 |
| Append/Consume | `DS_APPEND` (62), `DS_CONSUME` (61) | 61, 62 | p. 550 |
| Swap (XCHG with return) | `DS_STOREXCHG_RTN_{B32,B64}`, `DS_STOREXCHG_2ADDR_RTN_*`, `DS_STOREXCHG_2ADDR_STRIDE64_RTN_*` | 45–47, 109–111 | p. 544, 560 |

Notably:
* `DS_LOAD_2ADDR_*` lets you fetch two values in one instruction (same wave, different LDS
  addresses) — this is the closest thing to a wider-than-128b LDS load on gfx1150.
* `DS_LOAD_2ADDR_STRIDE64_*` multiplies offset by 64 elements (256B for B32, 512B for B64),
  useful for loading two rows separated by one matrix tile width.
* `DS_LOAD_B96` and `DS_LOAD_B128` exist (opcodes 254, 255) and load 12/16 bytes per lane —
  these are how you get 128-bit-per-lane LDS reads.

### 4.3 Cross-thread shuffle ops (the workhorse for matrix layout swizzling)

`DS_PERMUTE_B32` (forward / scatter) and `DS_BPERMUTE_B32` (backward / gather) operate over 32
lanes regardless of wave size (i.e. each half-wave acts as an independent 32-lane permute in
wave64). They use LDS hardware but allocate no LDS storage (p. 569–570 §16.15).

* Address is `lane_id * 4` (scaled to byte units).
* For `BPERMUTE`: each lane reads from `tmp[i] = VGPR[src_lane][DATA0]` where `src_lane` comes
  from this lane's `ADDR + OFFSET`, gated by `EXEC[src_lane]`.
* For `PERMUTE`: each lane writes to `tmp[dst_lane]`. If multiple sources hit the same dst, the
  highest-numbered active lane wins.

`DS_SWIZZLE_B32` (op 53, p. 547) provides predefined swizzle modes (FFT-decomposition, rotate,
full-share-within-4, limited-share-within-32) using its 16-bit `OFFSET` field for control. Useful
for matrix-element reordering without a software lane-id table.

### 4.4 LDS bank-conflict model (p. 120, §12.1)

* **64 banks total per WGP**, each bank = 32-bit (DWORD) wide × 512 entries × 1R/1W/clk.
* The 64 banks are split into **two 32-bank halves**, each half affiliated with one CU
  (SIMD32 pair).
* In **CU mode** a wave only sees 32 banks (its half) and can use 32 KiB of LDS effectively per
  wave.
* In **WGP mode** all 64 banks are visible to all waves; a single wave's 32 lanes still
  bank-conflict against 32 banks (DWORD addresses `0..31` map to banks `0..31`, `32..63` to banks
  `0..31`, etc.), so the **effective bank-conflict model for a single wave32 instruction is
  "32 banks, DWORD-stride"** — same as RDNA 3.0.

For GEMM tile codegen this means the standard XOR-swizzle (e.g.
`bank = (col ^ (row >> 1)) & 31`) you already use on gfx1100 works as-is. **Do not assume
`ds_load_tr` style "no-conflict transpose" on gfx1150.**

### 4.5 LDS atomics relevant to GEMM

* FP32 atomics: `DS_ADD_F32` / `DS_ADD_RTN_F32`, `DS_MIN_F32` / `DS_MAX_F32` (and 64-bit
  equivalents) — useful for accumulator/reduction patterns.
* INT32/INT64 atomics — full set
  (`ADD/SUB/AND/OR/XOR/MIN/MAX/CMPSTORE/STOREXCHG/INC/DEC`).
* `DS_APPEND` / `DS_CONSUME` for ring buffers.
* No FP16/BF16 native LDS atomics (use cmpstore loop or `DS_ADD_F32` after promotion).

---

## 5. Synchronization & pipeline

### 5.1 `s_waitcnt` family (p. 44, §5.6; p. 252–254 §16.5; p. 212–214 §16.2)

Four counters, all visible to the shader:

| Counter | Width | Tracks | Decrement timing |
|---|---|---|---|
| `VMcnt` | 6 bits (0..63) | VMEM **loads + samples + atomic-with-return** — i.e. anything that returns data to VGPRs | Decrements when load returns |
| `VScnt` | 6 bits | VMEM **stores + atomic-without-return** | Decrements when store completes |
| `LGKMcnt` | 6 bits | LDS + GDS + scalar memory (SMEM) + messages + FLAT (in addition to VM/VScnt) | Decrements when each completes |
| `EXPcnt` | 3 bits | Exports + LDS_param_load + LDS_direct_load | Decrements when complete |

* `S_WAITCNT` (combined, SOPP op 9, p. 252): SIMM16 packs
  `{VMcnt[5:0], LGKMcnt[5:0], 1'b0, EXPcnt[2:0]}`.
* `S_WAITCNT_VMCNT` (SOPK op 25, p. 212), `S_WAITCNT_VSCNT` (24), `S_WAITCNT_LGKMCNT` (27),
  `S_WAITCNT_EXPCNT` (26) — **separate per-counter waits exist** (this matches RDNA 3.0 / RDNA 4;
  it's CPOL / sync.scope logic that differs between generations, not the wait-counter mnemonics
  themselves).
* `S_WAIT_IDLE` (SOPP op 10, p. 254): "wait for all activity in the wave to complete" — the
  big-hammer barrier for kernel exit.
* `S_WAIT_EVENT` (SOPP op 11, p. 254): wait for export-ready event.

**No `s_wait_dscnt`, `s_wait_storecnt`, `s_wait_loadcnt`** etc. — those are RDNA 4 / gfx1201+
generation-specific (they break LGKMcnt into finer counters). On RDNA 3.5 you get one merged
LGKMcnt and the VM/VS split.

### 5.2 Barriers (p. 43, §5.5; p. 261, §16.5)

**Single barrier opcode**: `S_BARRIER` (SOPP op 61, p. 261).
* Synchronizes all waves of a work-group.
* Does **not** drain memory counters automatically — you must issue `S_WAITCNT 0` (or per-counter
  waits) **before** the barrier if the barrier protects a memory operation (PDF explicit warning:
  "Barrier instructions do not wait for any counters to go to zero before issuing", p. 261).
* Single-wave work-groups treat `S_BARRIER` as `S_NOP`.

**No split-barrier / async-barrier model on RDNA 3.5.** The PDF contains no `S_BARRIER_INIT`,
`S_BARRIER_SIGNAL`, `S_BARRIER_WAIT`, `S_BARRIER_LEAVE`, etc. — these are RDNA 4 / gfx1201
additions.

### 5.3 `S_DELAY_ALU` (p. 45–46 §5.7; p. 252–253 §16.5)

Inserts software-scheduled delay between dependent ALU instructions. Same encoding as RDNA 3.0:

```text
S_DELAY_ALU instid0(<dep>) | instskip(<n>) | instid1(<dep>)
```

Where `instid0` / `instid1` is one of:
* `INSTID_NO_DEP` (0)
* `INSTID_VALU_DEP_1..4` (1–4) — depends on Nth previous VALU instruction
* `INSTID_TRANS32_DEP_1..3` (5–7) — depends on Nth previous transcendental
  (`v_rcp`, `v_log`, `v_exp`, `v_rsq`, `v_sqrt`, `v_sin`, `v_cos`)
* `INSTID_FMA_ACCUM_CYCLE_1` (8) — single-cycle FMA accumulator penalty
* `INSTID_SALU_CYCLE_1..3` (9–11) — wait 1/2/3 cycles for prior SALU
* `INSTSKIP_SAME` (0), `INSTSKIP_NEXT` (1), `INSTSKIP_SKIP_1..4` (2–5) — distance to second
  instruction

For FlyDSL codegen: insert `S_DELAY_ALU` to hide back-to-back transcendentals or VALU
producer→consumer; **not required for correctness** but materially affects perf.

### 5.4 Other relevant scheduler ops

* `S_NOP <0..15>` (SOPP op 0, p. 249): insert 0–15 wait states.
* `S_SLEEP <0..127>` (op 3, p. 250): sleep 64*N..64*(N+1) clocks (poll loops).
* `S_SETPRIO 0..3` (op 53, p. 260): set wave priority (low to high).
* `S_CLAUSE` (op 5, p. 250): wraps 2–63 same-type instructions and locks the arbiter onto this
  wave for the clause. Clause types: VALU, SMEM, BUFFER/GLOBAL/SCRATCH load/store/atomic, FLAT
  load/store/atomic, IMAGE load/sample/store/atomic, IMAGE_BVH, **LDS** (loads/stores/atomics/
  bvh_stack can mix in one LDS clause). `S_DELAY_ALU` is illegal **inside** a VALU clause
  (p. 45).

### 5.5 Hard-coded NOPs around WMMA

Recap from §2.4: 1 V_NOP or independent VALU instruction is required between two WMMAs **only**
when the first's D matrix overlaps the second's A or B. The typical accumulator chain
`D = D + A·B` repeats with no NOPs. There is **no** "VALU-after-WMMA" NOP requirement — the PDF
only flags potential perf stalls (not correctness).

---

## 6. Scalar / vector ALU notes for DSL codegen

### 6.1 Inline literals and modifiers

* Inline constants: integers `-16..64` plus the 8 special FP values
  (`±0.5, ±1.0, ±2.0, ±4.0, 1/(2π)`) — same as RDNA 3.0 (p. 36, §4.1). Encoded with operand index
  `128–248`.
* One literal constant per VOP3 / VOP3P instruction; literals not allowed with DPP
  (p. 60–61, §7.2.2.3).
* WMMA accepts inline constants **only on C** (p. 75, §7.9).
* For F16/BF16 packed math, inline constants replicate low→high (p. 69, §7.5.1) — the OPSEL bit
  can be used to override and place the constant in only one half.

`V_FMAC_F32`, `V_FMAC_F16`, `V_PK_FMAC_F16` etc. all support standard NEG/ABS modifiers when
promoted to VOP3 encoding (p. 56, §7.1).

### 6.2 VOPD (dual-issue) — wave32 only (p. 70, §7.6; p. 384–388 §16.11)

VOPD encodes two VALU operations into one 64-bit instruction word. **Wave32 only** — VOPD is
skipped (treated as no-op? — actually "must not be used by wave64; it is skipped for wave64" per
p. 70).

| Slot | Allowed mnemonics (RDNA 3.5) |
|---|---|
| **OpcodeX** (4-bit op field) | `V_DUAL_FMAC_F32, FMAAK_F32, FMAMK_F32, MUL_F32, ADD_F32, SUB_F32, SUBREV_F32, MUL_DX9_ZERO_F32, MOV_B32, CNDMASK_B32, MAX_F32, MIN_F32, DOT2ACC_F32_F16, DOT2ACC_F32_BF16` (p. 384–385) |
| **OpcodeY** (5-bit op field) | All of OpcodeX **+ `V_DUAL_ADD_NC_U32`, `V_DUAL_LSHLREV_B32`, `V_DUAL_AND_B32`** (p. 386–388) |

Restrictions (p. 70, §7.6):
* Each instruction may use up to 2 VGPRs.
* At most 1 SGPR or 1 literal per instruction; combined the pair may use at most 2 SGPRs or
  1 SGPR + 1 literal, or share 1 literal.
* SRC0 may be VGPR/SGPR/constant; VSRC1 must be VGPR.
* **VGPR bank conflicts forbidden:** there are 4 VGPR banks (indexed by `SRC[1:0]`), each with 3
  read ports (one each for SRC0/1/2). `SRCX0` and `SRCY0` must use different banks; `VSRCX1` and
  `VSRCY1` must use different banks. Both X and Y reading SRC2 must split SRC2 even/odd.
* Destination VGPRs: one even, one odd.
* The two instructions must be data-independent.
* No DPP allowed.

For FlyDSL: the dual-issue patterns to target on gfx1150 are typically
`V_DUAL_FMAC_F32 + V_DUAL_FMAC_F32` (two independent FMAs),
`V_DUAL_FMAC_F32 + V_DUAL_ADD_NC_U32` (compute + index update),
`V_DUAL_DOT2ACC_F32_F16 + V_DUAL_DOT2ACC_F32_F16` (back-to-back dot accumulations). Compiler must
verify the bank/literal/SGPR constraints before emitting.

### 6.3 BF16 native ALU support

Limited. The opcodes that natively consume/produce BF16 in RDNA 3.5 are:
* `V_DOT2_F32_BF16` (VOP3P op 26, p. 379) — dot product with F32 accumulation.
* `V_DOT2ACC_F32_BF16` (VOP2/VOP3SD op 2, p. 267) and the dual-issue version
  `V_DUAL_DOT2ACC_F32_BF16` (p. 387).
* `V_WMMA_F32_16X16X16_BF16` and `V_WMMA_BF16_16X16X16_BF16` — already covered.

There is **no** `V_PK_ADD_BF16` / `V_PK_MUL_BF16` / `V_PK_FMA_BF16` packed BF16 ALU — for non-WMMA
BF16 arithmetic you must convert to F32, do the math, convert back. The conversion ops are
`v_cvt_pk_*` (p. 282–283 §16.7.1; full V_CVT_* list in §7.3 p. 65–66 and §16.8).

### 6.4 FP8 support — none

There are **no native FP8 (E4M3 / E5M2) conversion or arithmetic ops** in RDNA 3.5. Searching the
V_CVT_* list (p. 283–290) and the VOP3P list (p. 373–383) confirms this. Any FP8 software path
must materialise via byte-level packing/unpacking + V_CVT through F32.

### 6.5 SALU floating-point — new in RDNA 3.5 vs gfx1100

RDNA 3.5 adds scalar FP arithmetic to the SALU pipe (which on gfx1100 was integer-only):

* `S_ADD_F32, S_SUB_F32, S_MUL_F32, S_FMAC_F32, S_FMAAK_F32, S_FMAMK_F32, S_MIN_F32, S_MAX_F32,
  S_CEIL_F32, S_FLOOR_F32, S_TRUNC_F32, S_RNDNE_F32` (p. 53–54 §6.8; p. 235–236 §16.3; p. 244
  §16.4)
* `S_CMP_{LT,EQ,LE,GT,LG,GE,O,U,N*}_{F32,F16}` (p. 243–248 §16.4)
* `S_CVT_F32_I32, S_CVT_F32_U32, S_CVT_I32_F32, S_CVT_U32_F32, S_CVT_F16_F32, S_CVT_F32_F16,
  S_CVT_HI_F32_F16, S_CVT_PK_RTZ_F16_F32` (p. 53–54)
* `S_ADD_F16, S_SUB_F16, S_MUL_F16, S_FMAC_F16, S_MIN_F16, S_MAX_F16, S_CEIL_F16, S_FLOOR_F16,
  S_TRUNC_F16, S_RNDNE_F16` (p. 53–54; p. 237–238)

These are useful for moving uniform floating-point computations off the VALU pipe (e.g. address
arithmetic that involves a floating-point multiply, or a uniform softmax max-reduction tail).
LLVM exposes them via `s_*` builtins; FlyDSL's CSE + uniformity analysis can pick them up where
applicable.

### 6.6 V_DOT* (small-matrix workhorses)

Documented at p. 377–379 (§16.10). All present in RDNA 3.5:
* `V_DOT2_F32_F16`, `V_DOT2_F32_BF16` (ops 19, 26)
* `V_DOT4_I32_IU8`, `V_DOT4_U32_U8` (ops 22, 23)
* `V_DOT8_I32_IU4`, `V_DOT8_U32_U4` (ops 24, 25)
* `V_DOT2ACC_F32_F16`, `V_DOT2ACC_F32_BF16` (VOP2 ops 2)

These are smaller, 32-bit-VGPR-per-lane versions of WMMA — useful for short K dimensions or
per-thread "mini-MMA". The `V_FMA_MIX_*` family (`V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`,
`V_FMA_MIXHI_F16`, ops 32–34, p. 379–380) does mixed-precision (F16/F32) FMA.

---

## 7. Differences vs RDNA 4 / gfx1250 that block direct reuse

This section answers the explicit question: *"if I have WMMA / TDM / LDS_TR working on gfx1250,
what specifically would break or be missing on gfx1150?"*

### 7.1 Hard blockers (no workaround possible at the ISA level)

| Capability | gfx1250 has | gfx1150 status | Workaround on gfx1150 |
|---|---|---|---|
| **TDM (Tensor Direct-to-Memory)** | yes | **NOT PRESENT** | none — TDM is a wholly new instruction class. Use the regular `BUFFER_LOAD` → register → `DS_STORE` → barrier → `DS_LOAD` pipeline. |
| **`ds_load_tr16_b64` / `ds_load_tr8_b64` / `ds_load_tr_b6` (transpose load)** | yes | **NOT PRESENT** (verified by full opcode walk of §16.15, p. 535–571) | Use `DS_LOAD_B64`/`DS_LOAD_B128` + register-side shuffle (`v_dual_mov_b32`, `ds_bpermute_b32`, or layout in LDS pre-arranged so the natural load already produces the WMMA layout). |
| **WMMA FP8** (`V_WMMA_F32_16X16X32_FP8_FP8`, `BF8_BF8`, etc.) | yes (and 16x16x32 K-shape) | **NOT PRESENT** (only F16/BF16/IU8/IU4 16x16x16) | Pack/unpack FP8 in software through F16 (cost: one `V_CVT_F32_FP8`-equivalent missing → must do via byte mask + `V_CVT_F32_U32` + integer scaling). |
| **WMMA FP4** (`V_WMMA_*_FP4`) | yes (gfx1250 only) | **NOT PRESENT** | Software unpack to FP16/FP8. |
| **Scaled / block-scale WMMA** (`V_WMMA_*_SCALED_*`) | yes (gfx1250) | **NOT PRESENT** | Manual per-block scale multiply on accumulator. |
| **Sparse (V_SWMMAC)** | yes (gfx1170+ / gfx1201+) | **NOT PRESENT** | Densify / drop sparsity pass at compile time. |
| **Split barriers (`S_BARRIER_INIT/SIGNAL/WAIT`)** | yes | **NOT PRESENT** | Use plain `S_BARRIER` + manual `S_WAITCNT 0` ordering. Loses the latency-hiding benefit of split-barrier. |
| **CPOL field (unified cache policy)** | yes | NOT PRESENT — uses legacy GLC/SLC/DLC | Translate CPOL bits to the GLC/SLC/DLC trio per p. 36–38 cache-policy table. `DLC` is ignored on gfx1150 (no MALL). |
| **Async copy / cp.async-equivalent (global → LDS direct)** | (TDM provides this on gfx1250; no first-class on gfx1201) | **NOT PRESENT** | The classic "load-to-VGPR, store-to-LDS, sync, load-from-LDS-to-VGPR" pipeline is mandatory. |
| **K-shape 16x16x32 WMMA** | yes | **NOT PRESENT** (only K=16) | Issue 2 × WMMA(K=16) for K=32. |

### 7.2 Soft blockers (different encoding, same semantics)

| Capability | gfx1250 / RDNA 4 form | gfx1150 form | Notes |
|---|---|---|---|
| WMMA VGPR layout | each lane holds 8 elements (no duplication, "WMMA128bInsts") | each lane holds 8 elements **duplicated into the upper half-wave** ("WMMA256bInsts"; lanes 0-15 → 16-31 in wave32) | When porting, you must re-introduce the duplication step. The simplest is `v_dual_mov_b32 v_high, v_low` after the LDS load, or arrange the LDS layout so a `DS_LOAD_2ADDR_STRIDE64` already produces the duplicated form. (p. 75 §7.9) |
| Memory wait counters | finer-grained `loadcnt/storecnt/dscnt/kmcnt/...` | coarser `vmcnt/vscnt/lgkmcnt/expcnt` | Map your finer-grained waits into the LGKMcnt/VMcnt/VScnt buckets; potentially over-conservative. |
| WMMA opcode binary | re-encoded `_gfx1170` versions | original gfx11 encoding (op 64–69 in VOP3P) | LLVM picks the correct encoding via subtarget; from FlyDSL you reference the `llvm.amdgcn.wmma.*` (no `_gfx12` suffix) intrinsics. |

### 7.3 What's actually portable from a gfx1250 backend

A FlyDSL backend that already works on gfx1250 (per the user's statement) has working machinery
for:

* wave32 abstractions
* WMMA atom dispatch (M=N=K=16, F16/BF16/I8 dtypes)
* tiled `BUFFER_LOAD` / `DS_LOAD` via raw buffer/DS ops
* `s_waitcnt` insertion
* V_DOT*, V_FMA*, V_CVT* lowerings

This whole stack works on gfx1150 with **only two changes**:
1. Strip out the `ds_load_tr_*`, FP8/FP4/scaled/sparse, TDM, and split-barrier paths (i.e. don't
   emit them).
2. Add the **A/B matrix half-wave duplication** before WMMA (the "lanes 0–15 → 16–31" step from
   p. 75).

If your existing FlyDSL-RDNA 4 path already supports a "no-tr / no-fp8" subset for gfx1170-style
hardware that lacks gfx1250 specifics, that subset is essentially equivalent to gfx1150 — you
only have to add the duplication step and back off WMMA dtypes to the {F16, BF16, IU8, IU4} set.

### 7.4 Concrete intrinsic mapping (LLVM `llvm.amdgcn.*`)

For ROCDL / FlyDSL backends targeting `--mcpu=gfx1150`:

| Op | LLVM intrinsic | Notes |
|---|---|---|
| `V_WMMA_F32_16X16X16_F16` | `llvm.amdgcn.wmma.f32.16x16x16.f16` (with `i1 0` / `i1 1` for matrix layout / packed) | gfx11 family form; **not** the `_gfx12` form |
| `V_WMMA_F32_16X16X16_BF16` | `llvm.amdgcn.wmma.f32.16x16x16.bf16` | |
| `V_WMMA_F16_16X16X16_F16` | `llvm.amdgcn.wmma.f16.16x16x16.f16` | The C/D bit `OpSel[2]` controls hi/lo packing |
| `V_WMMA_BF16_16X16X16_BF16` | `llvm.amdgcn.wmma.bf16.16x16x16.bf16` | |
| `V_WMMA_I32_16X16X16_IU8` | `llvm.amdgcn.wmma.i32.16x16x16.iu8` (signed/unsigned via metadata args) | |
| `V_WMMA_I32_16X16X16_IU4` | `llvm.amdgcn.wmma.i32.16x16x16.iu4` | |
| `BUFFER_LOAD_B{32,64,96,128}` | `llvm.amdgcn.raw.ptr.buffer.load.{i32,v2i32,v3i32,v4i32}` (or `raw.buffer.load.*` on older) | uses V# resource |
| `GLOBAL_LOAD_B{32,64,96,128}` | LLVM emits via `load <ty>* addrspace(1)*` with appropriate alignment | |
| `DS_LOAD_B{32,64,96,128}` | `load <ty>* addrspace(3)*` | |
| `DS_BPERMUTE_B32` | `llvm.amdgcn.ds.bpermute` | |
| `DS_PERMUTE_B32` | `llvm.amdgcn.ds.permute` | |
| `S_BARRIER` | `llvm.amdgcn.s.barrier` | single barrier; no init/signal/wait variants |
| `S_WAITCNT*` | `llvm.amdgcn.s.waitcnt` family | |
| `S_DELAY_ALU` | `llvm.amdgcn.s.delay.alu` | |

Verify against the AMD LLVM source (`AMDGPU.td`, `AMDGPUUsage.html`) at the time of bring-up —
intrinsic names occasionally rename across LLVM versions, but the mapping shape is stable.

---

## 8. Summary: what FlyDSL needs for gfx1150 support

1. **Add a new backend dialect target** `FlyAMDGPU::RDNA35` (or fold into an existing RDNA
   dialect with a `gfx1150` subtarget flag).
2. **Reuse the existing RDNA 4 wave32 + buffer-load + DS-load infrastructure** — the
   per-thread/per-wave abstraction, raw buffer ops (`buffer_load_*`), raw DS ops
   (`ds_load_b*`, `ds_store_b*`), and `s_waitcnt` insertion all work unchanged.
3. **Restrict WMMA atom dispatch** to the 6 RDNA 3.5 dtypes: F16, BF16, IU8, IU4 — refuse to
   lower FP8/FP4/scaled/sparse atoms when targeting gfx1150 and emit a clear error pointing to
   "use FlyROCDL/CDNA4 or gfx1250 instead".
4. **Implement A/B matrix duplication** (`lanes 0-15 → 16-31`) in the WMMA copy-in op for
   gfx1150. Recommend: do it in LDS layout (cheaper) rather than via `v_dual_mov_b32` after load
   (more VGPR pressure).
5. **Drop any `ds_load_tr_*` lowering paths**; route those layouts through plain
   `DS_LOAD_B64/B128` + LDS-side swizzle or `DS_BPERMUTE_B32`.
6. **Drop any TDM / async-copy / split-barrier emission**; fall back to BUFFER_LOAD → register →
   DS_STORE + S_WAITCNT 0 + S_BARRIER.
7. **Keep `S_DELAY_ALU` insertion** (same encoding as RDNA 3 / 4).
8. **Keep VOPD dual-issue** (wave32-only, same op set as gfx1100/gfx1201 minus a few additions).
9. **Use legacy GLC/SLC/DLC cache hints** (CPOL not yet in this generation); `DLC=0` always
   (no MALL on Strix Point).
10. **Keep the V_DOT*, V_FMA_MIX_*, V_PK_*** ALU paths exactly as gfx1100 — they are unchanged.

After those edits, a FlyDSL kernel written against the RDNA 4 / gfx1250 wave32 path should
compile, link, and run on gfx1150 — at the cost of losing access to FP8/FP4/scaled/sparse/TDM/
`ds_load_tr` performance shortcuts. For a pure F16/BF16/IU8 GEMM, the resulting code path is
functionally identical to the RDNA 3.0 (gfx1100) backend, so any FlyDSL gfx1100 kernel is a
known-good reference for testing.

---

## Appendix A — PDF section index used in this report

* §1 Introduction (p. 3–8)
* §2 Shader Concepts — wave32/64, work-groups (p. 9–12)
* §3 Wave State (p. 13–34)
* §4.1.1 Cache Controls: SLC, GLC, DLC (p. 36–38)
* §5.5 Work-groups & Barriers (p. 43)
* §5.6 Data Dependency Resolution (s_waitcnt) (p. 44)
* §5.7 ALU Instruction Software Scheduling (s_delay_alu) (p. 45–46)
* §6.8 SALU Floating Point (p. 53–54)
* §7.5 Packed Math, VOP3P field layout (p. 67–69)
* §7.6 Dual Issue VALU (VOPD) (p. 70)
* §7.9 Wave Matrix Multiply Accumulate (WMMA) (p. 74–77)
* §9 Vector Memory Buffer Instructions (p. 82–93)
* §11 Global, Scratch, Flat (p. 113–119)
* §12 Data Share Operations (p. 120–133)
* §16.5 SOPP Instructions — `S_BARRIER`, `S_WAITCNT`, `S_DELAY_ALU`, `S_CLAUSE` (p. 249–262)
* §16.10 VOP3P Instructions — including WMMA opcodes (p. 373–383)
* §16.11 VOPD Instructions (p. 384–388)
* §16.15 LDS & GDS Instructions — full DS opcode listing (p. 535–571) — verified to contain no
  `DS_LOAD_TR_*`
* §16.16 MUBUF Instructions — buffer loads/stores/atomics (p. 573–590)
* §16.20 FLAT, Scratch and Global Instructions (p. 614–644)
