#!/usr/bin/env python3
"""WMMA GEMM kernel for RDNA4 (gfx120x, wave32).

4-warp LDS kernel inspired by Triton's 93 TFLOPS approach.

Architecture:
- 128x128x32 tiles, 4 warps (128 threads), 2x2 warp layout
- Each warp: 4 M-repeats x 4 N-repeats (64x64 output per warp)
- 2 K-steps per iteration (K=32, WMMA_K=16) -> 32 WMMAs per iter
- Double-buffered LDS (ping-pong): compute from buf[cur], prefetch to buf[1-cur]
- A[M,K] row-major GMEM, B_T[N,K] row-major GMEM
- K-padding on LDS stores for bank conflict avoidance

LDS layout (per buffer):
  A tile: 128 rows x (32+pad) cols x 2B, stored row-major
  B tile: 128 rows x (32+pad) cols x 2B, stored row-major
  Total per buffer: ~20KB, double-buffered: ~40KB

Pipeline: split GMEM load / LDS store with double buffering

Computes C[M,N] = A[M,K] @ B_T[N,K]^T
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.ir import InsertionPoint
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def create_wmma_gemm_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    reg_m=None,  # M-repeats per warp
    reg_n=None,  # N-repeats per warp
    reg_k=None,  # K-steps per tile (32/16=2)
    waves_m=None,  # warps in M dimension
    waves_n=None,  # warps in N dimension
    group_m=None,
    a_k_pad=None,  # K-padding for A in LDS (bank conflict avoidance)
    b_k_pad=None,  # K-padding for B in LDS
    use_native_bf16_acc=False,  # Experimental path; disabled by default.
):
    gpu_arch = get_rocm_arch()

    # Generic fallback keeps existing behavior.
    defaults = {
        "reg_m": 4,
        "reg_n": 4,
        "reg_k": 2,
        "waves_m": 2,
        "waves_n": 2,
        "group_m": 8,
        "a_k_pad": 8,
        "b_k_pad": 8,
    }
    if gpu_arch.startswith(("gfx115", "gfx110")):
        # RDNA3/3.5 baseline keeps the mature 128x128x32 tile.
        defaults.update(
            {
                "reg_m": 4,
                "reg_n": 4,
                "reg_k": 2,
                "waves_m": 2,
                "waves_n": 2,
                "group_m": 8,
                "a_k_pad": 8,
                "b_k_pad": 8,
            }
        )
        # For medium square tiles (e.g. 1024x1024), remap waves to 4x1 and
        # increase group swizzle for better locality/launch efficiency.
        if (
            M * N >= 1024 * 1024
            and M <= 1024
            and N <= 1024
            and M % (WMMA_M * 4 * 4) == 0
            and N % (WMMA_N * 4 * 1) == 0
        ):
            defaults.update(
                {
                    "waves_m": 4,
                    "waves_n": 1,
                    "group_m": 16,
                }
            )

    reg_m = defaults["reg_m"] if reg_m is None else reg_m
    reg_n = defaults["reg_n"] if reg_n is None else reg_n
    reg_k = defaults["reg_k"] if reg_k is None else reg_k
    waves_m = defaults["waves_m"] if waves_m is None else waves_m
    waves_n = defaults["waves_n"] if waves_n is None else waves_n
    group_m = defaults["group_m"] if group_m is None else group_m
    a_k_pad = defaults["a_k_pad"] if a_k_pad is None else a_k_pad
    b_k_pad = defaults["b_k_pad"] if b_k_pad is None else b_k_pad

    BLOCK_M = WMMA_M * reg_m * waves_m  # 16*4*2 = 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 16*4*2 = 128
    BLOCK_K = WMMA_K * reg_k  # 16*2 = 32
    NUM_WAVES = waves_m * waves_n  # 2*2 = 4
    WAVE_SIZE = 32
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    assert reg_k >= 2 and reg_k % 2 == 0

    # Loading: each thread loads 8 bf16 elements per load (128 bits = buffer_load_b128)
    LOAD_VEC = 8
    A_TILE_ELEMS = BLOCK_M * BLOCK_K  # 128*32 = 4096
    NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 4096/(128*8) = 4
    B_TILE_ELEMS = BLOCK_N * BLOCK_K  # 128*32 = 4096
    NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * LOAD_VEC)  # 4

    # LDS layout with K-padding for bank conflict avoidance
    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40
    LDS_A_SIZE = BLOCK_M * BLOCK_K_PAD_A  # 128*40 = 5120 elements
    LDS_B_SIZE = BLOCK_N * BLOCK_K_PAD_B  # 128*40 = 5120 elements
    LDS_ONE_BUF = LDS_A_SIZE + LDS_B_SIZE  # 10240 elements = 20KB
    LDS_TOTAL = 2 * LDS_ONE_BUF  # 20480 elements = 40KB

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles >= 2, "Need at least 2 K-tiles for prefetch pipeline"

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"
    use_native_bf16_acc = bool(use_native_bf16_acc) and in_dtype == "bf16" and out_dtype == "bf16"
    out_elem_bytes = 4 if out_dtype == "f32" else 2

    # Workgroup-barrier asm. RDNA 4 / gfx12+ supports split barriers
    # (s_barrier_signal/wait) and per-counter waits (s_wait_dscnt /
    # s_wait_storecnt) which let the wave overlap memory waits with the
    # barrier. RDNA 3 / 3.5 (gfx11 / gfx115x) only ship the single
    # S_BARRIER + combined S_WAITCNT family — see
    # docs/rdna35_research/01_isa_reference.md §5.1–5.2.
    if gpu_arch.startswith("gfx12"):
        _BARRIER_ASM = (
            "s_wait_dscnt 0x0\n"
            "s_wait_storecnt 0x0\n"
            "s_barrier_signal -1\n"
            "s_barrier_wait -1"
        )
    else:
        _BARRIER_ASM = (
            # On gfx11/11.5, we only need LDS ordering + workgroup sync here.
            # Waiting vmcnt(0) every iteration over-serializes global loads.
            "s_waitcnt lgkmcnt(0)\n"
            "s_barrier"
        )

    elem_bytes = 2  # bf16/f16 are both 2 bytes
    allocator = SmemAllocator(None, arch=gpu_arch)
    # Reserve LDS space (allocate_array needs an ir.Type, but we're outside MLIR
    # context here; manually compute offset instead).
    lds_byte_offset = allocator._align(allocator.ptr, elem_bytes)
    allocator.ptr = lds_byte_offset + LDS_TOTAL * elem_bytes

    @flyc.kernel
    def wmma_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
    ):
        in_ir_ty = T.bf16 if is_bf16 else T.f16
        v8_in_ty = T.vec(8, in_ir_ty)

        from flydsl.utils.smem_allocator import SmemPtr

        lds_base = allocator.get_base()
        lds_vec_ptr = SmemPtr(lds_base, lds_byte_offset, v8_in_ty, shape=(LDS_TOTAL // LOAD_VEC,))

        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        wave_id = tid // 32
        lane = tid % 32
        lane16 = lane % 16
        klane = lane // 16
        base8 = klane * 8

        # RDNA3/3.5 V_WMMA_F32_16X16X16_{F16,BF16} ("WMMA256bInsts") expects
        # each wave32 lane to provide a 16-wide A/B operand. FlyDSL fragments
        # carry only K/2 = 8 K-elements per lane (group 0 = K[0:8],
        # group 1 = K[8:16]); the high half is pulled from lane^16 via
        # ds_bpermute. C++/dialect-path equivalent:
        #   lib/Dialect/FlyROCDL/RDNA3/MmaAtom.cpp::expandTo16WideWmma256
        _lane_i32 = fx.Int32(lane)
        _paired_byte_addr = (_lane_i32 ^ fx.Int32(16)) * fx.Int32(4)

        def _expand_to_wmma256(v8):
            """Concatenate <8 x T> self with <8 x T> from lane^16 to <16 x T>."""
            src_dtype = v8.dtype
            packed = v8.bitcast(fx.Int32)  # <4 x i32>
            paired_words = []
            # NOTE: must use range_constexpr (compile-time unroll), not bare
            # range() — inside @flyc.kernel the AST rewriter turns range() into
            # scf_range and `i` would become a runtime index value rather than
            # a Python int, breaking vector.extract's static_position=[i].
            for i in range_constexpr(4):
                word_i32 = vector.extract(packed, static_position=[i])
                paired_i32 = rocdl.ds_bpermute(
                    fx.Int32.ir_type, _paired_byte_addr, word_i32
                )
                paired_words.append(paired_i32)
            paired_packed = fx.Vector.from_elements(paired_words, fx.Int32)
            paired_v8 = paired_packed.bitcast(src_dtype)
            return v8.shuffle(paired_v8, list(range(16)))

        # The lane-group bit `(lane & 16) != 0` distinguishes lane group 0
        # (matrix lanes 0..15) from lane group 1 (mirror lanes 16..31). We
        # need it as an i1/i32 to select between the two output reorder masks.
        _is_upper_group_i32 = _lane_i32 & fx.Int32(16)

        def _reorder_f32_acc(c8):
            """Re-permute per-lane <8 x f32> WMMA accumulator to the kernel's
            lane-local row-major fragment convention (mirrors
            ``MmaAtom.cpp::reorderF32AccLaneValues``).

            Without this, LLVM 23 ``rocdl.wmma.f32.16x16x16.{f16,bf16}`` returns
            the f32 result in a lane-local order that differs from what the
            kernel's store path expects, producing row-shuffled output.
            """
            # 1. Lane-local interleave: {0, 4, 1, 5, 2, 6, 3, 7}.
            lane_local = c8.shuffle(c8, [0, 4, 1, 5, 2, 6, 3, 7])

            # 2. Cross-lane bpermute on each i32 element from lane^16.
            packed = lane_local.bitcast(fx.Int32)  # <8 x i32>
            paired_words = []
            for i in range_constexpr(8):
                word_i32 = vector.extract(packed, static_position=[i])
                paired_i32 = rocdl.ds_bpermute(
                    fx.Int32.ir_type, _paired_byte_addr, word_i32
                )
                paired_words.append(paired_i32)
            paired_packed = fx.Vector.from_elements(paired_words, fx.Int32)
            paired_lane_local = paired_packed.bitcast(fx.Float32)  # <8 x f32>

            # 3. Lane-group select between mask {0,8,2,10,4,12,6,14} (group 0)
            #    and {9,1,11,3,13,5,15,7} (group 1, lane^16).
            out_g0 = lane_local.shuffle(paired_lane_local, [0, 8, 2, 10, 4, 12, 6, 14])
            out_g1 = lane_local.shuffle(paired_lane_local, [9, 1, 11, 3, 13, 5, 15, 7])
            is_upper = _is_upper_group_i32 != fx.Int32(0)
            # is_upper True (lane in 16..31) -> out_g1; else out_g0.
            return is_upper.select(out_g1, out_g0)

        def _wmma_fma_expanded(a_wide, b_wide, acc_wmma):
            """One WMMA FMA with expanded A/B operands.

            `acc_wmma` stays in the native WMMA accumulator lane layout.
            We only reorder once before global-store, not on every FMA.
            """
            # Fast path: BF16 native-accumulate WMMA (no f32 reorder pipeline).
            # Keep this path only for bf16->bf16 to preserve predictable numerics
            # for other output types.
            if const_expr(use_native_bf16_acc):
                a_i16 = a_wide.bitcast(fx.Int16)
                b_i16 = b_wide.bitcast(fx.Int16)
                c_i16_wide = _expand_to_wmma256(fx.Vector(acc_wmma)).bitcast(fx.Int16)
                # In current runtime bindings this symbol is exposed as the
                # underlying ODS op-builder (res, a, b, c) without explicit
                # op_sel argument.
                res_wide_i16 = rocdl.wmma_bf16_16x16x16_bf16(
                    c_i16_wide.type, a_i16, b_i16, c_i16_wide
                ).result
                res_wide = fx.Vector(res_wide_i16)
                res_lo_i16 = fx.Vector.from_elements(
                    [res_wide[i] for i in range_constexpr(8)], fx.Int16
                )
                return res_lo_i16.bitcast(fx.BFloat16)

            # const_expr() flags `is_bf16` as a Python compile-time branch so
            # the AST rewriter does not lower it into a runtime scf.if.
            if const_expr(is_bf16):
                a_i16 = a_wide.bitcast(fx.Int16)
                b_i16 = b_wide.bitcast(fx.Int16)
                res = rocdl.wmma_f32_16x16x16_bf16(
                    acc_wmma.type, a_i16, b_i16, acc_wmma
                ).result
            else:
                res = rocdl.wmma_f32_16x16x16_f16(
                    acc_wmma.type, a_wide, b_wide, acc_wmma
                ).result
            return res

        # Swizzle workgroup mapping for L2 locality
        effective_group_m = min(group_m, grid_m)
        num_pid_in_group = effective_group_m * grid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * effective_group_m
        group_size_m = effective_group_m

        pid_in_group = pid % num_pid_in_group
        bid_m = first_pid_m + (pid_in_group % group_size_m)
        bid_n = pid_in_group // group_size_m

        # 2x2 warp layout
        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        tile_m0 = bid_m * BLOCK_M
        tile_n0 = bid_n * BLOCK_N

        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(arg_bt, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)

        # ============================================================
        # Pre-compute GMEM offsets and LDS addresses
        # ============================================================
        a_lds_info = []
        for al in range_constexpr(NUM_A_LOADS):
            a_lin = tid * LOAD_VEC + (al * THREADS_PER_BLOCK * LOAD_VEC)
            a_load_row = a_lin // BLOCK_K
            a_load_col = a_lin % BLOCK_K
            lds_rel = a_load_row * BLOCK_K_PAD_A + a_load_col
            g_row = tile_m0 + a_load_row
            a_lds_info.append((g_row, a_load_col, lds_rel))

        b_lds_info = []
        for bl in range_constexpr(NUM_B_LOADS):
            b_lin = tid * LOAD_VEC + (bl * THREADS_PER_BLOCK * LOAD_VEC)
            b_load_row = b_lin // BLOCK_K
            b_load_col = b_lin % BLOCK_K
            lds_rel = LDS_A_SIZE + b_load_row * BLOCK_K_PAD_B + b_load_col
            g_row = tile_n0 + b_load_row
            b_lds_info.append((g_row, b_load_col, lds_rel))

        # ============================================================
        # Phase 1: Issue GMEM loads (non-blocking), return raw data
        # ============================================================
        def _gmem_load(k_base):
            """Issue buffer_loads for A+B tile. Returns list of raw v4f32."""
            raw_data = []
            for al in range_constexpr(NUM_A_LOADS):
                g_row, a_load_col, _ = a_lds_info[al]
                g_col = k_base + a_load_col
                elem_off = g_row * K + g_col
                f32_off = elem_off // 2
                a_raw = buffer_ops.buffer_load(a_rsrc, f32_off, vec_width=4, dtype=fx.Float32)
                raw_data.append(a_raw)

            for bl in range_constexpr(NUM_B_LOADS):
                g_row, b_load_col, _ = b_lds_info[bl]
                g_col = k_base + b_load_col
                elem_off = g_row * K + g_col
                f32_off = elem_off // 2
                b_raw = buffer_ops.buffer_load(bt_rsrc, f32_off, vec_width=4, dtype=fx.Float32)
                raw_data.append(b_raw)

            return raw_data  # [a0, a1, a2, a3, b0, b1, b2, b3] -- 8 x v4f32

        # ============================================================
        # Phase 2: Store loaded data to LDS
        # ============================================================
        def _lds_store(raw_data, buf_offset):
            """Store previously loaded data to LDS at buf_offset."""
            for al in range_constexpr(NUM_A_LOADS):
                _, _, lds_rel = a_lds_info[al]
                a_vec = raw_data[al].bitcast(fx.BFloat16 if is_bf16 else fx.Float16)
                lds_idx = buf_offset + lds_rel
                lds_vec_ptr.store(a_vec, [lds_idx // 8])

            for bl in range_constexpr(NUM_B_LOADS):
                _, _, lds_rel = b_lds_info[bl]
                b_vec = raw_data[NUM_A_LOADS + bl].bitcast(fx.BFloat16 if is_bf16 else fx.Float16)
                lds_idx = buf_offset + lds_rel
                lds_vec_ptr.store(b_vec, [lds_idx // 8])

        # ============================================================
        # LDS read helpers -- row-major with K-padding
        # ============================================================
        def _load_a_from_lds(rk, buf_offset):
            """Load A WMMA operands from LDS for K-step rk."""
            vecs = []
            col_base = 16 * rk + base8
            for rm in range_constexpr(reg_m):
                row = wave_m * (reg_m * WMMA_M) + 16 * rm + lane16
                lds_idx = buf_offset + row * BLOCK_K_PAD_A + col_base
                a_raw = lds_vec_ptr.load([lds_idx // 8])
                vecs.append(a_raw)
            return vecs

        def _load_b_from_lds(rk, buf_offset):
            """Load B WMMA operands from LDS for K-step rk."""
            vecs = []
            col_base = 16 * rk + base8
            for rn in range_constexpr(reg_n):
                row = wave_n * (reg_n * WMMA_N) + 16 * rn + lane16
                lds_idx = buf_offset + LDS_A_SIZE + row * BLOCK_K_PAD_B + col_base
                b_raw = lds_vec_ptr.load([lds_idx // 8])
                vecs.append(b_raw)
            return vecs

        def _barrier():
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=_BARRIER_ASM,
                constraints="",
                has_side_effects=True,
            )

        def _do_compute_rk(accs_in, rk, buf_offset):
            """Compute all WMMAs for one K-step.

            Pattern: load all B first, then for each A load 1 A -> 4 WMMAs.
            This keeps register pressure low: only 4 B + 1 A + 16 accs live.
            """
            new_accs = list(accs_in)
            # Load all B operands for this K-step first, and expand once.
            b_vecs = _load_b_from_lds(rk, buf_offset)
            b_wides = []
            for rn in range_constexpr(reg_n):
                b_wides.append(_expand_to_wmma256(b_vecs[rn]))
            # Then load A one at a time and do reg_n WMMAs per A
            for rm in range_constexpr(reg_m):
                a_vec = _load_a_single_from_lds(rk, rm, buf_offset)
                a_wide = _expand_to_wmma256(a_vec)
                for rn in range_constexpr(reg_n):
                    idx = rm * reg_n + rn
                    new_accs[idx] = _wmma_fma_expanded(
                        a_wide,
                        b_wides[rn],
                        new_accs[idx],
                    )
            return new_accs

        def _load_a_single_from_lds(rk, rm_val, buf_offset):
            """Load a single A WMMA operand from LDS for K-step rk, repeat rm_val."""
            col_base = 16 * rk + base8
            row = wave_m * (reg_m * WMMA_M) + 16 * rm_val + lane16
            lds_idx = buf_offset + row * BLOCK_K_PAD_A + col_base
            return lds_vec_ptr.load([lds_idx // 8])

        # ============================================================
        # Initialize accumulators -- 4x4 = 16 accumulators.
        # Keep them in native WMMA layout throughout the main loop and only
        # convert once before epilogue stores.
        # ============================================================
        if const_expr(use_native_bf16_acc):
            zero_acc = fx.full(8, 0.0, fx.BFloat16)
        else:
            zero_acc = fx.full(8, 0.0, fx.Float32)
        accs = [zero_acc for _ in range_constexpr(reg_m * reg_n)]

        # ============================================================
        # DOUBLE-BUFFERED PIPELINE WITH SPLIT LOAD/STORE
        # ============================================================

        c_lds_buf_stride = LDS_ONE_BUF

        # --- PROLOGUE ---
        prologue_data = _gmem_load(0)
        _lds_store(prologue_data, 0)
        _barrier()

        # --- MAIN LOOP: kt=0..num_k_tiles-2 (SCF loop) ---
        # Loop-carried: accs (reg_m*reg_n accumulators)
        n_acc = reg_m * reg_n
        init_state = list(accs)

        for iv, state in range(0, num_k_tiles - 1, 1, init=init_state):
            s_accs = list(state[:n_acc])

            # Ping-pong: even iterations read buf0/write buf1, odd reversed
            read_off = iv % 2 * c_lds_buf_stride
            write_off = (1 - iv % 2) * c_lds_buf_stride

            # 1. Issue GMEM loads for next tile (non-blocking)
            next_k = (iv + 1) * BLOCK_K
            next_data = _gmem_load(next_k)

            # 2. Compute from current read buffer
            for rk in range_constexpr(reg_k):
                s_accs = _do_compute_rk(s_accs, rk, read_off)

            # 3. Store loaded data to write buffer
            _lds_store(next_data, write_off)

            # 4. Barrier
            _barrier()

            results = yield list(s_accs)

        accs = list(results[:n_acc])

        # --- EPILOGUE: Last tile in LDS ---
        # After num_k_tiles-1 iterations, last written buffer is the read buffer
        last_read_off = ((num_k_tiles - 1) % 2) * c_lds_buf_stride
        for rk in range_constexpr(reg_k):
            accs = _do_compute_rk(accs, rk, last_read_off)

        # ============================================================
        # Store results to GMEM
        # ============================================================
        for rm in range_constexpr(reg_m):
            for rn in range_constexpr(reg_n):
                idx = rm * reg_n + rn
                if const_expr(use_native_bf16_acc):
                    acc_lane = accs[idx]
                else:
                    acc_lane = _reorder_f32_acc(accs[idx])
                wmma_m_off = wave_m * (reg_m * WMMA_M) + 16 * rm
                wmma_n_off = wave_n * (reg_n * WMMA_N) + 16 * rn
                g_col = tile_n0 + wmma_n_off + lane16
                c_out_elem_bytes = fx.Int32(out_elem_bytes)
                c_row_stride_bytes = fx.Int32(N * out_elem_bytes)
                row0 = tile_m0 + wmma_m_off + base8
                col_byte_off = fx.Int32(g_col) * c_out_elem_bytes
                byte_off = fx.Int32(row0) * c_row_stride_bytes + col_byte_off
                for si in range_constexpr(8):
                    val = acc_lane[si]
                    if const_expr(out_dtype == "bf16"):
                        val = fx.BFloat16(val)
                    elif const_expr(out_dtype == "f16"):
                        val = fx.Float16(val)
                    buffer_ops.buffer_store(val, c_rsrc, byte_off, offset_is_bytes=True)
                    if const_expr(si != 7):
                        byte_off = byte_off + c_row_stride_bytes

    # ── Host launcher ──────────────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        c1 = 1
        total_blocks = grid_m * grid_n
        bk = THREADS_PER_BLOCK

        launcher = wmma_gemm_kernel(arg_c, arg_a, arg_bt)
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(bk, c1, c1),
            stream=stream,
        )

    return launch_gemm, BLOCK_M, BLOCK_N, BLOCK_K
