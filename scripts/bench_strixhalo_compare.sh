#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Compare FlyDSL vs non-FlyDSL baselines on Strix Halo-class GPUs.
# - softmax/layernorm/rmsnorm: FlyDSL vs AIter (Triton)
# - gemm: FlyDSL RDNA WMMA (LDS kernel) vs torch.mm

set -eu
if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${FLY_BUILD_DIR:-${REPO_ROOT}/build-fly}"
if [ ! -d "${BUILD_DIR}" ] && [ -d "${REPO_ROOT}/build" ]; then
  BUILD_DIR="${REPO_ROOT}/build"
fi

PYTHON_PACKAGE_ROOT="${BUILD_DIR}/python_packages"
export PYTHONPATH="${PYTHON_PACKAGE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
MLIR_LIBS_DIR="${PYTHON_PACKAGE_ROOT}/flydsl/_mlir/_mlir_libs"
if [ -d "${MLIR_LIBS_DIR}" ]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

BENCH_LOG_DIR="${BENCH_LOG_DIR:-/tmp/flydsl_bench}"
mkdir -p "${BENCH_LOG_DIR}"

OPS="softmax,layernorm,rmsnorm,gemm"
SHAPES="4096,4096,bf16;8192,8192,bf16;32768,8192,bf16"
GEMM_SUITE="strix-balanced"
GEMM_SHAPES=""
WARMUP=20
ITERS=100
USE_AITER=1
CSV_PATH=""

_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/bench_strixhalo_compare.sh [options]

Options:
  --ops <csv>            Ops to run (default: softmax,layernorm,rmsnorm,gemm)
  --shapes <list>        Norm shapes: "M,N,dtype;M,N,dtype;..."
  --gemm-suite <name>    Predefined GEMM suite (default: strix-balanced)
  --dump-gemm-suite <name>  Print all shapes in suite and exit
  --list-gemm-suites     List available GEMM suites and exit
  --gemm-shapes <list>   GEMM shapes: "M,N,K,dtype;M,N,K,dtype;..."
  --warmup <n>           Warmup iterations (default: 20)
  --iters <n>            Timed iterations (default: 100)
  --csv <path>           Output CSV path (default: /tmp/flydsl_bench/strixhalo_compare_<ts>.csv)
  --no-aiter             Disable AIter baseline for norm ops
  -h, --help             Show this message

Examples:
  bash scripts/bench_strixhalo_compare.sh
  bash scripts/bench_strixhalo_compare.sh --ops gemm --gemm-suite strix-square
  bash scripts/bench_strixhalo_compare.sh --dump-gemm-suite strix-llm-384
  bash scripts/bench_strixhalo_compare.sh --ops softmax,gemm --iters 50
  bash scripts/bench_strixhalo_compare.sh --shapes "32768,8192,bf16" --gemm-shapes "1024,1024,1024,bf16"
USAGE
}

_die() {
  echo "error: $*" >&2
  echo "" >&2
  _usage >&2
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ops)
      shift
      [ "$#" -gt 0 ] || _die "--ops requires a value"
      OPS="$1"
      ;;
    --ops=*)
      OPS="${1#--ops=}"
      ;;
    --shapes)
      shift
      [ "$#" -gt 0 ] || _die "--shapes requires a value"
      SHAPES="$1"
      ;;
    --shapes=*)
      SHAPES="${1#--shapes=}"
      ;;
    --gemm-shapes)
      shift
      [ "$#" -gt 0 ] || _die "--gemm-shapes requires a value"
      GEMM_SHAPES="$1"
      ;;
    --gemm-shapes=*)
      GEMM_SHAPES="${1#--gemm-shapes=}"
      ;;
    --gemm-suite)
      shift
      [ "$#" -gt 0 ] || _die "--gemm-suite requires a value"
      GEMM_SUITE="$1"
      ;;
    --gemm-suite=*)
      GEMM_SUITE="${1#--gemm-suite=}"
      ;;
    --dump-gemm-suite)
      shift
      [ "$#" -gt 0 ] || _die "--dump-gemm-suite requires a value"
      export STRIX_DUMP_GEMM_SUITE="$1"
      ;;
    --dump-gemm-suite=*)
      export STRIX_DUMP_GEMM_SUITE="${1#--dump-gemm-suite=}"
      ;;
    --list-gemm-suites)
      export STRIX_LIST_GEMM_SUITES=1
      ;;
    --warmup)
      shift
      [ "$#" -gt 0 ] || _die "--warmup requires a value"
      WARMUP="$1"
      ;;
    --warmup=*)
      WARMUP="${1#--warmup=}"
      ;;
    --iters)
      shift
      [ "$#" -gt 0 ] || _die "--iters requires a value"
      ITERS="$1"
      ;;
    --iters=*)
      ITERS="${1#--iters=}"
      ;;
    --csv)
      shift
      [ "$#" -gt 0 ] || _die "--csv requires a value"
      CSV_PATH="$1"
      ;;
    --csv=*)
      CSV_PATH="${1#--csv=}"
      ;;
    --no-aiter)
      USE_AITER=0
      ;;
    -h|--help)
      _usage
      exit 0
      ;;
    *)
      _die "unknown argument '$1'"
      ;;
  esac
  shift
done

if [ -z "${CSV_PATH}" ]; then
  ts=$(date +%Y%m%d_%H%M%S)
  CSV_PATH="${BENCH_LOG_DIR}/strixhalo_compare_${ts}.csv"
fi

export STRIX_OPS="${OPS}"
export STRIX_SHAPES="${SHAPES}"
export STRIX_GEMM_SHAPES="${GEMM_SHAPES}"
export STRIX_GEMM_SUITE="${GEMM_SUITE}"
export STRIX_WARMUP="${WARMUP}"
export STRIX_ITERS="${ITERS}"
export STRIX_USE_AITER="${USE_AITER}"
export STRIX_CSV_PATH="${CSV_PATH}"
export STRIX_AITER_IMPL="${AITER_IMPL:-triton}"
export STRIX_LIST_GEMM_SUITES="${STRIX_LIST_GEMM_SUITES:-0}"
export STRIX_DUMP_GEMM_SUITE="${STRIX_DUMP_GEMM_SUITE:-}"

python3 - <<'PY'
import csv
import math
import os
import statistics

import torch

from flydsl.runtime.device import get_rocm_arch
from tests.kernels import benchmark_common as bc


def _fmt_us(v):
    if v is None:
        return "-"
    return f"{v:.1f}"


def _fmt_speedup(v):
    if v is None:
        return "-"
    return f"{v:.2f}x"


def _safe_float(v):
    if v is None:
        return None
    return float(v)


def _speedup(flydsl_us, baseline_us):
    if not flydsl_us or not baseline_us:
        return None
    if flydsl_us <= 0 or baseline_us <= 0:
        return None
    return baseline_us / flydsl_us


def _build_strix_llm_384():
    """Common LLM GEMM shapes (384 entries, bf16)."""
    rows = []

    # Decode/small-batch and medium prefill.
    m_small = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    hidden_main = [2048, 3072, 4096, 5120, 6144, 7168, 8192]
    for m in m_small:
        for h in hidden_main:
            rows.append((m, h, h, "bf16"))      # self projection
            rows.append((m, h, 3 * h, "bf16"))  # QKV projection
            rows.append((m, h, 4 * h, "bf16"))  # FFN up projection
            rows.append((m, 4 * h, h, "bf16"))  # FFN down projection

    # Large prefill.
    m_large = [4096, 8192, 16384, 32768]
    hidden_large = [2048, 4096, 8192]
    for m in m_large:
        for h in hidden_large:
            rows.append((m, h, h, "bf16"))
            rows.append((m, h, 4 * h, "bf16"))
            rows.append((m, 4 * h, h, "bf16"))

    # Generic square sanity set.
    square_dims = [256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216]
    for d in square_dims:
        rows.append((d, d, d, "bf16"))

    # A few extra common M=3072 prefill-like rectangulars.
    rows.append((3072, 4096, 4096, "bf16"))
    rows.append((3072, 4096, 16384, "bf16"))
    rows.append((3072, 16384, 4096, "bf16"))

    # De-dup while preserving order.
    dedup_rows = []
    seen = set()
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        dedup_rows.append(row)
    return dedup_rows


GEMM_SUITES = {
    # Fast smoke/regression checks.
    "quick": [
        (512, 512, 512, "bf16"),
        (1024, 1024, 1024, "bf16"),
    ],
    # Square-heavy set for comparing default kernels.
    "strix-square": [
        (512, 512, 512, "bf16"),
        (1024, 1024, 1024, "bf16"),
        (1536, 1536, 1536, "bf16"),
        (2048, 2048, 2048, "bf16"),
    ],
    # Balanced set: small/medium/large + rectangulars.
    "strix-balanced": [
        (512, 512, 512, "bf16"),
        (1024, 1024, 1024, "bf16"),
        (2048, 2048, 2048, "bf16"),
        (512, 2048, 512, "bf16"),
        (2048, 512, 2048, "bf16"),
        (1024, 4096, 1024, "bf16"),
        (4096, 1024, 1024, "bf16"),
    ],
    "strix-llm-384": _build_strix_llm_384(),
}


def _parse_list(spec, expected_fields):
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        fields = [x.strip() for x in chunk.split(",")]
        if len(fields) != expected_fields:
            raise ValueError(
                f"invalid spec '{chunk}', expected {expected_fields} comma-separated fields"
            )
        out.append(fields)
    return out


def _resolve_gemm_shapes(gemm_shapes_spec, gemm_suite):
    if gemm_shapes_spec.strip():
        parsed = _parse_list(gemm_shapes_spec, expected_fields=4)
        return [(int(M), int(N), int(K), dt) for M, N, K, dt in parsed]
    if gemm_suite not in GEMM_SUITES:
        raise ValueError(
            f"unknown gemm suite '{gemm_suite}', available: {sorted(GEMM_SUITES.keys())}"
        )
    return list(GEMM_SUITES[gemm_suite])


def _norm_dtype(dt):
    dt = dt.lower()
    if dt in ("bf16", "bfloat16"):
        return "bf16"
    if dt in ("f16", "fp16", "float16"):
        return "f16"
    if dt in ("f32", "fp32", "float32"):
        return "f32"
    raise ValueError(f"unsupported dtype '{dt}'")


def _gemm_dtype(dt):
    dt = dt.lower()
    if dt in ("bf16", "bfloat16"):
        return "bf16"
    if dt in ("f16", "fp16", "float16"):
        return "f16"
    raise ValueError(f"unsupported GEMM dtype '{dt}' (use f16 or bf16)")


def _torch_dtype(dt):
    if dt == "bf16":
        return torch.bfloat16
    if dt == "f16":
        return torch.float16
    if dt == "f32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype mapping for '{dt}'")


class _SkipShapeError(Exception):
    """Raised when a shape should be skipped, not counted as benchmark error."""


def _bench_flydsl_norm(op, M, N, dtype, warmup, iters):
    torch_dt = _torch_dtype(dtype)
    stream = torch.cuda.current_stream()

    if op == "softmax":
        from kernels.softmax_kernel import build_softmax_module

        launch_fn = build_softmax_module(M, N, dtype)
        a = torch.randn((M, N), device="cuda", dtype=torch_dt).contiguous()
        c = torch.empty((M, N), device="cuda", dtype=torch_dt)

        def run():
            launch_fn(a, c, M, stream=stream)

        return bc.bench_gpu_us_torch(run, warmup=warmup, iters=iters)

    if op == "layernorm":
        from kernels.layernorm_kernel import build_layernorm_module

        launch_fn = build_layernorm_module(M, N, dtype)
        x = torch.randn((M, N), device="cuda", dtype=torch_dt).contiguous()
        gamma = torch.randn((N,), device="cuda", dtype=torch_dt).contiguous()
        beta = torch.randn((N,), device="cuda", dtype=torch_dt).contiguous()
        y = torch.empty((M, N), device="cuda", dtype=torch_dt)

        def run():
            launch_fn(x, gamma, beta, y, M, stream=stream)

        return bc.bench_gpu_us_torch(run, warmup=warmup, iters=iters)

    if op == "rmsnorm":
        from kernels.rmsnorm_kernel import build_rmsnorm_module

        launch_fn = build_rmsnorm_module(M, N, dtype)
        x = torch.randn((M, N), device="cuda", dtype=torch_dt).contiguous()
        gamma = torch.randn((N,), device="cuda", dtype=torch_dt).contiguous()
        y = torch.empty((M, N), device="cuda", dtype=torch_dt)

        def run():
            launch_fn(x, gamma, y, M, stream=stream)

        return bc.bench_gpu_us_torch(run, warmup=warmup, iters=iters)

    raise ValueError(f"unsupported norm op '{op}'")


def _bench_torch_norm(op, M, N, dtype, warmup, iters):
    torch_dt = _torch_dtype(dtype)
    x = torch.randn((M, N), device="cuda", dtype=torch_dt).contiguous()

    if op == "softmax":
        return bc.bench_gpu_us_torch(
            lambda: torch.softmax(x, dim=1), warmup=warmup, iters=iters
        )

    if op == "layernorm":
        gamma = torch.randn((N,), device="cuda", dtype=torch_dt).contiguous()
        beta = torch.randn((N,), device="cuda", dtype=torch_dt).contiguous()
        return bc.bench_gpu_us_torch(
            lambda: torch.nn.functional.layer_norm(x, (N,), gamma, beta, 1e-5),
            warmup=warmup,
            iters=iters,
        )

    if op == "rmsnorm":
        gamma = torch.randn((N,), device="cuda", dtype=torch_dt).contiguous()
        return bc.bench_gpu_us_torch(
            lambda: x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + 1e-5) * gamma,
            warmup=warmup,
            iters=iters,
        )

    raise ValueError(f"unsupported norm op '{op}'")


def _run_norm_row(op, M, N, dtype, warmup, iters, use_aiter, aiter_impl):
    flydsl_us = _bench_flydsl_norm(
        op=op, M=M, N=N, dtype=dtype, warmup=warmup, iters=iters
    )
    baseline_name = "none"
    baseline_us = None
    note = ""
    if use_aiter:
        baseline_name = f"aiter-{aiter_impl}"
        baseline_us = bc._bench_aiter(
            op=op,
            impl=aiter_impl,
            M=M,
            N=N,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
        )
        if baseline_us is None:
            baseline_name = "torch"
            baseline_us = _bench_torch_norm(
                op=op, M=M, N=N, dtype=dtype, warmup=warmup, iters=iters
            )
            note = "AIter unavailable or kernel missing; fell back to torch baseline"
    else:
        baseline_name = "torch"
        baseline_us = _bench_torch_norm(
            op=op, M=M, N=N, dtype=dtype, warmup=warmup, iters=iters
        )
    return {
        "op": op,
        "shape": f"{M}x{N}",
        "dtype": dtype,
        "flydsl_gpu_us": _safe_float(flydsl_us),
        "baseline_name": baseline_name,
        "baseline_gpu_us": _safe_float(baseline_us),
        "speedup": _safe_float(_speedup(flydsl_us, baseline_us)),
        "status": "ok",
        "note": note,
    }


def _run_gemm_row(M, N, K, dtype, warmup, iters):
    from kernels.rdna_f16_gemm import create_wmma_gemm_module

    torch_dt = _torch_dtype(dtype)
    elem_size = torch.tensor([], device="cuda", dtype=torch_dt).element_size()

    try:
        launch_fn, block_m, block_n, block_k = create_wmma_gemm_module(
            M, N, K, in_dtype=dtype, out_dtype=dtype
        )
    except AssertionError as exc:
        raise _SkipShapeError(
            f"unsupported by RDNA WMMA tile constraints "
            f"(requires M%{16}==0 and divisibility by kernel block sizes)"
        ) from exc

    if M % block_m != 0 or N % block_n != 0 or K % block_k != 0:
        raise _SkipShapeError(
            f"unsupported by kernel tile: requires M%{block_m}=0, N%{block_n}=0, K%{block_k}=0"
        )

    # A, B, B_T, C, C_torch all stay resident for a row; keep margin for
    # runtime overhead/caches to avoid allocator thrash or OOM.
    required_bytes = elem_size * (M * K + K * N + K * N + M * N + M * N)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if required_bytes > int(free_bytes * 0.85):
        need_gib = required_bytes / (1024**3)
        free_gib = free_bytes / (1024**3)
        raise _SkipShapeError(
            f"estimated allocation {need_gib:.2f} GiB exceeds safe free memory {free_gib:.2f} GiB"
        )

    A = torch.randn((M, K), device="cuda", dtype=torch_dt)
    B = torch.randn((K, N), device="cuda", dtype=torch_dt)
    B_T = B.T.contiguous()
    # Keep output dtype aligned with torch baseline for fair comparison.
    C = torch.empty((M, N), device="cuda", dtype=torch_dt)
    C_torch = torch.empty((M, N), device="cuda", dtype=torch_dt)

    # Trigger JIT once so compile latency is excluded.
    launch_fn(C, A, B_T, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    flydsl_us = bc.bench_gpu_us_torch(
        lambda: launch_fn(C, A, B_T, stream=torch.cuda.current_stream()),
        warmup=warmup,
        iters=iters,
    )
    baseline_us = bc.bench_gpu_us_torch(
        lambda: torch.mm(A, B, out=C_torch), warmup=warmup, iters=iters
    )
    return {
        "op": "gemm",
        "shape": f"{M}x{N}x{K}",
        "dtype": dtype,
        "flydsl_gpu_us": _safe_float(flydsl_us),
        "baseline_name": "torch.mm",
        "baseline_gpu_us": _safe_float(baseline_us),
        "speedup": _safe_float(_speedup(flydsl_us, baseline_us)),
        "status": "ok",
        "note": "",
    }


def _print_rows(rows):
    print("\n" + "=" * 118)
    print("Strix Halo Compare (FlyDSL vs baseline)")
    print("=" * 118)
    print(
        f"{'op':10s} {'shape':18s} {'dtype':6s} {'FlyDSL(us)':>12s} "
        f"{'baseline':>12s} {'base(us)':>10s} {'speedup':>10s} {'status':>8s}"
    )
    print("-" * 118)
    for row in rows:
        print(
            f"{row['op']:10s} {row['shape']:18s} {row['dtype']:6s} "
            f"{_fmt_us(row['flydsl_gpu_us']):>12s} {row['baseline_name']:>12s} "
            f"{_fmt_us(row['baseline_gpu_us']):>10s} {_fmt_speedup(row['speedup']):>10s} "
            f"{row['status']:>8s}"
        )
        if row["note"]:
            print(f"  note: {row['note']}")
    print("-" * 118)


def _print_gemm_summary(rows):
    gemm_rows = [r for r in rows if r["op"] == "gemm" and r["status"] == "ok" and r["speedup"] is not None]
    if not gemm_rows:
        return

    speedups = [float(r["speedup"]) for r in gemm_rows]
    geomean = math.exp(sum(math.log(x) for x in speedups) / len(speedups))
    p50 = statistics.median(speedups)
    if len(speedups) == 1:
        p90 = speedups[0]
    else:
        q = statistics.quantiles(speedups, n=10, method="inclusive")
        p90 = q[8]
    worst_row = min(gemm_rows, key=lambda r: r["speedup"])

    print("GEMM Aggregate (FlyDSL/torch)")
    print(
        f"- geomean={geomean:.3f}x  p50={p50:.3f}x  p90={p90:.3f}x  "
        f"worst={float(worst_row['speedup']):.3f}x @ {worst_row['shape']}"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is not available")

    if os.environ.get("STRIX_LIST_GEMM_SUITES", "0") == "1":
        for suite_name in sorted(GEMM_SUITES.keys()):
            print(suite_name)
        return

    dump_suite = os.environ.get("STRIX_DUMP_GEMM_SUITE", "").strip()
    if dump_suite:
        if dump_suite not in GEMM_SUITES:
            raise ValueError(
                f"unknown gemm suite '{dump_suite}', available: {sorted(GEMM_SUITES.keys())}"
            )
        suite_rows = GEMM_SUITES[dump_suite]
        for M, N, K, dt in suite_rows:
            print(f"{M},{N},{K},{dt}")
        print(f"# total_shapes={len(suite_rows)}")
        return

    arch = str(get_rocm_arch())
    ops = [x.strip().lower() for x in os.environ["STRIX_OPS"].split(",") if x.strip()]
    ops_set = set(ops)
    allowed_ops = {"softmax", "layernorm", "rmsnorm", "gemm"}
    unknown_ops = sorted(ops_set - allowed_ops)
    if unknown_ops:
        raise ValueError(f"unsupported ops: {unknown_ops}")

    shapes = _parse_list(os.environ["STRIX_SHAPES"], expected_fields=3)
    gemm_shapes = _resolve_gemm_shapes(
        os.environ["STRIX_GEMM_SHAPES"], os.environ["STRIX_GEMM_SUITE"]
    )
    warmup = int(os.environ["STRIX_WARMUP"])
    iters = int(os.environ["STRIX_ITERS"])
    use_aiter = os.environ["STRIX_USE_AITER"] == "1"
    aiter_impl = os.environ["STRIX_AITER_IMPL"]
    csv_path = os.environ["STRIX_CSV_PATH"]

    print(f"[bench_strixhalo_compare] GPU arch: {arch}")
    if not arch.startswith("gfx115"):
        print(
            "[bench_strixhalo_compare] warning: target is optimized for Strix Halo (gfx115x). "
            "Continuing anyway."
        )

    rows = []
    error_count = 0

    for op in ("softmax", "layernorm", "rmsnorm"):
        if op not in ops_set:
            continue
        for M_s, N_s, dt_s in shapes:
            M = int(M_s)
            N = int(N_s)
            dtype = _norm_dtype(dt_s)
            try:
                row = _run_norm_row(
                    op=op,
                    M=M,
                    N=N,
                    dtype=dtype,
                    warmup=warmup,
                    iters=iters,
                    use_aiter=use_aiter,
                    aiter_impl=aiter_impl,
                )
            except Exception as exc:
                error_count += 1
                row = {
                    "op": op,
                    "shape": f"{M}x{N}",
                    "dtype": dtype,
                    "flydsl_gpu_us": None,
                    "baseline_name": f"aiter-{aiter_impl}" if use_aiter else "none",
                    "baseline_gpu_us": None,
                    "speedup": None,
                    "status": "error",
                    "note": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)

    if "gemm" in ops_set:
        if not (arch.startswith("gfx12") or arch.startswith("gfx11")):
            rows.append(
                {
                    "op": "gemm",
                    "shape": "-",
                    "dtype": "-",
                    "flydsl_gpu_us": None,
                    "baseline_name": "torch.mm",
                    "baseline_gpu_us": None,
                    "speedup": None,
                    "status": "skip",
                    "note": f"RDNA WMMA GEMM path requires gfx11x/gfx12x, got {arch}",
                }
            )
        else:
            gpu_faulted = False
            gpu_fault_note = ""
            for M, N, K, dt_s in gemm_shapes:
                dtype = _gemm_dtype(dt_s)
                if gpu_faulted:
                    rows.append(
                        {
                            "op": "gemm",
                            "shape": f"{M}x{N}x{K}",
                            "dtype": dtype,
                            "flydsl_gpu_us": None,
                            "baseline_name": "torch.mm",
                            "baseline_gpu_us": None,
                            "speedup": None,
                            "status": "skip",
                            "note": gpu_fault_note,
                        }
                    )
                    continue
                try:
                    row = _run_gemm_row(
                        M=M, N=N, K=K, dtype=dtype, warmup=warmup, iters=iters
                    )
                except _SkipShapeError as exc:
                    row = {
                        "op": "gemm",
                        "shape": f"{M}x{N}x{K}",
                        "dtype": dtype,
                        "flydsl_gpu_us": None,
                        "baseline_name": "torch.mm",
                        "baseline_gpu_us": None,
                        "speedup": None,
                        "status": "skip",
                        "note": str(exc),
                    }
                except Exception as exc:
                    error_count += 1
                    exc_note = f"{type(exc).__name__}: {exc}"
                    lowered = exc_note.lower()
                    if "illegal memory access" in lowered or "hiperrorillegaladdress" in lowered:
                        gpu_faulted = True
                        gpu_fault_note = (
                            f"skipped after prior GPU fault at {M}x{N}x{K}; "
                            "restart benchmark process to clear HIP error state"
                        )
                    row = {
                        "op": "gemm",
                        "shape": f"{M}x{N}x{K}",
                        "dtype": dtype,
                        "flydsl_gpu_us": None,
                        "baseline_name": "torch.mm",
                        "baseline_gpu_us": None,
                        "speedup": None,
                        "status": "error",
                        "note": exc_note,
                    }
                rows.append(row)

    if not rows:
        raise RuntimeError("no benchmark rows were produced; check --ops/shape arguments")

    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "op",
        "shape",
        "dtype",
        "flydsl_gpu_us",
        "baseline_name",
        "baseline_gpu_us",
        "speedup",
        "status",
        "note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _print_rows(rows)
    _print_gemm_summary(rows)
    print(f"[bench_strixhalo_compare] CSV written: {csv_path}")

    if error_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
PY

echo "[bench_strixhalo_compare] done"
