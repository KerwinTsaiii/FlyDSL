#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Benchmark "fusion-first" operators on Strix-class GPUs.
# Focus: MoE reduction now; track gaps for rmsnorm_reduce_add / MLA.

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

OPS="moe_reduce,rmsnorm_reduce_add,mla"
MOE_SHAPES="1,8,7168,f16,0;65,8,7168,f16,0;16384,6,5120,f16,0;129,8,7168,f16,1"
MLA_SHAPES="1,128;4,2048;32,8192"
WARMUP=10
ITERS=50
CSV_PATH=""

_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/bench_strixhalo_fusion.sh [options]

Options:
  --ops <csv>            Ops: moe_reduce,rmsnorm_reduce_add,mla
                         (default: moe_reduce,rmsnorm_reduce_add,mla)
  --moe-shapes <list>    "tokens,topk,model_dim,dtype,use_mask;..."
                         dtype: f16|bf16|f32 ; use_mask: 0|1
  --mla-shapes <list>    "batch,ctx_len;..."
  --warmup <n>           Warmup iterations (default: 10)
  --iters <n>            Timed iterations (default: 50)
  --csv <path>           Output CSV path
                         (default: /tmp/flydsl_bench/strixhalo_fusion_<ts>.csv)
  -h, --help             Show this message
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
    --moe-shapes)
      shift
      [ "$#" -gt 0 ] || _die "--moe-shapes requires a value"
      MOE_SHAPES="$1"
      ;;
    --moe-shapes=*)
      MOE_SHAPES="${1#--moe-shapes=}"
      ;;
    --mla-shapes)
      shift
      [ "$#" -gt 0 ] || _die "--mla-shapes requires a value"
      MLA_SHAPES="$1"
      ;;
    --mla-shapes=*)
      MLA_SHAPES="${1#--mla-shapes=}"
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
  CSV_PATH="${BENCH_LOG_DIR}/strixhalo_fusion_${ts}.csv"
fi

export STRIX_FUSION_OPS="${OPS}"
export STRIX_FUSION_MOE_SHAPES="${MOE_SHAPES}"
export STRIX_FUSION_MLA_SHAPES="${MLA_SHAPES}"
export STRIX_FUSION_WARMUP="${WARMUP}"
export STRIX_FUSION_ITERS="${ITERS}"
export STRIX_FUSION_CSV_PATH="${CSV_PATH}"

python3 - <<'PY'
import csv
import os

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


def _torch_dtype(dt):
    dt = dt.lower()
    if dt in ("f16", "fp16", "float16"):
        return torch.float16
    if dt in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dt in ("f32", "fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype '{dt}'")


def _run_moe_reduce_row(tokens, topk, model_dim, dtype, use_mask, warmup, iters):
    from kernels.moe_gemm_2stage import compile_moe_reduction

    torch_dt = _torch_dtype(dtype)
    device = torch.device("cuda")
    stream = torch.cuda.current_stream()

    reduce_exe = compile_moe_reduction(
        topk=topk,
        model_dim=model_dim,
        dtype_str=dtype,
        use_mask=use_mask,
    )
    x = torch.randn((tokens, topk, model_dim), device=device, dtype=torch_dt)
    y = torch.empty((tokens, model_dim), device=device, dtype=torch_dt)
    if use_mask:
        valid_mask = torch.randint(0, 2, (tokens, topk), device=device, dtype=torch.uint8)
        x_ref = x * valid_mask.to(torch.bool).unsqueeze(-1)
    else:
        valid_mask = torch.empty((0, topk), device=device, dtype=torch.uint8)
        x_ref = x
    y_torch = torch.empty_like(y)

    def launch_flydsl():
        reduce_exe(x, y, valid_mask, tokens, stream)

    def launch_torch():
        torch.sum(x_ref, dim=1, out=y_torch)

    flydsl_us = bc.bench_gpu_us_torch(launch_flydsl, warmup=warmup, iters=iters)
    baseline_us = bc.bench_gpu_us_torch(launch_torch, warmup=warmup, iters=iters)

    return {
        "op": "moe_reduce",
        "shape": f"{tokens}x{topk}x{model_dim}",
        "dtype": dtype,
        "flydsl_gpu_us": _safe_float(flydsl_us),
        "baseline_name": "torch.sum",
        "baseline_gpu_us": _safe_float(baseline_us),
        "speedup": _safe_float(_speedup(flydsl_us, baseline_us)),
        "status": "ok",
        "note": "mask=1" if use_mask else "mask=0",
    }


def _run_mla_row(batch_size, ctx_len):
    # MLA depends on aiter metadata + fp8 path. Keep this entry optional.
    try:
        from tests.kernels.test_mla_decode import run_single
    except BaseException as exc:
        return {
            "op": "mla",
            "shape": f"{batch_size}x{ctx_len}",
            "dtype": "fp8/fp8",
            "flydsl_gpu_us": None,
            "baseline_name": "none",
            "baseline_gpu_us": None,
            "speedup": None,
            "status": "skip",
            "note": f"mla benchmark unavailable: {type(exc).__name__}: {exc}",
        }
    try:
        _, us = run_single(batch_size=batch_size, ctx_len=ctx_len)
        return {
            "op": "mla",
            "shape": f"{batch_size}x{ctx_len}",
            "dtype": "fp8/fp8",
            "flydsl_gpu_us": _safe_float(us),
            "baseline_name": "none",
            "baseline_gpu_us": None,
            "speedup": None,
            "status": "ok",
            "note": "decode-only kernel time; baseline not wired",
        }
    except BaseException as exc:
        return {
            "op": "mla",
            "shape": f"{batch_size}x{ctx_len}",
            "dtype": "fp8/fp8",
            "flydsl_gpu_us": None,
            "baseline_name": "none",
            "baseline_gpu_us": None,
            "speedup": None,
            "status": "skip",
            "note": f"mla benchmark failed: {type(exc).__name__}: {exc}",
        }


def _print_rows(rows):
    print("\n" + "=" * 120)
    print("Strix Halo Fusion Benchmark")
    print("=" * 120)
    print(
        f"{'op':18s} {'shape':20s} {'dtype':9s} {'FlyDSL(us)':>12s} "
        f"{'baseline':>12s} {'base(us)':>10s} {'speedup':>10s} {'status':>8s}"
    )
    print("-" * 120)
    for row in rows:
        print(
            f"{row['op']:18s} {row['shape']:20s} {row['dtype']:9s} "
            f"{_fmt_us(row['flydsl_gpu_us']):>12s} {row['baseline_name']:>12s} "
            f"{_fmt_us(row['baseline_gpu_us']):>10s} {_fmt_speedup(row['speedup']):>10s} "
            f"{row['status']:>8s}"
        )
        if row["note"]:
            print(f"  note: {row['note']}")
    print("-" * 120)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is not available")

    arch = str(get_rocm_arch())
    ops = [x.strip().lower() for x in os.environ["STRIX_FUSION_OPS"].split(",") if x.strip()]
    ops_set = set(ops)
    allowed_ops = {"moe_reduce", "rmsnorm_reduce_add", "mla"}
    unknown_ops = sorted(ops_set - allowed_ops)
    if unknown_ops:
        raise ValueError(f"unsupported ops: {unknown_ops}")

    moe_shapes = [
        (int(t), int(k), int(d), dt.lower(), m.strip() in ("1", "true", "True"))
        for t, k, d, dt, m in _parse_list(os.environ["STRIX_FUSION_MOE_SHAPES"], expected_fields=5)
    ]
    mla_shapes = [
        (int(b), int(c))
        for b, c in _parse_list(os.environ["STRIX_FUSION_MLA_SHAPES"], expected_fields=2)
    ]
    warmup = int(os.environ["STRIX_FUSION_WARMUP"])
    iters = int(os.environ["STRIX_FUSION_ITERS"])
    csv_path = os.environ["STRIX_FUSION_CSV_PATH"]

    print(f"[bench_strixhalo_fusion] GPU arch: {arch}")

    rows = []
    error_count = 0

    if "moe_reduce" in ops_set:
        for tokens, topk, model_dim, dtype, use_mask in moe_shapes:
            try:
                row = _run_moe_reduce_row(
                    tokens=tokens,
                    topk=topk,
                    model_dim=model_dim,
                    dtype=dtype,
                    use_mask=use_mask,
                    warmup=warmup,
                    iters=iters,
                )
            except Exception as exc:
                error_count += 1
                row = {
                    "op": "moe_reduce",
                    "shape": f"{tokens}x{topk}x{model_dim}",
                    "dtype": dtype,
                    "flydsl_gpu_us": None,
                    "baseline_name": "torch.sum",
                    "baseline_gpu_us": None,
                    "speedup": None,
                    "status": "error",
                    "note": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)

    if "rmsnorm_reduce_add" in ops_set:
        rows.append(
            {
                "op": "rmsnorm_reduce_add",
                "shape": "-",
                "dtype": "-",
                "flydsl_gpu_us": None,
                "baseline_name": "none",
                "baseline_gpu_us": None,
                "speedup": None,
                "status": "skip",
                "note": "kernel not found in kernels/ (tracked gap for fusion roadmap)",
            }
        )

    if "mla" in ops_set:
        for batch_size, ctx_len in mla_shapes:
            rows.append(_run_mla_row(batch_size=batch_size, ctx_len=ctx_len))

    if not rows:
        raise RuntimeError("no benchmark rows were produced; check --ops arguments")

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
    print(f"[bench_strixhalo_fusion] CSV written: {csv_path}")
    if error_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
PY

echo "[bench_strixhalo_fusion] done"
