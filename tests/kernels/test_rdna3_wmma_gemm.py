#!/usr/bin/env python3
"""RDNA 3 / 3.5 WMMA GEMM correctness tests (gfx1100 / gfx1150, wave32).

Exercises the atom-based path through ``MmaOpRDNA3_WMMAType`` introduced
for the gfx1150 (Strix Point) PoC. RDNA 4 (gfx1201) and gfx1250 keep
their existing ``MmaOpGFX1250_WMMAType`` path and are not exercised here
(they have wider K-shapes and FP8 support that this atom does not cover).
"""

import logging
import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.rdna3_f16_gemm import create_rdna3_wmma_gemm_module
from tests.test_common import verify_output
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch())


def _requires_rdna3():
    if not (ARCH.startswith("gfx115") or ARCH.startswith("gfx110")):
        pytest.skip(f"RDNA3 / 3.5 WMMA tests require gfx110x or gfx115x, got {ARCH}")


# ── Single-warp WMMA correctness ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(16, 16, 16, id="16x16x16"),
        pytest.param(32, 16, 16, id="32x16x16"),
        pytest.param(16, 32, 16, id="16x32x16"),
        pytest.param(16, 16, 32, id="16x16x32"),
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(128, 128, 128, id="128x128x128"),
    ],
)
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_rdna3_wmma_f32_acc(M, N, K, dtype):
    """RDNA 3 / 3.5 WMMA with F32 accumulator, F16 / BF16 inputs."""
    _requires_rdna3()

    torch.manual_seed(42)
    torch_dtype = torch.float16 if dtype == "f16" else torch.bfloat16

    A = (torch.randn(M, K, dtype=torch_dtype, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch_dtype, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch_fn, _, _, _ = create_rdna3_wmma_gemm_module(
        M, N, K, in_dtype=dtype, out_dtype="f32"
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = A.float() @ B_T.float().T
    assert verify_output(C, C_ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "M, N, K",
    [
        pytest.param(16, 16, 16, id="16x16x16"),
        pytest.param(64, 64, 64, id="64x64x64"),
    ],
)
def test_rdna3_wmma_f16_acc(M, N, K):
    """RDNA 3 / 3.5 WMMA with F16 accumulator (same-precision)."""
    _requires_rdna3()

    torch.manual_seed(42)

    A = (torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1)
    B_T = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1)
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")

    launch_fn, _, _, _ = create_rdna3_wmma_gemm_module(
        M, N, K, in_dtype="f16", out_dtype="f16"
    )
    launch_fn(A, B_T, C, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    C_ref = (A.float() @ B_T.float().T).to(torch.float16)
    assert verify_output(C, C_ref, atol=0.1, rtol=0.1)
