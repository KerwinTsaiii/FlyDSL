#!/usr/bin/env python3
"""Regression tests for RDNA WMMA GEMM shape heuristics."""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.rdna_f16_gemm import resolve_wmma_gemm_config


@pytest.mark.parametrize(
    "M, N, K, expected",
    [
        pytest.param(
            128,
            128,
            128,
            {"waves_m": 2, "waves_n": 2, "group_m": 8, "a_k_pad": 8, "b_k_pad": 8},
            id="rdna35-baseline",
        ),
        pytest.param(
            1024,
            1024,
            1024,
            {"waves_m": 4, "waves_n": 1, "group_m": 16, "a_k_pad": 16, "b_k_pad": 16},
            id="medium-square-remap-and-pad16",
        ),
        pytest.param(
            256,
            7168,
            16384,
            {"waves_m": 4, "waves_n": 1, "group_m": 16, "a_k_pad": 16, "b_k_pad": 16},
            id="m256-broad-family-4x1",
        ),
        pytest.param(
            256,
            4096,
            16384,
            {"waves_m": 2, "waves_n": 1, "group_m": 16, "a_k_pad": 16, "b_k_pad": 16},
            id="m256-longk-subset-2x1-override",
        ),
        pytest.param(
            256,
            3072,
            9216,
            {"waves_m": 2, "waves_n": 2, "group_m": 8, "a_k_pad": 8, "b_k_pad": 8},
            id="m256-exclusion-n3072-keeps-baseline-waves",
        ),
        pytest.param(
            512,
            2048,
            8192,
            {"waves_m": 4, "waves_n": 1, "group_m": 16, "a_k_pad": 16, "b_k_pad": 16},
            id="small-m-longk-remap-4x1",
        ),
        pytest.param(
            128,
            8192,
            4096,
            {"waves_m": 2, "waves_n": 2, "group_m": 8, "a_k_pad": 16, "b_k_pad": 16},
            id="m128-elongated-pad16",
        ),
        pytest.param(
            2048,
            8192,
            24576,
            {"waves_m": 2, "waves_n": 2, "group_m": 8, "a_k_pad": 16, "b_k_pad": 16},
            id="very-large-k-pad16",
        ),
        pytest.param(
            2048,
            3072,
            12288,
            {"waves_m": 2, "waves_n": 1, "group_m": 8, "a_k_pad": 16, "b_k_pad": 16},
            id="k12288-medium-n-remap-2x1",
        ),
    ],
)
def test_rdna35_wmma_heuristic_shapes(M, N, K, expected):
    cfg = resolve_wmma_gemm_config(M, N, K, gpu_arch="gfx1151")
    assert cfg["reg_m"] == 4
    assert cfg["reg_n"] == 4
    assert cfg["reg_k"] == 2
    for key, value in expected.items():
        assert cfg[key] == value


def test_wmma_heuristics_do_not_apply_to_gfx12():
    cfg = resolve_wmma_gemm_config(1024, 1024, 1024, gpu_arch="gfx1201")
    assert cfg["waves_m"] == 2
    assert cfg["waves_n"] == 2
    assert cfg["group_m"] == 8
    assert cfg["a_k_pad"] == 8
    assert cfg["b_k_pad"] == 8


def test_explicit_config_overrides_heuristics():
    cfg = resolve_wmma_gemm_config(
        1024,
        1024,
        1024,
        gpu_arch="gfx1151",
        waves_m=3,
        waves_n=2,
        group_m=4,
        a_k_pad=24,
        b_k_pad=24,
    )
    assert cfg["waves_m"] == 3
    assert cfg["waves_n"] == 2
    assert cfg["group_m"] == 4
    assert cfg["a_k_pad"] == 24
    assert cfg["b_k_pad"] == 24
