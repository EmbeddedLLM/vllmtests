# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import sys
import os
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest

if 1:
    _path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, f"{_path}/../../")
    from aiter.tuned_gemm import tgemm


@perftest(num_iters=10)
def run_torch(x, weight, bias=None, otype=None, scaleA=None, scaleB=None):
    if x.dtype == dtypes.fp8:
        if scaleA is None:
            scaleA = torch.ones(1, dtype=dtypes.fp32, device=x.device)
        if scaleB is None:
            scaleB = torch.ones(1, dtype=dtypes.fp32, device=x.device)

        try:
            out = torch._scaled_mm(
                x,
                weight.t(),
                out_dtype=otype,
                scale_a=scaleA,
                scale_b=scaleB,
                bias=bias,
            )
        except RuntimeError:
            out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32)) * scaleA * scaleB
            out = (out.to(otype) + bias) if bias is not None else out.to(otype)
        return out
    if scaleA is not None:
        x = x * scaleA
    if scaleB is not None:
        weight = weight * scaleB
    return F.linear(x, weight, bias).to(otype)


@perftest()
def run_gemm_b(x, weight, bias=None, otype=None, scaleA=None, scaleB=None):
    return tgemm.mm(x, weight, bias, otype, scaleA, scaleB)


def test_gemm(dtype, m, n, k, bias=False, otype=None, scaleA=None, scaleB=None):
    dim = (m, n, k)
    x = torch.randn(m, k, dtype=otype, device="cuda").to(dtype)
    weight = torch.rand(n, k, dtype=otype, device="cuda").to(dtype)
    if bias:
        bias = torch.rand(n, dtype=otype, device="cuda")
    else:
        bias = None
    if scaleA is not None:
        scaleA = torch.tensor(scaleA, dtype=dtypes.fp32, device="cuda")
    if scaleB is not None:
        scaleB = torch.tensor(scaleB, dtype=dtypes.fp32, device="cuda")
    (a, *_), avg_a = run_torch(x, weight, bias, otype, scaleA, scaleB)
    (b, *_), avg_b = run_gemm_b(x, weight, bias, otype, scaleA, scaleB)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, B avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg)


test_gemm(
    dtypes.fp8, 128, 768, 4096, bias=False, otype=dtypes.bf16, scaleA=0.5, scaleB=0.5
)
test_gemm(dtypes.bf16, 128, 32, 8192)
# for dtype in [dtypes.fp16, dtypes.bf16]:
#     # # qkv_proj
#     # for (m, n, k) in [(4096, 1280, 8192),
#     #                   (128, 1280, 8192),
#     #                   (128, 1024, 8192),
#     #                   (128, 128, 8192),
#     #                   ]:
#     #     test_gemm(dtype, m, n, k)
#     # # attn_out
#     # for (m, n, k) in [(4096, 8192, 1024),
#     #                   (128, 8192, 1024)]:
#     #     test_gemm(dtype, m, n, k)
#     # test_gemm(dtype, 128, 1024, 8192)
#     test_gemm(dtype, 128, 32, 8192)
#     # # gating
#     # for (m, n, k) in [(4096, 32, 8192),
#     #                   (128, 32, 8192)]:
#     #     test_gemm(dtype, m, n, k)
#     # # gating
#     # for (m, n, k) in [(1, 19392, 8192),
#     #                   (128, 19392, 8192)]:
#     #     test_gemm(dtype, m, n, k)
