# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat as eirp
from typing_extensions import List

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest

block_shape = (128, 128)


@perftest(num_iters=5, needTrace=False, num_warmup=5, testGraph=False, num_rotate_args=0)
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
    return out.to(dtype)


@perftest(needTrace=False, num_warmup=10, testGraph=False, num_rotate_args=0)
def run_gemm_ck(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


@perftest(needTrace=False, num_warmup=10, testGraph=False, num_rotate_args=0)
def run_gemm_bpreshuffle_ck(x, weightshuffle, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, weightshuffle, x_scale, w_scale, dtype
    )


@benchmark()
def test_gemm(dtype, m, n, k, preshuffle=False):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    a, avg_a = run_torch(x, weight, x_scale, w_scale, dtype)
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    gemm_x_scale = x_scale_t if preshuffle else x_scale
    gemm_weight = shuffle_weight(weight, layout=(16, 16)) if preshuffle else weight
    run_func = run_gemm_bpreshuffle_ck if preshuffle else run_gemm_ck
    # torch.cuda.empty_cache()  # Clear cache
    # torch.cuda.synchronize()  # Ensure clean state
    # avg_a = 0
    # failed = 0
    b, avg_b = run_func(x, gemm_weight, gemm_x_scale, w_scale, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    failed = checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)

    return {"us": avg_b, "failed": failed}


@perftest(num_iters=5)
def run_torch2(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]

    x_scale_ = eirp(x_scale, "m k -> m (k repeat)", repeat=block_shape_k)
    x_scale_ = x_scale_[:m, :k]

    w_scale_ = eirp(w_scale, "n k -> (n repeat) k", repeat=block_shape_n)
    w_scale_ = eirp(w_scale_, "n k -> n (k repeat)", repeat=block_shape_k)
    w_scale_ = w_scale_[:n, :k]

    x_ = x.to(x_scale.dtype) * x_scale_
    weight_ = weight.to(w_scale.dtype) * w_scale_

    out = F.linear(x_.to(dtypes.fp32), weight_.to(dtypes.fp32))
    return out.to(dtype)


@perftest()
def run_asm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.flatmm_a8w8_blockscale_ASM(x, weight, x_scale, w_scale, dtype)


@benchmark()
def test_gemm_asm(dtype, m, n, k):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)

    x_scale = torch.rand([scale_k, scale_m], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_k, scale_n], dtype=dtypes.fp32, device="cuda")

    x_scale_trans = torch.transpose(x_scale, 0, 1)
    w_scale_trans = torch.transpose(w_scale, 0, 1)

    flat_weight = weight.view(n // 16, 16, k // 64, 4, 16)
    flat_weight = flat_weight.permute(0, 2, 3, 1, 4).contiguous()
    flat_weight = flat_weight.view(n, -1)

    a, avg_a = run_torch2(x, weight, x_scale_trans, w_scale_trans, dtype)
    b, avg_b = run_asm(x, flat_weight, x_scale, w_scale, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, asm avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b -1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)


l_dtype = ["bf16"]

# Shapes to represent untuned shapes
l_m = [10000, 16384, 20480, 32768, 65536, 128000, 131072, 260000, 262144]

# vLLM Graph Capture Size
# l_m = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 1024, 8192]

l_nk = [
    # DeepSeek-R1

    (1536, 7168),
    (3072, 1536),
    (576, 7168),
    (7168, 256),
    (7168, 2048),
    (4608, 7168),
    (7168, 2304),
    (512, 7168),
    (4096, 512),

    (2112,7168), # from vllm shared expert

    # # Qwen3-235B-FP8 2507
    (1280,4096), # TP8
    (4096,1024), # TP8
    
    # Qwen3-Coder-480B-FP8
    (1792,6144), # TP8
    (6144,1536), # TP8

    (3584, 6144), # TP4
    (6144, 3072), # TP4

]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-nk",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""N&K of mnk.
    e.g.: -nk 4096,512""",
)
parser.add_argument(
    "--preshuffle",
    nargs="+",
    default=[True, False],
    help="weight preshuffle or not",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.m is not None:
    l_m = [args.m]
if args.nk is not None:
    l_nk = [args.nk]
l_preshuffle: List[bool] = args.preshuffle

df = []
for dtype, m, (n, k), preshuffle in itertools.product(l_dtype, l_m, l_nk, l_preshuffle):
    # deepseek-r1
    ret = test_gemm(dtype, m, n, k, preshuffle)
    df.append(ret)
df = pd.DataFrame(df)
csv_filename = "gemm_test_results_nottunedshape_final_eager_rebuildaiter.csv"
df.to_csv(csv_filename, index=False)
aiter.logger.info(f"summary:\n{df}")
# for dtype in [dtypes.fp16]:
#     # deepseek-r1
#     for m in [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192, 16384, 20480]:
#         for (n, k) in [(1536,7168), (3072,1536), (7168, 256), (7168, 2048), (4608, 7168), (7168, 2304), (512, 7168), (4096, 512)][1:2]:
#             test_gemm_asm(dtype, m, n, k)
#             break
if df["failed"].any():
    print("Failed cases:", df[df["failed"] > 0], sep="\n")
        # Also save failed cases to a separate CSV
    failed_df = df[df["failed"] > 0]
    failed_csv_filename = "gemm_failed_cases_nottunedshape_final_eager_rebuildaiter.csv"
    failed_df.to_csv(failed_csv_filename, index=False)
    aiter.logger.info(f"Failed cases saved to {failed_csv_filename}")
